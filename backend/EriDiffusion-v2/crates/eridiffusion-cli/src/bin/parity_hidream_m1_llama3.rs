//! HiDream M1.5 — Llama-3.1-8B encoder parity vs HuggingFace `transformers`.
//!
//! Reads `/tmp/hidream_m1_llama3_ref_meta.json` + `/tmp/hidream_m1_llama3_ref.safetensors`
//! (produced by `tests/parity/hidream_m1_llama3_python_ref.py`), runs our
//! `Llama3Encoder` on the SAME token ids + attention mask, and compares each
//! of the 32 per-layer hidden states via `flame_core::parity::ParityHarness`.
//!
//! Tolerance: `min_cos = 0.9995`, `max_abs_ratio = unused` — the spec asks for
//! `max_abs < 0.1` directly, which we evaluate inline alongside the harness
//! cos check (the harness uses a *ratio* threshold which is more permissive
//! per-tensor; the spec wants an absolute cap on the worst element).
//!
//! Exit:
//!   0 — every layer passes cos >= 0.9995 AND max_abs < 0.1
//!   1 — at least one layer fails
//!   2 — setup/IO error (weights missing, ref dump missing, etc.)
//!
//! CLI:
//!   parity_hidream_m1_llama3 \
//!     --llama-path /path/to/unsloth-llama-3.1-8b-instruct-dir \
//!     [--ref-path /tmp/hidream_m1_llama3_ref.safetensors] \
//!     [--meta-path /tmp/hidream_m1_llama3_ref_meta.json]

use clap::Parser;
use eridiffusion_core::encoders::llama3::Llama3Encoder;
use flame_core::parity::{ParityHarness, ParityTolerance};
use flame_core::{DType, Tensor};
use std::path::PathBuf;
use std::process::ExitCode;

const PASS_MIN_COS: f32 = 0.9995;
const PASS_MAX_ABS: f32 = 0.1;

#[derive(Parser)]
#[command(name = "parity_hidream_m1_llama3")]
struct Args {
    /// Path to the Llama-3.1-8B-Instruct weights — either a directory of
    /// HF-sharded safetensors (`model-00001-of-00004.safetensors` ...) or a
    /// single consolidated safetensors file.
    #[arg(long)]
    llama_path: PathBuf,

    /// PyTorch reference dump emitted by `hidream_m1_llama3_python_ref.py`.
    #[arg(long, default_value = "/tmp/hidream_m1_llama3_ref.safetensors")]
    ref_path: PathBuf,

    /// JSON metadata sidecar emitted by the Python script (carries the
    /// input ids, mask, and config we have to match exactly).
    #[arg(long, default_value = "/tmp/hidream_m1_llama3_ref_meta.json")]
    meta_path: PathBuf,
}

#[derive(Debug)]
struct Meta {
    input_ids: Vec<i32>,
    attention_mask: Vec<bool>,
    num_layers: usize,
    hidden_size: usize,
    seq_len: usize,
    real_tokens: usize,
    pad_token_id: i32,
    prompt: String,
    model_name: String,
}

/// Tiny hand-rolled JSON reader for just the keys we need. Avoids dragging
/// `serde_json` into a leaf binary's dependency closure beyond what
/// eridiffusion-cli already has.
fn load_meta(path: &std::path::Path) -> anyhow::Result<Meta> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("read meta {}: {e}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| anyhow::anyhow!("parse meta {}: {e}", path.display()))?;

    let input_ids: Vec<i32> = v["input_ids"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("meta.input_ids: not an array"))?
        .iter()
        .map(|n| n.as_i64().unwrap_or(0) as i32)
        .collect();
    let attention_mask: Vec<bool> = v["attention_mask"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("meta.attention_mask: not an array"))?
        .iter()
        .map(|n| n.as_i64().unwrap_or(0) != 0)
        .collect();
    if input_ids.len() != attention_mask.len() {
        anyhow::bail!(
            "meta: input_ids len {} != attention_mask len {}",
            input_ids.len(),
            attention_mask.len()
        );
    }

    Ok(Meta {
        input_ids,
        attention_mask,
        num_layers: v["num_layers"].as_u64().unwrap_or(0) as usize,
        hidden_size: v["hidden_size"].as_u64().unwrap_or(0) as usize,
        seq_len: v["seq_len"].as_u64().unwrap_or(0) as usize,
        real_tokens: v["real_tokens"].as_u64().unwrap_or(0) as usize,
        pad_token_id: v["pad_token_id"].as_i64().unwrap_or(128_004) as i32,
        prompt: v["prompt"].as_str().unwrap_or("").to_string(),
        model_name: v["model_name"].as_str().unwrap_or("").to_string(),
    })
}

/// Compute `max(|a - b|)` in F32 over a (possibly BF16) tensor pair without
/// allocating a diff tensor on the GPU. We rely on the harness for cosine —
/// it also returns max_abs but expressed against the same F32 conversion,
/// which is what we want.
fn run() -> anyhow::Result<bool> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let args = Args::parse();

    log::info!("[parity-rs] meta:    {}", args.meta_path.display());
    log::info!("[parity-rs] ref:     {}", args.ref_path.display());
    log::info!("[parity-rs] weights: {}", args.llama_path.display());

    if !args.meta_path.exists() {
        anyhow::bail!(
            "ref metadata not found at {} — run \
             tests/parity/hidream_m1_llama3_python_ref.py first",
            args.meta_path.display()
        );
    }
    if !args.ref_path.exists() {
        anyhow::bail!(
            "ref tensors not found at {} — run \
             tests/parity/hidream_m1_llama3_python_ref.py first",
            args.ref_path.display()
        );
    }
    if !args.llama_path.exists() {
        anyhow::bail!(
            "Llama-3.1-8B weights not found at {} — download with: \
             huggingface-cli download unsloth/Meta-Llama-3.1-8B-Instruct \
             --local-dir <path>",
            args.llama_path.display()
        );
    }

    let meta = load_meta(&args.meta_path)?;
    log::info!(
        "[parity-rs] model={} prompt={:?} num_layers={} hidden={} seq_len={} real={}",
        meta.model_name,
        meta.prompt,
        meta.num_layers,
        meta.hidden_size,
        meta.seq_len,
        meta.real_tokens,
    );

    // ----- Wire up CUDA + encoder -------------------------------------------
    let device = flame_core::global_cuda_device();
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();

    log::info!("[parity-rs] loading Llama3Encoder ...");
    let mut encoder = Llama3Encoder::load(
        args.llama_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("--llama-path is not valid UTF-8"))?,
        &device,
    )
    .map_err(|e| anyhow::anyhow!("Llama3Encoder::load: {e}"))?;

    // Sanity: encoder's pad id should match what the Python side used. We
    // don't fail on mismatch because the test only cares about the produced
    // hidden states, which depend on `attention_mask`, not on `pad_token_id`
    // (those positions are masked out). But log it loudly.
    log::info!(
        "[parity-rs] meta.pad_token_id={} (Rust encoder default 128004)",
        meta.pad_token_id
    );

    // ----- Forward pass ------------------------------------------------------
    log::info!("[parity-rs] encoding ...");
    let ours_stacked = encoder
        .encode_all_hidden_states(&meta.input_ids, &meta.attention_mask)
        .map_err(|e| anyhow::anyhow!("encode_all_hidden_states: {e}"))?;
    let dims = ours_stacked.shape().dims().to_vec();
    log::info!("[parity-rs] our stacked output: {:?} dtype={:?}",
        dims, ours_stacked.dtype());

    if dims.len() != 4 || dims[0] != meta.num_layers
        || dims[2] != meta.seq_len
        || dims[3] != meta.hidden_size
    {
        anyhow::bail!(
            "shape mismatch: our output {:?}, ref expects [{}, 1, {}, {}]",
            dims, meta.num_layers, meta.seq_len, meta.hidden_size
        );
    }

    // ----- Per-layer parity --------------------------------------------------
    // We use the harness for the cosine math (well-tested) and compute
    // max_abs separately so we can apply the spec's *absolute* 0.1 cap
    // alongside the harness's ratio cap.
    let mut harness = ParityHarness::load(&args.ref_path, device.clone())
        .map_err(|e| anyhow::anyhow!("ParityHarness::load {}: {e}", args.ref_path.display()))?
        .with_tolerance(ParityTolerance {
            min_cos: PASS_MIN_COS,
            max_abs_ratio: 1.0, // disable ratio check; absolute cap handled below
        });

    println!();
    println!(
        "{:<10}  {:>10}  {:>12}  {:>12}  {:>10}  {}",
        "layer", "cos", "max_abs", "mean_abs", "ratio", "status"
    );

    let mut all_passed = true;
    let mut first_fail: Option<(usize, String)> = None;
    for i in 0..meta.num_layers {
        // narrow over dim 0 to get [1, 1, S, H], then drop the leading layer dim.
        let our_layer = ours_stacked
            .narrow(0, i, 1)
            .map_err(|e| anyhow::anyhow!("narrow layer {i}: {e}"))?;
        // The harness compares by element count + shape; reference is
        // [1, S, H]. Reshape ours to match.
        let our_layer_3d = our_layer
            .reshape(&[1, meta.seq_len, meta.hidden_size])
            .map_err(|e| anyhow::anyhow!("reshape layer {i}: {e}"))?;

        let key = format!("layer_{:02}", i);
        let r = harness
            .compare(&key, &our_layer_3d)
            .map_err(|e| anyhow::anyhow!("compare {key}: {e}"))?;

        let abs_ok = r.max_abs < PASS_MAX_ABS;
        let cos_ok = r.cos >= PASS_MIN_COS;
        let pass = abs_ok && cos_ok && r.note.is_none();
        let status = if pass {
            "OK"
        } else if r.note.is_some() {
            "ERR"
        } else if !cos_ok {
            "FAIL(cos)"
        } else {
            "FAIL(abs)"
        };
        println!(
            "{:<10}  {:>10.6}  {:>12.6e}  {:>12.6e}  {:>10.4}  {}{}",
            format!("layer_{:02}", i),
            r.cos,
            r.max_abs,
            r.mean_abs,
            r.max_abs_ratio,
            status,
            r.note.as_deref().map(|n| format!(" ({n})")).unwrap_or_default(),
        );
        if !pass {
            all_passed = false;
            if first_fail.is_none() {
                first_fail = Some((i, status.to_string()));
            }
        }
    }

    println!();
    println!("{}", harness.report());

    if all_passed {
        println!("[parity-rs] PASS — all {} layers cos >= {:.4} && max_abs < {}",
            meta.num_layers, PASS_MIN_COS, PASS_MAX_ABS);
        Ok(true)
    } else {
        let (first_idx, kind) = first_fail.unwrap();
        println!("[parity-rs] FAIL — first failure: layer_{:02} ({})", first_idx, kind);
        Ok(false)
    }
}

fn main() -> ExitCode {
    // `_` is fine; map run() into a 0/1/2 exit per the spec.
    match run() {
        Ok(true) => ExitCode::from(0),
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("[parity-rs] setup error: {e:#}");
            ExitCode::from(2)
        }
    }
}

// Tag — `Tensor` import used implicitly via the encoder/parity types only.
// Keep the explicit `use` so future edits don't strip the wrong thing.
#[allow(dead_code)]
fn _keep_tensor_import_alive(_: &Tensor) {}
#[allow(dead_code)]
fn _keep_dtype_import_alive(_: DType) {}
