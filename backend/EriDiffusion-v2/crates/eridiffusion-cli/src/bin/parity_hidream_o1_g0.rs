//! HiDream-O1 G0 — PyTorch inference parity for the trainer forward path.
//!
//! Loads HiDream-O1 via the same `HiDreamO1WeightLoader` the trainer uses,
//! builds an EMPTY LoraRegistry (A=N(0,1e-4), B=0 by default; --zero-lora-a
//! forces A=0 too for strict parity), reads the Python reference inputs +
//! output from `/tmp/hidream_o1_g0_python_ref.safetensors`, calls
//! `model.forward_lora(...)` with the same inputs, and compares the gathered
//! image-row output (`x_pred_rows`) against the Python reference via
//! `flame_core::parity::ParityHarness`.
//!
//! Tolerance (default):
//!   - cos      >= 0.999
//!   - max_abs <  0.5     (BF16 forward through a 36-layer 8B model — the
//!                         end-of-pipe absolute error is dominated by mantissa
//!                         truncation over the ~4096-dim hidden state; 0.5 is
//!                         the trainer-equivalence threshold we use elsewhere
//!                         for full-model bf16 parity).
//!
//! Exit codes:
//!   0 = PASS, 1 = FAIL, 2 = BLOCKED (missing deps / weights / ref dump).

use clap::Parser;
use flame_core::parity::{ParityHarness, ParityTolerance};
use flame_core::{DType, Shape, Tensor};
use eridiffusion_core::models::hidream_o1::{
    default_target_suffixes, HiDreamO1Config, HiDreamO1WeightLoader, LoraRegistry, MRopePositions,
};
use std::path::PathBuf;
use std::process::ExitCode;

const PASS_MIN_COS: f32 = 0.999;
const PASS_MAX_ABS: f32 = 0.5;
const SEED: u64 = 42;
const DEFAULT_MODEL_PATH: &str = "/home/alex/HiDream-O1-Image-Dev-weights";

#[derive(Parser)]
#[command(name = "parity_hidream_o1_g0")]
struct Args {
    /// HiDream-O1 weights dir.
    #[arg(long, default_value = DEFAULT_MODEL_PATH)]
    model_path: PathBuf,
    /// PyTorch reference dump.
    #[arg(long, default_value = "/tmp/hidream_o1_g0_python_ref.safetensors")]
    ref_path: PathBuf,
    /// JSON sidecar (carries shapes / pinning info).
    #[arg(long, default_value = "/tmp/hidream_o1_g0_python_ref_meta.json")]
    meta_path: PathBuf,
    /// LoRA rank used by the trainer (default 16). Must match the trainer's
    /// real config so we exercise the same code path.
    #[arg(long, default_value = "16")]
    rank: usize,
    /// LoRA alpha (default 16.0). Default config: scale = alpha/rank = 1.0.
    #[arg(long, default_value = "16.0")]
    lora_alpha: f32,
    /// Belt-and-suspenders: also force `A = 0` (default A ~ N(0, 1e-4)).
    /// Either way the LoRA delta is exactly zero because `B = 0` at init.
    /// But A=0 removes any micro-RNG drift in the optimizer/grad path so
    /// the test isolates "trainer forward equals inference forward" cleanly.
    #[arg(long, default_value_t = true)]
    zero_lora_a: bool,
    /// Tolerance overrides.
    #[arg(long, default_value_t = PASS_MIN_COS)]
    min_cos: f32,
    #[arg(long, default_value_t = PASS_MAX_ABS)]
    max_abs: f32,

    /// Deep-investigation mode: dump every decoder layer's hidden state and
    /// compare against `--per-layer-ref` (defaults to the per-layer Python
    /// ref). When this flag is set we IGNORE the regular `x_pred_rows`
    /// ref and instead drive a per-layer cosine cascade.
    #[arg(long, default_value = "")]
    per_layer_dump: String,
    /// Per-layer Python reference (default = output of
    /// tests/parity/hidream_o1_g0_per_layer_ref.py).
    #[arg(long, default_value = "/tmp/hidream_o1_g0_per_layer_ref.safetensors")]
    per_layer_ref: PathBuf,
}

fn decode_position_ids(pos: &Tensor) -> anyhow::Result<(Vec<u32>, Vec<u32>, Vec<u32>)> {
    let dims = pos.shape().dims().to_vec();
    if dims.len() != 2 || dims[0] != 3 {
        anyhow::bail!("position_ids: expected [3, S_total], got {:?}", dims);
    }
    let s_total = dims[1];
    // Ref tensor is I32 on disk; cast to F32 for to_vec_f32.
    let flat = pos.to_dtype(DType::F32)?.to_vec_f32()?;
    let mut t = Vec::with_capacity(s_total);
    let mut h = Vec::with_capacity(s_total);
    let mut w = Vec::with_capacity(s_total);
    for i in 0..s_total {
        t.push(flat[i] as u32);
        h.push(flat[s_total + i] as u32);
        w.push(flat[2 * s_total + i] as u32);
    }
    Ok((t, h, w))
}

/// Mirror `train_hidream_o1::gather_image_rows` — gather the contiguous tail
/// of image-mask=1 rows from `x_pred [1, S_total, 3072]`.
fn gather_image_rows(x_pred: &Tensor, vinput_mask: &Tensor) -> anyhow::Result<Tensor> {
    let xd = x_pred.shape().dims().to_vec();
    if xd.len() != 3 {
        anyhow::bail!("gather_image_rows: expected [B,S,C], got {:?}", xd);
    }
    let (_b, s_total, _c) = (xd[0], xd[1], xd[2]);
    let host = vinput_mask.to_dtype(DType::F32)?.to_vec_f32()?;
    let mut first: Option<usize> = None;
    let mut last: Option<usize> = None;
    for i in 0..s_total {
        if host[i] != 0.0 {
            first.get_or_insert(i);
            last = Some(i);
        }
    }
    let (first, last) = (
        first.ok_or_else(|| anyhow::anyhow!("vinput_mask has no image slots"))?,
        last.unwrap(),
    );
    let len = last - first + 1;
    let count = host.iter().filter(|&&x| x != 0.0).count();
    if count != len {
        anyhow::bail!(
            "non-contiguous image slots not supported (count {} != span {})",
            count,
            len
        );
    }
    Ok(x_pred.narrow(1, first, len)?)
}

fn run() -> anyhow::Result<bool> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
    let args = Args::parse();

    log::info!("[parity-rs] meta:    {}", args.meta_path.display());
    log::info!("[parity-rs] ref:     {}", args.ref_path.display());
    log::info!("[parity-rs] weights: {}", args.model_path.display());
    log::info!(
        "[parity-rs] LoRA: rank={} alpha={} zero_A={}",
        args.rank,
        args.lora_alpha,
        args.zero_lora_a
    );
    log::info!(
        "[parity-rs] tolerance: min_cos={:.4} max_abs={:.3}",
        args.min_cos,
        args.max_abs
    );

    for p in [&args.ref_path, &args.meta_path] {
        if !p.exists() {
            anyhow::bail!(
                "ref dump {} not found — run \
                 tests/parity/hidream_o1_g0_python_ref.py first",
                p.display()
            );
        }
    }
    if !args.model_path.exists() {
        anyhow::bail!("model path does not exist: {}", args.model_path.display());
    }

    // ── flame setup ───────────────────────────────────────────────────
    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();
    flame_core::rng::set_seed(SEED)
        .map_err(|e| anyhow::anyhow!("flame_core set_seed: {e}"))?;
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();

    // ── Load reference inputs first (cheap, fails early if format wrong). ─
    log::info!("[parity-rs] loading reference dump ...");
    let ref_tensors = flame_core::serialization::load_file(&args.ref_path, &device)
        .map_err(|e| anyhow::anyhow!("load ref {}: {e}", args.ref_path.display()))?;
    let get = |k: &str| -> anyhow::Result<&Tensor> {
        ref_tensors
            .get(k)
            .ok_or_else(|| anyhow::anyhow!("ref missing key {k:?}"))
    };

    let patches = get("patches")?.to_dtype(DType::BF16)?;
    let input_ids = get("input_ids")?.to_dtype(DType::I32)?;
    let position_ids = get("position_ids")?.clone();
    let vinput_mask = get("vinput_mask")?.to_dtype(DType::BF16)?;
    // token_types_bin = (raw > 0), required for the attention-mask TMS-row fix
    // (deep investigation 2026-05-17). Python ref dumps it under key
    // `token_types` already (`hidream_o1_g0_python_ref.py:237`).
    let token_types_bin = get("token_types")?.to_dtype(DType::BF16)?;
    let timestep_f32 = get("timestep")?.to_dtype(DType::F32)?;
    let x_pred_rows_ref = get("x_pred_rows")?.clone(); // F32 on disk

    log::info!(
        "[parity-rs] ref shapes: patches={:?} input_ids={:?} pos={:?} vmask={:?} \
         timestep={:?} x_pred_rows_ref={:?}",
        patches.shape().dims(),
        input_ids.shape().dims(),
        position_ids.shape().dims(),
        vinput_mask.shape().dims(),
        timestep_f32.shape().dims(),
        x_pred_rows_ref.shape().dims(),
    );

    // BF16 timestep (matches trainer; the embedder casts internally either way).
    let timestep = timestep_f32.to_dtype(DType::BF16)?;
    let (t_pos, h_pos, w_pos) = decode_position_ids(&position_ids)?;
    let pos_view = MRopePositions {
        t: &t_pos,
        h: &h_pos,
        w: &w_pos,
    };

    // ── Build the model the way the trainer does. ─────────────────────
    let cfg = HiDreamO1Config::dev_8b();
    log::info!(
        "[parity-rs] loading model from {} (num_layers={}, hidden={}) ...",
        args.model_path.display(),
        cfg.num_layers,
        cfg.hidden_size,
    );
    let loader = HiDreamO1WeightLoader::from_dir(&args.model_path)
        .map_err(|e| anyhow::anyhow!("HiDreamO1WeightLoader: {e}"))?;
    let mut model = loader
        .load_model(&cfg, &device)
        .map_err(|e| anyhow::anyhow!("load_model: {e}"))?;

    // ── Build empty LoRA registry. ────────────────────────────────────
    let mut lora = LoraRegistry::new(
        &cfg,
        args.rank,
        args.lora_alpha,
        default_target_suffixes(),
        SEED,
        &device,
    )
    .map_err(|e| anyhow::anyhow!("LoraRegistry::new: {e}"))?;
    log::info!(
        "[parity-rs] LoRA registry: {} adapters (rank={}, alpha={})",
        lora.len(),
        lora.rank,
        lora.alpha,
    );

    if args.zero_lora_a {
        // Force A = 0 in addition to B = 0. With both zero the LoRA delta is
        // identically the zero tensor — no RNG, no drift.
        let keys: Vec<String> = lora.adapters.keys().cloned().collect();
        for k in &keys {
            let ad = lora.adapters.get_mut(k).unwrap();
            let a_shape = Shape::from_dims(ad.a.tensor()?.shape().dims());
            let a_zero =
                Tensor::zeros_dtype(a_shape, DType::BF16, device.clone())?.requires_grad_(true);
            ad.a.set_data(a_zero)
                .map_err(|e| anyhow::anyhow!("zero A for {k}: {e}"))?;
        }
        log::info!("[parity-rs] forced A=0 on {} adapters", keys.len());
    }

    // ── Per-layer dump mode (deep investigation). ─────────────────────
    let per_layer_mode = !args.per_layer_dump.is_empty();
    if per_layer_mode {
        std::env::set_var("HIDREAM_DUMP_LAYERS", &args.per_layer_dump);
        log::info!(
            "[parity-rs] per-layer dump enabled → {} (will compare vs {})",
            args.per_layer_dump,
            args.per_layer_ref.display(),
        );
    }

    // ── Forward. ──────────────────────────────────────────────────────
    log::info!("[parity-rs] running forward_lora ...");
    let t0 = std::time::Instant::now();
    let x_pred_full = model
        .forward_lora(
            &input_ids,
            &timestep,
            &patches, // vinputs = noise z
            &pos_view,
            &vinput_mask,
            &token_types_bin,
            None,
            Some(&lora),
        )
        .map_err(|e| anyhow::anyhow!("forward_lora: {e}"))?;
    log::info!(
        "[parity-rs] forward done in {:.2}s, x_pred_full={:?} dtype={:?}",
        t0.elapsed().as_secs_f32(),
        x_pred_full.shape().dims(),
        x_pred_full.dtype(),
    );

    let x_pred_rows = gather_image_rows(&x_pred_full, &vinput_mask)?;
    log::info!(
        "[parity-rs] gathered x_pred_rows={:?} dtype={:?}",
        x_pred_rows.shape().dims(),
        x_pred_rows.dtype(),
    );

    // Shape sanity vs ref.
    if x_pred_rows.shape().dims() != x_pred_rows_ref.shape().dims() {
        anyhow::bail!(
            "x_pred_rows shape mismatch: ours={:?} ref={:?}",
            x_pred_rows.shape().dims(),
            x_pred_rows_ref.shape().dims(),
        );
    }

    // ── Parity compare. ───────────────────────────────────────────────
    let mut harness = ParityHarness::load(&args.ref_path, device.clone())
        .map_err(|e| anyhow::anyhow!("ParityHarness::load: {e}"))?
        .with_tolerance(ParityTolerance {
            min_cos: args.min_cos,
            max_abs_ratio: 1.0, // disabled; absolute cap handled below
        });

    println!();
    println!(
        "{:<14}  {:>10}  {:>12}  {:>12}  {:>10}  {}",
        "tensor", "cos", "max_abs", "mean_abs", "ratio", "status"
    );

    let r = harness
        .compare("x_pred_rows", &x_pred_rows)
        .map_err(|e| anyhow::anyhow!("compare x_pred_rows: {e}"))?;
    let cos_ok = r.cos >= args.min_cos;
    let abs_ok = r.max_abs < args.max_abs;
    let pass = cos_ok && abs_ok && r.note.is_none();
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
        "{:<14}  {:>10.6}  {:>12.6e}  {:>12.6e}  {:>10.4}  {}{}",
        "x_pred_rows",
        r.cos,
        r.max_abs,
        r.mean_abs,
        r.max_abs_ratio,
        status,
        r.note.as_deref().map(|n| format!(" ({n})")).unwrap_or_default(),
    );

    println!();
    println!("{}", harness.report());

    // ── Per-layer cascade (deep investigation). ──────────────────────
    if per_layer_mode {
        println!();
        println!("=== PER-LAYER COSINE CASCADE ===");
        let ours = flame_core::serialization::load_file(
            std::path::Path::new(&args.per_layer_dump),
            &device,
        )
        .map_err(|e| anyhow::anyhow!("load Rust dump {}: {e}", args.per_layer_dump))?;
        let theirs = flame_core::serialization::load_file(&args.per_layer_ref, &device)
            .map_err(|e| anyhow::anyhow!(
                "load Python per-layer ref {}: {e}", args.per_layer_ref.display()))?;

        println!(
            "{:<25}  {:>10}  {:>12}  {:>12}  {:>8}",
            "key", "cos", "max_abs", "mean_abs", "note"
        );

        // Order: input, layer_00 .. layer_35, final_norm.
        let mut keys: Vec<String> = vec!["hidden_input_layer_00".to_string()];
        for i in 0..cfg.num_layers {
            keys.push(format!("hidden_layer_{i:02}"));
        }
        keys.push("hidden_final_norm".to_string());

        let mut prev_cos = 1.0f32;
        let mut min_cos = 1.0f32;
        let mut min_cos_key = String::new();
        let mut biggest_drop = 0.0f32;
        let mut biggest_drop_key = String::new();
        for k in &keys {
            let (a, b) = match (ours.get(k), theirs.get(k)) {
                (Some(a), Some(b)) => (a, b),
                _ => {
                    println!("{:<25}  {:>10}  {:>12}  {:>12}  MISSING", k, "-", "-", "-");
                    continue;
                }
            };
            // Both should be F32. Compute cos, max_abs, mean_abs on the
            // host (these are small enough to round-trip).
            let av = a.to_dtype(DType::F32)?.to_vec_f32()?;
            let bv = b.to_dtype(DType::F32)?.to_vec_f32()?;
            if av.len() != bv.len() {
                println!(
                    "{:<25}  {:>10}  {:>12}  {:>12}  SHAPE_MISMATCH ours_len={} ref_len={}",
                    k, "-", "-", "-", av.len(), bv.len()
                );
                continue;
            }
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            let mut max_abs = 0.0f32;
            let mut sum_abs = 0.0f64;
            for (x, y) in av.iter().zip(bv.iter()) {
                let xf = *x as f64;
                let yf = *y as f64;
                dot += xf * yf;
                na += xf * xf;
                nb += yf * yf;
                let d = (x - y).abs();
                if d > max_abs { max_abs = d; }
                sum_abs += d as f64;
            }
            let cos = (dot / (na.sqrt() * nb.sqrt() + 1e-30)) as f32;
            let mean_abs = (sum_abs / av.len() as f64) as f32;
            let drop = prev_cos - cos;
            if drop > biggest_drop {
                biggest_drop = drop;
                biggest_drop_key = k.clone();
            }
            if cos < min_cos {
                min_cos = cos;
                min_cos_key = k.clone();
            }
            let note = if drop > 0.005 { "<<DROP>>" } else { "" };
            println!(
                "{:<25}  {:>10.6}  {:>12.6e}  {:>12.6e}  {}",
                k, cos, max_abs, mean_abs, note
            );
            prev_cos = cos;
        }
        println!();
        println!("min_cos={:.6} at {}", min_cos, min_cos_key);
        println!("biggest single-layer drop={:.6} at {}", biggest_drop, biggest_drop_key);

        // ── Step 2: Layer-0 fp64 isolated check. ─────────────────────
        if let (Some(rust_l0), Some(py_l0_bf16), Some(py_l0_fp64)) = (
            ours.get("hidden_layer_00"),
            theirs.get("hidden_layer_00"),
            theirs.get("hidden_layer_00_fp64"),
        ) {
            let r = rust_l0.to_dtype(DType::F32)?.to_vec_f32()?;
            let pb = py_l0_bf16.to_dtype(DType::F32)?.to_vec_f32()?;
            let pf = py_l0_fp64.to_dtype(DType::F32)?.to_vec_f32()?;
            let cos_rb_pb = cosf(&r, &pb);
            let cos_rb_pf = cosf(&r, &pf);
            let cos_pb_pf = cosf(&pb, &pf);
            println!();
            println!("=== STEP 2: LAYER-0 FP64 ISOLATED ===");
            println!("cos(Rust_bf16, Python_bf16) = {:.8}", cos_rb_pb);
            println!("cos(Rust_bf16, Python_fp64) = {:.8}", cos_rb_pf);
            println!("cos(Python_bf16, Python_fp64) = {:.8}", cos_pb_pf);
            println!();
            if cos_rb_pf >= cos_pb_pf - 5e-4 {
                println!(
                    "VERDICT: Rust matches fp64 truth as well as Python_bf16 does \
                     → no structural bug at layer 0 (both bf16 paths are equidistant from truth)"
                );
            } else {
                println!(
                    "VERDICT: Rust drifts further from fp64 than Python_bf16 \
                     does → POSSIBLE STRUCTURAL BUG at layer 0 \
                     (gap = {:.6})",
                    cos_pb_pf - cos_rb_pf
                );
            }
        } else {
            println!("[per-layer] WARN: missing layer-0 fp64 keys, skipping Step 2");
        }
    }

    if pass {
        println!(
            "[parity-rs] PASS — cos {:.6} >= {:.4} && max_abs {:.4e} < {}",
            r.cos, args.min_cos, r.max_abs, args.max_abs
        );
        Ok(true)
    } else {
        println!(
            "[parity-rs] FAIL — cos {:.6} (need >= {:.4}) max_abs {:.4e} (need < {})",
            r.cos, args.min_cos, r.max_abs, args.max_abs
        );
        Ok(false)
    }
}

fn cosf(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let xf = *x as f64;
        let yf = *y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    (dot / (na.sqrt() * nb.sqrt() + 1e-30)) as f32
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::from(0),
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("[parity-rs] setup error: {e:#}");
            ExitCode::from(2)
        }
    }
}
