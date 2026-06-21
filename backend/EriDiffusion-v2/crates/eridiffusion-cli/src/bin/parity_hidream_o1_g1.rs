//! HiDream-O1 G1 — Rust self-consistency parity gate.
//!
//! G0 verified that the trainer's `forward_lora` matches the Python reference
//! (cos=0.999811 on a 36-layer 8B bf16 model). G1 closes the remaining gap:
//! does the TRAINER forward path (`forward_lora` with an empty LoRA registry)
//! produce output bit-close to the INFERENCE forward path (`forward`, no LoRA),
//! using the SAME model instance and the SAME inputs?
//!
//! Both paths funnel into `forward_inner` (see model.rs), with the only
//! difference being `lora: Option<&LoraRegistry>`. When the registry's A and
//! B are exactly zero the LoRA delta is `scale * x @ A^T @ B^T = 0` — so the
//! two paths must be mathematically identical. Any drift here is a real
//! trainer/inference code-path divergence (extra `.contiguous()`, dtype
//! reshape, autograd-record difference, etc).
//!
//! Acceptance: cos >= 0.99999. This is Rust-vs-Rust, same dtype, same kernels,
//! no Python BF16 noise — the threshold is "exact-or-near-exact".
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

const PASS_MIN_COS: f32 = 0.99999;
const PASS_MAX_ABS_RATIO: f32 = 0.001;
const SEED: u64 = 42;
const DEFAULT_MODEL_PATH: &str = "/home/alex/HiDream-O1-Image-Dev-weights";

#[derive(Parser)]
#[command(name = "parity_hidream_o1_g1")]
struct Args {
    /// HiDream-O1 weights dir.
    #[arg(long, default_value = DEFAULT_MODEL_PATH)]
    model_path: PathBuf,
    /// Reference inputs dump (we only consume inputs, not the Python output).
    #[arg(long, default_value = "/tmp/hidream_o1_g0_python_ref.safetensors")]
    ref_path: PathBuf,
    /// LoRA rank for the empty registry.
    #[arg(long, default_value = "16")]
    rank: usize,
    /// LoRA alpha.
    #[arg(long, default_value = "16.0")]
    lora_alpha: f32,
    /// Tolerance overrides.
    #[arg(long, default_value_t = PASS_MIN_COS)]
    min_cos: f32,
    #[arg(long, default_value_t = PASS_MAX_ABS_RATIO)]
    max_abs_ratio: f32,
}

fn decode_position_ids(pos: &Tensor) -> anyhow::Result<(Vec<u32>, Vec<u32>, Vec<u32>)> {
    let dims = pos.shape().dims().to_vec();
    if dims.len() != 2 || dims[0] != 3 {
        anyhow::bail!("position_ids: expected [3, S_total], got {:?}", dims);
    }
    let s_total = dims[1];
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

fn run() -> anyhow::Result<bool> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
    let args = Args::parse();

    log::info!("[parity-g1] ref:     {}", args.ref_path.display());
    log::info!("[parity-g1] weights: {}", args.model_path.display());
    log::info!(
        "[parity-g1] LoRA: rank={} alpha={} (A=0, B=0)",
        args.rank,
        args.lora_alpha,
    );
    log::info!(
        "[parity-g1] tolerance: min_cos={:.6} max_abs_ratio={:.4}",
        args.min_cos,
        args.max_abs_ratio
    );

    if !args.ref_path.exists() {
        anyhow::bail!(
            "ref dump {} not found — run tests/parity/run_hidream_o1_g0.sh first",
            args.ref_path.display()
        );
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

    // ── Load reference inputs (we ignore the Python output). ─────────
    log::info!("[parity-g1] loading reference inputs ...");
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
    let token_types_bin = get("token_types")?.to_dtype(DType::BF16)?;
    let timestep = get("timestep")?.to_dtype(DType::BF16)?;

    log::info!(
        "[parity-g1] input shapes: patches={:?} input_ids={:?} pos={:?} vmask={:?} timestep={:?}",
        patches.shape().dims(),
        input_ids.shape().dims(),
        position_ids.shape().dims(),
        vinput_mask.shape().dims(),
        timestep.shape().dims(),
    );

    let (t_pos, h_pos, w_pos) = decode_position_ids(&position_ids)?;
    let pos_view = MRopePositions {
        t: &t_pos,
        h: &h_pos,
        w: &w_pos,
    };

    // ── Build the model once. ─────────────────────────────────────────
    let cfg = HiDreamO1Config::dev_8b();
    log::info!(
        "[parity-g1] loading model from {} (num_layers={}, hidden={}) ...",
        args.model_path.display(),
        cfg.num_layers,
        cfg.hidden_size,
    );
    let loader = HiDreamO1WeightLoader::from_dir(&args.model_path)
        .map_err(|e| anyhow::anyhow!("HiDreamO1WeightLoader: {e}"))?;
    let mut model = loader
        .load_model(&cfg, &device)
        .map_err(|e| anyhow::anyhow!("load_model: {e}"))?;

    // ── Build LoRA registry with A=0, B=0 (mathematically zero delta). ─
    let mut lora = LoraRegistry::new(
        &cfg,
        args.rank,
        args.lora_alpha,
        default_target_suffixes(),
        SEED,
        &device,
    )
    .map_err(|e| anyhow::anyhow!("LoraRegistry::new: {e}"))?;
    let keys: Vec<String> = lora.adapters.keys().cloned().collect();
    for k in &keys {
        let ad = lora.adapters.get_mut(k).unwrap();
        let a_shape = Shape::from_dims(ad.a.tensor()?.shape().dims());
        let a_zero =
            Tensor::zeros_dtype(a_shape, DType::BF16, device.clone())?.requires_grad_(true);
        ad.a.set_data(a_zero)
            .map_err(|e| anyhow::anyhow!("zero A for {k}: {e}"))?;
        // B is already zero at init.
    }
    log::info!(
        "[parity-g1] LoRA registry: {} adapters (A=0, B=0 forced)",
        lora.len(),
    );

    // ── PATH 1: inference forward (no LoRA). ──────────────────────────
    log::info!("[parity-g1] running INFERENCE path: model.forward(...) ...");
    let t0 = std::time::Instant::now();
    let x_inference = model
        .forward(
            &input_ids,
            &timestep,
            &patches,
            &pos_view,
            &vinput_mask,
            &token_types_bin,
            None,
        )
        .map_err(|e| anyhow::anyhow!("inference forward: {e}"))?;
    log::info!(
        "[parity-g1] inference forward done in {:.2}s, out={:?} dtype={:?}",
        t0.elapsed().as_secs_f32(),
        x_inference.shape().dims(),
        x_inference.dtype(),
    );

    // ── PATH 2: trainer forward (with empty LoRA registry). ───────────
    log::info!("[parity-g1] running TRAINER path: model.forward_lora(..., Some(&lora)) ...");
    let t1 = std::time::Instant::now();
    let x_trainer = model
        .forward_lora(
            &input_ids,
            &timestep,
            &patches,
            &pos_view,
            &vinput_mask,
            &token_types_bin,
            None,
            Some(&lora),
        )
        .map_err(|e| anyhow::anyhow!("trainer forward_lora: {e}"))?;
    log::info!(
        "[parity-g1] trainer forward done in {:.2}s, out={:?} dtype={:?}",
        t1.elapsed().as_secs_f32(),
        x_trainer.shape().dims(),
        x_trainer.dtype(),
    );

    // ── Shape sanity. ─────────────────────────────────────────────────
    if x_inference.shape().dims() != x_trainer.shape().dims() {
        anyhow::bail!(
            "output shape mismatch: inference={:?} trainer={:?}",
            x_inference.shape().dims(),
            x_trainer.shape().dims(),
        );
    }

    // ── Compare: build an in-memory ParityHarness by treating the
    //    inference path as the reference. Use F32 host vectors for
    //    a metric independent of BF16 storage.
    let inf_f32 = x_inference.to_dtype(DType::F32)?.to_vec_f32()?;
    let trn_f32 = x_trainer.to_dtype(DType::F32)?.to_vec_f32()?;
    if inf_f32.len() != trn_f32.len() {
        anyhow::bail!(
            "element-count mismatch: inference={} trainer={}",
            inf_f32.len(),
            trn_f32.len()
        );
    }

    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut max_inf_abs = 0.0f32;
    for (x, y) in inf_f32.iter().zip(trn_f32.iter()) {
        let xf = *x as f64;
        let yf = *y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_abs += d as f64;
        let ax = x.abs();
        if ax > max_inf_abs {
            max_inf_abs = ax;
        }
    }
    let cos = (dot / (na.sqrt() * nb.sqrt() + 1e-30)) as f32;
    let mean_abs = (sum_abs / inf_f32.len() as f64) as f32;
    let max_abs_ratio = if max_inf_abs > 0.0 {
        max_abs / max_inf_abs
    } else {
        0.0
    };

    // Also wire ParityHarness with the inference output saved to a temp
    // file, for symmetry with G0 and to exercise the harness path.
    let tmp_ref = std::env::temp_dir().join("hidream_o1_g1_inference_ref.safetensors");
    let mut to_save = std::collections::HashMap::new();
    to_save.insert("x_pred".to_string(), x_inference.clone());
    flame_core::serialization::save_file(&to_save, &tmp_ref)
        .map_err(|e| anyhow::anyhow!("save tmp ref: {e}"))?;
    let mut harness = ParityHarness::load(&tmp_ref, device.clone())
        .map_err(|e| anyhow::anyhow!("ParityHarness::load: {e}"))?
        .with_tolerance(ParityTolerance {
            min_cos: args.min_cos,
            max_abs_ratio: args.max_abs_ratio,
        });
    let hr = harness
        .compare("x_pred", &x_trainer)
        .map_err(|e| anyhow::anyhow!("compare x_pred: {e}"))?;

    println!();
    println!(
        "{:<14}  {:>10}  {:>12}  {:>12}  {:>10}  {}",
        "tensor", "cos", "max_abs", "mean_abs", "ratio", "status"
    );

    let cos_ok = hr.cos >= args.min_cos;
    let ratio_ok = hr.max_abs_ratio <= args.max_abs_ratio;
    let pass = cos_ok && ratio_ok && hr.note.is_none();
    let status = if pass {
        "OK"
    } else if hr.note.is_some() {
        "ERR"
    } else if !cos_ok {
        "FAIL(cos)"
    } else {
        "FAIL(ratio)"
    };
    println!(
        "{:<14}  {:>10.8}  {:>12.6e}  {:>12.6e}  {:>10.6}  {}{}",
        "x_pred",
        hr.cos,
        hr.max_abs,
        hr.mean_abs,
        hr.max_abs_ratio,
        status,
        hr.note.as_deref().map(|n| format!(" ({n})")).unwrap_or_default(),
    );

    println!();
    println!(
        "[parity-g1] hand-computed: cos={:.8} max_abs={:.6e} mean_abs={:.6e} ratio={:.6}",
        cos, max_abs, mean_abs, max_abs_ratio
    );
    println!("[parity-g1] inference max|x|={:.6e}", max_inf_abs);

    println!();
    println!("{}", harness.report());

    if pass {
        println!(
            "[parity-g1] PASS — cos {:.8} >= {:.6} && ratio {:.6} <= {:.4}",
            hr.cos, args.min_cos, hr.max_abs_ratio, args.max_abs_ratio
        );
        Ok(true)
    } else {
        println!(
            "[parity-g1] FAIL — cos {:.8} (need >= {:.6}) ratio {:.6} (need <= {:.4})",
            hr.cos, args.min_cos, hr.max_abs_ratio, args.max_abs_ratio
        );
        Ok(false)
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::from(0),
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("[parity-g1] setup error: {e:#}");
            ExitCode::from(2)
        }
    }
}
