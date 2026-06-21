//! parity_ideogram4_encoder — gate the EDv2 `Qwen3Encoder` (reused, configured
//! with Ideogram-4's 13 activation taps) vs the torch oracle `llm_features`
//! [1,651,53248] in `ideogram4_fx_predict.safetensors`. Feeds the fixture's
//! `token_ids` (no tokenization needed). Confirms stage 5 = a config away.
//!
//! Run (GPU):
//!   LIBTORCH=/home/alex/libs/libtorch LD_LIBRARY_PATH=$LIBTORCH/lib \
//!     cargo run --release --bin parity_ideogram4_encoder

use std::process::ExitCode;

use eridiffusion_core::encoders::qwen3::{Qwen3Config, Qwen3Encoder};
use flame_core::parity::{ParityHarness, ParityTolerance};

const FIXTURE: &str =
    "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_predict.safetensors";
const TOKENS: &str =
    "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_tokens.safetensors";
const TEXT_ENCODER: &str =
    "/home/alex/.serenity/models/ideogram-4-fp8/text_encoder/model.safetensors";
// Ideogram-4 QWEN3_VL_ACTIVATION_LAYERS (post-layer output indices).
const TAPS: [usize; 13] = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35];
const MIN_COS: f32 = 0.999;

fn run() -> anyhow::Result<bool> {
    let device = flame_core::global_cuda_device();

    let toks = flame_core::serialization::load_file(std::path::Path::new(TOKENS), &device)
        .map_err(|e| anyhow::anyhow!("load tokens: {e}"))?;
    let tok = toks
        .get("token_ids_f32")
        .ok_or_else(|| anyhow::anyhow!("tokens file missing token_ids_f32"))?;
    let token_ids: Vec<i32> = tok.to_vec_f32()?.iter().map(|&x| x.round() as i32).collect();
    let l = token_ids.len();
    println!("[encoder] {l} tokens; loading text_encoder (8.8GB fp8 → bf16)…");

    let raw = flame_core::serialization::load_file(std::path::Path::new(TEXT_ENCODER), &device)
        .map_err(|e| anyhow::anyhow!("load text_encoder: {e}"))?;
    // Qwen3-VL nests the LM under `language_model.` (+ a `visual.` tower);
    // Qwen3Encoder expects flat `model.*`. Remap, drop the vision tower.
    let mut w = std::collections::HashMap::with_capacity(raw.len());
    for (k, t) in raw {
        if let Some(rest) = k.strip_prefix("language_model.") {
            w.insert(format!("model.{rest}"), t);
        }
    }
    let mut cfg = Qwen3Config::qwen3_vl_text();
    cfg.extract_layers = TAPS.to_vec();
    let enc = Qwen3Encoder::new(w, cfg, device.clone());

    let out = enc.encode(&token_ids)?;
    let od = enc.output_dim();
    let out = out.reshape(&[1, l, od])?;
    println!("[encoder] output {:?} (klein tap-major order)", out.shape().dims());

    // klein stacks tap-major (idx = tap*H + h); Ideogram is hidden-major
    // (idx = h*ntaps + tap). De-interleave: [1,L,ntaps,H] → permute → [1,L,H,ntaps]
    // → [1,L,H*ntaps]. (.contiguous() before the final reshape — permute is a view.)
    let ntaps = TAPS.len();
    let hsz = od / ntaps;
    let out = out
        .reshape(&[1, l, ntaps, hsz])?
        .permute(&[0, 1, 3, 2])?
        .contiguous()?
        .reshape(&[1, l, od])?;

    let mut h = ParityHarness::load(FIXTURE, device.clone())
        .map_err(|e| anyhow::anyhow!("harness: {e}"))?
        .with_tolerance(ParityTolerance {
            min_cos: MIN_COS,
            max_abs_ratio: 1.0,
        });
    let r = h
        .compare("llm_features", &out)
        .map_err(|e| anyhow::anyhow!("compare llm_features: {e}"))?;
    let ok = r.cos >= MIN_COS;
    println!(
        "{:<14} {:>10.6} {:>12.4e} {:>12.4e}  {}",
        "llm_features",
        r.cos,
        r.max_abs,
        r.mean_abs,
        if ok { "OK" } else { "FAIL" }
    );
    Ok(ok)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::from(0),
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("[parity] error: {e:#}");
            ExitCode::from(2)
        }
    }
}
