//! parity_ideogram4_predict — gate the Rust Ideogram-4 predict stages 1–2
//! (`add_noise`, `flow_target`, `build_packed_inputs`) vs the torch oracle
//! fixture `ideogram4_fx_predict.safetensors` (real-giger, GH=GW=16, LT=651,
//! seq=907). Later stages (MRoPE, full `predict_velocity`) extend this bin.
//!
//! Run (GPU):
//!   LIBTORCH=/home/alex/libs/libtorch LD_LIBRARY_PATH=$LIBTORCH/lib \
//!     cargo run --release --bin parity_ideogram4_predict

use std::process::ExitCode;

use eridiffusion_core::models::ideogram;
use eridiffusion_core::models::ideogram_dit::IdeogramDit;
use flame_core::parity::{ParityHarness, ParityTolerance};

const FIXTURE: &str =
    "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_predict.safetensors";
const MROPE_FIXTURE: &str =
    "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_mrope.safetensors";
const TRANSFORMER: &str =
    "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors";
const INTERMEDIATES: &str =
    "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_intermediates.safetensors";
const T_FLOW: f32 = 0.7;
const GH: usize = 16;
const GW: usize = 16;
const MIN_COS: f32 = 0.999;

fn run() -> anyhow::Result<bool> {
    let device = flame_core::global_cuda_device();

    // inputs from the fixture
    let fx = flame_core::serialization::load_file(FIXTURE, &device)
        .map_err(|e| anyhow::anyhow!("load fixture: {e}"))?;
    let clean = fx
        .get("clean_latent")
        .ok_or_else(|| anyhow::anyhow!("fixture missing clean_latent"))?;
    let noise = fx
        .get("noise")
        .ok_or_else(|| anyhow::anyhow!("fixture missing noise"))?;
    let llm = fx
        .get("llm_features")
        .ok_or_else(|| anyhow::anyhow!("fixture missing llm_features"))?;

    // stage 1: flow helpers
    let noisy = ideogram::add_noise(clean, noise, T_FLOW)?;
    let target = ideogram::flow_target(noise, clean)?;
    // stage 2: packed inputs
    let packed = ideogram::build_packed_inputs(&noisy, llm, GH, GW, device.clone())?;

    // compare vs the torch dump
    let mut harness = ParityHarness::load(FIXTURE, device.clone())
        .map_err(|e| anyhow::anyhow!("ParityHarness::load: {e}"))?
        .with_tolerance(ParityTolerance {
            min_cos: MIN_COS,
            max_abs_ratio: 1.0,
        });

    println!(
        "{:<18} {:>10} {:>12} {:>12}  status",
        "tensor", "cos", "max_abs", "mean_abs"
    );
    let cases: [(&str, &flame_core::Tensor); 5] = [
        ("noisy", &noisy),
        ("target", &target),
        ("x", &packed.x),
        ("position_ids_f32", &packed.position_ids),
        ("indicator_f32", &packed.indicator),
    ];
    let mut all_ok = true;
    for (name, ours) in cases {
        let r = harness
            .compare(name, ours)
            .map_err(|e| anyhow::anyhow!("compare {name}: {e}"))?;
        let ok = r.cos >= MIN_COS && r.note.is_none();
        all_ok &= ok;
        println!(
            "{:<18} {:>10.6} {:>12.4e} {:>12.4e}  {}",
            name,
            r.cos,
            r.max_abs,
            r.mean_abs,
            if ok { "OK" } else { "FAIL" }
        );
    }

    // stage 3: interleaved MRoPE vs its own torch fixture.
    let (mrope_cos, mrope_sin) = ideogram::build_mrope(
        &packed.position_ids,
        ideogram::HEAD_DIM,
        ideogram::MROPE_SECTION,
        ideogram::MROPE_THETA,
        device.clone(),
    )?;
    let mut mrope = ParityHarness::load(MROPE_FIXTURE, device.clone())
        .map_err(|e| anyhow::anyhow!("ParityHarness::load mrope: {e}"))?
        .with_tolerance(ParityTolerance {
            min_cos: MIN_COS,
            max_abs_ratio: 1.0,
        });
    for (name, ours) in [("mrope_cos", &mrope_cos), ("mrope_sin", &mrope_sin)] {
        let r = mrope
            .compare(name, ours)
            .map_err(|e| anyhow::anyhow!("compare {name}: {e}"))?;
        let ok = r.cos >= MIN_COS && r.note.is_none();
        all_ok &= ok;
        println!(
            "{:<18} {:>10.6} {:>12.4e} {:>12.4e}  {}",
            name,
            r.cos,
            r.max_abs,
            r.mean_abs,
            if ok { "OK" } else { "FAIL" }
        );
    }

    // stage 4: full DiT forward → velocity (OT residency model — layers stream per-block).
    println!("[stage4] loading transformer (conditioning once + 34 layers streamed)…");
    let dit = IdeogramDit::load(TRANSFORMER, device.clone())?;
    let model_t = fx
        .get("model_t")
        .ok_or_else(|| anyhow::anyhow!("fixture missing model_t"))?;
    let x_bf = packed.x.to_dtype(flame_core::DType::BF16)?;
    let llm_bf = packed.llm_full.to_dtype(flame_core::DType::BF16)?;
    let mut dbg = std::collections::HashMap::new();
    let out = dit.forward(&x_bf, &llm_bf, model_t, &packed.indicator, &mrope_cos, &mrope_sin, Some(&mut dbg), 0)?; // [1,seq,128] f32

    // localize: compare DiT intermediates (input_proj → h_pre → block0 → block1).
    let mut interm = ParityHarness::load(INTERMEDIATES, device.clone())
        .map_err(|e| anyhow::anyhow!("ParityHarness::load intermediates: {e}"))?
        .with_tolerance(ParityTolerance { min_cos: MIN_COS, max_abs_ratio: 1.0 });
    let out_f32 = out.to_dtype(flame_core::DType::F32)?;
    for name in [
        "input_proj_out",
        "h_pre",
        "block0_out",
        "block1_out",
        "block8_out",
        "block16_out",
        "block33_out",
    ] {
        if let Some(t) = dbg.get(name) {
            let r = interm
                .compare(name, t)
                .map_err(|e| anyhow::anyhow!("compare {name}: {e}"))?;
            println!(
                "{:<18} {:>10.6} {:>12.4e} {:>12.4e}  {}",
                name,
                r.cos,
                r.max_abs,
                r.mean_abs,
                if r.cos >= MIN_COS { "OK" } else { "FAIL" }
            );
        }
    }
    let rt = interm
        .compare("transformer_out", &out_f32)
        .map_err(|e| anyhow::anyhow!("compare transformer_out: {e}"))?;
    println!(
        "{:<18} {:>10.6} {:>12.4e} {:>12.4e}  {}",
        "transformer_out",
        rt.cos,
        rt.max_abs,
        rt.mean_abs,
        if rt.cos >= MIN_COS { "OK" } else { "FAIL" }
    );
    // velocity = -out[:, NT:].reshape(GH,GW,128).permute(0,3,1,2)
    let seq = out.shape().dims()[1];
    let nimg = GH * GW;
    let nt = seq - nimg;
    let vel = out
        .narrow(1, nt, nimg)?
        .contiguous()?
        .reshape(&[1, GH, GW, 128])?
        .permute(&[0, 3, 1, 2])?
        .contiguous()?
        .mul_scalar(-1.0)?;
    let r = harness
        .compare("velocity", &vel)
        .map_err(|e| anyhow::anyhow!("compare velocity: {e}"))?;
    let ok = r.cos >= MIN_COS && r.note.is_none();
    all_ok &= ok;
    println!(
        "{:<18} {:>10.6} {:>12.4e} {:>12.4e}  {}",
        "velocity",
        r.cos,
        r.max_abs,
        r.mean_abs,
        if ok { "OK" } else { "FAIL" }
    );

    Ok(all_ok)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::from(0),
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("[parity-rs] error: {e:#}");
            ExitCode::from(2)
        }
    }
}
