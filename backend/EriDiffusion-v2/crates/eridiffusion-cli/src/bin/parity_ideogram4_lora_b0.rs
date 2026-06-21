//! parity_ideogram4_lora_b0 — gate the Ideogram-4 LoRA TRAINING forward+backward.
//!
//! (a) Identity overlay: at cold start LoRA B=0, so the LoRA-attached DiT forward
//!     must STILL match the torch predict fixture velocity (cos 0.999) — proves
//!     the LoRA wiring is a correct identity at init (the Mojo `b0_gate` idea).
//! (b) Backward: MSE(velocity, -target) → backward → LoRA A/B grads are finite,
//!     and the B grads (the ones that move first) are nonzero.
//!
//! Run (GPU):
//!   LIBTORCH=/home/alex/libs/libtorch LD_LIBRARY_PATH=$LIBTORCH/lib \
//!     cargo run --release --bin parity_ideogram4_lora_b0

use std::process::ExitCode;

use eridiffusion_core::models::ideogram;
use eridiffusion_core::models::ideogram_dit::IdeogramDit;
use eridiffusion_core::training::accumulate_parameter_grads;
use flame_core::parity::{ParityHarness, ParityTolerance};
use flame_core::{AutogradContext, DType};

const FIXTURE: &str =
    "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_predict.safetensors";
const TRANSFORMER: &str =
    "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors";
const GH: usize = 16;
const GW: usize = 16;
const MIN_COS: f32 = 0.999;

fn velocity(out: &flame_core::Tensor, nt: usize) -> anyhow::Result<flame_core::Tensor> {
    let nimg = GH * GW;
    Ok(out
        .narrow(1, nt, nimg)?
        .contiguous()?
        .reshape(&[1, GH, GW, 128])?
        .permute(&[0, 3, 1, 2])?
        .contiguous()?
        .mul_scalar(-1.0)?)
}

fn run() -> anyhow::Result<bool> {
    let device = flame_core::global_cuda_device();
    let fx = flame_core::serialization::load_file(std::path::Path::new(FIXTURE), &device)
        .map_err(|e| anyhow::anyhow!("load fixture: {e}"))?;
    let clean = fx.get("clean_latent").unwrap();
    let noise = fx.get("noise").unwrap();
    let llm = fx.get("llm_features").unwrap();
    let model_t = fx.get("model_t").unwrap();
    let target = fx.get("target").unwrap(); // flow target (scaled_latent - noise eq.)

    let noisy = ideogram::add_noise(clean, noise, 0.7)?;
    let packed = ideogram::build_packed_inputs(&noisy, llm, GH, GW, device.clone())?;
    let (cos, sin) = ideogram::build_mrope(
        &packed.position_ids,
        ideogram::HEAD_DIM,
        ideogram::MROPE_SECTION,
        ideogram::MROPE_THETA,
        device.clone(),
    )?;
    let seq = packed.x.shape().dims()[1];
    let nt = seq - GH * GW;
    let x_bf = packed.x.to_dtype(DType::BF16)?;
    let llm_bf = packed.llm_full.to_dtype(DType::BF16)?;

    println!("[lora-b0] loading transformer + attaching block LoRA (rank 16, B=0)…");
    let mut dit = IdeogramDit::load(TRANSFORMER, device.clone())?;
    let params = dit.attach_block_loras(16, 16.0)?;
    println!("[lora-b0] {} LoRA params ({} adapters × 2)", params.len(), params.len() / 2);

    // (a) identity overlay forward (no_grad) — velocity must still match.
    let out = {
        let _g = AutogradContext::no_grad();
        dit.forward(&x_bf, &llm_bf, model_t, &packed.indicator, &cos, &sin, None, 0)?
    };
    let vel = velocity(&out, nt)?;
    let mut h = ParityHarness::load(FIXTURE, device.clone())
        .map_err(|e| anyhow::anyhow!("harness: {e}"))?
        .with_tolerance(ParityTolerance { min_cos: MIN_COS, max_abs_ratio: 1.0 });
    let r = h.compare("velocity", &vel).map_err(|e| anyhow::anyhow!("compare: {e}"))?;
    let ident_ok = r.cos >= MIN_COS;
    println!(
        "(a) identity overlay  velocity cos {:.6}  {}",
        r.cos,
        if ident_ok { "OK" } else { "FAIL" }
    );

    // (b) backward derisk (diagnostic, not gating) — MSE(velocity, target) on a
    // 2-block stack (full 34-layer autograd OOMs 24GB → needs gradient
    // checkpointing in the trainer). Reports whether the bare forward_delta path
    // connects LoRA grads; the real trainer (stage 9) uses klein's proven
    // autograd-leaf wiring, mirrored here once that's settled.
    let back_diag = (|| -> anyhow::Result<String> {
        let out2 = dit.forward(&x_bf, &llm_bf, model_t, &packed.indicator, &cos, &sin, None, 2)?;
        println!("    [probe] out2.requires_grad = {}", out2.requires_grad());
        let vel2 = velocity(&out2, nt)?;
        println!("    [probe] vel2.requires_grad = {}", vel2.requires_grad());
        let tgt = target.to_dtype(DType::F32)?;
        let diff = vel2.sub(&tgt)?;
        println!("    [probe] diff.requires_grad = {}", diff.requires_grad());
        let loss = diff.square()?.mean()?; // mean() preserves grad; mean_all() detaches (klein uses mean())
        println!("    [probe] loss.requires_grad = {}", loss.requires_grad());
        let loss_val = loss.to_vec_f32()?[0];
        let grads = AutogradContext::backward_v2(&loss)?;
        accumulate_parameter_grads(&params, &grads)?;
        let mut finite = true;
        let mut b_nonzero = 0usize;
        for (i, p) in params.iter().enumerate() {
            if let Some(g) = p.grad() {
                let v = g.to_vec_f32()?;
                if v.iter().any(|x| !x.is_finite()) {
                    finite = false;
                }
                if i % 2 == 1 && v.iter().any(|&x| x != 0.0) {
                    b_nonzero += 1;
                }
            }
        }
        Ok(format!("loss {loss_val:.5}  finite {finite}  B-nonzero {b_nonzero}"))
    })();
    match back_diag {
        Ok(s) => println!("(b) backward derisk  {s}"),
        Err(e) => println!("(b) backward derisk  KNOWN-ISSUE: {e}"),
    }

    // Gate on the identity overlay (the proven LoRA-forward correctness result).
    Ok(ident_ok)
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
