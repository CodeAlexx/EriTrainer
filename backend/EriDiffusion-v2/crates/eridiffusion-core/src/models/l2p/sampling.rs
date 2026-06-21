//! L2P flow-matching sampler — Z-Image-style Euler + CFG, pixel-space.
//!
//! Self-contained copy of `inference_flame::sampling::l2p_sampling` (the
//! parts the `train_l2p` inline sampler uses). Operates on the
//! `super::dit::L2pDiT` training copy.
//!
//! Sign / timestep conventions are owned by `L2pDiT::forward_inner`:
//!   - The caller passes the flow-matching normalized timestep `v ∈ [0,1]`
//!     **as-is**. The DiT internally remaps to `(1 - v) * time_scale`.
//!   - The DiT internally negates the U-Net output before returning. The
//!     sampler MUST NOT re-negate.

use super::dit::L2pDiT;
use flame_core::{DType, Result, Shape, Tensor};
use std::sync::Arc;

/// Build the L2P FlowMatch sigma schedule (FLUX-shift form).
///
/// Matches DiffSynth's `FlowMatchScheduler.set_timesteps`
/// (`reference/diffsynth/diffusion/flow_match.py:103-118`):
/// ```python
/// sigmas = linspace(1, 0, num_steps + 1)[:-1]
/// sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
/// sigmas = cat([sigmas, zeros(1)])
/// ```
/// NOT the Klein/Qwen-Image `exp(μ)/(exp(μ)+(1/t-1))` curve — those diverge
/// from L2P's by 2-4× at low sigma (audit F1, `MATH_AUDIT_2026-05-22.md`).
pub fn build_l2p_sigma_schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    let shift = shift as f64;
    let mut sigmas: Vec<f32> = (0..num_steps)
        .map(|i| {
            let t = 1.0 - (i as f64) / (num_steps as f64);
            let s = shift * t / (1.0 + (shift - 1.0) * t);
            s as f32
        })
        .collect();
    sigmas.push(0.0);
    sigmas
}

/// CPU Box–Muller Gaussian noise (seeded `StdRng`). Deterministic across
/// hosts. Inlined copy of `inference_flame::sampling::klein_sampling::
/// box_muller_noise` to keep this module free of inference-flame deps.
fn box_muller_noise(numel: usize, seed: u64) -> Vec<f32> {
    use rand::prelude::*;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut v = Vec::with_capacity(numel);
    for _ in 0..numel / 2 {
        let u1: f32 = rng.gen::<f32>().max(1e-10);
        let u2: f32 = rng.gen::<f32>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        v.push(r * theta.cos());
        v.push(r * theta.sin());
    }
    if numel % 2 == 1 {
        let u1: f32 = rng.gen::<f32>().max(1e-10);
        let u2: f32 = rng.gen::<f32>();
        v.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos());
    }
    v
}

/// Generate initial pixel-space noise on CPU and upload to GPU as **F32**.
///
/// Shape: `[1, 3, height, width]`, F32. Per PORT_SPEC §"Special / things to
/// watch" #4, the L2P reference pipeline forces noise to F32 even though the
/// DiT is BF16. The caller is responsible for the F32→BF16 cast before
/// calling `l2p_euler_step`.
pub fn init_l2p_noise(
    height: usize,
    width: usize,
    seed: u64,
    device: &Arc<cudarc::driver::CudaDevice>,
) -> Result<Tensor> {
    let numel = 3 * height * width;
    let data = box_muller_noise(numel, seed);
    Tensor::from_vec_dtype(
        data,
        Shape::from_dims(&[1, 3, height, width]),
        device.clone(),
        DType::F32,
    )
}

/// One Euler step for L2P with optional classifier-free guidance.
///
/// **Dtype contract.** `x` must be BF16 — `L2pDiT::forward_inner`
/// `debug_assert_eq`'s on this. `init_l2p_noise` returns F32; the caller
/// must cast (`.to_dtype(DType::BF16)?`) before the first call.
///
/// **Sign / timestep contract.** `sigma` is the flow-matching normalized
/// `v ∈ [0, 1]` from the schedule. `L2pDiT::forward_inner` handles the L2P
/// sign + timestep inversion internally. Do NOT re-invert the timestep or
/// re-negate the prediction here.
///
/// **CFG.** `pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)`
/// (matches Python `base_pipeline_L2P.py:353`).
pub fn l2p_euler_step(
    model: &mut L2pDiT,
    x: &Tensor,
    sigma: f32,
    sigma_next: f32,
    cap_feats: &Tensor,
    cap_feats_uncond: Option<&Tensor>,
    cfg_scale: f32,
) -> Result<Tensor> {
    debug_assert_eq!(
        x.dtype(),
        DType::BF16,
        "l2p_euler_step expects BF16 x — caller must cast F32 noise to BF16"
    );

    let device = x.device().clone();
    let b = x.shape().dims()[0];

    // Timestep tensor: pass `sigma` (normalized) as-is. The DiT inverts and
    // scales internally.
    let sigma_tensor =
        Tensor::from_vec_dtype(vec![sigma; b], Shape::from_dims(&[b]), device, DType::BF16)?;

    // Conditional prediction. forward_inner already applies the
    // pipeline-level sign flip; do NOT negate here.
    let pred_cond = model.forward(x, &sigma_tensor, cap_feats)?;

    let pred = if let (Some(uncond_feats), true) = (cap_feats_uncond, cfg_scale > 1.0) {
        let pred_uncond = model.forward(x, &sigma_tensor, uncond_feats)?;
        let diff = pred_cond.sub(&pred_uncond)?;
        let scaled = diff.mul_scalar(cfg_scale)?;
        pred_uncond.add(&scaled)?
    } else {
        pred_cond
    };

    // Euler step: x_next = x + (sigma_next - sigma) * pred.
    let dsigma = sigma_next - sigma;
    let step = pred.mul_scalar(dsigma)?;
    x.add(&step)
}
