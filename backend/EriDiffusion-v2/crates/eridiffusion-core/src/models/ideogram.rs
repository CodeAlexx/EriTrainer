//! Ideogram-4 — training predict path (Rust port of the parity-verified Mojo
//! `Ideogram4Predict.mojo`, itself 1:1 from ai-toolkit ideogram4 pipeline.py
//! `predict_velocity`). Gated vs the torch oracle fixture
//! `ideogram4_fx_predict.safetensors` by the `parity_ideogram4_predict` bin.
//!
//! Stage 1–2: flow helpers (`add_noise` / `flow_target`) + the packed-input
//! builder. The DiT forward (MRoPE + `ideogram4_forward`) lands in a later stage.

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::{Result, Shape, Tensor};
use half::bf16;

// ── architecture constants (from Ideogram4Sampler.mojo) ──
pub const NUM_LAYERS: usize = 34;
pub const HIDDEN: usize = 4608;
pub const NUM_HEADS: usize = 18;
pub const HEAD_DIM: usize = 256;
pub const TEXT_FEATURE_DIM: usize = 53248;
pub const PACKED_CHANNELS: usize = 128;
pub const IMAGE_OFFSET: usize = 65536;
pub const LLM_TOKEN_INDICATOR: f32 = 3.0;
pub const OUTPUT_IMAGE_INDICATOR: f32 = 2.0;
pub const MROPE_SECTION: [usize; 3] = [24, 20, 20];
pub const MROPE_THETA: f32 = 5_000_000.0;

/// Flow-match add_noise: `noisy = (1 - t) * clean + t * noise`
/// (ai-toolkit pipeline.py; clean/noise persist F32 — the Euler-latent rule).
pub fn add_noise(clean: &Tensor, noise: &Tensor, t: f32) -> Result<Tensor> {
    clean.affine(1.0 - t, 0.0)?.add(&noise.affine(t, 0.0)?)
}

/// Flow-match loss target: `target = noise - clean` (ai-toolkit get_loss_target).
pub fn flow_target(noise: &Tensor, clean: &Tensor) -> Result<Tensor> {
    noise.sub(clean)
}

/// Ideogram-4 interleaved MRoPE (1:1 from `Ideogram4MRoPE.forward`).
/// `position_ids`: [1, L, 3] F32 (t, h, w). Returns (cos, sin) [1, L, head_dim] F32.
///
/// Computed host-side because two numeric details dominate at the 65536 image
/// positions: `inv_freq` is **bf16-rounded** (the real bf16 model's buffer, which
/// `forward` upcasts to f32), and the trig needs an **F64 range reduction** (F32
/// trig is wrong at pos ~ 65536). Per-index axis: t by default, H at d%3==1 &
/// d<section[1]*3, W at d%3==2 & d<section[2]*3; `emb = cat(freqs, freqs)`.
pub fn build_mrope(
    position_ids: &Tensor,
    head_dim: usize,
    sections: [usize; 3],
    theta: f32,
    device: Arc<CudaDevice>,
) -> Result<(Tensor, Tensor)> {
    let pos = position_ids.to_vec_f32()?; // [L*3], rows (t, h, w)
    let l = pos.len() / 3;
    let half_dim = head_dim / 2;
    let sec_h = sections[1] * 3;
    let sec_w = sections[2] * 3;
    let log_theta = (theta as f64).ln();
    let two_pi = std::f64::consts::TAU;

    let mut cos_v = vec![0f32; l * head_dim];
    let mut sin_v = vec![0f32; l * head_dim];
    for li in 0..l {
        for dd in 0..head_dim {
            let d = if dd < half_dim { dd } else { dd - half_dim }; // cat(freqs, freqs) undo
            let axis = if d % 3 == 1 && d < sec_h {
                1
            } else if d % 3 == 2 && d < sec_w {
                2
            } else {
                0
            };
            let pos_val = pos[li * 3 + axis];
            // inv = base^(-d/half), bf16-rounded to match the real model's buffer.
            let inv_f32 = ((-(d as f64) / half_dim as f64) * log_theta).exp() as f32;
            let inv = bf16::from_f32(inv_f32).to_f32();
            // F64 range reduction → accurate trig at large angle.
            let angle = (pos_val * inv) as f64;
            let k = (angle / two_pi + 0.5).floor();
            let reduced = angle - k * two_pi;
            let idx = li * head_dim + dd;
            cos_v[idx] = reduced.cos() as f32;
            sin_v[idx] = reduced.sin() as f32;
        }
    }
    let cos = Tensor::from_vec(cos_v, Shape::from_dims(&[1, l, head_dim]), device.clone())?;
    let sin = Tensor::from_vec(sin_v, Shape::from_dims(&[1, l, head_dim]), device)?;
    Ok((cos, sin))
}

/// The four packed tensors `ideogram4_forward` consumes (the `predict_velocity`
/// build block). `SEQ = NT + GH*GW`. Assumes b=1, text_mask all-ones.
pub struct PackedInputs {
    /// [1, SEQ, 128] — text region zeroed, image region = patchified latents.
    pub x: Tensor,
    /// [1, SEQ, 53248] — image region zeroed.
    pub llm_full: Tensor,
    /// [1, SEQ, 3] F32 — text rows [i,i,i]; image rows [off, h+off, w+off].
    pub position_ids: Tensor,
    /// [1, SEQ] F32 — text -> 3, image -> 2.
    pub indicator: Tensor,
}

/// Build the packed `[text ++ image]` transformer inputs.
/// `noisy`: [1, 128, GH, GW] F32. `llm`: [1, NT, 53248].
pub fn build_packed_inputs(
    noisy: &Tensor,
    llm: &Tensor,
    gh: usize,
    gw: usize,
    device: Arc<CudaDevice>,
) -> Result<PackedInputs> {
    let nt = llm.shape().dims()[1];
    let nimg = gh * gw;
    let seq = nt + nimg;

    // image_tokens = noisy.permute(0,2,3,1).reshape(1, nimg, 128)  (h outer, w inner)
    let image_tokens = noisy
        .permute(&[0, 2, 3, 1])?
        .reshape(&[1, nimg, PACKED_CHANNELS])?;

    // x = cat([zeros(1, nt, 128), image_tokens], dim=1)  (text region zeroed).
    // zeros must match the image-token dtype or cat rejects the dtype mismatch.
    let text_zeros = Tensor::zeros_dtype(
        Shape::from_dims(&[1, nt, PACKED_CHANNELS]),
        image_tokens.dtype(),
        device.clone(),
    )?;
    let x = Tensor::cat(&[&text_zeros, &image_tokens], 1)?;

    // llm_full = cat([llm, zeros(1, nimg, 53248)], dim=1)  (image region zeroed)
    let llm_zeros = Tensor::zeros_dtype(
        Shape::from_dims(&[1, nimg, TEXT_FEATURE_DIM]),
        llm.dtype(),
        device.clone(),
    )?;
    let llm_full = Tensor::cat(&[llm, &llm_zeros], 1)?;

    // position_ids [1, SEQ, 3]: text rows [i,i,i] (cumsum(all-ones)-1 == i);
    // image rows [t=0, h, w] + IMAGE_OFFSET, with h outer (tok/GW), w inner (tok%GW).
    let mut pos = Vec::<f32>::with_capacity(seq * 3);
    for i in 0..nt {
        let v = i as f32;
        pos.push(v);
        pos.push(v);
        pos.push(v);
    }
    let off = IMAGE_OFFSET as f32;
    for tok in 0..nimg {
        let h = (tok / gw) as f32;
        let w = (tok % gw) as f32;
        pos.push(off);
        pos.push(h + off);
        pos.push(w + off);
    }
    let position_ids = Tensor::from_vec(pos, Shape::from_dims(&[1, seq, 3]), device.clone())?;

    // indicator [1, SEQ]: text -> LLM_TOKEN_INDICATOR (3), image -> OUTPUT_IMAGE_INDICATOR (2).
    let mut ind = Vec::<f32>::with_capacity(seq);
    ind.resize(nt, LLM_TOKEN_INDICATOR);
    ind.resize(seq, OUTPUT_IMAGE_INDICATOR);
    let indicator = Tensor::from_vec(ind, Shape::from_dims(&[1, seq]), device)?;

    Ok(PackedInputs {
        x,
        llm_full,
        position_ids,
        indicator,
    })
}
