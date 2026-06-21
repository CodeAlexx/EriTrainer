//! DoRA (Weight-Decomposed LoRA) shared helper.
//!
//! DoRA = magnitude × normalized direction:
//!     `WP_dora = m * (WP / (||WP||_2.detach() + eps))`
//! where `WP = W_orig + ΔW` is the full reconstructed weight and `m` is a
//! learned per-axis magnitude initialized to `||W_orig||_2` along the
//! non-decomposed axis. The L2 norm is **detached** for backward (paper §4.3),
//! so gradients flow only through `m` and `WP`, not through the
//! renormalization itself.
//!
//! References:
//! - Paper: "DoRA: Weight-Decomposed Low-Rank Adaptation" (NVIDIA 2024).
//! - OneTrainer: `modules/module/LoRAModule.py:473-571` (`DoRAModule`).
//! - lycoris-upstream: `lycoris/modules/locon.py:239-260`
//!   (`apply_weight_decompose`).
//!
//! # Save format
//! When DoRA is enabled on an adapter, emit one extra tensor next to
//! `.lora_up`/`.lora_down`:
//!
//! - `<prefix>.dora_scale` — the magnitude tensor of shape `[out, 1]`
//!   for Linear or `[out, 1, 1, 1]` for Conv2d (when `wd_on_out=true`,
//!   the lycoris-upstream default). When `wd_on_out=false`, the shape is
//!   transposed to `[1, in]` / `[1, in, 1, 1]`.
//!
//! Loader convention (Phase 2 work in `loader.rs`): presence of the
//! `.dora_scale` key → DoRA is on for this adapter; absence → plain
//! adapter forward path. PEFT's `.magnitude_vector` should also be
//! accepted as an alias on load.
//!
//! # `wd_on_out` axis convention
//! - `wd_on_out = true` (lycoris-upstream default): norm is taken along
//!   the **input dims** of `WP`. For Linear `[out, in]`, reduce dim 1.
//!   For Conv2d `[out, in, kh, kw]`, reduce dims `{1, 2, 3}`. Magnitude
//!   shape is `[out, 1]` / `[out, 1, 1, 1]`.
//! - `wd_on_out = false` (OneTrainer default — note the flipped default):
//!   norm is taken along the **output dim**. For Linear, reduce dim 0;
//!   shape `[1, in]`. For Conv2d, reduce dim 0; shape `[1, in, kh, kw]`.
//!
//! # Integration plan (follow-up agent)
//!
//! Phase 1.1 lands constructor changes for `LoConModule`/`LoHaModule`/
//! `LoKrModule` in parallel; this module deliberately stays standalone so
//! the two streams of edits don't conflict. The follow-up agent should:
//!
//! 1. Add a `decompose: bool` arg to:
//!    - `LoConModule::new_linear` (`src/algorithms/locon.rs`)
//!    - `LoHaModule::new_linear` (`src/algorithms/loha.rs`)
//!    - `LoKrModule::new_linear` (`src/algorithms/lokr.rs`)
//!    - Future `OFTModule::new_linear` (when ported).
//! 2. Add an optional `dora_scale: Option<Tensor>` field on each struct
//!    plus a `wd_on_out: bool` config knob (default `true` to match
//!    lycoris-upstream).
//! 3. In each constructor, when `decompose=true`, allocate the magnitude
//!    via [`init_magnitude`] from the base weight and store it.
//! 4. In each `forward` / `get_diff_weight`, when `dora_scale.is_some()`:
//!    - Reconstruct `WP = base + ΔW` (algo-specific).
//!    - Call [`apply_weight_decompose`] with `m`, `wd_on_out`, `eps`.
//!    - Use the returned `WP_dora` as the effective weight (replaces the
//!      plain `base + ΔW` path; caller must NOT add base separately).
//! 5. Save: when `dora_scale.is_some()`, emit `<prefix>.dora_scale` in
//!    the safetensors writer alongside `lora_up`/`lora_down`.
//! 6. Load (`src/loader.rs:217-238`): drop the loud-skip branch and pass
//!    the parsed `.dora_scale` tensor into the algo builder. Also accept
//!    `.magnitude_vector` as an alias.
//!
//! # Detach plan
//! Uses [`flame_core::Tensor::detach`] (verified at `flame-core/src/
//! tensor.rs:3052`). The norm tensor is `detach`-ed before the divide,
//! so backward through the renormalization sees `||WP||` as a constant.

use crate::{Error, Result};
use flame_core::Tensor;

/// Apply DoRA weight decomposition to a fully-reconstructed weight tensor.
///
/// # Arguments
/// - `wp`: Full reconstructed weight `W + ΔW`.
///   - Linear: shape `[out, in]`.
///   - Conv2d: shape `[out, in, kh, kw]`.
/// - `m`: Magnitude tensor (learned parameter, F32 strongly recommended).
///   - `wd_on_out=true`: shape `[out, 1]` (Linear) or `[out, 1, 1, 1]` (Conv2d).
///   - `wd_on_out=false`: shape `[1, in]` (Linear) or `[1, in, kh, kw]` (Conv2d).
/// - `wd_on_out`: Axis convention.
///   - `true` (lycoris-upstream default): reduce input dims, preserve per-`out`.
///   - `false` (OneTrainer default): reduce output dim, preserve per-`in*`.
/// - `eps`: Epsilon added to the L2 norm before division. Set to
///   `f32::EPSILON` (≈1.19e-7) to mirror PyTorch `torch.finfo(float32).eps`,
///   or `0.0` if the OneTrainer `norm_epsilon=false` flag is desired.
///
/// # Returns
/// `m * (wp / (||wp||_2.detach() + eps))`, same shape as `wp`.
///
/// # Notes
/// - The norm is computed in `wp.dtype()`. For BF16 trainers, callers
///   should upcast `wp` to F32 first when the reduce-dim is wide
///   (>~1024) — BF16 sum-of-squares of a 4096-d vector loses precision
///   per the OT/lycoris convention.
/// - Autograd: gradient flows through `m` (multiplicative) and through
///   `wp` (via the divided form), but NOT through the norm denominator,
///   per paper §4.3.
pub fn apply_weight_decompose(
    wp: &Tensor,
    m: &Tensor,
    wd_on_out: bool,
    eps: f32,
) -> Result<Tensor> {
    let dims = wp.shape().dims();
    let reduce_dims: Vec<usize> = match (dims.len(), wd_on_out) {
        // Linear: [out, in]
        (2, true) => vec![1],   // reduce input axis → keepdim shape [out, 1]
        (2, false) => vec![0],  // reduce output axis → keepdim shape [1, in]
        // Conv2d: [out, in, kh, kw]
        (4, true) => vec![1, 2, 3], // reduce input + kernel dims
        (4, false) => vec![0],      // reduce output dim
        _ => {
            return Err(Error::InvalidOperation(format!(
                "apply_weight_decompose: WP must be 2D (Linear) or 4D (Conv2d), got rank {}",
                dims.len()
            )));
        }
    };

    // Compute ||WP||_2 along reduce_dims with keepdim, then DETACH so the
    // gradient does not flow through the renormalization (paper §4.3).
    //
    // Strategy: square → sum_dim_keepdim repeatedly along each reduce dim
    // (each keeps rank, so subsequent indices are still valid) → sqrt.
    let sq = wp.square()?;
    let mut norm = sq;
    for &d in &reduce_dims {
        norm = norm.sum_dim_keepdim(d)?;
    }
    let norm = norm.sqrt()?;

    // Detach: cuts the autograd edge through the norm so that backward
    // sees `||WP||` as a constant. Verified API at
    // `flame-core/src/tensor.rs:3052`.
    let norm = norm.detach()?;

    // Add epsilon to avoid div-by-zero on near-zero columns/rows.
    let denom = if eps > 0.0 {
        norm.add_scalar(eps)?
    } else {
        norm
    };

    // wp / denom — div() broadcasts internally (verified at
    // `flame-core/src/tensor_ops_extended.rs:1054-1078`), so the
    // `[out, 1]`-shaped `denom` lifts to `[out, in]` automatically.
    let direction = wp.div(&denom)?;

    // m * direction — mul() does NOT broadcast internally (verified at
    // `flame-core/src/tensor.rs:1915`), so we broadcast `m` to wp's shape
    // explicitly. Going through `broadcast_to` keeps autograd intact.
    let m_broadcast = if m.shape().dims() == dims {
        m.clone_result()?
    } else {
        m.broadcast_to(wp.shape())?
    };
    Ok(m_broadcast.mul(&direction)?)
}

/// Initialize the DoRA magnitude tensor from the original (pre-LoRA) weight.
///
/// Returns `||W_orig||_2 + eps` along the appropriate axis with `keepdim=true`.
///
/// # Arguments
/// - `w_orig`: Pre-LoRA base weight.
///   - Linear: `[out, in]`.
///   - Conv2d: `[out, in, kh, kw]`.
/// - `wd_on_out`: Same convention as [`apply_weight_decompose`].
/// - `eps`: Optional epsilon to add before storing (set to `0.0` to store
///   the bare norm; OT default is `0.0` here, with epsilon applied only
///   in the per-step divide).
///
/// # Returns
/// Magnitude tensor of shape `[out, 1, ...]` (if `wd_on_out=true`) or
/// `[1, in, ...]` (if `wd_on_out=false`). Same dtype as `w_orig`.
///
/// # Notes
/// - Caller is responsible for converting to F32 if the model trains in
///   BF16 — DoRA's `m` must be F32 storage to avoid norm-precision drift
///   (see scope §5.2). The most common pattern is:
///       let m = init_magnitude(&w_orig.to_dtype(F32)?, true, 0.0)?;
/// - Caller is responsible for the `requires_grad = true` flip and for
///   wiring the parameter into the optimizer state.
pub fn init_magnitude(w_orig: &Tensor, wd_on_out: bool, eps: f32) -> Result<Tensor> {
    let dims = w_orig.shape().dims();
    let reduce_dims: Vec<usize> = match (dims.len(), wd_on_out) {
        (2, true) => vec![1],
        (2, false) => vec![0],
        (4, true) => vec![1, 2, 3],
        (4, false) => vec![0],
        _ => {
            return Err(Error::InvalidOperation(format!(
                "init_magnitude: w_orig must be 2D (Linear) or 4D (Conv2d), got rank {}",
                dims.len()
            )));
        }
    };

    // Same square→sum_dim_keepdim→sqrt pattern as forward; init runs once
    // outside autograd, so detach is unnecessary but harmless if added.
    let sq = w_orig.square()?;
    let mut norm = sq;
    for &d in &reduce_dims {
        norm = norm.sum_dim_keepdim(d)?;
    }
    let norm = norm.sqrt()?;

    if eps > 0.0 {
        Ok(norm.add_scalar(eps)?)
    } else {
        Ok(norm)
    }
}
