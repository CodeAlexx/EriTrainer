//! AsymFlow loss + helpers.
//!
//! Milestone A1 of the AsymFlow trainer port (see
//! `EriDiffusion-v2/docs/asymflow_milestone_plan.md` §A1).
//!
//! Faithful port of LakonLab `lakonlab/models/diffusions/asymflow.py` lines
//! 12-119 (Apache 2.0, © 2026 Hansheng Chen). Three public entry points:
//!
//!   1. [`calc_shifted_signal_ratio`]   — mirror of `asymflow.py:12-15`.
//!   2. [`compute_asymflow_target`]     — mirror of `asymflow.py:93-111` (target
//!      construction: adaptive residual weighting + shifted-signal blend).
//!   3. [`asymflow_loss`]               — mirror of `asymflow.py:112-119`
//!      (velocity-weighted MSE in `pred_x_0` space).
//!
//! ## Deferred branches
//!
//! - **LPIPS / perceptual loss** (`asymflow.py:121-138`) — DEFERRED per the
//!   plan §2. `log_vars["loss_perceptual"] = 0.0` is reported unconditionally
//!   so downstream telemetry slots stay populated. Re-evaluate after A4.5.
//!
//! ## Why local `patchify_unpacked` / `unpatchify_unpacked` helpers
//!
//! The plan calls for reusing the four primitives in
//! `inference-flame::models::asymflux2`, but the published Rust `patchify` /
//! `unpatchify` only support `pack_channels = True` / `packed_channels = True`
//! (the rank-4 `(B, C*p², H/p, W/p)` packing used by the inference forward
//! pass). The AsymFlow loss path explicitly invokes `pack_channels = False`
//! and `packed_channels = False` (`asymflow.py:95, 98, 111, 132`), yielding a
//! rank-5 `(B, C, p², H/p, W/p)` layout where the patch-pixel axis is
//! retained so it can be reduced via `.mean(dim=2, keepdim=True)`.
//!
//! Adding the unpacked overloads to `inference-flame` would be a one-line
//! change but is outside this milestone's scope (constraint: no edits beyond
//! the four files listed in the prompt). The two helpers below mirror the
//! reference `pack_channels = False` branch using only existing flame-core
//! `reshape` / `permute` ops — no new primitives are introduced.
//!
//! ## Dtype convention
//!
//! Following the neighboring `loss_weight.rs` / `masked_loss.rs` modules,
//! all reductions happen at the input dtype. The caller is expected to have
//! already up-cast `x_0`, `latents_2`, `noise`, etc. to a consistent dtype
//! (BF16 or F32). Scalars (`sigma`, `loss_shift`) are `f32`.
//!
//! ## Shapes (rank-4 image data, the only case A1 exercises)
//!
//! For `B` batch, `C` channels (post-VAE-encode), `H × W` spatial, with
//! `patch_size = p` and `base_rank = k`:
//!
//! | Name                | Shape                              |
//! |---------------------|------------------------------------|
//! | `x_0`               | `(B, C_full, H, W)`                |
//! | `latents_2`         | `(B, C_low,  H, W)`                |
//! | `noise`             | `(B, C_full, H, W)` (`randn_like(x_0)`) |
//! | `proj_mat`          | `(C_full * p², k)` (Procrustes)    |
//! | `scale_buffer (s)`  | scalar                             |
//! | `student_pred_u`    | `(B, C_full, H, W)`                |
//! | `teacher_pred_u`    | `(B, C_full, H, W)`                |
//! | `sigma`             | scalar broadcast over B (taken as `f32`) |
//!
//! Rank-5 video tensors are out of scope for A1 (Klein is 2-D); the helpers
//! defensively reject non-rank-4 input.

use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;

// ───────────────────────────────────────────────────────────────────────────
// Public API
// ───────────────────────────────────────────────────────────────────────────

/// Shifted signal-to-noise ratio per `asymflow.py:12-15`:
///
/// ```python
/// def calc_shifted_signal_ratio(sigma, shift):
///     alpha = 1 - sigma
///     alpha_sq = alpha.square()
///     return alpha_sq / (alpha_sq + (shift * sigma).square())
/// ```
///
/// In our flow the timestep sampler is scalar-per-step (single `sigma` value
/// shared across the whole minibatch — Klein's `timestep_sampler` returns one
/// number per step, which is then broadcast). We therefore take `sigma` as a
/// plain `f32` and return a plain `f32` rather than a 1-element tensor; this
/// matches how `asymflow_loss` consumes the result (a single scalar multiplier
/// applied to a `(B, C, H, W)` residual block).
///
/// The Python signature accepts a tensor `sigma`, but every call site in the
/// reference uses a per-batch broadcastable scalar; the closed-form here is
/// numerically identical.
pub fn calc_shifted_signal_ratio(sigma: f32, shift: f32) -> f32 {
    let alpha = 1.0 - sigma;
    let alpha_sq = alpha * alpha;
    let shifted = shift * sigma;
    alpha_sq / (alpha_sq + shifted * shifted)
}

/// Intermediates returned alongside the AsymFlow target so callers (the
/// trainer step) can log them or feed them into the deferred LPIPS branch.
#[derive(Debug)]
pub struct AsymFlowTargetParts {
    /// Mean of the adaptive residual coefficient `res_coef`, as `f32`. Logged
    /// per the reference (`asymflow.py:118`).
    pub res_coef_mean: f32,
    /// Whether the shifted-signal-ratio shortcut was taken (caller asked for
    /// `loss_shift = None` ≡ `loss_shift = Some(_)` not provided). When false
    /// this is the `0.0` Python branch.
    pub used_loss_shift: bool,
    /// The scalar `shifted_signal_ratio` value actually used. `0.0` when
    /// `loss_shift = None`.
    pub shifted_signal_ratio: f32,
}

/// Compute the AsymFlow training target `tgt_x_0` and the auxiliary
/// `res_coef_mean` log value. Faithful port of `asymflow.py:93-111`.
///
/// ## Mapping from the Python forward_train
///
/// The reference function does a lot of work the trainer (A3) is responsible
/// for — sampling sigma, building `x_t`, running the student/teacher forwards
/// to obtain `pred_u` / `ref_u_low_rank`, and converting `pred_u → pred_x_0`
/// via `u_to_x_0(pred_u, x_t, sigma) = x_t - sigma * u`. By the time control
/// reaches this function we expect the trainer to have already produced:
///
///   - `x_0`              full-rank target (VAE / Oklab latent of the image)
///   - `x_0_low_rank`     low-rank reconstruction of `x_0`, equal to
///                        `unpatchify(unpack(pack(patchify(latents_2)) @ Pᵀ * s))`
///                        — i.e. `latents_2` lifted back into full-rank space.
///   - `pred_x_0`         student `u_to_x_0(student_pred_u, x_t, sigma)`.
///   - `ref_x_0_low_rank` teacher `u_to_x_0(teacher_pred_u, x_t_low_rank, sigma)`.
///
/// We chose to receive `(x_0, x_0_low_rank, pred_x_0, ref_x_0_low_rank,
/// patch_size, sigma, loss_shift)` rather than the full
/// `(x_0, latents_2, noise, student_pred_u, teacher_pred_u, …)` because the
/// `latents_2 → x_0_low_rank` lift requires `proj_mat` + `scale_buffer` which
/// the trainer already loads (per `extract_asymflow_buffers` in
/// `inference-flame::models::asymflux2:233`) and feeds into the teacher
/// forward as well. Keeping that single source of truth in the trainer
/// avoids double-loading the buffers here and keeps this module pure-math.
///
/// `loss_shift = None` matches Python's `if self.loss_shift is None: srr = 0`.
pub fn compute_asymflow_target(
    x_0: &Tensor,
    x_0_low_rank: &Tensor,
    pred_x_0: &Tensor,
    ref_x_0_low_rank: &Tensor,
    patch_size: usize,
    sigma: f32,
    loss_shift: Option<f32>,
) -> Result<(Tensor, AsymFlowTargetParts)> {
    let eps: f32 = 1e-4;

    let dims = x_0.shape().dims().to_vec();
    if dims.len() != 4 {
        return Err(Error::InvalidInput(format!(
            "compute_asymflow_target expects rank-4 (B,C,H,W), got rank {}",
            dims.len()
        )));
    }
    for (name, t) in [
        ("x_0_low_rank", x_0_low_rank),
        ("pred_x_0", pred_x_0),
        ("ref_x_0_low_rank", ref_x_0_low_rank),
    ] {
        if t.shape().dims() != dims.as_slice() {
            return Err(Error::InvalidInput(format!(
                "compute_asymflow_target: shape mismatch — x_0 is {:?} but {} is {:?}",
                dims,
                name,
                t.shape().dims()
            )));
        }
    }
    if patch_size == 0 {
        return Err(Error::InvalidInput(
            "compute_asymflow_target: patch_size must be > 0".into(),
        ));
    }

    // ── Python: low_rank_diff = patchify(x_0_low_rank - ref_x_0_low_rank, p,
    //                                    pack_channels=False)
    //          full_rank_diff = patchify(x_0 - pred_x_0.detach(), p,
    //                                    pack_channels=False)
    let low_rank_residual = x_0_low_rank.sub(ref_x_0_low_rank)?;
    let pred_x_0_detached = pred_x_0.detach()?;
    let full_rank_residual = x_0.sub(&pred_x_0_detached)?;

    let low_rank_diff = patchify_unpacked(&low_rank_residual, patch_size)?;
    let full_rank_diff = patchify_unpacked(&full_rank_residual, patch_size)?;

    // ── Python:
    //   num = (full_rank_diff * low_rank_diff).mean(dim=2, keepdim=True)
    //   den = low_rank_diff.square().mean(dim=2, keepdim=True).clamp(min=eps)
    //   res_coef = (num / den).clamp_(0.0, 1.0)
    let num = full_rank_diff
        .mul(&low_rank_diff)?
        .mean_dim(&[2], true)?;
    let den_raw = low_rank_diff.square()?.mean_dim(&[2], true)?;
    // clamp(min=eps): no max in Python, but our clamp requires both bounds.
    // Use f32::MAX as the upper bound — equivalent to "no upper clip".
    let den = den_raw.clamp(eps, f32::MAX)?;
    let res_coef = num.div(&den)?.clamp(0.0, 1.0)?;

    let res_coef_mean = tensor_scalar_mean_f32(&res_coef)?;

    // ── Python:
    //   if self.loss_shift is None: shifted_signal_ratio = 0.0
    //   else: shifted_signal_ratio = calc_shifted_signal_ratio(sigma, shift)
    let (shifted_signal_ratio, used_loss_shift) = match loss_shift {
        None => (0.0_f32, false),
        Some(shift) => (calc_shifted_signal_ratio(sigma, shift), true),
    };

    // ── Python:
    //   tgt_x_0 = x_0 - (1 - shifted_signal_ratio) * unpatchify(
    //       res_coef * low_rank_diff, p, packed_channels=False)
    //
    // `res_coef * low_rank_diff` broadcasts `(B, C, 1,    H/p, W/p)`
    //                                    over `(B, C, p², H/p, W/p)`.
    let res_dims = res_coef.shape().dims().to_vec();
    let diff_dims = low_rank_diff.shape().dims().to_vec();
    if res_dims.len() != 5 || diff_dims.len() != 5 {
        return Err(Error::InvalidInput(format!(
            "compute_asymflow_target: unpacked-patchify produced unexpected rank \
             (res_coef rank {}, low_rank_diff rank {})",
            res_dims.len(),
            diff_dims.len()
        )));
    }
    let res_coef_broadcast = res_coef.broadcast_to(&Shape::from_dims(&diff_dims))?;
    let weighted_low_rank_diff = res_coef_broadcast.mul(&low_rank_diff)?;
    let unpatched = unpatchify_unpacked(&weighted_low_rank_diff, patch_size)?;

    let blend_scale = 1.0 - shifted_signal_ratio;
    let scaled = unpatched.mul_scalar(blend_scale)?;
    let tgt_x_0 = x_0.sub(&scaled)?;

    Ok((
        tgt_x_0,
        AsymFlowTargetParts {
            res_coef_mean,
            used_loss_shift,
            shifted_signal_ratio,
        },
    ))
}

/// Final AsymFlow scalar loss + log dict — mirror of `asymflow.py:112-119`.
///
/// ```python
/// mse_loss = F.mse_loss(pred_x_0, tgt_x_0, reduction='none')
/// mse_loss = 0.5 * self.mse_loss_weight * (mse_loss / sigma_clamped.square()).mean()
/// loss = mse_loss
/// log_vars = dict(loss_diffusion=float(mse_loss), res_coef=float(res_coef.mean()))
/// ```
///
/// `sigma_clamped` corresponds to `sigma.clamp(min=self.sigma_min)` at
/// `asymflow.py:63`; we take the already-clamped value as `f32` so the
/// trainer owns `sigma_min` (the default is `1e-4`, per
/// `gaussian_flow.py:49`).
///
/// The deferred LPIPS branch is reported as
/// `log_vars["loss_perceptual"] = 0.0`. The Python `loss` term does not
/// include LPIPS when `perceptual_loss is None` — we match that.
pub fn asymflow_loss(
    pred_x_0: &Tensor,
    tgt_x_0: &Tensor,
    sigma_clamped: f32,
    mse_loss_weight: f32,
    target_parts: &AsymFlowTargetParts,
) -> Result<(Tensor, HashMap<String, f32>)> {
    if pred_x_0.shape().dims() != tgt_x_0.shape().dims() {
        return Err(Error::InvalidInput(format!(
            "asymflow_loss: pred_x_0 shape {:?} != tgt_x_0 shape {:?}",
            pred_x_0.shape().dims(),
            tgt_x_0.shape().dims()
        )));
    }
    if !sigma_clamped.is_finite() || sigma_clamped <= 0.0 {
        return Err(Error::InvalidInput(format!(
            "asymflow_loss: sigma_clamped must be finite and > 0, got {sigma_clamped}"
        )));
    }

    // F.mse_loss(..., reduction='none') == (pred - tgt) ** 2 elementwise.
    let diff = pred_x_0.sub(tgt_x_0)?;
    let sq = diff.square()?;
    // (sq / sigma_clamped**2).mean()  ≡  sq.mean() * (1 / sigma_clamped**2)
    // since sigma_clamped is a per-step scalar broadcast across all elements.
    // Doing the divide-then-mean form is slightly more arithmetic; the
    // mean-then-multiply form is numerically equivalent in F32 and avoids an
    // extra elementwise op on a large tensor.
    let mean_sq = sq.mean()?;
    let scale = 0.5 * mse_loss_weight / (sigma_clamped * sigma_clamped);
    let mse_loss = mean_sq.mul_scalar(scale)?;

    // log_vars: pull scalars off the GPU so downstream board logging can
    // serialize them without holding a tensor reference. Matches how
    // loss_weight.rs / masked_loss.rs return Tensor while a thin caller does
    // the scalar extraction — we do it inline here because the dict is
    // human-readable telemetry, not a hot-path tensor.
    let loss_scalar = tensor_scalar_mean_f32(&mse_loss)?;
    let mut log_vars: HashMap<String, f32> = HashMap::new();
    log_vars.insert("loss_diffusion".to_string(), loss_scalar);
    log_vars.insert("res_coef".to_string(), target_parts.res_coef_mean);
    // TODO: LPIPS (deferred per asymflow_milestone_plan.md section 2).
    // The full Python adds `perceptual_loss` into `loss` and pushes
    // `loss_perceptual` into `log_vars` only when `self.perceptual_loss
    // is not None`. We report a constant 0.0 so the trainer's CSV/board
    // schema stays uniform across runs.
    log_vars.insert("loss_perceptual".to_string(), 0.0);

    Ok((mse_loss, log_vars))
}

// ───────────────────────────────────────────────────────────────────────────
// Private helpers — unpacked patchify / unpatchify (see module docstring).
// ───────────────────────────────────────────────────────────────────────────

/// Patchify with `pack_channels=False`. Mirror of
/// `asymflux2.py:124-137` (else-branch):
///
/// ```python
/// latents = latents.reshape(bs, c, h/p, p, w/p, p).permute(0, 1, 3, 5, 2, 4)
/// latents = latents.reshape(bs, c, p*p, h/p, w/p)
/// ```
///
/// Returns a rank-5 tensor `(B, C, p², H/p, W/p)` with the patch-pixel axis
/// at dim 2 (so a `.mean_dim(&[2], true)` reduces over the `p²` patch pixels
/// the way the Python `mean(dim=2, keepdim=True)` does).
fn patchify_unpacked(x: &Tensor, patch_size: usize) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 4 {
        return Err(Error::InvalidInput(format!(
            "patchify_unpacked expects rank-4 (B,C,H,W), got rank {}",
            dims.len()
        )));
    }
    if patch_size == 0 {
        return Err(Error::InvalidInput(
            "patchify_unpacked: patch_size must be > 0".into(),
        ));
    }
    let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    if h % patch_size != 0 || w % patch_size != 0 {
        return Err(Error::InvalidInput(format!(
            "patchify_unpacked: H={h} and W={w} must both be multiples of patch_size={patch_size}"
        )));
    }
    let h_p = h / patch_size;
    let w_p = w / patch_size;

    // (B, C, H, W) → (B, C, H/p, p, W/p, p)
    let x = x.reshape(&[b, c, h_p, patch_size, w_p, patch_size])?;
    // permute(0, 1, 3, 5, 2, 4) → (B, C, p, p, H/p, W/p)
    let x = x.permute(&[0, 1, 3, 5, 2, 4])?;
    // reshape to (B, C, p², H/p, W/p) — materializes the view.
    x.reshape(&[b, c, patch_size * patch_size, h_p, w_p])
}

/// Inverse of [`patchify_unpacked`]. Mirror of
/// `asymflux2.py:140-156` (else-branch):
///
/// ```python
/// latents = latents.reshape(bs, c, p, p, h, w).permute(0, 1, 4, 2, 5, 3)
/// latents = latents.reshape(bs, c, h*p, w*p)
/// ```
///
/// Input is rank-5 `(B, C, p², H/p, W/p)`; output is rank-4 `(B, C, H, W)`.
fn unpatchify_unpacked(x: &Tensor, patch_size: usize) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 5 {
        return Err(Error::InvalidInput(format!(
            "unpatchify_unpacked expects rank-5 (B,C,p²,H/p,W/p), got rank {}",
            dims.len()
        )));
    }
    if patch_size == 0 {
        return Err(Error::InvalidInput(
            "unpatchify_unpacked: patch_size must be > 0".into(),
        ));
    }
    let p_sq = patch_size * patch_size;
    let (b, c, p2, h_p, w_p) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    if p2 != p_sq {
        return Err(Error::InvalidInput(format!(
            "unpatchify_unpacked: dim 2 must equal patch_size²={p_sq}, got {p2}"
        )));
    }

    // (B, C, p², H/p, W/p) → (B, C, p, p, H/p, W/p)
    let x = x.reshape(&[b, c, patch_size, patch_size, h_p, w_p])?;
    // permute(0, 1, 4, 2, 5, 3) → (B, C, H/p, p, W/p, p)
    let x = x.permute(&[0, 1, 4, 2, 5, 3])?;
    // reshape to (B, C, H, W).
    x.reshape(&[b, c, h_p * patch_size, w_p * patch_size])
}

/// Pull a scalar tensor (any shape) off the GPU as `f32` by mean-reducing.
/// For shape `[]` / `[1]` this is a no-op-ish identity; for general shapes it
/// matches Python's `float(t.mean())` semantics used in `log_vars`.
fn tensor_scalar_mean_f32(t: &Tensor) -> Result<f32> {
    let m = if t.shape().dims().is_empty() || t.shape().elem_count() == 1 {
        t.clone()
    } else {
        t.mean()?
    };
    let m_f32 = if m.dtype() == DType::F32 {
        m
    } else {
        m.to_dtype(DType::F32)?
    };
    let v = m_f32.to_vec()?;
    if v.is_empty() {
        return Err(Error::InvalidInput(
            "tensor_scalar_mean_f32: empty tensor".into(),
        ));
    }
    Ok(v[0])
}

// ───────────────────────────────────────────────────────────────────────────
// Tests — code-only milestone, NO GPU `cargo test` runs.
//
// Strategy: provide deterministic self-consistency tests that exercise every
// branch (degenerate target, residual round-trip, scalar formula, log_vars
// schema). The plan's golden-file test is `#[ignore]` since the dump file
// listed in §A1 acceptance does not exist yet — A0.5 report flagged that
// LakonLab can't be installed locally to generate it.
// ───────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use flame_core::{global_cuda_device, Shape, Tensor};

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    // ── calc_shifted_signal_ratio: pure-arithmetic, no GPU. ────────────────

    #[test]
    fn ssr_at_sigma_zero_is_one() {
        // alpha = 1, alpha² = 1; denom = 1 + 0 → ssr = 1.
        let s = calc_shifted_signal_ratio(0.0, 3.0);
        assert!(approx(s, 1.0, 1e-7), "got {s}");
    }

    #[test]
    fn ssr_at_sigma_one_is_zero() {
        // alpha = 0; numer = 0 → ssr = 0.
        let s = calc_shifted_signal_ratio(1.0, 3.0);
        assert!(approx(s, 0.0, 1e-7), "got {s}");
    }

    #[test]
    fn ssr_at_shift_one_is_alpha_sq_over_alpha_sq_plus_sigma_sq() {
        // shift = 1 collapses to the unshifted alpha²/(alpha² + sigma²).
        let sigma = 0.3_f32;
        let alpha = 1.0 - sigma;
        let expected = alpha * alpha / (alpha * alpha + sigma * sigma);
        let actual = calc_shifted_signal_ratio(sigma, 1.0);
        assert!(approx(actual, expected, 1e-7), "{actual} != {expected}");
    }

    #[test]
    fn ssr_decreases_with_larger_shift() {
        // For fixed sigma in (0, 1), shift ↑ ⇒ denominator ↑ ⇒ ratio ↓.
        let sigma = 0.5;
        let lo = calc_shifted_signal_ratio(sigma, 0.5);
        let hi = calc_shifted_signal_ratio(sigma, 3.0);
        assert!(hi < lo, "expected ssr({sigma},3.0)={hi} < ssr({sigma},0.5)={lo}");
        assert!(lo <= 1.0 && hi >= 0.0);
    }

    // ── patchify_unpacked + unpatchify_unpacked round-trip. ────────────────
    //
    // These DO touch the GPU at Tensor construction (allowed per the prompt:
    // "no model forward, no real safetensors load"). They run a couple of
    // tens of f32 elements through reshape/permute — no kernels are
    // recompiled, no real model is loaded.

    #[test]
    fn patchify_unpacked_round_trip() {
        let device = global_cuda_device();
        // (B=1, C=2, H=4, W=4), p=2 → (1, 2, 4, 2, 2).
        let b = 1usize;
        let c = 2usize;
        let h = 4usize;
        let w = 4usize;
        let p = 2usize;
        let n = b * c * h * w;
        let data: Vec<f32> = (0..n).map(|i| i as f32 + 0.5).collect();
        let x = Tensor::from_vec(data.clone(), Shape::from_dims(&[b, c, h, w]), device).unwrap();

        let patched = patchify_unpacked(&x, p).unwrap();
        assert_eq!(patched.shape().dims(), &[b, c, p * p, h / p, w / p]);

        let restored = unpatchify_unpacked(&patched, p).unwrap();
        assert_eq!(restored.shape().dims(), &[b, c, h, w]);

        let out = restored.to_vec().unwrap();
        assert_eq!(out.len(), n);
        for (i, (a, b)) in data.iter().zip(out.iter()).enumerate() {
            assert!(
                approx(*a, *b, 1e-6),
                "round-trip mismatch at i={i}: in={a} out={b}"
            );
        }
    }

    #[test]
    fn patchify_unpacked_shape_only_with_p3() {
        let device = global_cuda_device();
        // (1, 3, 6, 6), p=3 → (1, 3, 9, 2, 2).
        let n = 1 * 3 * 6 * 6;
        let x = Tensor::from_vec(
            vec![1.0_f32; n],
            Shape::from_dims(&[1, 3, 6, 6]),
            device.clone(),
        )
        .unwrap();
        let p = patchify_unpacked(&x, 3).unwrap();
        assert_eq!(p.shape().dims(), &[1, 3, 9, 2, 2]);
        let r = unpatchify_unpacked(&p, 3).unwrap();
        assert_eq!(r.shape().dims(), &[1, 3, 6, 6]);
    }

    #[test]
    fn patchify_unpacked_rejects_non_rank_4() {
        let device = global_cuda_device();
        let x = Tensor::from_vec(vec![0.0_f32; 4], Shape::from_dims(&[4]), device).unwrap();
        assert!(patchify_unpacked(&x, 2).is_err());
    }

    #[test]
    fn patchify_unpacked_rejects_indivisible_spatial() {
        let device = global_cuda_device();
        let x = Tensor::from_vec(
            vec![0.0_f32; 5 * 5],
            Shape::from_dims(&[1, 1, 5, 5]),
            device,
        )
        .unwrap();
        assert!(patchify_unpacked(&x, 2).is_err());
    }

    #[test]
    fn unpatchify_unpacked_rejects_wrong_psq() {
        let device = global_cuda_device();
        // p=2 ⇒ p²=4. Pass a tensor with dim 2 = 5 and expect error.
        let x = Tensor::from_vec(
            vec![0.0_f32; 1 * 1 * 5 * 2 * 2],
            Shape::from_dims(&[1, 1, 5, 2, 2]),
            global_cuda_device(),
        )
        .unwrap();
        assert!(unpatchify_unpacked(&x, 2).is_err());
        let _ = device; // silence unused warning if global_cuda_device is cheap-clone
    }

    // ── compute_asymflow_target degenerate cases. ──────────────────────────

    #[test]
    fn target_equals_x0_when_low_rank_residual_zero() {
        // If x_0_low_rank == ref_x_0_low_rank, low_rank_diff is identically 0
        // → res_coef = 0/eps clipped to [0,1] = 0
        // → tgt_x_0 = x_0 - (1-srr) * unpatchify(0) = x_0 - 0 = x_0.
        let device = global_cuda_device();
        let shape = Shape::from_dims(&[1, 2, 4, 4]);
        let n = shape.elem_count();
        let x_0 = Tensor::from_vec((0..n).map(|i| i as f32).collect(), shape.clone(), device.clone())
            .unwrap();
        let same = Tensor::from_vec(vec![7.0_f32; n], shape.clone(), device.clone()).unwrap();
        // pred_x_0 arbitrary; res_coef → 0 regardless.
        let pred_x_0 =
            Tensor::from_vec(vec![3.0_f32; n], shape.clone(), device.clone()).unwrap();

        let (tgt, parts) =
            compute_asymflow_target(&x_0, &same, &pred_x_0, &same, 2, 0.5, None).unwrap();

        // Shape preserved.
        assert_eq!(tgt.shape().dims(), &[1, 2, 4, 4]);

        // res_coef_mean is 0.0 (num is 0 ⇒ ratio is 0 ⇒ clamped to 0).
        assert!(
            approx(parts.res_coef_mean, 0.0, 1e-6),
            "res_coef_mean = {}",
            parts.res_coef_mean
        );

        // Element-wise equality vs x_0.
        let x_0_vec = x_0.to_vec().unwrap();
        let tgt_vec = tgt.to_vec().unwrap();
        for (i, (a, b)) in x_0_vec.iter().zip(tgt_vec.iter()).enumerate() {
            assert!(
                approx(*a, *b, 1e-5),
                "tgt[{i}] = {b}, expected x_0[{i}] = {a}"
            );
        }
    }

    #[test]
    fn target_finite_and_correct_shape_in_general_case() {
        let device = global_cuda_device();
        let shape = Shape::from_dims(&[2, 3, 8, 8]);
        let n = shape.elem_count();
        let x_0 =
            Tensor::from_vec((0..n).map(|i| (i % 17) as f32 * 0.1).collect(), shape.clone(), device.clone())
                .unwrap();
        let x_0_low_rank = Tensor::from_vec(
            (0..n).map(|i| ((i + 3) % 11) as f32 * 0.05).collect(),
            shape.clone(),
            device.clone(),
        )
        .unwrap();
        let pred_x_0 = Tensor::from_vec(
            (0..n).map(|i| ((i + 5) % 13) as f32 * 0.07).collect(),
            shape.clone(),
            device.clone(),
        )
        .unwrap();
        let ref_x_0_low_rank = Tensor::from_vec(
            (0..n).map(|i| ((i + 7) % 9) as f32 * 0.03).collect(),
            shape.clone(),
            device.clone(),
        )
        .unwrap();

        let (tgt, parts) = compute_asymflow_target(
            &x_0,
            &x_0_low_rank,
            &pred_x_0,
            &ref_x_0_low_rank,
            2,
            0.4,
            Some(3.0),
        )
        .unwrap();

        assert_eq!(tgt.shape().dims(), &[2, 3, 8, 8]);
        assert!(parts.used_loss_shift);
        let expected_srr = calc_shifted_signal_ratio(0.4, 3.0);
        assert!(
            approx(parts.shifted_signal_ratio, expected_srr, 1e-6),
            "{} != {}",
            parts.shifted_signal_ratio,
            expected_srr
        );
        assert!(parts.res_coef_mean >= 0.0 && parts.res_coef_mean <= 1.0);

        // All finite.
        for v in tgt.to_vec().unwrap() {
            assert!(v.is_finite(), "non-finite element in tgt_x_0: {v}");
        }
    }

    #[test]
    fn target_rejects_shape_mismatch() {
        let device = global_cuda_device();
        let s1 = Shape::from_dims(&[1, 2, 4, 4]);
        let s2 = Shape::from_dims(&[1, 2, 8, 8]);
        let a = Tensor::from_vec(vec![0.0_f32; s1.elem_count()], s1.clone(), device.clone()).unwrap();
        let b = Tensor::from_vec(vec![0.0_f32; s2.elem_count()], s2, device.clone()).unwrap();
        let r = compute_asymflow_target(&a, &b, &a, &a, 2, 0.5, None);
        assert!(r.is_err());
    }

    #[test]
    fn target_rejects_non_rank_4() {
        let device = global_cuda_device();
        let s = Shape::from_dims(&[4, 4]);
        let a = Tensor::from_vec(vec![0.0_f32; s.elem_count()], s.clone(), device.clone()).unwrap();
        let r = compute_asymflow_target(&a, &a, &a, &a, 2, 0.5, None);
        assert!(r.is_err());
    }

    // ── asymflow_loss. ─────────────────────────────────────────────────────

    #[test]
    fn loss_zero_when_pred_equals_target() {
        let device = global_cuda_device();
        let shape = Shape::from_dims(&[1, 2, 4, 4]);
        let n = shape.elem_count();
        let pred = Tensor::from_vec((0..n).map(|i| i as f32).collect(), shape.clone(), device.clone())
            .unwrap();
        let tgt = Tensor::from_vec((0..n).map(|i| i as f32).collect(), shape.clone(), device.clone())
            .unwrap();
        let parts = AsymFlowTargetParts {
            res_coef_mean: 0.25,
            used_loss_shift: false,
            shifted_signal_ratio: 0.0,
        };
        let (loss, log_vars) = asymflow_loss(&pred, &tgt, 0.5, 1.0, &parts).unwrap();
        let v = loss.to_vec().unwrap()[0];
        assert!(approx(v, 0.0, 1e-6), "expected 0 loss, got {v}");
        // log_vars schema invariants.
        assert!(log_vars.contains_key("loss_diffusion"));
        assert!(log_vars.contains_key("res_coef"));
        assert!(log_vars.contains_key("loss_perceptual"));
        assert!(approx(log_vars["loss_perceptual"], 0.0, 1e-9));
        assert!(approx(log_vars["res_coef"], 0.25, 1e-6));
    }

    #[test]
    fn loss_positive_when_pred_differs_from_target() {
        let device = global_cuda_device();
        let shape = Shape::from_dims(&[1, 1, 4, 4]);
        let n = shape.elem_count();
        let pred = Tensor::from_vec(vec![1.0_f32; n], shape.clone(), device.clone()).unwrap();
        let tgt = Tensor::from_vec(vec![0.0_f32; n], shape.clone(), device.clone()).unwrap();
        let parts = AsymFlowTargetParts {
            res_coef_mean: 0.0,
            used_loss_shift: false,
            shifted_signal_ratio: 0.0,
        };
        // diff = 1 everywhere, diff² = 1, mean = 1, scale = 0.5 * 1.0 / 0.5² = 2.
        let (loss, log_vars) = asymflow_loss(&pred, &tgt, 0.5, 1.0, &parts).unwrap();
        let v = loss.to_vec().unwrap()[0];
        assert!(approx(v, 2.0, 1e-5), "got {v}, expected 2.0");
        assert!(approx(log_vars["loss_diffusion"], 2.0, 1e-5));
    }

    #[test]
    fn loss_scales_with_mse_loss_weight() {
        let device = global_cuda_device();
        let shape = Shape::from_dims(&[1, 1, 2, 2]);
        let pred = Tensor::from_vec(vec![2.0_f32; 4], shape.clone(), device.clone()).unwrap();
        let tgt = Tensor::from_vec(vec![0.0_f32; 4], shape.clone(), device.clone()).unwrap();
        let parts = AsymFlowTargetParts {
            res_coef_mean: 0.0,
            used_loss_shift: false,
            shifted_signal_ratio: 0.0,
        };
        // Base: diff=2, sq=4, mean=4, scale = 0.5 * w / 1.0² = 0.5 * w.
        let (l1, _) = asymflow_loss(&pred, &tgt, 1.0, 1.0, &parts).unwrap();
        let (l10, _) = asymflow_loss(&pred, &tgt, 1.0, 10.0, &parts).unwrap();
        let v1 = l1.to_vec().unwrap()[0];
        let v10 = l10.to_vec().unwrap()[0];
        assert!(approx(v1, 2.0, 1e-5));
        assert!(approx(v10, 20.0, 1e-5));
    }

    #[test]
    fn loss_rejects_bad_sigma_clamped() {
        let device = global_cuda_device();
        let shape = Shape::from_dims(&[1, 1, 2, 2]);
        let pred = Tensor::from_vec(vec![1.0_f32; 4], shape.clone(), device.clone()).unwrap();
        let tgt = Tensor::from_vec(vec![0.0_f32; 4], shape.clone(), device.clone()).unwrap();
        let parts = AsymFlowTargetParts {
            res_coef_mean: 0.0,
            used_loss_shift: false,
            shifted_signal_ratio: 0.0,
        };
        assert!(asymflow_loss(&pred, &tgt, 0.0, 1.0, &parts).is_err());
        assert!(asymflow_loss(&pred, &tgt, f32::NAN, 1.0, &parts).is_err());
        assert!(asymflow_loss(&pred, &tgt, f32::INFINITY, 1.0, &parts).is_err());
    }

    // ── Plan-spec golden test — skipped, requires Python dump. ─────────────

    #[test]
    #[ignore = "needs tests/data/asymflow_golden.safetensors from LakonLab Python dump (A0.5 report: LakonLab unavailable locally)"]
    fn matches_python_golden_dump() {
        // When the golden file lands:
        //   1. Load (x_0, x_0_low_rank, pred_x_0, ref_x_0_low_rank, sigma,
        //      loss_shift, expected_tgt_x_0, expected_mse_loss) from
        //      tests/data/asymflow_golden.safetensors.
        //   2. Run compute_asymflow_target + asymflow_loss.
        //   3. Assert cos_sim(tgt_x_0, expected_tgt_x_0) >= 0.9999.
        //   4. Assert abs(mse_loss - expected_mse_loss) / expected_mse_loss < 0.01.
        unreachable!("ignored test; only runs when explicitly invoked");
    }
}
