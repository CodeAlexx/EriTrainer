//! Stochastic regularization for LyCORIS adapters.
//!
//! Mirrors upstream lycoris's three dropout knobs as documented at
//! `lycoris/modules/locon.py:43-66`.  Each algo's `forward` consults a
//! [`DropoutCfg`] to decide whether to:
//!
//! - **module_dropout** — zero the entire adapter delta for this step
//!   (skip the contribution).  Sampled once per `forward` from
//!   `Bernoulli(p)`.  This is the "skip whole adapter" knob; it does not
//!   touch individual leaves, just collapses the delta to zero so the
//!   downstream `base + delta` math returns base unchanged.
//!
//! - **rank_dropout** — Bernoulli mask over the rank axis applied to the
//!   intermediate `down_proj(x)` (shape `[..., rank]`).  When fired,
//!   entire rank dimensions go to zero for this step.  With
//!   `rank_dropout_scale=true` the surviving ranks scale by
//!   `1 / mean(mask)` to preserve expected magnitude (matches upstream
//!   `mid /= drop.mean()` at locon.py:297).
//!
//! - **dropout** — per-element Bernoulli on the *final* adapter delta
//!   (after `up_proj` and scale).  Standard PyTorch `nn.Dropout(p)`
//!   semantics with inverted scaling: surviving elements scale by
//!   `1 / (1 - p)` so expectation matches a no-dropout forward.  This
//!   reuses `flame_core::regularization::Dropout`.
//!
//! All three are no-ops when `training == false` or when the
//! corresponding `p == 0.0` — i.e. `DropoutCfg::default()` is byte-
//! identical to the pre-T1.2 forward path.
//!
//! ## Why not `flame_core::regularization::Dropout` for everything?
//!
//! `Dropout` covers per-element only.  rank_dropout needs a `[rank]`-
//! shaped mask broadcast over the rank axis (different geometry), and
//! module_dropout needs a host-side decision (whole-adapter early
//! return).  We keep this module thin and call `Dropout::forward` for
//! the per-element knob.

use crate::{Error, Result};
use cudarc::driver::CudaDevice;
use flame_core::regularization::Dropout as FlameDropout;
use flame_core::{rng, DType, Shape, Tensor};
use std::sync::Arc;

/// Per-adapter dropout configuration.  Every field defaulting to zero
/// (or `false` for `rank_dropout_scale`) reproduces the pre-T1.2
/// forward path bit-exact.
#[derive(Debug, Clone, Copy)]
pub struct DropoutCfg {
    /// Per-element dropout on the final adapter delta.  Range `[0, 1)`.
    pub dropout: f32,
    /// Per-rank Bernoulli on the down-projected intermediate.  Range
    /// `[0, 1)`.
    pub rank_dropout: f32,
    /// Per-step Bernoulli on the entire adapter (whole-delta skip).
    /// Range `[0, 1)`.
    pub module_dropout: f32,
    /// When `true`, divide the rank-mask by `mean(mask)` so the surviving
    /// ranks compensate for the dropped ones.  Matches upstream
    /// `rank_dropout_scale=True` (locon.py:297).
    pub rank_dropout_scale: bool,
}

impl Default for DropoutCfg {
    fn default() -> Self {
        Self {
            dropout: 0.0,
            rank_dropout: 0.0,
            module_dropout: 0.0,
            rank_dropout_scale: false,
        }
    }
}

impl DropoutCfg {
    /// True when **every** knob is the no-op default.  Algos use this
    /// to skip the dropout machinery entirely on the hot path.
    #[inline]
    pub fn is_off(&self) -> bool {
        self.dropout == 0.0 && self.rank_dropout == 0.0 && self.module_dropout == 0.0
    }
}

/// Decide whether to skip the whole adapter for this step.  Returns
/// `true` when `training` and a Bernoulli draw with `p = module_dropout`
/// fires.  Decision runs host-side via a 1-element `rand_on` →
/// `to_vec()`; cost is one tiny GPU launch + memcpy.
///
/// Callers should early-return a zero-delta tensor (or skip the add)
/// when this returns `true`.  Matches upstream locon.py:310.
pub fn should_skip_module(
    cfg: &DropoutCfg,
    training: bool,
    device: &Arc<CudaDevice>,
) -> Result<bool> {
    if !training || cfg.module_dropout == 0.0 {
        return Ok(false);
    }
    let r = rng::rand_on(device, &[1], DType::F32, 0)
        .map_err(Error::Flame)?
        .to_vec()
        .map_err(Error::Flame)?;
    Ok(r.first().copied().unwrap_or(1.0) < cfg.module_dropout)
}

/// Build a `[rank]`-shaped Bernoulli mask in `target_dtype`, ready to
/// broadcast against a `[..., rank]` intermediate.  Returns `None` when
/// not training or `rank_dropout == 0.0` so callers can take the fast
/// path.
///
/// With `rank_dropout_scale=true` the mask is scaled by `1/mean(mask)`
/// to preserve expected magnitude — matches upstream's
/// `drop /= drop.mean()` (locon.py:297).
pub fn rank_mask(
    cfg: &DropoutCfg,
    training: bool,
    rank: usize,
    target_dtype: DType,
    device: &Arc<CudaDevice>,
) -> Result<Option<Tensor>> {
    if !training || cfg.rank_dropout == 0.0 || rank == 0 {
        return Ok(None);
    }
    // Uniform [0, 1) per rank, then `>= p` keeps with prob `1 - p`.
    let u = rng::rand_on(device, &[rank], DType::F32, 0).map_err(Error::Flame)?;
    let thresh = u.full_like(cfg.rank_dropout).map_err(Error::Flame)?;
    let mask_f32 = u.ge(&thresh).map_err(Error::Flame)?;

    // Optionally rescale to preserve expectation.
    let mask_f32 = if cfg.rank_dropout_scale {
        let v = mask_f32.to_vec().map_err(Error::Flame)?;
        let n = v.len() as f32;
        let mean = v.iter().sum::<f32>() / n.max(1.0);
        if mean > 0.0 {
            mask_f32.mul_scalar(1.0 / mean).map_err(Error::Flame)?
        } else {
            // All ranks dropped this step — leave the mask at 0 so the
            // delta zeroes cleanly; no NaN from 1/0.
            mask_f32
        }
    } else {
        mask_f32
    };

    let mask = if target_dtype != DType::F32 {
        mask_f32.to_dtype(target_dtype).map_err(Error::Flame)?
    } else {
        mask_f32
    };
    Ok(Some(mask))
}

/// Per-element dropout on the final delta.  Delegates to
/// `flame_core::regularization::Dropout`, which handles the inverted
/// scaling (`1 / (1 - p)`) and the no-op when `training=false || p=0`.
pub fn apply_elem_dropout(
    cfg: &DropoutCfg,
    training: bool,
    delta: &Tensor,
) -> Result<Tensor> {
    if !training || cfg.dropout == 0.0 {
        return Ok(delta.clone());
    }
    let mut d = FlameDropout::new(cfg.dropout);
    d.train(training);
    d.forward(delta).map_err(Error::Flame)
}

/// Apply `mask` (shape `[rank]`) to `mid` (shape `[..., rank]`) by
/// broadcasting along the leading dims.  Reshapes `mask` to
/// `[1, ..., 1, rank]` matching `mid.rank()`, then multiplies.
pub fn apply_rank_mask(mid: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let dims = mid.dims().to_vec();
    if dims.is_empty() {
        return Err(Error::InvalidOperation(
            "apply_rank_mask: input has no dims".into(),
        ));
    }
    let rank = *dims.last().unwrap();
    let mask_dims = mask.dims();
    if mask_dims.len() != 1 || mask_dims[0] != rank {
        return Err(Error::InvalidOperation(format!(
            "apply_rank_mask: mask shape {:?} != [{}] (last dim of mid)",
            mask_dims, rank
        )));
    }
    let mut bshape = vec![1usize; dims.len()];
    *bshape.last_mut().unwrap() = rank;
    let mask_b = mask.reshape(&bshape).map_err(Error::Flame)?;
    mid.mul(&mask_b).map_err(Error::Flame)
}
