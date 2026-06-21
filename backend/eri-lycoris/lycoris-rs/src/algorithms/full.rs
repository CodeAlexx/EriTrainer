//! Full adapter — the trivial case. The on-disk tensors `diff` and (optional)
//! `diff_b` are direct weight/bias deltas.
//!
//! Used for fine-tuned full-rank weight deltas where no decomposition was
//! applied. Inference does:
//!     base.weight ← base.weight + strength * diff
//!     base.bias   ← base.bias   + strength * diff_b   (if diff_b present)
//!
//! Upstream save format: `lycoris/modules/full.py:128-132` (`custom_state_dict`).

use crate::{tensor_utils, Error, Result, StorageDtype};
use cudarc::driver::CudaDevice;
use flame_core::parameter::Parameter;
use flame_core::{Shape, Tensor};
use std::sync::Arc;

pub struct FullAdapter {
    /// Raw weight delta, wrapped in `Parameter` so the optimizer's in-place
    /// updates are visible through any accessor that reads it. The on-disk
    /// tensor matches the base weight shape.
    pub diff: Parameter,
    /// Optional bias delta (1D), shape matches the layer's bias if any.
    /// P0-7: previously `.diff_b` was silently dropped by the loader, so
    /// bias-using layers (linear projections, MLP outs, group norms) lost
    /// the bias delta entirely.
    pub diff_b: Option<Parameter>,
}

#[inline]
fn param_tensor(p: &Parameter) -> Result<Tensor> {
    p.tensor().map_err(Error::Flame)
}

impl FullAdapter {
    /// Construct a fresh trainable Full adapter with `requires_grad=true`
    /// leaves and storage dtype controlled by `dtype`.
    ///
    /// `weight_shape` is the base layer's weight shape (e.g. `[out, in]` for
    /// Linear, `[kh, kw, ic, oc]` for Flame conv2d). `bias_size` is `Some(n)`
    /// when the base layer has a bias of length `n`, else `None`.
    ///
    /// All leaves are zero-initialized so initial ΔW=0 and ΔB=0.
    pub fn new_for_training(
        weight_shape: Shape,
        bias_size: Option<usize>,
        device: Arc<CudaDevice>,
        dtype: StorageDtype,
    ) -> Result<Self> {
        let diff = tensor_utils::zeros_param(weight_shape, dtype, device.clone())?;
        let diff_b = match bias_size {
            None => None,
            Some(n) => Some(tensor_utils::zeros_param(
                Shape::from_dims(&[n]),
                dtype,
                device,
            )?),
        };
        Ok(Self {
            diff: Parameter::new(diff),
            diff_b: diff_b.map(Parameter::new),
        })
    }

    /// Returns `strength * diff`. Caller adds this to the base weight.
    pub fn delta_weight(&self, strength: f32) -> Result<Tensor> {
        let diff_t = param_tensor(&self.diff)?;
        if strength == 1.0 {
            // Avoid an unnecessary scalar mul kernel.
            return Ok(diff_t);
        }
        diff_t.mul_scalar(strength).map_err(Error::Flame)
    }

    /// Returns `Some(strength * diff_b)` when a bias delta is present.
    pub fn delta_bias(&self, strength: f32) -> Result<Option<Tensor>> {
        match &self.diff_b {
            None => Ok(None),
            Some(b) => {
                let bt = param_tensor(b)?;
                if strength == 1.0 {
                    Ok(Some(bt))
                } else {
                    Ok(Some(bt.mul_scalar(strength).map_err(Error::Flame)?))
                }
            }
        }
    }

    /// Trainable leaves: `[diff, diff_b?]`. Mirrors the `LycorisModule::parameters`
    /// contract for adapters that don't implement the full trait.
    /// `FullAdapter` is intentionally trait-free because it has no
    /// `forward(x)` (it's a pure weight delta), but the trainer needs the
    /// same accessor for optimizer collection.
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut out: Vec<Tensor> = Vec::with_capacity(2);
        out.push(param_tensor(&self.diff).expect("Full.diff mutex poisoned"));
        if let Some(ref b) = self.diff_b {
            out.push(param_tensor(b).expect("Full.diff_b mutex poisoned"));
        }
        out
    }

    /// Live `Parameter` handles for optimizer collection.
    pub fn parameters_handles(&self) -> Vec<Parameter> {
        let mut out: Vec<Parameter> = Vec::with_capacity(2);
        out.push(self.diff.clone());
        if let Some(ref b) = self.diff_b {
            out.push(b.clone());
        }
        out
    }
}
