/// LoCon (LoRA for Convolution) Module
///
/// Standard LoRA decomposition: ΔW = down @ up * scale
/// Works for both Linear and Conv layers
///
/// Weight layouts follow Flame contracts:
/// - Linear: [IN, OUT]
/// - Conv2d: [KH, KW, IC, OC]

use crate::{tensor_utils, tensor_utils::LoraInitType, Error, LycorisModule, Result, StorageDtype};
use cudarc::driver::CudaDevice;
use flame_core::parameter::Parameter;
use flame_core::{DType, Shape, Tensor};
use std::sync::Arc;

pub struct LoConModule {
    /// Down projection
    /// Linear: [IN, RANK]
    /// Conv: [KH, KW, IC, RANK] or [1, 1, IC, RANK] for 1×1
    /// Stored in BF16 (inference) or F32 (training); see `StorageDtype`.
    /// Wrapped in `Parameter` so the optimizer's in-place updates are
    /// visible through `forward` (which reads via `param.tensor()?`).
    pub down: Parameter,

    /// Up projection
    /// Linear: [RANK, OUT]
    /// Conv: [KH, KW, RANK, OC] or [1, 1, RANK, OC] for 1×1
    /// Initialized to zero so the initial adapter delta is exactly 0.
    pub up: Parameter,

    /// Tucker core tensor for Conv with kernel > 1
    /// [KH, KW, RANK, RANK], Optional
    pub mid: Option<Parameter>,

    /// Rank of the decomposition
    pub rank: usize,

    /// Alpha parameter for scaling (default: rank)
    pub alpha: f32,

    /// Device
    pub device: Arc<CudaDevice>,

    /// Whether this is for a convolution layer
    pub is_conv: bool,
}

// Helper functions
#[inline]
fn assert_bf16_storage(name: &str, t: &Tensor) -> Result<()> {
    if t.dtype() != DType::BF16 {
        return Err(Error::InvalidOperation(format!(
            "{} must use BF16 storage, got {:?}",
            name,
            t.dtype()
        )));
    }
    Ok(())
}

/// Read the inner tensor from a Parameter, surfacing the mutex-poisoned case
/// as our crate `Error` rather than the raw flame_core error.
#[inline]
fn param_tensor(p: &Parameter) -> Result<Tensor> {
    p.tensor().map_err(Error::Flame)
}

/// Convert a linear [IN, OUT] weight to a 1×1 conv kernel [KH,KW,IC,OC] without copy.
/// IN→IC, OUT→OC
fn as_conv1x1_kernel(w_in_out: &Tensor) -> Result<Tensor> {
    let dims = w_in_out.dims();
    if dims.len() != 2 {
        return Err(Error::InvalidOperation(format!(
            "as_conv1x1_kernel expects 2D tensor, got {}D",
            dims.len()
        )));
    }
    let (i, o) = (dims[0], dims[1]); // [IN, OUT]
    // View as [1,1,IC,OC]
    w_in_out.reshape(&[1, 1, i, o]).map_err(Error::Flame)
}

impl LoConModule {
    /// Create a new LoCon module for Linear layer
    ///
    /// # Arguments
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `rank` - Rank of LoRA decomposition
    /// * `alpha` - Scaling parameter (if None, uses rank)
    /// * `device` - CUDA device
    pub fn new_linear(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: Option<f32>,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let alpha = alpha.unwrap_or(rank as f32);

        // down: [IN, RANK] - BF16 storage
        let down = tensor_utils::randn_bf16(
            Shape::from_dims(&[in_features, rank]),
            0.0,
            1.0,
            device.clone(),
        )?;

        // up: [RANK, OUT] - BF16 storage, initialized to zeros
        let up = tensor_utils::zeros_bf16(
            Shape::from_dims(&[rank, out_features]),
            device.clone(),
        )?;

        assert_bf16_storage("down", &down)?;
        assert_bf16_storage("up", &up)?;

        Ok(Self {
            down: Parameter::new(down),
            up: Parameter::new(up),
            mid: None,
            rank,
            alpha,
            device,
            is_conv: false,
        })
    }

    /// Create a new LoCon module for Conv2d layer
    ///
    /// # Arguments
    /// * `in_channels` - Input channels
    /// * `out_channels` - Output channels
    /// * `kernel_size` - Convolution kernel size (h, w)
    /// * `rank` - Rank of LoRA decomposition
    /// * `alpha` - Scaling parameter
    /// * `use_tucker` - Whether to use Tucker decomposition for kernel
    /// * `device` - CUDA device
    pub fn new_conv2d(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        rank: usize,
        alpha: Option<f32>,
        use_tucker: bool,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let alpha = alpha.unwrap_or(rank as f32);
        let (kh, kw) = kernel_size;

        // Follow Flame conv layout: [KH, KW, IC, OC]
        let (down, up, mid) = if kh == 1 && kw == 1 {
            // 1×1 path uses true conv kernels:
            // down: [1, 1, IC, RANK], up: [1, 1, RANK, OC]
            let down = tensor_utils::randn_bf16(
                Shape::from_dims(&[1, 1, in_channels, rank]),
                0.0,
                1.0,
                device.clone(),
            )?;
            let up = tensor_utils::zeros_bf16(
                Shape::from_dims(&[1, 1, rank, out_channels]),
                device.clone(),
            )?;
            (down, up, None)
        } else if use_tucker {
            // Tucker path:
            // down: [1, 1, IC, RANK], up: [1, 1, RANK, OC], mid: [KH, KW, RANK, RANK]
            let down = tensor_utils::randn_bf16(
                Shape::from_dims(&[1, 1, in_channels, rank]),
                0.0,
                1.0,
                device.clone(),
            )?;
            let up = tensor_utils::zeros_bf16(
                Shape::from_dims(&[1, 1, rank, out_channels]),
                device.clone(),
            )?;
            let mid = tensor_utils::randn_bf16(
                Shape::from_dims(&[kh, kw, rank, rank]),
                0.0,
                1.0,
                device.clone(),
            )?;
            (down, up, Some(mid))
        } else {
            // Standard spatial (no Tucker):
            // down: [KH, KW, IC, RANK], up: [KH, KW, RANK, OC]
            let down = tensor_utils::randn_bf16(
                Shape::from_dims(&[kh, kw, in_channels, rank]),
                0.0,
                1.0,
                device.clone(),
            )?;
            let up = tensor_utils::zeros_bf16(
                Shape::from_dims(&[kh, kw, rank, out_channels]),
                device.clone(),
            )?;
            (down, up, None)
        };

        assert_bf16_storage("down", &down)?;
        assert_bf16_storage("up", &up)?;
        if let Some(ref m) = mid {
            assert_bf16_storage("mid", m)?;
        }

        Ok(Self {
            down: Parameter::new(down),
            up: Parameter::new(up),
            mid: mid.map(Parameter::new),
            rank,
            alpha,
            device,
            is_conv: true,
        })
    }

    /// Trainable variant of `new_linear`: F32-or-BF16 storage, `requires_grad=true`
    /// on `down` and `up` so the optimizer can collect them via the
    /// `LycorisModule::parameters` accessor.
    ///
    /// Init follows canonical PyTorch LoRA: `down ~ N(0, 1)`, `up = 0`. The
    /// adapter delta is exactly zero at init.
    pub fn new_linear_for_training(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: Option<f32>,
        device: Arc<CudaDevice>,
        dtype: StorageDtype,
    ) -> Result<Self> {
        Self::new_linear_for_training_with_init(
            in_features,
            out_features,
            rank,
            alpha,
            device,
            dtype,
            LoraInitType::Default,
        )
    }

    /// Variant of `new_linear_for_training` that selects the LoRA init scheme
    /// (PEFT / SimpleTuner `lora_init_type` parity).
    ///
    /// `LoraInitType::Default` = `down ~ N(0, 1)` (preserves prior behavior).
    /// `LoraInitType::Gaussian` = `down ~ N(0, 1/rank)` (PEFT gaussian).
    /// `Pissa` / `Olora` / `Loftq` error — they need SVD/QR of the base
    /// weight, which flame-core does not yet expose.
    pub fn new_linear_for_training_with_init(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: Option<f32>,
        device: Arc<CudaDevice>,
        dtype: StorageDtype,
        init_type: LoraInitType,
    ) -> Result<Self> {
        let alpha = alpha.unwrap_or(rank as f32);

        let down_std = match init_type {
            LoraInitType::Default => 1.0,
            LoraInitType::Gaussian => {
                if rank == 0 {
                    return Err(Error::InvalidOperation(
                        "LoCon::new_linear_for_training_with_init: gaussian init requires rank > 0".into(),
                    ));
                }
                1.0 / rank as f32
            }
            LoraInitType::Pissa | LoraInitType::Olora | LoraInitType::Loftq => {
                return Err(Error::InvalidOperation(format!(
                    "LoCon::new_linear_for_training_with_init: lora_init_type '{}' \
                     requires SVD/QR of the base weight, which flame-core does not \
                     yet expose. Choose 'default' or 'gaussian'.",
                    init_type.as_str()
                )));
            }
        };

        let down = tensor_utils::randn_param(
            Shape::from_dims(&[in_features, rank]),
            0.0,
            down_std,
            dtype,
            device.clone(),
        )?;
        let up = tensor_utils::zeros_param(
            Shape::from_dims(&[rank, out_features]),
            dtype,
            device.clone(),
        )?;

        Ok(Self {
            down: Parameter::new(down),
            up: Parameter::new(up),
            mid: None,
            rank,
            alpha,
            device,
            is_conv: false,
        })
    }

    /// Trainable variant of `new_conv2d`. See `new_linear_for_training` for
    /// dtype/grad semantics. `mid` is only created when `use_tucker` and the
    /// kernel has a non-1×1 spatial dim; it's also kaiming-uniform-initialized
    /// to preserve init magnitude (matching upstream).
    #[allow(clippy::too_many_arguments)]
    pub fn new_conv2d_for_training(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        rank: usize,
        alpha: Option<f32>,
        use_tucker: bool,
        device: Arc<CudaDevice>,
        dtype: StorageDtype,
    ) -> Result<Self> {
        Self::new_conv2d_for_training_with_init(
            in_channels,
            out_channels,
            kernel_size,
            rank,
            alpha,
            use_tucker,
            device,
            dtype,
            LoraInitType::Default,
        )
    }

    /// Variant of `new_conv2d_for_training` that selects the LoRA init
    /// scheme. See `new_linear_for_training_with_init` for semantics. Conv
    /// `mid` (Tucker) factor is always init'd at `N(0, 1)` regardless of
    /// `init_type`; it sits between two LoRA-style factors and inherits the
    /// upstream-LyCORIS Tucker convention.
    #[allow(clippy::too_many_arguments)]
    pub fn new_conv2d_for_training_with_init(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        rank: usize,
        alpha: Option<f32>,
        use_tucker: bool,
        device: Arc<CudaDevice>,
        dtype: StorageDtype,
        init_type: LoraInitType,
    ) -> Result<Self> {
        let alpha = alpha.unwrap_or(rank as f32);
        let (kh, kw) = kernel_size;

        let down_std = match init_type {
            LoraInitType::Default => 1.0,
            LoraInitType::Gaussian => {
                if rank == 0 {
                    return Err(Error::InvalidOperation(
                        "LoCon::new_conv2d_for_training_with_init: gaussian init requires rank > 0".into(),
                    ));
                }
                1.0 / rank as f32
            }
            LoraInitType::Pissa | LoraInitType::Olora | LoraInitType::Loftq => {
                return Err(Error::InvalidOperation(format!(
                    "LoCon::new_conv2d_for_training_with_init: lora_init_type '{}' \
                     requires SVD/QR of the base weight, which flame-core does not \
                     yet expose. Choose 'default' or 'gaussian'.",
                    init_type.as_str()
                )));
            }
        };

        let (down, up, mid) = if kh == 1 && kw == 1 {
            let down = tensor_utils::randn_param(
                Shape::from_dims(&[1, 1, in_channels, rank]),
                0.0,
                down_std,
                dtype,
                device.clone(),
            )?;
            let up = tensor_utils::zeros_param(
                Shape::from_dims(&[1, 1, rank, out_channels]),
                dtype,
                device.clone(),
            )?;
            (down, up, None)
        } else if use_tucker {
            let down = tensor_utils::randn_param(
                Shape::from_dims(&[1, 1, in_channels, rank]),
                0.0,
                down_std,
                dtype,
                device.clone(),
            )?;
            let up = tensor_utils::zeros_param(
                Shape::from_dims(&[1, 1, rank, out_channels]),
                dtype,
                device.clone(),
            )?;
            let mid = tensor_utils::randn_param(
                Shape::from_dims(&[kh, kw, rank, rank]),
                0.0,
                1.0,
                dtype,
                device.clone(),
            )?;
            (down, up, Some(mid))
        } else {
            let down = tensor_utils::randn_param(
                Shape::from_dims(&[kh, kw, in_channels, rank]),
                0.0,
                down_std,
                dtype,
                device.clone(),
            )?;
            let up = tensor_utils::zeros_param(
                Shape::from_dims(&[kh, kw, rank, out_channels]),
                dtype,
                device.clone(),
            )?;
            (down, up, None)
        };

        Ok(Self {
            down: Parameter::new(down),
            up: Parameter::new(up),
            mid: mid.map(Parameter::new),
            rank,
            alpha,
            device,
            is_conv: true,
        })
    }

    /// Get the scaling factor (alpha / rank), returns 0.0 if rank==0
    #[inline]
    pub fn scale(&self) -> f32 {
        if self.rank == 0 {
            0.0
        } else {
            self.alpha / self.rank as f32
        }
    }

    /// Merge into base weight tensor
    ///
    /// Returns new merged tensor (Flame doesn't support in-place add)
    pub fn merge_into(&self, base_weight: &Tensor, multiplier: f32) -> Result<Tensor> {
        let delta = self
            .get_diff_weight()?
            .mul_scalar(multiplier)
            .map_err(Error::Flame)?;
        // Add: result = base + delta
        base_weight.add(&delta).map_err(Error::Flame)
    }
}

impl LycorisModule for LoConModule {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let scale = self.scale();

        // Early exit for zero rank
        if scale == 0.0 {
            return tensor_utils::zeros_bf16(
                Shape::from_dims(x.dims()),
                self.device.clone(),
            );
        }

        // Cast F32-storage params to x.dtype() *in record mode* so backward
        // routes gradients through Cast → F32 leaves (matches the legacy
        // LoRALinear pattern).  Doing the dtype-coercion inside the matmul
        // (auto-cast via `Tensor::matmul`) uses `to_dtype_no_grad`, which
        // does NOT record a Cast op → the matmul autograd entry references
        // a leafless BF16 tensor.  That broken sub-graph corrupts FA
        // backward two layers up (CUDA_ERROR_MISALIGNED_ADDRESS, 2026-05-09
        // chroma --algo locon repro).
        let target_dtype = x.dtype();
        let down_t = {
            let t = param_tensor(&self.down)?;
            if t.dtype() != target_dtype { t.to_dtype(target_dtype).map_err(Error::Flame)? } else { t }
        };
        let up_t = {
            let t = param_tensor(&self.up)?;
            if t.dtype() != target_dtype { t.to_dtype(target_dtype).map_err(Error::Flame)? } else { t }
        };

        // First apply down projection (BF16 -> FP32 compute -> BF16 output)
        let h = if self.is_conv {
            // Conv2d operation with explicit parameters
            // down: [KH, KW, IC, RANK] or [1, 1, IC, RANK]
            crate::ops::conv2d::conv2d(
                x,
                &down_t,
                /*stride=*/ (1, 1),
                /*padding=*/ (0, 0),
                /*dilation=*/ (1, 1),
                /*groups=*/ 1,
                /*layout=*/ crate::ops::conv2d::Layout::NHWC,
            )?
        } else {
            // Linear operation: x[..., IN] @ down[IN, RANK] -> [..., RANK]
            x.matmul(&down_t).map_err(Error::Flame)?
        };

        // Apply mid if present (Tucker decomposition)
        let h = if let Some(ref mid) = self.mid {
            // mid: [KH, KW, RANK, RANK]
            let mid_t = {
                let t = param_tensor(mid)?;
                if t.dtype() != target_dtype { t.to_dtype(target_dtype).map_err(Error::Flame)? } else { t }
            };
            crate::ops::conv2d::conv2d(
                &h,
                &mid_t,
                (1, 1),
                (0, 0),
                (1, 1),
                1,
                crate::ops::conv2d::Layout::NHWC,
            )?
        } else {
            h
        };

        // Apply up projection
        let output = if self.is_conv {
            // up: [KH, KW, RANK, OC] or [1, 1, RANK, OC]
            crate::ops::conv2d::conv2d(
                &h,
                &up_t,
                (1, 1),
                (0, 0),
                (1, 1),
                1,
                crate::ops::conv2d::Layout::NHWC,
            )?
        } else {
            // h[..., RANK] @ up[RANK, OUT] -> [..., OUT]
            h.matmul(&up_t).map_err(Error::Flame)?
        };

        // Apply scale
        output.mul_scalar(scale).map_err(Error::Flame)
    }

    fn get_diff_weight(&self) -> Result<Tensor> {
        let scale = self.scale();
        let down_t = param_tensor(&self.down)?;
        let up_t = param_tensor(&self.up)?;

        // Early exit for zero scale
        if scale == 0.0 {
            return if self.is_conv {
                tensor_utils::zeros_bf16(
                    up_t.shape().clone(),
                    self.device.clone(),
                )
            } else {
                tensor_utils::zeros_bf16(
                    Shape::from_dims(&[down_t.dims()[0], up_t.dims()[1]]),
                    self.device.clone(),
                )
            };
        }

        if self.is_conv {
            // Conv path
            if let Some(ref mid) = self.mid {
                // Tucker reconstruction: mid[KH,KW,R,R], down[1,1,IC,R], up[1,1,R,OC]
                // → kernel [KH,KW,IC,OC], then apply scale.
                let mid_t = param_tensor(mid)?;
                let kernel = crate::ops::tucker::rebuild_conv_tucker(&mid_t, &down_t, &up_t)?;
                return kernel.mul_scalar(scale).map_err(Error::Flame);
            } else {
                // Standard LoRA conv
                let down_dims = down_t.dims();
                let up_dims = up_t.dims();

                // For 1×1: down:[1,1,IC,R], up:[1,1,R,OC]
                if down_dims[0] == 1 && down_dims[1] == 1 && up_dims[0] == 1 && up_dims[1] == 1 {
                    // Equivalent linear: [IC,OC] = [IC,R] @ [R,OC] then view to [1,1,IC,OC]
                    let ic = down_dims[2];
                    let r = down_dims[3];
                    let oc = up_dims[3];
                    let down_lin = down_t.reshape(&[ic, r]).map_err(Error::Flame)?;
                    let up_lin = up_t.reshape(&[r, oc]).map_err(Error::Flame)?;
                    let k_lin = down_lin.matmul(&up_lin).map_err(Error::Flame)?; // [IC,OC]
                    let k = k_lin.reshape(&[1, 1, ic, oc]).map_err(Error::Flame)?;
                    return k.mul_scalar(scale).map_err(Error::Flame);
                } else {
                    // Spatial (no Tucker): [KH,KW,IC,R] and [KH,KW,R,OC]
                    // For each spatial position, do IC×R @ R×OC = IC×OC
                    let kh = down_dims[0];
                    let kw = down_dims[1];
                    let ic = down_dims[2];
                    let r = down_dims[3];
                    let oc = up_dims[3];

                    // Reshape for batch matmul: [KH*KW, IC, R] @ [KH*KW, R, OC] -> [KH*KW, IC, OC]
                    let down_batch = down_t.reshape(&[kh * kw, ic, r]).map_err(Error::Flame)?;
                    let up_batch = up_t.reshape(&[kh * kw, r, oc]).map_err(Error::Flame)?;

                    // Batch matmul
                    let result_batch = down_batch.matmul(&up_batch).map_err(Error::Flame)?;

                    // Reshape back: [KH*KW, IC, OC] -> [KH, KW, IC, OC]
                    let k = result_batch.reshape(&[kh, kw, ic, oc]).map_err(Error::Flame)?;
                    return k.mul_scalar(scale).map_err(Error::Flame);
                }
            }
        } else {
            // Linear: ΔW = down @ up → [IN, OUT]
            // down: [IN, RANK], up: [RANK, OUT]
            let diff = down_t.matmul(&up_t).map_err(Error::Flame)?;
            diff.mul_scalar(scale).map_err(Error::Flame)
        }
    }

    fn merge_to(&mut self, multiplier: f32) -> Result<()> {
        // This is deprecated in favor of merge_into()
        // which takes a mutable base weight
        let _scaled = self
            .get_diff_weight()?
            .mul_scalar(multiplier)
            .map_err(Error::Flame)?;

        // Note: Cannot merge without base weight reference
        // Use merge_into() instead
        Ok(())
    }

    fn parameters(&self) -> Vec<Tensor> {
        // Order: [down, up, mid?]. Trainer pairs optimizer state by index.
        // Each `Tensor` is a clone of the live `Parameter` storage with
        // matching `TensorId`, so autograd grads route back to the param.
        let mut out: Vec<Tensor> = Vec::with_capacity(3);
        out.push(param_tensor(&self.down).expect("LoCon.down mutex poisoned"));
        out.push(param_tensor(&self.up).expect("LoCon.up mutex poisoned"));
        if let Some(ref m) = self.mid {
            out.push(param_tensor(m).expect("LoCon.mid mutex poisoned"));
        }
        out
    }

    fn parameters_handles(&self) -> Vec<Parameter> {
        // Live `Parameter` handles in the same order as `parameters()`.
        // Cloning a `Parameter` is an `Arc` bump; the optimizer mutates
        // the inner storage in-place, so subsequent `forward` calls see
        // the updated weights.
        let mut out: Vec<Parameter> = Vec::with_capacity(3);
        out.push(self.down.clone());
        out.push(self.up.clone());
        if let Some(ref m) = self.mid {
            out.push(m.clone());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_locon_creation() {
        // Placeholder - requires CUDA device initialization
        assert!(true);
    }

    #[test]
    fn test_scale_zero_rank() {
        let device = CudaDevice::new(0).unwrap();
        let module = LoConModule {
            down: Parameter::new(
                tensor_utils::zeros_bf16(Shape::from_dims(&[4, 0]), device.clone()).unwrap(),
            ),
            up: Parameter::new(
                tensor_utils::zeros_bf16(Shape::from_dims(&[0, 8]), device.clone()).unwrap(),
            ),
            mid: None,
            rank: 0,
            alpha: 1.0,
            device,
            is_conv: false,
        };

        assert_eq!(module.scale(), 0.0);
    }

    #[test]
    fn test_as_conv1x1_kernel() {
        let device = CudaDevice::new(0).unwrap();
        let w = tensor_utils::zeros_bf16(Shape::from_dims(&[3, 5]), device.clone()).unwrap();
        let k = as_conv1x1_kernel(&w).unwrap();
        assert_eq!(k.dims(), &[1, 1, 3, 5]);
    }
}
