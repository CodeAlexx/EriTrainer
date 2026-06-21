/// LoHa (LoRA with Hadamard Product) Module
///
/// ΔW = (w1a @ w1b) ⊙ (w2a @ w2b) * scale
/// where ⊙ is element-wise (Hadamard) product
///
/// Weight layouts follow Flame contracts:
/// - Linear: [IN, OUT]
/// - Conv2d: [KH, KW, IC, OC]

use crate::{tensor_utils, Error, LycorisModule, Result, StorageDtype};
use cudarc::driver::CudaDevice;
use flame_core::parameter::Parameter;
use flame_core::{DType, Shape, Tensor};
use std::sync::Arc;

pub struct LoHaModule {
    /// First down projection (w1a)
    /// Linear: [IN, RANK], Conv: [KH, KW, IC, RANK] or [1, 1, IC, RANK]
    pub w1a: Parameter,

    /// First up projection (w1b)
    /// Linear: [RANK, OUT], Conv: [KH, KW, RANK, OC] or [1, 1, RANK, OC]
    pub w1b: Parameter,

    /// Second down projection (w2a)
    /// Linear: [IN, RANK], Conv: [KH, KW, IC, RANK] or [1, 1, IC, RANK]
    pub w2a: Parameter,

    /// Second up projection (w2b)
    /// Linear: [RANK, OUT], Conv: [KH, KW, RANK, OC] or [1, 1, RANK, OC]
    pub w2b: Parameter,

    /// Tucker core 1 (optional): [KH, KW, RANK, RANK]
    pub t1: Option<Parameter>,

    /// Tucker core 2 (optional): [KH, KW, RANK, RANK]
    pub t2: Option<Parameter>,

    /// Rank of the decomposition
    pub rank: usize,

    /// Alpha parameter for scaling
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

#[inline]
fn param_tensor(p: &Parameter) -> Result<Tensor> {
    p.tensor().map_err(Error::Flame)
}

impl LoHaModule {
    /// Create a new LoHa module for Linear layer
    ///
    /// # Arguments
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `rank` - Rank of decomposition
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

        // w1a: [IN, RANK], initialized with normal(0, 1)
        let w1a = tensor_utils::randn_bf16(
            Shape::from_dims(&[in_features, rank]),
            0.0,
            1.0,
            device.clone(),
        )?;

        // w1b: [RANK, OUT], initialized with zeros
        let w1b = tensor_utils::zeros_bf16(
            Shape::from_dims(&[rank, out_features]),
            device.clone(),
        )?;

        // w2a: [IN, RANK], initialized with normal(0, 1)
        let w2a = tensor_utils::randn_bf16(
            Shape::from_dims(&[in_features, rank]),
            0.0,
            1.0,
            device.clone(),
        )?;

        // w2b: [RANK, OUT], initialized with normal(0, 0.1)
        let w2b = tensor_utils::randn_bf16(
            Shape::from_dims(&[rank, out_features]),
            0.0,
            0.1,
            device.clone(),
        )?;

        assert_bf16_storage("w1a", &w1a)?;
        assert_bf16_storage("w1b", &w1b)?;
        assert_bf16_storage("w2a", &w2a)?;
        assert_bf16_storage("w2b", &w2b)?;

        Ok(Self {
            w1a: Parameter::new(w1a),
            w1b: Parameter::new(w1b),
            w2a: Parameter::new(w2a),
            w2b: Parameter::new(w2b),
            t1: None,
            t2: None,
            rank,
            alpha,
            device,
            is_conv: false,
        })
    }

    /// Create a new LoHa module for Conv2d layer with optional Tucker decomposition
    ///
    /// # Arguments
    /// * `in_channels` - Input channels
    /// * `out_channels` - Output channels
    /// * `kernel_size` - Convolution kernel size (h, w)
    /// * `rank` - Rank of decomposition
    /// * `alpha` - Scaling parameter
    /// * `use_tucker` - Whether to use Tucker decomposition
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
        let (w1a, w1b, w2a, w2b, t1, t2) = if kh == 1 && kw == 1 {
            // 1×1 convolution
            let w1a = tensor_utils::randn_bf16(
                Shape::from_dims(&[1, 1, in_channels, rank]),
                0.0,
                1.0,
                device.clone(),
            )?;
            let w1b = tensor_utils::zeros_bf16(
                Shape::from_dims(&[1, 1, rank, out_channels]),
                device.clone(),
            )?;
            let w2a = tensor_utils::randn_bf16(
                Shape::from_dims(&[1, 1, in_channels, rank]),
                0.0,
                1.0,
                device.clone(),
            )?;
            let w2b = tensor_utils::randn_bf16(
                Shape::from_dims(&[1, 1, rank, out_channels]),
                0.0,
                0.1,
                device.clone(),
            )?;
            (w1a, w1b, w2a, w2b, None, None)
        } else if use_tucker {
            // Tucker decomposition: spatial kernels in t1/t2
            let w1a = tensor_utils::randn_bf16(
                Shape::from_dims(&[1, 1, in_channels, rank]),
                0.0,
                1.0,
                device.clone(),
            )?;
            let w1b = tensor_utils::zeros_bf16(
                Shape::from_dims(&[1, 1, rank, out_channels]),
                device.clone(),
            )?;
            let w2a = tensor_utils::randn_bf16(
                Shape::from_dims(&[1, 1, in_channels, rank]),
                0.0,
                1.0,
                device.clone(),
            )?;
            let w2b = tensor_utils::randn_bf16(
                Shape::from_dims(&[1, 1, rank, out_channels]),
                0.0,
                0.1,
                device.clone(),
            )?;

            let t1 = tensor_utils::randn_bf16(
                Shape::from_dims(&[kh, kw, rank, rank]),
                0.0,
                0.1,
                device.clone(),
            )?;
            let t2 = tensor_utils::randn_bf16(
                Shape::from_dims(&[kh, kw, rank, rank]),
                0.0,
                0.1,
                device.clone(),
            )?;

            (w1a, w1b, w2a, w2b, Some(t1), Some(t2))
        } else {
            // Standard spatial convolution
            let w1a = tensor_utils::randn_bf16(
                Shape::from_dims(&[kh, kw, in_channels, rank]),
                0.0,
                1.0,
                device.clone(),
            )?;
            let w1b = tensor_utils::zeros_bf16(
                Shape::from_dims(&[kh, kw, rank, out_channels]),
                device.clone(),
            )?;
            let w2a = tensor_utils::randn_bf16(
                Shape::from_dims(&[kh, kw, in_channels, rank]),
                0.0,
                1.0,
                device.clone(),
            )?;
            let w2b = tensor_utils::randn_bf16(
                Shape::from_dims(&[kh, kw, rank, out_channels]),
                0.0,
                0.1,
                device.clone(),
            )?;
            (w1a, w1b, w2a, w2b, None, None)
        };

        assert_bf16_storage("w1a", &w1a)?;
        assert_bf16_storage("w1b", &w1b)?;
        assert_bf16_storage("w2a", &w2a)?;
        assert_bf16_storage("w2b", &w2b)?;
        if let Some(ref t) = t1 {
            assert_bf16_storage("t1", t)?;
        }
        if let Some(ref t) = t2 {
            assert_bf16_storage("t2", t)?;
        }

        Ok(Self {
            w1a: Parameter::new(w1a),
            w1b: Parameter::new(w1b),
            w2a: Parameter::new(w2a),
            w2b: Parameter::new(w2b),
            t1: t1.map(Parameter::new),
            t2: t2.map(Parameter::new),
            rank,
            alpha,
            device,
            is_conv: true,
        })
    }

    /// Trainable variant of `new_linear` for LoHa with `requires_grad=true`
    /// leaves and storage policy controlled by `dtype`.
    ///
    /// Init matches upstream LyCORIS Python (`lycoris/modules/loha.py`,
    /// `use_scalar=False` default): `hada_w1_a ~ N(0, 0.1)`,
    /// `hada_w1_b ~ N(0, 1)`, `hada_w2_a = 0`, `hada_w2_b ~ N(0, 1)`. Saved
    /// names map directly: our `w1a → hada_w1_a`, `w1b → hada_w1_b`,
    /// `w2a → hada_w2_a`, `w2b → hada_w2_b` (see
    /// `eridiffusion-core::lycoris::collect_adapter_tensors`).
    ///
    /// Identity at init: `(w1a @ w1b) ⊙ (0 @ w2b) = (·) ⊙ 0 = 0`, so the
    /// adapter delta is exactly zero. Gradient pattern at step 0 (before
    /// any optimizer update) is **mathematically constrained**: only
    /// `w2a` (the zero leaf) receives a non-zero gradient — the other
    /// three are exact zeros because each is multiplied by either the
    /// upstream zero factor or the saved zero leaf. After step 1 (when
    /// the optimizer drives `w2a` off zero), all four matrices receive
    /// non-zero gradients in subsequent steps. This is identical to
    /// upstream PyTorch LyCORIS behavior; see
    /// `tests/autograd_smoke.rs::loha_linear_autograd_records_w2a_grad`.
    pub fn new_linear_for_training(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: Option<f32>,
        device: Arc<CudaDevice>,
        dtype: StorageDtype,
    ) -> Result<Self> {
        let alpha = alpha.unwrap_or(rank as f32);

        // Upstream `hada_w1_a ~ N(0, 0.1)` (loha.py line 149).
        let w1a = tensor_utils::randn_param(
            Shape::from_dims(&[in_features, rank]),
            0.0,
            0.1,
            dtype,
            device.clone(),
        )?;
        // Upstream `hada_w1_b ~ N(0, 1)` (loha.py line 148).
        let w1b = tensor_utils::randn_param(
            Shape::from_dims(&[rank, out_features]),
            0.0,
            1.0,
            dtype,
            device.clone(),
        )?;
        // Upstream `hada_w2_a = 0` when use_scalar=False (loha.py line 154).
        let w2a = tensor_utils::zeros_param(
            Shape::from_dims(&[in_features, rank]),
            dtype,
            device.clone(),
        )?;
        // Upstream `hada_w2_b ~ N(0, 1)` (loha.py line 150).
        let w2b = tensor_utils::randn_param(
            Shape::from_dims(&[rank, out_features]),
            0.0,
            1.0,
            dtype,
            device.clone(),
        )?;

        Ok(Self {
            w1a: Parameter::new(w1a),
            w1b: Parameter::new(w1b),
            w2a: Parameter::new(w2a),
            w2b: Parameter::new(w2b),
            t1: None,
            t2: None,
            rank,
            alpha,
            device,
            is_conv: false,
        })
    }

    /// Trainable variant of `new_conv2d` for LoHa. See `new_linear_for_training`
    /// for grad / dtype semantics. Init matches upstream LyCORIS Python
    /// (`lycoris/modules/loha.py`): `w1a ~ N(0, 0.1)`, `w1b ~ N(0, 1)`,
    /// `w2a = 0`, `w2b ~ N(0, 1)`; `t1, t2 ~ N(0, 0.1)` when Tucker is enabled.
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
        let alpha = alpha.unwrap_or(rank as f32);
        let (kh, kw) = kernel_size;

        let (w1a, w1b, w2a, w2b, t1, t2) = if kh == 1 && kw == 1 {
            let w1a = tensor_utils::randn_param(
                Shape::from_dims(&[1, 1, in_channels, rank]),
                0.0, 0.1, dtype, device.clone(),
            )?;
            let w1b = tensor_utils::randn_param(
                Shape::from_dims(&[1, 1, rank, out_channels]),
                0.0, 1.0, dtype, device.clone(),
            )?;
            let w2a = tensor_utils::zeros_param(
                Shape::from_dims(&[1, 1, in_channels, rank]),
                dtype, device.clone(),
            )?;
            let w2b = tensor_utils::randn_param(
                Shape::from_dims(&[1, 1, rank, out_channels]),
                0.0, 1.0, dtype, device.clone(),
            )?;
            (w1a, w1b, w2a, w2b, None, None)
        } else if use_tucker {
            let w1a = tensor_utils::randn_param(
                Shape::from_dims(&[1, 1, in_channels, rank]),
                0.0, 0.1, dtype, device.clone(),
            )?;
            let w1b = tensor_utils::randn_param(
                Shape::from_dims(&[1, 1, rank, out_channels]),
                0.0, 1.0, dtype, device.clone(),
            )?;
            let w2a = tensor_utils::zeros_param(
                Shape::from_dims(&[1, 1, in_channels, rank]),
                dtype, device.clone(),
            )?;
            let w2b = tensor_utils::randn_param(
                Shape::from_dims(&[1, 1, rank, out_channels]),
                0.0, 1.0, dtype, device.clone(),
            )?;
            let t1 = tensor_utils::randn_param(
                Shape::from_dims(&[kh, kw, rank, rank]),
                0.0, 0.1, dtype, device.clone(),
            )?;
            let t2 = tensor_utils::randn_param(
                Shape::from_dims(&[kh, kw, rank, rank]),
                0.0, 0.1, dtype, device.clone(),
            )?;
            (w1a, w1b, w2a, w2b, Some(t1), Some(t2))
        } else {
            let w1a = tensor_utils::randn_param(
                Shape::from_dims(&[kh, kw, in_channels, rank]),
                0.0, 0.1, dtype, device.clone(),
            )?;
            let w1b = tensor_utils::randn_param(
                Shape::from_dims(&[kh, kw, rank, out_channels]),
                0.0, 1.0, dtype, device.clone(),
            )?;
            let w2a = tensor_utils::zeros_param(
                Shape::from_dims(&[kh, kw, in_channels, rank]),
                dtype, device.clone(),
            )?;
            let w2b = tensor_utils::randn_param(
                Shape::from_dims(&[kh, kw, rank, out_channels]),
                0.0, 1.0, dtype, device.clone(),
            )?;
            (w1a, w1b, w2a, w2b, None, None)
        };

        Ok(Self {
            w1a: Parameter::new(w1a),
            w1b: Parameter::new(w1b),
            w2a: Parameter::new(w2a),
            w2b: Parameter::new(w2b),
            t1: t1.map(Parameter::new),
            t2: t2.map(Parameter::new),
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

impl LycorisModule for LoHaModule {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let scale = self.scale();

        // Early exit for zero rank
        if scale == 0.0 {
            return tensor_utils::zeros_bf16(Shape::from_dims(x.dims()), self.device.clone());
        }

        // See `LoConModule::forward` for the dtype-coercion rationale —
        // F32-storage params must be cast to `x.dtype()` *with autograd
        // recording* (i.e. NOT inside `Tensor::matmul`'s no-grad auto-cast)
        // so backward routes Cast → F32 leaf.
        let target_dtype = x.dtype();
        let coerce = |p: &Parameter| -> Result<Tensor> {
            let t = param_tensor(p)?;
            if t.dtype() != target_dtype { t.to_dtype(target_dtype).map_err(Error::Flame) } else { Ok(t) }
        };
        let w1a = coerce(&self.w1a)?;
        let w1b = coerce(&self.w1b)?;
        let w2a = coerce(&self.w2a)?;
        let w2b = coerce(&self.w2b)?;

        // Compute w1 and w2 with proper operations
        if self.is_conv {
            // Conv path: use conv2d operations
            let h1 = if let Some(ref t1) = self.t1 {
                let t1_t = coerce(t1)?;
                // Tucker: w1a → t1 → w1b
                let temp = crate::ops::conv2d::conv2d(
                    x,
                    &w1a,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    crate::ops::conv2d::Layout::NHWC,
                )
                ?;
                let temp = crate::ops::conv2d::conv2d(
                    &temp,
                    &t1_t,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    crate::ops::conv2d::Layout::NHWC,
                )
                ?;
                crate::ops::conv2d::conv2d(
                    &temp,
                    &w1b,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    crate::ops::conv2d::Layout::NHWC,
                )
                ?
            } else {
                // Direct: w1a → w1b
                let temp = crate::ops::conv2d::conv2d(
                    x,
                    &w1a,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    crate::ops::conv2d::Layout::NHWC,
                )
                ?;
                crate::ops::conv2d::conv2d(
                    &temp,
                    &w1b,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    crate::ops::conv2d::Layout::NHWC,
                )
                ?
            };

            let h2 = if let Some(ref t2) = self.t2 {
                let t2_t = coerce(t2)?;
                // Tucker: w2a → t2 → w2b
                let temp = crate::ops::conv2d::conv2d(
                    x,
                    &w2a,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    crate::ops::conv2d::Layout::NHWC,
                )
                ?;
                let temp = crate::ops::conv2d::conv2d(
                    &temp,
                    &t2_t,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    crate::ops::conv2d::Layout::NHWC,
                )
                ?;
                crate::ops::conv2d::conv2d(
                    &temp,
                    &w2b,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    crate::ops::conv2d::Layout::NHWC,
                )
                ?
            } else {
                // Direct: w2a → w2b
                let temp = crate::ops::conv2d::conv2d(
                    x,
                    &w2a,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    crate::ops::conv2d::Layout::NHWC,
                )
                ?;
                crate::ops::conv2d::conv2d(
                    &temp,
                    &w2b,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    crate::ops::conv2d::Layout::NHWC,
                )
                ?
            };

            // Hadamard product and scale
            let result = h1.mul(&h2)?;
            result.mul_scalar(scale).map_err(Error::Flame)
        } else {
            // Linear LoHa can apply
            //   x @ ((w1a @ w1b) * (w2a @ w2b))
            // without materializing the dense [IN, OUT] diff weight:
            //   y_o = sum_{r,s} (sum_i x_i*w1a_i_r*w2a_i_s) * w1b_r_o*w2b_s_o
            // This keeps the graph in rank^2 space and avoids SDXL-scale
            // dense adapter-weight activations for every target module.
            let x_dims = x.dims().to_vec();
            let in_features = w1a.dims()[0];
            let out_features = w1b.dims()[1];
            let rank = self.rank;
            let last = *x_dims.last().ok_or_else(|| {
                Error::InvalidOperation("LoHa linear forward: input has rank 0".into())
            })?;
            if last != in_features {
                return Err(Error::InvalidOperation(format!(
                    "LoHa linear forward: expected last dim {}, got {}",
                    in_features, last
                )));
            }
            let leading: usize = x_dims[..x_dims.len() - 1].iter().product();
            let x_flat = x.reshape(&[leading, in_features]).map_err(Error::Flame)?;
            let w2b_t = w2b.transpose().map_err(Error::Flame)?;

            let mut out_acc: Option<Tensor> = None;
            for s in 0..rank {
                let w2a_col = w2a
                    .narrow(1, s, 1)
                    .map_err(Error::Flame)?
                    .reshape(&[1, in_features])
                    .map_err(Error::Flame)?;
                let x_weighted = x_flat.mul(&w2a_col).map_err(Error::Flame)?;
                let z = x_weighted.matmul(&w1a).map_err(Error::Flame)?;

                let w2b_row = w2b_t
                    .narrow(1, s, 1)
                    .map_err(Error::Flame)?
                    .transpose()
                    .map_err(Error::Flame)?;
                let up = w1b.mul(&w2b_row).map_err(Error::Flame)?;
                let y = z.matmul(&up).map_err(Error::Flame)?;
                out_acc = Some(match out_acc {
                    Some(acc) => acc.add(&y).map_err(Error::Flame)?,
                    None => y,
                });
            }

            let out_flat = out_acc.ok_or_else(|| {
                Error::InvalidOperation("LoHa linear forward: rank must be > 0".into())
            })?;
            let out_flat = if scale == 1.0 {
                out_flat
            } else {
                out_flat.mul_scalar(scale).map_err(Error::Flame)?
            };

            let mut out_dims = x_dims;
            *out_dims.last_mut().expect("out_dims nonempty") = out_features;
            out_flat.reshape(&out_dims).map_err(Error::Flame)
        }
    }

    fn get_diff_weight(&self) -> Result<Tensor> {
        let scale = self.scale();
        let w1a = param_tensor(&self.w1a)?;
        let w1b = param_tensor(&self.w1b)?;
        let w2a = param_tensor(&self.w2a)?;
        let w2b = param_tensor(&self.w2b)?;

        // Early exit for zero scale
        if scale == 0.0 {
            return if self.is_conv {
                tensor_utils::zeros_bf16(w1b.shape().clone(), self.device.clone())
            } else {
                tensor_utils::zeros_bf16(
                    Shape::from_dims(&[w1a.dims()[0], w1b.dims()[1]]),
                    self.device.clone(),
                )
            };
        }

        if self.is_conv {
            // Conv path
            if let (Some(ref t1), Some(ref t2)) = (&self.t1, &self.t2) {
                let t1_t = param_tensor(t1)?;
                let t2_t = param_tensor(t2)?;
                // Tucker path: need full reconstruction
                // For now, use simplified approach via hadamard op
                crate::ops::hadamard::make_hadamard_weight_tucker(
                    &t1_t, &w1a, &w1b, &t2_t, &w2a, &w2b, scale,
                )
            } else {
                // Standard conv: compute kernel via hadamard
                let dims = w1a.dims();
                if dims[0] == 1 && dims[1] == 1 {
                    // 1×1 case: can use linear math
                    let ic = dims[2];
                    let r = dims[3];
                    let oc = w1b.dims()[3];

                    let w1a_lin = w1a.reshape(&[ic, r])?;
                    let w1b_lin = w1b.reshape(&[r, oc])?;
                    let w2a_lin = w2a.reshape(&[ic, r])?;
                    let w2b_lin = w2b.reshape(&[r, oc])?;

                    let w1 = w1a_lin.matmul(&w1b_lin)?;
                    let w2 = w2a_lin.matmul(&w2b_lin)?;
                    let diff = w1.mul(&w2)?;
                    let k = diff.reshape(&[1, 1, ic, oc])?;
                    k.mul_scalar(scale).map_err(Error::Flame)
                } else {
                    // Spatial case: use hadamard op
                    crate::ops::hadamard::make_hadamard_weight(
                        &w1a, &w1b, &w2a, &w2b, scale,
                    )
                }
            }
        } else {
            // Linear: w1 = w1a @ w1b, w2 = w2a @ w2b, diff = w1 ⊙ w2
            let w1 = w1a.matmul(&w1b)?;
            let w2 = w2a.matmul(&w2b)?;
            let diff = w1.mul(&w2)?;
            diff.mul_scalar(scale).map_err(Error::Flame)
        }
    }

    fn merge_to(&mut self, multiplier: f32) -> Result<()> {
        // Deprecated in favor of merge_into()
        let _scaled = self
            .get_diff_weight()?
            .mul_scalar(multiplier)
            .map_err(Error::Flame)?;
        Ok(())
    }

    fn parameters(&self) -> Vec<Tensor> {
        // Order: [w1a, w1b, w2a, w2b, t1?, t2?]. See LycorisModule docs.
        let mut out: Vec<Tensor> = Vec::with_capacity(6);
        out.push(param_tensor(&self.w1a).expect("LoHa.w1a mutex poisoned"));
        out.push(param_tensor(&self.w1b).expect("LoHa.w1b mutex poisoned"));
        out.push(param_tensor(&self.w2a).expect("LoHa.w2a mutex poisoned"));
        out.push(param_tensor(&self.w2b).expect("LoHa.w2b mutex poisoned"));
        if let Some(ref t) = self.t1 {
            out.push(param_tensor(t).expect("LoHa.t1 mutex poisoned"));
        }
        if let Some(ref t) = self.t2 {
            out.push(param_tensor(t).expect("LoHa.t2 mutex poisoned"));
        }
        out
    }

    fn parameters_handles(&self) -> Vec<Parameter> {
        let mut out: Vec<Parameter> = Vec::with_capacity(6);
        out.push(self.w1a.clone());
        out.push(self.w1b.clone());
        out.push(self.w2a.clone());
        out.push(self.w2b.clone());
        if let Some(ref t) = self.t1 {
            out.push(t.clone());
        }
        if let Some(ref t) = self.t2 {
            out.push(t.clone());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loha_creation() {
        // Placeholder - requires CUDA device initialization
        assert!(true);
    }

    #[test]
    fn test_scale_zero_rank() {
        let device = CudaDevice::new(0).unwrap();
        let module = LoHaModule {
            w1a: Parameter::new(
                tensor_utils::zeros_bf16(Shape::from_dims(&[4, 0]), device.clone()).unwrap(),
            ),
            w1b: Parameter::new(
                tensor_utils::zeros_bf16(Shape::from_dims(&[0, 8]), device.clone()).unwrap(),
            ),
            w2a: Parameter::new(
                tensor_utils::zeros_bf16(Shape::from_dims(&[4, 0]), device.clone()).unwrap(),
            ),
            w2b: Parameter::new(
                tensor_utils::zeros_bf16(Shape::from_dims(&[0, 8]), device.clone()).unwrap(),
            ),
            t1: None,
            t2: None,
            rank: 0,
            alpha: 1.0,
            device,
            is_conv: false,
        };

        assert_eq!(module.scale(), 0.0);
    }
}
