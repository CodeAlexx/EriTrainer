/// Tensor utility functions for LyCORIS
///
/// Helper functions to work with Flame tensors. Inference path (loader,
/// `apply_to`) uses the BF16 helpers as-is. Training path uses the
/// `*_param` variants which set `requires_grad=true` on the returned leaf
/// so flame_core's autograd records gradients into them and a trainer can
/// wrap them in `flame_core::parameter::Parameter` for the optimizer.

use crate::{Error, Result};
use cudarc::driver::CudaDevice;
use flame_core::{DType, Shape, Tensor};
use std::sync::Arc;

/// Storage dtype policy for trainable adapter leaves.
///
/// - `Bf16` (legacy / inference-flame compatible): all leaves stored in BF16,
///   compute also BF16. Matches the load-and-merge path used by
///   `inference-flame/src/lycoris.rs`.
/// - `F32` (EDv2 default for training): leaves stored in F32 (what AdamW
///   state expects), compute upcasts/downcasts as the algorithm forward
///   requires. Matches `eridiffusion-core/src/lora.rs::LoRALinear`.
///
/// Per-tensor: storage stays in the configured dtype. Compute paths upcast/
/// downcast as needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageDtype {
    /// BF16 storage, BF16 compute. Default for inference and back-compat.
    Bf16,
    /// F32 storage, BF16 compute (cast on every forward). EDv2 training default.
    F32,
}

impl StorageDtype {
    /// Map storage policy to the concrete flame_core DType used for the leaf.
    pub fn to_dtype(self) -> DType {
        match self {
            StorageDtype::Bf16 => DType::BF16,
            StorageDtype::F32 => DType::F32,
        }
    }
}

impl Default for StorageDtype {
    /// Default = BF16 to preserve existing inference-flame call sites.
    fn default() -> Self {
        StorageDtype::Bf16
    }
}

/// LoRA-style initialization scheme for the "down" / `_a` factor.
///
/// PEFT / SimpleTuner lora_init_type parity. Applies to LoCon (the canonical
/// LoRA decomposition); other LyCORIS algos (LoHa / LoKr / Full / OFT / BOFT)
/// retain their algorithm-specific upstream init regardless of this setting.
///
/// | Variant   | Down init                       | Up init |
/// | --------- | ------------------------------- | ------- |
/// | `Default` | `N(0, 1)` (current Phase-2b)    | zeros   |
/// | `Gaussian`| `N(0, 1/rank)` (PEFT gaussian)  | zeros   |
/// | `Pissa`   | SVD-of-base-weight (UNSUPPORTED — flame-core lacks SVD) |
/// | `Olora`   | QR-of-base-weight (UNSUPPORTED — flame-core lacks QR)   |
/// | `Loftq`   | quant-aware SVD (UNSUPPORTED — flame-core lacks SVD)    |
///
/// `Pissa`/`Olora`/`Loftq` parse and round-trip cleanly so config files can
/// carry them, but adapter construction errors with a clear "needs flame-core
/// linalg" message — see `LoConModule::new_linear_for_training_with_init`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraInitType {
    Default,
    Gaussian,
    Pissa,
    Olora,
    Loftq,
}

impl Default for LoraInitType {
    fn default() -> Self {
        LoraInitType::Default
    }
}

impl LoraInitType {
    /// Parse from string (case-insensitive). Matches SimpleTuner's
    /// `--lora_init_type` choices: `default | gaussian | pissa | olora | loftq`.
    /// Empty / `"none"` map to `Default`.
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        let lower = s.trim().to_ascii_lowercase();
        Ok(match lower.as_str() {
            "" | "default" | "none" => LoraInitType::Default,
            "gaussian" => LoraInitType::Gaussian,
            "pissa" => LoraInitType::Pissa,
            "olora" => LoraInitType::Olora,
            "loftq" => LoraInitType::Loftq,
            other => {
                return Err(format!(
                    "unknown lora_init_type '{other}'; expected one of: \
                     default, gaussian, pissa, olora, loftq"
                ))
            }
        })
    }

    /// Stable string identifier (matches `parse` lowercase form).
    pub fn as_str(&self) -> &'static str {
        match self {
            LoraInitType::Default => "default",
            LoraInitType::Gaussian => "gaussian",
            LoraInitType::Pissa => "pissa",
            LoraInitType::Olora => "olora",
            LoraInitType::Loftq => "loftq",
        }
    }

    /// Returns `true` when adapter construction with this init can succeed
    /// without external linalg (SVD/QR). Used by ctors to short-circuit
    /// before allocating tensors.
    pub fn is_supported(&self) -> bool {
        matches!(self, LoraInitType::Default | LoraInitType::Gaussian)
    }
}

/// Create a random tensor with BF16 dtype (inference / non-training).
///
/// Returned tensor has `requires_grad=false`. Use `randn_bf16_param` for
/// trainable leaves.
pub fn randn_bf16(
    shape: Shape,
    mean: f32,
    std: f32,
    device: Arc<CudaDevice>,
) -> Result<Tensor> {
    // Create FP32 random tensor first
    let tensor_f32 = Tensor::randn(shape.clone(), mean, std, device.clone())
        .map_err(|e| Error::Flame(e))?;

    // Convert to BF16
    tensor_f32.to_dtype(DType::BF16).map_err(|e| Error::Flame(e))
}

/// Create zeros tensor with BF16 dtype (inference / non-training).
///
/// Returned tensor has `requires_grad=false`. Use `zeros_bf16_param` for
/// trainable leaves.
pub fn zeros_bf16(shape: Shape, device: Arc<CudaDevice>) -> Result<Tensor> {
    Tensor::zeros_dtype(shape, DType::BF16, device).map_err(|e| Error::Flame(e))
}

/// Trainable random leaf in `storage` dtype with `requires_grad=true`.
///
/// Returned tensor is a leaf the optimizer can collect via the
/// `LycorisModule::parameters()` accessor. Wrap in
/// `flame_core::parameter::Parameter::new(t.clone())` at the trainer layer
/// to gain in-place AdamW updates.
pub fn randn_param(
    shape: Shape,
    mean: f32,
    std: f32,
    storage: StorageDtype,
    device: Arc<CudaDevice>,
) -> Result<Tensor> {
    let tensor_f32 = Tensor::randn(shape, mean, std, device).map_err(Error::Flame)?;
    let leaf = match storage {
        StorageDtype::F32 => tensor_f32,
        StorageDtype::Bf16 => tensor_f32.to_dtype(DType::BF16).map_err(Error::Flame)?,
    };
    Ok(leaf.requires_grad_(true))
}

/// Trainable zeros leaf in `storage` dtype with `requires_grad=true`.
///
/// Used for the canonical "up"-side LoRA leaf (Kohya init: zero so the
/// initial adapter delta is zero).
pub fn zeros_param(
    shape: Shape,
    storage: StorageDtype,
    device: Arc<CudaDevice>,
) -> Result<Tensor> {
    let dtype = storage.to_dtype();
    let leaf = Tensor::zeros_dtype(shape, dtype, device).map_err(Error::Flame)?;
    Ok(leaf.requires_grad_(true))
}

/// Back-compat shim: BF16-storage variant of `randn_param`. Equivalent to
/// `randn_param(..., StorageDtype::Bf16, ...)`.
pub fn randn_bf16_param(
    shape: Shape,
    mean: f32,
    std: f32,
    device: Arc<CudaDevice>,
) -> Result<Tensor> {
    randn_param(shape, mean, std, StorageDtype::Bf16, device)
}

/// Back-compat shim: BF16-storage variant of `zeros_param`. Equivalent to
/// `zeros_param(..., StorageDtype::Bf16, ...)`.
pub fn zeros_bf16_param(shape: Shape, device: Arc<CudaDevice>) -> Result<Tensor> {
    zeros_param(shape, StorageDtype::Bf16, device)
}

/// Create BF16 tensor with Kaiming uniform initialization
///
/// Python: torch.nn.init.kaiming_uniform_(tensor, a=sqrt(5))
/// Formula: U(-bound, bound) where bound = gain * sqrt(3 / fan_in)
/// gain = sqrt(2 / (1 + a²))
///
/// # Arguments
/// * `shape` - Tensor shape
/// * `a` - Negative slope parameter (use sqrt(5) for LoKr)
/// * `device` - CUDA device
pub fn kaiming_uniform_bf16(
    shape: Shape,
    a: f32,
    device: Arc<CudaDevice>,
) -> Result<Tensor> {
    let std = kaiming_std_for(&shape, a);
    // Note: PyTorch uses uniform distribution U(-bound, bound)
    // We approximate with normal distribution N(0, std) for simplicity
    // Exact uniform would require custom kernel or Flame API extension
    randn_bf16(shape, 0.0, std, device)
}

/// Trainable kaiming-uniform leaf with `requires_grad=true`, in `storage` dtype.
///
/// Same kaiming-as-normal approximation as `kaiming_uniform_bf16` (see
/// the note there). Use for the "down"-side / `_a` LoRA leaves; pair with
/// `zeros_param` for the "up"-side / `_b` leaves so the initial adapter
/// delta is exactly zero.
pub fn kaiming_uniform_param(
    shape: Shape,
    a: f32,
    storage: StorageDtype,
    device: Arc<CudaDevice>,
) -> Result<Tensor> {
    let std = kaiming_std_for(&shape, a);
    randn_param(shape, 0.0, std, storage, device)
}

#[inline]
fn kaiming_std_for(shape: &Shape, a: f32) -> f32 {
    let dims = shape.dims();
    let fan_in = if dims.len() >= 2 { dims[1] } else { dims[0] };
    let gain = (2.0 / (1.0 + a * a)).sqrt();
    gain * (3.0 / fan_in as f32).sqrt()
}

/// Create tensor from vec with BF16 dtype
pub fn from_vec_bf16(data: Vec<f32>, shape: Shape, device: Arc<CudaDevice>) -> Result<Tensor> {
    let tensor_f32 = Tensor::from_vec(data, shape, device).map_err(|e| Error::Flame(e))?;
    tensor_f32.to_dtype(DType::BF16).map_err(|e| Error::Flame(e))
}

/// Transpose 2D tensor (handles Flame's transpose API)
pub fn transpose_2d(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.dims();
    if dims.len() != 2 {
        return Err(Error::InvalidOperation(format!(
            "transpose_2d requires 2D tensor, got {}D",
            dims.len()
        )));
    }

    // Flame's transpose() swaps last two dimensions for 2D
    tensor.transpose().map_err(|e| Error::Flame(e))
}

/// Kronecker product implementation
///
/// Computes the Kronecker product of two tensors
pub fn kronecker_product(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_dims = a.dims();
    let b_dims = b.dims();

    if a_dims.len() != 2 || b_dims.len() != 2 {
        return Err(Error::InvalidOperation(
            "Kronecker product requires 2D tensors".to_string()
        ));
    }

    let (m, n) = (a_dims[0], a_dims[1]);
    let (p, q) = (b_dims[0], b_dims[1]);

    // Get data from tensors
    let a_data = a.to_vec().map_err(|e| Error::Flame(e))?;
    let b_data = b.to_vec().map_err(|e| Error::Flame(e))?;

    // Compute Kronecker product on CPU
    let mut result = vec![0.0f32; m * p * n * q];

    for i in 0..m {
        for j in 0..n {
            let a_val = a_data[i * n + j];
            for k in 0..p {
                for l in 0..q {
                    let b_val = b_data[k * q + l];
                    result[(i * p + k) * (n * q) + (j * q + l)] = a_val * b_val;
                }
            }
        }
    }

    // Create output tensor
    let output_shape = Shape::from_dims(&[m * p, n * q]);
    let output = Tensor::from_vec(result, output_shape, a.device().clone())
        .map_err(|e| Error::Flame(e))?;

    // Convert to BF16 if input was BF16
    if a.dtype() == DType::BF16 {
        output.to_dtype(DType::BF16).map_err(|e| Error::Flame(e))
    } else {
        Ok(output)
    }
}

/// Helper to get CudaDevice from Tensor
pub fn get_cuda_device(tensor: &Tensor) -> Arc<CudaDevice> {
    tensor.device().clone()
}
