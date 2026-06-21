//! LyCORIS-RS: Rust port of LyCORIS library for LoRA algorithms
//!
//! Implements LoCon, LoHa, LoKr, and Full adapters on top of `flame-core`.
//!
//! # Features
//! - BF16 storage with FP32 compute for numerical stability
//! - NHWC ↔ NCHW layout adapters
//! - safetensors loader with auto-detected adapter type
//! - Weight-merge `apply_to` API for inference-time fusion

pub mod algorithms;
pub mod dora;
pub mod dropout;
pub mod ops;
pub mod error;
pub mod dtype;
pub mod layout;
pub mod tensor_utils;
pub mod loader;

pub use error::{Error, Result};
pub use dtype::DType;
pub use layout::{TensorLayout, LayoutConverter};
pub use tensor_utils::{LoraInitType, StorageDtype};

// Re-export core Flame types
pub use flame_core::{Tensor, Shape, Device};
pub use flame_core::parameter::Parameter;

// Re-export adapter structs
pub use algorithms::full::FullAdapter;
pub use algorithms::locon::LoConModule as LoconAdapter;
pub use algorithms::loha::LoHaModule as LohaAdapter;
pub use algorithms::lokr::LoKrModule as LokrAdapter;
pub use algorithms::oft::OFTModule as OftAdapter;
pub use algorithms::boft::BOFTModule as BoftAdapter;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use cudarc::driver::CudaDevice;

/// Module trait for all LyCORIS modules
pub trait LycorisModule {
    /// Forward pass through the module
    fn forward(&self, x: &Tensor) -> Result<Tensor>;

    /// Get the differential weight ΔW
    fn get_diff_weight(&self) -> Result<Tensor>;

    /// Merge LoRA weights into base weights
    fn merge_to(&mut self, multiplier: f32) -> Result<()>;

    /// Trainable leaf tensors owned by this module, in a stable, defined
    /// order. The trainer wraps each into `flame_core::parameter::Parameter`
    /// (or registers them with its own optimizer) for the AdamW step.
    ///
    /// **Order contract** — kept stable so checkpoint resume can pair
    /// optimizer state by index:
    /// - LoCon: `[down, up]` (+ `mid` if Tucker)
    /// - LoHa:  `[w1a, w1b, w2a, w2b]` (+ `t1, t2` if Tucker)
    /// - LoKr:  one of the following, in this order whenever present:
    ///   `[w1?, w1a?, w1b?, w2?, w2a?, w2b?, t2?]`
    /// - Full:  `[diff]` (+ `diff_b` if present)
    ///
    /// Returned tensors should have `requires_grad=true` for any leaf
    /// constructed via the `*_param` / `_for_training` helpers; legacy
    /// loader-only adapters (BF16 storage, `requires_grad=false`) still
    /// surface here so the trainer can detect the policy mismatch.
    ///
    /// Each returned `Tensor` is a clone of the inner storage held by a
    /// `Parameter`. The clone preserves `TensorId` and `requires_grad`, so
    /// autograd correctly attributes gradients back to the parameter. Use
    /// [`parameters_handles`](Self::parameters_handles) when you need the
    /// `Parameter` wrapper itself (e.g. for the AdamW step path).
    fn parameters(&self) -> Vec<Tensor>;

    /// Trainable leaf **handles** (shared `Arc<Mutex<Tensor>>` wrappers) in
    /// the same stable order as [`parameters`](Self::parameters). Cloning a
    /// `Parameter` is cheap (Arc bump). The optimizer mutates each handle
    /// in-place via `Parameter::set_data` / `with_data_mut`; subsequent
    /// `Parameter::tensor()` reads return the updated storage, so forward
    /// passes pick up the new weights without any synchronization step.
    fn parameters_handles(&self) -> Vec<Parameter>;
}

/// Top-level adapter variant — one entry per Kohya/LyCORIS prefix in a
/// safetensors checkpoint.
pub enum LycorisAdapter {
    LoCon(LoconAdapter),
    LoHa(LohaAdapter),
    LoKr(LokrAdapter),
    Full(FullAdapter),
    /// Diag-OFT (orthogonal fine-tuning, Cayley-Neumann). Note: OFT is a
    /// **multiplicative** adapter (`W' = R^T·W`), so `delta_weight` errors
    /// here — `apply_to` cannot merge OFT without base-weight access. Use
    /// `OFTModule::apply_to_input` for forward-time application instead.
    OFT(OftAdapter),
    /// BOFT (butterfly OFT): `m` consecutive block-diagonal rotations with
    /// permuted block layouts between stages. Same multiplicative-on-input
    /// semantics as OFT — `delta_weight` errors here, use
    /// `BOFTModule::apply_to_input`.
    BOFT(BoftAdapter),
}

impl LycorisAdapter {
    /// Returns the unscaled ΔW for this adapter (alpha/rank already applied
    /// inside the math). Caller multiplies by `strength` and adds to base.
    pub fn delta_weight(&self) -> Result<Tensor> {
        match self {
            LycorisAdapter::LoCon(m) => m.get_diff_weight(),
            LycorisAdapter::LoHa(m)  => m.get_diff_weight(),
            LycorisAdapter::LoKr(m)  => m.get_diff_weight(),
            LycorisAdapter::Full(m)  => m.delta_weight(1.0),
            LycorisAdapter::OFT(m)   => m.get_diff_weight(),
            LycorisAdapter::BOFT(m)  => m.get_diff_weight(),
        }
    }

    /// Trainable leaves for this adapter, in the order documented on
    /// `LycorisModule::parameters`. Used by trainers to register adapter
    /// tensors with the optimizer.
    ///
    /// Returned tensors are cheap clones of the underlying `Parameter`'s
    /// data (preserving `TensorId` and `requires_grad`); they are *not*
    /// the live in-place storage. For optimizer-driven mutation use
    /// [`parameters_handles`](Self::parameters_handles).
    pub fn parameters(&self) -> Vec<Tensor> {
        match self {
            LycorisAdapter::LoCon(m) => m.parameters(),
            LycorisAdapter::LoHa(m)  => m.parameters(),
            LycorisAdapter::LoKr(m)  => m.parameters(),
            LycorisAdapter::Full(m)  => m.parameters(),
            LycorisAdapter::OFT(m)   => m.parameters(),
            LycorisAdapter::BOFT(m)  => m.parameters(),
        }
    }

    /// Live `Parameter` handles for this adapter, same stable order as
    /// `parameters()`. The trainer hands these directly to the optimizer
    /// so AdamW updates reach the adapter's internal storage.
    pub fn parameters_handles(&self) -> Vec<Parameter> {
        match self {
            LycorisAdapter::LoCon(m) => m.parameters_handles(),
            LycorisAdapter::LoHa(m)  => m.parameters_handles(),
            LycorisAdapter::LoKr(m)  => m.parameters_handles(),
            LycorisAdapter::Full(m)  => m.parameters_handles(),
            LycorisAdapter::OFT(m)   => m.parameters_handles(),
            LycorisAdapter::BOFT(m)  => m.parameters_handles(),
        }
    }
}

/// A collection of LyCORIS adapters keyed by Kohya prefix
/// (e.g. `lora_unet_down_blocks_0_attentions_0_proj_in`).
pub struct LycorisCollection {
    pub adapters: HashMap<String, LycorisAdapter>,
}

impl LycorisCollection {
    /// Load a LyCORIS safetensors file and auto-detect each adapter's type.
    pub fn load(path: &Path, device: Arc<CudaDevice>) -> anyhow::Result<Self> {
        loader::load(path, device)
    }

    /// Weight-merge mode. For each adapter, compute ΔW, reshape to base
    /// weight shape, and add `strength * ΔW` to the base tensor in place
    /// (replaces the entry in `weights`).
    ///
    /// `name_mapper` translates a LyCORIS adapter prefix into the caller's
    /// weight-dict key. Returning `None` skips that adapter.
    pub fn apply_to(
        &self,
        weights: &mut HashMap<String, Tensor>,
        strength: f32,
        name_mapper: impl Fn(&str) -> Option<String>,
    ) -> anyhow::Result<()> {
        loader::apply_collection(self, weights, strength, name_mapper)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_basic() {
        // Basic sanity test
        assert_eq!(2 + 2, 4);
    }
}
