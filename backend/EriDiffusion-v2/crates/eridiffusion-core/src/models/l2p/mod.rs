//! L2P (T2I-L2P, Tencent Youtu) — training copy of the pure-Rust port.
//!
//! L2P = Z-Image-Turbo DiT body + 16×16 pixel-space patchify +
//! MicroDiffusionModel U-Net head, with the standard `FinalLayer +
//! unpatchify` removed. Output is direct pixels, not VAE latents.
//!
//! This is a SELF-CONTAINED copy of `inference_flame::models::l2p` (plus
//! the training subset of `inference_flame::lora` and the `BlockLoader`
//! from `inference_flame::offload` and the `l2p_sampling` helpers). The
//! duplication is the established convention — see
//! `eridiffusion_core::models::klein` vs the inference-flame klein. The
//! `train_l2p` binary depends only on this module + flame_core; it has zero
//! `inference_flame` references.
//!
//! Module map:
//! - [`dit`]            — `L2pDiT` model + config + forward.
//! - [`local_decoder`]  — `MicroDiffusionModel` U-Net pixel head.
//! - [`rope`]           — 3-axis RoPE table builder.
//! - [`weight_loader`]  — safetensors → internal-key translator.
//! - [`lora`]           — training-mode `LoraStack` / `Slot` / `TrainEntry`.
//! - [`offload`]        — `BlockLoader` (dormant in resident-mode training).
//! - [`sampling`]       — sigma schedule + noise init + Euler step.
//! - [`block_trap`]     — intra-block probe registry (diagnostics).

pub mod block_trap;
pub mod dit;
pub mod local_decoder;
pub mod lora;
pub mod offload;
pub mod rope;
pub mod sampling;
pub mod weight_loader;

pub use dit::{L2pDiT, L2pDiTConfig};
pub use local_decoder::MicroDiffusionModel;
pub use lora::{LoraStack, Slot, TrainEntry};
pub use rope::build_3d_rope;
pub use weight_loader::{load_l2p_safetensors, translate_l2p_keys};
