//! Local HiDream-O1 trainer model helpers.
//!
//! This module is copied into EriTrainer's trainer core from the known-good
//! inference implementation so training code does not depend on the
//! `inference-flame` crate. Only the model, LoRA, MRoPE, trap, and weight
//! loader pieces needed by `prepare_hidream_o1`, `train_hidream_o1`, and parity
//! probes are exposed here; sampling/pipeline code stays outside the default
//! trainer dependency graph.

pub mod bottleneck_patch_embed;
pub mod decoder;
pub mod final_layer;
pub mod lora;
pub mod model;
pub mod mrope;
pub mod timestep_embedder;
pub mod trap;
pub mod weight_loader;

pub use bottleneck_patch_embed::BottleneckPatchEmbed;
pub use decoder::{hidream_o1_two_pass_attention, HiDreamDecoderLayer};
pub use final_layer::FinalLayer;
pub use lora::{
    default_resident_target_keys, default_target_suffixes, shape_for_suffix, LoraAdapter,
    LoraRegistry,
};
pub use model::HiDreamO1Model;
pub use mrope::{
    apply_interleaved_mrope, apply_mrope, build_mrope_positions, interleaved_mrope_cos_sin,
    MRopePositions,
};
pub use timestep_embedder::TimestepEmbedder;
pub use weight_loader::HiDreamO1WeightLoader;

/// Configuration for HiDream-O1-Image.
#[derive(Clone, Debug)]
pub struct HiDreamO1Config {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rope_theta: f32,
    pub mrope_section: [usize; 3],
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub attention_bias: bool,
    pub patch_size: usize,
    pub patch_in_channels: usize,
    pub bottleneck_dim: usize,
    pub tms_token_id: u32,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub fix_point: usize,
    pub timestep_freq_dim: usize,
}

impl HiDreamO1Config {
    /// HiDream-O1 Dev/Full defaults aligned with the Qwen3-VL 8B backbone.
    pub fn dev_8b() -> Self {
        let hidden_size = 4096;
        Self {
            hidden_size,
            num_layers: 36,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 12288,
            rope_theta: 5_000_000.0,
            mrope_section: [24, 20, 20],
            vocab_size: 151_936,
            rms_norm_eps: 1e-6,
            attention_bias: false,
            patch_size: 32,
            patch_in_channels: 3,
            bottleneck_dim: hidden_size / 4,
            tms_token_id: 151_673,
            image_token_id: 151_655,
            video_token_id: 151_656,
            vision_start_token_id: 151_652,
            fix_point: 4096,
            timestep_freq_dim: 256,
        }
    }
}
