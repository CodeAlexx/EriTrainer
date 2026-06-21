//! LTX-2 video DiT — T2V LoRA training port.
//!
//! Source of truth: `diffusers/models/transformers/transformer_ltx2.py`
//! (1350 LoC, audited 2026-05-05 — see `docs/LTX2_PORT_AUDIT_*.md`).
//!
//! ## Audio excision strategy (T2V-only)
//!
//! Per LTX2_PORT_AUDIT_REFERENCE.md §8.2 risk #1, naively dropping audio
//! sub-modules silently desyncs per-block modulation tensor shapes. The
//! safe excision is:
//!
//! 1. **Audio inputs** (`audio_hidden_states`, `audio_encoder_hidden_states`)
//!    are **never plumbed in**: T2V doesn't have them.
//! 2. **Audio output** is discarded (we only return the video output).
//! 3. **Audio sub-modules within each block** (audio_attn1, audio_attn2,
//!    audio_norm1/2/3, audio_ff, audio_to_video_attn, video_to_audio_attn)
//!    are **not invoked**. Reasoning: the only audio→video coupling is the
//!    `audio_to_video_attn` sub-module; with audio inputs at zero, that
//!    cross-attn produces zero output, which is gated by `a2v_gate` and
//!    added to video (zero contribution). So skipping it preserves block
//!    output bit-for-bit vs. the full block run.
//! 4. **Modulation tables**: the per-block `scale_shift_table[6]` (video
//!    self-attn + FF) and the global `time_embed` 6-mod-param output are
//!    consumed unchanged. The cross-attn modulation tables that gate the
//!    a2v / v2a paths are loaded into the model state but unused.
//! 5. **Audio time embedding** and `av_cross_attn_*` global modulation
//!    layers are NOT instantiated (no audio_hidden_states means no
//!    cross-attn that needs gating).
//!
//! ## Architecture summary
//!
//! - 48 layers, hidden=4096 (32 heads × 128 head_dim), inner_dim=4096.
//! - Patchify: spatial+temporal patch_size=1 → flatten `[B, 128, F, H, W]`
//!   to `[B, F*H*W, 128]`, then `proj_in: 128 → 4096`.
//! - 3D RoPE on (frame, h, w) axes, `rope_theta=10000`, dual flavors
//!   (interleaved / split). T2V-only port supports `interleaved` (LTX-2.0).
//! - QK-norm: RMSNorm across the full inner_dim before SDPA.
//! - AdaLN-Single: PixArt-style sin/cos timestep → SiLU → Linear → 6×dim
//!   modulation; applied to video self-attn (3) + video FF (3).
//! - Caption projection: PixArtAlphaTextProjection (Linear → GELU-tanh → Linear)
//!   from 3840 → 4096.
//! - Output: LayerNorm(no-affine) → AdaLN final shift/scale (2 mod params)
//!   → Linear(4096 → 128).
//!
//! ## Hardcoded structural constants
//! - `NUM_LAYERS = 48`, `INNER_DIM = 4096`, `HEAD_DIM = 128`, `HEADS = 32`,
//!   `IN_CHANNELS = 128`, `CAPTION_CHANNELS = 3840`, `NORM_EPS = 1e-6`.
//!
//! ## TODOs flagged for the Verify pass
//! - 3D RoPE: full pixel-coord pre-processing matching diffusers
//!   `prepare_video_coords` (scale by VAE factors, causal_offset clamp,
//!   /fps) is implemented; both `split` (LTX-2.3 default) and legacy
//!   `interleaved` are supported.
//! - AdaLN modulation cast to F32 (audit risk #3) — done at the
//!   `silu().linear()` step; subsequent multiplies stay BF16 by default.
//! - Block-level checkpointing not yet wired (clone the ERNIE pattern in
//!   a follow-up).

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::{parameter::Parameter, DType, Shape, Tensor};

use crate::config::TrainConfig;
use crate::lora::LoRALinear;
use crate::models::TrainableModel;
use crate::Result;

// ---------------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------------

pub const NUM_LAYERS: usize = 48;
pub const HEADS: usize = 32;
pub const HEAD_DIM: usize = 128;
pub const INNER_DIM: usize = HEADS * HEAD_DIM; // 4096
pub const AUDIO_HEADS: usize = 32;
pub const AUDIO_HEAD_DIM: usize = 64;
pub const AUDIO_INNER_DIM: usize = AUDIO_HEADS * AUDIO_HEAD_DIM; // 2048
pub const IN_CHANNELS: usize = 128;
pub const OUT_CHANNELS: usize = 128;
pub const AUDIO_IN_CHANNELS: usize = 128;
pub const AUDIO_OUT_CHANNELS: usize = 128;
pub const CAPTION_CHANNELS: usize = 3840;
pub const FFN_MULT: usize = 4;
pub const FFN_DIM: usize = INNER_DIM * FFN_MULT; // 16384
pub const AUDIO_FFN_DIM: usize = AUDIO_INNER_DIM * FFN_MULT; // 8192
pub const NORM_EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 10000.0;
pub const TIMESTEP_EMBED_DIM: usize = 256;
pub const VAE_SCALE_F: usize = 8;
pub const VAE_SCALE_HW: usize = 32;
pub const CAUSAL_OFFSET: usize = 1;
pub const AUDIO_SCALE_FACTOR: usize = 4;

/// Per-block LoRA slots. The first six slots preserve the original video-only
/// ordering. AV mode appends audio self/cross attention, modal cross attention,
/// and audio FF slots so old video LoRA checkpoints remain shape-compatible
/// when loaded with the video-only path.
pub const LORA_SLOTS_PER_BLOCK: usize = 24;

const LORA_SLOT_KEYS: [&str; LORA_SLOTS_PER_BLOCK] = [
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
    "audio_attn1.to_q",
    "audio_attn1.to_k",
    "audio_attn1.to_v",
    "audio_attn1.to_out.0",
    "audio_attn2.to_q",
    "audio_attn2.to_k",
    "audio_attn2.to_v",
    "audio_attn2.to_out.0",
    "audio_to_video_attn.to_q",
    "audio_to_video_attn.to_k",
    "audio_to_video_attn.to_v",
    "audio_to_video_attn.to_out.0",
    "video_to_audio_attn.to_q",
    "video_to_audio_attn.to_k",
    "video_to_audio_attn.to_v",
    "video_to_audio_attn.to_out.0",
    "audio_ff.net.0.proj",
    "audio_ff.net.2",
];

const LORA_SLOT_SHAPES: [(usize, usize); LORA_SLOTS_PER_BLOCK] = [
    (INNER_DIM, INNER_DIM),
    (INNER_DIM, INNER_DIM),
    (INNER_DIM, INNER_DIM),
    (INNER_DIM, INNER_DIM),
    (INNER_DIM, FFN_DIM),
    (FFN_DIM, INNER_DIM),
    (AUDIO_INNER_DIM, AUDIO_INNER_DIM),
    (AUDIO_INNER_DIM, AUDIO_INNER_DIM),
    (AUDIO_INNER_DIM, AUDIO_INNER_DIM),
    (AUDIO_INNER_DIM, AUDIO_INNER_DIM),
    (AUDIO_INNER_DIM, AUDIO_INNER_DIM),
    (AUDIO_INNER_DIM, AUDIO_INNER_DIM),
    (AUDIO_INNER_DIM, AUDIO_INNER_DIM),
    (AUDIO_INNER_DIM, AUDIO_INNER_DIM),
    (INNER_DIM, AUDIO_INNER_DIM),
    (AUDIO_INNER_DIM, AUDIO_INNER_DIM),
    (AUDIO_INNER_DIM, AUDIO_INNER_DIM),
    (AUDIO_INNER_DIM, INNER_DIM),
    (AUDIO_INNER_DIM, AUDIO_INNER_DIM),
    (INNER_DIM, AUDIO_INNER_DIM),
    (INNER_DIM, AUDIO_INNER_DIM),
    (AUDIO_INNER_DIM, AUDIO_INNER_DIM),
    (AUDIO_INNER_DIM, AUDIO_FFN_DIM),
    (AUDIO_FFN_DIM, AUDIO_INNER_DIM),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ltx2RopeType {
    Split,
    Interleaved,
}

fn strip_ltx2_weight_prefix(key: &str) -> &str {
    key.strip_prefix("model.diffusion_model.").unwrap_or(key)
}

fn should_load_ltx2_weight(key: &str, load_audio: bool) -> bool {
    let key = strip_ltx2_weight_prefix(key);
    if key.starts_with("video_embeddings_connector.")
        || key.starts_with("audio_embeddings_connector.")
        || key.starts_with("text_embedding_projection.")
    {
        return false;
    }
    if !load_audio
        && (key.starts_with("audio_")
            || key.starts_with("audio.")
            || key.starts_with("av_cross_attn_"))
    {
        return false;
    }
    if key.starts_with("proj_in.")
        || key.starts_with("patchify_proj.")
        || key.starts_with("proj_out.")
        || key.starts_with("time_embed.")
        || key.starts_with("adaln_single.")
        || key.starts_with("prompt_adaln_single.")
        || key.starts_with("caption_projection.")
        || key == "scale_shift_table"
    {
        return true;
    }
    if load_audio
        && (key.starts_with("audio_proj_in.")
            || key.starts_with("audio_patchify_proj.")
            || key.starts_with("audio_proj_out.")
            || key.starts_with("audio_time_embed.")
            || key.starts_with("audio_adaln_single.")
            || key.starts_with("audio_prompt_adaln_single.")
            || key.starts_with("audio_caption_projection.")
            || key.starts_with("av_cross_attn_")
            || key.starts_with("av_ca_")
            || key == "audio_scale_shift_table")
    {
        return true;
    }
    if !key.starts_with("transformer_blocks.") {
        return false;
    }
    let is_audio_block = key.contains(".audio")
        || key.contains("audio_")
        || key.contains("audio_to_video")
        || key.contains("video_to_audio")
        || key.contains("a2v_cross")
        || key.contains("v2a_cross");
    if is_audio_block && !load_audio
    {
        return false;
    }
    key.contains(".attn1.")
        || key.contains(".attn2.")
        || key.contains(".ff.")
        || key.contains(".audio_attn1.")
        || key.contains(".audio_attn2.")
        || key.contains(".audio_ff.")
        || key.contains(".audio_to_video_attn.")
        || key.contains(".video_to_audio_attn.")
        || key.contains(".audio_to_video_norm.")
        || key.contains(".video_to_audio_norm.")
        || key.ends_with(".scale_shift_table")
        || key.ends_with(".prompt_scale_shift_table")
        || key.ends_with(".audio_prompt_scale_shift_table")
        || key.ends_with(".video_a2v_cross_attn_scale_shift_table")
        || key.ends_with(".audio_a2v_cross_attn_scale_shift_table")
}

fn normalize_ltx2_weight_key(key: &str) -> Option<String> {
    let mut key = strip_ltx2_weight_prefix(key).to_string();
    if let Some(rest) = key.strip_prefix("patchify_proj.") {
        key = format!("proj_in.{rest}");
    } else if let Some(rest) = key.strip_prefix("audio_patchify_proj.") {
        key = format!("audio_proj_in.{rest}");
    } else if let Some(rest) = key.strip_prefix("adaln_single.") {
        key = format!("time_embed.{rest}");
    } else if let Some(rest) = key.strip_prefix("audio_adaln_single.") {
        key = format!("audio_time_embed.{rest}");
    } else if let Some(rest) = key.strip_prefix("av_ca_video_scale_shift_adaln_single.") {
        key = format!("av_cross_attn_video_scale_shift.{rest}");
    } else if let Some(rest) = key.strip_prefix("av_ca_audio_scale_shift_adaln_single.") {
        key = format!("av_cross_attn_audio_scale_shift.{rest}");
    } else if let Some(rest) = key.strip_prefix("av_ca_video_a2v_gate_adaln_single.") {
        key = format!("av_cross_attn_video_a2v_gate.{rest}");
    } else if let Some(rest) = key.strip_prefix("av_ca_audio_v2a_gate_adaln_single.") {
        key = format!("av_cross_attn_audio_v2a_gate.{rest}");
    } else if let Some(rest) = key.strip_prefix("av_ca_a2v_gate_adaln_single.") {
        key = format!("av_cross_attn_video_a2v_gate.{rest}");
    } else if let Some(rest) = key.strip_prefix("av_ca_v2a_gate_adaln_single.") {
        key = format!("av_cross_attn_audio_v2a_gate.{rest}");
    }
    key = key.replace(".q_norm.", ".norm_q.");
    key = key.replace(".k_norm.", ".norm_k.");
    Some(key)
}

fn ltx2_weight_alias(key: &str) -> Option<&'static str> {
    match key {
        "proj_in.weight" => Some("patchify_proj.weight"),
        "proj_in.bias" => Some("patchify_proj.bias"),
        "time_embed.emb.timestep_embedder.linear_1.weight" => {
            Some("adaln_single.emb.timestep_embedder.linear_1.weight")
        }
        "time_embed.emb.timestep_embedder.linear_1.bias" => {
            Some("adaln_single.emb.timestep_embedder.linear_1.bias")
        }
        "time_embed.emb.timestep_embedder.linear_2.weight" => {
            Some("adaln_single.emb.timestep_embedder.linear_2.weight")
        }
        "time_embed.emb.timestep_embedder.linear_2.bias" => {
            Some("adaln_single.emb.timestep_embedder.linear_2.bias")
        }
        "time_embed.linear.weight" => Some("adaln_single.linear.weight"),
        "time_embed.linear.bias" => Some("adaln_single.linear.bias"),
        "audio_proj_in.weight" => Some("audio_patchify_proj.weight"),
        "audio_proj_in.bias" => Some("audio_patchify_proj.bias"),
        "audio_time_embed.emb.timestep_embedder.linear_1.weight" => {
            Some("audio_adaln_single.emb.timestep_embedder.linear_1.weight")
        }
        "audio_time_embed.emb.timestep_embedder.linear_1.bias" => {
            Some("audio_adaln_single.emb.timestep_embedder.linear_1.bias")
        }
        "audio_time_embed.emb.timestep_embedder.linear_2.weight" => {
            Some("audio_adaln_single.emb.timestep_embedder.linear_2.weight")
        }
        "audio_time_embed.emb.timestep_embedder.linear_2.bias" => {
            Some("audio_adaln_single.emb.timestep_embedder.linear_2.bias")
        }
        "audio_time_embed.linear.weight" => Some("audio_adaln_single.linear.weight"),
        "audio_time_embed.linear.bias" => Some("audio_adaln_single.linear.bias"),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Ltx2Model {
    pub config: TrainConfig,
    pub device: Arc<CudaDevice>,
    pub weights: HashMap<String, Tensor>,
    pub lora_adapters: Vec<LoRALinear>,
    pub parameters: Vec<Parameter>,
    pub is_lora: bool,
    /// Number of latent video frames per training sample. Defaults to 1
    /// (image-as-frame bootstrap). Set explicitly for true video.
    pub num_frames: usize,
    rope_type: Ltx2RopeType,
    lora_slots_per_block: usize,
}

impl Ltx2Model {
    pub fn load(
        ckpt_paths: &[std::path::PathBuf],
        config: &TrainConfig,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        Self::load_with_audio(ckpt_paths, config, device, false)
    }

    pub fn load_with_audio(
        ckpt_paths: &[std::path::PathBuf],
        config: &TrainConfig,
        device: Arc<CudaDevice>,
        load_audio: bool,
    ) -> Result<Self> {
        let mut weights = HashMap::new();
        for p in ckpt_paths {
            let part = flame_core::serialization::load_file_filtered(p, &device, |key| {
                should_load_ltx2_weight(key, load_audio)
            })?;
            for (k, v) in part {
                if let Some(k) = normalize_ltx2_weight_key(&k) {
                    weights.insert(k, v.to_dtype(DType::BF16)?);
                }
            }
        }
        log::info!(
            "LTX-2: {} tensors loaded from {} shard(s)",
            weights.len(),
            ckpt_paths.len()
        );

        let is_lora = config.is_lora();
        let mut lora_adapters = Vec::new();
        let mut parameters = Vec::new();
        if is_lora {
            let rank = config.lora_rank as usize;
            let alpha = config.lora_alpha as f32;
            let active_lora_slots = if load_audio { LORA_SLOTS_PER_BLOCK } else { 6 };
            for block_idx in 0..NUM_LAYERS {
                for (slot_idx, &(in_dim, out_dim)) in LORA_SLOT_SHAPES
                    .iter()
                    .take(active_lora_slots)
                    .enumerate()
                {
                    let seed = 42u64 + (block_idx * active_lora_slots + slot_idx) as u64;
                    lora_adapters.push(LoRALinear::new(
                        in_dim,
                        out_dim,
                        rank,
                        alpha,
                        device.clone(),
                        seed,
                    )?);
                }
            }
            for l in &lora_adapters {
                parameters.extend(l.parameters());
            }
        } else {
            return Err(crate::EriDiffusionError::Model(
                "LTX-2 only supports LoRA training in this port".into(),
            ));
        }

        let rope_type = match std::env::var("FLAME_LTX2_ROPE")
            .unwrap_or_else(|_| "split".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "interleaved" => Ltx2RopeType::Interleaved,
            "split" | "" => Ltx2RopeType::Split,
            other => {
                log::warn!("unknown FLAME_LTX2_ROPE={other:?}; defaulting to split");
                Ltx2RopeType::Split
            }
        };
        log::info!("LTX-2 RoPE mode: {:?}", rope_type);

        Ok(Self {
            config: config.clone(),
            device,
            weights,
            lora_adapters,
            parameters,
            is_lora,
            num_frames: 1,
            rope_type,
            lora_slots_per_block: if load_audio { LORA_SLOTS_PER_BLOCK } else { 6 },
        })
    }

    fn lora_idx(&self, layer_idx: usize, slot: usize) -> usize {
        layer_idx * self.lora_slots_per_block + slot
    }

    fn w(&self, key: &str) -> Result<&Tensor> {
        if let Some(t) = self.weights.get(key) {
            return Ok(t);
        }
        if let Some(alias) = ltx2_weight_alias(key) {
            if let Some(t) = self.weights.get(alias) {
                return Ok(t);
            }
        }
        Err(crate::EriDiffusionError::Model(format!("LTX-2 missing weight: {key}")))
    }

    /// Try to load a weight; if missing, return a zero/identity tensor of
    /// the requested shape. Used for graceful pre-port runs against
    /// checkpoints that haven't been audited for key naming yet.
    fn w_or_zeros(&self, key: &str, shape: &[usize]) -> Result<Tensor> {
        if let Some(t) = self.weights.get(key) {
            Ok(t.clone())
        } else {
            log::warn!("LTX-2: weight '{key}' missing, using zeros");
            Tensor::zeros_dtype(Shape::from_dims(shape), DType::BF16, self.device.clone())
                .map_err(Into::into)
        }
    }

    fn linear(&self, x: &Tensor, w_key: &str, bias_key: Option<&str>) -> Result<Tensor> {
        let w = self.w(w_key)?;
        let mut out = x.matmul(&w.transpose()?)?;
        if let Some(bk) = bias_key {
            if let Some(b) = self.weights.get(bk) {
                out = out.add(b)?;
            }
        }
        Ok(out)
    }

    fn linear_lora(
        &self,
        x: &Tensor,
        w_key: &str,
        bias_key: Option<&str>,
        adapter_idx: usize,
    ) -> Result<Tensor> {
        let base = self.linear(x, w_key, bias_key)?;
        if self.is_lora {
            if let Some(adapter) = self.lora_adapters.get(adapter_idx) {
                let delta = adapter.forward_delta(x)?;
                return base.add(&delta).map_err(Into::into);
            }
        }
        Ok(base)
    }

    fn rms_norm_full(&self, x: &Tensor, scale_key: &str) -> Result<Tensor> {
        // RMSNorm over the inner_dim axis (last). Scale optional — LTX-2
        // sets `elementwise_affine=False` on most norms, so weight may
        // not exist; in that case pass None.
        let scale = self.weights.get(scale_key);
        flame_core::norm::rms_norm(x, &[INNER_DIM], scale, NORM_EPS).map_err(Into::into)
    }

    fn rms_norm_dim(&self, x: &Tensor, scale_key: &str, dim: usize) -> Result<Tensor> {
        let scale = self.weights.get(scale_key);
        flame_core::norm::rms_norm(x, &[dim], scale, NORM_EPS).map_err(Into::into)
    }

    /// QK-norm: `rms_norm_across_heads`. The diffusers RMSNorm for QK in
    /// LTX-2 normalizes across the full inner_dim (= heads * head_dim).
    /// Same shape contract as `rms_norm_full`.
    fn qk_norm(&self, x: &Tensor, scale_key: &str) -> Result<Tensor> {
        let scale = self.weights.get(scale_key);
        flame_core::norm::rms_norm(x, &[INNER_DIM], scale, NORM_EPS).map_err(Into::into)
    }

    fn qk_norm_dim(&self, x: &Tensor, scale_key: &str, dim: usize) -> Result<Tensor> {
        let scale = self.weights.get(scale_key);
        flame_core::norm::rms_norm(x, &[dim], scale, NORM_EPS).map_err(Into::into)
    }

    /// PixArt sinusoidal timestep embedding with `flip_sin_to_cos = True`.
    /// Diffusers `Timesteps(embedding_dim, flip_sin_to_cos=True, downscale_freq_shift=0)`.
    fn timestep_sin_emb(&self, t: &Tensor, dim: usize) -> Result<Tensor> {
        let t_v = t.to_vec()?;
        let half = dim / 2;
        let mut data = vec![0f32; t_v.len() * dim];
        for (bi, &tv) in t_v.iter().enumerate() {
            for j in 0..half {
                let f = (-(10000.0f32).ln() * (j as f32) / (half as f32)).exp();
                let arg = tv * f;
                // flip_sin_to_cos=True: cos first, then sin.
                data[bi * dim + j] = arg.cos();
                data[bi * dim + half + j] = arg.sin();
            }
        }
        Tensor::from_vec(
            data,
            Shape::from_dims(&[t_v.len(), dim]),
            self.device.clone(),
        )?
        .to_dtype(DType::BF16)
        .map_err(Into::into)
    }

    /// AdaLN-Single output: 6 × inner_dim modulation parameters from a
    /// (B,) timestep tensor. **Modulation cast to F32** at the linear
    /// step to mitigate audit risk #3 (BF16 modulation overflow); the
    /// downstream multiplies happen in BF16 after a final cast.
    fn ada_modulation(&self, timestep: &Tensor, num_mod_params: usize) -> Result<Tensor> {
        self.ada_modulation_for(timestep, "time_embed", INNER_DIM, num_mod_params)
    }

    fn ada_modulation_for(
        &self,
        timestep: &Tensor,
        prefix: &str,
        dim: usize,
        num_mod_params: usize,
    ) -> Result<Tensor> {
        let temb = self.timestep_sin_emb(timestep, TIMESTEP_EMBED_DIM)?;
        // PixArtAlphaCombinedTimestepSizeEmbeddings: linear → silu → linear.
        // In LTX-2 base config, size embedding is disabled
        // (`use_additional_conditions=False`), so just timestep MLP.
        let linear_1_w = format!("{prefix}.emb.timestep_embedder.linear_1.weight");
        let linear_1_b = format!("{prefix}.emb.timestep_embedder.linear_1.bias");
        let linear_2_w = format!("{prefix}.emb.timestep_embedder.linear_2.weight");
        let linear_2_b = format!("{prefix}.emb.timestep_embedder.linear_2.bias");
        let h1 = self.linear(
            &temb,
            &linear_1_w,
            Some(&linear_1_b),
        )?;
        let h1 = h1.silu()?;
        let h2 = self.linear(
            &h1,
            &linear_2_w,
            Some(&linear_2_b),
        )?;

        // Final SiLU then linear to (num_mod_params * inner_dim).
        let act = h2.silu()?;
        // Audit risk #3: cast act to F32 before the modulation linear.
        let act_f32 = act.to_dtype(DType::F32)?;
        // Per-LTX2AdaLayerNormSingle, weight name is `time_embed.linear`.
        let linear_w = format!("{prefix}.linear.weight");
        let linear_b = format!("{prefix}.linear.bias");
        let w = self.w(&linear_w)?.to_dtype(DType::F32)?;
        let out_dim = w.shape().dims()[0].max(num_mod_params * dim);
        let b = self
            .w_or_zeros(&linear_b, &[out_dim])?
            .to_dtype(DType::F32)?;
        let mod_out = act_f32.matmul(&w.transpose()?)?.add(&b)?;
        // Result shape: [B, num_mod_params * INNER_DIM]
        // Recast to BF16 for downstream block math.
        mod_out.to_dtype(DType::BF16).map_err(Into::into)
    }

    fn embedded_timestep_for(&self, timestep: &Tensor, prefix: &str) -> Result<Tensor> {
        let temb = self.timestep_sin_emb(timestep, TIMESTEP_EMBED_DIM)?;
        let linear_1_w = format!("{prefix}.emb.timestep_embedder.linear_1.weight");
        let linear_1_b = format!("{prefix}.emb.timestep_embedder.linear_1.bias");
        let linear_2_w = format!("{prefix}.emb.timestep_embedder.linear_2.weight");
        let linear_2_b = format!("{prefix}.emb.timestep_embedder.linear_2.bias");
        let h1 = self.linear(&temb, &linear_1_w, Some(&linear_1_b))?;
        let h1 = h1.silu()?;
        self.linear(&h1, &linear_2_w, Some(&linear_2_b))
    }

    fn zeros_mod(&self, b: usize, dim: usize) -> Result<Tensor> {
        Tensor::zeros_dtype(
            Shape::from_dims(&[b, 1, dim]),
            DType::BF16,
            self.device.clone(),
        )
        .map_err(Into::into)
    }

    fn ones_mod(&self, b: usize, dim: usize) -> Result<Tensor> {
        Tensor::ones_dtype(
            Shape::from_dims(&[b, 1, dim]),
            DType::BF16,
            self.device.clone(),
        )
        .map_err(Into::into)
    }

    /// 3D RoPE for video tokens. Mirrors diffusers `LTX2AudioVideoRotaryPosEmbed`.
    /// Returns interleaved cos/sin tensors of shape `[1, 1, n_tokens, INNER_DIM]`
    /// (broadcastable over (B, heads)).
    ///
    /// ## CPU-side implementation
    /// 3D RoPE math is integer-grid-driven and small; CPU computation is
    /// fine and avoids new GPU kernels. Output materialized to BF16.
    fn build_video_rope(
        &self,
        num_frames: usize,
        h_lat: usize,
        w_lat: usize,
        fps: f32,
    ) -> Result<(Tensor, Tensor)> {
        // 1. Build per-token (frame, h, w) coords in pixel space.
        //    pixel_coord = latent_coord * vae_scale; for the first frame,
        //    causal_offset shift + clamp at 0; then divide temporal by fps.
        let n_tokens = num_frames * h_lat * w_lat;
        let mut coords = vec![[0f32; 3]; n_tokens];
        let mut idx = 0;
        for f in 0..num_frames {
            for hi in 0..h_lat {
                for wi in 0..w_lat {
                    // Patch boundaries [start, end); use midpoint = start + 0.5
                    // (since patch_size=1, end = start + 1 → midpoint = start + 0.5).
                    let f_mid_lat = f as f32 + 0.5;
                    let h_mid_lat = hi as f32 + 0.5;
                    let w_mid_lat = wi as f32 + 0.5;

                    // Latent → pixel space.
                    let mut f_pix = f_mid_lat * VAE_SCALE_F as f32;
                    let h_pix = h_mid_lat * VAE_SCALE_HW as f32;
                    let w_pix = w_mid_lat * VAE_SCALE_HW as f32;

                    // Causal offset on temporal axis: shift by causal_offset - vae_scale_F,
                    // clamp at 0.
                    f_pix = (f_pix + CAUSAL_OFFSET as f32 - VAE_SCALE_F as f32).max(0.0);

                    // Temporal axis scaled by fps → seconds.
                    f_pix /= fps;

                    coords[idx] = [f_pix, h_pix, w_pix];
                    idx += 1;
                }
            }
        }

        // 2. Compute pow_indices. Each axis gets `inner_dim / (3 * 2)` freqs
        //    (3 axes, 2 components per axis = cos + sin → "num_rope_elems = 6").
        let num_pos_dims = 3usize;
        let num_rope_elems = num_pos_dims * 2;
        let freqs_per_axis = INNER_DIM / num_rope_elems; // 4096 / 6 = 682 (with remainder 4)
        let pad_per_freq = INNER_DIM - freqs_per_axis * num_rope_elems; // 4

        // pow_indices: theta ^ linspace(0, 1, freqs_per_axis), then * pi/2.
        let mut pow_indices = Vec::with_capacity(freqs_per_axis);
        for j in 0..freqs_per_axis {
            let lin = if freqs_per_axis > 1 {
                j as f32 / (freqs_per_axis - 1) as f32
            } else {
                0.0
            };
            pow_indices.push(ROPE_THETA.powf(lin) * std::f32::consts::FRAC_PI_2);
        }

        // 3. Per-token, per-axis: freq_per_axis values = (grid * 2 - 1) * pow_indices.
        //    grid = coord / max_position. For video: max_positions = (20, 2048, 2048).
        let max_positions = [20.0_f32, 2048.0, 2048.0];

        match self.rope_type {
            Ltx2RopeType::Interleaved => {
                // Output layout: cos and sin, each [n_tokens, INNER_DIM].
                // For interleaved: per-axis values are repeat_interleave(2),
                // then padded at the front.
                let mut cos_data = vec![0f32; n_tokens * INNER_DIM];
                let mut sin_data = vec![0f32; n_tokens * INNER_DIM];
                for (ti, c) in coords.iter().enumerate() {
                    let row = ti * INNER_DIM;
                    for p in 0..pad_per_freq {
                        cos_data[row + p] = 1.0;
                        sin_data[row + p] = 0.0;
                    }
                    let mut col = pad_per_freq;
                    for axis in 0..3 {
                        let grid = c[axis] / max_positions[axis];
                        let scaled = grid * 2.0 - 1.0;
                        for &pi in &pow_indices {
                            let arg = scaled * pi;
                            let cv = arg.cos();
                            let sv = arg.sin();
                            cos_data[row + col] = cv;
                            cos_data[row + col + 1] = cv;
                            sin_data[row + col] = sv;
                            sin_data[row + col + 1] = sv;
                            col += 2;
                        }
                    }
                }
                let cos = Tensor::from_vec(
                    cos_data,
                    Shape::from_dims(&[1, 1, n_tokens, INNER_DIM]),
                    self.device.clone(),
                )?
                .to_dtype(DType::BF16)?;
                let sin = Tensor::from_vec(
                    sin_data,
                    Shape::from_dims(&[1, 1, n_tokens, INNER_DIM]),
                    self.device.clone(),
                )?
                .to_dtype(DType::BF16)?;
                Ok((cos, sin))
            }
            Ltx2RopeType::Split => {
                // Split RoPE is the LTX-2.3 default. Frequencies are half
                // width, padded to INNER_DIM/2, then reshaped to
                // [B, HEADS, T, HEAD_DIM/2].
                let half_inner = INNER_DIM / 2;
                let split_pad = half_inner - freqs_per_axis * num_pos_dims;
                let half_head = HEAD_DIM / 2;
                let mut cos_flat = vec![0f32; n_tokens * half_inner];
                let mut sin_flat = vec![0f32; n_tokens * half_inner];
                for (ti, c) in coords.iter().enumerate() {
                    let row = ti * half_inner;
                    for p in 0..split_pad {
                        cos_flat[row + p] = 1.0;
                        sin_flat[row + p] = 0.0;
                    }
                    let mut col = split_pad;
                    for axis in 0..3 {
                        let grid = c[axis] / max_positions[axis];
                        let scaled = grid * 2.0 - 1.0;
                        for &pi in &pow_indices {
                            let arg = scaled * pi;
                            cos_flat[row + col] = arg.cos();
                            sin_flat[row + col] = arg.sin();
                            col += 1;
                        }
                    }
                }

                let mut cos_heads = vec![0f32; HEADS * n_tokens * half_head];
                let mut sin_heads = vec![0f32; HEADS * n_tokens * half_head];
                for ti in 0..n_tokens {
                    for d in 0..half_inner {
                        let h = d / half_head;
                        let hd = d % half_head;
                        let src = ti * half_inner + d;
                        let dst = (h * n_tokens + ti) * half_head + hd;
                        cos_heads[dst] = cos_flat[src];
                        sin_heads[dst] = sin_flat[src];
                    }
                }
                let cos = Tensor::from_vec(
                    cos_heads,
                    Shape::from_dims(&[1, HEADS, n_tokens, half_head]),
                    self.device.clone(),
                )?
                .to_dtype(DType::BF16)?;
                let sin = Tensor::from_vec(
                    sin_heads,
                    Shape::from_dims(&[1, HEADS, n_tokens, half_head]),
                    self.device.clone(),
                )?
                .to_dtype(DType::BF16)?;
                Ok((cos, sin))
            }
        }
    }

    fn build_temporal_rope(
        &self,
        positions_sec: &[f32],
        dim: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        let n_tokens = positions_sec.len();
        let freqs_per_axis = dim / 2;
        let half_head = head_dim / 2;
        let max_position = 20.0_f32;

        let mut pow_indices = Vec::with_capacity(freqs_per_axis);
        for j in 0..freqs_per_axis {
            let lin = if freqs_per_axis > 1 {
                j as f32 / (freqs_per_axis - 1) as f32
            } else {
                0.0
            };
            pow_indices.push(ROPE_THETA.powf(lin) * std::f32::consts::FRAC_PI_2);
        }

        match self.rope_type {
            Ltx2RopeType::Interleaved => {
                let mut cos_data = vec![0f32; n_tokens * dim];
                let mut sin_data = vec![0f32; n_tokens * dim];
                for (ti, &pos) in positions_sec.iter().enumerate() {
                    let row = ti * dim;
                    let scaled = (pos / max_position) * 2.0 - 1.0;
                    let mut col = 0;
                    for &pi in &pow_indices {
                        let arg = scaled * pi;
                        let cv = arg.cos();
                        let sv = arg.sin();
                        cos_data[row + col] = cv;
                        cos_data[row + col + 1] = cv;
                        sin_data[row + col] = sv;
                        sin_data[row + col + 1] = sv;
                        col += 2;
                    }
                }
                let cos = Tensor::from_vec(
                    cos_data,
                    Shape::from_dims(&[1, 1, n_tokens, dim]),
                    self.device.clone(),
                )?
                .to_dtype(DType::BF16)?;
                let sin = Tensor::from_vec(
                    sin_data,
                    Shape::from_dims(&[1, 1, n_tokens, dim]),
                    self.device.clone(),
                )?
                .to_dtype(DType::BF16)?;
                Ok((cos, sin))
            }
            Ltx2RopeType::Split => {
                let half_inner = dim / 2;
                let mut cos_flat = vec![0f32; n_tokens * half_inner];
                let mut sin_flat = vec![0f32; n_tokens * half_inner];
                for (ti, &pos) in positions_sec.iter().enumerate() {
                    let row = ti * half_inner;
                    let scaled = (pos / max_position) * 2.0 - 1.0;
                    for (col, &pi) in pow_indices.iter().enumerate() {
                        let arg = scaled * pi;
                        cos_flat[row + col] = arg.cos();
                        sin_flat[row + col] = arg.sin();
                    }
                }

                let mut cos_heads = vec![0f32; heads * n_tokens * half_head];
                let mut sin_heads = vec![0f32; heads * n_tokens * half_head];
                for ti in 0..n_tokens {
                    for d in 0..half_inner {
                        let h = d / half_head;
                        let hd = d % half_head;
                        let src = ti * half_inner + d;
                        let dst = (h * n_tokens + ti) * half_head + hd;
                        cos_heads[dst] = cos_flat[src];
                        sin_heads[dst] = sin_flat[src];
                    }
                }
                let cos = Tensor::from_vec(
                    cos_heads,
                    Shape::from_dims(&[1, heads, n_tokens, half_head]),
                    self.device.clone(),
                )?
                .to_dtype(DType::BF16)?;
                let sin = Tensor::from_vec(
                    sin_heads,
                    Shape::from_dims(&[1, heads, n_tokens, half_head]),
                    self.device.clone(),
                )?
                .to_dtype(DType::BF16)?;
                Ok((cos, sin))
            }
        }
    }

    fn build_audio_rope(&self, audio_frames: usize) -> Result<(Tensor, Tensor)> {
        const AUDIO_HOP_LENGTH: f32 = 160.0;
        const AUDIO_SAMPLE_RATE: f32 = 16000.0;
        let mel_to_sec_divisor = AUDIO_SAMPLE_RATE / AUDIO_HOP_LENGTH;
        let scale = AUDIO_SCALE_FACTOR as f32;
        let mut positions = Vec::with_capacity(audio_frames);
        for t in 0..audio_frames {
            let mel_start = t as f32 * scale;
            let mel_end = (t + 1) as f32 * scale;
            let start = (mel_start + CAUSAL_OFFSET as f32 - scale).max(0.0);
            let end = (mel_end + CAUSAL_OFFSET as f32 - scale).max(0.0);
            positions.push(((start / mel_to_sec_divisor) + (end / mel_to_sec_divisor)) * 0.5);
        }
        self.build_temporal_rope(&positions, AUDIO_INNER_DIM, AUDIO_HEADS, AUDIO_HEAD_DIM)
    }

    fn build_video_temporal_rope(
        &self,
        num_frames: usize,
        h_lat: usize,
        w_lat: usize,
        fps: f32,
    ) -> Result<(Tensor, Tensor)> {
        let n_tokens = num_frames * h_lat * w_lat;
        let mut positions = Vec::with_capacity(n_tokens);
        for f in 0..num_frames {
            let start_pix = (f * VAE_SCALE_F) as f32;
            let end_pix = ((f + 1) * VAE_SCALE_F) as f32;
            let start = (start_pix + CAUSAL_OFFSET as f32 - VAE_SCALE_F as f32).max(0.0);
            let end = (end_pix + CAUSAL_OFFSET as f32 - VAE_SCALE_F as f32).max(0.0);
            let mid = ((start / fps) + (end / fps)) * 0.5;
            for _ in 0..(h_lat * w_lat) {
                positions.push(mid);
            }
        }
        self.build_temporal_rope(&positions, AUDIO_INNER_DIM, AUDIO_HEADS, AUDIO_HEAD_DIM)
    }

    /// Apply LTX RoPE to Q/K shape `[B, T, INNER_DIM]`.
    ///
    /// **Note**: diffusers applies RoPE to the linear-output Q/K of shape
    /// `[B, T, inner_dim]` *before* the unflatten-to-heads step, then
    /// unflattens. The RoPE freq layout matches the flat inner_dim. That's
    /// what we do here too — caller passes Q/K shape `[B, T, INNER_DIM]`,
    /// our cos/sin is `[1, 1, T, INNER_DIM]` (broadcast over batch + a
    /// trivial unsqueezed head axis we skip by working pre-unflatten).
    fn apply_rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        self.apply_rope_dim(x, cos, sin, INNER_DIM, HEADS, HEAD_DIM)
    }

    fn apply_rope_dim(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        dim: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.len() != 3 || dims[2] != dim {
            return Err(crate::EriDiffusionError::Model(format!(
                "apply_rope expects [B, T, {dim}], got {dims:?}"
            )));
        }
        let (b, t, _) = (dims[0], dims[1], dims[2]);
        match self.rope_type {
            Ltx2RopeType::Interleaved => {
                let half = dim / 2;
                let xv = x.reshape(&[b, t, half, 2])?.contiguous()?;
                let x_real = xv.narrow(3, 0, 1)?.contiguous()?.reshape(&[b, t, half])?;
                let x_imag = xv.narrow(3, 1, 1)?.contiguous()?.reshape(&[b, t, half])?;
                let neg_imag = x_imag.mul_scalar(-1.0)?;
                let stacked = Tensor::cat(
                    &[
                        &neg_imag.reshape(&[b, t, half, 1])?,
                        &x_real.reshape(&[b, t, half, 1])?,
                    ],
                    3,
                )?
                .contiguous()?
                .reshape(&[b, t, dim])?;
                let cos_b = cos.reshape(&[1, t, dim])?.to_dtype(DType::BF16)?;
                let sin_b = sin.reshape(&[1, t, dim])?.to_dtype(DType::BF16)?;
                x.mul(&cos_b)?.add(&stacked.mul(&sin_b)?).map_err(Into::into)
            }
            Ltx2RopeType::Split => {
                let half_head = head_dim / 2;
                let xh = x
                    .reshape(&[b, t, heads, head_dim])?
                    .permute(&[0, 2, 1, 3])?
                    .contiguous()?;
                let split = xh.reshape(&[b, heads, t, 2, half_head])?;
                let first = split.narrow(3, 0, 1)?;
                let second = split.narrow(3, 1, 1)?;
                let cos_e = cos.unsqueeze(3)?.to_dtype(DType::BF16)?;
                let sin_e = sin.unsqueeze(3)?.to_dtype(DType::BF16)?;
                let first_out = first.mul(&cos_e)?.sub(&second.mul(&sin_e)?)?;
                let second_out = second.mul(&cos_e)?.add(&first.mul(&sin_e)?)?;
                Tensor::cat(&[&first_out, &second_out], 3)?
                    .contiguous()?
                    .reshape(&[b, heads, t, head_dim])?
                    .permute(&[0, 2, 1, 3])?
                    .contiguous()?
                    .reshape(&[b, t, dim])
                    .map_err(Into::into)
            }
        }
    }

    /// Reshape Q/K/V of `[B, T, INNER_DIM]` to `[B, HEADS, T, HEAD_DIM]` for SDPA.
    fn to_heads(&self, x: &Tensor, b: usize, t: usize) -> Result<Tensor> {
        self.to_heads_dim(x, b, t, HEADS, HEAD_DIM)
    }

    fn to_heads_dim(
        &self,
        x: &Tensor,
        b: usize,
        t: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        x.reshape(&[b, t, heads, head_dim])?
            .permute(&[0, 2, 1, 3])
            .map_err(Into::into)
    }

    /// Inverse of `to_heads` after SDPA: `[B, HEADS, T, HEAD_DIM]` → `[B, T, INNER_DIM]`.
    fn from_heads(&self, x: &Tensor, b: usize, t: usize) -> Result<Tensor> {
        self.from_heads_dim(x, b, t, HEADS, HEAD_DIM)
    }

    fn from_heads_dim(
        &self,
        x: &Tensor,
        b: usize,
        t: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        x.permute(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape(&[b, t, heads * head_dim])
            .map_err(Into::into)
    }

    fn prepare_video_text(&self, text_emb: &Tensor) -> Result<Tensor> {
        let text_dims = text_emb.shape().dims();
        if text_dims.len() != 3 {
            return Err(crate::EriDiffusionError::Model(format!(
                "LTX-2 text context expects [B,T,C], got {:?}",
                text_dims
            )));
        }
        match text_dims[2] {
            INNER_DIM => text_emb.clone_result().map_err(Into::into),
            CAPTION_CHANNELS => {
                let cap_h1 = self.linear(
                    text_emb,
                    "caption_projection.linear_1.weight",
                    Some("caption_projection.linear_1.bias"),
                )?;
                let cap_h1 = gelu_tanh(&cap_h1)?;
                self.linear(
                    &cap_h1,
                    "caption_projection.linear_2.weight",
                    Some("caption_projection.linear_2.bias"),
                )
            }
            other => Err(crate::EriDiffusionError::Model(format!(
                "LTX-2 video text context last dim must be {CAPTION_CHANNELS} raw or {INNER_DIM} projected, got {other}"
            ))),
        }
    }

    fn prepare_audio_text(&self, text_emb: &Tensor) -> Result<Tensor> {
        let text_dims = text_emb.shape().dims();
        if text_dims.len() != 3 {
            return Err(crate::EriDiffusionError::Model(format!(
                "LTX-2 audio text context expects [B,T,C], got {:?}",
                text_dims
            )));
        }
        match text_dims[2] {
            AUDIO_INNER_DIM => text_emb.clone_result().map_err(Into::into),
            CAPTION_CHANNELS => {
                let cap_h1 = self.linear(
                    text_emb,
                    "audio_caption_projection.linear_1.weight",
                    Some("audio_caption_projection.linear_1.bias"),
                )?;
                let cap_h1 = gelu_tanh(&cap_h1)?;
                self.linear(
                    &cap_h1,
                    "audio_caption_projection.linear_2.weight",
                    Some("audio_caption_projection.linear_2.bias"),
                )
            }
            other => Err(crate::EriDiffusionError::Model(format!(
                "LTX-2 audio text context last dim must be {CAPTION_CHANNELS} raw or {AUDIO_INNER_DIM} projected, got {other}"
            ))),
        }
    }

    fn text_prompt_modulation(
        &self,
        text: &Tensor,
        timestep: &Tensor,
        prefix: &str,
        table_key: &str,
        dim: usize,
    ) -> Result<Tensor> {
        let Some(table) = self.weights.get(table_key) else {
            return text.clone_result().map_err(Into::into);
        };
        let full = match self.ada_modulation_for(timestep, prefix, dim, 2) {
            Ok(t) => t,
            Err(_) => return text.clone_result().map_err(Into::into),
        };
        let dims = text.shape().dims();
        let b = dims[0];
        let seq = dims[1];
        let prompt = full.narrow(1, 0, 2 * dim)?.reshape(&[b, 1, 2, dim])?;
        let table = table.reshape(&[1, 1, 2, dim])?.to_dtype(DType::BF16)?;
        let combined = table.add(&prompt)?;
        let shift = combined.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let scale = combined.narrow(2, 1, 1)?.squeeze_dim(2)?;
        let shift = shift.expand(&[b, seq, dim])?;
        let scale = scale.expand(&[b, seq, dim])?;
        text.mul(&scale.add_scalar(1.0)?)?.add(&shift).map_err(Into::into)
    }

    fn block_ada_params_6(
        &self,
        table_key: &str,
        temb: &Tensor,
        b: usize,
        dim: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let table = self
            .w(table_key)?
            .narrow(0, 0, 6)?
            .reshape(&[1, 1, 6, dim])?
            .to_dtype(DType::BF16)?;
        let temb_6 = temb.narrow(1, 0, 6 * dim)?.reshape(&[b, 1, 6, dim])?;
        let ada = table.add(&temb_6)?;
        let shift_msa = ada.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let scale_msa = ada.narrow(2, 1, 1)?.squeeze_dim(2)?;
        let gate_msa = ada.narrow(2, 2, 1)?.squeeze_dim(2)?;
        let shift_mlp = ada.narrow(2, 3, 1)?.squeeze_dim(2)?;
        let scale_mlp = ada.narrow(2, 4, 1)?.squeeze_dim(2)?;
        let gate_mlp = ada.narrow(2, 5, 1)?.squeeze_dim(2)?;
        Ok((shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp))
    }

    fn block_ada_params_ca(
        &self,
        table_key: &str,
        temb: &Tensor,
        b: usize,
        dim: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let Some(table) = self.weights.get(table_key) else {
            return Ok((self.zeros_mod(b, dim)?, self.zeros_mod(b, dim)?, self.ones_mod(b, dim)?));
        };
        if table.shape().dims()[0] < 9 || temb.shape().dims()[1] < 9 * dim {
            return Ok((self.zeros_mod(b, dim)?, self.zeros_mod(b, dim)?, self.ones_mod(b, dim)?));
        }
        let table = table
            .narrow(0, 6, 3)?
            .reshape(&[1, 1, 3, dim])?
            .to_dtype(DType::BF16)?;
        let temb_ca = temb.narrow(1, 6 * dim, 3 * dim)?.reshape(&[b, 1, 3, dim])?;
        let ada = table.add(&temb_ca)?;
        let shift = ada.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let scale = ada.narrow(2, 1, 1)?.squeeze_dim(2)?;
        let gate = ada.narrow(2, 2, 1)?.squeeze_dim(2)?;
        Ok((shift, scale, gate))
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_forward(
        &self,
        query: &Tensor,
        context: &Tensor,
        prefix: &str,
        q_dim: usize,
        kv_dim: usize,
        out_dim: usize,
        heads: usize,
        head_dim: usize,
        q_rope: Option<(&Tensor, &Tensor)>,
        k_rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let b = query.shape().dims()[0];
        let tq = query.shape().dims()[1];
        let tk = context.shape().dims()[1];
        let q = self.linear(
            query,
            &format!("{prefix}.to_q.weight"),
            Some(&format!("{prefix}.to_q.bias")),
        )?;
        let k = self.linear(
            context,
            &format!("{prefix}.to_k.weight"),
            Some(&format!("{prefix}.to_k.bias")),
        )?;
        let v = self.linear(
            context,
            &format!("{prefix}.to_v.weight"),
            Some(&format!("{prefix}.to_v.bias")),
        )?;
        let q = self.qk_norm_dim(&q, &format!("{prefix}.norm_q.weight"), heads * head_dim)?;
        let k = self.qk_norm_dim(&k, &format!("{prefix}.norm_k.weight"), heads * head_dim)?;
        let q = if let Some((cos, sin)) = q_rope {
            self.apply_rope_dim(&q, cos, sin, heads * head_dim, heads, head_dim)?
        } else {
            q
        };
        let k = if let Some((cos, sin)) = k_rope {
            self.apply_rope_dim(&k, cos, sin, heads * head_dim, heads, head_dim)?
        } else {
            k
        };
        let qh = self.to_heads_dim(&q, b, tq, heads, head_dim)?;
        let kh = self.to_heads_dim(&k, b, tk, heads, head_dim)?;
        let vh = self.to_heads_dim(&v, b, tk, heads, head_dim)?;
        let attn_out = flame_core::attention::sdpa(&qh, &kh, &vh, None)?;
        let attn_out = self.from_heads_dim(&attn_out, b, tq, heads, head_dim)?;
        let _ = (q_dim, kv_dim, out_dim);
        self.linear(
            &attn_out,
            &format!("{prefix}.to_out.0.weight"),
            Some(&format!("{prefix}.to_out.0.bias")),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_forward_lora(
        &self,
        query: &Tensor,
        context: &Tensor,
        prefix: &str,
        heads: usize,
        head_dim: usize,
        q_rope: Option<(&Tensor, &Tensor)>,
        k_rope: Option<(&Tensor, &Tensor)>,
        q_slot: usize,
        k_slot: usize,
        v_slot: usize,
        out_slot: usize,
    ) -> Result<Tensor> {
        let b = query.shape().dims()[0];
        let tq = query.shape().dims()[1];
        let tk = context.shape().dims()[1];
        let q = self.linear_lora(
            query,
            &format!("{prefix}.to_q.weight"),
            Some(&format!("{prefix}.to_q.bias")),
            q_slot,
        )?;
        let k = self.linear_lora(
            context,
            &format!("{prefix}.to_k.weight"),
            Some(&format!("{prefix}.to_k.bias")),
            k_slot,
        )?;
        let v = self.linear_lora(
            context,
            &format!("{prefix}.to_v.weight"),
            Some(&format!("{prefix}.to_v.bias")),
            v_slot,
        )?;
        let q = self.qk_norm_dim(&q, &format!("{prefix}.norm_q.weight"), heads * head_dim)?;
        let k = self.qk_norm_dim(&k, &format!("{prefix}.norm_k.weight"), heads * head_dim)?;
        let q = if let Some((cos, sin)) = q_rope {
            self.apply_rope_dim(&q, cos, sin, heads * head_dim, heads, head_dim)?
        } else {
            q
        };
        let k = if let Some((cos, sin)) = k_rope {
            self.apply_rope_dim(&k, cos, sin, heads * head_dim, heads, head_dim)?
        } else {
            k
        };
        let qh = self.to_heads_dim(&q, b, tq, heads, head_dim)?;
        let kh = self.to_heads_dim(&k, b, tk, heads, head_dim)?;
        let vh = self.to_heads_dim(&v, b, tk, heads, head_dim)?;
        let attn_out = flame_core::attention::sdpa(&qh, &kh, &vh, None)?;
        let attn_out = self.from_heads_dim(&attn_out, b, tq, heads, head_dim)?;
        self.linear_lora(
            &attn_out,
            &format!("{prefix}.to_out.0.weight"),
            Some(&format!("{prefix}.to_out.0.bias")),
            out_slot,
        )
    }

    #[allow(clippy::type_complexity)]
    fn cross_modal_params(
        &self,
        pre: &str,
        v_ca_ss: &Tensor,
        a_ca_ss: &Tensor,
        v_ca_gate: &Tensor,
        a_ca_gate: &Tensor,
        b: usize,
    ) -> Result<(
        Tensor,
        Tensor,
        (Tensor, Tensor),
        (Tensor, Tensor),
        (Tensor, Tensor),
        (Tensor, Tensor),
    )> {
        let v_table_key = format!("{pre}.video_a2v_cross_attn_scale_shift_table");
        let a_table_key = format!("{pre}.audio_a2v_cross_attn_scale_shift_table");
        let Some(v_table) = self.weights.get(&v_table_key) else {
            return Ok((
                self.zeros_mod(b, INNER_DIM)?,
                self.zeros_mod(b, AUDIO_INNER_DIM)?,
                (self.zeros_mod(b, INNER_DIM)?, self.zeros_mod(b, INNER_DIM)?),
                (self.zeros_mod(b, INNER_DIM)?, self.zeros_mod(b, INNER_DIM)?),
                (self.zeros_mod(b, AUDIO_INNER_DIM)?, self.zeros_mod(b, AUDIO_INNER_DIM)?),
                (self.zeros_mod(b, AUDIO_INNER_DIM)?, self.zeros_mod(b, AUDIO_INNER_DIM)?),
            ));
        };
        let Some(a_table) = self.weights.get(&a_table_key) else {
            return Ok((
                self.zeros_mod(b, INNER_DIM)?,
                self.zeros_mod(b, AUDIO_INNER_DIM)?,
                (self.zeros_mod(b, INNER_DIM)?, self.zeros_mod(b, INNER_DIM)?),
                (self.zeros_mod(b, INNER_DIM)?, self.zeros_mod(b, INNER_DIM)?),
                (self.zeros_mod(b, AUDIO_INNER_DIM)?, self.zeros_mod(b, AUDIO_INNER_DIM)?),
                (self.zeros_mod(b, AUDIO_INNER_DIM)?, self.zeros_mod(b, AUDIO_INNER_DIM)?),
            ));
        };

        let v_ss = v_table
            .narrow(0, 0, 4)?
            .reshape(&[1, 1, 4, INNER_DIM])?
            .to_dtype(DType::BF16)?
            .add(&v_ca_ss.reshape(&[b, 1, 4, INNER_DIM])?)?;
        let video_a2v_scale = v_ss.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let video_a2v_shift = v_ss.narrow(2, 1, 1)?.squeeze_dim(2)?;
        let video_v2a_scale = v_ss.narrow(2, 2, 1)?.squeeze_dim(2)?;
        let video_v2a_shift = v_ss.narrow(2, 3, 1)?.squeeze_dim(2)?;
        let a2v_gate = v_table
            .narrow(0, 4, 1)?
            .reshape(&[1, 1, 1, INNER_DIM])?
            .to_dtype(DType::BF16)?
            .add(&v_ca_gate.reshape(&[b, 1, 1, INNER_DIM])?)?
            .squeeze_dim(2)?;

        let a_ss = a_table
            .narrow(0, 0, 4)?
            .reshape(&[1, 1, 4, AUDIO_INNER_DIM])?
            .to_dtype(DType::BF16)?
            .add(&a_ca_ss.reshape(&[b, 1, 4, AUDIO_INNER_DIM])?)?;
        let audio_a2v_scale = a_ss.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let audio_a2v_shift = a_ss.narrow(2, 1, 1)?.squeeze_dim(2)?;
        let audio_v2a_scale = a_ss.narrow(2, 2, 1)?.squeeze_dim(2)?;
        let audio_v2a_shift = a_ss.narrow(2, 3, 1)?.squeeze_dim(2)?;
        let v2a_gate = a_table
            .narrow(0, 4, 1)?
            .reshape(&[1, 1, 1, AUDIO_INNER_DIM])?
            .to_dtype(DType::BF16)?
            .add(&a_ca_gate.reshape(&[b, 1, 1, AUDIO_INNER_DIM])?)?
            .squeeze_dim(2)?;

        Ok((
            a2v_gate,
            v2a_gate,
            (video_a2v_scale, video_a2v_shift),
            (video_v2a_scale, video_v2a_shift),
            (audio_a2v_scale, audio_a2v_shift),
            (audio_v2a_scale, audio_v2a_shift),
        ))
    }

    fn optional_adaln_mod(
        &self,
        timestep: &Tensor,
        prefix: &str,
        dim: usize,
        num_mod_params: usize,
    ) -> Result<Tensor> {
        let linear_w = format!("{prefix}.linear.weight");
        if !self.weights.contains_key(&linear_w) {
            return Tensor::zeros_dtype(
                Shape::from_dims(&[timestep.shape().dims()[0], 1, num_mod_params * dim]),
                DType::BF16,
                self.device.clone(),
            )
            .map_err(Into::into);
        }
        self.ada_modulation_for(timestep, prefix, dim, num_mod_params)?
            .narrow(1, 0, num_mod_params * dim)?
            .reshape(&[timestep.shape().dims()[0], 1, num_mod_params * dim])
            .map_err(Into::into)
    }

    /// Forward.
    ///
    /// Inputs:
    /// - `latent`: `[B, IN_CHANNELS=128, F, H, W]` BF16.
    /// - `text_emb`: `[B, T_text, CAPTION_CHANNELS=3840]` BF16.
    /// - `timestep`: `[B]` F32 (will be passed through directly to `time_embed`,
    ///   pre-multiplied by `timestep_scale_multiplier=1000` by caller).
    /// - `fps`: video frame rate (used by RoPE coord builder; default 24.0).
    ///
    /// Output: `[B, OUT_CHANNELS=128, F, H, W]` BF16 (velocity prediction).
    pub fn forward(
        &mut self,
        latent: &Tensor,
        text_emb: &Tensor,
        timestep: &Tensor,
        fps: f32,
    ) -> Result<Tensor> {
        let dims = latent.shape().dims().to_vec();
        if dims.len() != 5 || dims[1] != IN_CHANNELS {
            return Err(crate::EriDiffusionError::Model(format!(
                "LTX-2 forward expects [B,128,F,H,W], got {:?}",
                dims
            )));
        }
        let (b, _c, f, h_lat, w_lat) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let n_tokens = f * h_lat * w_lat;

        // 1. Patchify [B, C, F, H, W] → [B, F*H*W, C].
        // permute to [B, F, H, W, C] then flatten (F*H*W).
        let x =
            latent
                .permute(&[0, 2, 3, 4, 1])?
                .contiguous()?
                .reshape(&[b, n_tokens, IN_CHANNELS])?;
        // proj_in: 128 → 4096
        let mut hidden = self.linear(&x, "proj_in.weight", Some("proj_in.bias"))?;

        // 2. Text context. LTX-2.0/V1 stores raw 3840-d Gemma features and
        // projects them here. LTX-2.3/V2 caches already-connector-processed
        // 4096-d video text embeddings, so applying caption_projection again
        // is both wrong and often impossible because those weights are absent.
        let encoder_hidden_states = self.prepare_video_text(text_emb)?;

        // 3. AdaLN-Single global modulation (6 mod params × inner_dim, plus
        //    embedded_timestep for output layer).
        let mod_full = self.ada_modulation(timestep, 6)?; // [B, 6+ * 4096]
        let mod_full = mod_full.narrow(1, 0, 6 * INNER_DIM)?;
        let mod_chunks = mod_full.chunk(6, 1)?;
        let shift_msa = mod_chunks[0].unsqueeze(1)?;
        let scale_msa = mod_chunks[1].unsqueeze(1)?;
        let gate_msa = mod_chunks[2].unsqueeze(1)?;
        let shift_mlp = mod_chunks[3].unsqueeze(1)?;
        let scale_mlp = mod_chunks[4].unsqueeze(1)?;
        let gate_mlp = mod_chunks[5].unsqueeze(1)?;

        // Embedded timestep for output layer (just the post-MLP F32 cast):
        // re-derive by running the MLP again without the final mod-linear.
        let temb_for_out = self.embedded_timestep_for(timestep, "time_embed")?;

        // 4. Build 3D RoPE for video self-attention.
        let (cos_b, sin_b) = self.build_video_rope(f, h_lat, w_lat, fps)?;

        // 5. Run 48 transformer blocks (T2V slice — see audio excision strategy
        //    in module docs).
        for i in 0..NUM_LAYERS {
            hidden = self.block_forward_t2v(
                hidden,
                &encoder_hidden_states,
                &shift_msa,
                &scale_msa,
                &gate_msa,
                &shift_mlp,
                &scale_mlp,
                &gate_mlp,
                &cos_b,
                &sin_b,
                i,
                b,
                n_tokens,
            )?;
        }

        // 6. Output layer: LayerNorm(no-affine) → AdaLN final shift/scale
        //    (2 mod params from `scale_shift_table[2, dim] + temb_for_out`).
        let x_n = flame_core::layer_norm::layer_norm(&hidden, &[INNER_DIM], None, None, NORM_EPS)?;

        // scale_shift_table is a parameter [2, 4096]; add the embedded_timestep.
        // diffusers: scale_shift_values = self.scale_shift_table[None, None] +
        //            embedded_timestep[:, :, None]
        // → shape [B, T_text(or 1), 2, dim] → unbind dim=2 → shift, scale.
        // For our T2V slice we treat embedded_timestep as [B, dim] (no per-text-token
        // expansion since we're not doing audio dual-stream). Reshape to [B, 1, 1, dim]
        // for broadcast.
        let sst = self
            .w("scale_shift_table")?
            .reshape(&[1, 1, 2, INNER_DIM])?
            .to_dtype(DType::BF16)?;
        let temb_b = temb_for_out.reshape(&[b, 1, 1, INNER_DIM])?;
        let scale_shift = sst.add(&temb_b)?;
        let shift = scale_shift.narrow(2, 0, 1)?.squeeze_dim(2)?; // [B, 1, INNER_DIM]
        let scale = scale_shift.narrow(2, 1, 1)?.squeeze_dim(2)?;
        let x_out = x_n.mul(&scale.add_scalar(1.0)?)?.add(&shift)?;

        // proj_out: 4096 → 128
        let projected = self.linear(&x_out, "proj_out.weight", Some("proj_out.bias"))?;
        // [B, n_tokens, 128] → [B, F, H, W, C] → [B, C, F, H, W]
        let unpacked = projected
            .reshape(&[b, f, h_lat, w_lat, OUT_CHANNELS])?
            .permute(&[0, 4, 1, 2, 3])?
            .contiguous()?;
        Ok(unpacked)
    }

    /// Audio-video forward. Video latent is `[B,128,F,H,W]`; audio latent is
    /// `[B,C,T,M]` with `C*M == 128` (Musubi/LTX audio VAE commonly uses
    /// `[B,8,T,16]`). Returns `(video_velocity, audio_velocity)`.
    pub fn forward_audio_video(
        &mut self,
        latent: &Tensor,
        audio_latent: &Tensor,
        text_emb: &Tensor,
        audio_text_emb: &Tensor,
        timestep: &Tensor,
        fps: f32,
    ) -> Result<(Tensor, Tensor)> {
        if self.lora_slots_per_block < LORA_SLOTS_PER_BLOCK {
            return Err(crate::EriDiffusionError::Model(
                "LTX-2 AV forward requires model loaded with audio weights".into(),
            ));
        }
        let dims = latent.shape().dims().to_vec();
        if dims.len() != 5 || dims[1] != IN_CHANNELS {
            return Err(crate::EriDiffusionError::Model(format!(
                "LTX-2 AV video forward expects [B,128,F,H,W], got {:?}",
                dims
            )));
        }
        let audio_dims = audio_latent.shape().dims().to_vec();
        if audio_dims.len() != 4 || audio_dims[0] != dims[0] || audio_dims[1] * audio_dims[3] != AUDIO_IN_CHANNELS {
            return Err(crate::EriDiffusionError::Model(format!(
                "LTX-2 AV audio forward expects [B,C,T,M] with C*M=128 and matching B, got {:?}",
                audio_dims
            )));
        }
        let (b, _c, f, h_lat, w_lat) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let n_video = f * h_lat * w_lat;
        let audio_c = audio_dims[1];
        let audio_t = audio_dims[2];
        let audio_m = audio_dims[3];

        let v_flat = latent
            .permute(&[0, 2, 3, 4, 1])?
            .contiguous()?
            .reshape(&[b, n_video, IN_CHANNELS])?;
        let a_flat = audio_latent
            .permute(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape(&[b, audio_t, AUDIO_IN_CHANNELS])?;

        let mut hidden = self.linear(&v_flat, "proj_in.weight", Some("proj_in.bias"))?;
        let mut audio_hidden =
            self.linear(&a_flat, "audio_proj_in.weight", Some("audio_proj_in.bias"))?;

        let video_text = self.prepare_video_text(text_emb)?;
        let audio_text = self.prepare_audio_text(audio_text_emb)?;

        let v_temb = self.ada_modulation_for(timestep, "time_embed", INNER_DIM, 9)?;
        let a_temb = self.ada_modulation_for(timestep, "audio_time_embed", AUDIO_INNER_DIM, 9)?;
        let v_ca_ss = self.optional_adaln_mod(
            timestep,
            "av_cross_attn_video_scale_shift",
            INNER_DIM,
            4,
        )?;
        let a_ca_ss = self.optional_adaln_mod(
            timestep,
            "av_cross_attn_audio_scale_shift",
            AUDIO_INNER_DIM,
            4,
        )?;
        let v_ca_gate = self.optional_adaln_mod(
            timestep,
            "av_cross_attn_video_a2v_gate",
            INNER_DIM,
            1,
        )?;
        let a_ca_gate = self.optional_adaln_mod(
            timestep,
            "av_cross_attn_audio_v2a_gate",
            AUDIO_INNER_DIM,
            1,
        )?;

        let temb_for_out = self.embedded_timestep_for(timestep, "time_embed")?;
        let audio_temb_for_out = self.embedded_timestep_for(timestep, "audio_time_embed")?;

        let (v_cos, v_sin) = self.build_video_rope(f, h_lat, w_lat, fps)?;
        let (a_cos, a_sin) = self.build_audio_rope(audio_t)?;
        let (ca_v_cos, ca_v_sin) = self.build_video_temporal_rope(f, h_lat, w_lat, fps)?;
        let (ca_a_cos, ca_a_sin) = self.build_audio_rope(audio_t)?;

        for i in 0..NUM_LAYERS {
            let (new_hidden, new_audio_hidden) = self.block_forward_av(
                hidden,
                audio_hidden,
                &video_text,
                &audio_text,
                timestep,
                &v_temb,
                &a_temb,
                &v_ca_ss,
                &a_ca_ss,
                &v_ca_gate,
                &a_ca_gate,
                &v_cos,
                &v_sin,
                &a_cos,
                &a_sin,
                &ca_v_cos,
                &ca_v_sin,
                &ca_a_cos,
                &ca_a_sin,
                i,
                b,
            )?;
            hidden = new_hidden;
            audio_hidden = new_audio_hidden;
        }

        let x_n = flame_core::layer_norm::layer_norm(&hidden, &[INNER_DIM], None, None, NORM_EPS)?;
        let sst = self
            .w("scale_shift_table")?
            .reshape(&[1, 1, 2, INNER_DIM])?
            .to_dtype(DType::BF16)?;
        let temb_b = temb_for_out.reshape(&[b, 1, 1, INNER_DIM])?;
        let scale_shift = sst.add(&temb_b)?;
        let shift = scale_shift.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let scale = scale_shift.narrow(2, 1, 1)?.squeeze_dim(2)?;
        let x_out = x_n.mul(&scale.add_scalar(1.0)?)?.add(&shift)?;
        let video_projected = self.linear(&x_out, "proj_out.weight", Some("proj_out.bias"))?;
        let video_out = video_projected
            .reshape(&[b, f, h_lat, w_lat, OUT_CHANNELS])?
            .permute(&[0, 4, 1, 2, 3])?
            .contiguous()?;

        let a_n = flame_core::layer_norm::layer_norm(
            &audio_hidden,
            &[AUDIO_INNER_DIM],
            None,
            None,
            NORM_EPS,
        )?;
        let a_sst = self
            .w("audio_scale_shift_table")?
            .reshape(&[1, 1, 2, AUDIO_INNER_DIM])?
            .to_dtype(DType::BF16)?;
        let a_temb_b = audio_temb_for_out.reshape(&[b, 1, 1, AUDIO_INNER_DIM])?;
        let a_scale_shift = a_sst.add(&a_temb_b)?;
        let a_shift = a_scale_shift.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let a_scale = a_scale_shift.narrow(2, 1, 1)?.squeeze_dim(2)?;
        let a_out = a_n.mul(&a_scale.add_scalar(1.0)?)?.add(&a_shift)?;
        let audio_projected =
            self.linear(&a_out, "audio_proj_out.weight", Some("audio_proj_out.bias"))?;
        let audio_out = audio_projected
            .reshape(&[b, audio_t, audio_c, audio_m])?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?;

        Ok((video_out, audio_out))
    }

    #[allow(clippy::too_many_arguments)]
    fn block_forward_av(
        &self,
        x: Tensor,
        audio_x: Tensor,
        video_text: &Tensor,
        audio_text: &Tensor,
        timestep: &Tensor,
        v_temb: &Tensor,
        a_temb: &Tensor,
        v_ca_ss: &Tensor,
        a_ca_ss: &Tensor,
        v_ca_gate: &Tensor,
        a_ca_gate: &Tensor,
        v_cos: &Tensor,
        v_sin: &Tensor,
        a_cos: &Tensor,
        a_sin: &Tensor,
        ca_v_cos: &Tensor,
        ca_v_sin: &Tensor,
        ca_a_cos: &Tensor,
        ca_a_sin: &Tensor,
        layer_idx: usize,
        b: usize,
    ) -> Result<(Tensor, Tensor)> {
        let lora_base = self.lora_idx(layer_idx, 0);
        let pre = format!("transformer_blocks.{layer_idx}");

        let (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.block_ada_params_6(&format!("{pre}.scale_shift_table"), v_temb, b, INNER_DIM)?;
        let (
            a_shift_msa,
            a_scale_msa,
            a_gate_msa,
            a_shift_mlp,
            a_scale_mlp,
            a_gate_mlp,
        ) = self.block_ada_params_6(
            &format!("{pre}.audio_scale_shift_table"),
            a_temb,
            b,
            AUDIO_INNER_DIM,
        )?;

        let t_video = x.shape().dims()[1];
        let t_audio = audio_x.shape().dims()[1];

        let v_norm = self.rms_norm_dim(&x, &format!("{pre}.norm1.weight"), INNER_DIM)?;
        let v_mod = v_norm.mul(&scale_msa.add_scalar(1.0)?)?.add(&shift_msa)?;
        let v_sa = self.attention_forward_lora(
            &v_mod,
            &v_mod,
            &format!("{pre}.attn1"),
            HEADS,
            HEAD_DIM,
            Some((v_cos, v_sin)),
            Some((v_cos, v_sin)),
            lora_base,
            lora_base + 1,
            lora_base + 2,
            lora_base + 3,
        )?;
        let mut x = x.add(&v_sa.mul(&gate_msa)?)?;

        let a_norm = self.rms_norm_dim(
            &audio_x,
            &format!("{pre}.audio_norm1.weight"),
            AUDIO_INNER_DIM,
        )?;
        let a_mod = a_norm.mul(&a_scale_msa.add_scalar(1.0)?)?.add(&a_shift_msa)?;
        let a_sa = self.attention_forward_lora(
            &a_mod,
            &a_mod,
            &format!("{pre}.audio_attn1"),
            AUDIO_HEADS,
            AUDIO_HEAD_DIM,
            Some((a_cos, a_sin)),
            Some((a_cos, a_sin)),
            lora_base + 6,
            lora_base + 7,
            lora_base + 8,
            lora_base + 9,
        )?;
        let mut audio_x = audio_x.add(&a_sa.mul(&a_gate_msa)?)?;

        let (v_shift_ca, v_scale_ca, v_gate_ca) =
            self.block_ada_params_ca(&format!("{pre}.scale_shift_table"), v_temb, b, INNER_DIM)?;
        let v_norm2 = self.rms_norm_dim(&x, &format!("{pre}.norm2.weight"), INNER_DIM)?;
        let v_mod2 = v_norm2.mul(&v_scale_ca.add_scalar(1.0)?)?.add(&v_shift_ca)?;
        let video_text = self.text_prompt_modulation(
            video_text,
            timestep,
            "prompt_adaln_single",
            &format!("{pre}.prompt_scale_shift_table"),
            INNER_DIM,
        )?;
        let v_ca = self.attention_forward(
            &v_mod2,
            &video_text,
            &format!("{pre}.attn2"),
            INNER_DIM,
            INNER_DIM,
            INNER_DIM,
            HEADS,
            HEAD_DIM,
            None,
            None,
        )?;
        x = x.add(&v_ca.mul(&v_gate_ca)?)?;

        let (a_shift_ca, a_scale_ca, a_gate_ca2) = self.block_ada_params_ca(
            &format!("{pre}.audio_scale_shift_table"),
            a_temb,
            b,
            AUDIO_INNER_DIM,
        )?;
        let a_norm2 = self.rms_norm_dim(
            &audio_x,
            &format!("{pre}.audio_norm2.weight"),
            AUDIO_INNER_DIM,
        )?;
        let a_mod2 = a_norm2.mul(&a_scale_ca.add_scalar(1.0)?)?.add(&a_shift_ca)?;
        let audio_text = self.text_prompt_modulation(
            audio_text,
            timestep,
            "audio_prompt_adaln_single",
            &format!("{pre}.audio_prompt_scale_shift_table"),
            AUDIO_INNER_DIM,
        )?;
        let a_ca = self.attention_forward_lora(
            &a_mod2,
            &audio_text,
            &format!("{pre}.audio_attn2"),
            AUDIO_HEADS,
            AUDIO_HEAD_DIM,
            None,
            None,
            lora_base + 10,
            lora_base + 11,
            lora_base + 12,
            lora_base + 13,
        )?;
        audio_x = audio_x.add(&a_ca.mul(&a_gate_ca2)?)?;

        let norm_a2v = self.rms_norm_dim(
            &x,
            &format!("{pre}.audio_to_video_norm.weight"),
            INNER_DIM,
        )?;
        let norm_v2a = self.rms_norm_dim(
            &audio_x,
            &format!("{pre}.video_to_audio_norm.weight"),
            AUDIO_INNER_DIM,
        )?;
        let (a2v_gate, v2a_gate, video_a2v_mod, video_v2a_mod, audio_a2v_mod, audio_v2a_mod) =
            self.cross_modal_params(pre.as_str(), v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate, b)?;

        let mod_video_a2v = norm_a2v
            .mul(&video_a2v_mod.0.add_scalar(1.0)?)?
            .add(&video_a2v_mod.1)?;
        let mod_audio_a2v = norm_v2a
            .mul(&audio_a2v_mod.0.add_scalar(1.0)?)?
            .add(&audio_a2v_mod.1)?;
        let a2v = self.attention_forward_lora(
            &mod_video_a2v,
            &mod_audio_a2v,
            &format!("{pre}.audio_to_video_attn"),
            AUDIO_HEADS,
            AUDIO_HEAD_DIM,
            Some((ca_v_cos, ca_v_sin)),
            Some((ca_a_cos, ca_a_sin)),
            lora_base + 14,
            lora_base + 15,
            lora_base + 16,
            lora_base + 17,
        )?;
        x = x.add(&a2v.mul(&a2v_gate)?)?;

        let mod_video_v2a = norm_a2v
            .mul(&video_v2a_mod.0.add_scalar(1.0)?)?
            .add(&video_v2a_mod.1)?;
        let mod_audio_v2a = norm_v2a
            .mul(&audio_v2a_mod.0.add_scalar(1.0)?)?
            .add(&audio_v2a_mod.1)?;
        let v2a = self.attention_forward_lora(
            &mod_audio_v2a,
            &mod_video_v2a,
            &format!("{pre}.video_to_audio_attn"),
            AUDIO_HEADS,
            AUDIO_HEAD_DIM,
            Some((ca_a_cos, ca_a_sin)),
            Some((ca_v_cos, ca_v_sin)),
            lora_base + 18,
            lora_base + 19,
            lora_base + 20,
            lora_base + 21,
        )?;
        audio_x = audio_x.add(&v2a.mul(&v2a_gate)?)?;

        let v_norm3 = self.rms_norm_dim(&x, &format!("{pre}.norm3.weight"), INNER_DIM)?;
        let v_mlp_in = v_norm3.mul(&scale_mlp.add_scalar(1.0)?)?.add(&shift_mlp)?;
        let ff1 = self.linear_lora(
            &v_mlp_in,
            &format!("{pre}.ff.net.0.proj.weight"),
            Some(&format!("{pre}.ff.net.0.proj.bias")),
            lora_base + 4,
        )?;
        let ff1 = gelu_tanh(&ff1)?;
        let ff2 = self.linear_lora(
            &ff1,
            &format!("{pre}.ff.net.2.weight"),
            Some(&format!("{pre}.ff.net.2.bias")),
            lora_base + 5,
        )?;
        x = x.add(&ff2.mul(&gate_mlp)?)?;

        let a_norm3 = self.rms_norm_dim(
            &audio_x,
            &format!("{pre}.audio_norm3.weight"),
            AUDIO_INNER_DIM,
        )?;
        let a_mlp_in = a_norm3.mul(&a_scale_mlp.add_scalar(1.0)?)?.add(&a_shift_mlp)?;
        let aff1 = self.linear_lora(
            &a_mlp_in,
            &format!("{pre}.audio_ff.net.0.proj.weight"),
            Some(&format!("{pre}.audio_ff.net.0.proj.bias")),
            lora_base + 22,
        )?;
        let aff1 = gelu_tanh(&aff1)?;
        let aff2 = self.linear_lora(
            &aff1,
            &format!("{pre}.audio_ff.net.2.weight"),
            Some(&format!("{pre}.audio_ff.net.2.bias")),
            lora_base + 23,
        )?;
        audio_x = audio_x.add(&aff2.mul(&a_gate_mlp)?)?;

        let _ = (t_video, t_audio);
        Ok((x, audio_x))
    }

    /// One block, T2V slice (audio sub-modules excised — see module docs
    /// for safety justification). Implements:
    ///   1. video self-attn with RoPE + AdaLN modulation
    ///   2. video cross-attn-text (no RoPE on text side)
    ///   3. video FF (gelu-approximate) with AdaLN modulation
    #[allow(clippy::too_many_arguments)]
    fn block_forward_t2v(
        &self,
        x: Tensor,
        text_proj: &Tensor,
        shift_msa: &Tensor,
        scale_msa: &Tensor,
        gate_msa: &Tensor,
        shift_mlp: &Tensor,
        scale_mlp: &Tensor,
        gate_mlp: &Tensor,
        cos_b: &Tensor,
        sin_b: &Tensor,
        layer_idx: usize,
        b: usize,
        t_video: usize,
    ) -> Result<Tensor> {
        let lora_base = self.lora_idx(layer_idx, 0);
        let pre = format!("transformer_blocks.{layer_idx}");

        // ── 1. Video self-attn ──
        let r = x.clone();
        let n = self.rms_norm_full(&x, &format!("{pre}.norm1.weight"))?;
        // Per-block scale_shift_table [6, dim]; first 3 rows = self-attn (shift, scale, gate).
        // We use the GLOBAL ada modulation for now (no per-block table addition) —
        // diffusers does `scale_shift_table[None, None] + temb.reshape(B, T, 6, -1)`.
        // That per-block table is small (6 × 4096); add it.
        let sst = self
            .w(&format!("{pre}.scale_shift_table"))?
            .reshape(&[1, 1, 6, INNER_DIM])?
            .to_dtype(DType::BF16)?;
        // global mods are already chunked; per-block table has 6 separate slots.
        let block_shift_msa = sst.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let block_scale_msa = sst.narrow(2, 1, 1)?.squeeze_dim(2)?;
        let block_gate_msa = sst.narrow(2, 2, 1)?.squeeze_dim(2)?;
        let block_shift_mlp = sst.narrow(2, 3, 1)?.squeeze_dim(2)?;
        let block_scale_mlp = sst.narrow(2, 4, 1)?.squeeze_dim(2)?;
        let block_gate_mlp = sst.narrow(2, 5, 1)?.squeeze_dim(2)?;

        let s_msa = block_shift_msa.add(shift_msa)?;
        let sc_msa = block_scale_msa.add(scale_msa)?;
        let g_msa = block_gate_msa.add(gate_msa)?;
        let s_mlp = block_shift_mlp.add(shift_mlp)?;
        let sc_mlp = block_scale_mlp.add(scale_mlp)?;
        let g_mlp = block_gate_mlp.add(gate_mlp)?;

        let m = n.mul(&sc_msa.add_scalar(1.0)?)?.add(&s_msa)?;

        let q = self.linear_lora(
            &m,
            &format!("{pre}.attn1.to_q.weight"),
            Some(&format!("{pre}.attn1.to_q.bias")),
            lora_base,
        )?;
        let k = self.linear_lora(
            &m,
            &format!("{pre}.attn1.to_k.weight"),
            Some(&format!("{pre}.attn1.to_k.bias")),
            lora_base + 1,
        )?;
        let v = self.linear_lora(
            &m,
            &format!("{pre}.attn1.to_v.weight"),
            Some(&format!("{pre}.attn1.to_v.bias")),
            lora_base + 2,
        )?;
        let q_n = self.qk_norm(&q, &format!("{pre}.attn1.norm_q.weight"))?;
        let k_n = self.qk_norm(&k, &format!("{pre}.attn1.norm_k.weight"))?;

        // RoPE on Q, K; not on V.
        let q_r = self.apply_rope(&q_n, cos_b, sin_b)?;
        let k_r = self.apply_rope(&k_n, cos_b, sin_b)?;

        let qh = self.to_heads(&q_r, b, t_video)?;
        let kh = self.to_heads(&k_r, b, t_video)?;
        let vh = self.to_heads(&v, b, t_video)?;
        let attn_out = flame_core::attention::sdpa(&qh, &kh, &vh, None)?;
        let attn_out = self.from_heads(&attn_out, b, t_video)?;
        let out = self.linear_lora(
            &attn_out,
            &format!("{pre}.attn1.to_out.0.weight"),
            Some(&format!("{pre}.attn1.to_out.0.bias")),
            lora_base + 3,
        )?;
        let x = r.add(&g_msa.mul(&out)?)?;

        // ── 2. Video cross-attn to text ──
        let r2 = x.clone();
        let n2 = self.rms_norm_full(&x, &format!("{pre}.norm2.weight"))?;
        // attn2 is cross-attn: K, V come from `text_proj`.
        let q2 = self.linear(
            &n2,
            &format!("{pre}.attn2.to_q.weight"),
            Some(&format!("{pre}.attn2.to_q.bias")),
        )?;
        let k2 = self.linear(
            text_proj,
            &format!("{pre}.attn2.to_k.weight"),
            Some(&format!("{pre}.attn2.to_k.bias")),
        )?;
        let v2 = self.linear(
            text_proj,
            &format!("{pre}.attn2.to_v.weight"),
            Some(&format!("{pre}.attn2.to_v.bias")),
        )?;
        let q2_n = self.qk_norm(&q2, &format!("{pre}.attn2.norm_q.weight"))?;
        let k2_n = self.qk_norm(&k2, &format!("{pre}.attn2.norm_k.weight"))?;

        let t_text = text_proj.shape().dims()[1];
        let q2h = self.to_heads(&q2_n, b, t_video)?;
        let k2h = self.to_heads(&k2_n, b, t_text)?;
        let v2h = self.to_heads(&v2, b, t_text)?;
        let attn2_out = flame_core::attention::sdpa(&q2h, &k2h, &v2h, None)?;
        let attn2_out = self.from_heads(&attn2_out, b, t_video)?;
        let out2 = self.linear(
            &attn2_out,
            &format!("{pre}.attn2.to_out.0.weight"),
            Some(&format!("{pre}.attn2.to_out.0.bias")),
        )?;
        let x = r2.add(&out2)?;

        // ── 3. Video FF ──
        let r3 = x.clone();
        let n3 = self.rms_norm_full(&x, &format!("{pre}.norm3.weight"))?;
        let m3 = n3.mul(&sc_mlp.add_scalar(1.0)?)?.add(&s_mlp)?;
        // ff.net.0.proj: 4096 → 16384 → GELU(tanh) → ff.net.2: 16384 → 4096
        let ff1 = self.linear_lora(
            &m3,
            &format!("{pre}.ff.net.0.proj.weight"),
            Some(&format!("{pre}.ff.net.0.proj.bias")),
            lora_base + 4,
        )?;
        // GELU tanh-approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // flame's `Tensor::gelu` is the exact GELU; for parity we hand-roll
        // the tanh approximation here.
        let ff1 = gelu_tanh(&ff1)?;
        let ff2 = self.linear_lora(
            &ff1,
            &format!("{pre}.ff.net.2.weight"),
            Some(&format!("{pre}.ff.net.2.bias")),
            lora_base + 5,
        )?;
        let x = r3.add(&g_mlp.mul(&ff2)?)?;

        Ok(x)
    }
}

/// GELU tanh-approx (matches PyTorch `gelu(approximate='tanh')`).
fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    // 0.044715, sqrt(2/pi) ≈ 0.7978845608.
    let cube = x.mul(x)?.mul(x)?.mul_scalar(0.044715)?;
    let arg = x.add(&cube)?.mul_scalar(0.7978845608)?;
    let t = arg.tanh()?;
    let inner = t.add_scalar(1.0)?;
    x.mul(&inner)?.mul_scalar(0.5).map_err(Into::into)
}

impl TrainableModel for Ltx2Model {
    fn forward(
        &mut self,
        noisy: &Tensor,
        timestep: &Tensor,
        context: &[Tensor],
        _p: Option<&Tensor>,
    ) -> Result<Tensor> {
        let txt = context.first().ok_or_else(|| {
            crate::EriDiffusionError::Model("LTX-2 needs text embeddings in context[0]".into())
        })?;
        // FPS not exposed via TrainableModel; default 24.0 (LTX-2 standard).
        Ltx2Model::forward(self, noisy, txt, timestep, 24.0)
    }
    fn parameters(&self) -> Vec<Parameter> {
        self.parameters.clone()
    }
    fn post_optimizer_step(&mut self) {}

    fn save_weights(&self, path: &str) -> Result<()> {
        if !self.is_lora {
            return Err(crate::EriDiffusionError::Model(
                "save_weights for non-LoRA LTX-2 not implemented".into(),
            ));
        }
        let mut out = HashMap::new();
        for (i, adapter) in self.lora_adapters.iter().enumerate() {
            let layer_idx = i / self.lora_slots_per_block;
            let slot = i % self.lora_slots_per_block;
            let prefix = format!("transformer_blocks.{layer_idx}.{}", LORA_SLOT_KEYS[slot]);
            adapter.save_tensors(&prefix, &mut out)?;
        }
        flame_core::serialization::save_file(&out, std::path::Path::new(path))
            .map_err(|e| crate::EriDiffusionError::Safetensors(format!("save_file: {e}")))?;
        Ok(())
    }

    fn load_weights(&mut self, path: &str) -> Result<()> {
        if !self.is_lora {
            return Err(crate::EriDiffusionError::Model(
                "load_weights for non-LoRA LTX-2 not implemented".into(),
            ));
        }
        let source = flame_core::serialization::load_file(std::path::Path::new(path), &self.device)
            .map_err(|e| crate::EriDiffusionError::Safetensors(format!("load_file: {e}")))?;
        for (i, adapter) in self.lora_adapters.iter().enumerate() {
            let layer_idx = i / self.lora_slots_per_block;
            let slot = i % self.lora_slots_per_block;
            let prefix = format!("transformer_blocks.{layer_idx}.{}", LORA_SLOT_KEYS[slot]);
            let has_new = source.contains_key(&format!("{prefix}.lora_A.weight"))
                && source.contains_key(&format!("{prefix}.lora_B.weight"));
            let has_legacy = source.contains_key(&format!("{prefix}.lora_A"))
                && source.contains_key(&format!("{prefix}.lora_B"));
            if has_new || has_legacy {
                adapter.load_tensors(&prefix, &source)?;
            } else if slot < 6 {
                return Err(crate::EriDiffusionError::Lora(format!(
                    "missing LTX-2 LoRA tensors for required video slot {prefix}"
                )));
            } else {
                log::warn!("LTX-2 LoRA checkpoint missing optional AV slot {prefix}; leaving initialized");
            }
        }
        Ok(())
    }
}
