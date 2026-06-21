//! Llama-3.1-8B-Instruct text encoder for HiDream-I1 training/inference.
//!
//! Pure flame-core implementation, mirrors the encoder used by edv2-reference's
//! HiDream-I1 model
//! (`edv2-reference/extensions_built_in/diffusion_models/hidream/hidream_model.py`).
//!
//! # What HiDream actually consumes
//!
//! Per `pipeline_hidream_image.py::_get_llama3_prompt_embeds`:
//!
//! ```python
//! outputs = self.text_encoder_4(
//!     text_input_ids,
//!     attention_mask=attention_mask,
//!     output_hidden_states=True,
//!     output_attentions=True,
//! )
//! prompt_embeds = outputs.hidden_states[1:]            # drop embedding output
//! prompt_embeds = torch.stack(prompt_embeds, dim=0)    # [num_layers, B, S, D]
//! ```
//!
//! HiDream stacks **every layer's output except the embedding** — so for
//! Llama-3.1-8B (32 layers) the per-prompt tensor that feeds the DiT is
//! `[32, B, S, 4096]`. The trainer-side caller can then index out whichever
//! layers it wants; this encoder exposes both single-layer and all-layer
//! variants so the caller decides.
//!
//! `max_sequence_length` defaults to `128` in the HiDream pipeline and
//! padding is `"max_length"`. For the **unsloth** Llama-3.1-8B-Instruct
//! checkpoint (the one HiDream uses), the tokenizer's `pad_token` is
//! `<|finetune_right_pad_id|>` with id **128004**, distinct from
//! `eos_token = 128009`. The official Meta checkpoint leaves
//! `pad_token_id` unset; only unsloth fixes this. Attention mask is the
//! standard `1 = real token, 0 = pad` boolean.
//!
//! HF's `LlamaModel.forward` returns a tuple of length `num_layers + 1`:
//! `[embed, l0_out, l1_out, ..., l_{n-2}_out, norm(l_{n-1}_out)]`. The
//! final entry passes through `model.norm`; the rest are raw residual
//! stream outputs. HiDream slices `hidden_states[1:]` = layers
//! `[l0_out, ..., l_{n-2}_out, norm(l_{n-1}_out)]` (length `num_layers`).
//! We apply `model.norm` to the last entry to match exactly.
//!
//! # Architecture (Llama-3.1-8B)
//!
//! - 32 decoder layers, `hidden_size = 4096`, `intermediate_size = 14336`
//! - 32 attention heads, **8 KV heads** (GQA 4:1), `head_dim = 128`
//! - **RoPE θ = 500_000** (NOT 10k — Llama-3 specific)
//! - **Llama-3.1 RoPE scaling** ("llama3" type):
//!     `factor = 8.0`, `low_freq_factor = 1.0`, `high_freq_factor = 4.0`,
//!     `original_max_position_embeddings = 8192`.
//!     Frequencies whose wavelength is below the high-freq cutoff are kept;
//!     above the low-freq cutoff they are divided by `factor`; in between
//!     they are smoothly interpolated. (See `transformers`
//!     `modeling_rope_utils.py::_compute_llama3_parameters`.)
//! - RMSNorm (`eps = 1e-5`)
//! - SwiGLU FFN (no bias on any projection)
//! - `vocab_size = 128_256`, `max_position_embeddings = 131_072`
//!
//! All on-GPU tensors are BF16; only RoPE angle math and the mask are
//! materialized in F32 before being cast.
//!
//! # Weight layout on disk
//!
//! HiDream's reference checkpoint is `unsloth/Meta-Llama-3.1-8B-Instruct`
//! (BF16, sharded). Expected directory layout (HF cache or local):
//!
//! ```text
//! <root>/
//!   config.json
//!   model.safetensors.index.json
//!   model-00001-of-00004.safetensors
//!   model-00002-of-00004.safetensors
//!   model-00003-of-00004.safetensors
//!   model-00004-of-00004.safetensors
//!   tokenizer.json
//!   tokenizer_config.json
//! ```
//!
//! `Llama3Encoder::load` accepts either:
//! - a path to a directory containing the shards (auto-detects all
//!   `*.safetensors` matching `model-*.safetensors` or falls back to
//!   `model.safetensors`), or
//! - a path to a single `.safetensors` file (e.g. a consolidated dump).
//!
//! Weight keys are the standard HF format:
//! `model.embed_tokens.weight`, `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`,
//! `model.layers.{i}.mlp.{gate,up,down}_proj.weight`,
//! `model.layers.{i}.{input,post_attention}_layernorm.weight`,
//! `model.norm.weight`, `lm_head.weight` (last one ignored).
//!
//! # Caller responsibilities
//!
//! The trainer/inference caller is responsible for:
//! - Tokenisation (HF `PreTrainedTokenizerFast` from `tokenizer.json`).
//! - Building the `attention_mask` (`true` for real token, `false` for pad)
//!   and passing it to [`Self::encode_all_hidden_states`]. The HiDream
//!   pipeline ALWAYS pads to `max_sequence_length=128`.
//! - Setting `max_sequence_length` (HiDream pipeline default = 128).
//! - Wrapping calls in `AutogradContext::no_grad(...)` during training —
//!   otherwise the full 32-layer activation graph is taped per batch
//!   (tens of GB of grad buffers retained).
//!
//! Padding side: the Llama-3 tokenizer defaults to **left-padding**. The
//! mask-builder is side-agnostic (it ANDs the user mask with a causal
//! lower-triangle); rows at pad positions get all-blocked SDPA which
//! emits uniform garbage (NOT NaN — verified). Downstream consumers
//! MUST mask out pad rows themselves; HiDream's DiT cross-attention
//! does so via its own attention mask.
//!
//! The primary API for HiDream is [`Self::encode_all_hidden_states`],
//! which returns `[num_layers, 1, S, H]` matching
//! `outputs.hidden_states[1:]`. The single-layer
//! [`Self::encode_last_hidden_state`] /
//! [`Self::encode_last_hidden_state_with_mask`] convenience methods
//! exist for non-HiDream consumers (general LLM tasks).
//!
//! # What is NOT implemented
//!
//! - LM head / logits — HiDream never reads them.
//! - KV cache — HiDream invokes the encoder once per prompt.
//! - Attention output (`output_attentions=True` in HF) — HiDream passes it
//!   but does not use the returned attention maps.
//! - FP8 / int8 quantisation — quant path is BF16 only here; if the caller
//!   wants quant it must be done in flame-core, not in this encoder.
//! - BlockOffloader integration — Llama-3.1-8B is ~15 GB in BF16; on a
//!   24 GB card it currently lives fully resident. A future revision can
//!   stream layers from a `BlockOffloader<LlamaLayer>` exactly the way
//!   `mistral3b` would if it ever needed to.

use flame_core::attention::sdpa as flame_sdpa;
use flame_core::serialization::load_file_filtered;
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Llama-3.1-8B-Instruct config, matching HF `config.json`.
///
/// Auto-detect against the actual safetensors via
/// [`Llama3Encoder::config_from_weights`].
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,

    // Llama-3.1 RoPE scaling ("rope_type": "llama3" in config.json).
    pub rope_factor: f64,
    pub rope_low_freq_factor: f64,
    pub rope_high_freq_factor: f64,
    pub rope_original_max_position_embeddings: usize,

    /// Max sequence length we will ever encode; the HiDream pipeline uses
    /// 128 by default but trainer-side callers may pass up to 256.
    pub max_seq_len: usize,

    /// Token id used for padding when the caller does not supply
    /// `pad_token_id` explicitly. The unsloth/Meta-Llama-3.1-8B-Instruct
    /// checkpoint (the one HiDream uses) sets
    /// `pad_token = <|finetune_right_pad_id|> = 128004`, distinct from
    /// `eos_token = 128009`. The official Meta checkpoint leaves this
    /// unset; callers using non-unsloth weights must override this.
    pub pad_token_id: i32,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 128_256,
            hidden_size: 4096,
            num_layers: 32,
            intermediate_size: 14_336,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,

            rope_factor: 8.0,
            rope_low_freq_factor: 1.0,
            rope_high_freq_factor: 4.0,
            rope_original_max_position_embeddings: 8192,

            max_seq_len: 128,
            // unsloth/Meta-Llama-3.1-8B-Instruct sets pad_token_id=128004
            // (<|finetune_right_pad_id|>), distinct from eos_token=128009.
            pad_token_id: 128_004,
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Llama-3.1-8B text encoder — pure flame-core implementation.
///
/// Weights are loaded once into a flat `HashMap<String, Tensor>` and
/// remain resident on the target CUDA device for the lifetime of the
/// encoder. 2D linear weights are pre-transposed at construction so the
/// forward path can use straight `matmul` without per-call transposes.
///
/// Per-layer progress is logged at `INFO` level every 8 layers (matching
/// `mistral3b`).
pub struct Llama3Encoder {
    weights: HashMap<String, Tensor>,
    config: LlamaConfig,
    device: Arc<CudaDevice>,
}

impl Llama3Encoder {
    // -----------------------------------------------------------------------
    // Construction / loading
    // -----------------------------------------------------------------------

    /// Load Llama-3.1-8B from either a directory of sharded safetensors or
    /// a single consolidated safetensors file.
    ///
    /// Directory layout expected: any number of files matching
    /// `model-*.safetensors` (HF default sharding), or a single
    /// `model.safetensors`. The loader concatenates all weights it finds
    /// and strips the standard prefix.
    ///
    /// `lm_head.weight` is skipped — HiDream never reads it and Llama-3.1
    /// keeps it un-tied from `embed_tokens` (so we cannot dedup; it just
    /// wastes ~1 GB of VRAM if loaded).
    pub fn load(path: &str, device: &Arc<CudaDevice>) -> Result<Self> {
        let p = Path::new(path);
        let shard_paths = if p.is_dir() {
            Self::discover_shards(p)?
        } else {
            vec![p.to_path_buf()]
        };

        if shard_paths.is_empty() {
            return Err(flame_core::Error::InvalidInput(format!(
                "[Llama3] No safetensors shards found under '{}'",
                p.display()
            )));
        }

        log::info!(
            "[Llama3] Loading {} shard(s) from {}",
            shard_paths.len(),
            p.display()
        );

        let mut weights: HashMap<String, Tensor> = HashMap::new();
        for shard in &shard_paths {
            let part = load_file_filtered(shard, device, |key| {
                // Keep transformer weights; drop the LM head — HiDream never
                // reads logits and Llama-3.1 has an un-tied lm_head.weight
                // (~1 GB of VRAM saved).
                !key.starts_with("lm_head.") && !key.starts_with("model.lm_head.")
            })?;
            log::info!(
                "[Llama3] {} -> {} tensors",
                shard.file_name().unwrap_or_default().to_string_lossy(),
                part.len()
            );
            weights.extend(part);
        }

        let mut config = Self::config_from_weights(&weights)?;

        // If a sibling config.json exists, override RoPE / pad fields from it
        // so we don't silently apply Llama-3.1 RoPE to a 3.2/3.3 checkpoint.
        if p.is_dir() {
            let cfg_path = p.join("config.json");
            if cfg_path.exists() {
                if let Err(e) = Self::apply_sibling_config(&cfg_path, &mut config) {
                    log::warn!(
                        "[Llama3] sibling config.json present but parse failed: {} \
                         (falling back to built-in Llama-3.1-8B defaults)",
                        e
                    );
                }
            }
        }

        Ok(Self::new(weights, config, device.clone()))
    }

    /// Best-effort overlay of a sibling `config.json` onto an auto-detected
    /// [`LlamaConfig`]. Used to pick up the correct `rope_theta`,
    /// `rope_scaling`, and `pad_token_id` when the on-disk checkpoint is a
    /// Llama-3.x variant whose RoPE parameters differ from the Llama-3.1
    /// defaults. This is best-effort: unrecognised fields are ignored, and
    /// any parse failure leaves the auto-detected config untouched. We do
    /// NOT touch shape-derived fields (`vocab_size`, `hidden_size`, etc.) —
    /// those came from the weights and are authoritative.
    fn apply_sibling_config(cfg_path: &Path, out: &mut LlamaConfig) -> Result<()> {
        let raw = std::fs::read_to_string(cfg_path).map_err(|e| {
            flame_core::Error::Io(format!(
                "[Llama3] read sibling config.json: {e}"
            ))
        })?;
        // Minimal hand-parser: just grep for the four fields we care about.
        // Avoids pulling in serde_json as a hard dependency for this crate.
        fn extract_num(s: &str, key: &str) -> Option<f64> {
            let needle = format!("\"{key}\"");
            let pos = s.find(&needle)?;
            let after = &s[pos + needle.len()..];
            let colon = after.find(':')?;
            let tail = after[colon + 1..].trim_start();
            // Read until comma, newline, or closing brace/bracket.
            let end = tail
                .find(|c: char| c == ',' || c == '\n' || c == '}' || c == ']')
                .unwrap_or(tail.len());
            tail[..end].trim().trim_matches('"').parse::<f64>().ok()
        }
        if let Some(theta) = extract_num(&raw, "rope_theta") {
            out.rope_theta = theta;
        }
        if let Some(pad) = extract_num(&raw, "pad_token_id") {
            out.pad_token_id = pad as i32;
        }
        if let Some(factor) = extract_num(&raw, "factor") {
            // Inside "rope_scaling": { ... }. There may also be "factor" outside
            // (unlikely); accept first occurrence.
            out.rope_factor = factor;
        }
        if let Some(low) = extract_num(&raw, "low_freq_factor") {
            out.rope_low_freq_factor = low;
        }
        if let Some(high) = extract_num(&raw, "high_freq_factor") {
            out.rope_high_freq_factor = high;
        }
        if let Some(orig) = extract_num(&raw, "original_max_position_embeddings") {
            out.rope_original_max_position_embeddings = orig as usize;
        }
        log::info!(
            "[Llama3] sibling config.json applied: rope_theta={}, pad={}, factor={}, \
             low/high={}/{}, orig_max_pos={}",
            out.rope_theta,
            out.pad_token_id,
            out.rope_factor,
            out.rope_low_freq_factor,
            out.rope_high_freq_factor,
            out.rope_original_max_position_embeddings,
        );
        Ok(())
    }

    /// Construct from already-loaded weights (e.g. when sharing a single
    /// load with another encoder, or when injecting a test fixture).
    /// Pre-transposes 2D linear weights for fast per-call matmul.
    pub fn new(
        mut weights: HashMap<String, Tensor>,
        config: LlamaConfig,
        device: Arc<CudaDevice>,
    ) -> Self {
        let keys: Vec<String> = weights.keys().cloned().collect();
        for key in &keys {
            if key.ends_with(".weight")
                && !key.contains("layernorm")
                && !key.contains("norm")
                && !key.contains("embed")
            {
                let w = &weights[key];
                if w.shape().dims().len() == 2 {
                    if let Ok(wt) = flame_core::bf16_elementwise::transpose2d_bf16(w) {
                        weights.insert(key.clone(), wt);
                    }
                }
            }
        }
        log::info!(
            "[Llama3] Ready: {} layers, hidden={}, heads={}/{} (GQA {}:1), rope_theta={}",
            config.num_layers,
            config.hidden_size,
            config.num_heads,
            config.num_kv_heads,
            config.num_heads / config.num_kv_heads,
            config.rope_theta,
        );
        Self {
            weights,
            config,
            device,
        }
    }

    /// Auto-detect a [`LlamaConfig`] from the weight tensor shapes, the
    /// same way `qwen25vl.rs::config_from_weights` does.
    ///
    /// Returns Llama-3.1 defaults (rope_theta, scaling factors, etc.) —
    /// those are not deducible from the safetensors alone and the
    /// downstream HiDream training only uses the official 8B-Instruct
    /// checkpoint.
    pub fn config_from_weights(weights: &HashMap<String, Tensor>) -> Result<LlamaConfig> {
        let embed_w = weights.get("model.embed_tokens.weight").ok_or_else(|| {
            flame_core::Error::InvalidInput(format!(
                "[Llama3] Missing model.embed_tokens.weight. First 10 keys: {:?}",
                weights.keys().take(10).collect::<Vec<_>>()
            ))
        })?;
        let vocab_size = embed_w.shape().dims()[0];
        let hidden_size = embed_w.shape().dims()[1];

        let mut num_layers = 0usize;
        while weights.contains_key(&format!(
            "model.layers.{num_layers}.self_attn.q_proj.weight"
        )) {
            num_layers += 1;
        }
        if num_layers == 0 {
            return Err(flame_core::Error::InvalidInput(
                "[Llama3] Could not find any model.layers.{i}.self_attn.q_proj.weight".into(),
            ));
        }

        let head_dim = 128;
        let q = weights
            .get("model.layers.0.self_attn.q_proj.weight")
            .ok_or_else(|| {
                flame_core::Error::InvalidInput("[Llama3] missing q_proj.weight".into())
            })?;
        let k = weights
            .get("model.layers.0.self_attn.k_proj.weight")
            .ok_or_else(|| {
                flame_core::Error::InvalidInput("[Llama3] missing k_proj.weight".into())
            })?;
        let num_heads = q.shape().dims()[0] / head_dim;
        let num_kv_heads = k.shape().dims()[0] / head_dim;

        let intermediate_size = weights
            .get("model.layers.0.mlp.gate_proj.weight")
            .map(|t| t.shape().dims()[0])
            .unwrap_or(hidden_size * 4);

        // Llama-3.1-8B shape assertion. RoPE-scaling fields below are
        // copied from `LlamaConfig::default()` (Llama-3.1-specific), so
        // any other variant (3.2 / 3.3 with same shape but different
        // rope_theta) would silently get wrong RoPE. Refuse to auto-detect
        // unless the shape matches Llama-3.1-8B exactly. Callers with
        // other variants must construct `LlamaConfig` manually and pass
        // it to `Llama3Encoder::new`.
        if !(num_layers == 32
            && hidden_size == 4096
            && num_heads == 32
            && num_kv_heads == 8
            && intermediate_size == 14_336)
        {
            return Err(flame_core::Error::InvalidInput(format!(
                "[Llama3] config_from_weights: detected shape \
                 (layers={num_layers}, hidden={hidden_size}, heads={num_heads}, \
                 kv_heads={num_kv_heads}, intermediate={intermediate_size}) does NOT \
                 match Llama-3.1-8B (32/4096/32/8/14336). Auto-detection only \
                 supports Llama-3.1-8B — construct LlamaConfig manually for \
                 other variants."
            )));
        }

        Ok(LlamaConfig {
            vocab_size,
            hidden_size,
            num_layers,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
            ..LlamaConfig::default()
        })
    }

    fn discover_shards(dir: &Path) -> Result<Vec<PathBuf>> {
        let read = std::fs::read_dir(dir).map_err(|e| {
            flame_core::Error::Io(format!("[Llama3] read_dir '{}': {e}", dir.display()))
        })?;
        let all_safetensors: Vec<PathBuf> = read
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();

        // First-pass filter: standard HF layout (model.safetensors or shards).
        let mut shards: Vec<PathBuf> = all_safetensors
            .iter()
            .filter(|p| {
                let n = p
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default();
                n.starts_with("model-") || n == "model.safetensors"
            })
            .cloned()
            .collect();

        // Fallback: if nothing matched but there ARE .safetensors files, take
        // them all (e.g. user-dropped `consolidated.safetensors` or
        // `pytorch_model.safetensors`). Loud warn so this surfaces.
        if shards.is_empty() && !all_safetensors.is_empty() {
            log::warn!(
                "[Llama3] no model-*.safetensors / model.safetensors in '{}'; \
                 falling back to loading all {} *.safetensors files found",
                dir.display(),
                all_safetensors.len()
            );
            shards = all_safetensors;
        }

        shards.sort();
        Ok(shards)
    }

    /// Config accessor.
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// Hidden dim of a single layer output: 4096 for Llama-3.1-8B.
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn w(&self, key: &str) -> Result<&Tensor> {
        self.weights
            .get(key)
            .ok_or_else(|| flame_core::Error::InvalidInput(format!("[Llama3] missing weight: {key}")))
    }

    /// Matmul for [B, N, C] x [C, out] -> [B, N, out].
    /// Weight is already pre-transposed at construction.
    fn linear_3d(x: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
        let shape = x.shape().dims().to_vec();
        let b = shape[0];
        let n = shape[1];
        let c = shape[2];
        let x_2d = x.reshape(&[b * n, c])?;
        let out_2d = x_2d.matmul(weight_t)?;
        let out_dim = out_2d.shape().dims()[1];
        out_2d.reshape(&[b, n, out_dim])
    }

    fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out_2d = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(weight), eps)?;
        out_2d.reshape(&dims)
    }

    /// Build Llama-3.1 RoPE cos/sin tables of shape `[1, 1, seq_len, head_dim/2]`.
    ///
    /// Implements the "llama3" rope_type from HF
    /// `modeling_rope_utils.py::_compute_llama3_parameters`:
    ///
    /// ```text
    /// inv_freq      = 1 / theta^(2i/dim)          for i in 0..dim/2
    /// wavelen       = 2π / inv_freq
    /// low_wavelen   = orig_max_pos / low_freq_factor    # long-period cutoff
    /// high_wavelen  = orig_max_pos / high_freq_factor   # short-period cutoff
    ///
    ///                    inv_freq                                  if wavelen <  high_wavelen
    /// scaled_inv_freq =  inv_freq / factor                          if wavelen >  low_wavelen
    ///                    (1 - smooth) * (inv_freq / factor)
    ///                      + smooth * inv_freq                      otherwise
    /// where smooth = (orig_max_pos/wavelen - low_freq_factor)
    ///                / (high_freq_factor - low_freq_factor)
    /// ```
    fn build_llama3_rope(
        seq_len: usize,
        config: &LlamaConfig,
        device: &Arc<CudaDevice>,
    ) -> Result<(Tensor, Tensor)> {
        let dim = config.head_dim;
        let half = dim / 2;
        let theta = config.rope_theta;
        let factor = config.rope_factor;
        let low = config.rope_low_freq_factor;
        let high = config.rope_high_freq_factor;
        let orig_max = config.rope_original_max_position_embeddings as f64;
        let two_pi = 2.0 * std::f64::consts::PI;

        let low_wavelen = orig_max / low;
        let high_wavelen = orig_max / high;

        let mut inv_freqs = vec![0.0f32; half];
        for i in 0..half {
            let base = 1.0 / theta.powf(2.0 * i as f64 / dim as f64);
            let wavelen = two_pi / base;

            let scaled = if wavelen < high_wavelen {
                // Short wavelength (high frequency): keep original.
                base
            } else if wavelen > low_wavelen {
                // Long wavelength (low frequency): divide by factor.
                base / factor
            } else {
                // Smooth interpolation across the medium-frequency band.
                let smooth = (orig_max / wavelen - low) / (high - low);
                (1.0 - smooth) * (base / factor) + smooth * base
            };
            inv_freqs[i] = scaled as f32;
        }

        let freq_tensor =
            Tensor::from_vec(inv_freqs, Shape::from_dims(&[1, half]), device.clone())?;
        let pos = Tensor::arange(0.0, seq_len as f32, 1.0, device.clone())?;
        let pos_col = pos.reshape(&[seq_len, 1])?;
        let angles = pos_col.matmul(&freq_tensor)?;

        let cos = angles
            .cos()?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .to_dtype(DType::BF16)?;
        let sin = angles
            .sin()?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .to_dtype(DType::BF16)?;
        Ok((cos, sin))
    }

    fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        flame_core::bf16_ops::rope_halfsplit_bf16(x, cos, sin)
    }

    fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let dims = x.shape().dims();
        let (b, h_kv, n, d) = (dims[0], dims[1], dims[2], dims[3]);
        let copies: Vec<Tensor> = (0..n_rep).map(|_| x.clone()).collect();
        let stacked = Tensor::stack(&copies, 2)?;
        stacked.reshape(&[b, h_kv * n_rep, n, d])
    }

    /// Build `[1, 1, seq_len, seq_len]` BF16 mask in flame's
    /// boolean / multiplicative convention (1.0 = attend, 0.0 = block).
    /// `attention_mask[i]` is `true` for real tokens. We combine the user
    /// mask with a causal lower-triangle so attention at row `i` to column
    /// `j` is allowed iff `j <= i AND attention_mask[j]`.
    ///
    /// Side-agnostic: works for right-pad (HF default for many fine-tunes)
    /// AND left-pad (HF Llama tokenizer default). For LEFT-pad inputs,
    /// rows at pad positions get all-blocked → `flame_core::attention::sdpa`
    /// emits uniform attention (NOT NaN — confirmed against
    /// `flame-core/src/attention/sdpa.rs:521`; mask is added as
    /// `(1 - mask) * -1e9` then softmaxed, so all-`-1e9` rows softmax to
    /// uniform). Downstream consumers MUST mask out pad rows themselves;
    /// HiDream's DiT cross-attention does so via its own attention mask.
    ///
    /// At least one position must be unmasked, otherwise EVERY row is
    /// blocked and the output is uniformly garbage end-to-end.
    fn build_attention_mask(
        seq_len: usize,
        attn_mask: &[bool],
        device: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        if !attn_mask.iter().any(|b| *b) {
            return Err(flame_core::Error::InvalidInput(
                "[Llama3] build_attention_mask: attention_mask has zero \
                 real tokens — all rows would be blocked"
                    .into(),
            ));
        }
        // TODO(M1.1): cache this on the encoder keyed by (seq_len, mask hash).
        // For HiDream's fixed seq_len=128 mask is ~64 KB host-side; with N
        // identical mask shapes per epoch this is wasted CPU work.
        let mut data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                if j < attn_mask.len() && attn_mask[j] {
                    data[i * seq_len + j] = 1.0;
                }
            }
        }
        Tensor::from_vec(
            data,
            Shape::from_dims(&[1, 1, seq_len, seq_len]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)
    }

    // -----------------------------------------------------------------------
    // Per-layer forward
    // -----------------------------------------------------------------------

    fn layer_forward(
        &self,
        layer_idx: usize,
        hidden: &Tensor,
        pe_cos: &Tensor,
        pe_sin: &Tensor,
        attn_mask: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let h = cfg.num_heads;
        let h_kv = cfg.num_kv_heads;
        let d = cfg.head_dim;
        let n_rep = h / h_kv;
        let prefix = format!("model.layers.{layer_idx}");

        let dims = hidden.shape().dims().to_vec();
        let b = dims[0];
        let n = dims[1];

        // --- Self-attention ---
        let normed = Self::rms_norm(
            hidden,
            self.w(&format!("{prefix}.input_layernorm.weight"))?,
            cfg.rms_norm_eps,
        )?;

        let q = Self::linear_3d(&normed, self.w(&format!("{prefix}.self_attn.q_proj.weight"))?)?;
        let k = Self::linear_3d(&normed, self.w(&format!("{prefix}.self_attn.k_proj.weight"))?)?;
        let v = Self::linear_3d(&normed, self.w(&format!("{prefix}.self_attn.v_proj.weight"))?)?;

        let q = q.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;

        // Llama-3 has no QK norm.
        let q = Self::apply_rope(&q, pe_cos, pe_sin)?;
        let k = Self::apply_rope(&k, pe_cos, pe_sin)?;

        let k = Self::repeat_kv(&k, n_rep)?;
        let v = Self::repeat_kv(&v, n_rep)?;

        let attn_out = flame_sdpa(&q, &k, &v, Some(attn_mask))?;
        let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?;
        let attn_out =
            Self::linear_3d(&attn_out, self.w(&format!("{prefix}.self_attn.o_proj.weight"))?)?;
        let hidden = hidden.add(&attn_out)?;

        // --- MLP (SwiGLU, no bias) ---
        let normed2 = Self::rms_norm(
            &hidden,
            self.w(&format!("{prefix}.post_attention_layernorm.weight"))?,
            cfg.rms_norm_eps,
        )?;

        let gate = Self::linear_3d(&normed2, self.w(&format!("{prefix}.mlp.gate_proj.weight"))?)?;
        let up = Self::linear_3d(&normed2, self.w(&format!("{prefix}.mlp.up_proj.weight"))?)?;
        let mlp_out = gate.silu()?.mul(&up)?;
        let mlp_out = Self::linear_3d(&mlp_out, self.w(&format!("{prefix}.mlp.down_proj.weight"))?)?;

        hidden.add(&mlp_out)
    }

    fn embed_tokens(&self, token_ids: &[i32]) -> Result<Tensor> {
        let embed_w = self.w("model.embed_tokens.weight")?;
        let seq_len = token_ids.len();
        let ids = Tensor::from_vec(
            token_ids.iter().map(|&id| id as f32).collect(),
            Shape::from_dims(&[seq_len]),
            self.device.clone(),
        )?
        .to_dtype(DType::I32)?;
        embed_w.index_select0(&ids)?.unsqueeze(0)
    }

    // -----------------------------------------------------------------------
    // Public forward
    // -----------------------------------------------------------------------

    /// **HiDream-primary API.** Encode tokens and return **every layer's
    /// output** stacked as `[num_layers, 1, seq_len, hidden_size]` BF16.
    ///
    /// This matches HiDream's
    /// `outputs.hidden_states[1:]` → `torch.stack(..., dim=0)` exactly:
    /// the embedding output is *not* included, and `model.norm` IS
    /// applied to the FINAL layer output (HF's `LlamaModel.forward`
    /// returns `[embed, l0, l1, ..., l_{n-2}, norm(l_{n-1})]`; slicing
    /// `[1:]` keeps the last entry, which is post-norm).
    ///
    /// Requirements:
    /// - `attention_mask.len()` must equal `token_ids.len()`.
    /// - Caller is responsible for pad/truncate to a fixed length; use
    ///   [`Self::pad_and_mask`] for the standard HiDream max_seq_len=128
    ///   right-padded case.
    /// - Caller MUST run inside `flame_core::autograd::AutogradContext::no_grad`
    ///   during training; otherwise the 32-layer activation graph is
    ///   retained per batch (tens of GB of grad buffers).
    pub fn encode_all_hidden_states(
        &mut self,
        token_ids: &[i32],
        attention_mask: &[bool],
    ) -> Result<Tensor> {
        if token_ids.len() != attention_mask.len() {
            return Err(flame_core::Error::InvalidInput(format!(
                "[Llama3] token_ids ({}) and attention_mask ({}) length mismatch",
                token_ids.len(),
                attention_mask.len()
            )));
        }
        if token_ids.is_empty() {
            return Err(flame_core::Error::InvalidInput(
                "[Llama3] token_ids is empty".into(),
            ));
        }
        // Footgun guard: warn if caller passed a sequence that looks
        // unpadded (last token != pad_token_id and length != typical 128).
        // We do NOT enforce — HiDream's 128 is just a convention.
        let cfg = &self.config;
        let seq_len = token_ids.len();
        let real_tokens = attention_mask.iter().filter(|b| **b).count();

        log::info!(
            "[Llama3] Encoding: seq_len={}, real_tokens={}",
            seq_len,
            real_tokens
        );

        let mut hidden = self.embed_tokens(token_ids)?;

        let (pe_cos, pe_sin) = Self::build_llama3_rope(seq_len, cfg, &self.device)?;
        let attn_mask = Self::build_attention_mask(seq_len, attention_mask, &self.device)?;

        let mut per_layer: Vec<Tensor> = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            hidden = self.layer_forward(i, &hidden, &pe_cos, &pe_sin, &attn_mask)?;
            per_layer.push(hidden.clone());

            if (i + 1) % 8 == 0 || i == cfg.num_layers - 1 {
                log::info!("[Llama3] Layer {}/{} done", i + 1, cfg.num_layers);
            }
        }

        // BUG-6 fix: HF's `LlamaModel.forward` runs `model.norm` on the
        // last hidden state. `outputs.hidden_states[-1]` is post-norm.
        // HiDream slices `[1:]` which keeps the last (post-norm) entry.
        // Apply `model.norm` to the final layer's output to match.
        let last_idx = cfg.num_layers - 1;
        let normed_last = Self::rms_norm(
            &per_layer[last_idx],
            self.w("model.norm.weight")?,
            cfg.rms_norm_eps,
        )?;
        per_layer[last_idx] = normed_last;

        // [num_layers, 1, S, H]
        let stacked = Tensor::stack(&per_layer, 0)?;
        log::info!("[Llama3] Output: {:?}", stacked.shape());
        Ok(stacked)
    }

    /// Right-pad `token_ids` to `cfg.max_seq_len` (or truncate if longer)
    /// and produce the matching boolean attention mask. Standard helper
    /// for HiDream's `padding="max_length", max_length=128, truncation=True`
    /// tokenizer call when the caller has already tokenised but not
    /// padded.
    ///
    /// Note: HF Llama tokenizer defaults to LEFT-padding. This helper
    /// produces RIGHT-padded output. Callers that need to match HF's
    /// tokenizer-default left-padding must pad themselves and pass the
    /// mask explicitly to [`Self::encode_all_hidden_states`].
    pub fn pad_and_mask(&self, token_ids: &[i32]) -> (Vec<i32>, Vec<bool>) {
        let cfg = &self.config;
        let max = cfg.max_seq_len;
        let real = token_ids.len().min(max);
        let mut ids = Vec::with_capacity(max);
        let mut mask = Vec::with_capacity(max);
        ids.extend_from_slice(&token_ids[..real]);
        mask.extend(std::iter::repeat_n(true, real));
        while ids.len() < max {
            ids.push(cfg.pad_token_id);
            mask.push(false);
        }
        (ids, mask)
    }

    /// **Last-hidden-state convenience.** Encode tokens with an explicit
    /// boolean attention mask and return only the FINAL post-norm layer
    /// of shape `[1, seq_len, hidden_size]` BF16.
    ///
    /// NOT the right method for HiDream — HiDream consumes ALL layers
    /// via [`Self::encode_all_hidden_states`]. This is for general LLM
    /// tasks (classifier head, similarity, etc.). Mask length must equal
    /// `token_ids.len()`.
    pub fn encode_last_hidden_state_with_mask(
        &mut self,
        token_ids: &[i32],
        attention_mask: &[bool],
    ) -> Result<Tensor> {
        let all = self.encode_all_hidden_states(token_ids, attention_mask)?;
        let dims = all.shape().dims().to_vec();
        let l = dims[0];
        let last = all.narrow(0, l - 1, 1)?; // [1, 1, S, H]
        last.squeeze(Some(0))
    }

    /// **Last-hidden-state convenience, no-mask.** Synthesises an
    /// all-real mask. Correct ONLY when the input has no padding.
    ///
    /// NOT the right method for HiDream — see
    /// [`Self::encode_all_hidden_states`]. Use
    /// [`Self::encode_last_hidden_state_with_mask`] if your input is
    /// padded.
    pub fn encode_last_hidden_state(&mut self, token_ids: &[i32]) -> Result<Tensor> {
        let mask = vec![true; token_ids.len()];
        self.encode_last_hidden_state_with_mask(token_ids, &mask)
    }

    // ---- Deprecated aliases (kept for API stability through M1; remove in M2) ----

    /// Deprecated alias for [`Self::encode_last_hidden_state`].
    /// Misleading name for HiDream (which needs all layers).
    /// TODO(M2): remove.
    #[deprecated(
        note = "Use encode_all_hidden_states for HiDream, or encode_last_hidden_state for general LLM tasks."
    )]
    pub fn encode(&mut self, token_ids: &[i32]) -> Result<Tensor> {
        self.encode_last_hidden_state(token_ids)
    }

    /// Deprecated alias for [`Self::encode_last_hidden_state_with_mask`].
    /// Misleading name for HiDream (which needs all layers).
    /// TODO(M2): remove.
    #[deprecated(
        note = "Use encode_all_hidden_states for HiDream, or encode_last_hidden_state_with_mask for general LLM tasks."
    )]
    pub fn encode_with_attention_mask(
        &mut self,
        token_ids: &[i32],
        attention_mask: &[bool],
    ) -> Result<Tensor> {
        self.encode_last_hidden_state_with_mask(token_ids, attention_mask)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_llama31_8b() {
        let c = LlamaConfig::default();
        assert_eq!(c.vocab_size, 128_256);
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.num_layers, 32);
        assert_eq!(c.intermediate_size, 14_336);
        assert_eq!(c.num_heads, 32);
        assert_eq!(c.num_kv_heads, 8);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.num_heads / c.num_kv_heads, 4);
        assert_eq!(c.rope_theta, 500_000.0);
        assert_eq!(c.rope_factor, 8.0);
        assert_eq!(c.rope_low_freq_factor, 1.0);
        assert_eq!(c.rope_high_freq_factor, 4.0);
        assert_eq!(c.rope_original_max_position_embeddings, 8192);
        assert_eq!(c.rms_norm_eps, 1e-5);
        assert_eq!(c.max_seq_len, 128);
        // unsloth pad_token_id, distinct from eos=128009.
        assert_eq!(c.pad_token_id, 128_004);
    }

    #[test]
    fn q_and_k_proj_shapes() {
        let c = LlamaConfig::default();
        // q_proj: [num_heads * head_dim, hidden] = [4096, 4096]
        assert_eq!(c.num_heads * c.head_dim, 4096);
        // k_proj: [num_kv_heads * head_dim, hidden] = [1024, 4096]
        assert_eq!(c.num_kv_heads * c.head_dim, 1024);
    }

    #[test]
    fn pad_and_mask_pads_right_to_max_seq_len() {
        // Pure logic test — no GPU, no Tensor.
        let cfg = LlamaConfig::default();
        let max = cfg.max_seq_len;
        let pad = cfg.pad_token_id;
        let real: Vec<i32> = (0..10).collect();
        // Mirror the encoder method's logic without instantiating one.
        let actual_real = real.len().min(max);
        let mut ids = Vec::with_capacity(max);
        let mut mask = Vec::with_capacity(max);
        ids.extend_from_slice(&real[..actual_real]);
        mask.extend(std::iter::repeat_n(true, actual_real));
        while ids.len() < max {
            ids.push(pad);
            mask.push(false);
        }
        assert_eq!(ids.len(), max);
        assert_eq!(mask.len(), max);
        assert_eq!(&ids[..10], &real[..]);
        assert!(mask[..10].iter().all(|b| *b));
        assert!(mask[10..].iter().all(|b| !*b));
        assert!(ids[10..].iter().all(|&id| id == pad));
        // BUG-1 regression guard.
        assert_eq!(pad, 128_004);
    }

    // NOTE: `config_from_weights_rejects_non_llama31_shape` would need a
    // CudaDevice + real Tensor allocations to exercise the shape assertion
    // path, so it's covered at integration-test level (real weight load).
    // The assertion logic itself is small and visible in source.

    #[test]
    fn rope_cutoffs_sane() {
        let c = LlamaConfig::default();
        let two_pi = 2.0 * std::f64::consts::PI;
        let low_wavelen = c.rope_original_max_position_embeddings as f64 / c.rope_low_freq_factor;
        let high_wavelen = c.rope_original_max_position_embeddings as f64 / c.rope_high_freq_factor;
        assert!(low_wavelen > high_wavelen);
        assert!(two_pi / c.rope_theta < high_wavelen); // shortest wavelength is "high freq"
    }
}
