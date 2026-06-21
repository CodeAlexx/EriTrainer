//! train_chroma — Chroma (Lodestone Rock's FLUX-derived DiT) LoRA training.
//!
//! Chroma1-HD is a de-distilled FLUX.1 derivative: same VAE (16-ch LDM), same
//! T5-XXL text encoder, similar dual-stream (19) + single-stream (38) DiT
//! topology. Differences from Flux that matter for training:
//!   - **No CLIP-pool branch.** Modulation is produced by the
//!     `distilled_guidance_layer` ("approximator") MLP fed only the timestep.
//!     No `vector_in`, no clip pool, no guidance value injection.
//!   - **NOT distilled** — real CFG at sample time (`sample_chroma` does 2
//!     forwards per step). At training time we use unconditional/conditional
//!     pairs the same way Flux training does (caption_dropout via
//!     `--null-text-cache`).
//!   - **Forward signature**: `model.forward(latent_nchw, t5_embed, timestep)`
//!     — chroma's training model handles patchify + RoPE + unpatchify
//!     internally. No `pack_latents` / `build_img_ids` at training time.
//!
//! Cached sample format (produced by `prepare_chroma`):
//!   - `latent`:   [1, 16, H/8, W/8] BF16 — RAW Flux VAE posterior
//!   - `t5_embed`: [1, T5, 4096]    BF16 — T5-XXL hidden states
//!
//! Latent shift/scale is applied here (trainer side) — matches `train_flux`
//! and the EDv2 H3 audit fix. The archive trainer used pre-scaled caches; we
//! deliberately diverged at the prepare step.
//!
//! Modern feature surface (mirrors train_flux.rs Phase 0+):
//!   - EMA shadow + `--ema-validation-swap`
//!   - Multi-resolution noise (default off → byte-invariant)
//!   - Timestep bias (default `none` → byte-invariant)
//!   - Caption dropout via `--null-text-cache` (T5-only; no clip-pool swap)
//!   - Validation harness (held-out cache + side-RNG)
//!   - min-SNR loss weighting (sigma form, FM)
//!   - Combined MSE + MAE + Huber loss
//!   - LR scheduler family + warmup + cycles + lr_min_factor
//!   - Optimizer family CLI (Phase 1 fallback to AdamW)
//!
//! Quant rule: BF16/F32 throughout. NO FP8, NO AdamW8bit. Matches existing
//! FLUX-family trainers (Klein/Flux).

use clap::Parser;
use eridiffusion_cli::{trainer_common, trainer_pipeline};
use eridiffusion_core::config::LrScheduler;
use eridiffusion_core::encoders::flux_vae::{SCALE, SHIFT};
use eridiffusion_core::encoders::t5_xxl::T5Encoder;
use eridiffusion_core::lycoris::{LoraInitType, LycorisAlgo, LycorisBundleConfig};
use eridiffusion_core::models::chroma::ChromaLoraBundle;
use eridiffusion_core::models::ChromaTrainingModel;
use eridiffusion_core::sampler::chroma_sampler;
use eridiffusion_core::training::checkpoint;
use eridiffusion_core::training::ema::ParameterEma;
use eridiffusion_core::training::features::ema_advanced::EmaConfig;
use eridiffusion_core::training::features::validation::ValidationLoop;
use eridiffusion_core::training::features::{
    caption_dropout, loss_weight, lr_schedule, noise_modifiers, sample_library::SampleLibrary,
    timestep_bias,
};
use eridiffusion_core::training::training_features::timestep_dist::TimestepConfig;
use eridiffusion_core::training::training_features::{Optimizer, OptimizerKind};
use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};
use std::path::PathBuf;

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const SEED: u64 = 42;
const CLIP_GRAD_NORM: f32 = 1.0;
// Matches sample_chroma + production chroma_gen.
const T5_SEQ_LEN: usize = 512;

fn tokenize_t5(
    tokenizer_path: &std::path::Path,
    prompt: &str,
    seq_len: usize,
) -> anyhow::Result<Vec<i32>> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("T5 tokenizer load: {e}"))?;
    let enc = tok
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("T5 tokenize: {e}"))?;
    let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
    ids.resize(seq_len, 0);
    Ok(ids)
}

fn build_joint_attention_mask(
    text_mask: &Tensor,
    n_img: usize,
    target_n_txt: usize,
) -> anyhow::Result<Option<Tensor>> {
    let dims = text_mask.shape().dims().to_vec();
    let (batch, mask_n_txt) = match dims.as_slice() {
        [n] => (1usize, *n),
        [b, n] => (*b, *n),
        other => anyhow::bail!("t5_attention_mask must be [T] or [B,T], got {:?}", other),
    };
    let values = text_mask.to_dtype(DType::F32)?.to_vec()?;
    let mut effective_values = vec![1.0f32; batch * target_n_txt];
    for b in 0..batch {
        let src_base = b * mask_n_txt;
        let dst_base = b * target_n_txt;
        let copy_len = mask_n_txt.min(target_n_txt);
        effective_values[dst_base..dst_base + copy_len]
            .copy_from_slice(&values[src_base..src_base + copy_len]);
    }

    if effective_values.iter().all(|v| *v > 0.5) {
        return Ok(None);
    }

    let total = target_n_txt + n_img;
    let mut data = vec![1.0f32; batch * total * total];
    for b in 0..batch {
        let mask_base = b * target_n_txt;
        let batch_base = b * total * total;
        for q in 0..total {
            let row = batch_base + q * total;
            for k in 0..target_n_txt {
                if effective_values[mask_base + k] <= 0.5 {
                    data[row + k] = 0.0;
                }
            }
        }
    }

    Ok(Some(Tensor::from_f32_to_bf16(
        data,
        Shape::from_dims(&[batch, 1, total, total]),
        text_mask.device().clone(),
    )?))
}

#[derive(Parser)]
struct Args {
    /// OT-format JSON config (optional). Falls back to TrainConfig::default().
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    cache_dir: PathBuf,
    /// Chroma transformer dir (containing `*.safetensors` shards) or single file.
    #[arg(long)]
    transformer: PathBuf,
    /// Training mode: `lora` (default) or `full`.
    #[arg(long, default_value = "lora")]
    mode: String,
    #[arg(long, default_value = "100")]
    steps: usize,
    #[arg(long, default_value = "16")]
    rank: usize,
    /// LoRA alpha. Convention: alpha = rank (effective scale = 1.0). Matches
    /// train_flux's H12 fix.
    #[arg(long, default_value = "16.0")]
    lora_alpha: f64,
    /// Default 1e-4. Chroma archive used 1e-4 in `boxjana_lora.toml`.
    #[arg(long, default_value = "1e-4")]
    lr: f32,
    #[arg(long, default_value = "output")]
    output_dir: PathBuf,
    /// Per-block weight streaming via BlockOffloader (required for 24GB VRAM
    /// at 1024² with 57 blocks resident).
    #[arg(long)]
    offload: bool,
    /// Reserved for future activation-offload pool sizing. As of 2026-05-12
    /// the pool is NOT installed for Chroma (see the comment block before
    /// `let params = model.parameters();` for the empirical reason). The
    /// flag is preserved so existing config files / CLI invocations remain
    /// valid when the pool is re-enabled.
    #[arg(long, default_value = "512")]
    offload_resolution: usize,
    /// Save a LoRA checkpoint every N steps (0 = end-only).
    #[arg(long, default_value = "0")]
    save_every: usize,

    // ── Periodic sample (mirrors train_qwenimage pattern) ────────────────
    /// Render a sample image every N steps. 0 disables. Also renders a
    /// step-0 baseline (LoRA=identity).
    #[arg(long, default_value = "0")]
    sample_every: usize,
    #[arg(long)]
    sample_prompt: Option<String>,
    #[arg(long, default_value = "")]
    sample_neg_prompt: String,
    /// Config-driven sample prompt library (JSON). Overrides
    /// config.validation_prompts_file. When set, every prompt is rendered at
    /// each sample checkpoint instead of the single --sample-prompt.
    #[arg(long)]
    validation_prompts_file: Option<PathBuf>,
    /// Flux/Chroma VAE safetensors (decoder loaded inside chroma_sampler).
    #[arg(long)]
    sample_vae: Option<PathBuf>,
    /// T5-XXL encoder weights (single safetensors).
    #[arg(long)]
    sample_t5: Option<PathBuf>,
    /// T5-XXL tokenizer.json.
    #[arg(long)]
    sample_t5_tokenizer: Option<PathBuf>,
    #[arg(long, default_value = "512")]
    sample_size: usize,
    #[arg(long, default_value = "50")]
    sample_steps: usize,
    #[arg(long, default_value = "4.0")]
    sample_cfg: f32,
    #[arg(long, default_value_t = SEED)]
    sample_seed: u64,
    /// Resume LoRA weights only (no optimizer state).
    #[arg(long)]
    resume_lora: Option<PathBuf>,
    /// Resume LoRA weights + AdamW optimizer state + step counter.
    #[arg(long)]
    resume_full: Option<PathBuf>,
    /// Save mode: `weights` (default) writes LoRA-only safetensors. `full`
    /// writes LoRA + AdamW + step in a single safetensors checkpoint.
    #[arg(long, default_value = "weights")]
    save_mode: String,

    // ── Modern feature surface (mirror train_flux.rs Phase 0+) ──
    #[arg(long)]
    min_snr_gamma: Option<f32>,
    #[arg(long, default_value_t = 0.0)]
    caption_dropout_probability: f32,
    /// Path to a single cache file produced by `prepare_chroma` from an empty-
    /// caption sample. When `--caption-dropout-probability > 0`, the trainer
    /// loads `t5_embed` from this file and swaps it in with probability `p`
    /// per step. Chroma has no CLIP-pool branch, so only T5 swaps.
    #[arg(long)]
    null_text_cache: Option<PathBuf>,
    #[arg(long, default_value_t = 1.0)]
    noise_offset_probability: f32,
    #[arg(long, default_value_t = 0.0)]
    gamma_input_perturbation: f32,
    #[arg(long, default_value_t = 0.0)]
    huber_strength: f32,
    #[arg(long, default_value_t = 0.0)]
    lr_min_factor: f32,
    #[arg(long)]
    validation_dataset_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    validation_every_steps: u64,
    #[arg(long, default_value_t = 0.0)]
    masked_loss_weight: f32,

    // EMA
    #[arg(long, default_value_t = false)]
    ema: bool,
    #[arg(long, default_value_t = 1.0)]
    ema_inv_gamma: f32,
    #[arg(long, default_value_t = 0.6667)]
    ema_power: f32,
    #[arg(long, default_value_t = 0)]
    ema_update_after_step: u64,
    #[arg(long, default_value_t = 0.0)]
    ema_min_decay: f32,
    #[arg(long, default_value_t = 0.9999)]
    ema_max_decay: f32,
    #[arg(long, default_value_t = false)]
    ema_validation_swap: bool,

    // Multi-resolution noise (default-off byte invariant)
    #[arg(long, default_value_t = 0)]
    multires_noise_iterations: usize,
    #[arg(long, default_value_t = 0.3)]
    multires_noise_discount: f32,

    // Timestep bias (default-off byte invariant)
    #[arg(long, default_value = "none")]
    timestep_bias_strategy: String,
    #[arg(long, default_value_t = 0.0)]
    timestep_bias_multiplier: f32,
    #[arg(long, default_value_t = 0.0)]
    timestep_bias_range_min: f32,
    #[arg(long, default_value_t = 1.0)]
    timestep_bias_range_max: f32,

    /// Timestep distribution. `uniform` (default — matches Chroma archive's
    /// `let u: f32 = rng.gen()` byte-for-byte after FLUX shift remap),
    /// `logit_normal` (FLUX preset), `sigmoid`, `heavy_tail`, `cos_map`,
    /// `inverted_parabola`.
    #[arg(long, default_value = "uniform")]
    timestep_distribution: String,
    /// Distribution-specific weight knob (default 0.0 — uniform legacy).
    #[arg(long, default_value_t = 0.0)]
    noising_weight: f32,
    /// Distribution-specific bias knob (default 0.0).
    #[arg(long, default_value_t = 0.0)]
    noising_bias: f32,

    /// Optional resolution-shift override. Default `auto` mirrors the chroma
    /// archive's `shift_for_resolution(...)` formula: linear blend 1.0..3.0
    /// across token counts 256..4096.
    #[arg(long, default_value_t = 0.0)]
    timestep_shift: f32,

    /// Phase 1: optimizer family CLI. `adamw` (default) is wired; others
    /// log a warning and fall back to AdamW (full dispatch is a Phase 5 task).
    #[arg(long, default_value = "adamw")]
    optimizer: String,
    /// LR scheduler family: `constant` (default), `linear`, `cosine`,
    /// `cosine_with_restarts`, `polynomial`. Default + `warmup_steps=0` is
    /// byte-equivalent to fixed-LR.
    #[arg(long, default_value = "constant")]
    lr_scheduler: String,
    #[arg(long, default_value_t = 0)]
    warmup_steps: usize,
    #[arg(long, default_value_t = 1.0)]
    lr_cycles: f32,

    // ── LyCORIS algo selection (Phase 2b) ──
    //
    // `--algo lora` (default) keeps the legacy LoRALinear path — byte-identical
    // training to pre-Phase-2b. Other values select LyCORIS algos via
    // `LycorisBundleConfig::new_with_config`. `lora_alpha` and `rank` are
    // shared with the legacy CLI flags above (no separate `--lycoris-rank`).
    /// LyCORIS algo: `lora` (default, legacy path) | `locon` | `loha` | `lokr`
    /// | `full` | `oft`. `full` and `oft` build successfully but their
    /// `forward_delta` will error inside the chroma forward pass —
    /// chroma's `base + delta_on_input` call pattern is incompatible with
    /// Full/OFT semantics. Phase 2c will wire a `merge_into_base` path.
    #[arg(long, default_value = "lora")]
    algo: String,
    /// LoKr Kronecker split factor (ignored for non-LoKr).
    #[arg(long, default_value_t = 16)]
    lokr_factor: i32,
    /// OFT block size (ignored for non-OFT).
    #[arg(long, default_value_t = 32)]
    oft_block_size: usize,
    /// OFT Cayley-Neumann series term count (ignored for non-OFT).
    #[arg(long, default_value_t = 5)]
    oft_neumann_terms: usize,
    /// LoCon / LoHa / LoKr conv variant — Tucker decomposition for non-1×1
    /// kernels. Chroma is linear-only so this is currently a no-op.
    #[arg(long, default_value_t = false)]
    use_tucker: bool,
    /// LoKr only: factorize both W1 *and* W2 (default false: only W2).
    #[arg(long, default_value_t = false)]
    decompose_both: bool,
    /// Enable DoRA (weight-decomposed LoRA). Applies to LoCon/LoHa/LoKr
    /// (Full inherits, OFT errors).
    ///
    /// Phase 2b limitation: chroma's bundle ctor doesn't have access to the
    /// streamed block weights at construction time, so DoRA's magnitude is
    /// initialized from `||I||_2 = 1` rather than `||W_orig||_2`. The
    /// trainer should still converge but will spend the first few hundred
    /// steps adjusting the magnitude. Phase 2c will wire pre-load
    /// magnitude init.
    #[arg(long, default_value_t = false)]
    dora: bool,
    /// DoRA magnitude axis. Default `true` matches lycoris-upstream
    /// (norm over input dims, magnitude shape `[out, 1]`).
    #[arg(long, default_value_t = true)]
    dora_wd_on_out: bool,
    #[arg(long, default_value_t = 1e-6)]
    dora_eps: f32,
    /// PEFT/SimpleTuner `--lora_init_type`. Applies to LoCon (the LoRA path)
    /// only. Choices: `default | gaussian | pissa | olora | loftq`. The
    /// PISSA/OLoRA/LoftQ variants parse but error at adapter construction
    /// because flame-core does not yet expose SVD/QR.
    #[arg(long, default_value = "default")]
    lora_init_type: String,
    /// SimpleTuner-style `lycoris_config preset.json`. Optional; when set,
    /// the file's top-level fields overlay the bundle defaults and its
    /// `apply_preset.module_algo_map` overrides apply per-target during
    /// adapter construction. See `LycorisConfigFile` doc for schema.
    #[arg(long)]
    lycoris_config: Option<PathBuf>,
    /// SimpleTuner-parity: perturbed-normal LoKr init.  Scale `>0`
    /// triggers `lokr_w1=1, lokr_w2 ~ N(μ_W, σ_W) · scale`.  No-op when
    /// algo != lokr or value is 0.0.  Chroma streams base weights via
    /// BlockOffloader; the helper warns when a slot's base is not
    /// resident at swap time.
    #[arg(long, default_value_t = 0.0)]
    init_lokr_norm: f32,
    /// SimpleTuner / edv2-reference `network.conv` — per-LyCORIS rank for
    /// CONV-layer targets (separate from linear `--rank`). `0` (default)
    /// = fall back to linear rank. Inert when no conv targets are wired
    /// in the model bundle (current state on all EDv2 trainers).
    #[arg(long, default_value_t = 0)]
    conv_rank: usize,
    /// SimpleTuner / edv2-reference `network.conv_alpha` — alpha for CONV
    /// targets. `0.0` (default) = fall back to linear `--lora-alpha`.
    #[arg(long, default_value_t = 0.0)]
    conv_alpha: f32,

    // ── Phase 5b: autograd v2 bridge opt-in ────────────────────────────────
    /// Route the backward pass through `AutogradContext::backward_v2`
    /// (`MatchInsertedDtype` policy → BF16 grads end-to-end). Default OFF
    /// preserves v3 byte-equivalence. See train_zimage.rs:269 for full doc.
    #[arg(long, default_value_t = false)]
    use_autograd_v2: bool,

    /// Opt OUT of autograd v2 and run the legacy v3 engine. v2 is the default
    /// as of 2026-05-30 (gate-on Stage 6a); v3 kept as the reference engine.
    /// `--use-autograd-v2` remains accepted as a back-compat no-op.
    #[arg(long, default_value_t = false, conflicts_with = "use_autograd_v2")]
    use_autograd_v3: bool,
}

/// Resolution-dependent timestep shift (matches the chroma archive's
/// `shift_for_resolution` and FLUX/Klein convention).
fn shift_for_resolution(h_lat: usize, w_lat: usize) -> f32 {
    let tokens = (h_lat / 2) * (w_lat / 2);
    let t = ((tokens as f32 - 256.0) / (4096.0 - 256.0)).clamp(0.0, 1.0);
    1.0 + t * (3.0 - 1.0)
}

/// Build a `TimestepConfig` from CLI args. Used to dispatch all 6 OT
/// distributions through the unified sampler.
///
/// **Byte-equivalence note:** the chroma archive's `logit_normal` and
/// `sigmoid` arms previously used `rand_distr::Normal` (Ziggurat) for the
/// underlying gaussian. The unified `TimestepConfig::sample_one` uses
/// Box-Muller polar form. With the same seed the two consume the same
/// number of `r#gen::<f32>()` draws but produce different normal samples,
/// so resumed runs that pinned a Ziggurat-derived sequence will diverge.
/// `Uniform` is bit-identical (no normal involved). The default
/// (`uniform, 0, 0`) is preserved exactly.
fn build_timestep_config(
    distribution: &str,
    weight: f32,
    bias: f32,
) -> anyhow::Result<TimestepConfig> {
    trainer_common::build_full_strength_timestep_config(distribution, weight, bias)
}

fn main() -> anyhow::Result<()> {
    use rand::SeedableRng;
    trainer_common::init_logging();
    let args = Args::parse();
    trainer_common::ensure_output_dir(&args.output_dir)?;

    let device = trainer_common::init_bf16_cuda();

    let mut config = trainer_common::load_train_config_or_default(args.config.as_deref())?;
    trainer_common::apply_lora_basics(&mut config, args.rank, args.lora_alpha, args.lr);

    // Plumb CLI args into config (default-off, modern feature rollout).
    config.min_snr_gamma = args.min_snr_gamma;
    config.caption_dropout_probability = args.caption_dropout_probability;
    config.noise_offset_probability = args.noise_offset_probability;
    config.gamma_input_perturbation = args.gamma_input_perturbation;
    config.huber_strength = args.huber_strength;
    config.lr_min_factor = args.lr_min_factor;
    config.validation_dataset_dir = args.validation_dataset_dir.clone();
    config.validation_every_steps = args.validation_every_steps;
    config.validation_prompts_file = args.validation_prompts_file.clone();
    config.masked_loss_weight = args.masked_loss_weight;
    if args.masked_loss_weight > 0.0 {
        log::warn!(
            "[masked-loss] --masked-loss-weight={:.3} requested but Chroma's prepare_chroma cache schema has no `latent_mask` field; flag is a no-op for this trainer.",
            args.masked_loss_weight
        );
    }
    config.ema_inv_gamma = args.ema_inv_gamma;
    config.ema_power = args.ema_power;
    config.ema_update_after_step = args.ema_update_after_step;
    config.ema_min_decay = args.ema_min_decay;

    log::info!(
        "[Chroma] loading transformer from {} (mode={}, rank={}, alpha={}, offload={})",
        args.transformer.display(),
        args.mode,
        args.rank,
        args.lora_alpha,
        args.offload,
    );

    // Phase 2b: parse the LyCORIS algo selector. `lora` (default) keeps the
    // legacy LoRALinear bundle constructed inside `ChromaTrainingModel::load`.
    // Anything else swaps the bundle in-place after model construction so we
    // don't have to re-plumb the per-trainer constructor signatures.
    //
    // NOTE: `LycorisAlgo::parse("lora")` aliases to `LycorisAlgo::LoCon`
    // (since LoCon-Linear is the canonical LoRA decomposition). For chroma
    // we need to distinguish LEGACY plain `LoRALinear` (byte-identical) from
    // the new `LycorisAdapter::LoCon` path, so re-map `"lora"` → `None` here
    // explicitly. Users who want the new LoCon path pass `--algo locon`.
    let algo_str = args.algo.trim().to_ascii_lowercase();
    let algo = if algo_str == "lora" || algo_str == "none" || algo_str.is_empty() {
        LycorisAlgo::None
    } else {
        LycorisAlgo::parse(&args.algo).map_err(|e| anyhow::anyhow!("--algo: {e}"))?
    };
    // Default storage (F32) inherited from `LycorisBundleConfig::default()`.
    // This matches the trainer-side AdamW state requirement; do NOT switch
    // to BF16/FP8 — chroma trainer is BF16/F32-only (see top-of-file rule).
    let lyc_config = LycorisBundleConfig {
        algo,
        rank: args.rank,
        alpha: args.lora_alpha as f32,
        factor: args.lokr_factor,
        conv_rank: args.conv_rank,
        conv_alpha: args.conv_alpha,
        block_size: args.oft_block_size,
        neumann_terms: args.oft_neumann_terms,
        use_tucker: args.use_tucker,
        decompose_both: args.decompose_both,
        use_scalar: false,
        dora: args.dora,
        dora_wd_on_out: args.dora_wd_on_out,
        dora_eps: args.dora_eps,
        init_type: LoraInitType::parse(&args.lora_init_type)
            .map_err(|e| anyhow::anyhow!("--lora_init_type: {e}"))?,
        ..LycorisBundleConfig::default()
    };
    let lyc_config =
        lyc_config.with_optional_lycoris_config_file(args.lycoris_config.as_deref())?;

    // Sample setup BEFORE DiT load: encode prompt + uncond with T5, drop
    // encoder. T5-XXL ~10 GB + Chroma DiT ~17 GB cannot coexist on 24 GB
    // VRAM; mirrors qwen/sample_chroma load order.
    let periodic_sample = args.sample_every > 0;
    // Config-driven sample set. Prompts come from --validation-prompts-file
    // (CLI) OR config.validation_prompts_file, NOT hardcoded. Falls back to the
    // single --sample-prompt only if no prompts file is given. Each entry:
    // (label "p{i+1}", encoded cond, encoded uncond).
    let prompts_file = args
        .validation_prompts_file
        .clone()
        .or_else(|| config.validation_prompts_file.clone());
    let mut sample_set: Vec<(String, Tensor, Tensor)> = Vec::new();
    // Prompt texts kept alive parallel to `sample_set` for board log_text.
    let mut sample_prompt_texts: Vec<String> = Vec::new();
    let sample_vae_path = if periodic_sample {
        let vae_path = args
            .sample_vae
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every>0 requires --sample-vae"))?;
        let t5_path = args
            .sample_t5
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every>0 requires --sample-t5"))?;
        let t5_tok = args
            .sample_t5_tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every>0 requires --sample-t5-tokenizer"))?;
        // Resolve the prompt list: prefer the config/CLI prompts file.
        let prompt_list: Vec<(String, String, String)> = if let Some(ref pf) = prompts_file {
            let lib = SampleLibrary::from_file(pf)?;
            log::info!(
                "[sample-setup] {} prompt(s) from {}",
                lib.len(),
                pf.display()
            );
            lib.prompts
                .iter()
                .enumerate()
                .map(|(i, p)| (format!("p{}", i + 1), p.prompt.clone(), p.negative.clone()))
                .collect()
        } else {
            let prompt = args.sample_prompt.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "--sample-every>0 requires --validation-prompts-file or --sample-prompt"
                )
            })?;
            vec![("p1".into(), prompt.clone(), args.sample_neg_prompt.clone())]
        };
        log::info!("[sample-setup] loading T5-XXL for prompt pre-encode...");
        let t5_path_str = t5_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("--sample-t5 not valid UTF-8"))?;
        let mut t5 =
            T5Encoder::load(t5_path_str, &device).map_err(|e| anyhow::anyhow!("T5 load: {e}"))?;
        for (label, ptext, ntext) in &prompt_list {
            let cond_tokens = tokenize_t5(t5_tok, ptext, T5_SEQ_LEN)?;
            let cond = t5
                .encode(&cond_tokens)
                .map_err(|e| anyhow::anyhow!("T5 cond encode: {e}"))?
                .to_dtype(DType::BF16)?;
            let uncond_tokens = tokenize_t5(t5_tok, ntext, T5_SEQ_LEN)?;
            let uncond = t5
                .encode(&uncond_tokens)
                .map_err(|e| anyhow::anyhow!("T5 uncond encode: {e}"))?
                .to_dtype(DType::BF16)?;
            log::info!(
                "[sample-setup] {label} cond={:?} uncond={:?}",
                cond.shape().dims(),
                uncond.shape().dims()
            );
            sample_set.push((label.clone(), cond, uncond));
            sample_prompt_texts.push(ptext.clone());
        }
        drop(t5);
        flame_core::cuda_alloc_pool::clear_pool_cache();
        flame_core::trim_cuda_mempool(0);
        log::info!(
            "[sample-setup] {} prompt(s) ready; periodic sample enabled (every {} steps).",
            sample_set.len(),
            args.sample_every
        );
        Some(vae_path.clone())
    } else {
        None
    };

    let mut model = if args.offload {
        ChromaTrainingModel::load_swapped(
            &args.transformer,
            &args.mode,
            args.rank,
            args.lora_alpha as f32,
            device.clone(),
            SEED,
        )?
    } else {
        ChromaTrainingModel::load(
            &args.transformer,
            &args.mode,
            args.rank,
            args.lora_alpha as f32,
            device.clone(),
            SEED,
        )?
    };

    // If a LyCORIS algo other than the legacy plain LoRA was requested, swap
    // the bundle. Plain `--algo lora` (or `lora`/`none`) keeps the legacy
    // bundle as-is so this branch is byte-equivalent to the pre-Phase-2b
    // pipeline.
    if algo != LycorisAlgo::None && args.mode == "lora" {
        log::info!(
            "[Chroma] LyCORIS algo='{}' rank={} alpha={} factor={} block_size={} dora={}",
            algo.as_str(),
            lyc_config.rank,
            lyc_config.alpha,
            lyc_config.factor,
            lyc_config.block_size,
            lyc_config.dora,
        );
        if matches!(algo, LycorisAlgo::Full | LycorisAlgo::Oft) {
            log::warn!(
                "[Chroma] algo='{}' selected — bundle construction will succeed, but \
                 forward_delta will error inside chroma's `base + delta_on_input` call \
                 pattern. Phase 2c will wire merge-into-base for these algos.",
                algo.as_str()
            );
        }
        let new_bundle = ChromaLoraBundle::new_with_config(&lyc_config, device.clone(), SEED)
            .map_err(|e| anyhow::anyhow!("LyCORIS bundle construction: {e}"))?;
        // SimpleTuner-parity perturbed-normal LoKr init.  Walks adapters
        // and dispatches `init_perturbed_normal_lokr` per-adapter via the
        // AdapterModule trait.  The base-weight lookup uses whatever the
        // model has resident — chroma's BlockOffloader streams blocks at
        // runtime, so missing slots get logged warnings rather than
        // failing.
        if matches!(algo, LycorisAlgo::LoKr) && args.init_lokr_norm > 0.0 {
            let skipped = new_bundle
                .apply_init_perturbed_normal(model.resident_weights(), args.init_lokr_norm)
                .map_err(|e| anyhow::anyhow!("init_lokr_norm: {e}"))?;
            log::info!(
                "[Chroma] --init_lokr_norm={} applied (skipped={} non-LoKr/missing/factored)",
                args.init_lokr_norm,
                skipped,
            );
        }
        model.bundle = Some(new_bundle);
    } else if algo == LycorisAlgo::None {
        // Explicit log: legacy path — no swap.
        log::info!("[Chroma] algo='lora' (legacy LoRALinear path, byte-identical)");
    }

    // ── Activation offload pool ─────────────────────────────────────────────
    // INTENTIONALLY NOT INSTALLED for Chroma as of 2026-05-12 — same
    // empirical reason as Z-Image (Stage 1, plan keen-crafting-jellyfish).
    // Chroma's per-block sub-tape (57 blocks × ~7 BF16-grad saves per
    // tape entry × many entries) consistently exhausts any slot budget
    // that fits in 62 GB host RAM, triggering partial offload + recompute
    // fallback. The fallback path produces NaN loss starting at step 2
    // (verified 5-step smoke: 944 slots / 16 per-block, raw BF16 = 42 GB
    // pinned, exhausted after 138 entries, NaN from step 2 onward,
    // 10–15 s/step vs ~5–7 s/step baseline).
    //
    // The `checkpoint_offload` call sites at models/chroma.rs:1352 and
    // :1397 are PRESERVED — they fall back to `checkpoint()` (recompute)
    // when no pool is installed (autograd.rs:1876-1879), giving the
    // existing baseline behaviour byte-equivalent.
    //
    // To revisit when flame-core gains per-block slot reservation +
    // eviction OR when the partial-offload NaN propagation bug is fixed.
    let _ = args.offload_resolution; // arg accepted for forward-compatibility

    let mut params = model.parameters();
    // Gate-on 6a: under v2 (default), flip LoRA params to MatchParamDtype so
    // BF16 grads from the bridge stay BF16 (Class A). --use-autograd-v3 skips.
    trainer_pipeline::apply_autograd_v2_grad_policy(&mut params, args.use_autograd_v3, "params");
    log::info!("[Chroma] {} trainable tensors", params.len());
    if params.is_empty() {
        anyhow::bail!("No trainable parameters — check mode={}", args.mode);
    }

    // OT preset optimizer: AdamW(β=(0.9, 0.999), ε=1e-8, wd=0.01).
    // Phase B (2026-05-10): unified Optimizer enum dispatches all kinds.
    let opt_kind =
        OptimizerKind::parse(&args.optimizer).map_err(|e| anyhow::anyhow!("--optimizer: {e}"))?;
    log::info!("[Chroma] optimizer={}", opt_kind.as_str());
    let mut opt = Optimizer::new(opt_kind, args.lr, 0.9, 0.999, 1e-8, 0.01);

    if args.resume_lora.is_some() && args.resume_full.is_some() {
        anyhow::bail!("Use either --resume-lora or --resume-full, not both");
    }

    let save_mode_full = match args.save_mode.as_str() {
        "full" => {
            if model.bundle.is_none() {
                anyhow::bail!("--save-mode full currently requires --mode lora");
            }
            true
        }
        "weights" => false,
        other => anyhow::bail!("--save-mode must be `weights` or `full`, got `{other}`"),
    };

    let mut start_step: usize = 0;
    if let Some(resume_path) = args.resume_lora.as_ref() {
        log::info!("Resuming LoRA weights from {}", resume_path.display());
        // `ChromaLoraBundle::load_weights` mutates adapters in-place via
        // Parameter::set_data, so the optimizer parameter list keeps the same
        // Parameter IDs.
        match model.bundle.as_ref() {
            Some(bundle) => {
                bundle
                    .load_weights(resume_path, device.clone())
                    .map_err(|e| anyhow::anyhow!("--resume-lora load: {e}"))?;
                log::info!("Resumed {} LoRA adapters", bundle.num_adapters());
            }
            None => anyhow::bail!("--resume-lora requires LoRA mode (--mode lora), not full"),
        }
    }
    if let Some(resume_path) = args.resume_full.as_ref() {
        log::info!("Full-resume from {}", resume_path.display());
        let loaded = checkpoint::load_full(resume_path, &device)?;
        let Some(bundle) = model.bundle.as_ref() else {
            anyhow::bail!("--resume-full requires LoRA mode (--mode lora)");
        };
        let named = bundle.named_parameters();
        checkpoint::apply_lora_weights(&loaded, &named)?;
        if let Optimizer::AdamW(ref mut adam) = opt {
            checkpoint::apply_to_optimizer(
                &loaded,
                adam,
                &named,
                args.rank,
                args.lora_alpha as f32,
            )?;
        } else {
            log::warn!(
                "[resume-full] optimizer state restore is only wired for AdamW; {:?} will resume weights with fresh optimizer state",
                opt.kind()
            );
        }
        start_step = loaded.header.step as usize;
        if start_step >= args.steps {
            log::warn!(
                "Resumed step ({start_step}) >= --steps ({}) — nothing to do.",
                args.steps
            );
            return Ok(());
        }
    }

    // EMA shadow. Build after resume so the shadow matches restored weights.
    let ema_cfg = EmaConfig {
        inv_gamma: args.ema_inv_gamma,
        power: args.ema_power,
        update_after_step: args.ema_update_after_step,
        min_decay: args.ema_min_decay,
        max_decay: args.ema_max_decay,
    };
    let mut ema: Option<ParameterEma> = if args.ema {
        let _g = AutogradContext::no_grad();
        let e = ParameterEma::new(&params, args.ema_max_decay)
            .map_err(|e| anyhow::anyhow!("EMA construction failed: {e}"))?;
        log::info!(
            "[ema] WIRED — {} shadow tensors, swap={}",
            e.len(),
            args.ema_validation_swap
        );
        Some(e)
    } else {
        None
    };

    // Timestep bias config (default-off byte invariance).
    let timestep_bias_cfg = {
        let strategy = timestep_bias::Strategy::parse(&args.timestep_bias_strategy)
            .map_err(|e| anyhow::anyhow!("--timestep-bias-strategy: {e}"))?;
        let cfg = timestep_bias::BiasConfig {
            strategy,
            multiplier: args.timestep_bias_multiplier,
            range_min: args.timestep_bias_range_min,
            range_max: args.timestep_bias_range_max,
        };
        if strategy != timestep_bias::Strategy::None {
            log::info!(
                "[timestep-bias] strategy={} multiplier={} range=[{}, {}]",
                strategy.as_str(),
                cfg.multiplier,
                cfg.range_min,
                cfg.range_max
            );
        }
        cfg
    };

    // Unified OneTrainer timestep distribution dispatch.
    let timestep_cfg = build_timestep_config(
        &args.timestep_distribution,
        args.noising_weight,
        args.noising_bias,
    )?;

    // Caption dropout: chroma has no clip-pool, so we only swap T5.
    let mut effective_caption_dropout_prob = args.caption_dropout_probability;
    let null_t5: Option<Tensor> = if effective_caption_dropout_prob > 0.0 {
        match args.null_text_cache.as_ref() {
            Some(p) => match flame_core::serialization::load_file(p, &device) {
                Ok(s) => {
                    let nt5 = s
                        .get("t5_embed")
                        .ok_or_else(|| anyhow::anyhow!("--null-text-cache missing 't5_embed'"))?
                        .to_dtype(DType::BF16)?;
                    log::info!(
                        "[caption-dropout] WIRED — prob={:.3} (null_t5={:?})",
                        effective_caption_dropout_prob,
                        nt5.shape().dims()
                    );
                    Some(nt5)
                }
                Err(e) => {
                    log::warn!(
                        "[caption-dropout] failed to load --null-text-cache {}: {e} — feature disabled",
                        p.display()
                    );
                    effective_caption_dropout_prob = 0.0;
                    None
                }
            },
            None => {
                log::warn!(
                    "caption_dropout_probability={:.3} requested but --null-text-cache not provided — feature disabled",
                    effective_caption_dropout_prob
                );
                effective_caption_dropout_prob = 0.0;
                None
            }
        }
    } else {
        None
    };

    let cache_files = trainer_common::list_cache_safetensors(&args.cache_dir)?;
    log::info!("[Chroma] {} cached samples", cache_files.len());

    // Seed both flame_core RNG and host RNG from SEED=42 (matches train_flux).
    trainer_common::set_flame_seed(SEED)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

    let board = trainer_common::open_board_writer(&args.output_dir, None);
    if let Some(b) = &board {
        // Run hyper-parameters → metadata.hparams + the dashboard's hparam
        // panel. JSON hand-built (no serde_json dep here).
        let hparams_json = format!(
            "{{\"model\":\"chroma\",\"steps\":{},\"rank\":{},\"lora_alpha\":{},\"lr\":{},\
             \"batch_size\":{},\"optimizer\":\"{}\",\"timestep_shift\":{},\
             \"sample_size\":{},\"sample_steps\":{},\"sample_cfg\":{},\"seed\":{}}}",
            args.steps,
            args.rank,
            args.lora_alpha,
            args.lr,
            1,
            opt_kind.as_str(),
            args.timestep_shift,
            args.sample_size,
            args.sample_steps,
            args.sample_cfg,
            SEED
        );
        b.log_hparams(&hparams_json, &[("steps_target", args.steps as f64)]);
    }
    let loop_config = trainer_pipeline::ManualTrainLoopConfig::new(
        "Chroma-lora",
        start_step,
        args.steps,
        cache_files.len(),
        1,
    );
    let mut loop_run = trainer_pipeline::ManualTrainLoopRun::new(loop_config);

    // Step-0 baseline sample (LoRA=identity) — every config-driven prompt.
    if periodic_sample {
        let vae_path = sample_vae_path.as_ref().unwrap();
        for (idx, (label, cond, uncond)) in sample_set.iter().enumerate() {
            let out_path = args
                .output_dir
                .join(format!("sample_step0_base_{label}.png"));
            log::info!("[sample step=0 {label}] BASELINE → {}", out_path.display());
            if let Err(e) = chroma_sampler::sample_image(
                &model,
                cond,
                uncond,
                args.sample_size,
                args.sample_size,
                args.sample_steps,
                args.sample_cfg,
                args.sample_seed,
                vae_path,
                &out_path,
                &device,
            ) {
                log::warn!("[sample step=0 {label}] failed: {e}");
            } else if let Some(b) = &board {
                b.log_image_png(&format!("samples/{label}"), 0, 0, &out_path);
                if let Some(t) = sample_prompt_texts.get(idx) {
                    b.log_text(&format!("prompts/{label}"), 0, t);
                }
            }
        }
        flame_core::cuda_alloc_pool::clear_pool_cache();
        flame_core::trim_cuda_mempool(0);
    }

    let validation_loop: Option<ValidationLoop> = if let (Some(dir), n) = (
        args.validation_dataset_dir.as_ref(),
        args.validation_every_steps,
    ) {
        if n > 0 {
            let v = ValidationLoop::new(dir, n)?;
            log::info!(
                "[validation] {} held-out samples, every {} steps",
                v.len(),
                n
            );
            Some(v)
        } else {
            None
        }
    } else {
        None
    };

    let sched: LrScheduler = lr_schedule::parse_cli_scheduler(&args.lr_scheduler);

    for step in loop_run.steps() {
        let cache_idx = step % cache_files.len();
        let sample = flame_core::serialization::load_file(&cache_files[cache_idx], &device)?;

        // RAW VAE posterior — apply `(raw - SHIFT) * SCALE` per step (same as
        // train_flux H3 fix). Chroma's DiT was trained on shift/scaled latents.
        let latent_raw = sample
            .get("latent")
            .ok_or_else(|| {
                anyhow::anyhow!("missing 'latent' in {}", cache_files[cache_idx].display())
            })?
            .to_dtype(DType::BF16)?;
        let t5 = sample
            .get("t5_embed")
            .ok_or_else(|| {
                anyhow::anyhow!("missing 't5_embed' in {}", cache_files[cache_idx].display())
            })?
            .to_dtype(DType::BF16)?;
        let t5_attention_mask = if let Some(mask) = sample
            .get("t5_attention_mask")
            .or_else(|| sample.get("t5_mask"))
        {
            Some(mask.to_dtype(DType::BF16)?)
        } else {
            None
        };

        // Caption dropout (T5-only — chroma has no clip pool).
        let t5 = if let Some(ref nt5) = null_t5 {
            caption_dropout::maybe_drop_caption(&t5, nt5, effective_caption_dropout_prob, &mut rng)?
        } else {
            t5
        };

        // VAE shift/scale. (raw - shift) * scale — multiply, not divide.
        let latent = latent_raw
            .add_scalar(-SHIFT)?
            .mul_scalar(SCALE)?
            .to_dtype(DType::BF16)?;

        let lat_dims = latent.shape().dims().to_vec();
        if lat_dims.len() != 4 {
            anyhow::bail!("expected 4D latent [B, C, H, W], got {:?}", lat_dims);
        }
        let (h_lat, w_lat) = (lat_dims[2], lat_dims[3]);
        let n_img = (h_lat / 2) * (w_lat / 2);

        // Resolution-dependent shift (default behaviour). Override via --timestep-shift.
        let shift = if args.timestep_shift > 0.0 {
            args.timestep_shift
        } else {
            shift_for_resolution(h_lat, w_lat)
        };

        // Sample base u and apply FLUX shift remap.
        let u_base = timestep_cfg.sample_one(&mut rng);
        // Default-off: Strategy::None returns u_base unchanged (the bias module
        // is in [0, NUM_TRAIN_TIMESTEPS] units, so we lift to that range and
        // back).
        let u_t = timestep_bias::apply_bias(
            u_base * NUM_TRAIN_TIMESTEPS as f32,
            NUM_TRAIN_TIMESTEPS as f32,
            &timestep_bias_cfg,
        ) / NUM_TRAIN_TIMESTEPS as f32;
        // Apply FLUX shift remap: sigma = shift * u / (1 + (shift - 1) * u).
        let sigma = shift * u_t / (1.0 + (shift - 1.0) * u_t);

        // Clean noise + multires + offset + perturbation. All default-off
        // are byte-invariant.
        let latent_f32 = latent.to_dtype(DType::F32)?;
        let noise = noise_modifiers::randn_f32(latent_f32.shape().clone(), device.clone())?;
        let noise = noise_modifiers::maybe_apply_multires_noise(
            &noise,
            args.multires_noise_iterations,
            args.multires_noise_discount,
            &mut rng,
        )?;
        let clean_noise = noise_modifiers::maybe_apply_offset_noise(
            &noise,
            config.offset_noise_weight as f32,
            args.noise_offset_probability,
            &mut rng,
        )?;
        let perturbed_noise = noise_modifiers::maybe_apply_input_perturbation(
            &clean_noise,
            args.gamma_input_perturbation,
            &mut rng,
        )?;

        // x_t = (1 - sigma) * latent + sigma * noise (FLUX flow-matching).
        let noisy_f32 = latent_f32
            .mul_scalar(1.0 - sigma)?
            .add(&perturbed_noise.mul_scalar(sigma)?)?;
        let noisy = noisy_f32.to_dtype(DType::BF16)?;
        // Rectified-flow target: noise - clean (matches archive `pipeline.rs`).
        let target = clean_noise.sub(&latent_f32)?;

        // Chroma's forward expects sigma directly as the timestep input
        // (not `sigma * 1000`). Matches `pipeline.rs::prepare_inputs` and
        // `sample_chroma`'s denoise loop.
        let timestep = Tensor::from_vec(vec![sigma], Shape::from_dims(&[1]), device.clone())?
            .to_dtype(DType::BF16)?;

        if step == 0 {
            log::info!(
                "step 0 | latent={:?} t5={:?} sigma={:.4} shift={:.2}",
                latent.shape().dims(),
                t5.shape().dims(),
                sigma,
                shift,
            );
        }

        let _profile_step = std::env::var("FLAME_PROFILE_STEP").ok().as_deref() == Some("1");
        let _t_fwd_start = std::time::Instant::now();
        if _profile_step {
            let _ = device.synchronize();
        }
        let attn_mask = if let Some(ref text_mask) = t5_attention_mask {
            build_joint_attention_mask(text_mask, n_img, t5.shape().dims()[1])?
        } else {
            None
        };
        let pred = model.forward_with_attention_mask(&noisy, &t5, &timestep, attn_mask.as_ref())?;
        if _profile_step {
            let _ = device.synchronize();
        }
        let _fwd_ms = _t_fwd_start.elapsed().as_secs_f64() * 1000.0;

        if pred.shape().dims() != target.shape().dims() {
            anyhow::bail!(
                "pred {:?} != target {:?}",
                pred.shape().dims(),
                target.shape().dims()
            );
        }

        // Combined MSE+MAE+Huber + min-SNR weighting (default config keeps
        // mse_strength=1, mae=0, huber=0 → straight MSE; min-snr=None is no-op).
        let pred_f32 = pred.to_dtype(DType::F32)?;
        let target_f32 = target.to_dtype(DType::F32)?;
        let raw_loss = loss_weight::combined_loss(
            &pred_f32,
            &target_f32,
            config.mse_strength as f32,
            config.mae_strength as f32,
            args.huber_strength,
        )?;
        let loss = loss_weight::apply_loss_weight(
            &raw_loss,
            sigma,
            config.loss_weight_fn,
            args.min_snr_gamma,
            true, // FM (sigma form)
        )?;
        let loss_val = loss.to_vec()?[0];

        let _t_bwd_start = std::time::Instant::now();
        if _profile_step {
            let _ = device.synchronize();
        }
        // Phase 5b / gate-on 6a: Route (ii) bridge. v2 is the default; backward
        // goes through `backward_v2` unless `--use-autograd-v3` opts into v3.
        let grads = trainer_pipeline::backward_loss(&loss, args.use_autograd_v3)?;
        if _profile_step {
            let _ = device.synchronize();
        }
        let _bwd_ms = _t_bwd_start.elapsed().as_secs_f64() * 1000.0;
        if _profile_step {
            log::info!("[profile] step {step} | fwd={_fwd_ms:.1}ms | bwd={_bwd_ms:.1}ms");
        }

        // Grad-flow diagnostic.  Runs at step 1 — NOT step 0 — because every
        // LoRA-style algo (LoRA, LoCon, LoHa, LoKr) initializes one factor
        // at zero so `delta = factor_a @ factor_b = 0` at step 0.  Backward
        // through `delta * weight` then forces half the leaves to zero
        // gradient by mathematical construction.  Step 1 (after the first
        // optimizer step has driven the zero leaves off zero) is when the
        // assertion can distinguish "real bug" from "expected zero-init
        // pattern".  See flame-core/docs/TRAINER_DIAGNOSTICS.md.
        if step == 1 {
            if let Some(bundle) = model.bundle.as_ref() {
                let named = bundle.named_parameters();
                let named_refs: Vec<(&str, &flame_core::parameter::Parameter)> =
                    named.iter().map(|(n, p)| (n.as_str(), p)).collect();
                let report = flame_core::diagnostics::assert_grad_flow(&grads, &named_refs)?;
                if report.is_clean() {
                    log::info!("[grad-flow] step 2 clean ({} params)", report.ok_count);
                } else {
                    log::warn!("{}", report.summary());
                }
            }
        }

        // Global L2 grad clip = 1.0 (preset default).
        let clip = trainer_pipeline::apply_gradient_map_clip(
            &params,
            &grads,
            trainer_pipeline::GradientClipOptions::clip_by_norm(CLIP_GRAD_NORM),
        )?;
        let total_norm = clip.total_norm;

        let cur_lr = lr_schedule::dispatch_lr(
            &sched,
            args.lr,
            step,
            args.steps,
            args.warmup_steps,
            args.lr_min_factor,
            args.lr_cycles,
        );
        trainer_pipeline::step_optimizer(&mut opt, &params, cur_lr, || {
            // Refresh chroma's LoRA cache so the LoRA contribution is visible
            // to the next forward (matches archive pattern).
            model.refresh_lora_cache();
            if let Some(ref mut e) = ema {
                e.update_with_schedule(&params, &ema_cfg, (step + 1) as u64)
                    .map_err(|err| {
                        anyhow::anyhow!("EMA update failed at step {}: {err}", step + 1)
                    })?;
            }
            Ok(())
        })?;
        // Flush GPU allocation pool (matches archive — chroma forward + bwd
        // accumulates a lot of intermediates).
        flame_core::cuda_alloc_pool::clear_pool_cache();
        device.synchronize().ok();

        loop_run.record_and_log(
            step,
            trainer_pipeline::TrainStepMetrics {
                loss_value: loss_val,
                grad_norm: total_norm,
                learning_rate: cur_lr,
            },
            board.as_ref(),
        );

        // Validation eval pass (no_grad), side-RNG seeded as `SEED ^ (step+1)`.
        if let Some(ref vloop) = validation_loop {
            if vloop.should_run(step + 1) {
                let mut sum = 0.0_f32;
                let mut count = 0_usize;
                for vfile in &vloop.cache_files {
                    let _g = AutogradContext::no_grad();
                    let sample = match flame_core::serialization::load_file(vfile, &device) {
                        Ok(s) => s,
                        Err(e) => {
                            log::warn!("[validation] load {} failed: {e}", vfile.display());
                            continue;
                        }
                    };
                    let v_latent_raw = match sample.get("latent") {
                        Some(t) => t.to_dtype(DType::BF16)?,
                        None => continue,
                    };
                    let v_t5 = match sample.get("t5_embed") {
                        Some(t) => t.to_dtype(DType::BF16)?,
                        None => continue,
                    };
                    let v_t5_attention_mask = if let Some(mask) = sample
                        .get("t5_attention_mask")
                        .or_else(|| sample.get("t5_mask"))
                    {
                        Some(mask.to_dtype(DType::BF16)?)
                    } else {
                        None
                    };
                    let v_latent = v_latent_raw
                        .add_scalar(-SHIFT)?
                        .mul_scalar(SCALE)?
                        .to_dtype(DType::BF16)?;
                    let v_dims = v_latent.shape().dims().to_vec();
                    let (vh, vw) = (v_dims[2], v_dims[3]);
                    let v_n_img = (vh / 2) * (vw / 2);
                    let v_shift = if args.timestep_shift > 0.0 {
                        args.timestep_shift
                    } else {
                        shift_for_resolution(vh, vw)
                    };
                    let mut vrng = rand::rngs::StdRng::seed_from_u64(SEED ^ (step as u64 + 1));
                    let v_u = timestep_cfg.sample_one(&mut vrng);
                    let v_u_t = timestep_bias::apply_bias(
                        v_u * NUM_TRAIN_TIMESTEPS as f32,
                        NUM_TRAIN_TIMESTEPS as f32,
                        &timestep_bias_cfg,
                    ) / NUM_TRAIN_TIMESTEPS as f32;
                    let v_sigma = v_shift * v_u_t / (1.0 + (v_shift - 1.0) * v_u_t);
                    let v_noise =
                        Tensor::randn(v_latent.shape().clone(), 0.0, 1.0, device.clone())?
                            .to_dtype(DType::BF16)?;
                    let v_noisy = v_latent
                        .mul_scalar(1.0 - v_sigma)?
                        .add(&v_noise.mul_scalar(v_sigma)?)?;
                    let v_target = v_noise.sub(&v_latent)?;
                    let v_timestep =
                        Tensor::from_vec(vec![v_sigma], Shape::from_dims(&[1]), device.clone())?
                            .to_dtype(DType::BF16)?;
                    let v_attn_mask = if let Some(ref text_mask) = v_t5_attention_mask {
                        build_joint_attention_mask(text_mask, v_n_img, v_t5.shape().dims()[1])?
                    } else {
                        None
                    };
                    let v_pred = match model.forward_with_attention_mask(
                        &v_noisy,
                        &v_t5,
                        &v_timestep,
                        v_attn_mask.as_ref(),
                    ) {
                        Ok(p) => p,
                        Err(e) => {
                            log::warn!("[validation] forward failed: {e}");
                            continue;
                        }
                    };
                    let v_loss = v_pred
                        .to_dtype(DType::F32)?
                        .sub(&v_target.to_dtype(DType::F32)?)?
                        .square()?
                        .mean()?;
                    let v_loss_val = v_loss.to_vec()?[0];
                    if v_loss_val.is_finite() {
                        sum += v_loss_val;
                        count += 1;
                    }
                    AutogradContext::clear();
                }
                if count > 0 {
                    let val_avg = sum / count as f32;
                    log::info!(
                        "[validation step={}] loss/val = {:.4} ({} samples)",
                        step + 1,
                        val_avg,
                        count
                    );
                    if let Some(b) = &board {
                        b.log_scalar("loss/val", (step + 1) as u64, val_avg as f64);
                    }
                }
            }
        }

        // Periodic save (weights-only — full mode is not yet wired).
        let step_num = step + 1;
        let save_now = trainer_common::cadence_fires(args.save_every, step_num, args.steps);
        let sample_now =
            periodic_sample && trainer_common::cadence_fires(args.sample_every, step_num, args.steps);
        if save_now || sample_now {
            trainer_pipeline::with_optional_ema_swap(
                ema.as_ref(),
                &params,
                args.ema_validation_swap && save_now,
                "mid",
                || {
                    if save_now {
                        let mid_ckpt = args
                            .output_dir
                            .join(format!("chroma_lora_step{step_num}.safetensors"));
                        trainer_pipeline::save_lora_checkpoint(
                            trainer_pipeline::CheckpointSaveOptions {
                                trainer: "train_chroma",
                                path: &mid_ckpt,
                                step: step_num as u64,
                                rank: args.rank,
                                alpha: args.lora_alpha as f32,
                                seed: SEED,
                                config_hash: "",
                                save_mode_full,
                                label: &format!("[save step {step_num}]"),
                            },
                            &opt,
                            || {
                                let Some(bundle) = model.bundle.as_ref() else {
                                    anyhow::bail!("--save-mode full requires LoRA mode");
                                };
                                Ok(bundle.named_parameters())
                            },
                            || {
                                model.save_weights(&mid_ckpt)?;
                                Ok(())
                            },
                        )?;
                    }

                    // Periodic in-loop sample — every config-driven prompt.
                    if sample_now {
                        let vae_path = sample_vae_path.as_ref().unwrap();
                        for (idx, (label, cond, uncond)) in sample_set.iter().enumerate() {
                            let out_path = args
                                .output_dir
                                .join(format!("sample_step{}_{label}.png", step_num));
                            log::info!(
                                "[sample step={} {label}] → {}",
                                step_num,
                                out_path.display()
                            );
                            if let Err(e) = chroma_sampler::sample_image(
                                &model,
                                cond,
                                uncond,
                                args.sample_size,
                                args.sample_size,
                                args.sample_steps,
                                args.sample_cfg,
                                args.sample_seed,
                                vae_path,
                                &out_path,
                                &device,
                            ) {
                                log::warn!("[sample step={} {label}] failed: {e}", step_num);
                            } else if let Some(b) = &board {
                                b.log_image_png(&format!("samples/{label}"), step_num as u64, 0, &out_path);
                                if let Some(t) = sample_prompt_texts.get(idx) {
                                    b.log_text(&format!("prompts/{label}"), step_num as u64, t);
                                }
                            }
                        }
                        flame_core::cuda_alloc_pool::clear_pool_cache();
                        flame_core::trim_cuda_mempool(0);
                    }
                    Ok(())
                },
            )?;
        }
    }

    // Final EMA swap. No restore — process exits.
    trainer_pipeline::swap_ema_for_final_save(ema.as_ref(), &params, args.ema_validation_swap)?;

    let completion = loop_run.finish();
    log::info!(
        "Training complete: {} new steps (total={}), avg loss={:.4}",
        completion.trained_steps,
        args.steps,
        completion.average_loss
    );
    trainer_pipeline::mark_board_completed(board.as_ref());

    let suffix = if args.mode == "full" { "full" } else { "lora" };
    let ckpt = args
        .output_dir
        .join(format!("chroma_{suffix}_{}steps.safetensors", args.steps));
    trainer_pipeline::save_lora_checkpoint(
        trainer_pipeline::CheckpointSaveOptions {
            trainer: "train_chroma",
            path: &ckpt,
            step: args.steps as u64,
            rank: args.rank,
            alpha: args.lora_alpha as f32,
            seed: SEED,
            config_hash: "",
            save_mode_full,
            label: "[final]",
        },
        &opt,
        || {
            let Some(bundle) = model.bundle.as_ref() else {
                anyhow::bail!("--save-mode full requires LoRA mode");
            };
            Ok(bundle.named_parameters())
        },
        || {
            model.save_weights(&ckpt)?;
            Ok(())
        },
    )?;

    // ── Final sample at the end of training (all config-driven prompts) ───
    if periodic_sample {
        let vae_path = sample_vae_path.as_ref().unwrap();
        for (idx, (label, cond, uncond)) in sample_set.iter().enumerate() {
            let out_path = args
                .output_dir
                .join(format!("sample_step{}_{label}.png", args.steps));
            log::info!(
                "[sample step={} FINAL {label}] → {}",
                args.steps,
                out_path.display()
            );
            if let Err(e) = chroma_sampler::sample_image(
                &model,
                cond,
                uncond,
                args.sample_size,
                args.sample_size,
                args.sample_steps,
                args.sample_cfg,
                args.sample_seed,
                vae_path,
                &out_path,
                &device,
            ) {
                log::warn!("[sample final {label}] failed: {e}");
            } else if let Some(b) = &board {
                b.log_image_png(&format!("samples/{label}"), args.steps as u64, 0, &out_path);
                if let Some(t) = sample_prompt_texts.get(idx) {
                    b.log_text(&format!("prompts/{label}"), args.steps as u64, t);
                }
            }
        }
        flame_core::cuda_alloc_pool::clear_pool_cache();
        flame_core::trim_cuda_mempool(0);
    }
    Ok(())
}
