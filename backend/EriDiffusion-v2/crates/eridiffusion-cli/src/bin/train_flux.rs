//! train_flux — FLUX.1 (Dev/Schnell) LoRA training binary.
//!
//! Mirrors `train_ernie.rs` per OT preset semantics (see `/home/alex/upstream Python/training_presets/#flux LoRA.json`):
//!   - learning_rate         = 0.0003
//!   - batch_size            = 4              (per-step here is 1; gradient acc not yet wired)
//!   - resolution            = 768            (dataset-side)
//!   - lora_rank / alpha     = 16 / 1.0       (preset doesn't pin these — common community defaults)
//!   - timestep_distribution = LOGIT_NORMAL   (preset L27)
//!   - dynamic_timestep_shifting = false      (preset L28)
//!   - timestep_shift        = 1.0            (default; identity shift — OT preset has dynamic_timestep_shifting=false;
//!                                            per-resolution shift is an INFERENCE knob, not used in OT training)
//!                                            BUG-FIX Set-B 2026-05-25: config.timestep_shift now wired via --timestep-shift.
//!   - noising_weight        = 0.0            (TrainConfig default → LOGIT_NORMAL scale = 1.0)
//!   - noising_bias          = 0.0            (TrainConfig default → LOGIT_NORMAL bias  = 0.0)
//!   - clip_grad_norm        = 1.0
//!   - train_dtype           = BF16
//!   - optimizer             = AdamW (β=(0.9, 0.999), ε=1e-8, wd=0.01)
//!
//! Variant flag (`--variant dev|schnell`): controls guidance handling.
//! Dev model has `guidance_in.in_layer.weight` → injects guidance=3.5 at training time.
//! Schnell has no guidance embed → guidance is ignored.
//!
//! Cached sample format (produced by prepare_flux):
//!   latent     [1, 16, H/8, W/8]  BF16   (RAW VAE posterior — unscaled, unpacked)
//!   t5_embed   [1, 512, 4096]     BF16
//!   clip_pool  [1, 768]           BF16
//!
//! Audit fix FLUX_VERIFY §H3: pre-fix the cache stored already-scaled,
//! already-packed latents AND `img_ids`/`txt_ids`. We now match upstream Python's
//! contract — cache is RAW posterior; `(latent - SHIFT) * SCALE` and
//! `pack_latents` happen at training time; position IDs are recomputed per
//! step from latent shape.

use clap::{Parser, ValueEnum};
use eridiffusion_cli::{trainer_common, trainer_pipeline};
use eridiffusion_core::config::LrScheduler;
use eridiffusion_core::encoders::flux_vae::{SCALE, SHIFT};
use eridiffusion_core::lycoris::{LycorisAlgo, LycorisBundleConfig};
use eridiffusion_core::models::{flux::FluxModel, TrainableModel};
use eridiffusion_core::sampler::flux_sampler::{build_img_ids, build_txt_ids, pack_latents};
use eridiffusion_core::training::checkpoint;
use eridiffusion_core::training::ema::ParameterEma;
use eridiffusion_core::training::features::ema_advanced::EmaConfig;
use eridiffusion_core::training::features::validation::ValidationLoop;
use eridiffusion_core::training::features::{
    loss_weight, lr_schedule, noise_modifiers, timestep_bias,
};
use eridiffusion_core::training::training_features::timestep_dist::TimestepConfig;
use eridiffusion_core::training::training_features::{Optimizer, OptimizerKind};
use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};
use std::path::PathBuf;

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const LOGIT_NORMAL_BIAS: f32 = 0.0; // TrainConfig.noising_bias default
const LOGIT_NORMAL_SCALE: f32 = 1.0; // noising_weight + 1.0 = 0.0 + 1.0
const TIMESTEP_SHIFT: f32 = 1.0; // dead — used only in the kept-for-reference fn; CLI uses args.timestep_shift
const SEED: u64 = 42;
const CLIP_GRAD_NORM: f32 = 1.0;

#[derive(Copy, Clone, ValueEnum, Debug)]
enum Variant {
    Dev,
    Schnell,
}

#[derive(Parser)]
struct Args {
    /// OT-format JSON config (optional). Falls back to TrainConfig::default() if absent.
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    cache_dir: PathBuf,
    /// FLUX transformer dir (containing `flux1-dev.safetensors` or shards).
    #[arg(long)]
    transformer: PathBuf,
    #[arg(long, value_enum, default_value_t = Variant::Dev)]
    variant: Variant,
    #[arg(long, default_value = "100")]
    steps: usize,
    #[arg(long, default_value = "16")]
    rank: usize,
    /// LoRA alpha. Convention: alpha = rank (effective scale = 1.0). FLUX_VERIFY §H12 /
    /// SKEPTIC §H12. Pre-fix default was 1.0 with rank=16 → effective scale = 0.0625.
    #[arg(long, default_value = "16.0")]
    lora_alpha: f64,
    /// Preset learning rate (3e-4).
    #[arg(long, default_value = "3e-4")]
    lr: f32,
    #[arg(long, default_value = "output")]
    output_dir: PathBuf,
    /// Per-block weight streaming (drops resident block weights, reloads per layer).
    #[arg(long)]
    offload: bool,
    /// Save a LoRA checkpoint every N steps (0 = end-only).
    #[arg(long, default_value = "0")]
    save_every: usize,
    /// Resume LoRA weights only (no optimizer state).
    #[arg(long, conflicts_with = "resume_full")]
    resume_lora: Option<PathBuf>,
    /// Full resume: LoRA weights + AdamW (m, v, t) + step counter.
    #[arg(long, conflicts_with = "resume_lora")]
    resume_full: Option<PathBuf>,
    /// Periodic + final save mode. Default `full` (LoRA + AdamW + step) for
    /// resumable runs. `weights` writes legacy weights-only files.
    #[arg(long, default_value = "full")]
    save_mode: String,

    // ── Phase 0 multi-feature rollout (default-off; Phase 1+ will consume) ──
    #[arg(long)]
    min_snr_gamma: Option<f32>,
    #[arg(long, default_value_t = 0.0)]
    caption_dropout_probability: f32,
    /// Path to a single cache file produced by `prepare_flux` from an empty-
    /// caption sample. When `--caption-dropout-probability > 0`, the trainer
    /// loads `t5_embed` + `clip_pool` from this file and swaps them in with
    /// probability `p` per step. If unset and dropout > 0, the feature is
    /// disabled with a warning (preserves prior behaviour).
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
    #[arg(long, num_args = 0..)]
    multi_backend_weights: Vec<f32>,
    /// Phase 2: paired with --multi-backend-weights. Klein-only wiring; other
    /// trainers accept-and-warn until per-model wiring lands.
    #[arg(long, num_args = 0..)]
    multi_backend_cache_dirs: Vec<std::path::PathBuf>,
    /// Phase 2: validation prompt library JSON (Klein-only wiring; other
    /// trainers accept-and-warn).
    #[arg(long)]
    validation_prompts_file: Option<std::path::PathBuf>,
    #[arg(long, default_value_t = 0.0)]
    masked_loss_weight: f32,
    /// Master switch for EMA shadow. See train_klein.rs for full doc.
    /// Default off → byte-identical to pre-flag commits.
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
    /// Swap shadow → live params at sample/save time.
    #[arg(long, default_value_t = false)]
    ema_validation_swap: bool,
    /// Pyramid / multi-resolution noise: number of additional resolution
    /// levels to mix into the per-step training noise. `0` (default) is a
    /// no-op — byte-identical to no-multires.
    #[arg(long, default_value_t = 0)]
    multires_noise_iterations: usize,
    /// Per-level discount factor for `--multires-noise-iterations`.
    #[arg(long, default_value_t = 0.3)]
    multires_noise_discount: f32,
    /// Multi-distribution timestep bias strategy. `none` is byte-identical.
    #[arg(long, default_value = "none")]
    timestep_bias_strategy: String,
    #[arg(long, default_value_t = 0.0)]
    timestep_bias_multiplier: f32,
    #[arg(long, default_value_t = 0.0)]
    timestep_bias_range_min: f32,
    #[arg(long, default_value_t = 1.0)]
    timestep_bias_range_max: f32,
    /// Timestep distribution. `logit_normal` (default — FLUX preset L27),
    /// `uniform`, `sigmoid`, `heavy_tail`, `cos_map`, `inverted_parabola`.
    #[arg(long, default_value = "logit_normal")]
    timestep_distribution: String,
    /// SD3-style flow-matching timestep shift applied AFTER the distribution
    /// sample: `t' = N*shift*t / ((shift-1)*t + N)`. `1.0` (default) is the
    /// identity (no shift). OneTrainer's Flux preset defaults to `1.0`
    /// (dynamic_timestep_shifting=false; per-resolution shift is an inference
    /// knob, not a training knob per the OT Flux preset). Mirrors the fix
    /// applied to train_zimage.rs and train_klein.rs — config.timestep_shift
    /// was previously silently ignored.
    #[arg(long, default_value_t = 1.0)]
    timestep_shift: f32,
    /// Distribution-specific weight knob (default 0.0 — FLUX preset).
    #[arg(long, default_value_t = 0.0)]
    noising_weight: f32,
    /// Distribution-specific bias knob (default 0.0 — FLUX preset).
    #[arg(long, default_value_t = 0.0)]
    noising_bias: f32,
    #[arg(long)]
    tread_route_pattern: Option<String>,
    /// Phase 1: optimizer family CLI surface (Phase 5 wires full dispatch).
    #[arg(long, default_value = "adamw")]
    optimizer: String,

    // ── Phase 6 multi-feature rollout (plumb-only; multi-backend wired in Klein) ──
    #[arg(long, num_args = 0..)]
    multi_backend_repeats: Vec<u32>,
    #[arg(long, default_value_t = false)]
    caption_tag_shuffle: bool,
    #[arg(long, default_value_t = false)]
    cache_clear_each_epoch: bool,
    #[arg(long, default_value_t = false)]
    cache_invalidate: bool,
    /// Phase 5: LR scheduler family. Default `constant` + `warmup_steps=0` is
    /// byte-equivalent to prior fixed-LR behaviour.
    #[arg(long, default_value = "constant")]
    lr_scheduler: String,
    /// Phase 5: linear LR warmup steps. Default 0 keeps prior behaviour.
    #[arg(long, default_value_t = 0)]
    warmup_steps: usize,
    /// Phase 5: cosine-with-restarts cycle count.
    #[arg(long, default_value_t = 1.0)]
    lr_cycles: f32,

    // ── Phase 2b: LyCORIS algo selection ───────────────────────────────
    // Default `lora` keeps the legacy `FluxLoraBundle` path (byte-identical
    // to pre-LyCORIS commits). Other values switch to `FluxLycorisBundle`:
    // - `locon` — LoRA + Conv2d. Linear-only here so identical to plain
    //   LoRA at the call-site, but uses the lycoris-rs leaves (different
    //   init RNG, different storage layout).
    // - `loha`  — Hadamard product LoRA.
    // - `lokr`  — Kronecker product LoRA. See `--lokr-factor`.
    // - `full`  — full-weight delta. Non-residual; trainer-side merge required
    //   (Phase 2b plumbs the bundle path; merge-side is a follow-up).
    // - `oft`   — Orthogonal Fine-Tuning (Diag-OFT). See `--oft-block-size`,
    //   `--oft-neumann-terms`.
    // Combine with `--dora` to layer DoRA on any of the above (except OFT).
    /// LyCORIS algorithm selector. Default `lora` keeps the legacy
    /// plain-LoRA path byte-identical.
    #[arg(long, default_value = "lora")]
    algo: String,
    /// LoKr Kronecker split factor (default 16). Ignored for non-LoKr algos.
    #[arg(long, default_value_t = 16)]
    lokr_factor: i32,
    /// OFT block size (default 32). Ignored for non-OFT algos.
    #[arg(long, default_value_t = 32)]
    oft_block_size: usize,
    /// OFT Cayley-Neumann series term count (default 5). Ignored for non-OFT.
    #[arg(long, default_value_t = 5)]
    oft_neumann_terms: usize,
    /// LoCon / LoHa / LoKr conv variant — Tucker decomposition for non-1×1.
    /// Linear adapters ignore this flag.
    #[arg(long, default_value_t = false)]
    use_tucker: bool,
    /// LoKr only: factorize both W1 and W2 (default false: only W2).
    #[arg(long, default_value_t = false)]
    decompose_both: bool,
    /// Enable DoRA (weight-decomposed LoRA) on the adapter stack. Applies to
    /// LoCon / LoHa / LoKr / Full; not OFT.
    #[arg(long, default_value_t = false)]
    dora: bool,
    /// DoRA axis convention. `true` (default) = lycoris-upstream norm over
    /// input dims; `false` = OneTrainer norm over output dim.
    #[arg(long, default_value_t = true)]
    dora_wd_on_out: bool,

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

/// LOGIT_NORMAL timestep sample matching OT `_get_timestep_discrete`.
///
/// Superseded by the unified `TimestepConfig` dispatch — kept for reference
/// (Box-Muller-vs-Ziggurat divergence is documented in the wiring report).
#[allow(dead_code)]
fn sample_timestep_logit_normal(rng: &mut rand::rngs::StdRng) -> f32 {
    trainer_common::sample_logit_normal_timestep(
        rng,
        NUM_TRAIN_TIMESTEPS,
        LOGIT_NORMAL_BIAS,
        LOGIT_NORMAL_SCALE,
        TIMESTEP_SHIFT,
        0.0,
        1.0,
    )
}

/// Build the unified `TimestepConfig` from CLI args.
fn build_timestep_config(
    distribution: &str,
    weight: f32,
    bias: f32,
) -> anyhow::Result<TimestepConfig> {
    trainer_common::build_full_strength_timestep_config(distribution, weight, bias)
}

fn collect_shards(path: &std::path::Path) -> anyhow::Result<Vec<PathBuf>> {
    trainer_common::collect_safetensor_shards(path, "")
}

fn main() -> anyhow::Result<()> {
    use rand::SeedableRng;
    trainer_common::init_logging();
    let args = Args::parse();
    // Phase 2: Klein-only wiring of multi-backend + validation prompts library.
    // Other trainers accept-and-warn so configs/launchers aren't broken; full
    // wiring is a follow-up after the per-model encoder + sample paths are
    // consolidated.
    trainer_common::warn_unsupported_multi_backend_flags(
        &args.multi_backend_cache_dirs,
        &args.multi_backend_weights,
    );
    trainer_common::warn_unsupported_validation_prompts_file(args.validation_prompts_file.as_deref());
    trainer_common::ensure_output_dir(&args.output_dir)?;

    let device = trainer_common::init_bf16_cuda();

    let mut config = trainer_common::load_train_config_or_default(args.config.as_deref())?;
    trainer_common::apply_lora_basics(&mut config, args.rank, args.lora_alpha, args.lr);

    // Phase 0 multi-feature rollout — plumb CLI args into config (default-off, unused yet).
    config.min_snr_gamma = args.min_snr_gamma;
    config.caption_dropout_probability = args.caption_dropout_probability;
    config.noise_offset_probability = args.noise_offset_probability;
    config.gamma_input_perturbation = args.gamma_input_perturbation;
    config.huber_strength = args.huber_strength;
    config.lr_min_factor = args.lr_min_factor;
    config.validation_dataset_dir = args.validation_dataset_dir.clone();
    config.validation_every_steps = args.validation_every_steps;
    config.multi_backend_weights = args.multi_backend_weights.clone();
    config.validation_prompts_file = args.validation_prompts_file.clone();
    config.masked_loss_weight = args.masked_loss_weight;
    if args.masked_loss_weight > 0.0 {
        log::warn!(
            "[masked-loss] --masked-loss-weight={:.3} requested but Flux's prepare_flux cache schema has no `latent_mask` field; flag is a no-op for this trainer.",
            args.masked_loss_weight
        );
    }
    config.ema_inv_gamma = args.ema_inv_gamma;
    config.ema_power = args.ema_power;
    config.ema_update_after_step = args.ema_update_after_step;
    config.ema_min_decay = args.ema_min_decay;
    config.tread_route_pattern = args.tread_route_pattern.clone();

    let shards = collect_shards(&args.transformer)?;
    log::info!(
        "[Flux] loading transformer from {} shard(s) (variant={:?}, rank={}, alpha={}, algo={})...",
        shards.len(),
        args.variant,
        args.rank,
        args.lora_alpha,
        args.algo
    );

    // Phase 2b: parse `--algo`. `lora`/`none` -> legacy `FluxLoraBundle`
    // (byte-identical to pre-LyCORIS commits). `LycorisAlgo::parse("lora")`
    // aliases to LoCon globally, so re-map it here to match SDXL/SD3.5:
    // users who want the LyCORIS LoCon path pass `--algo locon`.
    let algo_str = args.algo.trim().to_ascii_lowercase();
    let lyc_algo = if algo_str == "lora" || algo_str == "none" || algo_str.is_empty() {
        LycorisAlgo::None
    } else {
        LycorisAlgo::parse(&args.algo).map_err(|e| anyhow::anyhow!("--algo: {e}"))?
    };
    let lyc_cfg = if lyc_algo == LycorisAlgo::None {
        None
    } else {
        let mut cfg = LycorisBundleConfig::default();
        cfg.algo = lyc_algo;
        cfg.rank = args.rank;
        cfg.alpha = args.lora_alpha as f32;
        cfg.factor = args.lokr_factor;
        cfg.block_size = args.oft_block_size;
        cfg.neumann_terms = args.oft_neumann_terms;
        cfg.use_tucker = args.use_tucker;
        cfg.decompose_both = args.decompose_both;
        cfg.dora = args.dora;
        cfg.dora_wd_on_out = args.dora_wd_on_out;
        log::info!(
            "[Flux] LyCORIS config: algo={} rank={} alpha={} dora={} (wd_on_out={}) factor={} oft_block={} use_tucker={} decompose_both={}",
            cfg.algo.as_str(), cfg.rank, cfg.alpha, cfg.dora, cfg.dora_wd_on_out,
            cfg.factor, cfg.block_size, cfg.use_tucker, cfg.decompose_both,
        );
        Some(cfg)
    };

    // Per-shard load via FluxModel::load{,_with_lycoris} — pass the first
    // shard (typical Flux release ships a single `flux1-dev.safetensors`);
    // auto-detection of has_guidance happens from the keys.
    let mut model = FluxModel::load_with_lycoris(&shards[0], &config, device.clone(), lyc_cfg)?;
    // Variant override: if the user explicitly says Schnell, force guidance off even if
    // the checkpoint accidentally has guidance keys (and vice-versa for Dev).
    match args.variant {
        Variant::Dev => {
            if !model.has_guidance {
                log::warn!("--variant dev but loaded checkpoint has no guidance_in. Treating as Schnell-shaped Dev (no guidance injection).");
            }
            // Audit fix FLUX_VERIFY §H7 — `load()` defaults `guidance_value` to
            // 1.0 (matches OT canonical training default). 3.5 is inference-only.
        }
        Variant::Schnell => {
            model.has_guidance = false;
            model.guidance_value = 1.0;
        }
    }
    if args.offload {
        model.enable_offload(shards.clone())?;
        log::info!(
            "  block-offload enabled — per-block streaming from {} shards",
            shards.len()
        );
    }

    let mut params = model.parameters();
    // Gate-on 6a: under v2 (default), flip LoRA params to MatchParamDtype so
    // BF16 grads from the bridge stay BF16 (Class A). --use-autograd-v3 skips.
    trainer_pipeline::apply_autograd_v2_grad_policy(&mut params, args.use_autograd_v3, "params");
    log::info!("[Flux] {} trainable LoRA tensors", params.len());
    if params.is_empty() {
        anyhow::bail!("No trainable parameters — check TrainingMethod::Lora setup");
    }

    // OT preset optimizer: AdamW(β=(0.9, 0.999), ε=1e-8, wd=0.01).
    // Phase B (2026-05-10): unified Optimizer enum dispatches all kinds.
    let opt_kind =
        OptimizerKind::parse(&args.optimizer).map_err(|e| anyhow::anyhow!("--optimizer: {e}"))?;
    log::info!("[Flux] optimizer={}", opt_kind.as_str());
    let mut opt = Optimizer::new(opt_kind, args.lr, 0.9, 0.999, 1e-8, 0.01);

    // EMA shadow (Phase 3). See train_klein.rs for the same pattern.
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

    // Timestep bias config — defaults are byte-identical (Strategy::None).
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

    // Phase 1: caption_dropout. Flux has no inline encoder, so the user
    // supplies a `--null-text-cache` produced by `prepare_flux` on a single
    // empty-caption sample. We load `t5_embed` + `clip_pool` once and swap
    // them in per-step with the configured probability. Without
    // `--null-text-cache`, the feature is disabled with a warning.
    let mut effective_caption_dropout_prob = args.caption_dropout_probability;
    let null_text: Option<(Tensor, Tensor)> = if effective_caption_dropout_prob > 0.0 {
        match args.null_text_cache.as_ref() {
            Some(p) => match flame_core::serialization::load_file(p, &device) {
                Ok(s) => {
                    let nt5 = s
                        .get("t5_embed")
                        .ok_or_else(|| anyhow::anyhow!("--null-text-cache missing 't5_embed'"))?
                        .to_dtype(DType::BF16)?;
                    let nclip = s
                        .get("clip_pool")
                        .ok_or_else(|| anyhow::anyhow!("--null-text-cache missing 'clip_pool'"))?
                        .to_dtype(DType::BF16)?;
                    log::info!(
                        "[caption-dropout] WIRED — prob={:.3} (null_t5={:?}, null_clip_pool={:?})",
                        effective_caption_dropout_prob,
                        nt5.shape().dims(),
                        nclip.shape().dims()
                    );
                    Some((nt5, nclip))
                }
                Err(e) => {
                    log::warn!("[caption-dropout] failed to load --null-text-cache {}: {e} — feature disabled", p.display());
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

    if let Some(resume_path) = args.resume_lora.as_ref() {
        log::info!(
            "Resuming LoRA weights only (no optimizer state) from {}",
            resume_path.display()
        );
        model.load_weights(&resume_path.to_string_lossy())?;
    }

    let mut start_step: usize = 0;
    if let Some(resume_path) = args.resume_full.as_ref() {
        log::info!("Full-resume from {}", resume_path.display());
        let loaded = checkpoint::load_full(resume_path, &device)?;
        // Phase 2b: full-resume from either bundle. Order: legacy LoRA first
        // (mutually exclusive at construction time, so at most one matches).
        let named = if let Some(ref bundle) = model.bundle {
            bundle.named_parameters()
        } else if let Some(ref lb) = model.lycoris_bundle {
            lb.named_parameters()
        } else {
            anyhow::bail!("Flux full-resume requires LoRA or LyCORIS mode");
        };
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
                "[resume-full] non-AdamW resume not yet implemented for {:?}; LoRA weights restored, optimizer state reset",
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
        log::info!("Continuing from step {start_step}/{}", args.steps);
    }

    let save_mode_full = match args.save_mode.as_str() {
        "full" => true,
        "weights" => false,
        other => anyhow::bail!("--save-mode must be `full` or `weights`, got `{other}`"),
    };

    let cache_files = trainer_common::list_cache_safetensors(&args.cache_dir)?;
    log::info!("[Flux] {} cached samples", cache_files.len());

    // SEED=42 for both timestep RNG and noise RNG. Audit fix FLUX_VERIFY §H7 /
    // SKEPTIC §H10 / §M1 (`feedback_default_seed_42.md`): seed flame_core's
    // global RNG so `Tensor::randn` is deterministic too, not just our host RNG.
    trainer_common::set_flame_seed(SEED)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let board = trainer_common::open_board_writer(
        &args.output_dir,
        trainer_common::board_resume_step(start_step),
    );
    if let Some(b) = &board {
        // Full board wiring: run hyper-parameters → metadata.hparams + the
        // dashboard's hparam panel. JSON hand-built (no serde_json dep here).
        let hparams_json = format!(
            "{{\"model\":\"flux\",\"steps\":{},\"rank\":{},\"lora_alpha\":{},\"lr\":{},\
             \"batch_size\":{},\"optimizer\":\"{}\",\"timestep_shift\":{},\"seed\":{}}}",
            args.steps,
            args.rank,
            args.lora_alpha,
            args.lr,
            1,
            opt_kind.as_str(),
            args.timestep_shift,
            SEED
        );
        b.log_hparams(&hparams_json, &[("steps_target", args.steps as f64)]);
    }
    let loop_config = trainer_pipeline::ManualTrainLoopConfig::new(
        "FLUX.1-lora",
        start_step,
        args.steps,
        cache_files.len(),
        1,
    );
    let mut loop_run = trainer_pipeline::ManualTrainLoopRun::new(loop_config);

    // Validation harness — held-out cache + cadence. None at default
    // (validation_every_steps == 0). Mirrors train_klein.rs:608-624. Default-off
    // is byte-identical to pre-port behaviour: harness not constructed, no
    // per-step branch executed, training-side `rng` untouched.
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

        // Audit fix FLUX_VERIFY §H3 / SKEPTIC §H3: cache stores RAW VAE
        // posterior latent `[1, 16, H/8, W/8]` (no shift/scale, no patchify).
        // We apply `(raw - SHIFT) * SCALE` (correct BFL direction — H2) and
        // `pack_latents` at train time. Matches upstream Python contract
        // (`BaseFluxSetup.py:235`). Also: we no longer cache img_ids/txt_ids;
        // they are recomputed from latent shape per step (cheap, mirrors OT).
        let latent_raw = sample
            .get("latent")
            .ok_or_else(|| anyhow::anyhow!("missing 'latent'"))?
            .to_dtype(DType::BF16)?;
        let t5 = sample
            .get("t5_embed")
            .ok_or_else(|| anyhow::anyhow!("missing 't5_embed'"))?
            .to_dtype(DType::BF16)?;
        let clip_pool = sample
            .get("clip_pool")
            .ok_or_else(|| anyhow::anyhow!("missing 'clip_pool'"))?
            .to_dtype(DType::BF16)?;
        // Caption dropout: OT parity (F-C3). FluxModel.py:291-299 draws INDEPENDENT
        // Bernoullis for TE1 (CLIP pool) and TE2 (T5 hidden) via sequential rand.random()
        // calls. We match this with two independent draws from our rng:
        //   draw1 → CLIP pool (text_encoder_1_dropout_probability)
        //   draw2 → T5 hidden (text_encoder_2_dropout_probability)
        // Both use the same effective_caption_dropout_prob (OT default: same prob for both).
        // Default-off (prob == 0.0 OR null_text == None) draws no rng.
        let (t5, clip_pool) = if let Some((ref nt5, ref nclip)) = null_text {
            use rand::Rng;
            let drop_clip: bool = rng.r#gen::<f32>() < effective_caption_dropout_prob;
            let drop_t5: bool = rng.r#gen::<f32>() < effective_caption_dropout_prob;
            let out_clip = if drop_clip { nclip.clone() } else { clip_pool };
            let out_t5 = if drop_t5 { nt5.clone() } else { t5 };
            (out_t5, out_clip)
        } else {
            (t5, clip_pool)
        };

        // VAE shift/scale (H2: encode = `(raw - shift) * scale` — multiply,
        // not divide; pre-fix prepare_flux divided which left latents at
        // ~7.7× the variance the BFL DiT was trained on).
        let latent_scaled = latent_raw.add_scalar(-SHIFT)?.mul_scalar(SCALE)?;
        let (latent, h_tok, w_tok) = pack_latents(&latent_scaled)?;
        let latent_f32 = latent.to_dtype(DType::F32)?;
        let latent = latent_f32.to_dtype(DType::BF16)?;
        let n_txt = t5.shape().dims()[1];
        let img_ids = build_img_ids(h_tok, w_tok, device.clone())?.to_dtype(DType::BF16)?;
        let txt_ids = build_txt_ids(n_txt, device.clone())?.to_dtype(DType::BF16)?;

        // Flow-matching: t in [0, 1000), sigma = (idx+1)/1000.
        let raw_t_unshifted = timestep_cfg.sample_one(&mut rng) * NUM_TRAIN_TIMESTEPS as f32;
        // SD3-style flow-matching shift (identity when shift==1.0). Mirrors
        // OT `BaseFluxSetup.py:244-251` which passes
        // `shift = config.timestep_shift` to `_get_timestep_discrete`.
        // BUG-FIX: config.timestep_shift was previously silently ignored
        // (hardcoded 1.0). Same pattern fixed in train_zimage.rs + train_klein.rs.
        let raw_t = if (args.timestep_shift - 1.0).abs() < 1e-6 {
            raw_t_unshifted
        } else {
            let s = args.timestep_shift;
            let n = NUM_TRAIN_TIMESTEPS as f32;
            n * s * raw_t_unshifted / ((s - 1.0) * raw_t_unshifted + n)
        };
        // Default-off: Strategy::None → returns raw_t unchanged.
        let t_continuous =
            timestep_bias::apply_bias(raw_t, NUM_TRAIN_TIMESTEPS as f32, &timestep_bias_cfg);
        // OT parity (BUG-2): discretize the continuous timestep to integer
        // bins BEFORE deriving sigma and the model timestep, matching
        // `ModelSetupNoiseMixin.py:212` which returns `timestep.int()`. The
        // BFL DiT was distilled on integer timesteps; passing fractional
        // bins shifts the sinusoid phase off the trained band. Sigma and
        // model-t now share the same discretization.
        let sigma_idx = (t_continuous.floor() as usize).min(NUM_TRAIN_TIMESTEPS - 1);
        let sigma = (sigma_idx + 1) as f32 / NUM_TRAIN_TIMESTEPS as f32;
        let t_int = sigma_idx as f32;

        // Match OneTrainer: keep noising and target construction in F32,
        // then cast only the transformer input to BF16.
        let noise = noise_modifiers::randn_f32(latent_f32.shape().clone(), device.clone())?;
        // Pyramid / multi-resolution noise (additive). Default-off when
        // iterations == 0 → byte-identical.
        let noise = noise_modifiers::maybe_apply_multires_noise(
            &noise,
            args.multires_noise_iterations,
            args.multires_noise_discount,
            &mut rng,
        )?;
        // Phase 1: noise modifiers (default-off). Offset noise is part of the
        // clean noise distribution; input perturbation feeds the model input
        // only (target keeps the unperturbed noise). Default-off byte
        // invariance: gamma=0 → perturbed_noise == clean_noise.
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
        let noisy_f32 = perturbed_noise
            .mul_scalar(sigma)?
            .add(&latent_f32.mul_scalar(1.0 - sigma)?)?;
        let noisy = noisy_f32.to_dtype(DType::BF16)?;
        // Rectified-flow target: target = noise - clean.
        let target = clean_noise.sub(&latent_f32)?;

        // Audit fix FLUX_VERIFY §H1 / SKEPTIC §H1: pass `t / 1000 ∈ [0, 1)` to
        // the model — `flux.rs::timestep_embedding` then multiplies by 1000
        // exactly once. Pre-fix the trainer passed `t ∈ [0, 1000)` and the
        // embedder multiplied again → `t * 1_000_000` in the sinusoid arg.
        // Mirrors Klein's recently-landed fix (`train_klein.rs:147`).
        //
        // BUG-2 (Wave-2): use the integer-discretized timestep so the model
        // sees the same bin OT does (`BaseFluxSetup.py:289` passes
        // `timestep / 1000` where timestep is `.int()`).
        let t_model = t_int / NUM_TRAIN_TIMESTEPS as f32;
        let timestep = Tensor::from_vec(vec![t_model], Shape::from_dims(&[1]), device.clone())?;

        if step == 0 {
            log::info!("step 0 | latent={:?} t5={:?} clip={:?} img_ids={:?} txt_ids={:?} sigma={:.4} t_model={:.4}",
                latent.shape().dims(), t5.shape().dims(), clip_pool.shape().dims(),
                img_ids.shape().dims(), txt_ids.shape().dims(), sigma, t_model);
        }

        let context = vec![t5, img_ids, txt_ids];
        let pred = <FluxModel as TrainableModel>::forward(
            &mut model,
            &noisy,
            &timestep,
            &context,
            Some(&clip_pool),
        )?;

        if pred.shape().dims() != target.shape().dims() {
            anyhow::bail!(
                "pred {:?} != target {:?}",
                pred.shape().dims(),
                target.shape().dims()
            );
        }

        // F32 mean-MSE (CONSTANT loss-weight, mse_strength=1.0).
        // Phase 1: combined loss + loss-weighting. Default-off invariant.
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
            true,
        )?;
        let loss_val = loss.to_vec()?[0];

        let grads = trainer_pipeline::backward_loss(&loss, args.use_autograd_v3)?;

        // Grad-flow diagnostic.  Runs at step 1, not step 0, because LoRA-like
        // adapters intentionally start with one zero factor; after the first
        // optimizer step, all trainable leaves should receive real gradients.
        if step == 1 {
            let named = if let Some(ref b) = model.bundle {
                b.named_parameters()
            } else if let Some(ref lb) = model.lycoris_bundle {
                lb.named_parameters()
            } else {
                Vec::new()
            };
            let named_refs: Vec<(&str, &flame_core::parameter::Parameter)> =
                named.iter().map(|(n, p)| (n.as_str(), p)).collect();
            let report = flame_core::diagnostics::assert_grad_flow(&grads, &named_refs)?;
            if report.is_clean() {
                log::info!("[grad-flow] step 2 clean ({} params)", report.ok_count);
            } else {
                log::warn!("{}", report.summary());
            }
        }

        // clip_grad_norm = 1.0 (preset default; matches OT BaseFluxSetup).
        // Fusion Sprint Phase 5: device-resident global L2 norm — one D2H per step.
        let clip = trainer_pipeline::apply_gradient_map_clip(
            &params,
            &grads,
            trainer_pipeline::GradientClipOptions::clip_by_norm(CLIP_GRAD_NORM),
        )?;
        let total_norm = clip.total_norm;

        // Phase 5: dispatch LR per scheduler. Default Constant + warmup_steps=0
        // is byte-equivalent to prior fixed-LR behaviour.
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
            if let Some(ref mut e) = ema {
                e.update_with_schedule(&params, &ema_cfg, (step + 1) as u64)
                    .map_err(|err| {
                        anyhow::anyhow!("EMA update failed at step {}: {err}", step + 1)
                    })?;
            }
            Ok(())
        })?;
        model.post_optimizer_step();

        loop_run.record_and_log(
            step,
            trainer_pipeline::TrainStepMetrics {
                loss_value: loss_val,
                grad_norm: total_norm,
                learning_rate: cur_lr,
            },
            board.as_ref(),
        );

        // Validation eval pass (no_grad) every `validation_every_steps`.
        // Mirrors train_klein.rs:1063-1142. Validation uses its own SIDE-RNG
        // seeded as `SEED ^ (step as u64 + 1)` so it does NOT perturb the
        // training-side `rng` sequence (byte-invariance with feature off).
        // Runs BEFORE periodic save / EMA swap.
        // Flux specifics vs Klein:
        //   - cache fields: latent / t5_embed / clip_pool (no `text_embedding`)
        //   - VAE shift/scale + pack_latents must be replayed (matches train step)
        //   - forward takes `(noisy, timestep, context=[t5,img_ids,txt_ids], Some(clip_pool))`
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
                        None => {
                            log::warn!("[validation] {} missing latent", vfile.display());
                            continue;
                        }
                    };
                    let v_t5 = match sample.get("t5_embed") {
                        Some(t) => t.to_dtype(DType::BF16)?,
                        None => {
                            log::warn!("[validation] {} missing t5_embed", vfile.display());
                            continue;
                        }
                    };
                    let v_clip_pool = match sample.get("clip_pool") {
                        Some(t) => t.to_dtype(DType::BF16)?,
                        None => {
                            log::warn!("[validation] {} missing clip_pool", vfile.display());
                            continue;
                        }
                    };
                    // Replay VAE shift/scale + pack_latents identically to train step.
                    let v_latent_scaled = v_latent_raw.add_scalar(-SHIFT)?.mul_scalar(SCALE)?;
                    let (v_latent, v_h_tok, v_w_tok) = pack_latents(&v_latent_scaled)?;
                    let v_latent = v_latent.to_dtype(DType::BF16)?;
                    let v_n_txt = v_t5.shape().dims()[1];
                    let v_img_ids =
                        build_img_ids(v_h_tok, v_w_tok, device.clone())?.to_dtype(DType::BF16)?;
                    let v_txt_ids =
                        build_txt_ids(v_n_txt, device.clone())?.to_dtype(DType::BF16)?;

                    // SIDE-RNG: do NOT touch training-side `rng`.
                    let mut vrng = rand::rngs::StdRng::seed_from_u64(SEED ^ (step as u64 + 1));
                    let raw_t_unshifted =
                        timestep_cfg.sample_one(&mut vrng) * NUM_TRAIN_TIMESTEPS as f32;
                    // Mirror training-step shift exactly.
                    let raw_t = if (args.timestep_shift - 1.0).abs() < 1e-6 {
                        raw_t_unshifted
                    } else {
                        let s = args.timestep_shift;
                        let n = NUM_TRAIN_TIMESTEPS as f32;
                        n * s * raw_t_unshifted / ((s - 1.0) * raw_t_unshifted + n)
                    };
                    let t_continuous = timestep_bias::apply_bias(
                        raw_t,
                        NUM_TRAIN_TIMESTEPS as f32,
                        &timestep_bias_cfg,
                    );
                    let sigma_idx = (t_continuous.floor() as usize).min(NUM_TRAIN_TIMESTEPS - 1);
                    let sigma = (sigma_idx + 1) as f32 / NUM_TRAIN_TIMESTEPS as f32;
                    let v_noise =
                        Tensor::randn(v_latent.shape().clone(), 0.0, 1.0, device.clone())?
                            .to_dtype(DType::BF16)?;
                    let v_noisy = v_noise
                        .mul_scalar(sigma)?
                        .add(&v_latent.mul_scalar(1.0 - sigma)?)?;
                    let v_target = v_noise.sub(&v_latent)?;
                    // OT parity (F-C1/F-C2): validation path must use the integer-discretized
                    // sigma_idx (matching training path at line 738 + 780: `t_int = sigma_idx as f32;
                    // t_model = t_int / NUM_TRAIN_TIMESTEPS`). Pre-fix used t_continuous / 1000
                    // (fractional) while the training path used sigma_idx as f32 / 1000 (integer).
                    // BaseFluxSetup.py always calls _get_timestep_discrete → timestep.int().
                    let v_t_model = sigma_idx as f32 / NUM_TRAIN_TIMESTEPS as f32;
                    let v_timestep =
                        Tensor::from_vec(vec![v_t_model], Shape::from_dims(&[1]), device.clone())?;
                    let v_context = vec![v_t5, v_img_ids, v_txt_ids];
                    let v_pred = match <FluxModel as TrainableModel>::forward(
                        &mut model,
                        &v_noisy,
                        &v_timestep,
                        &v_context,
                        Some(&v_clip_pool),
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

        // Periodic save (full or weights-only).
        let step_num = step + 1;
        let save_now = trainer_common::cadence_fires(args.save_every, step_num, args.steps);
        if save_now {
            trainer_pipeline::with_optional_ema_swap(
                ema.as_ref(),
                &params,
                args.ema_validation_swap,
                "mid",
                || {
                    let mid_ckpt = args
                        .output_dir
                        .join(format!("flux_lora_step{step_num}.safetensors"));
                    trainer_pipeline::save_lora_checkpoint(
                        trainer_pipeline::CheckpointSaveOptions {
                            trainer: "train_flux",
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
                            // Phase 2b: prefer legacy LoRA bundle; fall back to LyCORIS.
                            Ok(if let Some(ref b) = model.bundle {
                                b.named_parameters()
                            } else if let Some(ref lb) = model.lycoris_bundle {
                                lb.named_parameters()
                            } else {
                                Vec::new()
                            })
                        },
                        || {
                            model.save_weights(&mid_ckpt.to_string_lossy())?;
                            Ok(())
                        },
                    )
                },
            )?;
        }
    }

    // Final EMA swap before final save. No restore — process exits.
    trainer_pipeline::swap_ema_for_final_save(ema.as_ref(), &params, args.ema_validation_swap)?;

    let completion = loop_run.finish();
    log::info!(
        "Training complete: {} new steps (total={}), avg loss={:.4}",
        completion.trained_steps,
        args.steps,
        completion.average_loss
    );
    trainer_pipeline::mark_board_completed(board.as_ref());

    let ckpt = args
        .output_dir
        .join(format!("flux_lora_{}steps.safetensors", args.steps));
    trainer_pipeline::save_lora_checkpoint(
        trainer_pipeline::CheckpointSaveOptions {
            trainer: "train_flux",
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
            // Phase 2b: prefer legacy LoRA bundle; fall back to LyCORIS.
            Ok(if let Some(ref b) = model.bundle {
                b.named_parameters()
            } else if let Some(ref lb) = model.lycoris_bundle {
                lb.named_parameters()
            } else {
                Vec::new()
            })
        },
        || {
            model.save_weights(&ckpt.to_string_lossy())?;
            Ok(())
        },
    )?;
    Ok(())
}
