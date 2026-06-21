//! train_qwenimage — Qwen-Image-2512 LoRA training binary.
//!
//! Patterned after `train_ernie`. Loads cached `(latent, text_embedding)`
//! samples from `prepare_qwenimage` and runs flow-matching LoRA training.
//!
//! Reference: `flame-diffusion/qwenimage-trainer/src/{main,pipeline}.rs`,
//! `musubi-tuner/qwen_image_train_network.py`.
//!
//! Schedule: logit-normal then qwen_shift remap
//!   `t = sigmoid(z); t_shifted = t * shift / (1 + (shift - 1) * t)`
//!   shift comes from `shift_for_resolution([w, h])` (linear-mu over image
//!   seq-len anchors 256→0.5, 8192→0.9, exp).
//!
//! Loss: MSE in F32 between pred and target = noise - latent.

use clap::Parser;
use eridiffusion_cli::{trainer_common, trainer_pipeline};
use eridiffusion_core::config::LrScheduler;
use eridiffusion_core::encoders::qwen25vl::Qwen25VLEncoder;
use eridiffusion_core::lycoris::{LoraInitType, LycorisAlgo, LycorisBundleConfig};
use eridiffusion_core::models::{qwenimage as qwen_model, QwenImageTrainingModel};
use eridiffusion_core::sampler::qwenimage_sampler;
use eridiffusion_core::training::checkpoint;
use eridiffusion_core::training::ema::ParameterEma;
use eridiffusion_core::training::features::{
    caption_dropout, ema_advanced::EmaConfig, loss_weight, lr_schedule, noise_modifiers,
    sample_library::SampleLibrary, timestep_bias, validation::ValidationLoop,
};
use eridiffusion_core::training::training_features::timestep_dist::TimestepConfig;
use eridiffusion_core::training::training_features::{Optimizer, OptimizerKind};
use flame_core::gradient_clip::GradientClipper;
use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};
use rand::{rngs::StdRng, SeedableRng};
use std::path::PathBuf;

const SEED_DEFAULT: u64 = 42;
const QWEN_PAD_ID: i32 = 151643;
/// Qwen-Image canonical prompt template (`pipeline_qwenimage.py::
/// PROMPT_TEMPLATE_ENCODE`). The DiT was trained against text embeddings
/// produced by this template; raw captions = OOD conditioning.
const PROMPT_PREFIX: &str =
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, \
     texture, quantity, text, spatial relationships of the objects and background:\
     <|im_end|>\n<|im_start|>user\n";
const PROMPT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";
/// Drop the system-prompt prefix (matches Python
/// `PROMPT_TEMPLATE_ENCODE_START_IDX`).
const DROP_IDX: usize = 34;

#[derive(Parser)]
struct Args {
    /// Qwen-Image-2512 transformer directory (the `transformer/` subdir, with
    /// `diffusion_pytorch_model-0000{N}-of-00009.safetensors` shards).
    #[arg(long)]
    model: PathBuf,
    /// Cache dir produced by `prepare_qwenimage`.
    #[arg(long)]
    cache_dir: PathBuf,
    #[arg(long, default_value = "3000")]
    steps: usize,
    #[arg(long, default_value = "16")]
    rank: usize,
    #[arg(long, default_value = "16.0")]
    lora_alpha: f32,
    /// OneTrainer "qwen LoRA 24GB" preset default = 3e-4.
    #[arg(long, default_value = "3e-4")]
    lr: f32,
    /// Resolution at which the cache was prepared (used for qwen_shift when
    /// `--dynamic-timestep-shifting` is set).
    #[arg(long, default_value = "512")]
    resolution: usize,
    #[arg(long, default_value = "200")]
    warmup_steps: usize,
    /// Fixed timestep shift. OneTrainer's `#qwen LoRA 24GB.json` preset
    /// defaults to `1.0` (no shift) at training time; the diffusers/musubi
    /// inference path uses a resolution-dependent shift via
    /// `--dynamic-timestep-shifting`. Default `1.0` matches OT.
    #[arg(long, default_value_t = 1.0)]
    qwen_shift: f32,
    /// When set, override `--qwen-shift` with a resolution-dependent shift
    /// derived from `shift_for_resolution([w, h])` (matches musubi/diffusers
    /// inference). Default off → byte-identical shift=1.0 (OT preset).
    #[arg(long, default_value_t = false)]
    dynamic_timestep_shifting: bool,
    /// Resume LoRA weights only (no optimizer state).
    #[arg(long, conflicts_with = "resume_full")]
    resume_lora: Option<PathBuf>,
    /// Full resume: LoRA + AdamW + step.
    #[arg(long, conflicts_with = "resume_lora")]
    resume_full: Option<PathBuf>,
    /// Save mode: `full` (LoRA + AdamW + step) or `weights` (legacy).
    #[arg(long, default_value = "full")]
    save_mode: String,
    #[arg(long, default_value = "0")]
    save_every: usize,
    #[arg(long, default_value = "output")]
    output_dir: PathBuf,
    #[arg(long, default_value_t = SEED_DEFAULT)]
    seed: u64,

    // ── Periodic-sample (mirrors train_ernie pattern) ─────────────────────
    /// Render samples every N steps. `0` disables. When > 0: renders a
    /// step-0 baseline (LoRA = identity, base-model output for sanity
    /// check), then every N steps, then a final sample after training.
    /// Two prompts are rendered each time so we see both pose/style
    /// variations as the LoRA imprints. Per-sample cost: ~30s + denoise.
    #[arg(long, default_value = "0")]
    sample_every: usize,
    /// First prompt — caption-style with the LoRA trigger word.
    #[arg(long)]
    sample_prompt_1: Option<String>,
    /// Second prompt — different scene/outfit but same trigger word style.
    #[arg(long)]
    sample_prompt_2: Option<String>,
    /// Negative prompt (shared across both prompts). Empty disables uncond.
    #[arg(long, default_value = "")]
    sample_neg_prompt: String,
    /// `qwen_image_vae.safetensors` (wan21 internal-key format) for the
    /// in-process VAE decode. Required if --sample-every > 0.
    #[arg(long)]
    sample_vae: Option<PathBuf>,
    /// Qwen2.5-VL text encoder dir (or single combined safetensors).
    /// Required if --sample-every > 0.
    #[arg(long)]
    sample_text_encoder: Option<PathBuf>,
    /// `tokenizer.json` for Qwen2.5-VL. Required if --sample-every > 0.
    #[arg(long)]
    sample_tokenizer: Option<PathBuf>,
    #[arg(long, default_value = "1024")]
    sample_size: usize,
    /// Inference-flame's qwenimage_gen defaults to 50 steps; lower (20) shows
    /// visible texture artifacts at 1024² even with norm-rescaled CFG.
    #[arg(long, default_value = "50")]
    sample_steps: usize,
    #[arg(long, default_value = "4.0")]
    sample_cfg: f32,
    #[arg(long, default_value = "42")]
    sample_seed: u64,
    #[arg(long, default_value_t = 512)]
    sample_max_text_len: usize,

    // ── Phase 0 multi-feature rollout (default-off; Phase 1+ will consume) ──
    #[arg(long)]
    min_snr_gamma: Option<f32>,
    #[arg(long, default_value_t = 0.0)]
    caption_dropout_probability: f32,
    /// Path to a single cache file produced by `prepare_qwenimage` from an
    /// empty-caption sample. When `--caption-dropout-probability > 0`, the
    /// trainer loads `text_embedding` from this file and swaps it in with
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
    /// Master switch for EMA shadow. When `true` an F32 shadow is built from
    /// current live params (post resume_lora / pre-step-0) and updated after
    /// each opt.step via `update_with_schedule` per the EmaConfig schedule
    /// (see `--ema-inv-gamma`, `--ema-power`, `--ema-min-decay`,
    /// `--ema-update-after-step`, `--ema-max-decay`). Training loss is
    /// byte-identical to `--ema=false` because the shadow is parallel — only
    /// `--ema-validation-swap` makes it visible at sample / checkpoint time.
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
    /// Phase 3: EMA decay clamp upper bound. The schedule clamps the
    /// per-step computed decay to `[ema_min_decay, ema_max_decay]`. Standard
    /// values: 0.999 (fast averaging), 0.9999 (default — diffusers EMAModel),
    /// 0.99999 (very slow averaging).
    #[arg(long, default_value_t = 0.9999)]
    ema_max_decay: f32,
    /// Phase 3: swap EMA shadow weights into live params at sample/checkpoint
    /// time. Default false. No effect when EMA is not constructed.
    #[arg(long, default_value_t = false)]
    ema_validation_swap: bool,
    #[arg(long)]
    tread_route_pattern: Option<String>,

    // ── Multi-resolution noise (default-off; byte-invariant when 0) ──────
    #[arg(long, default_value_t = 0)]
    multires_noise_iterations: usize,
    #[arg(long, default_value_t = 0.3)]
    multires_noise_discount: f32,

    // ── Timestep biasing (default `none` is byte-identity) ───────────────
    #[arg(long, default_value = "none")]
    timestep_bias_strategy: String,
    #[arg(long, default_value_t = 0.0)]
    timestep_bias_multiplier: f32,
    #[arg(long, default_value_t = 0.0)]
    timestep_bias_range_min: f32,
    #[arg(long, default_value_t = 1.0)]
    timestep_bias_range_max: f32,
    /// Timestep distribution. `logit_normal` (default — qwen preset),
    /// `uniform`, `sigmoid`, `heavy_tail`, `cos_map`, `inverted_parabola`.
    /// The qwen-shift remap is applied after the unified sampler.
    #[arg(long, default_value = "logit_normal")]
    timestep_distribution: String,
    /// Distribution-specific weight knob (default 0.0 — qwen preset).
    #[arg(long, default_value_t = 0.0)]
    noising_weight: f32,
    /// Distribution-specific bias knob (default 0.0 — qwen preset).
    #[arg(long, default_value_t = 0.0)]
    noising_bias: f32,
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
    /// Phase 5: LR scheduler family. Default `constant` is byte-equivalent to
    /// the legacy linear-warmup-then-flat path qwenimage has used since launch.
    /// Accepted: constant, linear, cosine, cosine_with_restarts, polynomial, rex.
    #[arg(long, default_value = "constant")]
    lr_scheduler: String,
    /// Phase 5: cosine-with-restarts cycle count. Ignored for other schedulers.
    #[arg(long, default_value_t = 1.0)]
    lr_cycles: f32,

    // ── LyCORIS algo selection (Phase 2b) ──
    //
    // `--algo lora` (default) keeps the legacy LoRALinear path — byte-identical
    // training to pre-Phase-2b. Other values select LyCORIS algos via
    // `QwenImageLoraBundle::new_with_config`. `lora_alpha` and `rank` are
    // shared with the legacy CLI flags above (no separate `--lycoris-rank`).
    /// LyCORIS algo: `lora` (default, legacy path) | `locon` | `loha` | `lokr`
    /// | `full` | `oft`. `full` and `oft` build successfully but their
    /// `forward_delta` will error inside the qwenimage forward pass —
    /// qwenimage's `base + delta_on_input` call pattern is incompatible
    /// with Full/OFT semantics. Phase 2c will wire a `merge_into_base` path.
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
    /// kernels. Qwen-Image-2512 is linear-only so this is currently a no-op.
    #[arg(long, default_value_t = false)]
    use_tucker: bool,
    /// LoKr only: factorize both W1 *and* W2 (default false: only W2).
    #[arg(long, default_value_t = false)]
    decompose_both: bool,
    /// Enable DoRA (weight-decomposed LoRA). Applies to LoCon/LoHa/LoKr
    /// (Full inherits, OFT errors).
    ///
    /// Phase 2b limitation: qwenimage's bundle ctor doesn't have access to
    /// the streamed block weights at construction time, so DoRA's magnitude
    /// is initialized from `||I||_2 = 1` rather than `||W_orig||_2`. Phase 2c
    /// will wire pre-load magnitude init.
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
    /// SimpleTuner-style `lycoris_config preset.json`. Optional; per-target
    /// `module_algo_map` overrides apply during adapter construction.
    #[arg(long)]
    lycoris_config: Option<PathBuf>,
    /// SimpleTuner-parity: perturbed-normal LoKr init.  Scale `>0`
    /// triggers `lokr_w1=1, lokr_w2 ~ N(μ_W, σ_W)·scale`.  No-op when
    /// algo != lokr or value is 0.0.
    ///
    /// Phase 2b note: qwenimage's resident weight map is not yet plumbed
    /// to the bundle's perturbed-init helper (block weights are streamed
    /// via BlockOffloader). When set with `--algo lokr`, the bundle method
    /// logs a warning and returns Ok(()) without touching adapters. Phase
    /// 2c will wire the resident `block_weights` map into the init call.
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
    /// Per-element dropout on the adapter delta (training only).
    #[arg(long, default_value_t = 0.0)]
    lora_dropout: f32,
    /// Per-rank Bernoulli on the down-projection intermediate.
    #[arg(long, default_value_t = 0.0)]
    rank_dropout: f32,
    /// Per-step Bernoulli on the entire adapter.
    #[arg(long, default_value_t = 0.0)]
    module_dropout: f32,
    /// Rescale rank-mask by `1/mean(mask)` to preserve expectation.
    #[arg(long, default_value_t = false)]
    rank_dropout_scale: bool,

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

/// Self-adjusting shift based on image sequence length.
/// musubi-tuner `qwen_image_utils.py:956-959` — anchors 256→0.5, 8192→0.9, exp.
fn shift_for_resolution(resolution: [usize; 2]) -> f32 {
    const VAE_SCALE: usize = 8;
    const PATCH_SIZE: usize = 2;
    const BASE_SEQ_LEN: f32 = 256.0;
    const MAX_SEQ_LEN: f32 = 8192.0;
    const BASE_SHIFT: f32 = 0.5;
    const MAX_SHIFT: f32 = 0.9;
    let [w, h] = resolution;
    let seq_len = ((h / VAE_SCALE / PATCH_SIZE) * (w / VAE_SCALE / PATCH_SIZE)) as f32;
    let m = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ_LEN - BASE_SEQ_LEN);
    let b = BASE_SHIFT - m * BASE_SEQ_LEN;
    let mu = seq_len * m + b;
    mu.exp()
}

/// Sample logit-normal then apply qwen_shift remap. Matches musubi
/// `t = sigmoid(z); t = t*shift / (1 + (shift-1)*t)` byte-for-byte.
///
/// Superseded by the unified `TimestepConfig` dispatch (see `apply_qwen_shift`).
#[allow(dead_code)]
fn sample_timestep_logit_normal_qwenshift(rng: &mut StdRng, shift: f32) -> f32 {
    let t = trainer_common::sample_logit_normal_timestep(rng, 1, 0.0, 1.0, 1.0, 0.0, 1.0);
    let shifted = shift * t / (1.0 + (shift - 1.0) * t);
    // Clamp to OT's discrete grid {1/1000, ..., 1.0}. OT samples a discrete
    // integer in [0, 1000) then divides by 1000, so sigma == 0 never occurs.
    // Continuous sampling here can hit sigma == 0 (degenerate clean input)
    // → clamp to the OT minimum.
    shifted.clamp(1.0 / 1000.0, 1.0)
}

/// Apply qwen-shift remap and clamp. Caller passes `t` from the unified
/// `TimestepConfig::sample_one` (in `[0, 1]`).
fn apply_qwen_shift(t: f32, shift: f32) -> f32 {
    let shifted = shift * t / (1.0 + (shift - 1.0) * t);
    shifted.clamp(1.0 / 1000.0, 1.0)
}

/// Build the unified `TimestepConfig` from CLI args.
fn build_timestep_config(
    distribution: &str,
    weight: f32,
    bias: f32,
) -> anyhow::Result<TimestepConfig> {
    trainer_common::build_full_strength_timestep_config(distribution, weight, bias)
}

fn main() -> anyhow::Result<()> {
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
    // `--validation-prompts-file` is now consumed below for config-driven
    // sampling (see `sample_set`); no longer a no-op warn for this trainer.
    if args.masked_loss_weight > 0.0 {
        log::warn!(
            "[masked-loss] --masked-loss-weight={:.3} requested but Qwen-Image's prepare_qwenimage cache schema has no `latent_mask` field; flag is a no-op for this trainer.",
            args.masked_loss_weight
        );
    }
    let device = trainer_common::init_bf16_cuda();

    let shift = if args.dynamic_timestep_shifting {
        shift_for_resolution([args.resolution, args.resolution])
    } else {
        args.qwen_shift
    };
    log::info!(
        "[train_qwenimage] model={}, cache={}, steps={}, rank={}, alpha={}, lr={}, res={}², shift={:.3}",
        args.model.display(), args.cache_dir.display(),
        args.steps, args.rank, args.lora_alpha, args.lr, args.resolution, shift,
    );

    // ── Sample-setup MUST run before DiT load ────────────────────────────
    // Qwen2.5-VL is ~14 GB BF16 on GPU. After the DiT (60-block BlockOffloader
    // + activation pool) loads, GPU has ~5 GB free which can't fit the TE.
    // So pre-encode prompts NOW (TE only resident on GPU), drop the TE, then
    // load DiT into the freed memory. Mirrors train_ernie with order swapped.
    let periodic_sample = args.sample_every > 0;
    // Config-driven sample set. When `--validation-prompts-file` is present,
    // EVERY prompt (prompt + per-prompt negative) is encoded here and rendered
    // at each checkpoint AND final. Otherwise we fall back to the single/dual
    // --sample-prompt-N path below. (train_qwenimage is fully CLI-driven — it
    // does not load a TrainConfig — so there is no config fallback source.)
    // Each entry: (label "p{i+1}", encoded cap, encoded uncond opt).
    let prompts_file = args.validation_prompts_file.clone();
    // (label "p{i+1}", prompt text, encoded cap, encoded uncond opt).
    let mut sample_set: Vec<(String, String, Tensor, Option<Tensor>)> = Vec::new();
    let (sample_cond_1, sample_cond_2, sample_uncond, sample_vae_path) = if periodic_sample {
        let te_path = args
            .sample_text_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-text-encoder"))?;
        let tok_path = args
            .sample_tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-tokenizer"))?;
        let vae_path = args
            .sample_vae
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-vae"))?
            .clone();
        let p1 = args
            .sample_prompt_1
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-prompt-1"))?
            .clone();
        let p2 = args
            .sample_prompt_2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-prompt-2"))?
            .clone();

        log::info!("[sample-setup] loading Qwen2.5-VL + tokenizer for prompt pre-encode...");
        let tok = tokenizers::Tokenizer::from_file(tok_path)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let te_weights = load_te_weights(te_path, &device)?;
        let te_cfg = Qwen25VLEncoder::config_from_weights(&te_weights)?;
        let te = Qwen25VLEncoder::new(te_weights, te_cfg, device.clone());

        let encode_one = |text: &str| -> anyhow::Result<Tensor> {
            // Match prepare_qwenimage's PROMPT_TEMPLATE_ENCODE wrap +
            // DROP_IDX slice + trailing-pad trim. The DiT was trained
            // against variable-length text embeddings (per-prompt content
            // length); padding to a fixed `sample_max_text_len` would
            // pollute joint attention with junk pad-token hiddens. Mirrors
            // inference-flame's `qwenimage_encode::encode_and_trim`.
            let wrapped = format!("{PROMPT_PREFIX}{text}{PROMPT_SUFFIX}");
            let enc = tok
                .encode(wrapped, false)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let raw_ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            let work_len = args.sample_max_text_len + DROP_IDX;
            let mut ids: Vec<i32> = raw_ids.iter().take(work_len).copied().collect();
            let real_len_pre_pad = ids.len();
            ids.resize(work_len, QWEN_PAD_ID);
            let real_len = real_len_pre_pad.min(work_len);
            if real_len <= DROP_IDX {
                anyhow::bail!(
                    "sample prompt tokenized to only {real_len} ids; expected > {DROP_IDX} after PROMPT_TEMPLATE_ENCODE wrap"
                );
            }
            let kept_len = real_len - DROP_IDX;
            let full_hidden = te.encode(&ids)?.to_dtype(DType::BF16)?;
            full_hidden
                .narrow(1, DROP_IDX, kept_len)
                .map_err(|e| anyhow::anyhow!("narrow: {e}"))
        };
        let cond_1 = encode_one(&p1)?;
        let cond_2 = encode_one(&p2)?;
        let uncond = if args.sample_cfg > 1.0 {
            Some(encode_one(&args.sample_neg_prompt)?)
        } else {
            None
        };
        // Config-driven set: encode every (prompt, negative) pair from the
        // prompts file while the text encoder is still resident. Per-prompt
        // negative encoded only when CFG > 1.0 (mirrors uncond gating above).
        if let Some(ref pf) = prompts_file {
            let lib = SampleLibrary::from_file(pf)?;
            log::info!(
                "[sample-setup] {} config-driven prompt(s) from {}",
                lib.len(),
                pf.display()
            );
            for (i, p) in lib.prompts.iter().enumerate() {
                let label = format!("p{}", i + 1);
                let cap = encode_one(&p.prompt)?;
                let unc = if args.sample_cfg > 1.0 {
                    Some(encode_one(&p.negative)?)
                } else {
                    None
                };
                log::info!(
                    "[sample-setup] {label} encoded cap={:?}",
                    cap.shape().dims()
                );
                sample_set.push((label, p.prompt.clone(), cap, unc));
            }
        }
        // Fallback when no prompts file: render the two --sample-prompt-N
        // entries (preserves prior behaviour). The step-0 baseline path still
        // uses cond_1/cond_2 directly from the returned tuple.
        if sample_set.is_empty() {
            sample_set.push(("p1".into(), p1.clone(), cond_1.clone(), uncond.clone()));
            sample_set.push(("p2".into(), p2.clone(), cond_2.clone(), uncond.clone()));
        }
        log::info!(
            "[sample-setup] dropping text encoder; cond_1={:?} cond_2={:?}{}",
            cond_1.shape().dims(),
            cond_2.shape().dims(),
            uncond
                .as_ref()
                .map(|u| format!(", uncond={:?}", u.shape().dims()))
                .unwrap_or_default(),
        );
        drop(te);
        flame_core::cuda_alloc_pool::clear_pool_cache();
        flame_core::trim_cuda_mempool(0);
        log::info!(
            "[sample-setup] periodic sample enabled (every {} steps; 2 prompts).",
            args.sample_every
        );
        (Some(cond_1), Some(cond_2), uncond, Some(vae_path))
    } else {
        (None, None, None, None)
    };

    // Phase 2b: parse the LyCORIS algo selector. `lora` (default) keeps the
    // legacy LoRALinear bundle constructed inside `QwenImageTrainingModel::load`.
    // Anything else swaps the bundle in-place after model construction so we
    // don't have to re-plumb the per-trainer constructor signatures.
    //
    // NOTE: `LycorisAlgo::parse("lora")` aliases to `LycorisAlgo::LoCon`
    // (since LoCon-Linear is the canonical LoRA decomposition). For
    // qwenimage we need to distinguish LEGACY plain `LoRALinear`
    // (byte-identical) from the new `LycorisAdapter::LoCon` path, so re-map
    // `"lora"` → `None` here explicitly. Users who want the new LoCon path
    // pass `--algo locon`.
    let algo_str = args.algo.trim().to_ascii_lowercase();
    let algo = if algo_str == "lora" || algo_str == "none" || algo_str.is_empty() {
        LycorisAlgo::None
    } else {
        LycorisAlgo::parse(&args.algo).map_err(|e| anyhow::anyhow!("--algo: {e}"))?
    };
    // Default storage (F32) inherited from `LycorisBundleConfig::default()`.
    // qwenimage trainer is BF16/F32-only; do NOT switch to FP8.
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

    log::info!("Loading Qwen-Image transformer...");
    let mut model = QwenImageTrainingModel::load(
        &args.model,
        args.rank,
        args.lora_alpha,
        /*full_finetune*/ false,
        device.clone(),
        args.seed,
    )?;

    // If a LyCORIS algo other than the legacy plain LoRA was requested, swap
    // the bundle. Plain `--algo lora` (or `lora`/`none`) keeps the legacy
    // bundle as-is so this branch is byte-equivalent to the pre-Phase-2b
    // pipeline.
    if algo != LycorisAlgo::None {
        log::info!(
            "[qwenimage] LyCORIS algo='{}' rank={} alpha={} factor={} block_size={} dora={}",
            algo.as_str(),
            lyc_config.rank,
            lyc_config.alpha,
            lyc_config.factor,
            lyc_config.block_size,
            lyc_config.dora,
        );
        if matches!(algo, LycorisAlgo::Full | LycorisAlgo::Oft) {
            log::warn!(
                "[qwenimage] algo='{}' selected — bundle construction will succeed, but \
                 forward_delta will error inside qwenimage's `base + delta_on_input` call \
                 pattern. Phase 2c will wire merge-into-base for these algos.",
                algo.as_str()
            );
        }
        let new_bundle =
            eridiffusion_core::models::qwenimage::QwenImageLoraBundle::new_with_config(
                &lyc_config,
                device.clone(),
                args.seed,
            )
            .map_err(|e| anyhow::anyhow!("LyCORIS bundle construction: {e}"))?;
        model.bundle = new_bundle;

        // SimpleTuner-style perturbed-normal LoKr init (Phase 2b: warns and
        // no-ops because qwenimage's base weights are streamed via the
        // BlockOffloader and not resident in a single map at this point).
        if matches!(algo, LycorisAlgo::LoKr) && args.init_lokr_norm > 0.0 {
            // `apply_init_perturbed_normal` walks LoKr adapters and looks up
            // their base weights in the provided map. We pass an empty map
            // here because qwenimage's `block_weights` are owned inside
            // `QwenImageTrainingModel` and are streamed at runtime; the
            // bundle method already logs a clear warning when the requested
            // scale is non-zero on qwenimage so users aren't silently
            // surprised.
            let empty: std::collections::HashMap<String, flame_core::Tensor> =
                std::collections::HashMap::new();
            model
                .bundle
                .apply_init_perturbed_normal(&empty, args.init_lokr_norm)
                .map_err(|e| anyhow::anyhow!("apply_init_perturbed_normal: {e}"))?;
        }
    } else {
        // Explicit log: legacy path — no swap.
        log::info!("[qwenimage] algo='lora' (legacy LoRALinear path, byte-identical)");
    }
    // Phase 2b: dropout flags (`--lora_dropout`, `--rank_dropout`,
    // `--module_dropout`, `--rank_dropout_scale`) are accepted on the CLI but
    // not yet wired into qwenimage's `add_lora` dispatch — Phase 2c will
    // route them through the adapter's `forward_delta` per-step. Default
    // values are `0.0`/`false`, matching pre-Phase-2b byte-identical behaviour.
    if args.lora_dropout > 0.0
        || args.rank_dropout > 0.0
        || args.module_dropout > 0.0
        || args.rank_dropout_scale
    {
        log::warn!(
            "[qwenimage] dropout flags (lora_dropout={}, rank_dropout={}, \
             module_dropout={}, rank_dropout_scale={}) are accepted but not yet \
             wired in qwenimage Phase 2b — they are no-ops for this run.",
            args.lora_dropout,
            args.rank_dropout,
            args.module_dropout,
            args.rank_dropout_scale,
        );
    }

    // Qwen uses boundary checkpointing in qwenimage.rs: only block inputs are
    // pushed to a grow-on-demand activation cache, then backward pulls them
    // and recomputes the block. This mirrors OneTrainer's conductor model
    // more closely than the legacy sub-tape pool, which tried to offload
    // hundreds of internal saved tensors per block and fell back after pool
    // exhaustion, paying an extra forward pass before the real checkpoint.
    let _activation_cache = {
        use eridiffusion_core::training::offload::setup_grow_activation_cache;
        let pack_h = args.resolution / 8 / 2;
        let pack_w = args.resolution / 8 / 2;
        let max_seq = pack_h * pack_w + 512;
        let boundary_bytes_per_block = max_seq * qwen_model::DIM * 2;
        let slab_bytes = 1usize << 30;
        match setup_grow_activation_cache(&device, slab_bytes) {
            Ok(cache) => {
                log::info!(
                    "[activation_offload] grow cache installed (slab={} MB, boundary≈{:.1}MB/block)",
                    slab_bytes / (1024 * 1024),
                    boundary_bytes_per_block as f64 / 1e6
                );
                Some(cache)
            }
            Err(e) => {
                log::warn!(
                    "[activation_offload] grow cache setup failed ({e}); falling back to plain checkpoint"
                );
                None
            }
        }
    };
    let mut params = model.parameters();
    // Gate-on 6a: under v2 (default), flip LoRA params to MatchParamDtype so
    // BF16 grads from the bridge stay BF16 (Class A). --use-autograd-v3 skips.
    trainer_pipeline::apply_autograd_v2_grad_policy(&mut params, args.use_autograd_v3, "params");
    log::info!("Loaded {} trainable LoRA tensors", params.len());
    if params.is_empty() {
        anyhow::bail!("No trainable parameters — model produced empty param list");
    }

    // Cache enumeration.
    trainer_common::ensure_output_dir(&args.output_dir)?;
    let cache_files = trainer_common::list_cache_safetensors(&args.cache_dir)?;
    log::info!("Found {} cached samples", cache_files.len());

    // Sentinel check: prepare_qwenimage writes `_meta.json` with the prep
    // resolution + max_text_len. Warn loud if the cache was produced at a
    // different resolution than `--resolution`. Legacy caches without the
    // sentinel proceed silently (user is on their own for cross-resolution
    // contamination).
    let meta_path = args.cache_dir.join("_meta.json");
    if meta_path.exists() {
        if let Ok(s) = std::fs::read_to_string(&meta_path) {
            let res_match = s.contains(&format!("\"resolution\": {}", args.resolution));
            if !res_match {
                log::warn!(
                    "[cache-meta] {} prep settings = {} — but trainer --resolution = {}. Possible OOD cache reuse.",
                    meta_path.display(), s.trim(), args.resolution
                );
            }
        }
    } else {
        log::debug!("[cache-meta] no _meta.json sentinel — legacy cache or hand-managed");
    }

    // Validation harness — held-out cache + cadence. None at default (byte-identity).
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

    let opt_kind =
        OptimizerKind::parse(&args.optimizer).map_err(|e| anyhow::anyhow!("--optimizer: {e}"))?;
    log::info!("[Qwen-Image] optimizer={}", opt_kind.as_str());
    // Phase 1: caption_dropout. Qwen-Image has no inline encoder, so the user
    // supplies a `--null-text-cache` produced by `prepare_qwenimage` on a
    // single empty-caption sample. We load `text_embedding` once and swap it
    // in per-step with the configured probability. Without `--null-text-cache`,
    // the feature is disabled with a warning.
    let mut effective_caption_dropout_prob = args.caption_dropout_probability;
    let null_text: Option<Tensor> = if effective_caption_dropout_prob > 0.0 {
        match args.null_text_cache.as_ref() {
            Some(p) => match flame_core::serialization::load_file(p, &device) {
                Ok(s) => {
                    let nt = s
                        .get("text_embedding")
                        .ok_or_else(|| {
                            anyhow::anyhow!("--null-text-cache missing 'text_embedding'")
                        })?
                        .to_dtype(DType::BF16)?;
                    log::info!(
                        "[caption-dropout] WIRED — prob={:.3} (null_text_embedding={:?})",
                        effective_caption_dropout_prob,
                        nt.shape().dims()
                    );
                    Some(nt)
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
    let mut optimizer = Optimizer::new(opt_kind, args.lr, 0.9, 0.999, 1e-8, 0.01);

    // EMA shadow (Phase 3 advanced). Built from current live params (post
    // resume_lora / pre-step-0). Updated after each opt.step via
    // `update_with_schedule`. Optional swap into live params at sample /
    // checkpoint time when --ema-validation-swap is set.
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
            "[ema] WIRED — {} shadow tensors, inv_gamma={} power={} update_after_step={} min_decay={} max_decay={} validation_swap={}",
            e.len(),
            ema_cfg.inv_gamma,
            ema_cfg.power,
            ema_cfg.update_after_step,
            ema_cfg.min_decay,
            ema_cfg.max_decay,
            args.ema_validation_swap,
        );
        Some(e)
    } else {
        None
    };

    // Timestep bias config — defaults are byte-identical (Strategy::None).
    // qwenimage operates in continuous-sigma space [0, 1] (logit-normal then
    // qwen_shift remap), so we pass `total = 1.0` to apply_bias rather than
    // the `NUM_TRAIN_TIMESTEPS = 1000` convention used by Klein/Z-Image.
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

    let mut start_step = 0usize;

    // Resume.
    if let Some(path) = &args.resume_lora {
        log::info!("Resuming LoRA weights from {}", path.display());
        let loaded = checkpoint::load_full(path, &device)?;
        let named = qwen_named_parameters(&model);
        checkpoint::apply_lora_weights(&loaded, &named)?;
    } else if let Some(path) = &args.resume_full {
        log::info!("Full-resume from {}", path.display());
        let loaded = checkpoint::load_full(path, &device)?;
        let named = qwen_named_parameters(&model);
        checkpoint::apply_lora_weights(&loaded, &named)?;
        if let Optimizer::AdamW(ref mut adam) = optimizer {
            checkpoint::apply_to_optimizer(&loaded, adam, &named, args.rank, args.lora_alpha)?;
        } else {
            log::warn!(
                "[resume-full] non-AdamW resume not yet implemented for {:?}; LoRA weights restored, optimizer state reset",
                optimizer.kind()
            );
        }
        start_step = loaded.header.step as usize;
        if start_step >= args.steps {
            log::warn!(
                "Resumed step {} >= --steps {}; nothing to do.",
                start_step,
                args.steps
            );
            return Ok(());
        }
        log::info!("Continuing from step {}/{}", start_step, args.steps);
    }

    // No step-0 / no in-loop sampling — per "do it once, like inference".
    // The single sample lands AFTER training completes, with the activation
    // offload pool torn down so the 1024² VAE decode has full GPU.

    let clipper = GradientClipper::clip_by_norm(1.0);
    let mut rng = StdRng::seed_from_u64(args.seed);
    let board = trainer_common::open_board_writer(
        &args.output_dir,
        trainer_common::board_resume_step(start_step),
    );
    if let Some(b) = &board {
        // Full board wiring: run hyper-parameters → metadata.hparams + the
        // dashboard's hparam panel. JSON hand-built (no serde_json dep here).
        let hparams_json = format!(
            "{{\"model\":\"qwenimage\",\"steps\":{},\"rank\":{},\"lora_alpha\":{},\"lr\":{},\
             \"warmup_steps\":{},\"batch_size\":{},\"optimizer\":\"{}\",\"timestep_shift\":{},\
             \"sample_size\":{},\"sample_steps\":{},\"sample_cfg\":{},\"seed\":{}}}",
            args.steps,
            args.rank,
            args.lora_alpha,
            args.lr,
            args.warmup_steps,
            1,
            opt_kind.as_str(),
            shift,
            args.sample_size,
            args.sample_steps,
            args.sample_cfg,
            args.seed
        );
        b.log_hparams(&hparams_json, &[("steps_target", args.steps as f64)]);
    }
    let loop_config = trainer_pipeline::ManualTrainLoopConfig::new(
        "QwenImage-lora",
        start_step,
        args.steps,
        cache_files.len(),
        1,
    );
    let mut loop_run = trainer_pipeline::ManualTrainLoopRun::new(loop_config);

    // Step-0 baseline sample: LoRA = identity, so this is the base model
    // output. Rendered at 512² to avoid tearing down the 19 GB activation
    // pool mid-process (FINAL sample handles teardown since process exits).
    if periodic_sample && start_step == 0 {
        let cond_1 = sample_cond_1.as_ref().unwrap();
        let cond_2 = sample_cond_2.as_ref().unwrap();
        let uncond = sample_uncond.as_ref();
        let vae_path = sample_vae_path.as_ref().unwrap();
        for (idx, cond) in [cond_1, cond_2].iter().enumerate() {
            let out_path = args
                .output_dir
                .join(format!("sample_step0_base_p{}.png", idx + 1));
            log::info!(
                "[sample step=0 BASELINE p{}] → {}",
                idx + 1,
                out_path.display()
            );
            if let Err(e) = qwenimage_inline_sample(
                &mut model,
                cond,
                uncond,
                vae_path,
                &out_path,
                512,
                args.sample_steps,
                args.sample_cfg,
                args.sample_seed,
                &device,
            ) {
                log::warn!("[sample step=0 p{}] failed: {e}", idx + 1);
            }
            flame_core::cuda_alloc_pool::clear_pool_cache();
            flame_core::trim_cuda_mempool(0);
        }
    }

    log::info!("Training {} steps from step={}", args.steps, start_step);
    let sched: LrScheduler = args.lr_scheduler.parse().unwrap_or_else(|e: String| {
        log::warn!("[lr_scheduler] {e} — falling back to Constant");
        LrScheduler::Constant
    });
    for step in loop_run.steps() {
        // Phase 5: dispatch via LrScheduler enum. Default `Constant` is
        // byte-equivalent to the legacy hand-rolled linear-warmup-then-flat path.
        let current_lr = lr_schedule::dispatch_lr(
            &sched,
            args.lr,
            step,
            args.steps,
            args.warmup_steps,
            args.lr_min_factor,
            args.lr_cycles,
        );
        optimizer.set_lr(current_lr);

        // Load one cached sample.
        let idx = step % cache_files.len();
        let tensors = flame_core::serialization::load_file(&cache_files[idx], &device)?;
        let latent = tensors
            .get("latent")
            .ok_or_else(|| anyhow::anyhow!("cache missing 'latent'"))?
            .to_dtype(DType::BF16)?;
        let txt_embed = tensors
            .get("text_embedding")
            .ok_or_else(|| anyhow::anyhow!("cache missing 'text_embedding'"))?
            .to_dtype(DType::BF16)?;
        // Caption dropout: single Bernoulli per step swaps text_embedding
        // with null cache. Default-off (prob == 0.0 OR null_text == None)
        // draws no rng.
        let txt_embed = if let Some(ref nt) = null_text {
            caption_dropout::maybe_drop_caption(
                &txt_embed,
                nt,
                effective_caption_dropout_prob,
                &mut rng,
            )?
        } else {
            txt_embed
        };

        let lat_dims = latent.shape().dims().to_vec();
        let (b, _c, latent_h, latent_w) = (lat_dims[0], lat_dims[1], lat_dims[2], lat_dims[3]);
        let _ = b;

        // Sample timestep with qwen_shift.
        let raw_t = apply_qwen_shift(timestep_cfg.sample_one(&mut rng), shift);
        // Default-off: Strategy::None → returns raw_t unchanged. qwenimage
        // sigma is already in [0, 1], so we pass total=1.0.
        let t_continuous = timestep_bias::apply_bias(raw_t, 1.0, &timestep_bias_cfg);
        // OT parity (Q-C2/Q-C3): BaseQwenSetup.py:110 calls _get_timestep_discrete which
        // returns timestep.int() (ModelSetupNoiseMixin.py:212). OT model gets
        // `timestep / 1000` where timestep is the INTEGER index ∈ [0,999].
        // Our t_continuous ∈ [0,1] → discretize to integer bin, derive sigma and
        // model timestep from the same discrete index. Matches _add_noise_discrete:
        //   sigma = (sigma_idx + 1) / 1000  (FlowMatchingMixin.py:23-24)
        //   model input = sigma_idx / 1000  (BaseQwenSetup.py:142)
        const NUM_TRAIN_TIMESTEPS_QWEN: usize = 1000;
        let sigma_idx = ((t_continuous * NUM_TRAIN_TIMESTEPS_QWEN as f32).floor() as usize)
            .min(NUM_TRAIN_TIMESTEPS_QWEN - 1);
        let sigma = (sigma_idx + 1) as f32 / NUM_TRAIN_TIMESTEPS_QWEN as f32;
        let model_t = sigma_idx as f32 / NUM_TRAIN_TIMESTEPS_QWEN as f32;
        let timestep = Tensor::from_vec(vec![model_t], Shape::from_dims(&[1]), device.clone())?
            .to_dtype(DType::BF16)?;

        // Flow-matching: x_t = (1 - sigma) * latent + sigma * noise; target = noise - latent.
        let latent_f32 = latent.to_dtype(DType::F32)?;
        let noise = noise_modifiers::randn_f32(latent_f32.shape().clone(), device.clone())?;
        // Multi-resolution noise (default-off). When iterations==0 returns
        // noise.clone() with no rng draw — byte-identical to baseline.
        let noise = noise_modifiers::maybe_apply_multires_noise(
            &noise,
            args.multires_noise_iterations,
            args.multires_noise_discount,
            &mut rng,
        )?;
        // Phase 1: noise modifiers (default-off). Qwen-Image trainer doesn't
        // load TrainConfig JSON — `offset_noise_weight` is hardcoded 0.0.
        // Offset noise is part of the clean noise distribution; input
        // perturbation feeds model input only.
        let clean_noise = noise_modifiers::maybe_apply_offset_noise(
            &noise,
            0.0,
            args.noise_offset_probability,
            &mut rng,
        )?;
        let perturbed_noise = noise_modifiers::maybe_apply_input_perturbation(
            &clean_noise,
            args.gamma_input_perturbation,
            &mut rng,
        )?;
        let xt_f32 = latent_f32
            .mul_scalar(1.0 - sigma)?
            .add(&perturbed_noise.mul_scalar(sigma)?)?;
        let xt = xt_f32.to_dtype(DType::BF16)?;
        let target = clean_noise.sub(&latent_f32)?;

        // Pack [B, 16, H, W] → [B, H/2 * W/2, 64] for forward.
        let xt_packed = qwen_model::pack_latents(&xt)?;
        let target_packed = qwen_model::pack_latents(&target)?;

        // Forward.
        AutogradContext::clear();
        let pred = model.forward(&xt_packed, &timestep, &txt_embed, latent_h, latent_w)?;

        // MSE loss in F32.
        // Phase 1: combined loss + per-step weighting. Default-off invariant.
        let pred_f32 = pred.to_dtype(DType::F32)?;
        let target_f32 = target_packed.to_dtype(DType::F32)?;
        let raw_loss =
            loss_weight::combined_loss(&pred_f32, &target_f32, 1.0, 0.0, args.huber_strength)?;
        let loss = loss_weight::apply_loss_weight(
            &raw_loss,
            sigma,
            eridiffusion_core::config::LossWeight::Constant,
            args.min_snr_gamma,
            true,
        )?;

        let loss_val: f32 = loss.to_vec()?.first().copied().unwrap_or(f32::NAN);
        if !loss_val.is_finite() {
            anyhow::bail!("step {}: non-finite loss {}", step + 1, loss_val);
        }

        // Backward.
        // Phase 5b / gate-on 6a: Route (ii) bridge. v2 is the default; backward
        // goes through `backward_v2` unless `--use-autograd-v3` opts into v3.
        let grads = trainer_pipeline::backward_loss(&loss, args.use_autograd_v3)?;

        // Step-1 grad-flow diagnostic.  Catches the recurring "BF16 fused
        // inference op missing autograd registration" bug class before it
        // wastes a 3000-step run.  Returns a report; only panics when
        // `FLAME_ASSERT_GRAD_FLOW=1` is set.  Note: qwen-image has 4
        // architecturally-zero block-59 txt-stream params (`add_q_proj`,
        // `to_add_out`, `txt_mlp.net.0.proj`, `txt_mlp.net.2`) that will
        // legitimately appear in the report — do NOT enable the panic flag
        // for qwen; use it on klein/sdxl/etc. where 0-dead is the contract.
        // Step 1 (NOT step 0): LoRA-style algos init one factor at zero so
        // half the leaves have zero grad at step 0 by construction.  Step 1
        // (after the first optimizer step has moved them off zero) is the
        // earliest meaningful check.
        if step == 1 {
            let named = qwen_named_parameters(&model);
            let named_refs: Vec<(&str, &flame_core::parameter::Parameter)> =
                named.iter().map(|(n, p)| (n.as_str(), p)).collect();
            let report = flame_core::diagnostics::assert_grad_flow(&grads, &named_refs)?;
            if report.is_clean() {
                log::info!("[grad-flow] step 2 clean ({} params)", report.ok_count);
            } else {
                log::warn!("{}", report.summary());
            }
        }

        // Per-tensor non-finite-grad guard.  Some early-step / extreme-sigma
        // combinations push `attn.to_out.0` post-SDPA activations to BF16-
        // overflow magnitudes (block 1's `img_attn` saw max_abs=4096 in
        // boxjana 512² runs).  The corresponding lora_B gradient blows up
        // (lora_B grad = (input @ A^T)^T @ grad_out, A is large-init), and
        // the upstream gradient that feeds block 0's lora_B picks up NaN.
        // Without per-tensor zeroing, the global-norm clipper still passes
        // NaN-element grads through (clamp leaves NaN → NaN), poisoning the
        // AdamW first/second moments forever.  Mirrors PyTorch's
        // `clip_grad_norm_(error_if_nonfinite=False)` behavior of dropping
        // the non-finite tensor's update for that step.
        let mut nan_skipped = 0usize;
        let mut nan_names: Vec<String> = Vec::new();
        let names_by_id = if step == 0 || step % 100 == 99 {
            let mut names = std::collections::HashMap::new();
            for (name, param) in qwen_named_parameters(&model) {
                names.insert(param.id(), name);
            }
            Some(names)
        } else {
            None
        };
        for param in params.iter() {
            if let Some(g) = grads.get(param.id()) {
                let g = if g.dtype() == DType::F32 {
                    g.clone()
                } else {
                    g.to_dtype(DType::F32)?
                };
                let g_vec = g.to_vec()?;
                let any_bad = g_vec.iter().any(|x| !x.is_finite());
                if any_bad {
                    nan_skipped += 1;
                    if let Some(ref ns) = names_by_id {
                        if let Some(n) = ns.get(&param.id()) {
                            nan_names.push(n.clone());
                        }
                    }
                    let zero_g = g.zeros_like_with_dtype(DType::F32)?;
                    param.set_grad(zero_g)?;
                } else {
                    param.set_grad(g)?;
                }
            }
        }
        if nan_skipped > 0 {
            if names_by_id.is_some() {
                log::warn!(
                    "[grad-guard] step={} zeroed {} non-finite-grad params: {:?}",
                    step + 1,
                    nan_skipped,
                    nan_names
                );
            } else {
                log::warn!(
                    "[grad-guard] step={} zeroed {} non-finite-grad params",
                    step + 1,
                    nan_skipped
                );
            }
        }

        // Clip + step.
        let grad_norm = trainer_pipeline::clip_parameter_slot_grads(&params, &clipper)?;

        trainer_pipeline::step_optimizer(&mut optimizer, &params, current_lr, || {
            model.refresh_lora_cache();
            if let Some(ref mut e) = ema {
                e.update_with_schedule(&params, &ema_cfg, (step + 1) as u64)
                    .map_err(|err| {
                        anyhow::anyhow!("EMA update failed at step {}: {err}", step + 1)
                    })?;
            }
            Ok(())
        })?;
        flame_core::cuda_alloc_pool::clear_pool_cache();

        let step_num = step + 1;
        loop_run.record_and_log(
            step,
            trainer_pipeline::TrainStepMetrics {
                loss_value: loss_val,
                grad_norm,
                learning_rate: current_lr,
            },
            board.as_ref(),
        );

        // Validation eval pass (no_grad) every `validation_every_steps`.
        // step+1 because `step` here is 0-based; ValidationLoop::should_run
        // expects the 1-based completed-step number.
        //
        // Mirrors the training step's math:
        //   - logit-normal then qwen_shift remap (same `shift` as training)
        //   - sigma == t_continuous (Qwen-Image works in continuous [0,1])
        //   - flow-matching x_t = (1-sigma)*lat + sigma*noise; tgt = noise-lat
        //   - pack [B,16,H,W]→[B,H/2*W/2,64]; forward(xt_packed, ts, txt, H, W)
        // Side-RNG is `args.seed ^ (step+1)` so this never perturbs the
        // training-side rng draws (byte invariance for default-off).
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
                    let v_lat = match sample.get("latent") {
                        Some(t) => t.to_dtype(DType::BF16)?,
                        None => {
                            log::warn!("[validation] {} missing latent", vfile.display());
                            continue;
                        }
                    };
                    let v_txt = match sample.get("text_embedding") {
                        Some(t) => t.to_dtype(DType::BF16)?,
                        None => {
                            log::warn!("[validation] {} missing text_embedding", vfile.display());
                            continue;
                        }
                    };
                    let v_dims = v_lat.shape().dims().to_vec();
                    let (v_lat_h, v_lat_w) = (v_dims[2], v_dims[3]);

                    let mut vrng = rand::rngs::StdRng::seed_from_u64(args.seed ^ (step as u64 + 1));
                    let v_sigma = apply_qwen_shift(timestep_cfg.sample_one(&mut vrng), shift);
                    let v_timestep =
                        Tensor::from_vec(vec![v_sigma], Shape::from_dims(&[1]), device.clone())?
                            .to_dtype(DType::BF16)?;
                    let v_noise = Tensor::randn(v_lat.shape().clone(), 0.0, 1.0, device.clone())?
                        .to_dtype(DType::BF16)?;
                    let v_xt = v_lat
                        .mul_scalar(1.0 - v_sigma)?
                        .add(&v_noise.mul_scalar(v_sigma)?)?;
                    let v_target = v_noise.sub(&v_lat)?;

                    let v_xt_packed = match qwen_model::pack_latents(&v_xt) {
                        Ok(t) => t,
                        Err(e) => {
                            log::warn!("[validation] pack_latents failed: {e}");
                            continue;
                        }
                    };
                    let v_target_packed = match qwen_model::pack_latents(&v_target) {
                        Ok(t) => t,
                        Err(e) => {
                            log::warn!("[validation] pack_latents (target) failed: {e}");
                            continue;
                        }
                    };
                    let v_pred =
                        match model.forward(&v_xt_packed, &v_timestep, &v_txt, v_lat_h, v_lat_w) {
                            Ok(p) => p,
                            Err(e) => {
                                log::warn!("[validation] forward failed: {e}");
                                continue;
                            }
                        };
                    let v_loss = v_pred
                        .to_dtype(DType::F32)?
                        .sub(&v_target_packed.to_dtype(DType::F32)?)?
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

        if trainer_common::cadence_fires(args.save_every, step_num, args.steps) {
            trainer_pipeline::with_optional_ema_swap(
                ema.as_ref(),
                &params,
                args.ema_validation_swap,
                "mid",
                || {
                    let path = args
                        .output_dir
                        .join(format!("qwenimage_lora_step{}.safetensors", step_num));
                    save_ckpt(
                        &path,
                        &model,
                        &optimizer,
                        args.rank,
                        args.lora_alpha,
                        args.seed,
                        &args.save_mode,
                        step_num,
                    )?;
                    Ok(())
                },
            )?;
        }

        // Periodic in-loop sample: both prompts at 512² (1024² would need
        // activation pool teardown, which is unsafe mid-training).
        if periodic_sample && trainer_common::cadence_fires(args.sample_every, step_num, args.steps) {
            let vae_path = sample_vae_path.as_ref().unwrap();
            // Render EVERY config-driven prompt in `sample_set` (loaded from the
            // prompts file, or the --sample-prompt-N fallback). Each →
            // sample_step{N}_{label}.png + board image/text.
            for (label, ptext, cond, unc) in &sample_set {
                let out_path = args
                    .output_dir
                    .join(format!("sample_step{}_{label}.png", step_num));
                log::info!(
                    "[sample step={} {label}] → {}",
                    step_num,
                    out_path.display()
                );
                if let Err(e) = qwenimage_inline_sample(
                    &mut model,
                    cond,
                    unc.as_ref(),
                    vae_path,
                    &out_path,
                    512,
                    args.sample_steps,
                    args.sample_cfg,
                    args.sample_seed,
                    &device,
                ) {
                    log::warn!("[sample step={} {label}] failed: {e}", step_num);
                } else if let Some(b) = &board {
                    b.log_image_png(&format!("samples/{label}"), step_num as u64, 0, &out_path);
                    b.log_text(&format!("prompts/{label}"), step_num as u64, ptext);
                }
                flame_core::cuda_alloc_pool::clear_pool_cache();
                flame_core::trim_cuda_mempool(0);
            }
        }
    }

    // Final EMA swap (covers both final save and final sample below). No
    // restore at the very end — process exits, no further training. Skipped
    // when --ema-validation-swap is off or no EMA was constructed.
    trainer_pipeline::swap_ema_for_final_save(ema.as_ref(), &params, args.ema_validation_swap)?;

    let final_path = args
        .output_dir
        .join(format!("qwenimage_lora_{}steps.safetensors", args.steps));
    save_ckpt(
        &final_path,
        &model,
        &optimizer,
        args.rank,
        args.lora_alpha,
        args.seed,
        &args.save_mode,
        args.steps,
    )?;

    // Final sample after training — single pass, both prompts, 1024².
    //
    // Tear down the activation offload pool first: it holds ~19 GB of GPU
    // staging buffers that the VAE decoder needs for the 1024² mid-block
    // attention. With the pool gone, the 1024² decode fits on 24 GB GPU.
    // (Once training is complete the pool isn't needed anymore.)
    if periodic_sample {
        log::info!(
            "[sample FINAL] tearing down activation offload pool to free GPU for 1024² decode..."
        );
        flame_core::autograd::clear_activation_offload_pool();
        flame_core::cuda_alloc_pool::clear_pool_cache();
        flame_core::trim_cuda_mempool(0);

        let vae_path = sample_vae_path.as_ref().unwrap();
        // Final sample: render EVERY config-driven prompt at full --sample-size.
        for (label, ptext, cond, unc) in &sample_set {
            let out_path = args
                .output_dir
                .join(format!("sample_step{}_final_{label}.png", args.steps));
            log::info!(
                "[sample FINAL step={} {label}] → {}",
                args.steps,
                out_path.display()
            );
            if let Err(e) = qwenimage_inline_sample(
                &mut model,
                cond,
                unc.as_ref(),
                vae_path,
                &out_path,
                args.sample_size,
                args.sample_steps,
                args.sample_cfg,
                args.sample_seed,
                &device,
            ) {
                log::warn!("[sample FINAL {label}] failed: {e}");
            } else if let Some(b) = &board {
                b.log_image_png(&format!("samples/{label}"), args.steps as u64, 0, &out_path);
                b.log_text(&format!("prompts/{label}"), args.steps as u64, ptext);
            }
            flame_core::cuda_alloc_pool::clear_pool_cache();
            flame_core::trim_cuda_mempool(0);
        }
    }
    let completion = loop_run.finish();
    log::info!(
        "Training complete: {} new steps (total {}). avg loss={:.4}. Saved to {}",
        completion.trained_steps,
        args.steps,
        completion.average_loss,
        final_path.display(),
    );
    trainer_pipeline::mark_board_completed(board.as_ref());
    Ok(())
}

/// Build the canonical `(name, Parameter)` pairs for `checkpoint::save_full`
/// and `apply_to_optimizer`. The names match `QwenImageLoraBundle::save`
/// byte-for-byte so resumed AdamW state lines up with live params.
///
/// Iteration order: deterministic (sorted by `(block_idx, target_suffix)`).
/// Required because `HashMap` iteration is random across runs and
/// save→reload must produce a stable key→tensor mapping.
fn qwen_named_parameters(
    model: &QwenImageTrainingModel,
) -> Vec<(String, flame_core::parameter::Parameter)> {
    use eridiffusion_core::adapter::AdapterModule;
    use eridiffusion_core::models::qwenimage::QwenImageLoraBundle;

    // Legacy plain-LoRA path — byte-identical to pre-Phase-2b.
    let mut entries: Vec<((usize, &str), &eridiffusion_core::lora::LoRALinear)> = model
        .bundle
        .adapters
        .iter()
        .map(|(&(idx, target), lora)| ((idx, QwenImageLoraBundle::target_suffix(target)), lora))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut out: Vec<(String, flame_core::parameter::Parameter)> =
        Vec::with_capacity(entries.len() * 2);
    for ((block_idx, suffix), lora) in entries {
        let prefix = format!("transformer_blocks.{block_idx}.{suffix}");
        // LoRALinear::parameters() returns [lora_a, lora_b]; the safetensors
        // save scheme is `{prefix}.lora_A.weight` then `{prefix}.lora_B.weight`.
        out.push((format!("{prefix}.lora_A.weight"), lora.lora_a().clone()));
        out.push((format!("{prefix}.lora_B.weight"), lora.lora_b().clone()));
    }

    // LyCORIS adapter path. When `--algo lora` (default) this map is empty and
    // the loop is a no-op, preserving byte-identical legacy behaviour.
    let mut lyc_entries: Vec<((usize, &str), &eridiffusion_core::adapter::LycorisLinear)> = model
        .bundle
        .lycoris_adapters
        .iter()
        .map(|(&(idx, target), arc)| {
            (
                (idx, QwenImageLoraBundle::target_suffix(target)),
                arc.as_ref(),
            )
        })
        .collect();
    lyc_entries.sort_by(|a, b| a.0.cmp(&b.0));
    for ((block_idx, suffix), adapter) in lyc_entries {
        let prefix = format!("transformer_blocks.{block_idx}.{suffix}");
        // `to_parameters()` and `named_tensors()` are zipped per the
        // `AdapterModule` contract — same length, same order.
        let params = adapter.to_parameters();
        let names = adapter.named_tensors();
        for ((leaf, _), p) in names.into_iter().zip(params.into_iter()) {
            out.push((format!("{prefix}.{leaf}"), p));
        }
    }
    out
}

/// In-process sample call. Pre-encoded prompts skip the (huge) text-encoder
/// load. Disables FLAME_CHECKPOINT inside the sampler scope (no autograd
/// during inference) and restores it on exit so training continues with
/// the same recompute behavior.
fn qwenimage_inline_sample(
    model: &mut QwenImageTrainingModel,
    cond: &Tensor,
    uncond: Option<&Tensor>,
    vae_path: &std::path::Path,
    out_path: &std::path::Path,
    size: usize,
    steps: usize,
    cfg: f32,
    seed: u64,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<()> {
    qwenimage_sampler::sample_image(
        model, cond, uncond, size, size, steps, cfg, seed, vae_path, out_path, device,
    )
    .map_err(|e| anyhow::anyhow!("qwenimage sample: {e}"))?;
    Ok(())
}

/// Load Qwen2.5-VL text encoder weights from a directory of shards or a
/// single combined safetensors file.
fn load_te_weights(
    path: &std::path::Path,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<std::collections::HashMap<String, Tensor>> {
    if path.is_file() {
        return flame_core::serialization::load_file(path, device)
            .map_err(|e| anyhow::anyhow!("text-encoder load: {e}"));
    }
    let mut all = std::collections::HashMap::new();
    for entry in
        std::fs::read_dir(path).map_err(|e| anyhow::anyhow!("read_dir {}: {e}", path.display()))?
    {
        let p = entry.map_err(|e| anyhow::anyhow!("entry: {e}"))?.path();
        if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            let part = flame_core::serialization::load_file(&p, device)
                .map_err(|e| anyhow::anyhow!("text-encoder shard {}: {e}", p.display()))?;
            all.extend(part);
        }
    }
    Ok(all)
}

fn save_ckpt(
    path: &std::path::Path,
    model: &QwenImageTrainingModel,
    optimizer: &Optimizer,
    rank: usize,
    alpha: f32,
    seed: u64,
    mode: &str,
    step: usize,
) -> anyhow::Result<()> {
    let save_mode_full = mode != "weights";
    if !save_mode_full {
        model.save_weights(path)?;
        log::info!("[save] {} (weights only)", path.display());
        return Ok(());
    }

    let named = qwen_named_parameters(model);
    if named.is_empty() {
        // Until qwenimage gets a named_parameters() helper, fall back to
        // weights-only for the periodic save (full-resume from this file is
        // disabled by the named.is_empty() guard above).
        log::warn!("[save] qwenimage missing named_parameters; falling back to weights-only");
        model.save_weights(path)?;
        return Ok(());
    }

    trainer_pipeline::save_lora_checkpoint_strict(
        trainer_pipeline::CheckpointSaveOptions {
            trainer: "train_qwenimage",
            path,
            step: step as u64,
            rank,
            alpha,
            seed,
            config_hash: "",
            save_mode_full,
            label: "[save]",
        },
        optimizer,
        || Ok(named),
        || {
            model.save_weights(path)?;
            Ok(())
        },
    )
}
