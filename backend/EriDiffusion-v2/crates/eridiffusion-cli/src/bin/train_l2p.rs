//! train_l2p — L2P (T2I-L2P, Tencent Youtu) LoRA training binary.
//!
//! L2P = Z-Image-Turbo DiT body + 16×16 pixel-space patchify +
//! MicroDiffusionModel U-Net head. This trainer fine-tunes the DiT body
//! only via LoRA — the U-Net remains frozen.
//!
//! # Pipeline per step
//!
//! 1. Load cached `{pixel, cap_feats}` (from `prepare_l2p`).
//!    - `pixel`     : BF16 `[3, H, W]`,  normalized to [-1, 1]
//!    - `cap_feats` : BF16 `[1, seq, 2560]`
//! 2. Sample timestep `v ∈ (0, 1]` via LOGIT_NORMAL (matches Z-Image's
//!    flow-matching schedule and L2P's `FlowMatchScheduler "Z-Image"` preset).
//! 3. Rectified flow: `noisy = (1 - v) * clean + v * noise`,  v=sigma.
//! 4. Target = `clean - noise`. **L2P's `forward_inner` ends with
//!    `mul_scalar(-1.0)`** (sign-flip convention), so `pred ≈ -velocity ≈
//!    clean - noise`. Comparing pred against (clean - noise) gives the
//!    same sign on both sides.
//! 5. Forward: `pred = model.forward(noisy, sigma, cap_feats)` (returns
//!    BF16 `[1, 3, H, W]`).
//! 6. Loss = mean MSE in F32. Backward, AdamW (BF16 stoch-round via the
//!    AdamW family), step.
//!
//! # LoRA scope
//!
//! DiT-only — 34 attention blocks (2 noise_refiner + 2 context_refiner +
//! 30 main layers) × 5 weight keys per block = **170 LoRA modules**.
//! Per-block targets (post-translation, pre-transposed [in, out] shape):
//!   - `attention.qkv.weight`         [3840, 11520]
//!   - `attention.out.weight`         [3840, 3840]
//!   - `feed_forward.w1.weight`       [3840, 10240]
//!   - `feed_forward.w2.weight`       [10240, 3840]
//!   - `feed_forward.w3.weight`       [3840, 10240]
//!
//! U-Net `local_decoder.*` Conv2d/MaxPool2d weights are excluded from
//! the LoRA target set. Their `requires_grad` stays false (Conv2d weights
//! constructed inside `MicroDiffusionModel::new` are not autograd
//! Parameters), so they receive no gradient and are not added to the
//! optimizer — frozen by construction.

use clap::Parser;
use flame_core::diagnostics;
use flame_core::parameter::Parameter;
use flame_core::serialization::{save_file, save_tensors, SerializationFormat};
use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use eridiffusion_cli::{trainer_common, trainer_pipeline};
use eridiffusion_core::config::GradientCheckpointing;
use eridiffusion_core::training::training_features::{Optimizer, OptimizerKind};
use rand::Rng as _;

use eridiffusion_core::models::l2p::sampling::{
    build_l2p_sigma_schedule, init_l2p_noise, l2p_euler_step,
};
use eridiffusion_core::models::l2p::{
    weight_loader::translate_l2p_keys, L2pDiT, LoraStack, Slot, TrainEntry,
};

// -------------------------------------------------------------------------
// Hyperparameters mirroring Z-Image preset / L2P train_run.sh
// -------------------------------------------------------------------------

// DiT dimensions — pinned from L2pDiTConfig::default() in
// inference-flame/src/models/l2p/dit.rs.
const DIM: usize = 3840;
const MLP_HIDDEN: usize = 10240;
const NUM_LAYERS: usize = 30;

// Sample-prompt text-encoding constants — MUST match prepare_l2p exactly
// (chat template, pad token, pad length, extract layer) so the inline
// sampler's cap_feats share the training-time distribution.
const ZIMAGE_TEMPLATE_PRE: &str = "<|im_start|>user\n";
const ZIMAGE_TEMPLATE_POST: &str = "<|im_end|>\n<|im_start|>assistant\n";
const PAD_TOKEN_ID: i32 = 151643;
const TXT_PAD_LEN: usize = 512;
/// Z-Image canonical extract layer: layer 34 of Qwen3-4B (penultimate).
const QWEN3_EXTRACT_LAYER: usize = 34;

#[derive(Parser)]
struct Args {
    /// L2P single-file safetensors (merged Z-Image-Turbo + L2P deltas).
    #[arg(
        long,
        default_value = "/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors"
    )]
    model: PathBuf,
    /// Directory of `prepare_l2p` outputs (one safetensors per sample).
    #[arg(long)]
    cache: PathBuf,
    /// Where to write LoRA checkpoints.
    #[arg(long)]
    output: PathBuf,
    /// Total training steps. ai-toolkit UI default = 3000 for ZImage L2P
    /// (`ui/src/app/jobs/new/jobConfig.ts:69`); 1000 was undertrained on
    /// the 22-image box1jana dataset (subject identity didn't bind to the
    /// trigger token). 200 still works as a smoke target via explicit flag.
    #[arg(long, default_value = "3000")]
    steps: usize,
    /// Learning rate. Default `1e-4` matches Ostris ai-toolkit's ZImage-L2P
    /// LoRA recipe (their range is 1e-4 .. 4e-4). Earlier `5e-5` from L2P's
    /// train_run.sh was tuned for full DiT fine-tune, not LoRA — too low
    /// for rank-16 LoRA training (LoRA needs roughly 5-10× the full-FT LR).
    #[arg(long, default_value_t = 1e-4)]
    lr: f32,
    /// LoRA rank.
    #[arg(long, default_value_t = 16)]
    lora_rank: usize,
    /// LoRA alpha (effective scale = alpha / rank).
    #[arg(long, default_value_t = 16.0)]
    lora_alpha: f32,
    /// Square training resolution. 512 is the smoke target. 1024² works
    /// for inference but blows the activation budget for training on
    /// 24 GB cards (per PORT_STATE.md activation estimate).
    #[arg(long, default_value_t = 512)]
    resolution: usize,
    /// Per-step gradient-clip global L2 norm.
    #[arg(long, default_value_t = 1.0)]
    clip_grad_norm: f32,
    /// Random seed (data shuffling + noise). Default matches the project
    /// convention (`SEED=42` across the codebase per CONTEXT.md).
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Save LoRA every N steps. `0` disables periodic save (final-only).
    /// Default 1000 → checkpoints at 1000, 2000, 3000 — matches the user
    /// directive "sample every 1000 steps" since post-hoc sampling reads
    /// these checkpoints.
    #[arg(long, default_value_t = 1000)]
    save_every: usize,
    /// Resume from a previously-saved LoRA safetensors (PEFT format produced
    /// by this trainer's `save_lora_peft`). Copies saved weights into the
    /// fresh LoRA Parameters; Adam optimizer state is reinitialized (Adam
    /// recovers in ~50-100 steps so this is acceptable for short runs).
    /// Pair with `--start-step` so save filenames advance correctly.
    #[arg(long)]
    resume: Option<PathBuf>,
    /// Step counter offset for logging + save filenames when resuming.
    /// E.g. `--resume step250.safetensors --start-step 250 --steps 500`
    /// continues for 500 more steps, saving as `l2p_lora_step500.safetensors`
    /// etc. (Adam state still resets; only weights resume.)
    #[arg(long, default_value_t = 0)]
    start_step: usize,
    /// Enable gradient checkpointing — recompute activations during backward
    /// to cut peak memory ~3-4×. Required for 512²+ training on 24 GB.
    /// Adds ~30% compute per step (~2.5 → 3.3 s/step).
    #[arg(long, default_value_t = false)]
    grad_checkpoint: bool,
    /// SQLite scalars DB for the BoardWriter (per-step loss / grad_norm).
    #[arg(long, default_value = "./l2p_train.board.db")]
    log_db: PathBuf,
    /// (RETIRED) Number of training-time scheduler steps. Was used to size
    /// a shift-warped sigma lookup table; now we sample sigma uniformly in
    /// `(0, 1]` directly (Ostris ai-toolkit recipe). The flag is kept for
    /// back-compat and only controls the discrete granularity (t_int in
    /// `1..=train_num_steps`, sigma = t_int / train_num_steps). Default
    /// 1000 matches ai-toolkit's `linspace(1000, 1)`.
    #[arg(long, default_value_t = 1000)]
    train_num_steps: usize,
    /// (RETIRED) FlowMatch sigma shift. Was used to warp the training
    /// schedule; now ignored — Ostris ai-toolkit uses unshifted uniform
    /// sigma sampling. Inference still uses shift=3.0 (see
    /// `build_l2p_sigma_schedule` in `l2p_sampling.rs`).
    #[arg(long, default_value_t = 1.0)]
    train_shift: f32,
    /// Optimizer family. `adamw` is the canonical project default (BF16
    /// grad → F32 moments via the default `GradDtypePolicy::CastToF32`).
    #[arg(long, default_value = "adamw")]
    optimizer: String,

    // ── Inline sampling (config-driven, mirrors train_klein) ─────────────
    /// Render a validation image every N steps (and at the first training
    /// step + at the end). `0` disables inline sampling entirely. Requires
    /// `--validation-prompts-file`, `--sample-qwen3`, `--sample-tokenizer`
    /// when > 0.
    #[arg(long, default_value_t = 0)]
    sample_every: usize,
    /// JSON prompt library (`SampleLibrary`) for periodic + final samples.
    /// When absent, inline sampling is skipped (L2P had no sampler before,
    /// so absent = no inline sample — byte-identical to the prior trainer).
    #[arg(long)]
    validation_prompts_file: Option<PathBuf>,
    /// Qwen3-4B text encoder (single file or shard dir) used ONLY to encode
    /// the validation prompts at startup. Dropped before the DiT loads.
    /// Default matches prepare_l2p's encoder path.
    #[arg(
        long,
        default_value = "/home/alex/.serenity/models/text_encoders/qwen_3_4b.safetensors"
    )]
    sample_qwen3: PathBuf,
    /// Tokenizer.json (Qwen3 BPE). Shared with Z-Image / prepare_l2p.
    #[arg(
        long,
        default_value = "/home/alex/.serenity/models/zimage_base/tokenizer/tokenizer.json"
    )]
    sample_tokenizer: PathBuf,
    /// Square sample resolution. L2P inference reference is 1024².
    #[arg(long, default_value_t = 1024)]
    sample_size: usize,
    /// Sampler steps (Euler). ai-toolkit ZImage-L2P recipe = 30.
    #[arg(long, default_value_t = 30)]
    sample_steps: usize,
    /// CFG scale for sampling. L2P README / ai-toolkit recipe = 2.0.
    #[arg(long, default_value_t = 2.0)]
    sample_cfg: f32,
    /// Inference sigma-schedule shift. L2P uses 3.0 (FLUX-shift form).
    #[arg(long, default_value_t = 3.0)]
    sample_shift: f32,
    /// Seed for sample noise.
    #[arg(long, default_value_t = 42)]
    sample_seed: u64,

    /// OT-compatible JSON `TrainConfig`. When provided, its fields OVERRIDE
    /// the corresponding CLI defaults below (the merge happens in `main`
    /// right after parse). This is the same `--config` mechanism every other
    /// EdV2 trainer uses (commit `82db138`) — L2P was the lone trainer that
    /// hardcoded its hyperparameters with no config-file path. Fields OT's
    /// TrainConfig does NOT carry (`steps`, the `sample_*` knobs) stay as
    /// CLI flags, identical to `train_zimage`.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Opt OUT of autograd v2 and run the legacy v3 engine. v2 is the default
    /// as of 2026-05-30 (gate-on Stage 6a); v3 kept as the reference engine.
    #[arg(long, default_value_t = false)]
    use_autograd_v3: bool,
}

// (build_timestep_config removed 2026-05-22 — L2P uses uniform unshifted
//  sigma sampling per the Ostris ai-toolkit ZImage-L2P recipe
//  (timestep_type='linear'), NOT LOGIT_NORMAL. Inference sigma schedule
//  lives in eridiffusion_core::models::l2p::sampling::build_l2p_sigma_schedule.)

// -------------------------------------------------------------------------
// LoRA target table — per-block weight keys + (in, out) dims + slot
// -------------------------------------------------------------------------

/// adaLN_modulation.0 input dim = `t_embedder` hidden = 256 for Z-Image-Turbo.
/// Output dim = 4 * DIM (chunks into scale_msa / gate_msa / scale_mlp / gate_mlp).
const T_EMB_DIM: usize = 256;
const ADALN_OUT: usize = 4 * DIM; // 15360

/// One LoRA target: where it applies in the model, how it's saved, what
/// portion of the base output it adds delta to.
#[derive(Clone)]
struct L2pLoraTarget {
    /// Internal model weight key the LoRA delta gets added to.
    weight_key: String,
    /// Logical name used in the saved safetensors file. For the Q/K/V split
    /// across our fused qkv weight, this maps to ai-toolkit's separate
    /// `attention.to_q/to_k/to_v.weight` keys — `LoraStack::load`
    /// (`inference-flame/src/lora.rs::map_prefix_diffusion_model`) recognizes
    /// these and reassembles them onto the fused qkv with the correct
    /// `Slot::RowRange`. Same trick for `attention.to_out.0` → `attention.out`.
    save_key: String,
    in_dim: usize,
    out_dim: usize,
    slot: Slot,
}

/// 8 LoRA modules per `layers.{i}` block, matching ai-toolkit's discovery
/// (`lora_special.py:367-374` + `z_image.py:386-387`): every `nn.Linear`
/// under `layers` is hooked. Specifically:
///   - `attention.to_q` / `to_k` / `to_v` — 3 separate LoRAs targeting
///     row-partitions of our fused `attention.qkv.weight` via
///     `Slot::RowRange`. Each gets an independent rank-`R` direction
///     (effective rank 3R vs the 1R we got from a single LoRA on the
///     fused weight). This recovers the per-projection expressivity that
///     ai-toolkit's separate `nn.Linear` modules carry natively.
///   - `attention.to_out.0` — output projection, saved with the ai-toolkit
///     key. Loads onto our `attention.out.weight` via the loader mapper.
///   - `feed_forward.w1 / w2 / w3` — SwiGLU FFN.
///   - `adaLN_modulation.0` — per-block scale/shift/gate Linear. **The
///     missing-modulation LoRA was the single biggest convergence gap vs
///     ai-toolkit** per audit `AITOOLKIT_L2P_GRAD_AUDIT.md` § "wrong-bucket
///     LoRA target list" — every block's modulation drives all downstream
///     scale + gate values, so excluding it starved the LoRA of the
///     strongest per-block adaptation lever.
fn l2p_block_targets(block_prefix: &str) -> Vec<L2pLoraTarget> {
    vec![
        // Q / K / V — three LoRAs on the fused qkv weight, each targeting
        // a third of the [3*DIM] output via Slot::RowRange.
        L2pLoraTarget {
            weight_key: format!("{block_prefix}.attention.qkv.weight"),
            save_key: format!("{block_prefix}.attention.to_q"),
            in_dim: DIM,
            out_dim: DIM,
            slot: Slot::RowRange { start: 0, len: DIM },
        },
        L2pLoraTarget {
            weight_key: format!("{block_prefix}.attention.qkv.weight"),
            save_key: format!("{block_prefix}.attention.to_k"),
            in_dim: DIM,
            out_dim: DIM,
            slot: Slot::RowRange {
                start: DIM,
                len: DIM,
            },
        },
        L2pLoraTarget {
            weight_key: format!("{block_prefix}.attention.qkv.weight"),
            save_key: format!("{block_prefix}.attention.to_v"),
            in_dim: DIM,
            out_dim: DIM,
            slot: Slot::RowRange {
                start: 2 * DIM,
                len: DIM,
            },
        },
        // Output projection.
        L2pLoraTarget {
            weight_key: format!("{block_prefix}.attention.out.weight"),
            save_key: format!("{block_prefix}.attention.to_out.0"),
            in_dim: DIM,
            out_dim: DIM,
            slot: Slot::Full,
        },
        // SwiGLU FFN.
        L2pLoraTarget {
            weight_key: format!("{block_prefix}.feed_forward.w1.weight"),
            save_key: format!("{block_prefix}.feed_forward.w1"),
            in_dim: DIM,
            out_dim: MLP_HIDDEN,
            slot: Slot::Full,
        },
        L2pLoraTarget {
            weight_key: format!("{block_prefix}.feed_forward.w2.weight"),
            save_key: format!("{block_prefix}.feed_forward.w2"),
            in_dim: MLP_HIDDEN,
            out_dim: DIM,
            slot: Slot::Full,
        },
        L2pLoraTarget {
            weight_key: format!("{block_prefix}.feed_forward.w3.weight"),
            save_key: format!("{block_prefix}.feed_forward.w3"),
            in_dim: DIM,
            out_dim: MLP_HIDDEN,
            slot: Slot::Full,
        },
        // Per-block modulation linear — the previously-missing target.
        L2pLoraTarget {
            weight_key: format!("{block_prefix}.adaLN_modulation.0.weight"),
            save_key: format!("{block_prefix}.adaLN_modulation.0"),
            in_dim: T_EMB_DIM,
            out_dim: ADALN_OUT,
            slot: Slot::Full,
        },
    ]
}

/// 30 main `layers` blocks × 8 targets = 240 entries. Matches ai-toolkit's
/// production discovery exactly. Refiner blocks (`noise_refiner.*`,
/// `context_refiner.*`) remain EXCLUDED per ai-toolkit's
/// `get_transformer_block_names() == ["layers"]`.
fn enumerate_lora_targets() -> Vec<L2pLoraTarget> {
    let mut out = Vec::with_capacity(NUM_LAYERS * 8);
    for i in 0..NUM_LAYERS {
        out.extend(l2p_block_targets(&format!("layers.{i}")));
    }
    out
}

/// Build a fresh LoRA Parameter pair for a single target.
///
/// Shape convention matches `LoraStack` training apply (`(x @ down) @ up`):
///   down: [in,   rank]  — Kaiming-ish small init (1/sqrt(rank) std)
///   up:   [rank, out ]  — zero init (canonical LoRA convention)
///
/// Both dtypes **F32** to match Ostris ai-toolkit's LoRA recipe
/// (`BaseSDTrainProcess.py:1800` does `self.network.force_to(... dtype=torch.float32)`
/// before training starts, then the LoRA matmul chain runs in F32, with
/// the delta cast to the base BF16 dtype only at the final add). BF16
/// LoRA weights (our prior default) suffered grad-magnitude collapse on
/// long-sequence reductions: sum-of-~9K-terms with mixed signs in BF16
/// flushes the gradient signal to ~zero through cancellation. See
/// `inference-flame/src/models/l2p/AITOOLKIT_L2P_GRAD_AUDIT.md` § "F32 vs
/// BF16" for the full derivation. `apply_training`
/// (`inference-flame/src/lora.rs`) auto-detects the F32 dtype on the LoRA
/// Parameters and (a) casts x to F32 via autograd-tracked `to_dtype`, (b)
/// runs the matmul chain in F32, (c) casts the delta back to BF16 before
/// adding to the BF16 base — bit-equivalent to ai-toolkit's
/// `ToolkitModuleMixin.forward` chain.
fn make_lora_pair(
    name: &str,
    in_dim: usize,
    out_dim: usize,
    rank: usize,
    device: &Arc<flame_core::CudaDevice>,
    seed: u64,
) -> anyhow::Result<(Parameter, Parameter)> {
    // Match PyTorch nn.Linear default init = `kaiming_uniform_(a=sqrt(5))`
    // (which is what ai-toolkit's LoRAModule uses at `lora_special.py:120`).
    // For uniform(-bound, bound) with bound = sqrt(6 / (in * (1 + a^2)))
    // and a = sqrt(5), bound = 1/sqrt(in_features). Variance = bound^2/3 =
    // 1/(3·in_features), so std ≈ 1/sqrt(3·in_features). For in_features
    // 2560-3840 this gives std ≈ 0.010 — matches the ~0.009 std observed
    // on ai-toolkit's trained-from-zero baseline. Prior `1/sqrt(rank)` =
    // 0.25 was ~25x too large, causing the LoRA-A weights to start huge
    // and the resulting delta to overwhelm the base output by step 1000,
    // producing a destroyed inference output regardless of training
    // direction quality. The bug was diagnosed by per-key magnitude diff
    // vs the ai-toolkit step-1000 baseline (see
    // `inference-flame/src/models/l2p/AITOOLKIT_L2P_GRAD_AUDIT.md`).
    let down_std = 1.0_f32 / ((in_dim as f32) * 3.0).sqrt();
    let down = Tensor::randn_seeded(
        Shape::from_dims(&[in_dim, rank]),
        0.0,
        down_std,
        seed,
        device.clone(),
    )?
    .to_dtype(DType::F32)?
    .requires_grad_(true);
    let up = Tensor::zeros_dtype(
        Shape::from_dims(&[rank, out_dim]),
        DType::F32,
        device.clone(),
    )?
    .requires_grad_(true);
    let _ = name; // diagnostic placeholder
    Ok((Parameter::new(down), Parameter::new(up)))
}

fn main() -> anyhow::Result<()> {
    use rand::SeedableRng;

    // -------------------------------------------------------------------
    // Pre-flight env warnings. We don't FORCE the env vars at runtime —
    // setting them mid-process is racy on multi-thread. We diagnose.
    // -------------------------------------------------------------------
    if std::env::var("FLAME_ALLOC_POOL").as_deref() != Ok("0") {
        eprintln!(
            "WARNING: FLAME_ALLOC_POOL is not set to 0. L2P training is known to OOM \
             without this. Recommend `FLAME_ALLOC_POOL=0 ... train_l2p ...`."
        );
    }
    if std::env::var("FLAME_AUTOGRAD_OFF").as_deref() == Ok("1") {
        eprintln!("FATAL: FLAME_AUTOGRAD_OFF=1 disables training. Unset before running.");
        std::process::exit(2);
    }
    if std::env::var("FLAME_ASSERT_GRAD_FLOW").as_deref() != Ok("1") {
        eprintln!(
            "WARNING: FLAME_ASSERT_GRAD_FLOW is not set to 1. Recommended for catching \
             dead-leaf training bugs early."
        );
    }

    trainer_common::init_logging();
    let mut args = Args::parse();

    // -------------------------------------------------------------------
    // Config-file merge (parity with every other EdV2 trainer; commit
    // `82db138`). When `--config <json>` is given, the OT-compatible
    // `TrainConfig` is the source of truth for the training hyperparameters
    // it carries — they OVERRIDE the CLI defaults. This removes L2P's status
    // as the lone trainer that could only be configured via source-baked
    // CLI defaults. `steps` and the `sample_*` knobs are not part of OT's
    // TrainConfig schema, so they remain CLI flags (same as `train_zimage`).
    // -------------------------------------------------------------------
    if let Some(ref cfg_path) = args.config {
        let c = trainer_common::load_train_config(cfg_path)
            .map_err(|e| anyhow::anyhow!("--config {}: {e}", cfg_path.display()))?;
        args.lr = c.learning_rate as f32;
        args.lora_rank = c.lora_rank as usize;
        args.lora_alpha = c.lora_alpha as f32;
        args.clip_grad_norm = c.clip_grad_norm as f32;
        args.save_every = c.save_every as usize;
        args.train_shift = c.timestep_shift as f32;
        args.grad_checkpoint = matches!(c.gradient_checkpointing, GradientCheckpointing::On);
        if !c.optimizer.name.is_empty() {
            args.optimizer = c.optimizer.name.clone();
        }
        // Step count + the full inline-sampler block now live in the config
        // (sentinel 0 / None = "not specified", keep the CLI default). With a
        // complete config, the launch command carries NO tunable params —
        // only `--config` + per-run paths. This is the whole point: nothing
        // tunable is hardcoded on the command line.
        if c.steps > 0 {
            args.steps = c.steps as usize;
        }
        args.sample_every = c.sample_every as usize;
        if c.sample_size > 0 {
            args.sample_size = c.sample_size as usize;
        }
        if c.sample_steps > 0 {
            args.sample_steps = c.sample_steps as usize;
        }
        if c.sample_cfg > 0.0 {
            args.sample_cfg = c.sample_cfg;
        }
        if c.sample_shift > 0.0 {
            args.sample_shift = c.sample_shift;
        }
        if let Some(seed) = c.sample_seed {
            args.sample_seed = seed;
        }
        // CLI `--validation-prompts-file` wins if explicitly given; otherwise
        // take it from the config so a single file fully describes a run.
        if args.validation_prompts_file.is_none() {
            args.validation_prompts_file = c.validation_prompts_file.clone();
        }
        log::info!(
            "[config] {} → steps={} lr={} rank={} alpha={} clip={} save_every={} optimizer={} grad_ckpt={}",
            cfg_path.display(),
            args.steps,
            args.lr,
            args.lora_rank,
            args.lora_alpha,
            args.clip_grad_norm,
            args.save_every,
            args.optimizer,
            args.grad_checkpoint,
        );
        log::info!(
            "[config]   sampler → every={} size={} steps={} cfg={} shift={} seed={}",
            args.sample_every,
            args.sample_size,
            args.sample_steps,
            args.sample_cfg,
            args.sample_shift,
            args.sample_seed,
        );
    }
    trainer_common::ensure_output_dir(&args.output)?;

    let device = trainer_common::init_bf16_cuda();

    // -------------------------------------------------------------------
    // 0. Inline-sample setup (config-driven, mirrors train_klein).
    //
    // MUST run BEFORE the L2P DiT load: the merged L2P checkpoint (~10 GB
    // BF16-resident) + Qwen3-4B (~8 GB) do not co-reside comfortably on a
    // 24 GB card. Encode all validation prompts now, drop Qwen3, then load
    // the DiT. `sample_set`: (label, cap_feats, cap_feats_uncond).
    //
    // Inline sampling is gated on BOTH `--sample-every > 0` AND a prompts
    // file being present. If either is absent, sampling is skipped — L2P
    // had no sampler before, so absent = no-op (byte-identical behaviour).
    // -------------------------------------------------------------------
    let periodic = args.sample_every > 0 && args.validation_prompts_file.is_some();
    // (label, cap_feats, cap_feats_uncond, prompt_text). prompt_text is kept
    // so the board can log_text it alongside each rendered sample.
    let mut sample_set: Vec<(String, Tensor, Tensor, String)> = Vec::new();
    if periodic {
        use eridiffusion_core::encoders::qwen3::Qwen3Encoder;
        use eridiffusion_core::training::features::sample_library::SampleLibrary;
        let prompts_file = args.validation_prompts_file.as_ref().unwrap();
        log::info!(
            "[sample-setup] loading Qwen3 + tokenizer to encode sample prompts (before DiT load)..."
        );
        let qwen_w = l2p_load_qwen3(&args.sample_qwen3, &device)?;
        let mut qcfg = Qwen3Encoder::config_from_weights(&qwen_w)?;
        // Z-Image / L2P canonical extract layer = 34 (penultimate). Must match
        // prepare_l2p so sample cap_feats share the training distribution.
        qcfg.extract_layers = vec![QWEN3_EXTRACT_LAYER];
        let qwen = Qwen3Encoder::new(qwen_w, qcfg, device.clone());
        let tok = tokenizers::Tokenizer::from_file(&args.sample_tokenizer)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

        let lib = SampleLibrary::from_file(prompts_file)?;
        log::info!(
            "[sample-setup] {} prompt(s) from {}",
            lib.len(),
            prompts_file.display()
        );
        for (i, p) in lib.prompts.iter().enumerate() {
            let label = format!("p{}", i + 1);
            let cap = l2p_encode_prompt(&qwen, &tok, &p.prompt)?;
            // Unconditional embedding = the prompt's negative (empty by
            // default in SampleLibrary). CFG uses pred_uncond + cfg*(cond-uncond).
            let unc = l2p_encode_prompt(&qwen, &tok, &p.negative)?;
            log::info!(
                "[sample-setup] {label} encoded cap={:?}",
                cap.shape().dims()
            );
            sample_set.push((label, cap, unc, p.prompt.clone()));
        }
        drop(qwen);
        flame_core::trim_cuda_mempool(0);
        log::info!(
            "[sample-setup] Qwen3 dropped; {} prompt(s) ready. Sample every {} steps.",
            sample_set.len(),
            args.sample_every
        );
    } else if args.sample_every > 0 {
        log::warn!(
            "[sample-setup] --sample-every={} but no --validation-prompts-file; inline sampling DISABLED.",
            args.sample_every
        );
    }

    // -------------------------------------------------------------------
    // 1. Load + translate + construct L2pDiT.
    // -------------------------------------------------------------------
    log::info!(
        "[1/5] Loading L2P safetensors from {}...",
        args.model.display()
    );
    let source = flame_core::serialization::load_file(&args.model, &device)?;
    let internal = translate_l2p_keys(source)?;
    log::info!(
        "  translated {} keys ({} after fuse + rename)",
        internal.len(),
        internal.len()
    );
    let mut model = L2pDiT::new_resident(internal, device.clone());
    if args.grad_checkpoint {
        model.set_grad_checkpoint(true);
        log::info!("[1/5] Gradient checkpointing ENABLED (peak activation memory ~3-4× lower, ~30% compute overhead per step).");
    }

    // -------------------------------------------------------------------
    // 2. Build LoRA Parameters + assemble training-mode LoraStack.
    // -------------------------------------------------------------------
    let targets = enumerate_lora_targets();
    let n_targets = targets.len();
    log::info!(
        "[2/5] Building DiT-only LoRA: rank={} alpha={} → {} modules",
        args.lora_rank,
        args.lora_alpha,
        n_targets,
    );
    let scale = args.lora_alpha / args.lora_rank as f32;
    let mut train_map: HashMap<String, Vec<TrainEntry>> = HashMap::new();
    let mut params: Vec<Parameter> = Vec::new();
    let mut named: Vec<(String, Parameter)> = Vec::new();
    for (idx, target) in targets.into_iter().enumerate() {
        // Per-target seed offset so each module's down init is distinct.
        // We seed off the save_key (which is unique across Q/K/V splits)
        // rather than weight_key (which Q/K/V share).
        let (down, up) = make_lora_pair(
            &target.save_key,
            target.in_dim,
            target.out_dim,
            args.lora_rank,
            &device,
            args.seed + idx as u64,
        )?;
        params.push(down.clone());
        params.push(up.clone());
        // PEFT/ai-toolkit save format names. Save under the save_key — the
        // inference loader's `map_prefix_diffusion_model` translates these
        // ai-toolkit-style names back to internal `attention.qkv.weight` +
        // `Slot::RowRange` at load time, and `attention.to_out.0` →
        // `attention.out.weight` + `Slot::Full`. So one safetensors file
        // is interoperable across both our trainer-built LoRAs AND any
        // ai-toolkit-trained LoRA placed in the same dir.
        named.push((
            format!("diffusion_model.{}.lora_A.weight", target.save_key),
            down.clone(),
        ));
        named.push((
            format!("diffusion_model.{}.lora_B.weight", target.save_key),
            up.clone(),
        ));
        // Multiple entries can share the same weight_key (Q, K, V all hit
        // `attention.qkv.weight` with different `Slot::RowRange`s). The Vec
        // means LoraStack's apply iterates and adds all matching deltas.
        train_map
            .entry(target.weight_key.clone())
            .or_default()
            .push(TrainEntry {
                slot: target.slot,
                down,
                up,
                scale,
            });
    }
    let total_entries: usize = train_map.values().map(|v| v.len()).sum();
    if total_entries != n_targets {
        anyhow::bail!(
            "expected {} total LoRA entries across train_map, got {}",
            n_targets,
            total_entries,
        );
    }
    log::info!(
        "  built {} Parameters ({} train entries across {} weight keys; \
         Q/K/V splits and adaLN_modulation included)",
        params.len(),
        total_entries,
        train_map.len(),
    );

    // Gate-on 6a: under v2 (default), flip LoRA params to MatchParamDtype so
    // BF16 grads from the bridge stay BF16 (Class A). --use-autograd-v3 skips.
    trainer_pipeline::apply_autograd_v2_grad_policy(&mut params, args.use_autograd_v3, "params");

    // Optional: resume LoRA weights from a previous checkpoint. Adam state
    // is NOT restored (acceptable tradeoff — Adam recovers in ~50 steps).
    if let Some(resume_path) = &args.resume {
        load_lora_resume(resume_path, &named, &device)?;
        log::info!(
            "[2/5] Resumed LoRA weights from {} (start_step={})",
            resume_path.display(),
            args.start_step,
        );
    }

    let stack = Arc::new(LoraStack::new_training(train_map));
    model.set_lora(stack);

    // -------------------------------------------------------------------
    // 3. Optimizer + timestep config + BoardWriter.
    // -------------------------------------------------------------------
    let opt_kind =
        OptimizerKind::parse(&args.optimizer).map_err(|e| anyhow::anyhow!("--optimizer: {e}"))?;
    log::info!("[3/5] Optimizer: {} lr={}", opt_kind.as_str(), args.lr);
    // C2 fix 2026-05-25: match ai-toolkit's ACTUAL optimizer construction.
    // optimizer.py:71 → `AdamW8bit(params, lr, eps=1e-6, **optimizer_params)`.
    // l2p_boxjana_baseline.yaml sets optimizer=adamw8bit + lr only (NO
    // optimizer_params), so weight_decay falls through to bitsandbytes'
    // AdamW8bit default = 0.01.
    //   eps: 1e-8 → 1e-6 (explicit in optimizer.py:71).
    //   weight_decay: stays 0.01 (bnb default; the auditor's 1e-4 came from
    //   jobConfig.ts, a UI default the hand-written baseline never uses — REVERTED).
    let mut opt = Optimizer::new(opt_kind, args.lr, 0.9, 0.999, 1e-6, 0.01);

    // Training-time sigma sampling: Ostris ai-toolkit ZImage-L2P recipe.
    // Sample sigma UNIFORMLY in `(0, 1]` (mirrors `linspace(1000, 1)` with
    // uniform index). The previous FLUX-shift-warped table concentrated
    // training mass near low sigma (near-clean), biasing the LoRA toward
    // identity. ai-toolkit's `timestep_type='linear'` is the production-
    // tested choice (commit 6102370 `Add support for ZImage L2P`).
    log::info!(
        "[3/5] Training sigma sampling: uniform t_int in 1..={} → sigma = t_int / {} (unshifted, ai-toolkit recipe)",
        args.train_num_steps,
        args.train_num_steps,
    );
    let _ = args.train_shift; // intentionally unused — see Args::train_shift doc.

    let board = trainer_common::open_board_writer(&args.output, None);
    if let Some(b) = &board {
        // Full board wiring: run hyper-parameters → metadata.hparams + the
        // dashboard's hparam panel. JSON hand-built (no serde dep here).
        let hparams_json = format!(
            "{{\"model\":\"l2p\",\"steps\":{},\"rank\":{},\"lora_alpha\":{},\"lr\":{},\
             \"batch_size\":{},\"optimizer\":\"{}\",\"seed\":{},\"resolution\":{},\
             \"clip_grad_norm\":{},\"start_step\":{},\"grad_checkpoint\":{}}}",
            args.steps,
            args.lora_rank,
            args.lora_alpha,
            args.lr,
            1,
            opt_kind.as_str(),
            args.seed,
            args.resolution,
            args.clip_grad_norm,
            args.start_step,
            args.grad_checkpoint,
        );
        b.log_hparams(&hparams_json, &[("steps_target", args.steps as f64)]);
    }
    // The `--log-db` flag is preserved for forward-compat; BoardWriter::open
    // currently picks the path under `--output`. Surface a warning if the
    // user wanted a different DB path.
    let _ = &args.log_db;

    // -------------------------------------------------------------------
    // 4. Enumerate cached samples.
    // -------------------------------------------------------------------
    let cache_files = trainer_common::list_cache_safetensors(&args.cache)?;
    log::info!("[4/5] Found {} cached samples", cache_files.len());

    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);

    // -------------------------------------------------------------------
    // 5. Training loop.
    // -------------------------------------------------------------------
    log::info!(
        "[5/5] Starting {} training steps @ resolution {}²",
        args.steps,
        args.resolution
    );
    let loop_config = trainer_pipeline::ManualTrainLoopConfig::new(
        "L2P-lora",
        0,
        args.steps,
        cache_files.len(),
        1,
    )
    .with_progress_target(args.start_step, args.steps + args.start_step);
    let mut loop_run = trainer_pipeline::ManualTrainLoopRun::new(loop_config);

    for step in loop_run.steps() {
        // ── Load one cached sample ────────────────────────────────────
        let cache_idx = step % cache_files.len();
        let sample = flame_core::serialization::load_file(&cache_files[cache_idx], &device)?;
        let pixel = sample
            .get("pixel")
            .ok_or_else(|| anyhow::anyhow!("cache {cache_idx} missing 'pixel'"))?
            .to_dtype(DType::BF16)?;
        // pixel arrives as [3, H, W]; reshape to [1, 3, H, W].
        let pixel = {
            let d = pixel.shape().dims().to_vec();
            if d.len() != 3 {
                anyhow::bail!("pixel shape {:?} != [3, H, W]", d);
            }
            pixel.reshape(&[1, d[0], d[1], d[2]])?
        };
        let cap_feats = sample
            .get("cap_feats")
            .ok_or_else(|| anyhow::anyhow!("cache {cache_idx} missing 'cap_feats'"))?
            .to_dtype(DType::BF16)?;

        // ── Sample timestep + build noisy / target ───────────────────
        //
        // Python L2P (loss.py:6-13 with train_L2P.py:89 `num_inference_steps=500`):
        //   timestep_id = randint(0, len(scheduler.timesteps))     # uniform [0, 500)
        //   sigma       = scheduler.sigmas[timestep_id]            # FLUX-shifted
        //   noisy       = (1 - sigma) * clean + sigma * noise
        //   target      = noise - clean
        //   pred        = -DiT(noisy, timestep=sigma*1000 → forward divides by 1000)
        //   loss        = MSE(pred, target)
        //
        // Our path mirrors exactly: uniform idx → lookup shift-warped sigma →
        // pass that sigma as `v ∈ [0,1]` to L2pDiT.forward (which applies
        // `(1-v)*time_scale` internally — net effect identical to Python's
        // `timestep / 1000` pre-divide).
        // Uniform sigma sampling (Ostris ai-toolkit ZImage-L2P recipe).
        // t_int ∈ {1, 2, ..., train_num_steps} → sigma = t_int / train_num_steps
        // gives a discrete uniform distribution on `(0, 1]` with granularity
        // 1/train_num_steps. Matches `timestep_type='linear'` in ai-toolkit.
        let t_int: usize = (rng.gen::<u32>() as usize) % args.train_num_steps + 1;
        let sigma = t_int as f32 / args.train_num_steps as f32;
        let v_in = sigma;

        let noise = Tensor::randn(pixel.shape().clone(), 0.0, 1.0, device.clone())?
            .to_dtype(DType::BF16)?;
        // Rectified flow noisy: (1 - sigma) * clean + sigma * noise.
        let noisy = pixel
            .mul_scalar(1.0 - sigma)?
            .add(&noise.mul_scalar(sigma)?)?;
        // Target = (noise - clean) per Python L2P's `FlowMatchScheduler.training_target`
        // (reference/diffsynth/diffusion/flow_match.py:172-174: `target = noise - sample`).
        //
        // Python's pipeline applies the SAME negation we do: `model_fn_z_image`
        // returns `-DiT(...)`. Loss in Python: MSE(model_fn_output, training_target)
        //                                    = MSE(-v_raw, noise - clean).
        // Our pred path is identical: L2pDiT.forward returns `-v_raw` (via the
        // `mul_scalar(-1.0)` at the tail of `forward_inner`). So our target
        // must match Python's: `noise - clean`.
        //
        // (Earlier this was `clean - noise`. That inverts both sides of the MSE
        // and the model can't learn — loss saturates at ~4*var ≈ 5.5 in BF16.
        // Fix landed 2026-05-22 after a 300-step smoke confirmed the inversion.)
        let target = noise.sub(&pixel)?;

        let timestep = Tensor::from_vec(vec![v_in], Shape::from_dims(&[1]), device.clone())?
            .to_dtype(DType::BF16)?;

        if step == 0 {
            log::info!(
                "step 0 | pixel={:?} cap={:?} sigma={:.4} (v_in={:.4})",
                pixel.shape().dims(),
                cap_feats.shape().dims(),
                sigma,
                v_in,
            );
        }

        // ── Forward ──────────────────────────────────────────────────
        let pred = model.forward(&noisy, &timestep, &cap_feats)?;
        if pred.shape().dims() != target.shape().dims() {
            anyhow::bail!(
                "pred {:?} != target {:?}",
                pred.shape().dims(),
                target.shape().dims()
            );
        }

        // ── Loss = mean MSE in F32 ───────────────────────────────────
        let pred_f32 = pred.to_dtype(DType::F32)?;
        let target_f32 = target.to_dtype(DType::F32)?;
        let diff = pred_f32.sub(&target_f32)?;
        let loss = diff.mul(&diff)?.mean()?;
        let loss_val = loss.to_vec()?[0];

        let grads = trainer_pipeline::backward_loss(&loss, args.use_autograd_v3)?;

        // Grad-flow check at step 1 (LoRA-B is zero-init so step-0
        // through `delta = down @ up` is identically zero; backward
        // through `delta * weight` produces zero gradients on down by
        // mathematical construction. Step 1 is the first step where the
        // assertion can distinguish "real bug" from "expected zero").
        if step == 1 {
            let named_refs: Vec<(&str, &Parameter)> =
                named.iter().map(|(n, p)| (n.as_str(), p)).collect();
            match diagnostics::assert_grad_flow(&grads, &named_refs) {
                Ok(report) if report.is_clean() => {
                    log::info!("[grad-flow] step 1 clean ({} params)", report.ok_count);
                }
                Ok(report) => log::warn!("{}", report.summary()),
                Err(e) => log::warn!("[grad-flow] check failed: {e}"),
            }
        }

        // ── Per-group grad norms (LoRA-A vs LoRA-B) ─────────────────
        // The global grad_norm hides the typical pattern in LoRA training
        // where lora_B (zero-init) grads are tiny early on while lora_A
        // (random-init) grads dominate. Logging the two separately surfaces
        // whether convergence is actually happening on the B side — the
        // load-bearing direction since identity (B=0) means no LoRA effect.
        let mut grad_refs_a: Vec<&Tensor> = Vec::with_capacity(named.len() / 2);
        let mut grad_refs_b: Vec<&Tensor> = Vec::with_capacity(named.len() / 2);
        for (name, p) in &named {
            if let Some(g) = grads.get(p.id()) {
                if name.ends_with(".lora_A.weight") {
                    grad_refs_a.push(g);
                } else if name.ends_with(".lora_B.weight") {
                    grad_refs_b.push(g);
                }
            }
        }
        let norm_a = if grad_refs_a.is_empty() {
            0.0_f32
        } else {
            flame_core::ops::grad_norm::global_l2_norm(&grad_refs_a)?.item()? as f32
        };
        let norm_b = if grad_refs_b.is_empty() {
            0.0_f32
        } else {
            flame_core::ops::grad_norm::global_l2_norm(&grad_refs_b)?.item()? as f32
        };

        // ── Grad clip + assign ──────────────────────────────────────
        let clip = trainer_pipeline::apply_gradient_map_clip(
            &params,
            &grads,
            trainer_pipeline::GradientClipOptions::clip_by_norm(args.clip_grad_norm),
        )?;
        let total_norm = clip.total_norm;

        // ── Optimizer step ──────────────────────────────────────────
        trainer_pipeline::step_optimizer(&mut opt, &params, args.lr, || Ok(()))?;

        // ── Logging ─────────────────────────────────────────────────
        // Progress uses an absolute display target when --start-step is set.
        loop_run.record_and_log(
            step,
            trainer_pipeline::TrainStepMetrics {
                loss_value: loss_val,
                grad_norm: total_norm,
                learning_rate: args.lr,
            },
            board.as_ref(),
        );
        // Per-group breakdown: A = down-projection (random-init), B = up-
        // projection (zero-init). Watch lora_B/lora_A ratio over time —
        // healthy LoRA training drives B up from 0 within the first ~100
        // steps. If lora_B norm stays orders of magnitude below lora_A,
        // the trainer is identity-stuck regardless of total loss.
        println!(
            "[L2P-lora]   grad lora_A={:.3e}  grad lora_B={:.3e}  B/A={:.3e}",
            norm_a,
            norm_b,
            if norm_a > 0.0 { norm_b / norm_a } else { 0.0 },
        );
        if let Some(b) = board.as_ref() {
            let abs_step = (step + 1 + args.start_step) as u64;
            b.log_scalars(
                abs_step,
                &[
                    ("grad_norm/lora_A", norm_a as f64),
                    ("grad_norm/lora_B", norm_b as f64),
                ],
            );
        }

        // ── LoRA-B nonzero-ratio diagnostic at step 1 (paired with
        //    grad-flow). After one optimizer step, LoRA-B should have
        //    moved off zero on the modules that saw gradient.
        if step == 1 {
            let mut nonzero = 0usize;
            let mut total = 0usize;
            for (name, p) in &named {
                if !name.contains(".lora_B.weight") {
                    continue;
                }
                total += 1;
                if let Ok(t) = p.tensor() {
                    // abs.sum > 0 ⇒ at least one element has moved off zero
                    // (LoRA-B is zero-init so this is the correct test).
                    let s = t
                        .to_dtype(DType::F32)
                        .and_then(|f| f.mul(&f))
                        .and_then(|f| f.sum())
                        .and_then(|f| f.item())
                        .unwrap_or(0.0);
                    if s > 0.0 {
                        nonzero += 1;
                    }
                }
            }
            log::info!(
                "[lora-B-nonzero] step 1: {}/{} modules off-zero",
                nonzero,
                total
            );
        }

        // ── Periodic save ───────────────────────────────────────────
        // `step_num` is the global step (including --start-step offset)
        // so resumed runs save under filenames continuing from where the
        // previous run left off.
        let step_num = step + 1 + args.start_step;
        let total_target_steps = args.steps + args.start_step;
        if trainer_common::cadence_fires(args.save_every, step_num, total_target_steps) {
            let path = args
                .output
                .join(format!("l2p_lora_step{step_num}.safetensors"));
            if let Err(e) = save_lora_peft(&named, &path) {
                log::warn!("[save step {step_num}] {e}");
            } else {
                log::info!("[save step {step_num}] {}", path.display());
            }
        }

        // ── Periodic inline sample (config-driven) ──────────────────────
        // Sample-at-start (first training step) + every `--sample-every`
        // steps. Skipped on the final step (a final sample fires after the
        // loop). Mirrors train_klein's cadence. `sample_set` is empty when
        // inline sampling is disabled → loop is a no-op.
        if periodic
            && (step == 0
                || trainer_common::cadence_fires(args.sample_every, step_num, total_target_steps))
            && (step + 1) < args.steps
        {
            for (label, cap, unc, prompt) in &sample_set {
                let sample_out = args
                    .output
                    .join(format!("sample_step{step_num}_{label}.png"));
                log::info!(
                    "[sample step={step_num} {label}] \"{prompt}\" → {}",
                    sample_out.display()
                );
                if let Err(e) = l2p_inline_sample(
                    &mut model,
                    cap,
                    unc,
                    &sample_out,
                    args.sample_size,
                    args.sample_steps,
                    args.sample_cfg,
                    args.sample_shift,
                    args.sample_seed,
                    &device,
                ) {
                    log::warn!("[sample step={step_num} {label}] failed: {e}");
                } else if let Some(b) = board.as_ref() {
                    b.log_image_png(&format!("samples/{label}"), step_num as u64, 0, &sample_out);
                }
            }
        }
    }

    // ── Final save ────────────────────────────────────────────────────
    let final_step = args.steps + args.start_step;
    let final_path = args
        .output
        .join(format!("l2p_lora_step{}.safetensors", final_step));
    save_lora_peft(&named, &final_path)?;
    let completion = loop_run.finish();
    log::info!(
        "Training complete: {} steps, avg loss = {:.4} → {}",
        args.steps,
        completion.average_loss,
        final_path.display()
    );

    // ── Final inline sample (all config-driven prompts) ─────────────────
    if periodic {
        for (label, cap, unc, prompt) in &sample_set {
            let sample_out = args
                .output
                .join(format!("sample_step{}_{label}.png", final_step));
            log::info!(
                "[sample final {label}] \"{prompt}\" → {}",
                sample_out.display()
            );
            if let Err(e) = l2p_inline_sample(
                &mut model,
                cap,
                unc,
                &sample_out,
                args.sample_size,
                args.sample_steps,
                args.sample_cfg,
                args.sample_shift,
                args.sample_seed,
                &device,
            ) {
                log::warn!("[sample final {label}] failed: {e}");
            } else if let Some(b) = board.as_ref() {
                b.log_image_png(
                    &format!("samples/{label}"),
                    final_step as u64,
                    0,
                    &sample_out,
                );
            }
        }
    }

    trainer_pipeline::mark_board_completed(board.as_ref());
    Ok(())
}

/// Save LoRA in PEFT/ai-toolkit format. Each Parameter is written under
/// its already-namespaced key (`diffusion_model.<weight_key>.lora_A.weight`
/// / `...lora_B.weight`).
fn save_lora_peft(named: &[(String, Parameter)], path: &std::path::Path) -> anyhow::Result<()> {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (name, p) in named {
        let t = p.tensor()?;
        // The trainer stores down as [in, rank] and up as [rank, out] (native
        // matmul convention for `x_cast.matmul(&down).matmul(&up)`). PEFT
        // convention is PyTorch `nn.Linear` weight layout: lora_A.weight =
        // [rank, in], lora_B.weight = [out, rank]. The inference-side
        // `LoraStack::load` UNCONDITIONALLY transposes both — assuming
        // PEFT convention. Without this transpose-on-save the loader
        // double-transposes and inference produces shape errors at the
        // LoRA matmul. Fix landed 2026-05-22 after a 1000-step LoRA
        // produced bit-identical renders with/without --lora.
        let needs_transpose = name.ends_with(".lora_A.weight") || name.ends_with(".lora_B.weight");
        let t_out = if needs_transpose && t.shape().dims().len() == 2 {
            t.transpose()?.contiguous()?
        } else {
            t
        };
        // Cast to BF16 on save: training keeps LoRA Parameters in F32 (matches
        // ai-toolkit), but PEFT/diffusers/ai-toolkit-comfy-loader expect BF16
        // safetensors. F32 saves are 2× larger and not every consumer auto-
        // casts. F32 training precision is preserved by the optimizer state
        // (Adam moments stay F32 in memory); only the snapshot is downcast.
        let t_out = if t_out.dtype() != flame_core::DType::BF16 && t_out.shape().dims().len() == 2 {
            t_out.to_dtype(flame_core::DType::BF16)?
        } else {
            t_out
        };
        tensors.insert(name.clone(), t_out);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    save_file(&tensors, path).map_err(|e| anyhow::anyhow!("save_file: {e}"))?;
    // Silence the unused-import warning when save_tensors path isn't used.
    let _ = save_tensors
        as fn(
            &HashMap<String, Tensor>,
            &std::path::Path,
            SerializationFormat,
        ) -> flame_core::Result<()>;
    Ok(())
}

/// Resume: load LoRA weights from a PEFT-format safetensors written by
/// `save_lora_peft`, undo the save-time transpose, and copy the tensors
/// into the freshly-constructed LoRA Parameters in-place. Adam optimizer
/// state is NOT loaded — it reinitializes (Adam recovers in ~50-100 steps).
fn load_lora_resume(
    path: &std::path::Path,
    named: &[(String, Parameter)],
    device: &Arc<flame_core::CudaDevice>,
) -> anyhow::Result<()> {
    let loaded = flame_core::serialization::load_file(path, device)
        .map_err(|e| anyhow::anyhow!("load_file({}): {e}", path.display()))?;
    let mut hit = 0usize;
    let mut missing: Vec<&str> = Vec::new();
    for (name, p) in named {
        let saved = match loaded.get(name) {
            Some(t) => t,
            None => {
                missing.push(name);
                continue;
            }
        };
        // save_lora_peft writes PEFT layout: lora_A.weight = [rank, in],
        // lora_B.weight = [out, rank] (transposed from native [in, rank] /
        // [rank, out]). Undo that here so the in-memory Parameter keeps
        // the trainer's native shape.
        let needs_transpose = name.ends_with(".lora_A.weight") || name.ends_with(".lora_B.weight");
        let target_dtype = p.dtype()?;
        let t_native = if needs_transpose && saved.shape().dims().len() == 2 {
            saved.transpose()?.contiguous()?
        } else {
            saved.clone()
        };
        let t_native = if t_native.dtype() != target_dtype {
            t_native.to_dtype(target_dtype)?
        } else {
            t_native
        };
        // Shape sanity check before overwriting.
        if t_native.shape().dims() != p.shape().dims() {
            anyhow::bail!(
                "resume shape mismatch for {name}: file gives {:?} (after un-transpose) vs Parameter expects {:?}",
                t_native.shape().dims(),
                p.shape().dims(),
            );
        }
        p.set_data(t_native)
            .map_err(|e| anyhow::anyhow!("set_data({name}): {e}"))?;
        hit += 1;
    }
    if hit == 0 {
        anyhow::bail!(
            "resume from {} matched 0 of {} expected LoRA keys — wrong file?",
            path.display(),
            named.len(),
        );
    }
    if !missing.is_empty() {
        log::warn!(
            "[resume] {}/{} keys loaded; {} missing (first 3: {:?})",
            hit,
            named.len(),
            missing.len(),
            &missing[..missing.len().min(3)],
        );
    } else {
        log::info!("[resume] loaded all {} LoRA tensors", hit);
    }
    Ok(())
}

// -------------------------------------------------------------------------
// Inline sampling helpers (config-driven, mirrors train_klein).
// -------------------------------------------------------------------------

/// Load Qwen3-4B weights from a single safetensors file OR a shard
/// directory. Mirror of prepare_l2p's `load_qwen3_weights`.
fn l2p_load_qwen3(
    path: &std::path::Path,
    device: &Arc<flame_core::CudaDevice>,
) -> anyhow::Result<HashMap<String, Tensor>> {
    if path.is_file() {
        return Ok(flame_core::serialization::load_file(path, device)?);
    }
    let mut all = HashMap::new();
    for entry in std::fs::read_dir(path)? {
        let p = entry?.path();
        if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            let part = flame_core::serialization::load_file(&p, device)?;
            all.extend(part);
        }
    }
    if all.is_empty() {
        anyhow::bail!("no safetensors found at Qwen3 path {}", path.display());
    }
    Ok(all)
}

/// Encode one prompt to `cap_feats` `[1, valid_len, 2560]` BF16, exactly as
/// prepare_l2p does: Z-Image chat template, pad to `TXT_PAD_LEN` with the
/// pad token, encode, then strip pad positions (`narrow(1, 0, valid_len)`).
/// An empty prompt (the SampleLibrary default negative) yields the
/// unconditional embedding.
fn l2p_encode_prompt(
    qwen: &eridiffusion_core::encoders::qwen3::Qwen3Encoder,
    tok: &tokenizers::Tokenizer,
    prompt: &str,
) -> anyhow::Result<Tensor> {
    let wrapped = format!(
        "{ZIMAGE_TEMPLATE_PRE}{}{ZIMAGE_TEMPLATE_POST}",
        prompt.trim()
    );
    let enc = tok
        .encode(wrapped.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
    let valid_len = ids.len().min(TXT_PAD_LEN);
    ids.resize(TXT_PAD_LEN, PAD_TOKEN_ID);
    let text_hidden = qwen.encode(&ids)?.to_dtype(DType::BF16)?;
    Ok(text_hidden.narrow(1, 0, valid_len)?.contiguous()?)
}

/// Self-contained L2P inline sampler: Euler denoise loop in pixel-space,
/// then NCHW float [-1,1] → packed RGB u8 → PNG. Mirrors the orchestration
/// in `inference-flame/src/bin/l2p_infer.rs` (denoise + PNG) and klein's
/// `klein_inline_sample` (self-contained, defined in the trainer).
///
/// Runs under `AutogradContext::no_grad` so the multi-step denoise graph is
/// never retained — sampling must not pollute the training tape or leak
/// activation memory across the ~30 denoise steps.
#[allow(clippy::too_many_arguments)]
fn l2p_inline_sample(
    model: &mut L2pDiT,
    cap_feats: &Tensor,
    cap_feats_uncond: &Tensor,
    out_png: &std::path::Path,
    size: usize,
    steps: usize,
    cfg_scale: f32,
    shift: f32,
    seed: u64,
    device: &Arc<flame_core::CudaDevice>,
) -> anyhow::Result<()> {
    let _g = AutogradContext::no_grad();

    // F32 noise per L2P convention → cast to BF16 (the DiT debug-asserts BF16).
    let x_f32 = init_l2p_noise(size, size, seed, device)?;
    let mut x = x_f32.to_dtype(DType::BF16)?;
    drop(x_f32);

    let sigmas = build_l2p_sigma_schedule(steps, shift);
    let uncond_ref = if cfg_scale > 1.0 {
        Some(cap_feats_uncond)
    } else {
        None
    };
    for step in 0..steps {
        x = l2p_euler_step(
            model,
            &x,
            sigmas[step],
            sigmas[step + 1],
            cap_feats,
            uncond_ref,
            cfg_scale,
        )?;
    }

    // Pixel-space [-1,1] BF16 → F32 → packed RGB u8 (NCHW). Identical to
    // l2p_infer.rs:343-357.
    let rgb_f32 = x.to_dtype(DType::F32)?;
    drop(x);
    let data = rgb_f32.to_vec()?;
    let d = rgb_f32.shape().dims();
    let (out_h, out_w) = (d[2], d[3]);
    let mut pixels = vec![0u8; out_h * out_w * 3];
    for y in 0..out_h {
        for xp in 0..out_w {
            for c in 0..3 {
                let idx = c * out_h * out_w + y * out_w + xp;
                let val = (127.5 * (data[idx].clamp(-1.0, 1.0) + 1.0)) as u8;
                pixels[(y * out_w + xp) * 3 + c] = val;
            }
        }
    }
    let img = image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("failed to build RgbImage"))?;
    img.save(out_png)
        .map_err(|e| anyhow::anyhow!("save PNG {}: {e}", out_png.display()))?;
    Ok(())
}
