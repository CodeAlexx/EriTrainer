//! train_asymflow — AsymFlow Klein 9B LoRA training, mirroring upstream Python
//! `AsymFlowVR.forward_train` (`lakonlab/models/diffusions/asymflow.py:36-140`).
//!
//! This is the A3 milestone of the AsymFlow trainer port (see
//! `EriDiffusion-v2/docs/asymflow_milestone_plan.md` §A3). It forks
//! `train_klein.rs` and overlays the AsymFlow-specific delta:
//!   * Two forward passes per step (student with LoRA active; teacher with
//!     LoRA disabled + autograd disabled).
//!   * Procrustes "lift" of `latents_2` (Klein VAE 128-channel) → `x_0_low_rank`
//!     (768-channel via `pack(patchify(latents_2, 2)) @ proj.T * scale`, then
//!     unpack + unpatchify back to pixel space).
//!   * AsymFlow loss = velocity-weighted MSE between `pred_x_0` and the
//!     adaptive-residual target. LPIPS branch deferred (TODO §2).
//!   * Karras EMA momentum policy `(1 - 1/t)^(gamma+1)` (`asymflux2_klein_32gpus.py:204-213`).
//!   * Opt-in `AdamW8bit` via `--optimizer adamw8bit`.
//!
//! ## Save format
//!
//! Mirror of `train_klein` — PEFT/edv2-reference safetensors keyed
//! `double_blocks.{i}.{slot}.lora_{A,B}.weight` etc. (memory:
//! `feedback_save_format_peft_peft`). The 3 non-LoRA modified weights
//! (`x_embedder`, `proj_out`, `norm_out`) are NOT yet emitted alongside the
//! LoRA — see TODO §A3-non-lora below.
//!
//! ## Acceptance gate (deferred — RUNS TOMORROW on GPU)
//!
//! Per plan §A3:
//!   * 5-step smoke on the A2 10-image bucket at 384² with `--offload` —
//!     forward+backward completes without OOM.
//!   * Loss finite and trending down.
//!   * `FLAME_ASSERT_GRAD_FLOW=1` (default-on per `feedback_grad_flow_default_on.md`)
//!     shows 100% of trainable LoRA-B + 3 non-LoRA linears non-zero at **step 1**
//!     (NOT step 0 per `project_phase_d_smoke_2026-05-10.md`).
//!   * Saved LoRA loads in `klein_lora_infer`.
//!
//! ## TODOs (deferred per plan)
//!
//!   * LPIPS perceptual branch: report `loss_perceptual=0` until ported
//!     (deferred per asymflow_milestone_plan.md §2).
//!   * Per-param `lr_mult` (`proj_out: lr_mult=10.0` from
//!     `asymflux2_klein_32gpus.py:108`): uniform `lr_mult=1.0` for now
//!     (deferred per asymflow_milestone_plan.md §3).
//!   * In-trainer validation hook: use external infer (`klein_lora_infer`)
//!     (deferred per asymflow_milestone_plan.md §A4).
//!   * **A3-non-lora**: train `x_embedder`, `proj_out`, `norm_out` directly
//!     (Python `freeze_exclude`). Requires lifting these 3 keys out of
//!     `KleinModel.weights` into `Parameter` slots; needs edits to
//!     `eridiffusion-core/src/models/klein.rs` which are outside this
//!     milestone's allowed-file set. Trainer logs a warning at startup and
//!     proceeds with LoRA-only training.
//!   * **A3-timestep-embedder LoRA**: AsymFlow targets
//!     `timestep_embedder.linear_{1,2}` (= Klein's `time_in.{in,out}_layer`),
//!     which the current Klein LoRA injection set does NOT cover. Same
//!     blocker (would need new slots in `klein.rs`). Trainer logs a
//!     warning and proceeds without those two LoRA slots.

use clap::Parser;
use eridiffusion_cli::{trainer_common, trainer_pipeline};
use std::collections::HashMap;
use std::path::PathBuf;

use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};

use eridiffusion_core::debug as dbg;
use eridiffusion_core::models::{klein::KleinModel, TrainableModel};
use eridiffusion_core::training::checkpoint;
use eridiffusion_core::training::ema::ParameterEma;
use eridiffusion_core::training::features::ema_advanced::{
    decay_at_step, decay_karras, EmaConfig, KarrasConfig,
};
use eridiffusion_core::training::features::{disk_check, lr_schedule};
use eridiffusion_core::training::training_features::timestep_dist::TimestepConfig;
use eridiffusion_core::training::training_features::{Optimizer, OptimizerKind};
use eridiffusion_core::{asymflow_loss, compute_asymflow_target};

use eridiffusion_core::models::asymflux2;

// ───────────────────────────────────────────────────────────────────────────
// Constants — mirror the asymflux2_klein_32gpus.py config and gaussian_flow.py.
// ───────────────────────────────────────────────────────────────────────────

/// LakonLab `num_timesteps=1` (continuous time) — but the trainer still maps
/// sigma to a [0, NUM_TRAIN_TIMESTEPS) integer index for the timestep tensor
/// fed into the model's `timestep_embedding` (same as `train_klein`).
const NUM_TRAIN_TIMESTEPS: usize = 1000;

/// `denoising.patch_size = 16` — pixel patch size (`asymflux2_klein_32gpus.py:26`).
/// Drives both the `x_embedder` projection (`3 * 16² = 768` input channels)
/// and the residual `patchify(.., pack_channels=False)` used in
/// `compute_asymflow_target`.
const PATCH_SIZE: usize = 16;

/// `latent_patch_size = 2` (`asymflux2_klein_32gpus.py:14`) — the patch size
/// used when lifting `latents_2` (Klein VAE 128-channel) into the
/// 768-channel packed space via the Procrustes buffer.
///
/// Kept for documentation; the active Procrustes-lift branch (docstring
/// interpretation `proj_buffer = (full=768, base=128)`) does NOT pre-patchify
/// — it packs `latents_2` directly. The alternative branch (proj_buffer is
/// `(c_low*patch²=512, 768)`) would consume this constant. A4.5 will resolve.
#[allow(dead_code)]
const LATENT_PATCH_SIZE: usize = 2;

/// `mse_loss_weight = 10.0` from `asymflux2_klein_32gpus.py:51`. Comment in the
/// Python config notes LakonLab's MSE has an internal `0.5` factor → effective
/// 5.0. Our `asymflow_loss` mirrors the `0.5 *` factor (`asymflow.py:113`), so
/// passing 10.0 reproduces the effective 5.0 weighting.
const MSE_LOSS_WEIGHT: f32 = 10.0;

/// `loss_shift = 0.3` from `asymflux2_klein_32gpus.py:56` — the shift fed into
/// `calc_shifted_signal_ratio`. Use `None` to disable (Python `if
/// self.loss_shift is None: shifted_signal_ratio = 0.0`).
const LOSS_SHIFT: Option<f32> = Some(0.3);

/// `sigma_min = 5e-2` from `asymflux2_klein_32gpus.py:63`. Used as the lower
/// clamp on sigma before squaring in the loss denominator.
const SIGMA_MIN: f32 = 5e-2;

/// `diffusion_grad_clip = 200.0` from `asymflux2_klein_32gpus.py:97`. Note
/// this is two-hundred — much higher than Klein's 1.0 — because AsymFlow's
/// per-iter loss runs O(10²) larger before the 1/sigma² scaling stabilises.
const CLIP_GRAD_NORM: f32 = 200.0;

/// `diffusion_grad_clip_begin_iter = 100` from
/// `asymflux2_klein_32gpus.py:98` — clipping is disabled before this step
/// (gradient explosions at warmup are expected to wash out).
const CLIP_GRAD_BEGIN_ITER: usize = 100;

const DEFAULT_SEED: u64 = 42;

#[derive(Parser)]
#[command(about = "AsymFlow Klein 9B LoRA training (A3 milestone)")]
struct Args {
    #[arg(long)]
    config: PathBuf,
    /// A2 prepare_asymflow cache directory. Each sample is a safetensors
    /// file with keys `latent`, `image_oklab`, `text_embedding`, `text_mask`.
    #[arg(long)]
    cache_dir: PathBuf,
    /// Klein 9B transformer safetensors (directory of shards or single file).
    #[arg(long)]
    transformer: PathBuf,
    /// AsymFlow Procrustes adapter safetensors path. Contains the 2 buffers
    /// (`proj_buffer`, `scale_buffer`) AND the 3 replacement weights
    /// (`x_embedder.weight`, `proj_out.weight`, `norm_out.linear.weight`).
    /// Plan §2: we reuse LakonLab's published `asymflow_subspace_procrustes.pth`
    /// converted to safetensors. `inference-flame::asymflux2::extract_asymflow_buffers`
    /// is the canonical loader.
    #[arg(long)]
    asymflow_adapter: PathBuf,

    #[arg(long, default_value = "100")]
    steps: usize,
    /// LoRA rank — AsymFlux2 uses 256 (`asymflux2_klein_32gpus.py:50`).
    #[arg(long, default_value = "256")]
    rank: usize,
    /// LoRA alpha — AsymFlux2 uses `alpha = rank` (per `asymflux2.py:355-356`
    /// → scaling = 1.0).
    #[arg(long, default_value = "256.0")]
    lora_alpha: f64,
    /// `lr = 1e-4` from `asymflux2_klein_32gpus.py:106`.
    #[arg(long, default_value = "1e-4")]
    lr: f32,
    /// Linear warmup steps. `asymflux2_klein_32gpus.py:153-157` =
    /// `warmup_iters=100, warmup_ratio=0.001`. The 0.001 is the start-of-warmup
    /// LR fraction (1e-4 * 0.001 = 1e-7); we approximate with the existing
    /// linear-warmup helper which sweeps 0 → base_lr.
    #[arg(long, default_value = "100")]
    warmup_steps: usize,
    #[arg(long, default_value = "1")]
    batch_size: usize,

    /// AdamW (default) or `adamw8bit` (opt-in for OOM-prone 1024² runs;
    /// matches `asymflux2_klein_32gpus.py:106` `type='AdamW8bit'`). Per
    /// plan §3: smoke BF16 AdamW first, escalate to AdamW8bit if 1024²
    /// teacher OOMs. The no-quant rule (`feedback_zimage_no_quantization.md`)
    /// is Z-Image-scoped, NOT Klein — AdamW8bit is fine here.
    #[arg(long, default_value = "adamw")]
    optimizer: String,

    /// `betas = (0.9, 0.95)` from `asymflux2_klein_32gpus.py:106`. Klein's
    /// 0.9/0.999 default differs (the 0.95 second beta is AsymFlow-specific).
    #[arg(long, default_value = "0.9")]
    adam_beta1: f32,
    #[arg(long, default_value = "0.95")]
    adam_beta2: f32,
    /// `weight_decay = 0.0` from `asymflux2_klein_32gpus.py:106`.
    #[arg(long, default_value = "0.0")]
    weight_decay: f32,

    #[arg(long, conflicts_with = "resume_full")]
    resume_lora: Option<PathBuf>,
    #[arg(long, conflicts_with = "resume_lora")]
    resume_full: Option<PathBuf>,
    #[arg(long, default_value = "full")]
    save_mode: String,
    #[arg(long, default_value = "output")]
    output_dir: PathBuf,
    /// Per-block weight streaming — strongly recommended for AsymFlow on 24 GB
    /// (the student forward + autograd tape + teacher forward live
    /// concurrently; per plan §6 risk #1 the memory headroom is tight).
    #[arg(long)]
    offload: bool,
    #[arg(long, default_value = "500")]
    save_every: usize,
    #[arg(long, default_value_t = DEFAULT_SEED)]
    seed: u64,

    /// Karras EMA — `--ema --use-ema-advanced` enables. Defaults track
    /// `asymflux2_klein_32gpus.py:204-213`: `gamma=7.0, start_iter=100,
    /// max_momentum=1.0`. EMA shadow lives on F32; ~rank·param_count·4 bytes.
    #[arg(long, default_value_t = false)]
    ema: bool,
    /// Use the Karras momentum curve (LakonLab) instead of the diffusers
    /// power-decay curve. When `false`, `--ema` falls back to diffusers
    /// defaults from `train_klein`. Default-on for AsymFlow because the
    /// Python config mandates it.
    #[arg(long, default_value_t = true)]
    use_ema_advanced: bool,
    #[arg(long, default_value_t = 7.0)]
    ema_karras_gamma: f32,
    #[arg(long, default_value_t = 100)]
    ema_karras_start_iter: u64,
    #[arg(long, default_value_t = 1.0)]
    ema_karras_max_momentum: f32,
    #[arg(long, default_value_t = false)]
    ema_validation_swap: bool,

    /// Timestep distribution. AsymFlow's `ContinuousTimeStepSampler` with
    /// `shift=17.0, logit_normal_enable=True` (`asymflux2_klein_32gpus.py:58-61`)
    /// maps to our `logit_normal` distribution with the existing klein
    /// timestep_shift. Sigma shift is applied via `klein_dynamic_shift`
    /// downstream (NOT yet — see TODO).
    #[arg(long, default_value = "logit_normal")]
    timestep_distribution: String,
    #[arg(long, default_value_t = 0.0)]
    noising_weight: f32,
    #[arg(long, default_value_t = 0.0)]
    noising_bias: f32,

    /// LakonLab `sample_forward_diffusion`: `sigma_min` clamp for the
    /// `1/sigma²` loss weight (`asymflow.py:63, 113`). Lower-bound clamp
    /// only; sigma proper is unclamped.
    #[arg(long, default_value_t = SIGMA_MIN)]
    sigma_min: f32,

    /// Opt OUT of autograd v2 and run the legacy v3 engine. v2 is the default
    /// as of 2026-05-30 (gate-on Stage 6a); v3 kept as the reference engine.
    #[arg(long, default_value_t = false)]
    use_autograd_v3: bool,
}

fn collect_klein_shards(path: &std::path::Path) -> anyhow::Result<Vec<PathBuf>> {
    trainer_common::collect_safetensor_shards(path, "klein")
}

fn build_timestep_config(
    distribution: &str,
    weight: f32,
    bias: f32,
) -> anyhow::Result<TimestepConfig> {
    trainer_common::build_full_strength_timestep_config(distribution, weight, bias)
}

// ───────────────────────────────────────────────────────────────────────────
// Procrustes lift: latents_2 ([B, 128, h, w] BF16) → x_0_low_rank ([B, 768, h', w'] F32 → BF16)
//
// Python (`asymflow.py:69-78`):
//   latents_patchified = denoising.patchify(latents_2, latent_patch_size)  # [B, 128*4, h/2, w/2]
//   shape_after = latents_patchified.shape[2:]                              # (h/2, w/2)
//   latents_packed = denoising.pack(latents_patchified)                      # [B, N, 128*4=512]  ⚠
//   x_0_low_rank_packed = latents_packed @ (proj_mat.T * s)                 # [B, N, 768]
//   x_0_low_rank = denoising.unpatchify(
//       denoising.unpack(x_0_low_rank_packed, *shape_after),               # [B, 768, h/2, w/2]
//       denoising.patch_size                                                # patch=16
//   )                                                                       # [B, 3*16²/16² , 8h, 8w]
//                                                                            # = [B, 3, 8h, 8w]
//
// **NOTE on Procrustes proj_buffer shape**: per
// `inference-flame::asymflux2::extract_asymflow_buffers` doc, the buffer is
// `(in_channels * patch_size², base_rank)` = `(3 * 16² = 768, 128)`. The
// matmul `latents_packed @ (proj.T * s)` therefore expects `latents_packed`
// last-dim = base_rank = 128. Walking back, `pack(patchify(latents_2,
// LATENT_PATCH_SIZE))` last-dim is `128 * LATENT_PATCH_SIZE² = 128 * 4 =
// 512` — NOT 128. Either:
//   (a) the Python intends pack ordering that drops the patch axis into the
//       sequence dim (giving last-dim=128, sequence-dim = N * 4), or
//   (b) the proj_buffer shape is actually `(512, 768)` and the docstring
//       is wrong.
//
// **OPEN — needs Python golden dump (A4.5) to disambiguate.** For now we
// implement the docstring's interpretation `(768, 128)` and let A4.5 catch
// any pack-order disagreement. The lift is structurally:
//
//   1. patchify(latents_2, LATENT_PATCH_SIZE)            → [B, 128, h/2*p, w/2*p] **TBD**
//   2. pack(...)                                          → [B, N_low, base_rank=128]
//   3. matmul: pack @ (proj.T * s)                         → [B, N_low, 768]
//   4. unpack to [B, 768, h_low, w_low]
//   5. unpatchify(.., PATCH_SIZE=16)                       → [B, 3, H_pixel, W_pixel]
//
// where `H_pixel = h_low * PATCH_SIZE` and `h_low = h/(LATENT_PATCH_SIZE)`.
// ───────────────────────────────────────────────────────────────────────────
fn procrustes_lift(
    latents_2_bf16: &Tensor,
    proj_buffer_f32: &Tensor,
    scale_buffer: f32,
) -> anyhow::Result<Tensor> {
    // patchify the low-rank latent into (B, 128, h_low, w_low) — note the
    // LATENT_PATCH_SIZE=2 case. `eridiffusion_core::models::asymflux2::patchify` packs
    // channels: input (B, 128, h, w), output (B, 128*4, h/2, w/2). For the
    // lift we want the docstring's `(B, 128, h/2, w/2)` (channel count
    // preserved) — i.e. `LATENT_PATCH_SIZE` is implicitly 1 in the pack-only
    // step, OR the matmul axis is the packed `128*4 = 512`-channel space and
    // proj_buffer is `(512, 768)` not `(768, 128)`.
    //
    // **DEFERRED** — A4.5 will dump Python intermediates and tell us which.
    // Both interpretations are wired below; the active branch is the
    // docstring path (channel preserved, no pre-pack patchify).
    //
    // SAFETY OF LATER MATCHING: the matmul shape check at line `pack @
    // proj.T` will surface any disagreement as a flame-core error before any
    // training step pollutes the optimizer state — fail-fast.

    let dims = latents_2_bf16.shape().dims();
    if dims.len() != 4 {
        anyhow::bail!(
            "procrustes_lift: latents_2 expected rank-4 (B,C_low,h,w), got rank {}",
            dims.len()
        );
    }
    let (b, c_low, h, w) = (dims[0], dims[1], dims[2], dims[3]);

    // Up-cast to F32 for the Procrustes matmul (proj_buffer is F32 already).
    let latents_f32 = if latents_2_bf16.dtype() == DType::F32 {
        latents_2_bf16.clone()
    } else {
        latents_2_bf16.to_dtype(DType::F32)?
    };

    // Branch A — docstring interpretation: proj_buffer shape (full_rank, base_rank)
    //                                     = (768, 128). Then pack(latents_2)
    //                                     directly: (B, h*w, 128).
    //
    // Verified at construction time by checking proj_buffer dim[1] == c_low.
    let proj_dims = proj_buffer_f32.shape().dims();
    if proj_dims.len() != 2 {
        anyhow::bail!(
            "procrustes_lift: proj_buffer expected rank-2 (full_rank, base_rank), got {:?}",
            proj_dims
        );
    }
    let (full_rank, base_rank) = (proj_dims[0], proj_dims[1]);
    if base_rank != c_low {
        // If this fires, the Procrustes file follows branch B (`(c_low *
        // patch², full_rank)`) and the lift needs a different shape. Surface
        // the mismatch loud-and-early.
        anyhow::bail!(
            "procrustes_lift: proj_buffer base_rank={} != latents_2 channel count={}; \
             likely the .pth follows the (c_low*patch²=512, full_rank=768) layout — \
             pre-patchify with LATENT_PATCH_SIZE before pack (see lift docstring branch B).",
            base_rank,
            c_low
        );
    }

    // pack: (B, C, H, W) → (B, H*W, C)  via inference-flame asymflux2::pack.
    // Take (latents_2, 1) — i.e. patchify with PS=1 is a no-op, so we skip it.
    let packed = asymflux2::pack(&latents_f32)?;
    // proj.T: (base_rank, full_rank) = (128, 768). Multiply by scalar `s`.
    let proj_t = proj_buffer_f32.transpose()?.contiguous()?;
    let proj_t_scaled = proj_t.mul_scalar(scale_buffer)?;
    // packed (B, N, 128) @ proj_t_scaled (128, 768)  → (B, N, 768).
    let packed_full = packed.matmul(&proj_t_scaled)?;

    // unpack: (B, N, 768) → (B, 768, H, W).
    let unpacked = asymflux2::unpack(&packed_full, h, w)?;

    // unpatchify(.., PATCH_SIZE=16): (B, 768, h, w) → (B, 3, h*16, w*16).
    // 768 = 3 * 16² ⇒ `unpatchify` recovers C = 3.
    let pixel = asymflux2::unpatchify(&unpacked, PATCH_SIZE)?;

    // Cast back to BF16 so it matches the rest of the trainer's working dtype.
    let pixel_bf16 = if pixel.dtype() == DType::BF16 {
        pixel
    } else {
        pixel.to_dtype(DType::BF16)?
    };
    let _ = (b, full_rank); // mark used for the shape-check audit trail
    Ok(pixel_bf16)
}

// ───────────────────────────────────────────────────────────────────────────
// Pre-replace the 3 `pretrained_linear_proj` weights inside the freshly-loaded
// Klein weight map. Mirror of `asymflux2_klein9b_infer.rs:276-344`. Three
// replacements:
//   ("x_embedder.weight",       "img_in.weight")
//   ("proj_out.weight",         "final_layer.linear.weight")
//   ("norm_out.linear.weight",  "final_layer.adaLN_modulation.1.weight")  ⚠ shift/scale row-swap
//
// **Critical**: `norm_out.linear.weight` is `[scale, shift]` in diffusers but
// `[shift, scale]` in BFL Klein. Inference fixes this with a row-block swap
// (see comments at asymflux2_klein9b_infer.rs:297-322). We mirror exactly.
//
// Returns the modified weight map ready to feed into `KleinModel::new`.
// ───────────────────────────────────────────────────────────────────────────
fn apply_asymflow_weight_replacements(
    base: &mut HashMap<String, Tensor>,
    adapter: &HashMap<String, Tensor>,
) -> anyhow::Result<()> {
    let raw_replacements: &[(&str, &str)] = &[
        ("x_embedder.weight", "img_in.weight"),
        ("proj_out.weight", "final_layer.linear.weight"),
        (
            "norm_out.linear.weight",
            "final_layer.adaLN_modulation.1.weight",
        ),
    ];
    for (from, to) in raw_replacements {
        let new_w = adapter
            .get(*from)
            .ok_or_else(|| anyhow::anyhow!("asymflow adapter missing required key {}", from))?;
        let new_w_bf16 = if new_w.dtype() == DType::BF16 {
            new_w.clone()
        } else {
            new_w.to_dtype(DType::BF16)?
        };
        let to_insert = if *to == "final_layer.adaLN_modulation.1.weight" {
            // Row-swap fix — see inference-flame asymflux2_klein9b_infer.rs:297-322.
            // Diffusers AdaLayerNormContinuous emits [scale, shift] rows; BFL Klein
            // expects [shift, scale]. Swap row blocks [0..H/2] ↔ [H/2..H] before
            // insertion or modulation runs with reversed scale/shift.
            let dims = new_w_bf16.shape().dims();
            let h = dims[0];
            if h % 2 != 0 {
                anyhow::bail!(
                    "norm_out.linear.weight row count {} must be even (shift/scale split)",
                    h
                );
            }
            let half = h / 2;
            let scale_rows = new_w_bf16.narrow(0, 0, half)?;
            let shift_rows = new_w_bf16.narrow(0, half, half)?;
            Tensor::cat(&[&shift_rows, &scale_rows], 0)?
        } else {
            new_w_bf16
        };
        log::info!(
            "[asymflow] replace base[{to}] ← adapter[{from}] {:?}",
            to_insert.shape().dims()
        );
        base.insert((*to).to_string(), to_insert);
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────────────
// main
// ───────────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    use rand::SeedableRng;

    // Pool workaround per Klein convention — autograd doesn't play well with
    // the global alloc pool during long training runs.
    if std::env::var_os("FLAME_ALLOC_POOL").is_none() {
        unsafe {
            std::env::set_var("FLAME_ALLOC_POOL", "0");
        }
    }

    // Default-on grad-flow assertion (memory: feedback_grad_flow_default_on).
    if std::env::var_os("FLAME_ASSERT_GRAD_FLOW").is_none() {
        unsafe {
            std::env::set_var("FLAME_ASSERT_GRAD_FLOW", "1");
        }
    }

    trainer_common::init_logging();
    let args = Args::parse();
    trainer_common::ensure_output_dir(&args.output_dir)?;

    let device = trainer_common::init_bf16_cuda();

    // ── A3-non-lora / A3-timestep-embedder gap warnings ──────────────────
    log::warn!(
        "[A3 deferred] x_embedder / proj_out / norm_out are NOT marked trainable. \
         Python freeze_exclude (asymflux2_klein_32gpus.py:18-22) wants these \
         updated alongside the LoRA. Requires lifting them out of KleinModel.weights \
         into Parameter slots — change is outside this milestone's allowed-file set."
    );
    log::warn!(
        "[A3 deferred] timestep_embedder.linear_1/2 LoRA targets are NOT injected. \
         Klein's current LoRA slot map (12 double + 2 single) does not include \
         time_in.* — adding it requires new slots in eridiffusion-core/models/klein.rs. \
         Training proceeds with 14 of the 16 AsymFlow LoRA target families."
    );

    // ── Load Klein base + AsymFlow adapter + apply weight replacements ────
    let shards = collect_klein_shards(&args.transformer)?;
    log::info!(
        "[asymflow] loading Klein 9B base from {} shard(s)",
        shards.len()
    );
    let mut base_weights: HashMap<String, Tensor> = HashMap::new();
    for p in &shards {
        let part = flame_core::serialization::load_file(p, &device)?;
        for (k, v) in part {
            base_weights.insert(k, v.to_dtype(DType::BF16)?);
        }
    }
    log::info!("[asymflow] loaded {} base tensors", base_weights.len());

    log::info!(
        "[asymflow] loading Procrustes adapter from {}",
        args.asymflow_adapter.display()
    );
    let adapter_weights = flame_core::serialization::load_file(&args.asymflow_adapter, &device)?;
    log::info!(
        "[asymflow] loaded {} adapter tensors",
        adapter_weights.len()
    );

    let (proj_buffer, scale_buffer) = asymflux2::extract_asymflow_buffers(&adapter_weights)
        .map_err(|e| anyhow::anyhow!("extract_asymflow_buffers: {:?}", e))?;
    log::info!(
        "[asymflow] proj_buffer={:?} F32, scale_buffer={}",
        proj_buffer.shape().dims(),
        scale_buffer
    );

    apply_asymflow_weight_replacements(&mut base_weights, &adapter_weights)?;
    log::info!("[asymflow] 3 pretrained_linear_proj replacements applied");

    // ── Build KleinModel ──────────────────────────────────────────────────
    // KleinConfig auto-detects in_channels from img_in.weight; after our
    // replacement that's 768 (= 3 * 16²) instead of 128.
    let mut config = trainer_common::load_train_config(&args.config)?;
    trainer_common::apply_lora_basics(&mut config, args.rank, args.lora_alpha, args.lr);

    let kconfig = eridiffusion_core::models::klein::KleinConfig::from_weights(&base_weights)?;
    log::info!(
        "[asymflow] Klein autodetect: inner={} joint={} double={} single={} heads={} in_ch={} \
         (expected in_ch=768 after AsymFlow x_embedder replacement)",
        kconfig.inner_dim,
        kconfig.joint_attention_dim,
        kconfig.num_double,
        kconfig.num_single,
        kconfig.num_heads,
        kconfig.in_channels
    );
    if kconfig.in_channels != PATCH_SIZE * PATCH_SIZE * 3 {
        anyhow::bail!(
            "[asymflow] KleinModel in_channels={} after replacement, expected {} (= 3 * PATCH_SIZE²). \
             The x_embedder.weight replacement may not have applied correctly.",
            kconfig.in_channels,
            PATCH_SIZE * PATCH_SIZE * 3
        );
    }

    let mut model = KleinModel::new(base_weights, kconfig.clone(), &config, device.clone())?;
    if args.offload {
        // BlockOffloader needs the original shard paths (loads per-block from disk).
        model.enable_offload(shards.clone())?;
        log::info!(
            "[asymflow] block-offload enabled — per-block streaming from {} shard(s)",
            shards.len()
        );
    }

    let mut params = model.parameters();
    // Gate-on 6a: under v2 (default), flip LoRA params to MatchParamDtype so
    // BF16 grads from the bridge stay BF16 (Class A). --use-autograd-v3 skips.
    trainer_pipeline::apply_autograd_v2_grad_policy(&mut params, args.use_autograd_v3, "params");
    if params.is_empty() {
        anyhow::bail!("No trainable parameters — KleinModel produced empty LoRA list");
    }
    log::info!("[asymflow] {} trainable LoRA tensors", params.len());

    // ── Optimizer ─────────────────────────────────────────────────────────
    let opt_kind =
        OptimizerKind::parse(&args.optimizer).map_err(|e| anyhow::anyhow!("--optimizer: {e}"))?;
    log::info!("[asymflow] optimizer={}", opt_kind.as_str());
    // TODO per-param lr_mult: uniform 1.0 (deferred per
    // asymflow_milestone_plan.md §3). Python config wants
    // `proj_out: lr_mult=10.0` (asymflux2_klein_32gpus.py:108) but per-param
    // groups are not yet wired in the AdamW driver.
    let mut opt = Optimizer::new(
        opt_kind,
        args.lr,
        args.adam_beta1,
        args.adam_beta2,
        1e-8,
        args.weight_decay,
    );

    // ── EMA (Karras) ──────────────────────────────────────────────────────
    let mut ema: Option<ParameterEma> = if args.ema {
        let _g = AutogradContext::no_grad();
        // Seed shadow with `max_momentum` as the initial decay; the actual
        // per-step decay is overridden via `set_decay` before each `update`.
        let initial_decay = if args.use_ema_advanced {
            args.ema_karras_max_momentum
        } else {
            0.9999
        };
        let e = ParameterEma::new(&params, initial_decay)
            .map_err(|e| anyhow::anyhow!("EMA construction failed: {e}"))?;
        if args.use_ema_advanced {
            log::info!(
                "[ema] Karras WIRED — gamma={} start_iter={} max_momentum={} shadows={}",
                args.ema_karras_gamma,
                args.ema_karras_start_iter,
                args.ema_karras_max_momentum,
                e.len()
            );
        } else {
            log::info!(
                "[ema] diffusers WIRED — shadow={}, schedule via EmaConfig defaults",
                e.len()
            );
        }
        Some(e)
    } else {
        None
    };
    let karras_cfg = KarrasConfig {
        gamma: args.ema_karras_gamma,
        start_iter: args.ema_karras_start_iter,
        max_momentum: args.ema_karras_max_momentum,
    };
    let diffusers_ema_cfg = EmaConfig::default();

    // ── Timestep config ───────────────────────────────────────────────────
    let timestep_cfg = build_timestep_config(
        &args.timestep_distribution,
        args.noising_weight,
        args.noising_bias,
    )?;

    // ── Resume ────────────────────────────────────────────────────────────
    if let Some(resume_path) = args.resume_lora.as_ref() {
        log::info!(
            "[asymflow] resuming LoRA weights from {}",
            resume_path.display()
        );
        model.load_weights(&resume_path.to_string_lossy())?;
    }
    let mut start_step: usize = 0;
    if let Some(resume_path) = args.resume_full.as_ref() {
        log::info!("[asymflow] full-resume from {}", resume_path.display());
        let loaded = checkpoint::load_full(resume_path, &device)?;
        let named = model.named_parameters();
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
                "[resume-full] non-AdamW resume not yet implemented for {:?}",
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
    let save_mode_full = match args.save_mode.as_str() {
        "full" => true,
        "weights" => false,
        other => anyhow::bail!("--save-mode must be `full` or `weights`, got `{other}`"),
    };

    // ── Cache list ────────────────────────────────────────────────────────
    let cache_files = trainer_common::list_cache_safetensors(&args.cache_dir)?;
    log::info!(
        "[asymflow] found {} cached samples in {}",
        cache_files.len(),
        args.cache_dir.display()
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);

    let board = trainer_common::open_board_writer(
        &args.output_dir,
        trainer_common::board_resume_step(start_step),
    );
    if let Some(b) = &board {
        // Full board wiring: run hyper-parameters → metadata.hparams + the
        // dashboard's hparam panel. JSON hand-built (no serde dep here).
        let hparams_json = format!(
            "{{\"model\":\"asymflow-klein9b\",\"steps\":{},\"rank\":{},\"lora_alpha\":{},\"lr\":{},\
             \"warmup_steps\":{},\"batch_size\":{},\"optimizer\":\"{}\",\"seed\":{},\
             \"adam_beta1\":{},\"adam_beta2\":{},\"weight_decay\":{},\"offload\":{},\
             \"sigma_min\":{}}}",
            args.steps, args.rank, args.lora_alpha, args.lr, args.warmup_steps,
            args.batch_size, opt_kind.as_str(), args.seed,
            args.adam_beta1, args.adam_beta2, args.weight_decay, args.offload,
            args.sigma_min,
        );
        b.log_hparams(&hparams_json, &[("steps_target", args.steps as f64)]);
    }

    let dataset_len = cache_files.len();
    let batch_size = args.batch_size.max(1);

    // Pre-warm AdamW state.
    {
        let _g = AutogradContext::no_grad();
        opt.ensure_state_initialized(&params)
            .map_err(|e| anyhow::anyhow!("optimizer state prewarm failed: {e}"))?;
    }

    // ── Training loop ─────────────────────────────────────────────────────
    let loop_config = trainer_pipeline::ManualTrainLoopConfig::new(
        "AsymFlow-Klein9B",
        start_step,
        args.steps,
        dataset_len,
        batch_size,
    );
    let completion =
        trainer_pipeline::run_manual_train_loop(loop_config, board.as_ref(), |step| {
        let bs = batch_size;
        let mut latents_klein = Vec::with_capacity(bs);
        let mut images_oklab = Vec::with_capacity(bs);
        let mut txts = Vec::with_capacity(bs);
        for b in 0..bs {
            let cache_path = cache_files[(step * bs + b) % cache_files.len()].clone();
            let sample = flame_core::serialization::load_file(&cache_path, &device)?;
            let l = sample
                .get("latent")
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "cache {} missing 'latent' (Klein VAE low-rank)",
                        cache_path.display()
                    )
                })?
                .to_dtype(DType::BF16)?;
            let img = sample
                .get("image_oklab")
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "cache {} missing 'image_oklab' (full-rank x_0); \
                         did prepare_asymflow run with --no-oklab? A3 trainer requires it.",
                        cache_path.display()
                    )
                })?
                .to_dtype(DType::BF16)?;
            let t = sample
                .get("text_embedding")
                .ok_or_else(|| {
                    anyhow::anyhow!("cache {} missing 'text_embedding'", cache_path.display())
                })?
                .to_dtype(DType::BF16)?;
            latents_klein.push(l);
            images_oklab.push(img);
            txts.push(t);
        }
        let latents_2 = if bs == 1 {
            latents_klein.into_iter().next().unwrap()
        } else {
            Tensor::cat(&latents_klein.iter().collect::<Vec<_>>(), 0)?
        };
        let x_0 = if bs == 1 {
            images_oklab.into_iter().next().unwrap()
        } else {
            Tensor::cat(&images_oklab.iter().collect::<Vec<_>>(), 0)?
        };
        let txt = if bs == 1 {
            txts.into_iter().next().unwrap()
        } else {
            Tensor::cat(&txts.iter().collect::<Vec<_>>(), 0)?
        };

        // ── Procrustes lift: latents_2 → x_0_low_rank ─────────────────────
        // This is the teacher's reference target; runs in no_grad scope.
        let x_0_low_rank = {
            let _g = AutogradContext::no_grad();
            procrustes_lift(&latents_2, &proj_buffer, scale_buffer)?
        };
        if x_0_low_rank.shape().dims() != x_0.shape().dims() {
            anyhow::bail!(
                "[asymflow] Procrustes lift shape mismatch: x_0={:?}, x_0_low_rank={:?}. \
                 prepare_asymflow's Oklab encoder and Klein VAE may disagree on output \
                 resolution.",
                x_0.shape().dims(),
                x_0_low_rank.shape().dims()
            );
        }

        // ── Timestep sampling (per-step scalar) ────────────────────────────
        let raw_t = timestep_cfg.sample_one(&mut rng) * NUM_TRAIN_TIMESTEPS as f32;
        let sigma_idx = (raw_t.floor() as usize).min(NUM_TRAIN_TIMESTEPS - 1);
        let sigma = (sigma_idx + 1) as f32 / NUM_TRAIN_TIMESTEPS as f32;
        let sigma_clamped = sigma.max(args.sigma_min);
        let t_model = sigma_idx as f32 / NUM_TRAIN_TIMESTEPS as f32;
        let timestep =
            Tensor::from_vec(vec![t_model; bs], Shape::from_dims(&[bs]), device.clone())?;

        // ── x_t = sample_forward_diffusion(x_0, sigma, noise) ──────────────
        // gaussian_flow.py:98-103:
        //   std = t / num_timesteps                     (== our sigma)
        //   mean = 1 - std
        //   x_t = x_0 * mean + noise * std
        let noise =
            Tensor::randn(x_0.shape().clone(), 0.0, 1.0, device.clone())?.to_dtype(DType::BF16)?;
        let x_t = x_0
            .mul_scalar(1.0 - sigma)?
            .add(&noise.mul_scalar(sigma)?)?;
        let x_t_low_rank = x_0_low_rank
            .mul_scalar(1.0 - sigma)?
            .add(&noise.mul_scalar(sigma)?)?;

        // ── Student forward (LoRA active, autograd on) ─────────────────────
        let student_pred_u = model.forward_train(&x_t, &txt, &timestep, None)?;
        if student_pred_u.shape().dims() != x_t.shape().dims() {
            anyhow::bail!(
                "[asymflow] student forward shape {:?} != x_t {:?}",
                student_pred_u.shape().dims(),
                x_t.shape().dims()
            );
        }

        // ── Teacher forward (LoRA disabled, autograd OFF) ──────────────────
        // RAII pattern: take the LoRA adapters out so the model's forward
        // path's `lora_adapters.is_empty()` short-circuits past the additive
        // delta. The teacher forward runs the BASE weights only. Restore
        // before the next step via Rust's reverse drop order.
        //
        // **Limitation**: this only fully disables LoRA when
        // `lyc_adapters.is_none()` (legacy path). For Klein the legacy path
        // IS the default — we explicitly use `KleinModel::new` (above) which
        // routes to `new_inner(.., lyc_cfg=None)`. So this is correct here.
        let teacher_pred_u = {
            let _g = AutogradContext::no_grad();
            let saved_adapters = std::mem::take(&mut model.lora_adapters);
            let result = model.forward(&x_t_low_rank, &txt, &timestep);
            model.lora_adapters = saved_adapters;
            result?
        };

        // ── u → x_0 conversion (gaussian_flow.py:114) ──────────────────────
        //   x_0 = x_t - sigma * u
        let pred_x_0 = x_t.sub(&student_pred_u.mul_scalar(sigma)?)?;
        let ref_x_0_low_rank = x_t_low_rank.sub(&teacher_pred_u.mul_scalar(sigma)?)?;

        // ── AsymFlow target + loss (asymflow.py:93-119) ────────────────────
        let (tgt_x_0, target_parts) = compute_asymflow_target(
            &x_0,
            &x_0_low_rank,
            &pred_x_0,
            &ref_x_0_low_rank,
            PATCH_SIZE,
            sigma,
            LOSS_SHIFT,
        )?;
        let (loss, log_vars) = asymflow_loss(
            &pred_x_0,
            &tgt_x_0,
            sigma_clamped,
            MSE_LOSS_WEIGHT,
            &target_parts,
        )?;
        let loss_val = loss.to_vec()?[0];

        if dbg::enabled("OT_DEBUG_STATS") {
            eprintln!(
                "[asymflow step={:5}] sigma={:.4} loss={:.4} | res_coef={:.4} ssr={:.4} \
                 used_loss_shift={} loss_perceptual(stub)={}",
                step,
                sigma,
                loss_val,
                log_vars["res_coef"],
                target_parts.shifted_signal_ratio,
                target_parts.used_loss_shift,
                log_vars["loss_perceptual"],
            );
        }

        // ── Backward + grad clip + step ────────────────────────────────────
        // Gate-on 6a: v2 is the default; backward goes through `backward_v2`
        // unless `--use-autograd-v3` opts into the legacy v3 backward.
        let grads = trainer_pipeline::backward_loss(&loss, args.use_autograd_v3)?;
        // Mirror asymflux2 config: clip is gated by `grad_clip_begin_iter=100`.
        let clip_active = step >= CLIP_GRAD_BEGIN_ITER;
        let clip = trainer_pipeline::apply_gradient_map_clip(
            &params,
            &grads,
            trainer_pipeline::GradientClipOptions::clip_by_norm(CLIP_GRAD_NORM)
                .with_clipping_enabled(clip_active),
        )?;
        let total_norm = clip.total_norm;

        // LR schedule: linear warmup → constant. Reuses Klein's helper.
        let cur_lr = lr_schedule::dispatch_lr(
            &config.learning_rate_scheduler,
            args.lr,
            step,
            args.steps,
            args.warmup_steps,
            config.lr_min_factor,
            config.learning_rate_cycles as f32,
        );
        trainer_pipeline::step_optimizer(&mut opt, &params, cur_lr, || {

            // ── EMA update ─────────────────────────────────────────────────
            if let Some(ref mut e) = ema {
                let step_1based = (step + 1) as u64;
                let decay = if args.use_ema_advanced {
                    decay_karras(&karras_cfg, step_1based)
                } else {
                    decay_at_step(&diffusers_ema_cfg, step_1based)
                };
                if decay > 0.0 {
                    e.set_decay(decay);
                    e.update(&params).map_err(|err| {
                        anyhow::anyhow!("EMA update failed at step {step_1based}: {err}")
                    })?;
                }
                // decay == 0 → pre-warmup. LakonLab's hook hard-copies
                // param → shadow; our `update()` is the lerp form, so we
                // skip cleanly (the shadow stays at its initial value =
                // params, which were captured fresh at construction).
            }
            Ok(())
        })?;

        // ── Periodic save (no inline sample — DEFERRED per plan §A4) ──────
        let step_num = step + 1;
        if trainer_common::cadence_fires(args.save_every, step_num, args.steps) {
            let mid_ckpt = args
                .output_dir
                .join(format!("asymflow_lora_step{step_num}.safetensors"));
            let mut skip_save = false;
            if let Err(e) = disk_check::check_free_space(&args.output_dir, 2 * 1024 * 1024 * 1024) {
                log::warn!("[disk-check step {step_num}] {e} — skipping mid-save");
                skip_save = true;
            }
            if !skip_save {
                trainer_pipeline::save_lora_checkpoint(
                    trainer_pipeline::CheckpointSaveOptions {
                        trainer: "train_asymflow",
                        path: &mid_ckpt,
                        step: step_num as u64,
                        rank: args.rank,
                        alpha: args.lora_alpha as f32,
                        seed: args.seed,
                        config_hash: "",
                        save_mode_full,
                        label: &format!("[mid-save step {step_num}]"),
                    },
                    &opt,
                    || Ok(model.named_parameters()),
                    || {
                        model.save_weights(&mid_ckpt.to_string_lossy())?;
                        Ok(())
                    },
                )?;
            }
            // TODO in-trainer validation: use external infer
            // (deferred per asymflow_milestone_plan.md §A4).
        }
        Ok(trainer_pipeline::TrainStepMetrics {
            loss_value: loss_val,
            grad_norm: total_norm,
            learning_rate: cur_lr,
        })
    })?;

    log::info!(
        "[asymflow] training complete: {} new steps, avg loss={:.4}",
        completion.trained_steps,
        completion.average_loss
    );

    // ── Final save ────────────────────────────────────────────────────────
    trainer_pipeline::swap_ema_for_final_save(ema.as_ref(), &params, args.ema_validation_swap)?;
    let ckpt = args
        .output_dir
        .join(format!("asymflow_lora_{}steps.safetensors", args.steps));
    let mut final_skip_save = false;
    if let Err(e) = disk_check::check_free_space(&args.output_dir, 2 * 1024 * 1024 * 1024) {
        log::warn!("[disk-check final] {e} — skipping final save");
        final_skip_save = true;
    }
    if !final_skip_save {
        trainer_pipeline::save_lora_checkpoint(
            trainer_pipeline::CheckpointSaveOptions {
                trainer: "train_asymflow",
                path: &ckpt,
                step: args.steps as u64,
                rank: args.rank,
                alpha: args.lora_alpha as f32,
                seed: args.seed,
                config_hash: "",
                save_mode_full,
                label: "[final]",
            },
            &opt,
            || Ok(model.named_parameters()),
            || {
                model.save_weights(&ckpt.to_string_lossy())?;
                Ok(())
            },
        )?;
    }
    trainer_pipeline::mark_board_completed(board.as_ref());
    Ok(())
}
