//! train_hidream_o1 — HiDream-O1 (Qwen3-VL pixel-DiT) LoRA training (MVP).
//!
//! HiDream-O1 is a single-model pixel-level DiT — Qwen3-VL 8B text spine plus
//! three added heads (`x_embedder`, `t_embedder1`, `final_layer2`) that
//! operate on raw `PATCH_SIZE=32` RGB patches in `[-1, 1]`. There is NO VAE
//! and NO separate text encoder: `embed_tokens` runs inside the forward and
//! consumes `input_ids` directly. See
//! `EriDiffusion-v2/docs/hidream_o1_trainer_analysis.md` §1 for the full
//! refresher.
//!
//! Production training targets the **HiDream-O1-Image-Full** checkpoint. Older
//! local notes and Dev-weight defaults came from an accidental download of a
//! non-training/dev variant and should not be used for trainer parity.
//!
//! ## Cache contract (consumed from `prepare_hidream_o1` output)
//!
//! Each `sample_NNNNNN.safetensors` carries (all F32 on disk per
//! `prepare_hidream_o1` M2 — flame-core's safetensors writer dtype-erases):
//!
//!   - `patches`      [1, L, 3072]  pixel patches in `[-1, 1]`, L=(H/32)(W/32)
//!   - `input_ids`    [1, S_text]   Qwen3-VL chat-template ids incl. boi+tms
//!   - `position_ids` [3, S_total]  3D MRoPE T/H/W stacked
//!   - `vinput_mask`  [1, S_total]  1.0 at image slots (`token_types == 1`)
//!   - `token_types`  [1, S_total]  1.0 at image slots + TMS row (cache v2)
//!   - `image_grid`   [3]           (1, H/32, W/32) — unused here, kept for parity
//!
//! Top-level `_meta.json` carries `format: "hidream-o1-v2"` which we validate.
//!
//! ## Training step (aligned to edv2-reference `HidreamO1Model`, 2026-05-16)
//!
//! Source of truth:
//!   - `edv2-reference/extensions_built_in/diffusion_models/hidream/hidream_o1_model.py`
//!     `add_noise` (line 48-57) and `get_loss_target` (line 517-521).
//!   - `edv2-reference/extensions_built_in/diffusion_models/hidream/src/hidream_o1/pipeline.py`
//!     `DEFAULT_NOISE_SCALE = 8.0`, `T_EPS = 0.001`, `PATCH_SIZE = 32`.
//!
//! ```text
//!   noise_scale = 8.0          # DEFAULT_NOISE_SCALE
//!   u           ~ linspace(1.0, 0.001, 1000)[randint(0, 999)]
//!   t           = shift * u / (1 + (shift - 1) * u)
//!   t_eps       = 1e-3         # T_EPS
//!
//!   z    ~ N(0, 1), same shape as patches
//!   z_s  = z * noise_scale                        # SCALED noise
//!   x_t  = (1 - t) * patches + t * z_s            # add_noise()
//!   target = (z_s - patches).detach()             # velocity target, for parity/debug
//!
//!   x_pred = model.forward_lora(x_t, ids, pos, mask, ...)  # model emits x0-style
//!   # Convert model x0-style output to velocity to match `target`. From
//!   # `hidream_o1_model.py:469-473`:
//!   pred  = (x_t - x_pred) / max(t, t_eps)         # = z_s - patches if perfect
//!   # Production training defaults to velocity MSE because that is the
//!   # ai-toolkit O1 contract. Low-sigma samples can spike the logged loss
//!   # through the 1 / t^2 weighting; use the board's x0 loss only as a
//!   # diagnostic, not as the trained objective.
//! ```
//!
//! The conversion `pred = (x_t - x_pred)/t` (in float32) reproduces the
//! reference Python implementation's `get_noise_prediction` exactly.  The
//! `x0` objective is retained only for ablation/debug runs.
//!
//! ## Diagnostics
//!
//! Set `FLAME_ASSERT_GRAD_FLOW=1` to enable per-step LoRA grad-flow assertion
//! at step 1 (mirrors train_klein / train_chroma).  Per
//! `feedback_grad_flow_default_on.md` this should be the default; we surface
//! it in the doc rather than hard-coding `std::env::set_var` so users can
//! still opt out for production runs.
//!
//! ## Offload/checkpoint speed note (2026-05-20)
//!
//! HiDream-O1 now uses the Klein-style flame-core `BlockOffloader` path:
//! native `[Cout, Cin]` weights, `FLAME_LAYER_OFFLOAD_FRACTION=0.77` by
//! default, `plan_layer_access` in forward and checkpoint replay, and
//! `checkpoint_offload_boundary` around each decoder block. A 512 validation
//! run measured about 3.1 s/step after warmup.
//!
//! O1 attention is now structured instead of materialized: the decoder uses
//! flame-core `sdpa_prefix_causal_full`, so the AR/text prefix is causal and
//! image rows are full-attention without building the old `[B, 1, S, S]` mask.
//! Flame records the mixed attention as one custom autograd op: forward uses
//! prefix causal + suffix masked attention, while backward recomputes once as
//! the exact prefix-causal/full mask. This avoids the old two-SDPA shared-K/V
//! gradient collapse and the cuDNN plan failures seen on O1's non-aligned
//! suffix shape. Remaining speed/parity gaps are lower-layer/model-kernel
//! issues, not this SDPA mask path.

use clap::Parser;
use eridiffusion_cli::{trainer_common, trainer_pipeline};
use eridiffusion_core::models::hidream_o1::{
    build_mrope_positions, default_resident_target_keys, default_target_suffixes, HiDreamO1Config,
    HiDreamO1WeightLoader, LoraRegistry, MRopePositions,
};
use eridiffusion_core::training::checkpoint;
use eridiffusion_core::training::training_features::{Optimizer, OptimizerKind};
use flame_core::parameter::Parameter;
use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};
use std::collections::{BTreeSet, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};

const SEED: u64 = 42;
const CLIP_GRAD_NORM: f32 = 1.0;
const DEFAULT_MODEL_PATH: &str = "/home/alex/HiDream-O1-Image-Full-weights";
/// Reference `DEFAULT_NOISE_SCALE` (pipeline.py:15). Noise is scaled by this
/// factor in both add_noise and the loss target.
const NOISE_SCALE: f32 = 8.0;
/// Reference `T_EPS` (pipeline.py:16). Lower bound on `t` (and `1-t`) to
/// avoid the divide-by-zero in `(x_t - x_pred)/t`.
const T_EPS_AT: f32 = 1.0e-3;
/// Reference flowmatch scheduler `shift` (hidream_o1_model.py:33-36 +
/// `set_train_timesteps`@sampler.py:161). With `use_dynamic_shifting=False`,
/// the shift mapping is `sigma_shifted = shift * u / (1 + (shift-1) * u)`.
const FLOW_SHIFT: f32 = 3.0;
/// Number of entries in the ai-toolkit flowmatch sigma grid
/// (`custom_flowmatch_sampler.py:136` `num_timesteps = 1000`).
const SHIFT_GRID_NUM: usize = 1000;
/// Pre-shift sigma grid maximum, mirroring ai-toolkit's
/// `_sigma_to_t(sigma_max) / num_train_timesteps` for
/// `FlowMatchEulerDiscreteScheduler(shift=3.0)`:
/// `_sigma_to_t(1.0) = 1000.0`, `/1000 = 1.0` (`custom_flowmatch_sampler.py:137,141`).
const SHIFT_SIGMA_MAX: f64 = 1.0;
/// Pre-shift sigma grid minimum, mirroring ai-toolkit's
/// `_sigma_to_t(sigma_min) / num_train_timesteps`. The diffusers
/// `FlowMatchEulerDiscreteScheduler(shift=3.0)` computes
/// `sigma_min = 0.002994012087583542`, and `_sigma_to_t(sigma_min) = 2.994012087583542`,
/// `/1000 = 0.002994012087583542` (`custom_flowmatch_sampler.py:138,141`).
/// PREVIOUS BUG: ours used `(1000 - idx) / 1000`, i.e. a linspace floor of
/// 0.001 instead of 0.002994, letting the reachable shifted sigma drop to
/// ~0.005976 (idx=998) vs ai-toolkit's ~0.011881 — over-amplifying the
/// 1/sigma velocity weighting at low sigma. See AUDIT/SKEPTIC_HIDREAM_2026-05-25.
const SHIFT_SIGMA_MIN: f64 = 0.002994012087583542;

#[derive(Parser)]
struct Args {
    /// TrainConfig JSON file (optional). Falls back to TrainConfig::default().
    #[arg(long)]
    config: Option<PathBuf>,
    /// Cache dir written by prepare_hidream_o1 (contains `_meta.json` +
    /// per-sample `.safetensors`).
    #[arg(long)]
    cache_dir: PathBuf,
    /// Optional max total sequence length (`vinput_mask.shape[-1]`) for 24GB
    /// runs. Overlong cached samples are skipped before training starts.
    #[arg(long, default_value_t = 0)]
    max_seq_len: usize,
    /// Disable ai-toolkit-style per-epoch cache shuffling. The default keeps
    /// shuffled epochs even though samples are cached on disk.
    #[arg(long, default_value_t = false)]
    no_cache_shuffle: bool,
    /// HiDream-O1 model dir (containing `model.safetensors.index.json` +
    /// shards + `tokenizer.json`).
    #[arg(long, default_value = DEFAULT_MODEL_PATH)]
    model_path: PathBuf,
    #[arg(long, default_value = "768")]
    steps: usize,
    /// Global step offset for LoRA-only resume runs. Example: resume a
    /// step-1000 LoRA with `--start-step 1000 --steps 1000` to continue cache
    /// order and save the final checkpoint as 2000 steps.
    #[arg(long, default_value_t = 0)]
    start_step: usize,
    /// LoRA rank. Reference yaml default: 32 (`train_lora_hidream_48.yaml:26`).
    #[arg(long, default_value = "32")]
    rank: usize,
    /// LoRA alpha. Reference yaml default: 32 (`train_lora_hidream_48.yaml:27`).
    #[arg(long, default_value = "32.0")]
    lora_alpha: f32,
    /// Explicit ablation: also train O1 head adapters
    /// (`x_embedder`, `t_embedder1`, `final_layer2`). Disabled by default
    /// because ai-toolkit's current HiDream-O1 training surface saves 252
    /// Qwen language-layer adapters.
    #[arg(long, default_value_t = false, conflicts_with = "no_resident_lora")]
    include_resident_lora: bool,
    /// Compatibility no-op for older scripts. The default already skips O1
    /// head adapters to match ai-toolkit training.
    #[arg(
        long,
        default_value_t = false,
        conflicts_with = "include_resident_lora"
    )]
    no_resident_lora: bool,
    /// Learning rate. Kept conservative for O1; override explicitly when
    /// matching a config that uses a higher value.
    #[arg(long, default_value = "1e-4")]
    lr: f32,
    /// Save a LoRA checkpoint every N steps (0 = end-only). Reference yaml
    /// default: 250 (`train_lora_hidream_48.yaml:36` → `save_every: 250`).
    #[arg(long, default_value = "250")]
    save_every: usize,
    /// In-trainer sampling cadence. **Deferred to O1-M4.** Any non-zero value
    /// logs a warning and is ignored. Use `hidream_o1_infer` externally with
    /// a saved LoRA to visualize progress.
    #[arg(long, default_value = "0")]
    sample_every: usize,
    #[arg(long, default_value = "output")]
    output_dir: PathBuf,
    /// `weights` (default — LoRA-only safetensors) or `full` (LoRA + AdamW
    /// state + step counter). Full checkpoint state is available only for
    /// `--optimizer adamw`; other optimizers fail loudly instead of writing a
    /// checkpoint that looks full but only contains weights.
    #[arg(long, default_value = "weights")]
    save_mode: String,
    /// Resume LoRA weights only.
    #[arg(long, conflicts_with = "resume_full")]
    resume_lora: Option<PathBuf>,
    /// Resume LoRA + AdamW state + step counter.
    /// Full AdamW state restore is available only for `--optimizer adamw`.
    #[arg(long, conflicts_with = "resume_lora")]
    resume_full: Option<PathBuf>,

    // ── Training-step knobs ─────────────────────────────────────────────
    /// Timestep distribution. Default `shift` matches the edv2-reference HiDream-O1
    /// reference config and the flowmatch scheduler config (`shift=3.0`).
    /// `linear` and `uniform` are retained only for ablation.
    #[arg(long, default_value = "shift")]
    timestep_distribution: String,
    /// Flowmatch shift constant. Reference default = 3.0 (per the scheduler
    /// kwargs in `hidream_o1_model.py:32-36`). Only used when
    /// `--timestep-distribution shift`.
    #[arg(long, default_value_t = FLOW_SHIFT)]
    flow_shift: f32,
    /// Optimizer. Reference default = `adamw8bit`
    /// (`train_lora_hidream_48.yaml:57`). HiDream-O1 yaml passes only
    /// `optimizer: "adamw8bit"` and `lr: 2e-4`, no `optimizer_params`. The
    /// downstream `bitsandbytes.optim.AdamW8bit(params, lr, eps=1e-6)` call
    /// (`edv2-reference/toolkit/optimizer.py:71`) therefore takes bitsandbytes
    /// defaults for betas=(0.9, 0.999) and weight_decay=1e-2; only `eps` is
    /// overridden (`1e-6` instead of the torch `1e-8` default).
    ///
    /// Default = `adamw8bit` to match the reference HiDream-O1 config. Use
    /// `adamw` for optimizer ablations and for full-state checkpoints.
    #[arg(long, default_value = "adamw8bit")]
    optimizer: String,
    /// AdamW β1 momentum coefficient. Default = 0.9 (bitsandbytes / torch
    /// AdamW default — edv2-reference's HiDream-O1 yaml does not override).
    #[arg(long, default_value_t = 0.9)]
    adamw_beta1: f32,
    /// AdamW β2 second-moment coefficient. Default = 0.999 (bitsandbytes /
    /// torch AdamW default — edv2-reference's HiDream-O1 yaml does not override).
    #[arg(long, default_value_t = 0.999)]
    adamw_beta2: f32,
    /// AdamW ε. Default = 1e-6 (edv2-reference `optimizer.py:67,71,77,79`
    /// hard-codes this for every Adam-family path, overriding torch's 1e-8).
    #[arg(long, default_value_t = 1.0e-6)]
    adamw_eps: f32,
    /// AdamW weight decay. bitsandbytes AdamW8bit default = 0.01, and
    /// ai-toolkit's O1 config does not override it.
    #[arg(long, default_value_t = 1.0e-2)]
    adamw_weight_decay: f32,
    /// Clamp scalar loss before backward. Disabled by default; the reference
    /// TrainConfig leaves `max_loss` unset, and clamping a scalar MSE
    /// above the cap has zero derivative, silently turning those samples into
    /// no-op optimizer steps.
    #[arg(long, default_value_t = 0.0)]
    max_loss: f32,
    /// Loss objective. `velocity` matches ai-toolkit HiDream-O1:
    /// target = noise * noise_scale - patches and pred = (x_t - x0_pred)/sigma.
    /// `x0` is retained only as an ablation/debug path.
    #[arg(long, default_value = "velocity")]
    loss_objective: String,
    /// Probability of dropping the caption to an empty prompt on each sample.
    /// ai-toolkit's Eri2 O1 config uses 0.05. Because EDv2 caches token ids,
    /// the trainer rebuilds the empty prompt stream on demand for the current
    /// image grid instead of requiring a second cache.
    #[arg(long, default_value_t = 0.05)]
    caption_dropout_probability: f32,
    /// Let optimizer steps proceed when some LoRA parameters have no gradient.
    /// This is only for debugging partial adapter surfaces; production should
    /// fail fast because missing grads usually mean a detach, wrong key, or
    /// offload/checkpoint routing bug.
    #[arg(long, default_value_t = false)]
    allow_missing_lora_grads: bool,
    /// Log aggregate LoRA A/B magnitudes every N optimizer steps. Set 0 to
    /// disable. This catches A-collapse or B-overgrowth early in long runs.
    #[arg(long, default_value_t = 100)]
    lora_stats_every: usize,
    /// Scale LoRA B matrices only when writing weights-only exports. The
    /// in-memory train state and full checkpoints remain raw so resume math is
    /// unchanged. Production LoRA exports must default to the trained delta
    /// (`1.0`); use lower values only for explicit debug strength sweeps.
    #[arg(long, default_value_t = 1.0)]
    export_scale: f32,

    /// Opt OUT of autograd v2 and run the legacy v3 engine. v2 is the default
    /// as of 2026-05-30 (gate-on Stage 6a); v3 kept as the reference engine.
    #[arg(long, default_value_t = false)]
    use_autograd_v3: bool,
}

/// Distribution mode for `t` sampling. `Shift` mirrors the reference
/// HiDream-O1 configs. `Linear` and `Uniform` are retained for
/// ablations.
#[derive(Clone, Copy, Debug)]
enum TstepMode {
    Linear,
    Uniform,
    Shift,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LossObjective {
    X0,
    Velocity,
}

fn parse_loss_objective(s: &str) -> anyhow::Result<LossObjective> {
    match s.trim().to_ascii_lowercase().as_str() {
        "x0" | "clean" | "clean-patch" | "clean_patch" => Ok(LossObjective::X0),
        "velocity" | "vel" | "reference" | "parity" => Ok(LossObjective::Velocity),
        other => anyhow::bail!("--loss-objective: expected `x0` or `velocity`, got `{other}`"),
    }
}

fn parse_tstep_mode(s: &str) -> anyhow::Result<TstepMode> {
    match s.trim().to_ascii_lowercase().as_str() {
        "linear" => Ok(TstepMode::Linear),
        "uniform" => Ok(TstepMode::Uniform),
        "shift" | "flowmatch" => Ok(TstepMode::Shift),
        other => anyhow::bail!(
            "--timestep-distribution: expected `linear` (default), `shift`, or `uniform`, got `{other}`"
        ),
    }
}

/// Sample `t in (t_eps, 1 - t_eps)` from the configured distribution.
fn sample_t<R: rand::Rng>(rng: &mut R, mode: TstepMode, shift: f32) -> f32 {
    match mode {
        // edv2-reference `CustomFlowMatchScheduler.set_train_timesteps(linear)`
        // creates timesteps 1000..1, then the balanced sampler draws integer
        // indices in [0, 999). This gives sigma/t in [1.0, 0.002].
        TstepMode::Linear => {
            let idx: usize = rng.gen_range(0..999);
            (1000.0 - idx as f32) / 1000.0
        }
        TstepMode::Uniform => rng.r#gen::<f32>().clamp(T_EPS_AT, 1.0 - T_EPS_AT),
        // Flow-matching shift grid — bit-mirror of ai-toolkit
        // `custom_flowmatch_sampler.py:136-161` for the `shift` timestep type
        // with `use_dynamic_shifting=False`, `shift_terminal=None`, and no
        // karras/exponential/beta/invert (all defaults for the HiDream-O1
        // scheduler, verified against `FlowMatchEulerDiscreteScheduler(shift=3.0)`):
        //
        //   timesteps = np.linspace(_sigma_to_t(sigma_max),    # = 1000.0
        //                           _sigma_to_t(sigma_min),     # = 2.994012087583542
        //                           num_timesteps)              # = 1000, endpoints inclusive
        //   sigmas    = timesteps / num_train_timesteps         # → linspace(1.0, 0.002994…)
        //   sigmas    = shift * sigmas / (1 + (shift - 1) * sigmas)   # :161
        //
        // numpy linspace value at index i (both endpoints inclusive):
        //   sigma_raw[i] = sigma_max + i * (sigma_min - sigma_max) / (num - 1)
        // The `balanced` sampler then draws idx via `torch.randint(0, 999)` ⇒
        // idx ∈ [0, 998]; we mirror that with `gen_range(0..SHIFT_GRID_NUM-1)`.
        // PREVIOUS BUG used `(1000 - idx)/1000` (linspace floor 0.001), giving
        // a reachable shifted sigma ~0.005976 vs ai-toolkit's ~0.011881.
        TstepMode::Shift => {
            let idx: usize = rng.gen_range(0..(SHIFT_GRID_NUM - 1));
            // numpy/torch build the grid in float64-equivalent precision; do
            // the same and cast once at the end so the grid is bit-identical.
            let n = (SHIFT_GRID_NUM - 1) as f64;
            let sigma_raw =
                SHIFT_SIGMA_MAX + (idx as f64) * (SHIFT_SIGMA_MIN - SHIFT_SIGMA_MAX) / n;
            let s = shift as f64;
            let sigma_shifted = s * sigma_raw / (1.0 + (s - 1.0) * sigma_raw);
            (sigma_shifted as f32).clamp(T_EPS_AT, 1.0 - T_EPS_AT)
        }
    }
}

fn apply_chat_template_t2i(prompt: &str) -> String {
    let mut s = String::new();
    s.push_str("<|im_start|>user\n");
    s.push_str(prompt);
    s.push_str("<|im_end|>\n");
    s.push_str("<|im_start|>assistant\n");
    s.push_str("<|boi_token|>");
    s.push_str("<|tms_token|>");
    s
}

fn build_prompt_conditioning(
    tokenizer: &tokenizers::Tokenizer,
    config: &HiDreamO1Config,
    prompt: &str,
    h_patches: usize,
    w_patches: usize,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<(Tensor, Tensor, Tensor, Tensor)> {
    if h_patches == 0 || w_patches == 0 {
        anyhow::bail!("invalid image grid {h_patches}x{w_patches}");
    }
    let image_len = h_patches * w_patches;
    let template = apply_chat_template_t2i(prompt);
    let enc = tokenizer
        .encode(template.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenize failed: {e}"))?;
    let text_ids: Vec<u32> = enc.get_ids().to_vec();
    let txt_seq_len = text_ids.len();

    let mut full_ids: Vec<u32> = Vec::with_capacity(txt_seq_len + image_len);
    full_ids.extend_from_slice(&text_ids);
    full_ids.push(config.vision_start_token_id);
    for _ in 1..image_len {
        full_ids.push(config.image_token_id);
    }
    let all_seq_len = full_ids.len();

    let (t_pos, h_pos, w_pos) = build_mrope_positions(
        &full_ids,
        config.image_token_id,
        config.video_token_id,
        config.vision_start_token_id,
        &[(1, h_patches, w_patches)],
        &[1],
        Some(config.fix_point),
    );

    let input_ids_f: Vec<f32> = text_ids.iter().map(|&id| id as f32).collect();
    let input_ids = Tensor::from_vec(
        input_ids_f,
        Shape::from_dims(&[1, txt_seq_len]),
        device.clone(),
    )?
    .to_dtype(DType::I32)?;

    let mut pos_f: Vec<f32> = Vec::with_capacity(3 * all_seq_len);
    pos_f.extend(t_pos.iter().map(|&v| v as f32));
    pos_f.extend(h_pos.iter().map(|&v| v as f32));
    pos_f.extend(w_pos.iter().map(|&v| v as f32));
    let position_ids =
        Tensor::from_vec(pos_f, Shape::from_dims(&[3, all_seq_len]), device.clone())?;

    let mut vmask = vec![0.0_f32; all_seq_len];
    for i in txt_seq_len..(txt_seq_len + image_len) {
        vmask[i] = 1.0;
    }
    let vinput_mask = Tensor::from_vec(vmask, Shape::from_dims(&[1, all_seq_len]), device.clone())?
        .to_dtype(DType::BF16)?;

    let mut token_types = vec![0.0_f32; all_seq_len];
    for i in txt_seq_len..(txt_seq_len + image_len) {
        token_types[i] = 1.0;
    }
    if txt_seq_len > 0 {
        token_types[txt_seq_len - 1] = 1.0;
    }
    let token_types_bin = Tensor::from_vec(
        token_types,
        Shape::from_dims(&[1, all_seq_len]),
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;

    Ok((input_ids, position_ids, vinput_mask, token_types_bin))
}

fn decode_image_grid(grid: &Tensor) -> anyhow::Result<(usize, usize)> {
    let dims = grid.shape().dims().to_vec();
    if dims.as_slice() != [3] {
        anyhow::bail!("image_grid: expected [3], got {:?}", dims);
    }
    let v = grid.to_dtype(DType::F32)?.to_vec_f32()?;
    let h = v[1].round() as usize;
    let w = v[2].round() as usize;
    if !v[1].is_finite() || !v[2].is_finite() || h == 0 || w == 0 {
        anyhow::bail!("image_grid: invalid values {:?}", v);
    }
    Ok((h, w))
}

fn expected_lora_adapter_keys(cfg: &HiDreamO1Config, include_resident: bool) -> Vec<String> {
    let mut keys = Vec::new();
    for layer_idx in 0..cfg.num_layers {
        for suffix in default_target_suffixes() {
            keys.push(format!("layers.{layer_idx}.{suffix}"));
        }
    }
    if include_resident {
        keys.extend(
            default_resident_target_keys()
                .iter()
                .map(|k| (*k).to_string()),
        );
    }
    keys.sort();
    keys
}

fn preview_keys(keys: &[String]) -> String {
    let mut out = keys.iter().take(12).cloned().collect::<Vec<_>>().join(", ");
    if keys.len() > 12 {
        out.push_str(&format!(", ... ({} total)", keys.len()));
    }
    out
}

fn validate_lora_surface(
    lora: &LoraRegistry,
    cfg: &HiDreamO1Config,
    include_resident: bool,
) -> anyhow::Result<()> {
    let expected: BTreeSet<String> = expected_lora_adapter_keys(cfg, include_resident)
        .into_iter()
        .collect();
    let actual: BTreeSet<String> = lora.adapter_keys().into_iter().collect();
    let missing: Vec<String> = expected.difference(&actual).cloned().collect();
    let extra: Vec<String> = actual.difference(&expected).cloned().collect();
    if !missing.is_empty() || !extra.is_empty() {
        anyhow::bail!(
            "LoRA adapter surface mismatch: expected {} adapters, got {}. missing=[{}] extra=[{}]",
            expected.len(),
            actual.len(),
            preview_keys(&missing),
            preview_keys(&extra)
        );
    }
    Ok(())
}

fn require_lora_grad_coverage(
    grads: &flame_core::GradientMap,
    named: &[(String, Parameter)],
    step_num: usize,
    allow_missing_zero_a: bool,
) -> anyhow::Result<()> {
    let mut missing = Vec::new();
    let mut allowed_a = 0usize;
    for (name, p) in named {
        if grads.contains(p.id()) {
            continue;
        }
        if allow_missing_zero_a && name.ends_with(".lora_A") {
            allowed_a += 1;
        } else {
            missing.push(name.clone());
        }
    }
    if !missing.is_empty() {
        anyhow::bail!(
            "missing LoRA gradients at step {step_num}: {} of {} parameters absent. [{}]",
            missing.len(),
            named.len(),
            preview_keys(&missing)
        );
    }
    if allowed_a > 0 {
        log::warn!(
            "[grad-coverage] step {step_num}: allowed {allowed_a} missing LoRA A grads on fresh zero-B init"
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct LoraMagnitudeStats {
    adapters: usize,
    a_mean_abs: f64,
    b_mean_abs: f64,
    a_max_abs: f32,
    b_max_abs: f32,
}

fn tensor_abs_stats(t: &Tensor) -> anyhow::Result<(f64, f32, usize)> {
    let values = t.to_dtype(DType::F32)?.to_vec_f32()?;
    let mut sum = 0.0f64;
    let mut max = 0.0f32;
    for v in values.iter().copied() {
        let a = v.abs();
        sum += a as f64;
        max = max.max(a);
    }
    Ok((sum, max, values.len()))
}

fn lora_magnitude_stats(lora: &LoraRegistry) -> anyhow::Result<LoraMagnitudeStats> {
    let mut a_sum = 0.0f64;
    let mut b_sum = 0.0f64;
    let mut a_max = 0.0f32;
    let mut b_max = 0.0f32;
    let mut a_count = 0usize;
    let mut b_count = 0usize;
    let mut adapters = 0usize;

    for (_key, a, b) in lora.iter_trainable() {
        let (s, m, n) = tensor_abs_stats(&a.tensor()?)?;
        a_sum += s;
        a_max = a_max.max(m);
        a_count += n;

        let (s, m, n) = tensor_abs_stats(&b.tensor()?)?;
        b_sum += s;
        b_max = b_max.max(m);
        b_count += n;
        adapters += 1;
    }

    Ok(LoraMagnitudeStats {
        adapters,
        a_mean_abs: if a_count > 0 {
            a_sum / a_count as f64
        } else {
            0.0
        },
        b_mean_abs: if b_count > 0 {
            b_sum / b_count as f64
        } else {
            0.0
        },
        a_max_abs: a_max,
        b_max_abs: b_max,
    })
}

fn log_lora_magnitude_stats(label: &str, lora: &LoraRegistry) -> anyhow::Result<()> {
    let s = lora_magnitude_stats(lora)?;
    log::info!(
        "[lora-stats] {label}: adapters={} A.mean_abs={:.6e} B.mean_abs={:.6e} \
         A.max_abs={:.6e} B.max_abs={:.6e} B/A={:.4}",
        s.adapters,
        s.a_mean_abs,
        s.b_mean_abs,
        s.a_max_abs,
        s.b_max_abs,
        if s.a_mean_abs > 0.0 {
            s.b_mean_abs / s.a_mean_abs
        } else {
            0.0
        },
    );
    Ok(())
}

/// Validate the cache's `_meta.json` header.  Mirrors `prepare_hidream_o1`'s
/// emitted format string and ensures we don't accidentally consume a cache
/// produced for a different model family.
fn validate_meta(cache_dir: &std::path::Path) -> anyhow::Result<()> {
    let meta_path = cache_dir.join("_meta.json");
    let raw = std::fs::read_to_string(&meta_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", meta_path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| anyhow::anyhow!("parse {}: {e}", meta_path.display()))?;
    let fmt = v
        .get("format")
        .and_then(|x| x.as_str())
        .ok_or_else(|| anyhow::anyhow!("{} missing `format`", meta_path.display()))?;
    // v2 (2026-05-17) added `token_types` (`token_types_bin = (raw > 0)`)
    // to fix the attention-mask TMS-row parity bug; v1 caches must be
    // re-generated by `prepare_hidream_o1`. See
    // `EriDiffusion-v2/docs/hidream_o1_g0_deep_investigation.md`.
    //
    // v3 (2026-05-18) adds multi-resolution AR-preserving rectangular
    // buckets. The per-sample tensor schema is identical to v2 (same six
    // fields, same dtypes); v3 only declares that `image_grid`
    // (H/32, W/32) may vary across samples in the cache directory. The
    // trainer's per-step `load_file` already pulls per-sample shapes, so
    // v2 caches load transparently as "v3 with one square bucket". Accept
    // either; reject anything else.
    if fmt != "hidream-o1-v2" && fmt != "hidream-o1-v3" {
        anyhow::bail!(
            "cache format mismatch: {} reports `{fmt}`, expected `hidream-o1-v2` or \
             `hidream-o1-v3`. Re-run `prepare_hidream_o1` to regenerate the cache; v1 \
             caches lack the `token_types` field added in the 2026-05-17 attention-mask \
             bug fix.",
            meta_path.display()
        );
    }
    Ok(())
}

fn safetensors_last_dim(path: &Path, tensor_name: &str) -> anyhow::Result<usize> {
    let mut file =
        std::fs::File::open(path).map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?;
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .map_err(|e| anyhow::anyhow!("read header len {}: {e}", path.display()))?;
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)
        .map_err(|e| anyhow::anyhow!("read header {}: {e}", path.display()))?;
    let v: serde_json::Value = serde_json::from_slice(&header)
        .map_err(|e| anyhow::anyhow!("parse safetensors header {}: {e}", path.display()))?;
    let shape = v
        .get(tensor_name)
        .and_then(|x| x.get("shape"))
        .and_then(|x| x.as_array())
        .ok_or_else(|| anyhow::anyhow!("{} missing shape for `{tensor_name}`", path.display()))?;
    shape
        .last()
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| anyhow::anyhow!("{} has invalid shape for `{tensor_name}`", path.display()))
}

/// Decode the on-disk F32 `position_ids: [3, S_total]` tensor into three
/// `Vec<u32>`s for `MRopePositions`.
fn decode_position_ids(pos: &Tensor) -> anyhow::Result<(Vec<u32>, Vec<u32>, Vec<u32>)> {
    let dims = pos.shape().dims().to_vec();
    if dims.len() != 2 || dims[0] != 3 {
        anyhow::bail!("position_ids: expected [3, S_total], got {:?}", dims);
    }
    let s_total = dims[1];
    let flat = pos.to_dtype(DType::F32)?.to_vec_f32()?;
    let mut t = Vec::with_capacity(s_total);
    let mut h = Vec::with_capacity(s_total);
    let mut w = Vec::with_capacity(s_total);
    for i in 0..s_total {
        t.push(flat[i] as u32);
        h.push(flat[s_total + i] as u32);
        w.push(flat[2 * s_total + i] as u32);
    }
    Ok((t, h, w))
}

/// Gather rows of `x_pred [B, S_total, 3072]` where `vinput_mask[b, i] != 0`.
///
/// Cache layout per `prepare_hidream_o1`: the image slots are the **tail** of
/// the stream (`txt_seq_len..(txt_seq_len + L)`). We exploit this to skip a
/// general per-row gather kernel and just `narrow` along dim 1 for the
/// last-L rows. The MVP assumes batch=1 (which prepare currently produces).
///
/// If the cache ever interleaves image slots, replace this with a host-side
/// mask scan and `index_select`.
fn gather_image_rows(x_pred: &Tensor, vinput_mask: &Tensor) -> anyhow::Result<Tensor> {
    let xd = x_pred.shape().dims().to_vec();
    if xd.len() != 3 {
        anyhow::bail!("gather_image_rows: x_pred must be [B,S,C], got {:?}", xd);
    }
    let (b, s_total, _c) = (xd[0], xd[1], xd[2]);
    let md = vinput_mask.shape().dims().to_vec();
    if md.len() != 2 || md[0] != b || md[1] != s_total {
        anyhow::bail!(
            "gather_image_rows: vinput_mask shape {:?} != [{},{}]",
            md,
            b,
            s_total
        );
    }
    let host = vinput_mask.to_dtype(DType::F32)?.to_vec_f32()?;
    // Find the first and last non-zero index across the stream (per-batch
    // MVP: b==1). We assume contiguous tail layout (matches prep).
    let mut first: Option<usize> = None;
    let mut last: Option<usize> = None;
    for i in 0..s_total {
        if host[i] != 0.0 {
            first.get_or_insert(i);
            last = Some(i);
        }
    }
    let (first, last) = (
        first.ok_or_else(|| anyhow::anyhow!("vinput_mask has no image slots"))?,
        last.unwrap(),
    );
    let len = last - first + 1;
    // Sanity: count of 1's must equal `len` (i.e. tail is contiguous).
    let count = host.iter().filter(|&&x| x != 0.0).count();
    if count != len {
        anyhow::bail!(
            "gather_image_rows: non-contiguous image slots not yet supported \
             (got {count} non-zero, span [{first}..{}] len {len}). \
             TODO(O1-M3.1): index_select fallback.",
            last + 1
        );
    }
    Ok(x_pred.narrow(1, first, len)?)
}

fn main() -> anyhow::Result<()> {
    use rand::seq::SliceRandom;
    use rand::Rng;
    use rand::SeedableRng;
    trainer_common::init_logging();
    let args = Args::parse();
    if !args.export_scale.is_finite() || args.export_scale <= 0.0 {
        anyhow::bail!(
            "--export-scale must be finite and > 0 for weights-only LoRA exports, got {}",
            args.export_scale
        );
    }
    if !args.caption_dropout_probability.is_finite()
        || args.caption_dropout_probability < 0.0
        || args.caption_dropout_probability > 1.0
    {
        anyhow::bail!(
            "--caption-dropout-probability must be finite and in [0,1], got {}",
            args.caption_dropout_probability
        );
    }
    trainer_common::ensure_output_dir(&args.output_dir)?;
    validate_meta(&args.cache_dir)?;

    let device = trainer_common::init_bf16_cuda();
    // BUG-8 fix: set the global flame RNG seed BEFORE model load so that any
    // ephemeral random init in the loader path (Linear::new, RMSNorm::new etc.
    // before safetensors overwrite) is deterministic across runs. Previously
    // this was set just before the train loop, leaving the loader at the
    // default RNG state.
    trainer_common::set_flame_seed(SEED)?;

    let mut config = trainer_common::load_train_config_or_default(args.config.as_deref())?;
    trainer_common::apply_lora_basics(&mut config, args.rank, args.lora_alpha as f64, args.lr);

    if args.sample_every > 0 {
        log::warn!(
            "[hidream_o1] --sample-every={} ignored: in-trainer sampling deferred to O1-M4. \
             Use `hidream_o1_infer --lora-path ...` externally between checkpoints.",
            args.sample_every
        );
    }

    let tstep_mode = parse_tstep_mode(&args.timestep_distribution)?;
    let loss_objective = parse_loss_objective(&args.loss_objective)?;
    let include_resident_lora = args.include_resident_lora;
    if args.include_resident_lora {
        log::warn!(
            "[hidream_o1] --include-resident-lora selected 257-adapter O1-head ablation; \
             this does not match ai-toolkit's current 252-adapter O1 training surface"
        );
    }
    if args.no_resident_lora {
        log::info!(
            "[hidream_o1] --no-resident-lora is already the default for ai-toolkit O1 parity"
        );
    }
    log::info!(
        "[hidream_o1] tstep_mode={:?} flow_shift={} noise_scale={} t_eps={} max_loss={} loss_objective={:?} resident_heads={} export_scale={}",
        tstep_mode,
        args.flow_shift,
        NOISE_SCALE,
        T_EPS_AT,
        args.max_loss,
        loss_objective,
        include_resident_lora,
        args.export_scale,
    );

    let save_mode_full = match args.save_mode.as_str() {
        "weights" => false,
        "full" => true,
        other => anyhow::bail!("--save-mode must be `weights` or `full`, got `{other}`"),
    };
    let opt_kind =
        OptimizerKind::parse(&args.optimizer).map_err(|e| anyhow::anyhow!("--optimizer: {e}"))?;
    if save_mode_full && opt_kind != OptimizerKind::AdamW {
        anyhow::bail!(
            "--save-mode=full is implemented only for --optimizer adamw, got `{}`. \
             Re-run with --save-mode=weights or switch to --optimizer adamw.",
            opt_kind.as_str()
        );
    }
    if args.resume_full.is_some() && opt_kind != OptimizerKind::AdamW {
        anyhow::bail!(
            "--resume-full restores AdamW optimizer state and requires --optimizer adamw, got `{}`. \
             Use --resume-lora to resume weights only with a fresh optimizer.",
            opt_kind.as_str()
        );
    }

    // ── Load model + tokenizer. The tokenizer is needed for on-the-fly empty
    //    prompt streams when caption dropout is enabled.
    let hd_cfg = HiDreamO1Config::dev_8b();
    log::info!(
        "[hidream_o1] loading model from {} (num_layers={}, hidden={})",
        args.model_path.display(),
        hd_cfg.num_layers,
        hd_cfg.hidden_size,
    );
    let tokenizer_path = args.model_path.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer::from_file({}): {e}", tokenizer_path.display()))?;
    let mut empty_conditioning_cache: HashMap<(usize, usize), (Tensor, Tensor, Tensor, Tensor)> =
        HashMap::new();
    let loader = HiDreamO1WeightLoader::from_dir(&args.model_path)
        .map_err(|e| anyhow::anyhow!("HiDreamO1WeightLoader: {e}"))?;
    // The base model's parameters are all `requires_grad=false` (loader runs
    // under no_grad). The trainable surface is purely the LoRA registry.
    let mut model = loader
        .load_model(&hd_cfg, &device)
        .map_err(|e| anyhow::anyhow!("HiDreamO1WeightLoader::load_model: {e}"))?;

    // ── Build LoRA registry. The reference LoRA code keeps adapter weights and
    // branch math in F32, then casts the residual back to base dtype.
    let mut lora = if let Some(resume) = args.resume_lora.as_ref() {
        log::info!(
            "[hidream_o1] resuming LoRA registry from {}",
            resume.display()
        );
        LoraRegistry::from_safetensors_with_dtype(resume, &hd_cfg, &device, DType::F32)
            .map_err(|e| anyhow::anyhow!("LoraRegistry::from_safetensors: {e}"))?
    } else {
        LoraRegistry::new_with_dtype_and_resident(
            &hd_cfg,
            args.rank,
            args.lora_alpha,
            default_target_suffixes(),
            SEED,
            &device,
            DType::F32,
            include_resident_lora,
        )
        .map_err(|e| anyhow::anyhow!("LoraRegistry::new: {e}"))?
    };
    log::info!(
        "[hidream_o1] LoRA registry: {} adapters, rank={}, alpha={}, dtype=F32, resident_heads={}",
        lora.len(),
        lora.rank,
        lora.alpha,
        include_resident_lora,
    );
    validate_lora_surface(&lora, &hd_cfg, include_resident_lora)?;
    if args.lora_stats_every > 0 {
        log_lora_magnitude_stats("init", &lora)?;
    }

    // HiDream-O1 uses boundary checkpointing in inference-flame's decoder
    // loop. Keep the grow cache alive for the full run so checkpointed
    // block I/O can spill to pinned host memory instead of retaining every
    // streamed block in the backward tape. The cursor must still be reset
    // after every backward pass, otherwise pinned slabs grow monotonically
    // across steps until the process gets host-OOM killed.
    let activation_cache = {
        use eridiffusion_core::training::offload::setup_grow_activation_cache;
        let slab_bytes = 1usize << 30;
        match setup_grow_activation_cache(&device, slab_bytes) {
            Ok(cache) => {
                log::info!(
                    "[activation_offload] grow cache installed (slab={} MB); HiDream-O1 decoder boundary checkpointing active",
                    slab_bytes / (1024 * 1024)
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

    // ── Flatten registry into a Vec<Parameter> for the optimizer.
    let mut params: Vec<Parameter> = lora.parameters();
    // Gate-on 6a: under v2 (default), flip LoRA params to MatchParamDtype so
    // BF16 grads from the bridge stay BF16 (Class A). --use-autograd-v3 skips.
    trainer_pipeline::apply_autograd_v2_grad_policy(&mut params, args.use_autograd_v3, "params");
    log::info!("[hidream_o1] {} trainable parameters", params.len());
    if params.is_empty() {
        anyhow::bail!("LoRA registry produced no trainable parameters");
    }

    log::info!(
        "[hidream_o1] optimizer={} lr={} betas=({}, {}) eps={} wd={}",
        opt_kind.as_str(),
        args.lr,
        args.adamw_beta1,
        args.adamw_beta2,
        args.adamw_eps,
        args.adamw_weight_decay,
    );
    // Default is reference-style AdamW8bit. Keep `--optimizer adamw` available
    // for ablations and for full-state checkpoints.
    let mut opt = Optimizer::new(
        opt_kind,
        args.lr,
        args.adamw_beta1,
        args.adamw_beta2,
        args.adamw_eps,
        args.adamw_weight_decay,
    );

    let mut start_step = args.start_step;
    if let Some(resume_path) = args.resume_full.as_ref() {
        if args.start_step != 0 {
            anyhow::bail!(
                "--resume-full already carries the step counter; omit --start-step \
                 or use --resume-lora for weights-only resume"
            );
        }
        log::info!("[hidream_o1] full-resume from {}", resume_path.display());
        let loaded = checkpoint::load_full(resume_path, &device)?;
        if loaded.header.rank != args.rank {
            anyhow::bail!(
                "checkpoint rank={} but --rank={} — LoRA shapes are incompatible",
                loaded.header.rank,
                args.rank
            );
        }
        if (loaded.header.alpha - args.lora_alpha).abs() > 1e-6 {
            anyhow::bail!(
                "checkpoint alpha={} but --lora-alpha={} — LoRA scale would diverge",
                loaded.header.alpha,
                args.lora_alpha
            );
        }
        let named = lora.named_parameters();
        checkpoint::apply_lora_weights(&loaded, &named)?;
        if let Optimizer::AdamW(ref mut adam) = opt {
            checkpoint::apply_to_optimizer(&loaded, adam, &named, args.rank, args.lora_alpha)?;
        }
        start_step = loaded.header.step as usize;
        log::info!(
            "[hidream_o1] continuing from step {start_step}; running {} additional steps",
            args.steps
        );
    }

    // ── Index cache files.
    // BUG-6 fix: only `sample_NNNNNN.safetensors` to avoid picking up
    // companions (e.g. `features.safetensors`) or stale `*.partial` artifacts
    // from a crashed prep run.
    let mut cache_files = trainer_common::list_cache_safetensors_or_empty(&args.cache_dir)?;
    cache_files.retain(|p| {
        p.file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with("sample_"))
            .unwrap_or(false)
    });
    if args.max_seq_len > 0 {
        let before = cache_files.len();
        let mut kept = Vec::with_capacity(before);
        let mut skipped: Vec<(PathBuf, usize)> = Vec::new();
        for path in cache_files {
            let seq_len = safetensors_last_dim(&path, "vinput_mask")?;
            if seq_len <= args.max_seq_len {
                kept.push(path);
            } else {
                skipped.push((path, seq_len));
            }
        }
        if !skipped.is_empty() {
            let longest = skipped
                .iter()
                .max_by_key(|(_, seq_len)| *seq_len)
                .map(|(path, seq_len)| {
                    format!(
                        "{} ({seq_len})",
                        path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("<unknown>")
                    )
                })
                .unwrap_or_else(|| "<none>".to_string());
            log::warn!(
                "[hidream_o1] --max-seq-len={} skipped {}/{} cached samples; longest skipped: {}",
                args.max_seq_len,
                skipped.len(),
                before,
                longest
            );
        }
        cache_files = kept;
    }
    if cache_files.is_empty() {
        anyhow::bail!("No cached samples in {:?}", args.cache_dir);
    }
    log::info!("[hidream_o1] {} cached samples", cache_files.len());

    // (flame_core::rng::set_seed moved above model-load — BUG-8.)
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let mut epoch_order: Vec<usize> = (0..cache_files.len()).collect();
    let mut epoch_order_epoch: Option<usize> = None;
    let shuffle_cache = !args.no_cache_shuffle;
    log::info!(
        "[hidream_o1] cache_order={} caption_dropout_probability={}",
        if shuffle_cache {
            "shuffle_each_epoch"
        } else {
            "sorted_cyclic"
        },
        args.caption_dropout_probability
    );

    let board = trainer_common::open_board_writer(&args.output_dir, None);
    if let Some(b) = &board {
        // Full board wiring: run hyper-parameters → metadata.hparams + the
        // dashboard's hparam panel. JSON hand-built (no serde_json dep here).
        let hparams_json = format!(
            "{{\"model\":\"hidream_o1\",\"steps\":{},\"start_step\":{},\"rank\":{},\
             \"lora_alpha\":{},\"lr\":{},\"optimizer\":\"{}\",\"flow_shift\":{},\"seed\":{}}}",
            args.steps,
            start_step,
            args.rank,
            args.lora_alpha,
            args.lr,
            opt_kind.as_str(),
            args.flow_shift,
            SEED
        );
        b.log_hparams(
            &hparams_json,
            &[("steps_target", (start_step + args.steps) as f64)],
        );
    }

    let mut max_loss_clamps = 0usize;
    let mut max_raw_loss = 0f32;

    let total_target_steps = start_step + args.steps;
    let loop_config = trainer_pipeline::ManualTrainLoopConfig::new(
        "HiDreamO1-lora",
        start_step,
        total_target_steps,
        cache_files.len(),
        1,
    );
    let mut loop_run = trainer_pipeline::ManualTrainLoopRun::new(loop_config);
    for step in loop_run.steps() {
        flame_core::debug_finite::reset();

        let epoch = step / cache_files.len();
        let epoch_pos = step % cache_files.len();
        if epoch_order_epoch != Some(epoch) {
            epoch_order.clear();
            epoch_order.extend(0..cache_files.len());
            if shuffle_cache {
                let mut epoch_rng = rand::rngs::StdRng::seed_from_u64(
                    SEED.wrapping_add((epoch as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                );
                epoch_order.shuffle(&mut epoch_rng);
            }
            epoch_order_epoch = Some(epoch);
            if step == start_step || epoch_pos == 0 {
                log::info!(
                    "[hidream_o1] epoch {} cache order {}",
                    epoch + 1,
                    if shuffle_cache { "shuffled" } else { "sorted" }
                );
            }
        }
        let cache_idx = epoch_order[epoch_pos];
        let sample = flame_core::serialization::load_file(&cache_files[cache_idx], &device)?;
        let path_disp = cache_files[cache_idx].display().to_string();

        let patches = sample
            .get("patches")
            .ok_or_else(|| anyhow::anyhow!("missing `patches` in {path_disp}"))?
            .to_dtype(DType::BF16)?;
        let mut input_ids = sample
            .get("input_ids")
            .ok_or_else(|| anyhow::anyhow!("missing `input_ids` in {path_disp}"))?
            .to_dtype(DType::I32)?;
        let mut position_ids = sample
            .get("position_ids")
            .ok_or_else(|| anyhow::anyhow!("missing `position_ids` in {path_disp}"))?
            .to_dtype(DType::F32)?;
        let mut vinput_mask = sample
            .get("vinput_mask")
            .ok_or_else(|| anyhow::anyhow!("missing `vinput_mask` in {path_disp}"))?
            .to_dtype(DType::BF16)?;
        // Cache v2 (2026-05-17): `token_types_bin = (raw > 0)`. Drives the
        // structured prefix-causal/full attention split so the TMS/image rows
        // get full-attention, matching `qwen3_vl_transformers.py:1501`.
        let mut token_types_bin = sample
            .get("token_types")
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "missing `token_types` in {path_disp} (cache v1 detected — \
                     re-run prepare_hidream_o1 to regenerate v2 cache)"
                )
            })?
            .to_dtype(DType::BF16)?;
        if args.caption_dropout_probability > 0.0
            && rng.r#gen::<f32>() < args.caption_dropout_probability
        {
            let grid = sample
                .get("image_grid")
                .ok_or_else(|| anyhow::anyhow!("missing `image_grid` in {path_disp}"))?;
            let (h_patches, w_patches) = decode_image_grid(grid)?;
            let cond = match empty_conditioning_cache.get(&(h_patches, w_patches)) {
                Some(cond) => cond.clone(),
                None => {
                    let cond = build_prompt_conditioning(
                        &tokenizer, &hd_cfg, "", h_patches, w_patches, &device,
                    )?;
                    empty_conditioning_cache.insert((h_patches, w_patches), cond.clone());
                    cond
                }
            };
            input_ids = cond.0;
            position_ids = cond.1;
            vinput_mask = cond.2;
            token_types_bin = cond.3;
            if step == 0 {
                log::info!(
                    "step 0 | caption dropped to empty prompt for image_grid={}x{}",
                    h_patches,
                    w_patches
                );
            }
        }
        flame_core::debug_finite::check("g2.patches", &patches)?;
        flame_core::debug_finite::check("g2.vinput_mask", &vinput_mask)?;
        flame_core::debug_finite::check("g2.token_types", &token_types_bin)?;

        let (t_pos, h_pos, w_pos) = decode_position_ids(&position_ids)?;
        let pos_view = MRopePositions {
            t: &t_pos,
            h: &h_pos,
            w: &w_pos,
        };

        // ── Sample timestep (flowmatch shift by default, see TstepMode).
        let t_scalar = sample_t(&mut rng, tstep_mode, args.flow_shift);
        // Noise ~ N(0, 1), then SCALED by NOISE_SCALE before use in both the
        // noisy input and the loss target — see edv2-reference
        // `hidream_o1_model.py:53-56` (`scaled_noise = noise * noise_scale`)
        // and `:520-521` (`target = noise*noise_scale - latents`).
        let noise = Tensor::randn(patches.shape().clone(), 0.0, 1.0, device.clone())?
            .to_dtype(DType::BF16)?;
        let scaled_noise = noise.mul_scalar(NOISE_SCALE)?;
        flame_core::debug_finite::check("g2.scaled_noise", &scaled_noise)?;
        // Linear flow matching with scaled noise:
        //   x_t = (1 - t) * patches + t * (noise * noise_scale)
        let noisy = patches
            .mul_scalar(1.0 - t_scalar)?
            .add(&scaled_noise.mul_scalar(t_scalar)?)?;
        flame_core::debug_finite::check("g2.noisy", &noisy)?;
        // HiDream-O1's model expects timestep as denoising PROGRESS
        // (1=clean, 0=noisy) — inverted from the canonical convention used
        // for `noisy`. Mirror edv2-reference `hidream_o1_model.py:439, 446`:
        //   t_pixeldit = (1.0 - timestep / 1000.0)
        // Our `t_scalar` is already in canonical [eps, 1-eps] continuous
        // (not `/1000` discrete), so the equivalent is `1.0 - t_scalar`.
        let t_pixeldit = 1.0 - t_scalar;
        let timestep = Tensor::from_vec(vec![t_pixeldit], Shape::from_dims(&[1]), device.clone())?
            .to_dtype(DType::BF16)?;

        // Loss target = `(scaled_noise - patches).detach()` — the
        // edv2-reference flow-matching velocity with scaled noise.
        // `hidream_o1_model.py:517-521 get_loss_target`.
        let target_full = scaled_noise.sub(&patches)?.detach()?;
        flame_core::debug_finite::check("g2.target_full", &target_full)?;

        if step == 0 {
            log::info!(
                "step 0 | patches={:?} input_ids={:?} vinput_mask={:?} t={:.4} \
                 noise_scale={} target={:?}",
                patches.shape().dims(),
                input_ids.shape().dims(),
                vinput_mask.shape().dims(),
                t_scalar,
                NOISE_SCALE,
                target_full.shape().dims(),
            );
        }

        // ── Forward with LoRA routed through every decoder layer's 7 linears.
        // TODO(O1-G2): verify autograd memory pressure with BlockOffloader
        // during 50-step overfit smoke — Skeptic Q1 (lora.rs review). If
        // saved weight Arcs in the backward tape stay pinned across 36 layers,
        // peak GPU memory could exceed the 14 GB budget at 512².
        let x_pred = model.forward_lora(
            &input_ids,
            &timestep,
            &noisy,
            &pos_view,
            &vinput_mask,
            &token_types_bin,
            None,
            Some(&lora),
        )?;
        flame_core::debug_finite::check("g2.x_pred", &x_pred)?;

        // Gather only the image rows so the loss doesn't reward fitting text
        // positions (matches inference `pipeline.py:329`).
        let x_rows = gather_image_rows(&x_pred, &vinput_mask)?;
        flame_core::debug_finite::check("g2.x_rows", &x_rows)?;
        // `target_full` is already shaped to the image rows (it's a function
        // of `patches` and `noise`, both of shape `[1, L, 3072]`). No gather
        // needed on the target side.
        if x_rows.shape().dims() != target_full.shape().dims() {
            anyhow::bail!(
                "shape mismatch: x_rows={:?} target={:?}",
                x_rows.shape().dims(),
                target_full.shape().dims()
            );
        }

        // ── Loss. HiDream-O1 emits x0-style clean-patch predictions.
        // The stable EDv2 objective trains that native output directly.
        // The velocity path is retained for reference parity/debugging.
        let x0_pred_f32 = x_rows.to_dtype(DType::F32)?;
        let clean_f32 = patches.to_dtype(DType::F32)?;
        let x0_loss = x0_pred_f32.sub(&clean_f32)?.square()?.mean()?;
        flame_core::debug_finite::check("g2.x0_loss", &x0_loss)?;
        let x0_loss_val = x0_loss.to_vec()?[0];
        if !x0_loss_val.is_finite() {
            anyhow::bail!("NaN/Inf x0 loss at step {step}: {x0_loss_val}");
        }

        // Convert model x0-style output to velocity, then MSE against
        // (scaled_noise - patches). Per reference Python
        // `hidream_o1_model.py:467-473`:
        //   sigma = max(t, T_EPS)
        //   pred  = (latent_model_input.float() - x0_pred.float()) / sigma
        //   return pred.to(in_dtype)   # cast back to BF16
        // SDTrainer.py:739, 806 then re-casts `pred.float()` for MSE.
        // The BF16 round-trip TRUNCATES the 1/sigma-amplified F32 difference
        // into BF16 mantissa precision. Skipping that cast leaves the full
        // F32 amplification in `pred`, blowing up the MSE (and lora_B grad)
        // when sigma is small. Mirror the reference exactly: F32 sub+div → BF16 → F32.
        // Pred has the same shape as target_full ([1, L, 3072]).
        let sigma = t_scalar.max(T_EPS_AT);
        // noisy was built from `patches` + `scaled_noise` (both BF16); gather
        // the image rows out of `noisy` too — they're already image-aligned
        // (1:1 with target_full), no narrow needed since `noisy` is shaped
        // exactly to image rows.
        let pred_f32 = noisy
            .to_dtype(DType::F32)?
            .sub(&x0_pred_f32)?
            .mul_scalar(1.0 / sigma)?
            // Reference parity: round-trip through in_dtype=BF16 before MSE.
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?;
        flame_core::debug_finite::check("g2.pred_f32", &pred_f32)?;
        let target_f32 = target_full.to_dtype(DType::F32)?;
        flame_core::debug_finite::check("g2.target_f32", &target_f32)?;
        let velocity_loss = pred_f32.sub(&target_f32)?.square()?.mean()?;
        flame_core::debug_finite::check("g2.velocity_loss", &velocity_loss)?;
        let velocity_loss_val = velocity_loss.to_vec()?[0];
        if !velocity_loss_val.is_finite() {
            anyhow::bail!("NaN/Inf velocity loss at step {step}: {velocity_loss_val}");
        }

        let raw_loss = match loss_objective {
            LossObjective::X0 => x0_loss,
            LossObjective::Velocity => velocity_loss,
        };
        flame_core::debug_finite::check("g2.raw_loss", &raw_loss)?;
        let raw_loss_val = raw_loss.to_vec()?[0];
        if !raw_loss_val.is_finite() {
            anyhow::bail!("NaN/Inf loss at step {step}: {raw_loss_val}");
        }
        max_raw_loss = max_raw_loss.max(raw_loss_val);

        let loss = if args.max_loss > 0.0 {
            if raw_loss_val > args.max_loss {
                max_loss_clamps += 1;
                if max_loss_clamps <= 10 || max_loss_clamps % 50 == 0 {
                    log::warn!(
                        "[max-loss] step {} raw loss {:.4} > {:.4}; clamping before backward",
                        step + 1,
                        raw_loss_val,
                        args.max_loss
                    );
                }
            }
            raw_loss.clamp(0.0, args.max_loss)?
        } else {
            raw_loss
        };
        flame_core::debug_finite::check("g2.loss", &loss)?;
        let loss_val = loss.to_vec()?[0];
        if !loss_val.is_finite() {
            anyhow::bail!("NaN/Inf loss at step {step}: {loss_val}");
        }

        // ── Backward. Gate-on 6a: v2 is the default; backward goes through
        // `backward_v2` unless `--use-autograd-v3` opts into the legacy v3.
        let grads = trainer_pipeline::backward_loss(&loss, args.use_autograd_v3)?;
        let named_lora_params = lora.named_parameters();
        if args.allow_missing_lora_grads {
            log::warn!(
                "[grad-coverage] --allow-missing-lora-grads is enabled; optimizer may skip detached adapters"
            );
        } else {
            require_lora_grad_coverage(&grads, &named_lora_params, step + 1, step == 0)?;
        }

        // Grad-flow diagnostic — runs at step 1 (lora_B starts zero so step 0
        // is mathematically zero-grad on the A side). This is fail-closed for
        // O1 because a dead adapter can otherwise produce a numerically
        // nonzero checkpoint that does not actually learn.
        if step == 1 {
            let named_refs: Vec<(&str, &Parameter)> = named_lora_params
                .iter()
                .map(|(n, p)| (n.as_str(), p))
                .collect();
            let report = flame_core::diagnostics::check_grad_flow(&grads, &named_refs)?;
            if report.is_clean() {
                log::info!("[grad-flow] step 1 clean ({} params)", report.ok_count);
            } else {
                anyhow::bail!("{}", report.summary());
            }
        }

        // ── Global L2 grad clip = 1.0.
        let clip = trainer_pipeline::apply_gradient_map_clip(
            &params,
            &grads,
            trainer_pipeline::GradientClipOptions::clip_by_norm(CLIP_GRAD_NORM)
                .require_gradients()
                .require_finite_norm(),
        )
        .map_err(|err| anyhow::anyhow!("grad clipping failed at step {}: {err}", step + 1))?;
        let total_norm = clip.total_norm;

        // ── Optimizer step (constant LR for MVP; LR scheduling is a Klein-
        //     surface feature deferred to O1-M3.1).
        trainer_pipeline::step_optimizer(&mut opt, &params, args.lr, || Ok(()))?;
        // Re-sync the LoRA registry with whatever the optimizer just wrote.
        // No-op today (LoraAdapter holds the Parameter and reads via
        // `a_tensor()` / `b_tensor()`), but kept here so a future
        // optimizer-replaces-storage path stays obvious.
        let _ = &mut lora;

        device.synchronize().ok();
        if let Some(cache) = &activation_cache {
            let mut cache = cache
                .lock()
                .map_err(|_| anyhow::anyhow!("grow activation cache mutex poisoned"))?;
            let in_flight = cache.in_flight();
            if in_flight != 0 {
                log::warn!(
                    "[activation_offload] resetting grow cache with {in_flight} in-flight entries"
                );
            }
            cache.reset();
        }
        AutogradContext::clear();
        flame_core::cuda_alloc_pool::clear_pool_cache();

        let step_num = step + 1;
        loop_run.record_and_log(
            step,
            trainer_pipeline::TrainStepMetrics {
                loss_value: loss_val,
                grad_norm: total_norm,
                learning_rate: args.lr,
            },
            board.as_ref(),
        );
        if let Some(b) = &board {
            b.log_scalars(
                step_num as u64,
                &[
                    ("hidream_o1/sigma", sigma as f64),
                    ("hidream_o1/loss_x0", x0_loss_val as f64),
                    ("hidream_o1/loss_velocity", velocity_loss_val as f64),
                ],
            );
        }
        if args.lora_stats_every > 0 && (step_num == 1 || step_num % args.lora_stats_every == 0) {
            log_lora_magnitude_stats(&format!("step {step_num}"), &lora)?;
        }

        // ── Periodic save.
        if trainer_common::cadence_fires(args.save_every, step_num, total_target_steps) {
            let mid_ckpt = args
                .output_dir
                .join(format!("hidream_o1_lora_step{step_num}.safetensors"));
            trainer_pipeline::save_lora_checkpoint(
                trainer_pipeline::CheckpointSaveOptions {
                    trainer: "train_hidream_o1",
                    path: &mid_ckpt,
                    step: step_num as u64,
                    rank: args.rank,
                    alpha: args.lora_alpha,
                    seed: SEED,
                    config_hash: "",
                    save_mode_full,
                    label: &format!("[save step {step_num}]"),
                },
                &opt,
                || Ok(lora.named_parameters()),
                || {
                    lora.save_safetensors_with_export_scale(&mid_ckpt, args.export_scale)?;
                    Ok(())
                },
            )?;
        }
    }

    let completion = loop_run.finish();
    log::info!(
        "Training complete: {} steps, avg loss={:.4}, max raw loss={:.4}, max-loss clamps={}",
        total_target_steps,
        completion.average_loss,
        max_raw_loss,
        max_loss_clamps
    );
    trainer_pipeline::mark_board_completed(board.as_ref());

    let final_ckpt = args.output_dir.join(format!(
        "hidream_o1_lora_{}steps.safetensors",
        total_target_steps
    ));
    trainer_pipeline::save_lora_checkpoint(
        trainer_pipeline::CheckpointSaveOptions {
            trainer: "train_hidream_o1",
            path: &final_ckpt,
            step: total_target_steps as u64,
            rank: args.rank,
            alpha: args.lora_alpha,
            seed: SEED,
            config_hash: "",
            save_mode_full,
            label: "[final]",
        },
        &opt,
        || Ok(lora.named_parameters()),
        || {
            lora.save_safetensors_with_export_scale(&final_ckpt, args.export_scale)?;
            Ok(())
        },
    )?;
    Ok(())
}
