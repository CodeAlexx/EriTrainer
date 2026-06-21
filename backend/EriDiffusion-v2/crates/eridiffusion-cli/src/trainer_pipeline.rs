//! Shared trainer pipeline boundary.
//!
//! This mirrors OneTrainer's split:
//!
//! - the trainer pipeline owns lifecycle services: logging, preflight,
//!   config, cache discovery, runtime device setup, timing, progress, save
//!   cadence, and sampler orchestration;
//! - model adapters own model-specific details: model loading, cache tensor
//!   schema, forward/loss construction, model-specific validation/sample code,
//!   and checkpoint key layout.
//!
//! The existing `train_*` binaries are being migrated toward this boundary
//! incrementally. Until their full loops are collapsed, they should call the
//! services here through `trainer_common` and keep model-specific code local.

use eridiffusion_core::config::TrainConfig;
use eridiffusion_core::training::board::BoardWriter;
use eridiffusion_core::training::checkpoint::{self, CkptHeader};
use eridiffusion_core::training::ema::ParameterEma;
use eridiffusion_core::training::training_features::{Optimizer, OptimizerKind};
use eridiffusion_core::training::{accumulate_parameter_grads, clip_parameter_grads};
use flame_core::gradient_clip::{GradientClipStrategy, GradientClipper};
use flame_core::{AutogradContext, CudaDevice, GradientMap, Parameter, Tensor};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

pub struct TrainerServices;

impl TrainerServices {
    pub fn init_logging(&self) {
        crate::trainer_runtime::init_logging();
    }

    pub fn ensure_output_dir(&self, path: &Path) -> anyhow::Result<()> {
        crate::trainer_preflight::ensure_output_dir(path)
    }

    pub fn bf16_device(&self) -> Arc<CudaDevice> {
        crate::trainer_runtime::global_bf16_device()
    }

    pub fn device(&self) -> Arc<CudaDevice> {
        crate::trainer_runtime::global_device()
    }

    pub fn set_seed(&self, seed: u64) -> anyhow::Result<()> {
        crate::trainer_runtime::set_flame_seed(seed)
    }

    pub fn cache_safetensors(&self, path: &Path) -> anyhow::Result<Vec<PathBuf>> {
        crate::trainer_cache::list_safetensors(path)
    }

    pub fn cache_safetensors_or_empty(&self, path: &Path) -> anyhow::Result<Vec<PathBuf>> {
        crate::trainer_cache::list_safetensors_or_empty(path)
    }
}

pub trait ModelTrainer {
    fn model_id(&self) -> &'static str;

    fn preflight(&self, _services: &TrainerServices) -> anyhow::Result<()> {
        Ok(())
    }

    fn train(&mut self, services: &TrainerServices) -> anyhow::Result<()>;
}

pub fn services() -> TrainerServices {
    TrainerServices
}

#[derive(Debug, Clone)]
pub struct SimpleTrainLoopConfig {
    pub tag: String,
    pub total_steps: usize,
    pub warmup_steps: usize,
    pub resume_step: usize,
    pub dataset_len: usize,
    pub batch_size: usize,
}

impl SimpleTrainLoopConfig {
    pub fn fresh(
        tag: impl Into<String>,
        total_steps: usize,
        dataset_len: usize,
        batch_size: usize,
        warmup_steps: usize,
    ) -> Self {
        Self {
            tag: tag.into(),
            total_steps,
            warmup_steps,
            resume_step: 0,
            dataset_len,
            batch_size,
        }
    }

    pub fn with_resume_step(mut self, resume_step: usize) -> Self {
        self.resume_step = resume_step;
        self
    }
}

#[derive(Debug, Clone)]
pub struct ManualTrainLoopConfig {
    pub tag: String,
    pub start_step: usize,
    pub total_steps: usize,
    pub progress_resume_step: usize,
    pub progress_total_steps: usize,
    pub dataset_len: usize,
    pub batch_size: usize,
}

impl ManualTrainLoopConfig {
    pub fn new(
        tag: impl Into<String>,
        start_step: usize,
        total_steps: usize,
        dataset_len: usize,
        batch_size: usize,
    ) -> Self {
        Self {
            tag: tag.into(),
            start_step,
            total_steps,
            progress_resume_step: start_step,
            progress_total_steps: total_steps,
            dataset_len,
            batch_size,
        }
    }

    pub fn with_progress_target(mut self, resume_step: usize, total_steps: usize) -> Self {
        self.progress_resume_step = resume_step;
        self.progress_total_steps = total_steps;
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TrainStepMetrics {
    pub loss_value: f32,
    pub grad_norm: f32,
    pub learning_rate: f32,
}

pub struct ManualTrainLoopRun {
    config: ManualTrainLoopConfig,
    started_at: Instant,
    total_loss: f32,
}

impl ManualTrainLoopRun {
    pub fn new(config: ManualTrainLoopConfig) -> Self {
        Self {
            config,
            started_at: Instant::now(),
            total_loss: 0.0,
        }
    }

    pub fn steps(&self) -> std::ops::Range<usize> {
        self.config.start_step..self.config.total_steps
    }

    pub fn started_at(&self) -> Instant {
        self.started_at
    }

    pub fn elapsed_secs_f64(&self) -> f64 {
        self.started_at.elapsed().as_secs_f64()
    }

    pub fn total_loss(&self) -> f32 {
        self.total_loss
    }

    pub fn average_loss_so_far(&self, current_step: usize) -> f32 {
        let trained_steps = current_step
            .saturating_add(1)
            .saturating_sub(self.config.start_step);
        if trained_steps > 0 {
            self.total_loss / trained_steps as f32
        } else {
            0.0
        }
    }

    pub fn record_and_log(
        &mut self,
        step: usize,
        metrics: TrainStepMetrics,
        board: Option<&BoardWriter>,
    ) {
        self.record_and_log_at(step - self.config.start_step, metrics, board);
    }

    pub fn record_and_log_at(
        &mut self,
        progress_step: usize,
        metrics: TrainStepMetrics,
        board: Option<&BoardWriter>,
    ) {
        let tag = self.config.tag.clone();
        self.record_and_log_as(&tag, progress_step, metrics, board);
    }

    pub fn record_and_log_as(
        &mut self,
        tag: &str,
        progress_step: usize,
        metrics: TrainStepMetrics,
        board: Option<&BoardWriter>,
    ) {
        self.total_loss += metrics.loss_value;
        log_training_step_with_resume(
            tag,
            progress_step,
            self.config.progress_resume_step,
            self.config.progress_total_steps,
            self.config.dataset_len,
            self.config.batch_size,
            metrics.loss_value,
            metrics.grad_norm,
            metrics.learning_rate,
            self.started_at,
            board,
        );
    }

    pub fn completion(&self) -> TrainingCompletion {
        training_completion(
            self.config.start_step,
            self.config.total_steps,
            self.total_loss,
        )
    }

    pub fn finish(self) -> TrainingCompletion {
        self.completion()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SimpleOptimizerConfig {
    pub kind: OptimizerKind,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
}

impl SimpleOptimizerConfig {
    pub fn adamw_style(kind: OptimizerKind, lr: f32, max_grad_norm: f32) -> Self {
        Self {
            kind,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            max_grad_norm,
        }
    }

    pub fn from_optimizer_name(
        optimizer: &str,
        lr: f32,
        max_grad_norm: f32,
    ) -> anyhow::Result<Self> {
        let kind = OptimizerKind::parse(optimizer)
            .map_err(|err| anyhow::anyhow!("optimizer: {err}"))?;
        Ok(Self::adamw_style(kind, lr, max_grad_norm))
    }

    pub fn build(self) -> Optimizer {
        Optimizer::new(
            self.kind,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.weight_decay,
        )
    }
}

pub struct StepLoss {
    pub loss: Tensor,
    pub loss_value: Option<f32>,
}

impl StepLoss {
    pub fn new(loss: Tensor) -> Self {
        Self {
            loss,
            loss_value: None,
        }
    }

    pub fn with_value(loss: Tensor, loss_value: f32) -> Self {
        Self {
            loss,
            loss_value: Some(loss_value),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GradientClipStats {
    pub matched_gradients: usize,
    pub total_norm: f32,
    pub scale: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct GradientClipOptions {
    pub max_norm: f32,
    pub clipping_enabled: bool,
    pub require_gradients: bool,
    pub require_finite_norm: bool,
}

impl GradientClipOptions {
    pub fn clip_by_norm(max_norm: f32) -> Self {
        Self {
            max_norm,
            clipping_enabled: true,
            require_gradients: false,
            require_finite_norm: false,
        }
    }

    pub fn with_clipping_enabled(mut self, enabled: bool) -> Self {
        self.clipping_enabled = enabled;
        self
    }

    pub fn require_gradients(mut self) -> Self {
        self.require_gradients = true;
        self
    }

    pub fn require_finite_norm(mut self) -> Self {
        self.require_finite_norm = true;
        self
    }
}

pub fn apply_gradient_map_clip(
    params: &[Parameter],
    grads: &GradientMap,
    options: GradientClipOptions,
) -> anyhow::Result<GradientClipStats> {
    let grad_refs: Vec<&Tensor> = params.iter().filter_map(|p| grads.get(p.id())).collect();
    if options.require_gradients && grad_refs.is_empty() {
        anyhow::bail!("no gradients matched trainable parameters");
    }

    let total_norm = if grad_refs.is_empty() {
        0.0
    } else {
        flame_core::ops::grad_norm::global_l2_norm(&grad_refs)?.item()? as f32
    };
    if options.require_finite_norm && !total_norm.is_finite() {
        anyhow::bail!("non-finite grad_norm: {total_norm}");
    }

    let scale = if options.clipping_enabled && total_norm > options.max_norm {
        options.max_norm / total_norm
    } else {
        1.0
    };
    for param in params {
        if let Some(grad) = grads.get(param.id()) {
            let grad = if scale < 1.0 {
                grad.mul_scalar(scale)?
            } else {
                grad.clone()
            };
            param.set_grad(grad)?;
        }
    }

    Ok(GradientClipStats {
        matched_gradients: grad_refs.len(),
        total_norm,
        scale,
    })
}

pub fn clip_parameter_slot_grads(
    params: &[Parameter],
    clipper: &GradientClipper,
) -> anyhow::Result<f32> {
    let mut grad_tensors: Vec<Tensor> = Vec::new();
    let mut owners: Vec<usize> = Vec::new();
    for (idx, param) in params.iter().enumerate() {
        if let Some(grad) = param.grad() {
            grad_tensors.push(grad);
            owners.push(idx);
        }
    }

    let mut grad_refs: Vec<&mut Tensor> = grad_tensors.iter_mut().collect();
    let norm = clipper.clip_grads(&mut grad_refs)?;
    for (owner, grad) in owners.into_iter().zip(grad_tensors.into_iter()) {
        params[owner].set_grad(grad)?;
    }
    Ok(norm)
}

/// Minimal OneTrainer-style model adapter for trainers whose common loop is:
/// build one loss tensor, run backward, apply optimizer, log progress, save.
pub trait SimpleStepTrainer {
    fn trainable_parameters(&self) -> &[Parameter];

    fn step_loss(&mut self, step: usize) -> anyhow::Result<StepLoss>;

    fn save_final(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

pub fn backward_loss(loss: &Tensor, use_autograd_v3: bool) -> anyhow::Result<GradientMap> {
    if use_autograd_v3 {
        return Ok(loss.backward()?);
    }

    #[cfg(feature = "autograd_v2")]
    {
        Ok(AutogradContext::backward_v2(loss)?)
    }
    #[cfg(not(feature = "autograd_v2"))]
    {
        anyhow::bail!(
            "autograd v2 is the default but this binary was built without the \
             `autograd_v2` feature. Rebuild with the feature, or pass --use-autograd-v3."
        );
    }
}

pub fn step_optimizer(
    optimizer: &mut Optimizer,
    params: &[Parameter],
    learning_rate: f32,
    after_step: impl FnOnce() -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    {
        let _guard = AutogradContext::no_grad();
        optimizer.set_lr(learning_rate);
        optimizer.step(params)?;
        optimizer.zero_grad(params);
        after_step()?;
    }
    AutogradContext::clear();
    Ok(())
}

pub fn step_adamw_optimizer(
    optimizer: &mut flame_core::adam::AdamW,
    params: &[Parameter],
    learning_rate: f32,
    after_step: impl FnOnce() -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    {
        let _guard = AutogradContext::no_grad();
        optimizer.set_lr(learning_rate);
        optimizer.step(params)?;
        optimizer.zero_grad(params);
        after_step()?;
    }
    AutogradContext::clear();
    Ok(())
}

pub fn apply_autograd_v2_grad_policy(
    params: &mut [Parameter],
    use_autograd_v3: bool,
    label: &str,
) {
    let count = params.len();
    apply_autograd_v2_grad_policy_to_iter(
        params.iter_mut(),
        count,
        use_autograd_v3,
        label,
    );
}

pub fn apply_autograd_v2_grad_policy_to_iter<'a, I>(
    params: I,
    count: usize,
    use_autograd_v3: bool,
    label: &str,
) where
    I: IntoIterator<Item = &'a mut Parameter>,
{
    if use_autograd_v3 {
        return;
    }

    for param in params {
        param.set_grad_dtype_policy(flame_core::parameter::GradDtypePolicy::MatchParamDtype);
    }
    log::info!(
        "[autograd_v2] flipped {} {label} to MatchParamDtype grad policy",
        count,
    );
}

pub fn with_optional_ema_swap<T>(
    ema: Option<&ParameterEma>,
    params: &[Parameter],
    enabled: bool,
    label: &str,
    action: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let backup = if enabled {
        match ema {
            Some(ema) => {
                let _guard = AutogradContext::no_grad();
                Some(ema.swap_with_live(params).map_err(|err| {
                    anyhow::anyhow!("EMA swap_with_live ({label}) failed: {err}")
                })?)
            }
            None => None,
        }
    } else {
        None
    };

    let result = action();

    if let (Some(backup), Some(ema)) = (backup, ema) {
        let _guard = AutogradContext::no_grad();
        if let Err(err) = ema.restore_swapped(params, backup) {
            return match result {
                Ok(_) => Err(anyhow::anyhow!("EMA restore_swapped ({label}) failed: {err}")),
                Err(action_err) => Err(anyhow::anyhow!(
                    "{action_err}; also failed to restore EMA swap ({label}): {err}"
                )),
            };
        }
    }

    result
}

pub fn swap_ema_for_final_save(
    ema: Option<&ParameterEma>,
    params: &[Parameter],
    enabled: bool,
) -> anyhow::Result<bool> {
    if !enabled {
        return Ok(false);
    }

    match ema {
        Some(ema) => {
            let _guard = AutogradContext::no_grad();
            ema.swap_with_live(params)
                .map_err(|err| anyhow::anyhow!("EMA swap_with_live (final) failed: {err}"))?;
            log::info!("[ema] swapped EMA shadow into live params for final save");
            Ok(true)
        }
        None => Ok(false),
    }
}

pub struct CheckpointSaveOptions<'a> {
    pub trainer: &'a str,
    pub path: &'a Path,
    pub step: u64,
    pub rank: usize,
    pub alpha: f32,
    pub seed: u64,
    pub config_hash: &'a str,
    pub save_mode_full: bool,
    pub label: &'a str,
}

#[derive(Debug, Clone, Copy)]
pub struct TrainingCompletion {
    pub trained_steps: usize,
    pub average_loss: f32,
}

pub fn training_completion(
    start_step: usize,
    total_steps: usize,
    total_loss: f32,
) -> TrainingCompletion {
    let trained_steps = total_steps - start_step;
    let average_loss = if trained_steps > 0 {
        total_loss / trained_steps as f32
    } else {
        0.0
    };

    TrainingCompletion {
        trained_steps,
        average_loss,
    }
}

pub fn mark_board_completed(board: Option<&BoardWriter>) {
    if let Some(board) = board {
        board.set_status("completed");
    }
}

pub fn complete_training_run(
    board: Option<&BoardWriter>,
    start_step: usize,
    total_steps: usize,
    total_loss: f32,
) -> TrainingCompletion {
    let completion = training_completion(start_step, total_steps, total_loss);
    log::info!(
        "Training complete: {} new steps (total={}), avg loss={:.4}",
        completion.trained_steps,
        total_steps,
        completion.average_loss
    );
    mark_board_completed(board);
    completion
}

pub fn log_training_step_with_resume(
    tag: &str,
    step: usize,
    resume_step: usize,
    total_steps: usize,
    dataset_len: usize,
    batch_size: usize,
    loss: f32,
    grad_norm: f32,
    learning_rate: f32,
    started_at: Instant,
    board: Option<&BoardWriter>,
) {
    eridiffusion_core::training::progress::log_step_with_resume(
        tag,
        step,
        resume_step,
        total_steps,
        dataset_len,
        batch_size,
        loss,
        grad_norm,
        learning_rate,
        started_at,
        board,
    );
}

pub fn log_training_step(
    tag: &str,
    step: usize,
    total_steps: usize,
    dataset_len: usize,
    batch_size: usize,
    loss: f32,
    grad_norm: f32,
    learning_rate: f32,
    started_at: Instant,
    board: Option<&BoardWriter>,
) {
    eridiffusion_core::training::progress::log_step(
        tag,
        step,
        total_steps,
        dataset_len,
        batch_size,
        loss,
        grad_norm,
        learning_rate,
        started_at,
        board,
    );
}

pub fn save_lora_checkpoint(
    options: CheckpointSaveOptions<'_>,
    optimizer: &Optimizer,
    named_parameters: impl FnOnce() -> anyhow::Result<Vec<(String, Parameter)>>,
    save_weights: impl FnOnce() -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    save_lora_checkpoint_impl(
        options,
        optimizer,
        named_parameters,
        save_weights,
        CheckpointFailurePolicy::Warn,
    )
}

pub fn save_lora_checkpoint_strict(
    options: CheckpointSaveOptions<'_>,
    optimizer: &Optimizer,
    named_parameters: impl FnOnce() -> anyhow::Result<Vec<(String, Parameter)>>,
    save_weights: impl FnOnce() -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    save_lora_checkpoint_impl(
        options,
        optimizer,
        named_parameters,
        save_weights,
        CheckpointFailurePolicy::Propagate,
    )
}

pub fn save_lora_checkpoint_adamw(
    options: CheckpointSaveOptions<'_>,
    optimizer: &flame_core::adam::AdamW,
    named_parameters: impl FnOnce() -> anyhow::Result<Vec<(String, Parameter)>>,
    save_weights: impl FnOnce() -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    if options.save_mode_full {
        let header = CkptHeader::from_adamw(
            options.trainer,
            options.step,
            optimizer,
            options.rank,
            options.alpha,
            options.seed,
            options.config_hash.to_string(),
        );
        let named = named_parameters()?;
        if let Err(err) = checkpoint::save_full(options.path, &named, optimizer, &header) {
            log::warn!("{} full save failed: {err}", options.label);
        }
    } else if let Err(err) = save_weights() {
        log::warn!("{} save_weights failed: {err}", options.label);
    } else {
        log::info!("{} {}", options.label, options.path.display());
    }

    Ok(())
}

#[derive(Clone, Copy)]
enum CheckpointFailurePolicy {
    Warn,
    Propagate,
}

fn save_lora_checkpoint_impl(
    options: CheckpointSaveOptions<'_>,
    optimizer: &Optimizer,
    named_parameters: impl FnOnce() -> anyhow::Result<Vec<(String, Parameter)>>,
    save_weights: impl FnOnce() -> anyhow::Result<()>,
    failure_policy: CheckpointFailurePolicy,
) -> anyhow::Result<()> {
    if options.save_mode_full {
        if let Optimizer::AdamW(adam) = optimizer {
            let header = CkptHeader::from_adamw(
                options.trainer,
                options.step,
                adam,
                options.rank,
                options.alpha,
                options.seed,
                options.config_hash.to_string(),
            );
            let named = named_parameters()?;
            if let Err(err) = checkpoint::save_full(options.path, &named, adam, &header) {
                match failure_policy {
                    CheckpointFailurePolicy::Warn => {
                        log::warn!("{} full save failed: {err}", options.label);
                    }
                    CheckpointFailurePolicy::Propagate => {
                        return Err(anyhow::anyhow!("{} full save failed: {err}", options.label));
                    }
                }
            }
        } else {
            log::warn!(
                "{} full-state save not yet implemented for {:?}; saving weights only",
                options.label,
                optimizer.kind()
            );
            if let Err(err) = save_weights() {
                match failure_policy {
                    CheckpointFailurePolicy::Warn => {
                        log::warn!("{} weights-only save failed: {err}", options.label);
                    }
                    CheckpointFailurePolicy::Propagate => {
                        return Err(anyhow::anyhow!(
                            "{} weights-only save failed: {err}",
                            options.label
                        ));
                    }
                }
            } else {
                log::info!(
                    "{} weights-only checkpoint: {}",
                    options.label,
                    options.path.display()
                );
            }
        }
    } else if let Err(err) = save_weights() {
        match failure_policy {
            CheckpointFailurePolicy::Warn => {
                log::warn!("{} save_weights failed: {err}", options.label);
            }
            CheckpointFailurePolicy::Propagate => {
                return Err(anyhow::anyhow!("{} save_weights failed: {err}", options.label));
            }
        }
    } else {
        log::info!("{} {}", options.label, options.path.display());
    }

    Ok(())
}

pub fn run_manual_train_loop(
    config: ManualTrainLoopConfig,
    board: Option<&BoardWriter>,
    mut train_step: impl FnMut(usize) -> anyhow::Result<TrainStepMetrics>,
) -> anyhow::Result<TrainingCompletion> {
    let mut run = ManualTrainLoopRun::new(config);

    for step in run.steps() {
        let metrics = train_step(step)?;
        run.record_and_log(step, metrics, board);
    }

    Ok(run.finish())
}

pub fn run_simple_step_trainer<T: SimpleStepTrainer>(
    trainer: &mut T,
    loop_config: SimpleTrainLoopConfig,
    optimizer_config: SimpleOptimizerConfig,
    board: Option<&BoardWriter>,
) -> anyhow::Result<()> {
    let mut opt = optimizer_config.build();
    let clipper = GradientClipper::new(GradientClipStrategy::ClipByNorm {
        max_norm: optimizer_config.max_grad_norm,
    });

    let mut lr_config = TrainConfig::default();
    lr_config.learning_rate = optimizer_config.lr as f64;

    let t_start = Instant::now();
    for step in 0..loop_config.total_steps {
        let step_loss = trainer.step_loss(step)?;
        let loss_val = match step_loss.loss_value {
            Some(value) => value,
            None => step_loss.loss.to_vec_f32()?[0],
        };

        let grads = backward_loss(&step_loss.loss, false)?;
        let params = trainer.trainable_parameters();
        accumulate_parameter_grads(params, &grads)?;
        let grad_norm = clip_parameter_grads(params, &clipper)?;

        let cur_lr = eridiffusion_core::training::levers::lr(
            &lr_config,
            optimizer_config.lr,
            step,
            loop_config.total_steps,
            loop_config.warmup_steps,
        );
        step_optimizer(&mut opt, params, cur_lr, || Ok(()))?;

        log_training_step_with_resume(
            &loop_config.tag,
            step,
            loop_config.resume_step,
            loop_config.total_steps,
            loop_config.dataset_len,
            loop_config.batch_size,
            loss_val,
            grad_norm,
            cur_lr,
            t_start,
            board,
        );
    }

    log::info!("Training complete: {} steps", loop_config.total_steps);
    trainer.save_final()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_train_loop_config_defaults_to_fresh_run() {
        let cfg = SimpleTrainLoopConfig::fresh("ideogram", 10, 3, 1, 2);
        assert_eq!(cfg.tag, "ideogram");
        assert_eq!(cfg.total_steps, 10);
        assert_eq!(cfg.dataset_len, 3);
        assert_eq!(cfg.batch_size, 1);
        assert_eq!(cfg.warmup_steps, 2);
        assert_eq!(cfg.resume_step, 0);
    }

    #[test]
    fn simple_optimizer_config_parses_existing_optimizer_names() {
        let cfg = SimpleOptimizerConfig::from_optimizer_name("adamw", 1e-4, 1.0).unwrap();
        assert_eq!(cfg.kind, OptimizerKind::AdamW);
        assert_eq!(cfg.lr, 1e-4);
        assert_eq!(cfg.max_grad_norm, 1.0);
    }

    #[test]
    fn manual_train_loop_config_captures_resume_shape() {
        let cfg = ManualTrainLoopConfig::new("manual", 4, 10, 8, 2);
        assert_eq!(cfg.tag, "manual");
        assert_eq!(cfg.start_step, 4);
        assert_eq!(cfg.total_steps, 10);
        assert_eq!(cfg.progress_resume_step, 4);
        assert_eq!(cfg.progress_total_steps, 10);
        assert_eq!(cfg.dataset_len, 8);
        assert_eq!(cfg.batch_size, 2);
    }

    #[test]
    fn manual_train_loop_config_can_override_progress_target() {
        let cfg = ManualTrainLoopConfig::new("manual", 0, 6, 8, 2)
            .with_progress_target(10, 16);
        assert_eq!(cfg.start_step, 0);
        assert_eq!(cfg.total_steps, 6);
        assert_eq!(cfg.progress_resume_step, 10);
        assert_eq!(cfg.progress_total_steps, 16);
    }

    #[test]
    fn manual_train_loop_run_accumulates_completion() {
        let cfg = ManualTrainLoopConfig::new("manual", 2, 4, 8, 2);
        let mut run = ManualTrainLoopRun::new(cfg);
        run.record_and_log(
            2,
            TrainStepMetrics {
                loss_value: 1.5,
                grad_norm: 0.25,
                learning_rate: 1e-4,
            },
            None,
        );
        run.record_and_log(
            3,
            TrainStepMetrics {
                loss_value: 0.5,
                grad_norm: 0.20,
                learning_rate: 1e-4,
            },
            None,
        );

        let completion = run.completion();
        assert_eq!(completion.trained_steps, 2);
        assert_eq!(completion.average_loss, 1.0);
    }

    #[test]
    fn manual_train_loop_run_tracks_custom_progress_records() {
        let cfg = ManualTrainLoopConfig::new("manual", 0, 4, 8, 2)
            .with_progress_target(10, 14);
        let mut run = ManualTrainLoopRun::new(cfg);
        run.record_and_log_as(
            "manual-fwd-only",
            2,
            TrainStepMetrics {
                loss_value: 3.0,
                grad_norm: 0.0,
                learning_rate: 0.0,
            },
            None,
        );

        assert_eq!(run.total_loss(), 3.0);
        assert_eq!(run.average_loss_so_far(2), 1.0);
    }

    #[test]
    fn gradient_clip_options_enable_common_guards() {
        let cfg = GradientClipOptions::clip_by_norm(2.0)
            .with_clipping_enabled(false)
            .require_gradients()
            .require_finite_norm();
        assert_eq!(cfg.max_norm, 2.0);
        assert!(!cfg.clipping_enabled);
        assert!(cfg.require_gradients);
        assert!(cfg.require_finite_norm);
    }
}
