//! Training tab — BASE SCHEDULE / OPTIMIZER / PRECISION & MEMORY / EMA & TARGETS
//! / TEXT ENCODER / TRANSFORMER / NOISE & TIMESTEPS / MASKED TRAINING [not wired]
//! / LOSS / LAYER FILTER (mirrors the Mojo TrainingTab panel-for-panel).
//!
//! Honesty discipline (from the Mojo UI):
//! - MASKED TRAINING panel title carries `[not wired]` (no runner consumes it).
//! - The PEFT selector is decorative (`[LORA only]`): adapter_algo is never
//!   emitted to any runner. Both are surfaced via `ignored_lever_summary`.

use eframe::egui;

use crate::config::{
    device_options, ema_options, layer_filter_preset_options, loss_fn_options,
    loss_scaler_options, loss_weight_options, lr_scaler_options, optimizer_options,
    peft_options, precision_options, resolution_options, scheduler_options,
    timestep_distribution_options, TrainConfig,
};
use crate::widgets::{
    combo_row, combo_str_row, drag_row, field_row, form_panel, slider_row, toggle_row,
};

pub fn render(ui: &mut egui::Ui, cfg: &mut TrainConfig) {
    let optimizer_opts = optimizer_options();
    let scheduler_opts = scheduler_options();
    let precision_opts = precision_options();
    let lr_scaler_opts = lr_scaler_options();
    let ema_opts = ema_options();
    let device_opts = device_options();
    let resolution_opts = resolution_options();
    let timestep_opts = timestep_distribution_options();
    let loss_fn_opts = loss_fn_options();
    let loss_weight_opts = loss_weight_options();
    let loss_scaler_opts = loss_scaler_options();
    let layer_preset_opts = layer_filter_preset_options();
    let peft_opts = peft_options();

    form_panel(ui, "BASE SCHEDULE", "Serenity epoch, batch, and LR cycle fields", |ui| {
        slider_row(ui, "Epochs", &mut cfg.epochs, 1.0, 50.0);
        slider_row(ui, "Local Batch", &mut cfg.batch_size, 1.0, 16.0);
        slider_row(ui, "Accum Steps", &mut cfg.gradient_accumulation_steps, 1.0, 16.0);
        drag_row(ui, "Warmup", &mut cfg.learning_rate_warmup_steps, 10.0);
        if cfg.learning_rate_warmup_steps < 0.0 {
            cfg.learning_rate_warmup_steps = 0.0;
        }
        drag_row(ui, "LR Min Factor", &mut cfg.learning_rate_min_factor, 0.01);
        drag_row(ui, "LR Cycles", &mut cfg.learning_rate_cycles, 0.1);
        combo_str_row(ui, "tr_lr_scaler", "LR Scaler", &lr_scaler_opts, &mut cfg.learning_rate_scaler);
        drag_row(ui, "Seed", &mut cfg.seed, 1.0);
        if cfg.seed < 0.0 {
            cfg.seed = 0.0;
        }
        drag_row(ui, "Max Steps", &mut cfg.max_train_steps, 50.0);
        if cfg.max_train_steps < 1.0 {
            cfg.max_train_steps = 1.0;
        }
    });

    form_panel(ui, "OPTIMIZER", "Optimizer, scheduler, learning rates", |ui| {
        combo_row(ui, "tr_optimizer", "Optimizer", &optimizer_opts, &mut cfg.optimizer_index);
        combo_row(ui, "tr_scheduler", "Scheduler", &scheduler_opts, &mut cfg.scheduler_index);
        drag_row(ui, "Learning Rate", &mut cfg.learning_rate, 1e-5);
        if cfg.learning_rate < 1e-6 {
            cfg.learning_rate = 1e-6;
        }
        drag_row(ui, "Text Enc LR", &mut cfg.text_encoder_learning_rate, 1e-5);
        if cfg.text_encoder_learning_rate < 0.0 {
            cfg.text_encoder_learning_rate = 0.0;
        }
        drag_row(ui, "Transformer LR", &mut cfg.transformer_learning_rate, 1e-5);
        if cfg.transformer_learning_rate < 0.0 {
            cfg.transformer_learning_rate = 0.0;
        }
        drag_row(ui, "Weight Decay", &mut cfg.weight_decay, 0.001);
        drag_row(ui, "Clip Grad", &mut cfg.clip_grad_norm, 0.1);
    });

    form_panel(ui, "PRECISION & MEMORY", "Precision and VRAM switches", |ui| {
        combo_str_row(ui, "tr_train_dtype", "Train DType", &precision_opts, &mut cfg.train_dtype);
        combo_str_row(ui, "tr_fallback_dtype", "Fallback DType", &precision_opts, &mut cfg.fallback_train_dtype);
        toggle_row(ui, "Gradient CKPT", &mut cfg.gradient_checkpointing, "Enabled");
        toggle_row(ui, "Act Offload", &mut cfg.activation_offloading, "Enabled");
        slider_row(ui, "Offload Fraction", &mut cfg.layer_offload_fraction, 0.0, 1.0);
        toggle_row(ui, "Autocast Cache", &mut cfg.enable_autocast_cache, "Enabled");
        combo_str_row(ui, "tr_resolution", "Resolution", &resolution_opts, &mut cfg.resolution);
        field_row(ui, "Frames", &cfg.frames);
        toggle_row(ui, "Circular Pad", &mut cfg.force_circular_padding, "Force");
        combo_str_row(ui, "tr_train_device", "Train Device", &device_opts, &mut cfg.train_device);
    });

    form_panel(ui, "EMA & TARGETS", "EMA and trainable model sections", |ui| {
        combo_str_row(ui, "tr_ema", "EMA", &ema_opts, &mut cfg.ema_mode);
        drag_row(ui, "EMA Decay", &mut cfg.ema_decay, 0.001);
        drag_row(ui, "EMA Update", &mut cfg.ema_update_step_interval, 1.0);
        toggle_row(ui, "Transformer", &mut cfg.train_transformer, "Train");
        toggle_row(ui, "Text Encoder", &mut cfg.train_text_encoder, "Train");
        drag_row(ui, "TE Stop After", &mut cfg.text_encoder_stop_after, 1.0);
        drag_row(ui, "Tr Stop After", &mut cfg.transformer_stop_after, 1.0);
        field_row(ui, "Backend", &cfg.backend_target);
    });

    form_panel(ui, "TEXT ENCODER", "Serenity text encoder controls", |ui| {
        toggle_row(ui, "Train", &mut cfg.train_text_encoder, "Enabled");
        slider_row(ui, "Caption Dropout", &mut cfg.caption_dropout, 0.0, 0.5);
        drag_row(ui, "Stop After", &mut cfg.text_encoder_stop_after, 1.0);
        drag_row(ui, "Learning Rate", &mut cfg.text_encoder_learning_rate, 1e-5);
        field_row(ui, "Sequence Len", &cfg.text_encoder_sequence_length);
        field_row(ui, "Clip Skip", "not used by Flux2");
    });

    form_panel(ui, "TRANSFORMER", "Serenity transformer controls", |ui| {
        toggle_row(ui, "Train", &mut cfg.train_transformer, "Enabled");
        drag_row(ui, "Stop After", &mut cfg.transformer_stop_after, 1.0);
        drag_row(ui, "Learning Rate", &mut cfg.transformer_learning_rate, 1e-5);
        toggle_row(ui, "Attention Mask", &mut cfg.transformer_attention_mask, "Force");
        drag_row(ui, "Guidance", &mut cfg.transformer_guidance_scale, 0.1);
        field_row(ui, "Target", "Flux2 transformer");
    });

    form_panel(ui, "NOISE & TIMESTEPS", "Flow matching noising and timestep controls", |ui| {
        slider_row(ui, "Offset Noise", &mut cfg.offset_noise_weight, 0.0, 1.0);
        slider_row(ui, "Perturb Noise", &mut cfg.perturbation_noise_weight, 0.0, 1.0);
        combo_str_row(ui, "tr_timestep_dist", "Distribution", &timestep_opts, &mut cfg.timestep_distribution);
        slider_row(ui, "Min Strength", &mut cfg.min_noising_strength, 0.0, 1.0);
        slider_row(ui, "Max Strength", &mut cfg.max_noising_strength, 0.0, 1.0);
        drag_row(ui, "Noising Weight", &mut cfg.noising_weight, 0.1);
        drag_row(ui, "Noising Bias", &mut cfg.noising_bias, 0.1);
        drag_row(ui, "Time Shift", &mut cfg.timestep_shift, 0.1);
        toggle_row(ui, "Dynamic Shift", &mut cfg.dynamic_timestep_shifting, "Enabled");
        field_row(ui, "Prediction", "flow matching");
    });

    // Capability honesty: NO runner consumes masked-training — these widgets
    // are snapshot-only and excluded from every runner emission. The
    // [not wired] suffix + capability-table warning keep them from lying.
    form_panel(
        ui,
        "MASKED TRAINING [not wired]",
        "No runner consumes these yet - excluded from launch config",
        |ui| {
            toggle_row(ui, "Masked", &mut cfg.masked_training, "Enabled");
            slider_row(ui, "Unmasked Prob", &mut cfg.unmasked_probability, 0.0, 1.0);
            slider_row(ui, "Unmasked Weight", &mut cfg.unmasked_weight, 0.0, 2.0);
            toggle_row(ui, "Normalize Loss", &mut cfg.normalize_masked_area_loss, "Enabled");
            slider_row(ui, "Prior Weight", &mut cfg.masked_prior_preservation_weight, 0.0, 2.0);
            toggle_row(ui, "Custom Cond", &mut cfg.custom_conditioning_image, "Image");
        },
    );

    form_panel(ui, "LOSS", "Serenity loss mix and scaling", |ui| {
        // T1.A loss-fn selector (levers_loss_grad trainers, zimage first).
        // SEPARATE from the mse/mae/huber strength rows below — do not conflate.
        combo_str_row(ui, "tr_loss_fn", "Loss Fn", &loss_fn_opts, &mut cfg.loss_fn);
        drag_row(ui, "MSE", &mut cfg.mse_strength, 0.1);
        drag_row(ui, "MAE", &mut cfg.mae_strength, 0.1);
        drag_row(ui, "log-cosh", &mut cfg.log_cosh_strength, 0.1);
        drag_row(ui, "Huber", &mut cfg.huber_strength, 0.1);
        drag_row(ui, "Huber Delta", &mut cfg.huber_delta, 0.1);
        drag_row(ui, "SmoothL1 Beta", &mut cfg.smooth_l1_beta, 0.1);
        drag_row(ui, "VB", &mut cfg.vb_loss_strength, 0.1);
        combo_str_row(ui, "tr_loss_weight_fn", "Weight Fn", &loss_weight_opts, &mut cfg.loss_weight_fn);
        drag_row(ui, "Gamma", &mut cfg.loss_weight_strength, 0.1);
        // Min-SNR gamma (flow): SimpleTuner ε-style min(SNR,γ)/SNR weight,
        // 0 = off. NOTE divisor /SNR, not the klein loss_weight_fn lever.
        drag_row(ui, "Min-SNR Flow", &mut cfg.min_snr_gamma_flow, 0.1);
        combo_str_row(ui, "tr_loss_scaler", "Loss Scaler", &loss_scaler_opts, &mut cfg.loss_scaler);
        field_row(ui, "Backend", "Serenity loss mix");
    });

    form_panel(ui, "LAYER FILTER", "Target layer preset and explicit filter", |ui| {
        combo_str_row(ui, "tr_layer_preset", "Preset", &layer_preset_opts, &mut cfg.layer_filter_preset);
        field_row(ui, "Filter", &cfg.layer_filter);
        toggle_row(ui, "Regex", &mut cfg.layer_filter_regex, "Enabled");
        // PEFT selector is decorative: adapter_algo is never emitted to any
        // runner (UI always trains plain LoRA). Marked until LoKr/LoHa/OFT land.
        combo_str_row(ui, "tr_peft", "PEFT [LORA only]", &peft_opts, &mut cfg.peft_type);
        field_row(ui, "Target", "transformer blocks");
    });
}
