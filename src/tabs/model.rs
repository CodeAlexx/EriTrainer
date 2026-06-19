//! Model tab — Base Model + Output panels (mirrors the Mojo ModelTab).
//!
//! Honesty discipline: every wired runner trains LoRA only, so the Train
//! Method selector is labelled `[LoRA only]` and surfaced via
//! `ignored_lever_summary` when non-default.

use eframe::egui;

use crate::config::{
    architecture_options, model_type_options, output_format_options, precision_options,
    training_method_options, TrainConfig,
};
use crate::widgets::{browse_row, combo_row, combo_str_row, field_row, form_panel, toggle_row};

pub fn render(ui: &mut egui::Ui, cfg: &mut TrainConfig) {
    let method_opts = training_method_options();
    let model_opts = model_type_options();
    let arch_opts = architecture_options();
    let format_opts = output_format_options();
    let precision_opts = precision_options();

    form_panel(ui, "BASE MODEL", "Method, architecture, and checkpoint", |ui| {
        combo_row(ui, "model_train_method", "Train Method [LoRA only]", &method_opts, &mut cfg.training_method_index);
        // Model Type / Architecture drive the launch identity: when either
        // changes, re-apply the per-model preset so cfg.model_type /
        // backend_target / recipe defaults follow the selection (the launcher,
        // "Live target" label, and ignored_lever_summary all read them).
        let model_changed =
            combo_row(ui, "model_model_type", "Model Type", &model_opts, &mut cfg.model_type_index);
        let arch_changed =
            combo_row(ui, "model_architecture", "Architecture", &arch_opts, &mut cfg.architecture_index);
        if model_changed || arch_changed {
            cfg.apply_model_preset(model_changed);
        }
        browse_row(ui, "Base Model", &mut cfg.base_model_path, false);
        // Path to the EDv2 trainer --config JSON (dataset/recipe). Required to
        // launch; e.g. EriDiffusion-v2/configs/klein9b_alina.json.
        browse_row(ui, "Run Config", &mut cfg.run_config_path, false);
        field_row(ui, "VAE", &cfg.vae_override);
        toggle_row(ui, "Transformer", &mut cfg.train_transformer, "Train transformer");
        toggle_row(ui, "Text Encoder", &mut cfg.train_text_encoder, "Train text encoder");
    });

    form_panel(ui, "OUTPUT", "Destination, format, and backend", |ui| {
        browse_row(ui, "Destination", &mut cfg.output_dir, true);
        combo_str_row(ui, "model_output_format", "Format", &format_opts, &mut cfg.output_model_format);
        combo_str_row(ui, "model_output_dtype", "Output DType", &precision_opts, &mut cfg.output_dtype);
        field_row(ui, "Backend", &cfg.backend_target);
        field_row(ui, "Checkpoint", &cfg.base_model_path);
        toggle_row(ui, "Bundle Embeds", &mut cfg.bundle_additional_embeddings, "Enabled");
    });
}
