//! Captioner tab — Caption Model / Folder Processing panels (mirrors the Mojo
//! CaptionerTab settings surface).
//!
//! Honesty discipline: the captioner runner is a separate pure-Mojo bridge the
//! EDv2 launcher does not invoke yet — the action buttons are `[not wired]`.

use eframe::egui;

use crate::config::{
    captioner_attention_options, captioner_model_options, captioner_quant_options,
    captioner_resolution_options, TrainConfig,
};
use crate::widgets::{combo_row, edit_row, field_row, form_panel, slider_row, toggle_row};

pub fn render(ui: &mut egui::Ui, cfg: &mut TrainConfig) {
    let model_opts = captioner_model_options();
    let quant_opts = captioner_quant_options();
    let attention_opts = captioner_attention_options();
    let resolution_opts = captioner_resolution_options();

    form_panel(ui, "CAPTION MODEL", "Qwen VL caption model settings", |ui| {
        combo_row(ui, "cap_model", "Model", &model_opts, &mut cfg.captioner_model_index);
        let is_custom = model_opts
            .get(cfg.captioner_model_index)
            .map(|s| s == "Custom...")
            .unwrap_or(false);
        if is_custom {
            edit_row(ui, "Custom ID", &mut cfg.captioner_custom_model_id);
        }
        combo_row(ui, "cap_quant", "Quant", &quant_opts, &mut cfg.captioner_quant_index);
        combo_row(ui, "cap_attention", "Attention", &attention_opts, &mut cfg.captioner_attention_index);
        combo_row(ui, "cap_resolution", "Resolution", &resolution_opts, &mut cfg.captioner_resolution_index);
        field_row(ui, "Backend", "Pure Mojo command bridge");
    });

    form_panel(ui, "FOLDER PROCESSING", "Recursive scan, prompt controls, sidecar output", |ui| {
        edit_row(ui, "Folder", &mut cfg.captioner_folder_path);
        edit_row(ui, "Prompt", &mut cfg.captioner_prompt);
        toggle_row(ui, "Skip Existing", &mut cfg.captioner_skip_existing, ".txt exists");
        toggle_row(ui, "Summary", &mut cfg.captioner_summary_mode, "Short summary");
        toggle_row(ui, "One Sentence", &mut cfg.captioner_one_sentence_mode, "Constrain output");
        toggle_row(ui, "Retain Preview", &mut cfg.captioner_retain_preview, "On skipped media");
        slider_row(ui, "Max Tokens", &mut cfg.captioner_max_tokens, 32.0, 512.0);
        // The captioner runner is not invoked by this launcher yet.
        ui.horizontal(|ui| {
            ui.add_enabled(false, egui::Button::new("Scan Folder [not wired]"));
            ui.add_enabled(false, egui::Button::new("Queue Caption Run [not wired]"));
        });
    });
}
