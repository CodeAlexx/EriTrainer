//! LoRA / OFT tab — Parameters / Targets & OFT panels (mirrors the Mojo
//! LoraTab).
//!
//! Honesty discipline: the PEFT Type selector here is decorative (every wired
//! runner trains plain LoRA); a non-LORA value is surfaced via
//! `ignored_lever_summary`.

use eframe::egui;

use crate::config::{peft_options, precision_options, TrainConfig};
use crate::widgets::{combo_str_row, field_row, form_panel, slider_row, toggle_row};

pub fn render(ui: &mut egui::Ui, cfg: &mut TrainConfig) {
    let peft_opts = peft_options();
    let precision_opts = precision_options();

    form_panel(ui, "LORA PARAMETERS", "Adapter type, rank, alpha, dropout", |ui| {
        combo_str_row(ui, "lora_peft", "PEFT Type [LORA only]", &peft_opts, &mut cfg.peft_type);
        field_row(ui, "LoRA Name", &cfg.lora_model_name);
        slider_row(ui, "LoRA Rank", &mut cfg.lora_rank, 1.0, 256.0);
        slider_row(ui, "LoRA Alpha", &mut cfg.lora_alpha, 1.0, 256.0);
        slider_row(ui, "Dropout", &mut cfg.lora_dropout, 0.0, 0.5);
        combo_str_row(ui, "lora_weight_dtype", "Weight DType", &precision_opts, &mut cfg.lora_weight_dtype);
    });

    form_panel(ui, "TARGETS & OFT", "Module target and OFT controls", |ui| {
        field_row(ui, "Transformer", "double/single projections");
        field_row(ui, "Targets", "qkv, proj, mlp, modulation");
        slider_row(ui, "OFT Block Size", &mut cfg.oft_block_size, 2.0, 32.0);
        toggle_row(ui, "COFT", &mut cfg.oft_coft, "Enabled");
        toggle_row(ui, "Bundle Embeds", &mut cfg.bundle_additional_embeddings, "Enabled");
        field_row(ui, "Backend", &cfg.backend_target);
    });
}
