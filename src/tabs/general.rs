//! General tab — Workspace / Debug & Validation / Device / Multi-GPU panels
//! (mirrors the Mojo GeneralTab).

use eframe::egui;

use crate::config::{device_options, precision_options, TrainConfig};
use crate::widgets::{combo_str_row, drag_row, edit_row, field_row, form_panel, toggle_row};

pub fn render(ui: &mut egui::Ui, cfg: &mut TrainConfig) {
    form_panel(ui, "WORKSPACE", "Paths, cache policy, and safety", |ui| {
        edit_row(ui, "Workspace", &mut cfg.workspace_dir);
        edit_row(ui, "Cache", &mut cfg.cache_dir);
        toggle_row(ui, "Continue Backup", &mut cfg.continue_last_backup, "Continue from last backup");
        toggle_row(ui, "Only Cache", &mut cfg.only_cache, "Only cache");
        toggle_row(ui, "Overwrite", &mut cfg.prevent_overwrites, "Prevent overwrites");
        drag_row(ui, "Dataloader", &mut cfg.dataloader_threads, 1.0);
    });

    form_panel(ui, "DEBUG & VALIDATION", "SerenityBoard, debug, and validation", |ui| {
        toggle_row(ui, "Debug Mode", &mut cfg.debug_mode, "Enabled");
        field_row(ui, "Debug Dir", &cfg.debug_dir);
        toggle_row(ui, "SerenityBoard", &mut cfg.tensorboard, "Enabled");
        toggle_row(ui, "Always On", &mut cfg.tensorboard_always_on, "Keep SerenityBoard open");
        field_row(ui, "Board Port", &cfg.tensorboard_port);
        toggle_row(ui, "Validation", &mut cfg.validation, "Enabled");
    });

    let device_opts = device_options();
    let precision_opts = precision_options();
    form_panel(ui, "DEVICE", "Placement and precision defaults", |ui| {
        combo_str_row(ui, "general_train_device", "Train Device", &device_opts, &mut cfg.train_device);
        combo_str_row(ui, "general_temp_device", "Temp Device", &device_opts, &mut cfg.temp_device);
        combo_str_row(ui, "general_precision", "Train DType", &precision_opts, &mut cfg.train_dtype);
        toggle_row(ui, "Gradient CKPT", &mut cfg.gradient_checkpointing, "Enabled");
        toggle_row(ui, "Act Offload", &mut cfg.activation_offloading, "Enabled");
    });

    form_panel(ui, "MULTI-GPU", "Distributed trainer switches", |ui| {
        toggle_row(ui, "Multi-GPU", &mut cfg.multi_gpu, "Enabled");
        field_row(ui, "Devices", &cfg.device_indexes);
        toggle_row(ui, "Fused Reduce", &mut cfg.fused_gradient_reduce, "Enabled");
        toggle_row(ui, "Async Reduce", &mut cfg.async_gradient_reduce, "Enabled");
        drag_row(ui, "Layer Offload", &mut cfg.layer_offload_fraction, 0.05);
    });
}
