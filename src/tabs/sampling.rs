//! Sampling tab — Settings / Prompts panels (mirrors the Mojo SamplingTab;
//! the sample media virtual grid is not part of the launcher UI).

use eframe::egui;

use crate::config::{sample_sampler_options, TrainConfig};
use crate::widgets::{browse_row, combo_str_row, drag_row, edit_row, field_row, form_panel, toggle_row};

pub fn render(ui: &mut egui::Ui, cfg: &mut TrainConfig) {
    let sampler_opts = sample_sampler_options();

    form_panel(ui, "SAMPLING SETTINGS", "Serenity sampling cadence and sampler", |ui| {
        browse_row(ui, "Sample Dir", &mut cfg.sample_output_dir, true);
        drag_row(ui, "Sample After", &mut cfg.sample_after, 10.0);
        drag_row(ui, "Skip First", &mut cfg.sample_skip_first, 1.0);
        drag_row(ui, "Steps", &mut cfg.sample_steps, 1.0);
        drag_row(ui, "CFG", &mut cfg.sample_cfg, 0.1);
        combo_str_row(ui, "sm_sampler", "Sampler", &sampler_opts, &mut cfg.sample_sampler);
        toggle_row(ui, "SerenityBoard", &mut cfg.samples_to_tensorboard, "Send samples");
        toggle_row(ui, "Non-EMA", &mut cfg.non_ema_sampling, "Use non-EMA model");
        field_row(ui, "Preset", &cfg.sampler_preset);
    });

    form_panel(ui, "SAMPLE PROMPTS", "Prompts shown in the preview panel", |ui| {
        for (i, sample) in cfg.samples.iter_mut().enumerate() {
            edit_row(ui, &format!("Prompt {}", i + 1), &mut sample.prompt);
            edit_row(ui, &format!("Negative {}", i + 1), &mut sample.negative_prompt);
            drag_row_i32(ui, &format!("Seed {}", i + 1), &mut sample.seed);
        }
    });
}

/// Small i32 drag helper for sample seeds (the shared drag_row is f32).
fn drag_row_i32(ui: &mut egui::Ui, label: &str, value: &mut i32) {
    ui.horizontal(|ui| {
        ui.add_sized([178.0, ui.spacing().interact_size.y], egui::Label::new(label));
        ui.add(egui::DragValue::new(value).speed(1.0));
    });
}
