//! Validations tab — Validations list / Validation Detail panels (mirrors the
//! Mojo ConceptTab). Concepts are read-only summary rows here; editing the
//! concept set is not yet wired into the launcher.

use eframe::egui;

use crate::config::TrainConfig;
use crate::widgets::{field_row, form_panel, toggle_row};

pub fn render(ui: &mut egui::Ui, cfg: &mut TrainConfig) {
    form_panel(ui, "VALIDATIONS", "Serenity trainer gates", |ui| {
        if cfg.concepts.is_empty() {
            ui.label(egui::RichText::new("No validations configured").weak());
        } else {
            for c in cfg.concepts.iter() {
                field_row(
                    ui,
                    &c.name,
                    &format!("{} imgs x{}", c.image_count, c.repeats),
                );
            }
        }
    });

    form_panel(ui, "VALIDATION DETAIL", "Current training inputs", |ui| {
        // Borrow the first concept's display fields, then its mutable toggle.
        let (path, trigger, image_count) = cfg
            .concepts
            .first()
            .map(|c| (c.path.clone(), c.trigger.clone(), c.image_count))
            .unwrap_or_default();
        if cfg.concepts.is_empty() {
            field_row(ui, "Validations", "No validations configured");
        } else {
            field_row(ui, "Dataset", &path);
            field_row(ui, "Trigger", &trigger);
            field_row(ui, "Images", &image_count.to_string());
            field_row(ui, "Cache", &cfg.cache_dir);
            field_row(ui, "Model", &cfg.base_model_path);
            field_row(ui, "Steps", &cfg.max_train_steps.to_string());
            if let Some(c) = cfg.concepts.first_mut() {
                toggle_row(ui, "Enabled", &mut c.enabled, "Run validation gates");
            }
        }
    });
}
