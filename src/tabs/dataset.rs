//! Dataset tab — Dataset / Preview & Captions panels (mirrors the Mojo
//! DatasetTab; the media virtual grid is not part of the launcher UI).
//!
//! Honesty discipline: Aspect Buckets is `[not wired]` — bucketing is baked
//! into the prepared caches and no runner consumes the toggle.

use eframe::egui;

use crate::config::{resolution_options, TrainConfig};
use crate::widgets::{combo_str_row, edit_row, field_row, form_panel, slider_row, toggle_row};

pub fn render(ui: &mut egui::Ui, cfg: &mut TrainConfig) {
    let resolution_opts = resolution_options();

    form_panel(ui, "DATASET", "Image path, concept file, and cache policy", |ui| {
        edit_row(ui, "Dataset Path", &mut cfg.dataset_path);
        field_row(ui, "Concept File", &cfg.concept_file_name);
        combo_str_row(ui, "ds_resolution", "Resolution", &resolution_opts, &mut cfg.resolution);
        // No runner consumes this toggle — bucketing is baked into the cache.
        toggle_row(ui, "Aspect Buckets [not wired]", &mut cfg.aspect_ratio_bucketing, "Enabled");
        toggle_row(ui, "Latent Caching", &mut cfg.latent_caching, "Enabled");
        toggle_row(ui, "Clear Cache", &mut cfg.clear_cache_before_training, "Before training");
    });

    form_panel(ui, "PREVIEW & CAPTIONS", "Dataset summary", |ui| {
        field_row(ui, "Caption Source", "sidecar text files");
        slider_row(ui, "Caption Dropout", &mut cfg.caption_dropout, 0.0, 0.5);
        let trigger = cfg
            .concepts
            .first()
            .map(|c| c.trigger.clone())
            .unwrap_or_default();
        field_row(ui, "Trigger", &trigger);
        field_row(ui, "Cache Format", "safetensors latent/cap");
    });
}
