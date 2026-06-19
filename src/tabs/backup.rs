//! Backup tab — Backup / Save panels. Mirrors the OneTrainer/Mojo backup
//! field set. These fields ride along in the runner config snapshot; the
//! save-cadence keys (`save_every`/`save_skip_first`) reach the runner.

use eframe::egui;

use crate::config::TrainConfig;
use crate::widgets::{drag_row, edit_row, form_panel, toggle_row};

pub fn render(ui: &mut egui::Ui, cfg: &mut TrainConfig) {
    form_panel(ui, "BACKUP", "Rolling backup cadence", |ui| {
        drag_row(ui, "Backup After", &mut cfg.backup_after, 10.0);
        toggle_row(ui, "Rolling", &mut cfg.rolling_backup, "Enabled");
        drag_row(ui, "Rolling Count", &mut cfg.rolling_backup_count, 1.0);
        toggle_row(ui, "Before Save", &mut cfg.backup_before_save, "Backup before save");
    });

    form_panel(ui, "SAVE", "Checkpoint save cadence and retention", |ui| {
        drag_row(ui, "Save Every", &mut cfg.save_every, 10.0);
        drag_row(ui, "Skip First", &mut cfg.save_skip_first, 1.0);
        drag_row(ui, "Max Keep", &mut cfg.save_max_keep, 1.0);
        edit_row(ui, "Filename Prefix", &mut cfg.save_filename_prefix);
    });
}
