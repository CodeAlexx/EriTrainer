//! Cloud tab — Cloud / Sync panels (mirrors the Mojo CloudTab).
//!
//! Honesty discipline: the launcher runs trainers locally; cloud dispatch is
//! not wired, so the panel is labelled `[not wired]`.

use eframe::egui;

use crate::config::{cloud_type_options, TrainConfig};
use crate::widgets::{combo_row, edit_row, field_row, form_panel, toggle_row};

pub fn render(ui: &mut egui::Ui, cfg: &mut TrainConfig) {
    let cloud_opts = cloud_type_options();

    form_panel(ui, "CLOUD [not wired]", "Cloud dispatch is not wired - local launch only", |ui| {
        combo_row(ui, "cloud_type", "Cloud Type", &cloud_opts, &mut cfg.cloud_type_index);
        edit_row(ui, "Host", &mut cfg.cloud_host);
        edit_row(ui, "Port", &mut cfg.cloud_port);
        edit_row(ui, "User", &mut cfg.cloud_user);
        edit_row(ui, "Remote Dir", &mut cfg.cloud_workspace_dir);
        toggle_row(ui, "Delete Work Dir", &mut cfg.cloud_delete_workspace, "After run");
    });

    form_panel(ui, "SYNC", "File sync and post-run behavior", |ui| {
        field_row(ui, "Upload", "config, concepts, cache metadata");
        field_row(ui, "Download", "samples, checkpoints, logs");
        field_row(ui, "File Sync", "Native SCP / Fabric SFTP");
        field_row(ui, "On Finish", "None / Stop / Delete");
    });
}
