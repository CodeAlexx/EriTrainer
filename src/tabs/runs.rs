//! Runs tab — past/active run history. Honest stub: the launcher does not yet
//! persist a run index, so this surface lists nothing rather than faking it.

use eframe::egui;

use crate::config::TrainConfig;
use crate::widgets::{field_row, form_panel};

pub fn render(ui: &mut egui::Ui, cfg: &mut TrainConfig) {
    form_panel(ui, "RUNS [not yet built]", "Run history is not yet persisted by the launcher", |ui| {
        field_row(ui, "Current Run", &cfg.run_name);
        field_row(ui, "Backend", &cfg.backend_target);
        field_row(ui, "Workspace", &cfg.workspace_dir);
        ui.label(
            egui::RichText::new("No past runs indexed yet — start a run to populate this.")
                .weak()
                .small(),
        );
    });
}
