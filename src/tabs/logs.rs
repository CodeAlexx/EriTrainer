//! Logs tab — live trainer stdout. Honest stub: the log buffer lives on the
//! Runtime (backbone-owned). Until the backbone exposes it to tabs, this
//! surface says so rather than rendering an empty pane that looks functional.

use eframe::egui;

use crate::config::TrainConfig;
use crate::widgets::form_panel;

pub fn render(ui: &mut egui::Ui, _cfg: &mut TrainConfig) {
    form_panel(ui, "LOGS [not yet built]", "Live trainer stdout streams in the right rail", |ui| {
        ui.label(
            egui::RichText::new(
                "Trainer stdout is parsed into the right-rail live stats. A full scrollback \
                 log pane is not yet wired into this tab.",
            )
            .weak(),
        );
    });
}
