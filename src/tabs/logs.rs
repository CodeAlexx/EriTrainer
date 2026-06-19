//! Logs tab — the live trainer stdout scrollback (the Runtime's 200-line ring
//! buffer), plus the artifact paths. Mono terminal-style pane on a near-black
//! background, auto-scrolled to the newest line.

use eframe::egui;

use crate::runtime::Runtime;
use crate::tokens;
use crate::widgets::{field_row, form_panel};

pub fn render(ui: &mut egui::Ui, rt: &Runtime) {
    let t = tokens::current();

    form_panel(ui, "LOGS", "Live trainer stdout (newest at the bottom)", |ui| {
        field_row(ui, "Command", text_or(&rt.last_command, "—"));
        field_row(ui, "Source", text_or(&rt.progress_source, "waiting"));
        field_row(ui, "Lines", &rt.logs.len().to_string());
        ui.add_space(6.0);

        // Terminal-style pane.
        egui::Frame::new()
            .fill(egui::Color32::from_rgb(12, 10, 8))
            .stroke(egui::Stroke::new(1.0, t.line))
            .corner_radius(5)
            .inner_margin(10)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .max_height(440.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        if rt.logs.is_empty() {
                            ui.label(
                                egui::RichText::new("no log lines yet — start a run")
                                    .monospace()
                                    .size(12.0)
                                    .color(t.ink_mute),
                            );
                        }
                        for line in &rt.logs {
                            // Color by severity prefix (fail-loud lines stand out).
                            let upper = line.to_uppercase();
                            let color = if upper.contains("FAILED") || upper.contains("BLOCKED") {
                                t.err
                            } else if upper.contains("[STDERR]") || upper.contains("WARN") {
                                t.warn
                            } else if line.contains("complete") || line.contains("launch ") {
                                t.ok
                            } else {
                                egui::Color32::from_rgb(217, 207, 189)
                            };
                            ui.label(egui::RichText::new(line).monospace().size(12.0).color(color));
                        }
                    });
            });
    });
}

fn text_or<'a>(s: &'a str, fallback: &'a str) -> &'a str {
    if s.trim().is_empty() {
        fallback
    } else {
        s
    }
}
