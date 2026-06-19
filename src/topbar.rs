//! Top action bar, matching the Mojo `TopBar.mojo` (two rows).
//!
//! Row 1: `SECTION — <label>` · run-name edit · `Live target — <model_type>` ·
//!        Start/Stop (Start when idle → `rt.start(cfg)`, Stop when running →
//!        `rt.stop()`).
//! Row 2: `VALIDATION — <validate()>` plus, when `ignored_lever_summary()` is
//!        non-empty, `  |  WARNING <summary>`; then Pause/Resume, Sample, Save
//!        each suffixed `[not wired]` and DISABLED.
//!
//! Honesty discipline (from the Mojo TopBar): any control the launched runner
//! does not consume is named here, before launch, and rendered disabled — no
//! widget may silently lie. Pause/Sample/Save are not wired yet.

use eframe::egui;

use crate::config::{Section, TrainConfig};
use crate::runtime::Runtime;
use crate::theme;

pub fn top_bar(ui: &mut egui::Ui, cfg: &mut TrainConfig, rt: &mut Runtime, section: Section) {
    ui.add_space(4.0);

    // --- Row 1: section · run name · live target · Start/Stop ---
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(format!("SECTION — {}", section.label()))
                .strong()
                .color(theme::TEXT_STRONG),
        );
        ui.separator();
        ui.add(
            egui::TextEdit::singleline(&mut cfg.run_name)
                .hint_text("run name")
                .desired_width(220.0),
        );
        ui.separator();
        ui.label(
            egui::RichText::new(format!("Live target — {}", cfg.model_type))
                .color(theme::TEXT_WEAK),
        );

        // Right-align the Start/Stop button.
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if rt.has_running {
                let stop = egui::Button::new(
                    egui::RichText::new("Stop training").color(egui::Color32::WHITE),
                )
                .fill(egui::Color32::from_rgb(176, 64, 64));
                if ui.add(stop).clicked() {
                    rt.stop();
                }
            } else {
                let start = egui::Button::new(
                    egui::RichText::new("Start training").color(egui::Color32::BLACK),
                )
                .fill(theme::ACCENT);
                if ui.add(start).clicked() {
                    rt.start(cfg);
                }
            }
        });
    });

    ui.add_space(2.0);

    // --- Row 2: validation + capability warning · Pause/Resume · Sample · Save ---
    ui.horizontal(|ui| {
        let ignored = cfg.ignored_lever_summary();
        ui.label(
            egui::RichText::new(format!("VALIDATION — {}", cfg.validate()))
                .color(theme::TEXT_WEAK),
        );
        if !ignored.is_empty() {
            ui.label(
                egui::RichText::new(format!("|  WARNING {}", ignored))
                    .color(theme::WARN),
            );
        }
        // Newly-wired models are launch-wired but not smoke-tested — say so loud.
        if !crate::config::model_verified(&cfg.model_type) {
            ui.label(
                egui::RichText::new("|  UNVERIFIED — launch wired but not smoke-tested")
                    .strong()
                    .color(theme::WARN),
            );
        }

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // Save is rightmost in right_to_left layout → matches LTR order
            // Pause/Resume · Sample · Save when laid out from the right.
            ui.add_enabled(false, egui::Button::new("Save checkpoint [not wired]"));
            ui.add_enabled(false, egui::Button::new("Sample [not wired]"));
            // Pause/Resume mirrors the running state, but stays disabled until
            // the wave-3 command-file protocol lands.
            let pause_label = if rt.has_running && rt.paused {
                "Resume [not wired]"
            } else {
                "Pause [not wired]"
            };
            ui.add_enabled(false, egui::Button::new(pause_label));
        });
    });

    ui.add_space(2.0);
}
