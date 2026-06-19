//! Dark "serenity" visuals, matching the Mojo `trainer_theme` palette: dense
//! near-black panels, a slightly lighter window/widget fill, weak vs strong
//! text, and a cyan accent used for the active nav row and the RUNNING pill.
//!
//! The palette constants are public so nav/topbar/rail can paint accents that
//! agree with the global `Visuals` set here.

use eframe::egui;

// --- Serenity palette (approximate; dense dark) ---

/// App background behind the 3-column shell.
pub const BG: egui::Color32 = egui::Color32::from_rgb(15, 17, 22);
/// Panel fill (nav / rail / central body).
pub const PANEL: egui::Color32 = egui::Color32::from_rgb(18, 20, 26);
/// Window / widget fill (text edits, buttons, form panels).
pub const WINDOW: egui::Color32 = egui::Color32::from_rgb(24, 27, 34);
/// Slightly raised fill for hovered / active widgets.
pub const RAISED: egui::Color32 = egui::Color32::from_rgb(31, 35, 44);
/// Hairline separators / weak strokes.
pub const STROKE: egui::Color32 = egui::Color32::from_rgb(44, 49, 60);

/// Primary text.
pub const TEXT_STRONG: egui::Color32 = egui::Color32::from_rgb(222, 226, 234);
/// Secondary / label text.
pub const TEXT_WEAK: egui::Color32 = egui::Color32::from_rgb(150, 157, 170);

/// Cyan accent: active nav row, RUNNING pill, progress bar.
pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(56, 189, 208);
/// Amber: PAUSED pill, capability warnings.
pub const WARN: egui::Color32 = egui::Color32::from_rgb(228, 178, 76);
/// Muted slate: IDLE pill.
pub const IDLE: egui::Color32 = egui::Color32::from_rgb(108, 116, 130);

pub fn apply(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();

    visuals.panel_fill = PANEL;
    visuals.window_fill = WINDOW;
    visuals.extreme_bg_color = BG;
    visuals.faint_bg_color = egui::Color32::from_rgb(21, 24, 30);
    visuals.override_text_color = Some(TEXT_STRONG);
    visuals.hyperlink_color = ACCENT;
    visuals.selection.bg_fill = ACCENT.linear_multiply(0.35);
    visuals.selection.stroke = egui::Stroke::new(1.0, ACCENT);

    visuals.widgets.noninteractive.bg_fill = PANEL;
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, STROKE);
    visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, TEXT_WEAK);

    visuals.widgets.inactive.bg_fill = WINDOW;
    visuals.widgets.inactive.weak_bg_fill = WINDOW;
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, STROKE);
    visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT_STRONG);

    visuals.widgets.hovered.bg_fill = RAISED;
    visuals.widgets.hovered.weak_bg_fill = RAISED;
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, ACCENT.linear_multiply(0.6));
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, TEXT_STRONG);

    visuals.widgets.active.bg_fill = RAISED;
    visuals.widgets.active.weak_bg_fill = RAISED;
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, TEXT_STRONG);

    visuals.window_stroke = egui::Stroke::new(1.0, STROKE);
    visuals.window_corner_radius = egui::CornerRadius::same(6);

    ctx.set_visuals(visuals);
}
