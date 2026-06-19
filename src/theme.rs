//! Theme: design-token visuals (dark default + light), Inter + JetBrains Mono
//! fonts, the typography scale and the dense instrument-panel spacing from the
//! Rust Trainer design handoff.
//!
//! The legacy palette constants (BG/PANEL/…/ACCENT) are kept so the existing
//! chrome (nav/topbar/rail) keeps compiling; they now source from
//! `tokens::dark()` so the whole UI shares one palette. New widgets read
//! `tokens::current()` directly (theme-aware).

use std::sync::Arc;

use eframe::egui;

use crate::tokens::{self, Tokens};

// --- Legacy palette consts used by the chrome (nav/topbar/rail),
//     single-sourced from the dark tokens. ---
pub const RAISED: egui::Color32 = tokens::dark().line_2;
pub const TEXT_STRONG: egui::Color32 = tokens::dark().ink;
pub const TEXT_WEAK: egui::Color32 = tokens::dark().ink_dim;
pub const ACCENT: egui::Color32 = tokens::dark().accent;
pub const WARN: egui::Color32 = tokens::dark().warn;
pub const IDLE: egui::Color32 = tokens::dark().ink_mute;

/// Install Inter (proportional) + JetBrains Mono (monospace). Call once.
pub fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "inter".to_owned(),
        Arc::new(egui::FontData::from_static(include_bytes!(
            "../assets/Inter-Regular.ttf"
        ))),
    );
    fonts.font_data.insert(
        "jbmono".to_owned(),
        Arc::new(egui::FontData::from_static(include_bytes!(
            "../assets/JetBrainsMono-Regular.ttf"
        ))),
    );
    if let Some(p) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
        p.insert(0, "inter".to_owned());
    }
    if let Some(m) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
        m.insert(0, "jbmono".to_owned());
    }
    ctx.set_fonts(fonts);
}

/// The typography scale + dense spacing from the handoff.
fn install_style(ctx: &egui::Context) {
    use egui::{FontFamily, FontId, TextStyle};
    let mut style = (*ctx.style()).clone();
    style.text_styles = [
        (TextStyle::Heading, FontId::new(14.0, FontFamily::Proportional)),
        (TextStyle::Body, FontId::new(13.0, FontFamily::Proportional)),
        (TextStyle::Button, FontId::new(13.0, FontFamily::Proportional)),
        (TextStyle::Monospace, FontId::new(12.0, FontFamily::Monospace)),
        (TextStyle::Small, FontId::new(11.0, FontFamily::Proportional)),
    ]
    .into();
    // Instrument-panel density (handoff §Density).
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(8.0, 5.0);
    style.spacing.interact_size.y = 28.0;
    ctx.set_style(style);
}

/// Build `Visuals` from a token set.
fn visuals_for(t: Tokens, light: bool) -> egui::Visuals {
    let mut v = if light {
        egui::Visuals::light()
    } else {
        egui::Visuals::dark()
    };
    v.panel_fill = t.bg_2;
    v.window_fill = t.panel;
    v.extreme_bg_color = t.bg;
    v.faint_bg_color = t.bg_3;
    v.override_text_color = Some(t.ink);
    v.hyperlink_color = t.accent;
    v.selection.bg_fill = t.accent_soft;
    v.selection.stroke = egui::Stroke::new(1.0, t.accent);

    v.widgets.noninteractive.bg_fill = t.bg_2;
    v.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, t.line);
    v.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, t.ink_dim);

    v.widgets.inactive.bg_fill = t.bg_3;
    v.widgets.inactive.weak_bg_fill = t.bg_3;
    v.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, t.line);
    v.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, t.ink);

    v.widgets.hovered.bg_fill = t.bg_3;
    v.widgets.hovered.weak_bg_fill = t.bg_3;
    v.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, t.line_2);
    v.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, t.ink);

    v.widgets.active.bg_fill = t.accent_soft;
    v.widgets.active.weak_bg_fill = t.accent_soft;
    v.widgets.active.bg_stroke = egui::Stroke::new(1.0, t.accent);
    v.widgets.active.fg_stroke = egui::Stroke::new(1.0, t.ink);

    v.window_stroke = egui::Stroke::new(1.0, t.line);
    v.window_corner_radius = egui::CornerRadius::same(6);
    v
}

/// Full install: fonts + typography/density + dark visuals. Call once at start.
pub fn apply(ctx: &egui::Context) {
    install_fonts(ctx);
    install_style(ctx);
    set_theme(ctx, false);
}

/// Switch dark/light (re-applies visuals only; fonts/style persist).
pub fn set_theme(ctx: &egui::Context, light: bool) {
    tokens::set_active(light);
    ctx.set_visuals(visuals_for(tokens::current(), light));
}
