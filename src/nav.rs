//! Left navigation column, matching the Mojo `_sidebar`: header (Serenity /
//! Native trainer UI), a CONFIGURE group of the 12 Section rows with the active
//! one accented, then a PROJECT footer with the run name.

use eframe::egui;

use crate::config::Section;
use crate::theme;

pub fn left_nav(ui: &mut egui::Ui, current: &mut Section, run_name: &str) {
    ui.add_space(10.0);
    ui.label(
        egui::RichText::new("Serenity")
            .size(22.0)
            .strong()
            .color(theme::TEXT_STRONG),
    );
    ui.label(
        egui::RichText::new("Native trainer UI")
            .size(12.0)
            .color(theme::TEXT_WEAK),
    );

    ui.add_space(10.0);
    ui.separator();
    ui.add_space(6.0);

    group_header(ui, "CONFIGURE");
    for s in Section::ALL {
        nav_row(ui, current, s);
    }

    ui.add_space(8.0);
    ui.separator();
    ui.add_space(6.0);

    group_header(ui, "PROJECT");
    ui.label(
        egui::RichText::new(if run_name.is_empty() {
            "(unnamed run)"
        } else {
            run_name
        })
        .color(theme::TEXT_STRONG),
    );
}

fn group_header(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .size(11.0)
            .color(theme::TEXT_WEAK),
    );
    ui.add_space(2.0);
}

/// One full-width nav row. The active row gets an accent background + accent
/// text; others use the weak text color and highlight on hover.
fn nav_row(ui: &mut egui::Ui, current: &mut Section, s: Section) {
    let active = *current == s;
    let text_color = if active {
        theme::ACCENT
    } else {
        theme::TEXT_STRONG
    };

    // Full-width clickable row so the accent fill spans the column.
    let desired = egui::vec2(ui.available_width(), 26.0);
    let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::click());

    if active {
        ui.painter()
            .rect_filled(rect, 4.0, theme::ACCENT.linear_multiply(0.18));
        // Accent bar on the left edge of the active row.
        let bar = egui::Rect::from_min_size(rect.min, egui::vec2(3.0, rect.height()));
        ui.painter().rect_filled(bar, 0.0, theme::ACCENT);
    } else if resp.hovered() {
        ui.painter().rect_filled(rect, 4.0, theme::RAISED);
    }

    let text_pos = egui::pos2(rect.left() + 12.0, rect.center().y);
    ui.painter().text(
        text_pos,
        egui::Align2::LEFT_CENTER,
        s.label(),
        egui::FontId::proportional(14.0),
        text_color,
    );

    if resp.clicked() {
        *current = s;
    }
}
