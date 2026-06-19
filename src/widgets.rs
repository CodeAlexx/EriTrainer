//! Form-row widget helpers (Builder-B). Mirrors the Mojo `form` module:
//! `slider_row`, `drag_row`, `combo_row`, `toggle_row`, `field_row`, `edit_row`,
//! all inside titled `form_panel`s. Each row is a labelled left column + an
//! editable right control, matching the two-column Serenity layout.

use eframe::egui;

const LABEL_W: f32 = 178.0;

/// A titled, framed group panel (≈ Mojo `begin_form_panel`/`end_form_panel`).
pub fn form_panel<R>(
    ui: &mut egui::Ui,
    title: &str,
    subtitle: &str,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    egui::Frame::group(ui.style())
        .show(ui, |ui| {
            ui.label(egui::RichText::new(title).strong());
            if !subtitle.is_empty() {
                ui.label(egui::RichText::new(subtitle).weak().small());
            }
            ui.separator();
            add_contents(ui)
        })
        .inner
}

/// Read-only label:value row (≈ Mojo `field_row`).
pub fn field_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.add_sized([LABEL_W, ui.spacing().interact_size.y], egui::Label::new(label));
        ui.label(egui::RichText::new(value).weak());
    });
}

/// Editable text row (≈ Mojo `edit_row`).
pub fn edit_row(ui: &mut egui::Ui, label: &str, value: &mut String) {
    ui.horizontal(|ui| {
        ui.add_sized([LABEL_W, ui.spacing().interact_size.y], egui::Label::new(label));
        ui.text_edit_singleline(value);
    });
}

/// Bounded slider row (≈ Mojo `slider_row`).
pub fn slider_row(ui: &mut egui::Ui, label: &str, value: &mut f32, min: f32, max: f32) {
    ui.horizontal(|ui| {
        ui.add_sized([LABEL_W, ui.spacing().interact_size.y], egui::Label::new(label));
        ui.add(egui::Slider::new(value, min..=max));
    });
}

/// Unbounded drag row (≈ Mojo `drag_row`).
pub fn drag_row(ui: &mut egui::Ui, label: &str, value: &mut f32, speed: f32) {
    ui.horizontal(|ui| {
        ui.add_sized([LABEL_W, ui.spacing().interact_size.y], egui::Label::new(label));
        ui.add(egui::DragValue::new(value).speed(speed));
    });
}

/// Combo row keyed by index into `options` (≈ Mojo `select_index_row`).
/// Returns true when the selection changed.
pub fn combo_row(
    ui: &mut egui::Ui,
    id: &str,
    label: &str,
    options: &[String],
    selected: &mut usize,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.add_sized([LABEL_W, ui.spacing().interact_size.y], egui::Label::new(label));
        let current = options.get(*selected).cloned().unwrap_or_default();
        egui::ComboBox::from_id_salt(id)
            .selected_text(current)
            .show_ui(ui, |ui| {
                for (i, opt) in options.iter().enumerate() {
                    if ui.selectable_value(selected, i, opt).changed() {
                        changed = true;
                    }
                }
            });
    });
    changed
}

/// Combo row keyed by the selected String value (≈ Mojo `select_string_row`).
/// Returns true when the selection changed.
pub fn combo_str_row(
    ui: &mut egui::Ui,
    id: &str,
    label: &str,
    options: &[String],
    selected: &mut String,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.add_sized([LABEL_W, ui.spacing().interact_size.y], egui::Label::new(label));
        egui::ComboBox::from_id_salt(id)
            .selected_text(selected.clone())
            .show_ui(ui, |ui| {
                for opt in options.iter() {
                    if ui.selectable_value(selected, opt.clone(), opt).changed() {
                        changed = true;
                    }
                }
            });
    });
    changed
}

/// Toggle (checkbox) row (≈ Mojo `toggle_row`). `on_text` is the trailing
/// caption shown next to the checkbox.
pub fn toggle_row(ui: &mut egui::Ui, label: &str, value: &mut bool, on_text: &str) {
    ui.horizontal(|ui| {
        ui.add_sized([LABEL_W, ui.spacing().interact_size.y], egui::Label::new(label));
        ui.checkbox(value, on_text);
    });
}
