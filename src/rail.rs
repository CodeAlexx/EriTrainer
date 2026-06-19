//! Right live-status rail, matching the Mojo `_status_rail`:
//!   - status pill (IDLE / RUNNING / PAUSED, color-coded)
//!   - progress bar (`rt.progress_fraction()`)
//!   - STATS group (Step, Epoch, Loss, Smooth, Grad, LR, Speed s/step, ETA,
//!     Status, Command, Source)
//!   - ARTIFACTS group (Samples, Checkpoints, Backend)
//!   - HARDWARE group (GPU Model, Driver, GPU util/temp, VRAM, CPU Model,
//!     CPU util, RAM)
//!
//! Empty hardware names render as "unavailable" (honest, not fabricated).

use eframe::egui;

use crate::runtime::Runtime;
use crate::theme;

pub fn status_rail(ui: &mut egui::Ui, rt: &Runtime) {
    ui.add_space(10.0);
    ui.label(
        egui::RichText::new("LIVE STATUS")
            .strong()
            .color(theme::TEXT_STRONG),
    );
    ui.add_space(6.0);

    status_pill(ui, rt);
    ui.add_space(6.0);

    ui.add(
        egui::ProgressBar::new(rt.progress_fraction())
            .show_percentage()
            .fill(theme::ACCENT),
    );
    ui.add_space(8.0);
    ui.separator();

    // --- STATS ---
    group_header(ui, "STATS");
    let l = &rt.live;
    grid(ui, "rail_stats", |ui| {
        stat(ui, "Step", &format!("{} / {}", l.step, l.total_steps));
        stat(ui, "Epoch", &format!("{} / {}", l.epoch, l.total_epochs));
        stat(ui, "Loss", &format!("{:.4}", l.loss));
        stat(ui, "Smooth", &format!("{:.4}", l.smooth_loss));
        stat(ui, "Grad", &format!("{:.4}", l.grad_norm));
        stat(ui, "LR", &format!("{:.2e}", l.learning_rate));
        stat(ui, "Speed", &format!("{:.1} s/step", l.speed_s_step));
        stat(ui, "ETA", &hms(l.eta_secs));
        stat(ui, "Status", text_or(&rt.status_text, "idle"));
        stat(ui, "Command", text_or(&rt.last_command, "—"));
        stat(ui, "Source", text_or(&rt.progress_source, "waiting"));
    });
    ui.add_space(8.0);
    ui.separator();

    // --- ARTIFACTS ---
    group_header(ui, "ARTIFACTS");
    grid(ui, "rail_artifacts", |ui| {
        stat(ui, "Samples", &rt.samples.len().to_string());
        stat(ui, "Checkpoints", &rt.checkpoints.len().to_string());
        stat(ui, "Backend", text_or(&rt.backend_label, "unavailable"));
    });
    ui.add_space(8.0);
    ui.separator();

    // --- HARDWARE ---
    group_header(ui, "HARDWARE");
    grid(ui, "rail_hardware", |ui| {
        stat(ui, "GPU Model", text_or(&rt.gpu_name, "unavailable"));
        stat(ui, "Driver", text_or(&rt.gpu_driver, "unavailable"));
        stat(ui, "GPU", &format!("{:.0}% {}C", l.gpu_util, l.temp_c));
        stat(ui, "VRAM", &format!("{:.1} / {:.1} GB", l.vram_gb, l.vram_total_gb));
        stat(ui, "CPU Model", text_or(&rt.cpu_name, "unavailable"));
        stat(ui, "CPU", &format!("{:.0}%", l.cpu_util));
        stat(ui, "RAM", &format!("{:.1} / {:.1} GB", l.ram_gb, l.ram_total_gb));
    });
}

fn status_pill(ui: &mut egui::Ui, rt: &Runtime) {
    let (text, color) = if rt.has_running {
        if rt.paused {
            ("PAUSED", theme::WARN)
        } else {
            ("RUNNING", theme::ACCENT)
        }
    } else {
        ("IDLE", theme::IDLE)
    };

    let galley = ui.painter().layout_no_wrap(
        text.to_string(),
        egui::FontId::proportional(12.0),
        egui::Color32::BLACK,
    );
    let pad = egui::vec2(12.0, 5.0);
    let size = galley.size() + pad * 2.0;
    let (rect, _resp) = ui.allocate_exact_size(size, egui::Sense::hover());
    ui.painter().rect_filled(rect, 10.0, color);
    ui.painter().galley(rect.min + pad, galley, egui::Color32::BLACK);
}

fn group_header(ui: &mut egui::Ui, text: &str) {
    ui.add_space(6.0);
    ui.label(
        egui::RichText::new(text)
            .size(11.0)
            .color(theme::TEXT_WEAK),
    );
    ui.add_space(2.0);
}

fn grid(ui: &mut egui::Ui, id: &str, body: impl FnOnce(&mut egui::Ui)) {
    egui::Grid::new(id)
        .num_columns(2)
        .striped(true)
        .spacing(egui::vec2(10.0, 4.0))
        .show(ui, body);
}

fn stat(ui: &mut egui::Ui, key: &str, value: &str) {
    ui.label(egui::RichText::new(key).color(theme::TEXT_WEAK));
    ui.label(egui::RichText::new(value).color(theme::TEXT_STRONG));
    ui.end_row();
}

/// `value` if non-empty (after trim), else the fallback.
fn text_or<'a>(value: &'a str, fallback: &'a str) -> &'a str {
    if value.trim().is_empty() {
        fallback
    } else {
        value
    }
}

/// Format seconds as `H:MM:SS`.
fn hms(secs: u64) -> String {
    format!("{}:{:02}:{:02}", secs / 3600, (secs % 3600) / 60, secs % 60)
}
