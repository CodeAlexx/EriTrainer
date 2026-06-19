//! EriTrainer — native egui training UI for the EriDiffusion-v2 Rust trainers.
//!
//! Mirrors the Mojo `serenity-trainer` UI: a 3-column shell (left nav / center
//! tabs + top action bar / right live-status rail) that launches the existing
//! EDv2 `train_*` binaries and tails their stdout progress line. It is a
//! launcher UI — it does NOT replace the standalone CLI trainers.

mod config;
mod nav;
mod rail;
mod runtime;
mod shell;
mod sysmetrics;
mod theme;
mod tabs;
mod tokens;
mod topbar;
mod widgets;

use eframe::egui;

use config::{Section, TrainConfig};
use runtime::Runtime;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1480.0, 920.0])
            .with_min_inner_size([1100.0, 720.0])
            .with_title("EriTrainer"),
        ..Default::default()
    };
    eframe::run_native(
        "EriTrainer",
        options,
        Box::new(|cc| {
            theme::apply(&cc.egui_ctx);
            let mut app = EriTrainerApp::default();
            // Screenshot/testing hook: open on a named section.
            if let Ok(name) = std::env::var("ERITRAINER_SECTION") {
                if let Some(s) = Section::from_name(&name) {
                    app.section = s;
                }
            }
            if let Ok(idx) = std::env::var("ERITRAINER_ARCH") {
                if let Ok(i) = idx.parse::<usize>() {
                    app.cfg.architecture_index = i;
                    app.cfg.apply_model_preset(false);
                }
            }
            Ok(Box::new(app))
        }),
    )
}

#[derive(Default)]
struct EriTrainerApp {
    cfg: TrainConfig,
    rt: Runtime,
    section: Section,
}

impl eframe::App for EriTrainerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.rt.tick();

        egui::SidePanel::left("nav")
            .resizable(false)
            .exact_width(shell::NAV_W)
            .show(ctx, |ui| {
                nav::left_nav(ui, &mut self.section, &self.cfg.run_name);
            });

        egui::SidePanel::right("rail")
            .resizable(false)
            .exact_width(shell::RAIL_W)
            .show(ctx, |ui| {
                rail::status_rail(ui, &self.rt);
            });

        egui::TopBottomPanel::top("topbar").show(ctx, |ui| {
            topbar::top_bar(ui, &mut self.cfg, &mut self.rt, self.section);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                tabs::render(self.section, ui, &mut self.cfg, &self.rt);
            });
        });

        // Keep the live rail ticking on a timer even when idle, so the
        // hardware stats (GPU/CPU/RAM) refresh without needing input events.
        // Faster cadence while a run is active. (Mirror of the Mojo UI ticking
        // `trainer_ui_tick_and_apply` unconditionally every frame.)
        let cadence = if self.rt.has_running { 500 } else { 1000 };
        ctx.request_repaint_after(std::time::Duration::from_millis(cadence));
    }
}
