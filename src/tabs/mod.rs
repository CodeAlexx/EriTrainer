//! Tab dispatch. Every section is populated as a titled form surface; the few
//! that the launcher cannot drive yet (Runs/Logs) render an honest "not yet
//! built" panel rather than a blank that looks functional.

use eframe::egui;

use crate::config::{Section, TrainConfig};
use crate::runtime::Runtime;

mod backup;
mod captioner;
mod cloud;
mod concepts;
mod dataset;
mod general;
mod logs;
mod lora;
mod model;
mod runs;
mod sampling;
mod training;

pub fn render(section: Section, ui: &mut egui::Ui, cfg: &mut TrainConfig, rt: &Runtime) {
    match section {
        Section::General => general::render(ui, cfg),
        Section::Model => model::render(ui, cfg),
        Section::Lora => lora::render(ui, cfg),
        Section::Dataset => dataset::render(ui, cfg),
        Section::Captioner => captioner::render(ui, cfg),
        Section::Validations => concepts::render(ui, cfg),
        Section::Training => training::render(ui, cfg),
        Section::Sampling => sampling::render(ui, cfg),
        Section::Backup => backup::render(ui, cfg),
        Section::Cloud => cloud::render(ui, cfg),
        Section::Runs => runs::render(ui, cfg),
        Section::Logs => logs::render(ui, rt),
    }
}
