//! Sampling tab — Settings / Prompts panels (mirrors the Mojo SamplingTab;
//! the sample media virtual grid is not part of the launcher UI).

use eframe::egui;

use crate::config::{sample_sampler_options, TrainConfig};
use crate::tokens;
use crate::widgets::{
    browse_row, combo_str_row, drag_row, edit_row, field_row, form_panel, toggle_row,
};

pub fn render(ui: &mut egui::Ui, cfg: &mut TrainConfig) {
    let sampler_opts = sample_sampler_options();
    let t = tokens::current();

    // Per-model in-trainer sampling status (honesty: say when the asset fields
    // below actually take effect for the selected model).
    let (supported, note) = match cfg.model_type.as_str() {
        "klein" | "zimage" => (true, "In-trainer sampling: set Sample VAE + Encoder (Qwen3) + Tokenizer below."),
        "chroma" => (true, "In-trainer sampling: set Sample VAE + Encoder (T5) + Tokenizer below."),
        "ernie" => (true, "In-trainer sampling: set Sample VAE + Encoder (text) + Tokenizer below."),
        "l2p" => (true, "In-trainer sampling: set Sample Encoder (Qwen3) + Tokenizer below (pixel-space, no VAE)."),
        "hidream" => (true, "In-trainer sampling: enabled by 'Sample After' alone (assets load from the model dir)."),
        "sd35" => (true, "In-trainer sampling: set CLIP-L (Encoder/Tokenizer) + CLIP-G + T5 below (VAE optional — falls back to checkpoint)."),
        "sdxl" | "anima" => (false, "No in-trainer sampling for this model — it uses a separate sample bin."),
        "ideogram4" => (false, "In-trainer sampling not wired for Ideogram-4 yet (training verified; sample via serenitymojo ideogram4_generate_lora)."),
        _ => (false, "Unknown model."),
    };
    ui.label(
        egui::RichText::new(note)
            .size(11.5)
            .color(if supported { t.ok } else { t.warn }),
    );
    ui.add_space(4.0);

    form_panel(ui, "SAMPLING SETTINGS", "Serenity sampling cadence and sampler", |ui| {
        browse_row(ui, "Sample Dir", &mut cfg.sample_output_dir, true);
        drag_row(ui, "Sample After", &mut cfg.sample_after, 10.0);
        drag_row(ui, "Skip First", &mut cfg.sample_skip_first, 1.0);
        drag_row(ui, "Steps", &mut cfg.sample_steps, 1.0);
        drag_row(ui, "CFG", &mut cfg.sample_cfg, 0.1);
        drag_row(ui, "Size", &mut cfg.sample_size, 64.0);
        combo_str_row(ui, "sm_sampler", "Sampler", &sampler_opts, &mut cfg.sample_sampler);
        toggle_row(ui, "SerenityBoard", &mut cfg.samples_to_tensorboard, "Send samples");
        toggle_row(ui, "Non-EMA", &mut cfg.non_ema_sampling, "Use non-EMA model");
        field_row(ui, "Preset", &cfg.sampler_preset);
    });

    form_panel(ui, "SAMPLE ASSETS", "Paths used to render in-trainer samples", |ui| {
        browse_row(ui, "Sample VAE", &mut cfg.sample_vae_path, false);
        browse_row(ui, "Sample Encoder", &mut cfg.sample_encoder_path, false);
        browse_row(ui, "Sample Tokenizer", &mut cfg.sample_tokenizer_path, false);
        // SD3.5 is a 3-encoder model: the generic Encoder/Tokenizer above are
        // its CLIP-L; these add CLIP-G and T5 (all required to sample).
        if cfg.model_type == "sd35" {
            browse_row(ui, "Sample CLIP-G", &mut cfg.sample_clip_g_path, false);
            browse_row(ui, "CLIP-G Tokenizer", &mut cfg.sample_clip_g_tokenizer_path, false);
            browse_row(ui, "Sample T5", &mut cfg.sample_t5_path, false);
            browse_row(ui, "T5 Tokenizer", &mut cfg.sample_t5_tokenizer_path, false);
        }
    });

    form_panel(ui, "SAMPLE PROMPTS", "Prompts shown in the preview panel", |ui| {
        for (i, sample) in cfg.samples.iter_mut().enumerate() {
            edit_row(ui, &format!("Prompt {}", i + 1), &mut sample.prompt);
            edit_row(ui, &format!("Negative {}", i + 1), &mut sample.negative_prompt);
            drag_row_i32(ui, &format!("Seed {}", i + 1), &mut sample.seed);
        }
    });
}

/// Small i32 drag helper for sample seeds (the shared drag_row is f32).
fn drag_row_i32(ui: &mut egui::Ui, label: &str, value: &mut i32) {
    ui.horizontal(|ui| {
        ui.add_sized([178.0, ui.spacing().interact_size.y], egui::Label::new(label));
        ui.add(egui::DragValue::new(value).speed(1.0));
    });
}
