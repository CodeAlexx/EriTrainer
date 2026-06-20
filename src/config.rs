//! TrainConfig model + the left-nav Section enum.
//!
//! This mirrors the Mojo `TrainerConfigModel.mojo` field set (the OneTrainer
//! organization). Every field that a runner does NOT consume must be surfaced
//! by `ignored_lever_summary` so no widget silently lies (honesty discipline).
//!
//! The capability honesty mirrors the Mojo `trainer_ui_*_lever_*` helpers:
//! only klein/zimage/hidream/ideogram4 runners consume the T1 levers
//! (optimizer/warmup/loss_fn/min_snr_gamma_flow/ema/caption_dropout); the
//! masked-training, full-FT training_method, and non-LORA PEFT widgets reach
//! NO runner and are decorative.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Section {
    General,
    Model,
    Lora,
    Dataset,
    Captioner,
    Validations,
    Training,
    Sampling,
    Backup,
    Cloud,
    Runs,
    Logs,
}

impl Default for Section {
    fn default() -> Self {
        Section::Model // open on Model, matching the Mojo trainer's default
    }
}

impl Section {
    pub const ALL: [Section; 12] = [
        Section::General,
        Section::Model,
        Section::Lora,
        Section::Dataset,
        Section::Captioner,
        Section::Validations,
        Section::Training,
        Section::Sampling,
        Section::Backup,
        Section::Cloud,
        Section::Runs,
        Section::Logs,
    ];

    /// Resolve a section by name (for the `ERITRAINER_SECTION` screenshot hook).
    pub fn from_name(s: &str) -> Option<Section> {
        Some(match s.trim().to_lowercase().as_str() {
            "general" => Section::General,
            "model" => Section::Model,
            "lora" | "lora/oft" | "oft" => Section::Lora,
            "dataset" => Section::Dataset,
            "captioner" => Section::Captioner,
            "validations" | "concepts" => Section::Validations,
            "training" => Section::Training,
            "sampling" => Section::Sampling,
            "backup" => Section::Backup,
            "cloud" => Section::Cloud,
            "runs" => Section::Runs,
            "logs" => Section::Logs,
            _ => return None,
        })
    }

    pub fn label(&self) -> &'static str {
        match self {
            Section::General => "General",
            Section::Model => "Model",
            Section::Lora => "LoRA / OFT",
            Section::Dataset => "Dataset",
            Section::Captioner => "Captioner",
            Section::Validations => "Validations",
            Section::Training => "Training",
            Section::Sampling => "Sampling",
            Section::Backup => "Backup",
            Section::Cloud => "Cloud",
            Section::Runs => "Runs",
            Section::Logs => "Logs",
        }
    }
}

// ── Option lists (mirror TrainerConfigModel.mojo) ───────────────────────────
// Helper to build a Vec<String> from string literals.
fn opts(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

pub fn training_method_options() -> Vec<String> {
    opts(&["LoRA", "Fine Tune", "Embedding"])
}
pub fn model_type_options() -> Vec<String> {
    opts(&[
        "IDEOGRAM_4",
        "FLUX_2",
        "STABLE_DIFFUSION_XL_10_BASE",
        "STABLE_DIFFUSION_35",
        "CHROMA_1",
        "ERNIE_IMAGE",
        "ANIMA",
        "Z_IMAGE",
        "Z_IMAGE_L2P",
        "LTX_2_VIDEO",
        "WAN_22_VIDEO",
        "HIDREAM_O1",
        "FLUX_1_DEV",
        "QWEN_IMAGE",
        "ACE_STEP",
        "SLIDER_KLEIN",
        "ASYMFLOW",
        "SENSENOVA_U1",
    ])
}
pub fn architecture_options() -> Vec<String> {
    opts(&[
        "Ideogram4 FP8",
        "Klein 9B",
        "Flux2 Dev",
        "SDXL 1.0",
        "Chroma1 HD",
        "Ernie Image",
        "Anima",
        "Z-Image",
        "Z-Image L2P",
        "LTX-2 AV",
        "Wan2.2 T2V 14B",
        "HiDream O1",
        "Flux.1 Dev",
        "Qwen-Image",
        "ACE-Step",
        "Slider (Klein)",
        "AsymFlow",
        "SenseNova U1",
    ])
}
pub fn optimizer_options() -> Vec<String> {
    opts(&[
        "ADAMW8BIT",
        "ADAMW",
        "CAME",
        "ADAFACTOR",
        "MUON",
        "SCHEDULE_FREE_ADAMW",
    ])
}
pub fn scheduler_options() -> Vec<String> {
    opts(&["COSINE", "CONSTANT", "LINEAR"])
}
pub fn precision_options() -> Vec<String> {
    opts(&["BFLOAT_16", "FLOAT_16", "FLOAT_32"])
}
pub fn cloud_type_options() -> Vec<String> {
    opts(&["NONE", "RUNPOD", "LINUX"])
}
pub fn output_format_options() -> Vec<String> {
    opts(&["SAFETENSORS", "CKPT", "INTERNAL", "DIFFUSERS"])
}
pub fn device_options() -> Vec<String> {
    opts(&["cuda", "cpu"])
}
pub fn lr_scaler_options() -> Vec<String> {
    opts(&["NONE", "LINEAR", "SQRT", "COSINE"])
}
pub fn ema_options() -> Vec<String> {
    opts(&["OFF", "EMA"])
}
pub fn timestep_distribution_options() -> Vec<String> {
    opts(&["UNIFORM", "LOGIT_NORMAL", "MODE", "COSINE", "SIGMOID"])
}
pub fn loss_weight_options() -> Vec<String> {
    opts(&["MIN_SNR_GAMMA", "NONE", "P2", "DEBIASED_ESTIMATION"])
}
pub fn loss_scaler_options() -> Vec<String> {
    opts(&["NONE", "MIN_SNR_GAMMA", "P2"])
}
pub fn loss_fn_options() -> Vec<String> {
    opts(&["mse", "huber", "smooth_l1"])
}
pub fn layer_filter_preset_options() -> Vec<String> {
    opts(&["full", "attention", "mlp", "double_blocks", "single_blocks"])
}
pub fn peft_options() -> Vec<String> {
    opts(&["LORA", "LOKR", "LOHA", "OFT"])
}
pub fn sample_sampler_options() -> Vec<String> {
    opts(&[
        "Ideogram4 FlowMatch",
        "FlowMatch Euler",
        "Euler",
        "DDIM",
        "DPM++ 2M",
        "UniPC",
    ])
}
pub fn resolution_options() -> Vec<String> {
    opts(&["512", "768", "1024", "1280"])
}
pub fn captioner_model_options() -> Vec<String> {
    opts(&[
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3-VL-4B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Custom...",
    ])
}
pub fn captioner_quant_options() -> Vec<String> {
    opts(&["None", "8-bit", "4-bit"])
}
pub fn captioner_attention_options() -> Vec<String> {
    opts(&["flash_attention_2", "eager"])
}
pub fn captioner_resolution_options() -> Vec<String> {
    opts(&["auto", "auto_high", "fast", "high"])
}

/// A dataset concept (mirrors `TrainerUIConcept`).
#[derive(Clone, Serialize, Deserialize)]
pub struct Concept {
    pub name: String,
    pub path: String,
    pub trigger: String,
    pub image_count: i32,
    pub repeats: i32,
    pub concept_type: String,
    pub enabled: bool,
}

impl Default for Concept {
    fn default() -> Self {
        Self {
            name: String::new(),
            path: String::new(),
            trigger: String::new(),
            image_count: 0,
            repeats: 1,
            concept_type: String::from("STANDARD"),
            enabled: true,
        }
    }
}

/// A sample prompt (mirrors `TrainerUISample`).
#[derive(Clone, Serialize, Deserialize)]
pub struct Sample {
    pub prompt: String,
    pub negative_prompt: String,
    pub seed: i32,
}

impl Default for Sample {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            seed: 42,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    // --- Identity / backend ---
    pub run_name: String,
    /// Selects which EDv2 `train_<model_type>` binary to launch.
    pub model_type: String,
    pub backend_target: String,

    // --- Selectors (index into the option-list helpers above) ---
    pub training_method_index: usize,
    pub model_type_index: usize,
    pub architecture_index: usize,
    pub optimizer_index: usize,
    pub scheduler_index: usize,
    pub cloud_type_index: usize,
    pub captioner_model_index: usize,
    pub captioner_quant_index: usize,
    pub captioner_attention_index: usize,
    pub captioner_resolution_index: usize,

    // --- General: workspace / debug & validation ---
    pub workspace_dir: String,
    pub cache_dir: String,
    pub debug_mode: bool,
    pub debug_dir: String,
    pub tensorboard: bool,
    pub tensorboard_always_on: bool,
    pub tensorboard_port: String,
    pub validation: bool,
    pub continue_last_backup: bool,
    pub prevent_overwrites: bool,
    pub only_cache: bool,
    pub dataloader_threads: f32,

    // --- General: device / multi-gpu ---
    pub train_device: String,
    pub temp_device: String,
    pub multi_gpu: bool,
    pub device_indexes: String,
    pub fused_gradient_reduce: bool,
    pub async_gradient_reduce: bool,

    // --- Model: base model + output ---
    pub base_model_path: String,
    /// Path to the EDv2 trainer `--config` JSON (e.g. configs/klein9b_alina.json).
    /// Required to launch; the runner reads dataset/recipe from it.
    pub run_config_path: String,
    /// Secondary checkpoint for two-model trainers (asymflow `--asymflow-adapter`,
    /// wan22 `--high-noise`). Empty for single-checkpoint models.
    pub aux_model_path: String,
    pub vae_override: String,
    pub output_dir: String,
    pub output_model_format: String,
    pub output_dtype: String,
    pub model_arch: String,
    pub bundle_additional_embeddings: bool,

    // --- Dataset ---
    pub dataset_path: String,
    pub concept_file_name: String,
    pub concepts: Vec<Concept>,
    pub aspect_ratio_bucketing: bool,
    pub latent_caching: bool,
    pub cache_text_embeddings: bool,
    pub clear_cache_before_training: bool,
    pub resolution: String,
    pub caption_extension: String,
    pub caption_dropout: f32,

    // --- Training: base schedule ---
    pub epochs: f32,
    pub batch_size: f32,
    pub gradient_accumulation_steps: f32,
    pub learning_rate_warmup_steps: f32,
    pub learning_rate_min_factor: f32,
    pub learning_rate_cycles: f32,
    pub learning_rate_scaler: String,
    pub seed: f32,
    pub max_train_steps: f32,

    // --- Training: optimizer / LRs ---
    pub learning_rate: f32,
    pub text_encoder_learning_rate: f32,
    pub transformer_learning_rate: f32,
    pub weight_decay: f32,
    pub clip_grad_norm: f32,

    // --- Training: precision & memory ---
    pub train_dtype: String,
    pub fallback_train_dtype: String,
    pub gradient_checkpointing: bool,
    pub activation_offloading: bool,
    pub layer_offload_fraction: f32,
    pub enable_autocast_cache: bool,
    pub frames: String,
    pub force_circular_padding: bool,

    // --- Training: EMA & targets ---
    pub ema_mode: String,
    pub ema_decay: f32,
    pub ema_update_step_interval: f32,
    pub train_transformer: bool,
    pub train_text_encoder: bool,
    pub text_encoder_stop_after: f32,
    pub transformer_stop_after: f32,

    // --- Training: text encoder / transformer panels ---
    pub text_encoder_sequence_length: String,
    pub transformer_attention_mask: bool,
    pub transformer_guidance_scale: f32,

    // --- Training: noise & timesteps ---
    pub offset_noise_weight: f32,
    pub perturbation_noise_weight: f32,
    pub timestep_distribution: String,
    pub min_noising_strength: f32,
    pub max_noising_strength: f32,
    pub noising_weight: f32,
    pub noising_bias: f32,
    pub timestep_shift: f32,
    pub dynamic_timestep_shifting: bool,

    // --- Training: masked training [not wired] ---
    pub masked_training: bool,
    pub unmasked_probability: f32,
    pub unmasked_weight: f32,
    pub normalize_masked_area_loss: bool,
    pub masked_prior_preservation_weight: f32,
    pub custom_conditioning_image: bool,

    // --- Training: loss ---
    pub loss_fn: String,
    pub mse_strength: f32,
    pub mae_strength: f32,
    pub log_cosh_strength: f32,
    pub huber_strength: f32,
    pub huber_delta: f32,
    pub smooth_l1_beta: f32,
    pub vb_loss_strength: f32,
    pub loss_weight_fn: String,
    pub loss_weight_strength: f32,
    pub min_snr_gamma_flow: f32,
    pub loss_scaler: String,
    pub quantized_resident: String,

    // --- Training: layer filter ---
    pub layer_filter_preset: String,
    pub layer_filter: String,
    pub layer_filter_regex: bool,
    pub peft_type: String,

    // --- LoRA / OFT ---
    pub lora_model_name: String,
    pub lora_rank: f32,
    pub lora_alpha: f32,
    pub lora_dropout: f32,
    pub lora_weight_dtype: String,
    pub oft_block_size: f32,
    pub oft_coft: bool,

    // --- Sampling ---
    pub sample_output_dir: String,
    pub samples: Vec<Sample>,
    pub sample_after: f32,
    pub sample_skip_first: f32,
    pub sample_cfg: f32,
    pub sample_steps: f32,
    pub sample_sampler: String,
    pub sampler_preset: String,
    pub samples_to_tensorboard: bool,
    pub non_ema_sampling: bool,
    // In-trainer sample-asset paths (consumed by the *_args sample wiring).
    pub sample_size: f32,
    pub sample_vae_path: String,
    pub sample_encoder_path: String,
    pub sample_tokenizer_path: String,
    // SD3.5 extra sample encoders (it needs CLIP-L + CLIP-G + T5): the generic
    // encoder/tokenizer above map to CLIP-L; these add CLIP-G and T5.
    pub sample_clip_g_path: String,
    pub sample_clip_g_tokenizer_path: String,
    pub sample_t5_path: String,
    pub sample_t5_tokenizer_path: String,

    // --- Backup ---
    pub backup_after: f32,
    pub rolling_backup: bool,
    pub rolling_backup_count: f32,
    pub backup_before_save: bool,
    pub save_every: f32,
    pub save_skip_first: f32,
    pub save_max_keep: f32,
    pub save_filename_prefix: String,

    // --- Captioner ---
    pub captioner_custom_model_id: String,
    pub captioner_folder_path: String,
    pub captioner_prompt: String,
    pub captioner_skip_existing: bool,
    pub captioner_summary_mode: bool,
    pub captioner_one_sentence_mode: bool,
    pub captioner_retain_preview: bool,
    pub captioner_max_tokens: f32,

    // --- Cloud ---
    pub cloud_host: String,
    pub cloud_port: String,
    pub cloud_user: String,
    pub cloud_workspace_dir: String,
    pub cloud_delete_workspace: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            // Identity / backend — Klein is the M1 vertical.
            run_name: String::from("boxjana_klein9b_lora_v1"),
            model_type: String::from("klein"),
            backend_target: String::from("klein"),

            training_method_index: 0,
            model_type_index: 1, // FLUX_2
            architecture_index: 1, // Klein 9B
            optimizer_index: 1, // ADAMW
            scheduler_index: 0, // COSINE
            cloud_type_index: 0, // NONE
            captioner_model_index: 3, // Qwen3-VL-8B-Instruct
            captioner_quant_index: 1, // 8-bit
            captioner_attention_index: 0,
            captioner_resolution_index: 0,

            workspace_dir: String::from("/home/alex/trainings/boxjana_klein9b_lora_v1"),
            cache_dir: String::new(),
            debug_mode: false,
            debug_dir: String::from("debug"),
            tensorboard: true,
            tensorboard_always_on: false,
            tensorboard_port: String::from("6006"),
            validation: false,
            continue_last_backup: false,
            prevent_overwrites: true,
            only_cache: false,
            dataloader_threads: 8.0,

            train_device: String::from("cuda"),
            temp_device: String::from("cpu"),
            multi_gpu: false,
            device_indexes: String::from("0"),
            fused_gradient_reduce: false,
            async_gradient_reduce: false,

            base_model_path: String::new(),
            run_config_path: String::new(),
            aux_model_path: String::new(),
            vae_override: String::new(),
            output_dir: String::from("/home/alex/mojodiffusion/output"),
            output_model_format: String::from("SAFETENSORS"),
            output_dtype: String::from("FLOAT_16"),
            model_arch: String::from("klein9b"),
            bundle_additional_embeddings: true,

            dataset_path: String::new(),
            concept_file_name: String::from("concepts.json"),
            concepts: vec![Concept {
                name: String::from("boxjana"),
                path: String::new(),
                trigger: String::from("box1jana"),
                image_count: 22,
                repeats: 1,
                concept_type: String::from("STANDARD"),
                enabled: true,
            }],
            aspect_ratio_bucketing: true,
            latent_caching: true,
            cache_text_embeddings: true,
            clear_cache_before_training: false,
            resolution: String::from("512"),
            caption_extension: String::from("txt"),
            // Default OFF (0.0) keeps default runs byte-identical to baselines.
            caption_dropout: 0.0,

            epochs: 1.0,
            batch_size: 1.0,
            gradient_accumulation_steps: 1.0,
            learning_rate_warmup_steps: 0.0,
            learning_rate_min_factor: 0.0,
            learning_rate_cycles: 1.0,
            learning_rate_scaler: String::from("NONE"),
            seed: 42.0,
            max_train_steps: 3000.0,

            learning_rate: 4e-4,
            text_encoder_learning_rate: 1e-5,
            transformer_learning_rate: 4e-4,
            weight_decay: 0.01,
            clip_grad_norm: 1.0,

            train_dtype: String::from("BFLOAT_16"),
            fallback_train_dtype: String::from("BFLOAT_16"),
            gradient_checkpointing: true,
            activation_offloading: false,
            layer_offload_fraction: 0.0,
            enable_autocast_cache: true,
            frames: String::from("25"),
            force_circular_padding: false,

            ema_mode: String::from("OFF"),
            ema_decay: 0.999,
            ema_update_step_interval: 5.0,
            train_transformer: true,
            train_text_encoder: false,
            text_encoder_stop_after: 30.0,
            transformer_stop_after: 0.0,

            text_encoder_sequence_length: String::from("512"),
            transformer_attention_mask: false,
            transformer_guidance_scale: 1.0,

            offset_noise_weight: 0.0,
            perturbation_noise_weight: 0.0,
            timestep_distribution: String::from("UNIFORM"),
            min_noising_strength: 0.0,
            max_noising_strength: 1.0,
            noising_weight: 1.0,
            noising_bias: 0.0,
            timestep_shift: 1.0,
            dynamic_timestep_shifting: false,

            masked_training: false,
            unmasked_probability: 0.0,
            unmasked_weight: 0.0,
            normalize_masked_area_loss: false,
            masked_prior_preservation_weight: 1.0,
            custom_conditioning_image: false,

            loss_fn: String::from("mse"), // T1.A default-off
            mse_strength: 1.0,
            mae_strength: 0.0,
            log_cosh_strength: 0.0,
            huber_strength: 0.0,
            huber_delta: 1.0,
            smooth_l1_beta: 1.0,
            vb_loss_strength: 0.0,
            loss_weight_fn: String::from("MIN_SNR_GAMMA"),
            loss_weight_strength: 5.0,
            min_snr_gamma_flow: 0.0, // 0.0 = off
            loss_scaler: String::from("NONE"),
            quantized_resident: String::from("OFF"), // T2.B default-off

            layer_filter_preset: String::from("full"),
            layer_filter: String::new(),
            layer_filter_regex: false,
            peft_type: String::from("LORA"),

            lora_model_name: String::from("boxjana_klein9b_lora_v1"),
            lora_rank: 16.0,
            lora_alpha: 16.0,
            lora_dropout: 0.0,
            lora_weight_dtype: String::from("FLOAT_32"),
            oft_block_size: 4.0,
            oft_coft: false,

            sample_output_dir: String::from("/home/alex/mojodiffusion/output"),
            samples: vec![
                Sample {
                    prompt: String::from(
                        "box1jana, 512x512 portrait photo, confident smile, dark hair, sleek modern styling, simple studio background, natural skin detail, sharp focus.",
                    ),
                    negative_prompt: String::new(),
                    seed: 42,
                },
                Sample {
                    prompt: String::from(
                        "box1jana, 512x512 seated portrait on an ornate chair, dark turtleneck, playful expression, soft studio light, clean white background.",
                    ),
                    negative_prompt: String::new(),
                    seed: 43,
                },
            ],
            sample_after: 500.0,
            sample_skip_first: 0.0,
            sample_cfg: 7.0,
            sample_steps: 20.0,
            sample_sampler: String::from("FlowMatch Euler"),
            sampler_preset: String::from("KLEIN_20"),
            samples_to_tensorboard: true,
            non_ema_sampling: false,
            sample_size: 512.0,
            sample_vae_path: String::new(),
            sample_encoder_path: String::new(),
            sample_tokenizer_path: String::new(),
            sample_clip_g_path: String::new(),
            sample_clip_g_tokenizer_path: String::new(),
            sample_t5_path: String::new(),
            sample_t5_tokenizer_path: String::new(),

            backup_after: 500.0,
            rolling_backup: true,
            rolling_backup_count: 5.0,
            backup_before_save: true,
            save_every: 500.0,
            save_skip_first: 0.0,
            save_max_keep: 4.0,
            save_filename_prefix: String::from("boxjana_klein9b_lora_v1"),

            captioner_custom_model_id: String::new(),
            captioner_folder_path: String::new(),
            captioner_prompt: String::from("Describe this media."),
            captioner_skip_existing: true,
            captioner_summary_mode: false,
            captioner_one_sentence_mode: false,
            captioner_retain_preview: true,
            captioner_max_tokens: 128.0,

            cloud_host: String::new(),
            cloud_port: String::from("22"),
            cloud_user: String::from("root"),
            cloud_workspace_dir: String::from("/workspace/serenity"),
            cloud_delete_workspace: false,
        }
    }
}

// ── Model checkpoint / cache constants (mirror TrainerConfigModel.mojo
// SERENITY_* comptime aliases verbatim, audit values 2026-06-11) ────────────
const SERENITY_KLEIN9B_CHECKPOINT: &str =
    "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors";
const SERENITY_BOXJANA_KLEIN_CACHE: &str =
    "/home/alex/EriDiffusion/EriDiffusion-v2/cache/alina_klein9b";
const SERENITY_IDEOGRAM4_BASE: &str = "/home/alex/.serenity/models/ideogram-4-fp8";
const SERENITY_IDEOGRAM4_CACHE: &str =
    "/home/alex/trainings/ideogram4_giger_cache/cache.safetensors";
const SERENITY_SDXL_CHECKPOINT: &str =
    "/home/alex/.serenity/models/checkpoints/sdxl_unet_bf16.safetensors";
const SERENITY_SDXL_CACHE: &str =
    "/home/alex/EriDiffusion/EriDiffusion-v2/cache/eri2_sdxl_512_smoke";
const SERENITY_CHROMA_CHECKPOINT: &str =
    "/home/alex/.serenity/models/checkpoints/chroma1_hd_bf16.safetensors";
const SERENITY_CHROMA_CACHE: &str = "/home/alex/datasets/boxjana_chroma_edv2_512";
const SERENITY_ERNIE_CHECKPOINT: &str = "/home/alex/models/ERNIE-Image/transformer";
const SERENITY_ERNIE_CACHE: &str =
    "/home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_ernie_512_FIXED";
const SERENITY_ANIMA_CHECKPOINT: &str =
    "/home/alex/.serenity/models/anima/split_files/diffusion_models/anima-base-v1.0.safetensors";
const SERENITY_ANIMA_CACHE: &str =
    "/home/alex/EriDiffusion/EriDiffusion-v2/cache/anima_synth_smoke";
// train_zimage --model wants a single safetensors file (mmap), NOT a sharded
// dir — verified 2026-06-19 (the diffusers `zimage_base/transformer` dir errors
// "Is a directory"). Point at the single-file base checkpoint.
const SERENITY_ZIMAGE_CHECKPOINT: &str =
    "/home/alex/.serenity/models/checkpoints/z_image_base_bf16.safetensors";
const SERENITY_ZIMAGE_CACHE: &str = "/home/alex/mojodiffusion/output/alina_zimage_cache";
const SERENITY_L2P_CHECKPOINT: &str =
    "/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors";
const SERENITY_L2P_CACHE: &str = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_l2p_512";
const SERENITY_HIDREAM_CHECKPOINT: &str = "/home/alex/HiDream-O1-Image-Dev-weights";
const SERENITY_HIDREAM_STAGE: &str = "/home/alex/trainings/ideogram4_giger_stage";
// SD3.5: default to the medium checkpoint (5GB, fits 24GB without offload —
// train_sd35 has no --offload). sd3.5_large.safetensors (16GB) is also present.
const SERENITY_SD35_CHECKPOINT: &str =
    "/home/alex/.serenity/models/checkpoints/sd3.5_medium.safetensors";
const SERENITY_SD35_CACHE: &str =
    "/home/alex/EriDiffusion/EriDiffusion-v2/cache/eri2_sd35_512_smoke";
const SERENITY_WAN22_CHECKPOINT: &str =
    "/home/alex/.serenity/models/checkpoints/wan2.2_t2v_low_noise_14b_fp16.safetensors";

/// model_type option index -> canonical architecture option index (mirrors
/// the Mojo `_arch_index_for_model_type`). Returns `None` for
/// STABLE_DIFFUSION_35 (index 3): no trainable runner yet.
fn arch_index_for_model_type(model_type_index: usize) -> Option<usize> {
    match model_type_index {
        0 => Some(0),  // IDEOGRAM_4
        1 => Some(1),  // FLUX_2 -> Klein 9B is the trainable FLUX_2 default
        2 => Some(3),  // STABLE_DIFFUSION_XL_10_BASE
        4 => Some(4),  // CHROMA_1
        5 => Some(5),  // ERNIE_IMAGE
        6 => Some(6),  // ANIMA
        7 => Some(7),  // Z_IMAGE
        8 => Some(8),  // Z_IMAGE_L2P
        9 => Some(9),  // LTX_2_VIDEO
        10 => Some(10), // WAN_22_VIDEO
        11 => Some(11), // HIDREAM_O1
        12 => Some(12), // FLUX_1_DEV
        13 => Some(13), // QWEN_IMAGE
        14 => Some(14), // ACE_STEP
        15 => Some(15), // SLIDER_KLEIN
        16 => Some(16), // ASYMFLOW
        17 => Some(17), // SENSENOVA_U1
        _ => None,     // 3 = STABLE_DIFFUSION_35
    }
}

/// architecture option index -> model_type option index (mirrors the Mojo
/// `_model_type_for_arch_index`).
fn model_type_for_arch_index(architecture_index: usize) -> Option<usize> {
    match architecture_index {
        0 => Some(0),     // Ideogram4 FP8
        1 | 2 => Some(1), // Klein 9B / Flux2 Dev
        3 => Some(2),     // SDXL 1.0
        4 => Some(4),     // Chroma1 HD
        5 => Some(5),     // Ernie Image
        6 => Some(6),     // Anima
        7 => Some(7),     // Z-Image
        8 => Some(8),     // Z-Image L2P
        9 => Some(9),     // LTX-2 AV
        10 => Some(10),   // Wan2.2 T2V 14B
        11 => Some(11),   // HiDream O1
        12 => Some(12),   // Flux.1 Dev
        13 => Some(13),   // Qwen-Image
        14 => Some(14),   // ACE-Step
        15 => Some(15),   // Slider (Klein)
        16 => Some(16),   // AsymFlow
        17 => Some(17),   // SenseNova U1
        _ => None,
    }
}

impl TrainConfig {
    /// Rewrite the per-model launch identity + recipe defaults from the
    /// selected model/architecture (port of `trainer_ui_apply_model_preset`).
    ///
    /// `prefer_model_type = true` means the Model Type combo changed; resolve
    /// the canonical architecture from it (and rewrite architecture_index to
    /// match). `false` means the Architecture combo changed; resolve the
    /// model_type from it.
    ///
    /// NOTE: in the Mojo the launch-binary name is `backend_target`; here the
    /// launcher (runtime.rs `train_{model_type}`) and the "Live target" label
    /// both read `model_type`, so this sets BOTH to that same binary name.
    pub fn apply_model_preset(&mut self, prefer_model_type: bool) {
        // Resolve the canonical architecture from whichever selector changed.
        let arch: usize;
        if prefer_model_type {
            match arch_index_for_model_type(self.model_type_index) {
                Some(a) => arch = a,
                None => {
                    // STABLE_DIFFUSION_35 (model-type-only; no Architecture-combo
                    // entry). EDv2 has train_sd35, so apply the full SD3.5 recipe.
                    self.backend_target = String::from("sd35");
                    self.model_type = String::from("sd35");
                    self.model_type_index = 3;
                    self.base_model_path = String::from(SERENITY_SD35_CHECKPOINT);
                    self.cache_dir = String::from(SERENITY_SD35_CACHE);
                    self.model_arch = String::from("sd35");
                    self.sample_sampler = String::from("FlowMatch Euler");
                    self.sampler_preset = String::from("SD35_20");
                    self.learning_rate = 0.0001;
                    self.lora_rank = 16.0;
                    self.lora_alpha = 16.0;
                    self.timestep_shift = 3.0;
                    return;
                }
            }
        } else {
            arch = self.architecture_index;
            if let Some(mt) = model_type_for_arch_index(self.architecture_index) {
                self.model_type_index = mt;
            }
        }

        match arch {
            1 | 2 => {
                self.backend_target = String::from("klein");
                self.model_type = String::from("klein");
                self.model_type_index = 1;
                self.architecture_index = 1;
                self.base_model_path = String::from(SERENITY_KLEIN9B_CHECKPOINT);
                self.cache_dir = String::from(SERENITY_BOXJANA_KLEIN_CACHE);
                self.model_arch = String::from("klein9b");
                self.sample_sampler = String::from("FlowMatch Euler");
                self.sampler_preset = String::from("KLEIN_20");
            }
            0 => {
                self.backend_target = String::from("ideogram4");
                self.model_type = String::from("ideogram4");
                self.model_type_index = 0;
                self.architecture_index = 0;
                self.base_model_path = String::from(SERENITY_IDEOGRAM4_BASE);
                self.cache_dir = String::from(SERENITY_IDEOGRAM4_CACHE);
                self.model_arch = String::from("ideogram4");
                self.sample_sampler = String::from("Ideogram4 FlowMatch");
                self.sampler_preset = String::from("V4_DEFAULT_20");
            }
            3 => {
                // SDXL — serenitymojo train_sdxl_real (eps-pred conv-UNet LoRA).
                self.backend_target = String::from("sdxl");
                self.model_type = String::from("sdxl");
                self.model_type_index = 2;
                self.architecture_index = 3;
                self.base_model_path = String::from(SERENITY_SDXL_CHECKPOINT);
                self.cache_dir = String::from(SERENITY_SDXL_CACHE);
                self.model_arch = String::from("sdxl10");
                self.sample_sampler = String::from("Euler");
                self.sampler_preset = String::from("SDXL_20");
                self.learning_rate = 0.0001;
                // train_sdxl_real is compiled for rank 16 (fails loud otherwise).
                self.lora_rank = 16.0;
                self.lora_alpha = 16.0;
                self.timestep_shift = 1.0;
            }
            4 => {
                // Chroma1-HD — serenitymojo train_chroma_real (flow-match).
                self.backend_target = String::from("chroma");
                self.model_type = String::from("chroma");
                self.model_type_index = 4;
                self.architecture_index = 4;
                self.base_model_path = String::from(SERENITY_CHROMA_CHECKPOINT);
                self.cache_dir = String::from(SERENITY_CHROMA_CACHE);
                self.model_arch = String::from("chroma1hd");
                self.sample_sampler = String::from("FlowMatch Euler");
                self.sampler_preset = String::from("CHROMA_20");
                self.learning_rate = 0.0001;
                // train_chroma_real is compiled for rank 16 (fails loud otherwise).
                self.lora_rank = 16.0;
                self.lora_alpha = 16.0;
                self.timestep_shift = 1.15;
            }
            5 => {
                // Ernie Image — serenitymojo train_ernie_real.
                // NOTE canonical lora_alpha is 1.0, not rank.
                self.backend_target = String::from("ernie");
                self.model_type = String::from("ernie");
                self.model_type_index = 5;
                self.architecture_index = 5;
                self.base_model_path = String::from(SERENITY_ERNIE_CHECKPOINT);
                self.cache_dir = String::from(SERENITY_ERNIE_CACHE);
                self.model_arch = String::from("ernie_image");
                self.sample_sampler = String::from("FlowMatch Euler");
                self.sampler_preset = String::from("ERNIE_20");
                self.learning_rate = 0.0003;
                self.lora_alpha = 1.0;
                self.timestep_shift = 1.0;
            }
            6 => {
                // Anima — serenitymojo train_anima_real.
                self.backend_target = String::from("anima");
                self.model_type = String::from("anima");
                self.model_type_index = 6;
                self.architecture_index = 6;
                self.base_model_path = String::from(SERENITY_ANIMA_CHECKPOINT);
                self.cache_dir = String::from(SERENITY_ANIMA_CACHE);
                self.model_arch = String::from("anima");
                self.sample_sampler = String::from("FlowMatch Euler");
                self.sampler_preset = String::from("ANIMA_20");
                self.learning_rate = 0.0001;
                self.lora_alpha = 16.0;
                self.timestep_shift = 1.0;
            }
            7 => {
                // Z-Image — serenitymojo train_zimage_real (compiled for
                // rank=16, alpha=1.0, lr=3e-4; fails loud on any other).
                self.backend_target = String::from("zimage");
                self.model_type = String::from("zimage");
                self.model_type_index = 7;
                self.architecture_index = 7;
                self.base_model_path = String::from(SERENITY_ZIMAGE_CHECKPOINT);
                self.cache_dir = String::from(SERENITY_ZIMAGE_CACHE);
                self.model_arch = String::from("zimage");
                self.sample_sampler = String::from("FlowMatch Euler");
                self.sampler_preset = String::from("ZIMAGE_20");
                self.learning_rate = 0.0003;
                self.lora_rank = 16.0;
                self.lora_alpha = 1.0;
                self.timestep_shift = 1.0;
            }
            8 => {
                // Z-Image L2P (pixel-space, VAE-less) — serenitymojo
                // train_l2p_real (compiled for rank=16, alpha=16, lr=3e-4,
                // shift=3.0; fails loud on any other). No prepared pixel cache
                // exists yet — trainer preflight fails loud until built.
                self.backend_target = String::from("l2p");
                self.model_type = String::from("l2p");
                self.model_type_index = 8;
                self.architecture_index = 8;
                self.base_model_path = String::from(SERENITY_L2P_CHECKPOINT);
                self.cache_dir = String::from(SERENITY_L2P_CACHE);
                self.model_arch = String::from("zimage_l2p");
                self.sample_sampler = String::from("FlowMatch Euler");
                self.sampler_preset = String::from("L2P_20");
                self.learning_rate = 0.0003;
                self.lora_rank = 16.0;
                self.lora_alpha = 16.0;
                self.timestep_shift = 3.0;
                // ~19.5GB pixel-space checkpoint: grad checkpointing on by
                // default so the default launch fits a 24GB card.
                self.gradient_checkpointing = true;
            }
            9 => {
                // LTX-2 AV — NO production trainer yet. Routed to an unwired
                // backend so launch fails loudly instead of silently running a
                // legacy/unfaithful path.
                self.backend_target = String::from("ltx2");
                self.model_type = String::from("ltx2");
                self.model_type_index = 9;
                self.architecture_index = 9;
                self.base_model_path = String::new();
                self.model_arch = String::from("ltx2_av");
                self.frames = String::from("25");
            }
            10 => {
                // Wan2.2-T2V 14B — train_wan22_real exists but its RoPE tables
                // are placeholders and it is config-less smoke-mode. Unwired
                // backend: fail-loud until the trainer is made faithful.
                self.backend_target = String::from("wan22");
                self.model_type = String::from("wan22");
                self.model_type_index = 10;
                self.architecture_index = 10;
                self.base_model_path = String::from(SERENITY_WAN22_CHECKPOINT);
                self.model_arch = String::from("wan22_t2v_14b");
                self.frames = String::from("1");
            }
            11 => {
                // HiDream-O1 — serenitymojo train_hidream_o1_real. Weights dir
                // is comptime in the trainer; cache_dir is the stage-A dir.
                // Recipe: lr 1e-4, rank 32, alpha=rank.
                self.backend_target = String::from("hidream");
                self.model_type = String::from("hidream");
                self.model_type_index = 11;
                self.architecture_index = 11;
                self.base_model_path = String::from(SERENITY_HIDREAM_CHECKPOINT);
                self.cache_dir = String::from(SERENITY_HIDREAM_STAGE);
                self.model_arch = String::from("hidream_o1");
                self.sample_sampler = String::from("FlowMatch Euler");
                self.sampler_preset = String::from("HIDREAM_20");
                self.learning_rate = 0.0001;
                self.lora_rank = 32.0;
                self.lora_alpha = 32.0;
                self.timestep_shift = 3.0;
            }
            // --- UNVERIFIED models (wired from the CLI; not smoke-tested) ---
            12 => {
                self.backend_target = String::from("flux");
                self.model_type = String::from("flux");
                self.model_type_index = 12;
                self.architecture_index = 12;
                self.base_model_path = String::new();
                self.model_arch = String::from("flux1_dev");
            }
            13 => {
                self.backend_target = String::from("qwenimage");
                self.model_type = String::from("qwenimage");
                self.model_type_index = 13;
                self.architecture_index = 13;
                self.base_model_path = String::new();
                self.model_arch = String::from("qwen_image");
            }
            14 => {
                self.backend_target = String::from("acestep");
                self.model_type = String::from("acestep");
                self.model_type_index = 14;
                self.architecture_index = 14;
                self.base_model_path = String::new();
                self.model_arch = String::from("acestep");
            }
            15 => {
                // Slider (Klein) — uses the Klein 9B transformer.
                self.backend_target = String::from("slider_klein");
                self.model_type = String::from("slider_klein");
                self.model_type_index = 15;
                self.architecture_index = 15;
                self.base_model_path = String::from(SERENITY_KLEIN9B_CHECKPOINT);
                self.model_arch = String::from("klein9b_slider");
            }
            16 => {
                // AsymFlow — Klein 9B fork (student/teacher); also needs the
                // adapter in Aux Model (Model tab).
                self.backend_target = String::from("asymflow");
                self.model_type = String::from("asymflow");
                self.model_type_index = 16;
                self.architecture_index = 16;
                self.base_model_path = String::from(SERENITY_KLEIN9B_CHECKPOINT);
                self.model_arch = String::from("klein9b_asymflow");
            }
            17 => {
                self.backend_target = String::from("u1");
                self.model_type = String::from("u1");
                self.model_type_index = 17;
                self.architecture_index = 17;
                self.base_model_path = String::new();
                self.model_arch = String::from("sensenova_u1");
            }
            _ => {}
        }
    }

    /// The optimizer enum the dropdown selects (label form, e.g. "ADAMW").
    pub fn optimizer_label(&self) -> String {
        optimizer_options()
            .get(self.optimizer_index)
            .cloned()
            .unwrap_or_default()
    }

    /// The optimizer enum string emitted into the runner train config. The
    /// dropdown labels map verbatim except ADAMW8BIT -> ADAMW_8BIT (mirrors
    /// the Mojo `optimizer_runner_value`).
    pub fn optimizer_runner_value(&self) -> String {
        let label = self.optimizer_label();
        if label == "ADAMW8BIT" {
            String::from("ADAMW_8BIT")
        } else {
            label
        }
    }

    /// Lever keys the selected backend's runner actually CONSUMES (mirrors
    /// `trainer_ui_supported_lever_keys`). Only klein/zimage/hidream/ideogram4
    /// consume the T1 levers; every other runner consumes none of them.
    fn supported_lever_keys(&self) -> Vec<&'static str> {
        match self.backend_target.as_str() {
            "klein" | "zimage" | "hidream" | "ideogram4" => vec![
                "optimizer",
                "warmup",
                "loss_fn",
                "min_snr_gamma_flow",
                "ema",
                "caption_dropout",
            ],
            _ => Vec::new(),
        }
    }

    /// Non-default lever / decorative widgets, stable display order (mirrors
    /// `trainer_ui_active_lever_keys`). Defaults produce an EMPTY list.
    fn active_lever_keys(&self) -> Vec<&'static str> {
        let mut keys: Vec<&'static str> = Vec::new();
        if self.optimizer_runner_value() != "ADAMW" {
            keys.push("optimizer");
        }
        if self.learning_rate_warmup_steps as i64 > 0 {
            keys.push("warmup");
        }
        if self.loss_fn != "mse" {
            keys.push("loss_fn");
        }
        if self.min_snr_gamma_flow != 0.0 {
            keys.push("min_snr_gamma_flow");
        }
        if self.ema_mode != "OFF" {
            keys.push("ema");
        }
        if self.caption_dropout > 0.0 {
            keys.push("caption_dropout");
        }
        if self.masked_training {
            keys.push("masked_training");
        }
        if self.training_method_index != 0 {
            keys.push("training_method");
        }
        if self.peft_type != "LORA" {
            keys.push("peft_type");
        }
        keys
    }

    /// "<model> ignores: a, b" for every non-default widget the selected
    /// model's runner does not consume; empty when nothing is ignored
    /// (mirrors `trainer_ui_ignored_lever_summary`). Shown in the top bar
    /// before launch.
    pub fn ignored_lever_summary(&self) -> String {
        let active = self.active_lever_keys();
        let supported = self.supported_lever_keys();
        let ignored: Vec<&str> = active
            .into_iter()
            .filter(|k| !supported.contains(k))
            .collect();
        if ignored.is_empty() {
            String::new()
        } else {
            format!("{} ignores: {}", self.backend_target, ignored.join(", "))
        }
    }

    pub fn validate(&self) -> String {
        if self.base_model_path.trim().is_empty() {
            String::from("INVALID: base model is required")
        } else if self.dataset_path.trim().is_empty() {
            String::from("INVALID: dataset path is required")
        } else if self.lora_rank < 1.0 {
            String::from("INVALID: LoRA rank must be >= 1")
        } else if self.learning_rate <= 0.0 {
            String::from("INVALID: learning rate must be > 0")
        } else if self.model_type.trim().is_empty() {
            String::from("INVALID: model_type empty")
        } else if self.max_train_steps < 1.0 {
            String::from("INVALID: max_train_steps < 1")
        } else {
            String::from("Ready")
        }
    }
}

/// Models whose launch path has been smoke-verified end-to-end against the real
/// EDv2 binary. The rest are wired but UNVERIFIED — do not rely until tested.
pub fn model_verified(model_type: &str) -> bool {
    matches!(
        model_type,
        "ideogram4" | "klein" | "sdxl" | "zimage" | "chroma" | "ernie" | "anima" | "hidream" | "sd35"
            | "l2p"
    )
}

// --- Config persistence (save/load the UI config as JSON) ---

impl TrainConfig {
    /// Serialize this config to `path` (creates parent dirs).
    pub fn save_to(&self, path: &std::path::Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
        }
        let json = serde_json::to_string_pretty(self).map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(path, json).map_err(|e| format!("write {}: {e}", path.display()))
    }

    /// Load a config from `path`.
    pub fn load_from(path: &std::path::Path) -> Result<TrainConfig, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))
    }
}

/// Directory where saved UI configs live (`$HOME/.config/eritrainer/configs`).
pub fn configs_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
    std::path::PathBuf::from(home).join(".config/eritrainer/configs")
}

/// List saved configs as (name, path), sorted by name.
pub fn list_saved_configs() -> Vec<(String, std::path::PathBuf)> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(configs_dir()) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.extension().map_or(false, |x| x == "json") {
                if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                    out.push((stem.to_string(), p.clone()));
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[cfg(test)]
mod persistence_tests {
    use super::*;

    #[test]
    fn config_save_load_roundtrip() {
        let mut cfg = TrainConfig::default();
        cfg.run_name = "roundtrip_test".into();
        cfg.model_type = "klein".into();
        cfg.learning_rate = 1.23e-4;
        cfg.lora_rank = 24.0;
        let path = std::env::temp_dir().join("eritrainer_cfg_roundtrip.json");
        cfg.save_to(&path).expect("save");
        let loaded = TrainConfig::load_from(&path).expect("load");
        assert_eq!(loaded.run_name, "roundtrip_test");
        assert_eq!(loaded.model_type, "klein");
        assert!((loaded.learning_rate - 1.23e-4).abs() < 1e-9);
        assert!((loaded.lora_rank - 24.0).abs() < 1e-6);
        let _ = std::fs::remove_file(&path);
    }
}

#[cfg(test)]
mod preset_tests {
    use super::*;

    // Proves the block-defect fix: selecting a model actually drives the launch
    // identity (model_type → `train_{model_type}`) and backend_target, instead
    // of staying frozen at the Default "klein".

    #[test]
    fn switching_architecture_drives_launch_identity() {
        let mut cfg = TrainConfig::default();
        cfg.architecture_index = 3; // SDXL via the Architecture combo
        cfg.apply_model_preset(false);
        assert_eq!(cfg.model_type, "sdxl"); // launcher -> train_sdxl
        assert_eq!(cfg.backend_target, "sdxl"); // honesty + live-target label
        assert_eq!(cfg.model_type_index, 2);
        assert!((cfg.lora_rank - 16.0).abs() < 1e-6);
        assert!((cfg.learning_rate - 1e-4).abs() < 1e-9);
    }

    #[test]
    fn switching_model_type_resolves_architecture() {
        let mut cfg = TrainConfig::default();
        cfg.model_type_index = 11; // HiDream-O1 via the Model Type combo
        cfg.apply_model_preset(true);
        assert_eq!(cfg.architecture_index, 11);
        assert_eq!(cfg.backend_target, cfg.model_type); // both set to the binary
        assert!(!cfg.model_type.is_empty());
    }

    #[test]
    fn sd35_resolves_to_sd35_backend_with_recipe() {
        // SD3.5 has no Architecture-combo entry (model-type-only). EDv2 has
        // train_sd35, so selecting it applies the full SD3.5 recipe (was
        // fail-loud while the Mojo had no SD3.5 trainer).
        let mut cfg = TrainConfig::default();
        cfg.model_type_index = 3; // STABLE_DIFFUSION_35
        cfg.apply_model_preset(true);
        assert_eq!(cfg.model_type, "sd35");
        assert_eq!(cfg.backend_target, "sd35");
        assert!(!cfg.base_model_path.is_empty());
    }

    #[test]
    fn honesty_warning_follows_selected_backend() {
        // Klein does not consume masked_training -> it must be named as ignored,
        // and the summary must reference the SELECTED backend (not stale klein).
        let mut cfg = TrainConfig::default();
        cfg.architecture_index = 1; // Klein
        cfg.apply_model_preset(false);
        cfg.masked_training = true;
        let warn = cfg.ignored_lever_summary();
        assert!(warn.contains("klein"), "warn={warn}");
        assert!(warn.contains("masked_training"), "warn={warn}");

        cfg.architecture_index = 3; // switch to SDXL
        cfg.apply_model_preset(false);
        let warn2 = cfg.ignored_lever_summary();
        assert!(warn2.contains("sdxl"), "warn2={warn2}");
    }

    #[test]
    fn all_models_selectable_via_preset() {
        assert_eq!(architecture_options().len(), 18);
        assert_eq!(model_type_options().len(), 18);
        let cases = [
            (12, "flux"),
            (13, "qwenimage"),
            (14, "acestep"),
            (15, "slider_klein"),
            (16, "asymflow"),
            (17, "u1"),
        ];
        for (idx, mt) in cases {
            let mut cfg = TrainConfig::default();
            cfg.architecture_index = idx;
            cfg.apply_model_preset(false);
            assert_eq!(cfg.model_type, mt, "arch index {idx} -> model_type");
        }
    }
}
