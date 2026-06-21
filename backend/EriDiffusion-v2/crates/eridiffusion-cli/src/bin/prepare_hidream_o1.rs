//! prepare_hidream_o1 — image+caption → cached pixel patches + tokenized
//! Qwen3-VL prompt stream for HiDream-O1-Image LoRA training.
//!
//! HiDream-O1 is a **pixel-level Unified Transformer** (no VAE). The model
//! is a Qwen3-VL 8B backbone with three added heads (`x_embedder`,
//! `t_embedder1`, `final_layer2`) that operates directly on
//! `PATCH_SIZE=32` patches of raw RGB pixels in `[-1, 1]` range. Confirmed
//! against:
//!
//!   - `/home/alex/HiDream-O1-Image/models/pipeline.py:17` (`PATCH_SIZE=32`),
//!     :291-295 (initial noise on `[B,3,H,W]` pixel space — no VAE encode).
//!   - `/home/alex/HiDream-O1-Image/models/qwen3_vl_transformers.py:1389-1391,
//!     1525-1526` (forward eats `vinputs=[B, L, 3*32*32=3072]`).
//!   - `/home/alex/EriDiffusion/inference-flame/src/models/hidream_o1/
//!     bottleneck_patch_embed.rs:67-90` (`patchify`).
//!   - `/home/alex/EriDiffusion/EriDiffusion-v2/docs/hidream_o1_trainer_analysis.md`
//!     §1 ("Pixel-level, no VAE") and §4.1 (cache schema).
//!
//! Cache shape (per-sample `.safetensors`, all stored as F32 — the
//! flame-core serialization layer dtype-erases everything to F32):
//!
//!   - `patches`      F32 [1, L, 3072]   pre-patchified pixels in [-1, 1]
//!                                       L = (H/32) * (W/32)
//!   - `input_ids`    F32 [1, S_text]    Qwen3-VL chat-template-tokenised
//!                                       prompt incl. `<|boi_token|>` +
//!                                       `<|tms_token|>` (token ids as f32;
//!                                       max vocab 152k < 2^24 → exact)
//!   - `position_ids` F32 [3, S_total]   3D MRoPE T/H/W positions stacked
//!                                       S_total = S_text + L
//!   - `vinput_mask`  F32 [1, S_total]   1.0 at image slots, 0.0 elsewhere
//!                                       (`token_types == 1`)
//!   - `token_types`  F32 [1, S_total]   1.0 at image slots AND the TMS row,
//!                                       0.0 elsewhere (`token_types > 0`,
//!                                       added in cache v2, 2026-05-17).
//!                                       Used by the trainer for the structured
//!                                       prefix-causal/full attention split so
//!                                       the TMS row gets full-attention (matches
//!                                       `qwen3_vl_transformers.py:1501`).
//!   - `image_grid`   F32 [3]            (1.0, H/32, W/32)
//!
//! ## Cache schema versioning
//!
//! - **v1**: legacy layout WITHOUT `token_types`. Re-generate.
//! - **v2** (current): adds `token_types`. Trainer M3 refuses v1.
//!
//! Mirrors the inference path in `inference-flame/src/models/hidream_o1/
//! pipeline.rs:216-321` (`build_t2i_input`). The inference pipeline calls
//! `build_t2i_input` from a `HiDreamO1Pipeline` instance which requires
//! loading the 8B model; this prep binary re-implements the same logic
//! standalone so we don't load weights just to tokenize.
//!
//! ## Sample invocation
//!
//! ```bash
//! cargo run --release --bin prepare_hidream_o1 -- \
//!     --input-dir <your-dataset> \
//!     --output-dir <your-cache-dir> \
//!     --model-path /home/alex/HiDream-O1-Image-Dev-weights \
//!     --resolution 512
//! ```

use clap::Parser;
use flame_core::{serialization::save_file, DType, Shape, Tensor};
use std::collections::HashMap;
use std::path::PathBuf;

use eridiffusion_core::models::hidream_o1::{
    bottleneck_patch_embed::BottleneckPatchEmbed, build_mrope_positions, HiDreamO1Config,
};

/// Bake-in special-token id constants are validated against the tokenizer
/// at startup — same contract as `HiDreamO1Pipeline::new` in
/// `inference-flame/src/models/hidream_o1/pipeline.rs:110-119`. A mismatch
/// means the cache would silently miss the timestep injection or the
/// vision-start marker.
fn validate_token_id(
    tokenizer: &tokenizers::Tokenizer,
    token_str: &str,
    expected_id: u32,
) -> anyhow::Result<()> {
    let id = tokenizer
        .token_to_id(token_str)
        .ok_or_else(|| anyhow::anyhow!("tokenizer missing special token {}", token_str))?;
    if id != expected_id {
        anyhow::bail!(
            "tokenizer token-id mismatch for {}: bake-in {} vs tokenizer {}",
            token_str,
            expected_id,
            id
        );
    }
    Ok(())
}

/// Build the chat-template string for a T2I prompt. Byte-identical to
/// `HiDreamO1Pipeline::apply_chat_template_t2i` (pipeline.rs:180-198) so
/// the cached `input_ids` match what the inference forward consumes.
///
/// Produces:
/// ```text
/// <|im_start|>user
/// {prompt}<|im_end|>
/// <|im_start|>assistant
/// <|boi_token|><|tms_token|>
/// ```
fn apply_chat_template_t2i(prompt: &str) -> String {
    let mut s = String::new();
    s.push_str("<|im_start|>user\n");
    s.push_str(prompt);
    s.push_str("<|im_end|>\n");
    s.push_str("<|im_start|>assistant\n");
    s.push_str("<|boi_token|>");
    s.push_str("<|tms_token|>"); // TIMESTEP_TOKEN_NUM == 1
    s
}

#[derive(Parser)]
#[command(name = "prepare_hidream_o1")]
struct Args {
    /// Directory containing `*.{jpg,png,webp,...}` paired with same-stem `*.txt` captions.
    #[arg(long)]
    input_dir: PathBuf,
    /// Per-sample `.safetensors` written here, plus a top-level `_meta.json`.
    #[arg(long)]
    output_dir: PathBuf,
    /// HiDream-O1-Image weights directory (used for `tokenizer.json` only —
    /// model weights are NOT loaded by this prep binary).
    #[arg(long, default_value = "/home/alex/HiDream-O1-Image-Dev-weights")]
    model_path: PathBuf,
    /// Square training resolution(s). Comma-separated list of one or more
    /// values, each a multiple of 32 (patch size). 512 is the
    /// analysis-recommended single-bucket default for 24 GB (see
    /// `docs/hidream_o1_trainer_analysis.md` §8 risk #4).
    ///
    /// Multi-bucket: `--resolution 512,768,1024` matches the edv2-reference
    /// convention (`config/examples/train_lora_hidream_48.yaml` `datasets.
    /// resolution: [512, 768, 1024]`). Each source image is assigned to
    /// the bucket whose pixel count is closest to the source pixel count
    /// (`H*W` distance). Per-sample `patches`/`position_ids`/`image_grid`
    /// fields already encode the per-sample L = (H/32)*(W/32), so the
    /// trainer reads each sample independently — varying-L samples are
    /// transparently interleaved (batch_size=1 today; per-bucket dataloader
    /// batching is a separate trainer-side M3 concern).
    ///
    /// The resampling mode (Lanczos3) is the project-wide convention; the
    /// edv2-reference reference path uses `Image.BICUBIC` — see the
    /// `_meta.json` `resampler` field for cross-loader diagnostic.
    #[arg(long, default_value = "512", value_delimiter = ',')]
    resolution: Vec<u32>,
    #[arg(long, default_value_t = true)]
    skip_existing: bool,
    #[arg(long, default_value_t = 0)]
    max_samples: usize,
    /// Image augmentations at prep time. All default-off → byte-identical
    /// caches. Mirrors `prepare_chroma.rs` (see §image_aug docs there).
    #[arg(long, default_value_t = false)]
    aug_flip: bool,
    #[arg(long, default_value_t = 0.0)]
    aug_brightness: f32,
    #[arg(long, default_value_t = 0.0)]
    aug_contrast: f32,
    #[arg(long, default_value_t = 0)]
    aug_seed: u64,
}

/// Canonical 1024-base rectangular bucket list, mirroring edv2-reference's
/// `toolkit/buckets.py:resolutions_1024`. Each entry is a (W, H) target
/// computed so that W*H ≈ 1024² with aspect ratios spanning extreme wide
/// (16:1) through square (1:1) through extreme portrait (1:16).
///
/// `bucket_list_for_resolution` scales these by `R / 1024` and snaps W, H
/// down to a multiple of `divisibility` (must be a multiple of 32 for
/// HiDream-O1's PATCH_SIZE=32 constraint).
const RESOLUTIONS_1024: &[(u32, u32)] = &[
    // Square
    (1024, 1024),
    // Landscape
    (2048, 512), (1984, 512), (1920, 512), (1856, 512),
    (1792, 576), (1728, 576), (1664, 576),
    (1600, 640), (1536, 640),
    (1472, 704), (1408, 704), (1344, 704),
    (1344, 768), (1280, 768),
    (1216, 832), (1152, 832),
    (1152, 896), (1088, 896),
    (1088, 960), (1024, 960),
    // Portrait (mirror)
    (960, 1024), (960, 1088),
    (896, 1088), (896, 1152),
    (832, 1152), (832, 1216),
    (768, 1280), (768, 1344),
    (704, 1344), (704, 1408), (704, 1472),
    (640, 1536), (640, 1600),
    (576, 1664), (576, 1728), (576, 1792),
    (512, 1856), (512, 1920), (512, 1984), (512, 2048),
];

/// Scale the 1024-base bucket list to a target resolution R and snap to
/// `divisibility`. Mirrors edv2-reference `buckets.py:get_bucket_sizes` lines
/// 59-74. For HiDream-O1, divisibility is 32 (PATCH_SIZE).
fn bucket_list_for_resolution(resolution: u32, divisibility: u32) -> Vec<(u32, u32)> {
    let scaler = resolution as f64 / 1024.0;
    let mut out = Vec::with_capacity(RESOLUTIONS_1024.len());
    for &(w, h) in RESOLUTIONS_1024 {
        let mut nw = (w as f64 * scaler) as u32;
        let mut nh = (h as f64 * scaler) as u32;
        nw -= nw % divisibility;
        nh -= nh % divisibility;
        if nw == 0 || nh == 0 {
            continue;
        }
        out.push((nw, nh));
    }
    // Dedup
    out.sort_unstable();
    out.dedup();
    out
}

/// Pick the bucket that minimizes removed (cropped) pixels after an
/// AR-preserving "fit larger" scale. Mirrors edv2-reference
/// `buckets.py:get_bucket_for_image_size` lines 101-127:
///   scale = max(bucket_w/w, bucket_h/h)  → both dims ≥ bucket dims
///   removed_pixels = (new_w - bucket_w) * new_h + (new_h - bucket_h) * new_w
fn pick_bucket_for_image(src_w: u32, src_h: u32, buckets: &[(u32, u32)]) -> (u32, u32) {
    // Exact match first
    for &(bw, bh) in buckets {
        if bw == src_w && bh == src_h {
            return (bw, bh);
        }
    }
    let sw = src_w as f64;
    let sh = src_h as f64;
    let mut best = buckets[0];
    let mut best_removed = f64::INFINITY;
    for &(bw, bh) in buckets {
        let scale_w = bw as f64 / sw;
        let scale_h = bh as f64 / sh;
        let scale = scale_w.max(scale_h);
        let new_w = sw * scale;
        let new_h = sh * scale;
        let removed = (new_w - bw as f64) * new_h + (new_h - bh as f64) * new_w;
        if removed < best_removed {
            best_removed = removed;
            best = (bw, bh);
        }
    }
    best
}

fn main() -> anyhow::Result<()> {
    // Disable flame_core CUDA alloc pool — same rationale as prepare_klein
    // / prepare_chroma. Pool retains slabs; one-pass prep grows host RSS by
    // ~1 GB per sample and OOM-kills a 62 GB box around sample 75.
    if std::env::var_os("FLAME_ALLOC_POOL").is_none() {
        // SAFETY: single-threaded at this point.
        unsafe {
            std::env::set_var("FLAME_ALLOC_POOL", "0");
        }
    }
    env_logger::init();
    let args = Args::parse();

    let patch_size = 32u32;
    if args.resolution.is_empty() {
        anyhow::bail!("--resolution must contain at least one value");
    }
    for &r in &args.resolution {
        if r % patch_size != 0 {
            anyhow::bail!(
                "--resolution values must each be a multiple of 32 (the HiDream-O1 patch size); \
                 valid examples: 512, 768, 1024, 2048. Got {}.",
                r
            );
        }
    }
    let mut resolutions: Vec<u32> = args.resolution.clone();
    resolutions.sort_unstable();
    resolutions.dedup();
    // Build the union of AR-preserving rectangular buckets across all
    // requested resolutions. Mirrors edv2-reference's behaviour where each
    // `--resolution R` produces ~40 rectangular (W, H) candidates spanning
    // ARs from 16:1 → 1:1 → 1:16. Per-image assignment then picks the
    // bucket with the smallest cropped-pixel count. Divisibility = 32 to
    // honour HiDream-O1's PATCH_SIZE.
    let mut rect_buckets: Vec<(u32, u32)> = Vec::new();
    for &r in &resolutions {
        rect_buckets.extend(bucket_list_for_resolution(r, patch_size));
    }
    rect_buckets.sort_unstable();
    rect_buckets.dedup();
    if rect_buckets.is_empty() {
        anyhow::bail!(
            "--resolution {:?} produced no valid rectangular buckets at patch_size={}",
            resolutions, patch_size
        );
    }
    log::info!(
        "[hidream_o1] requested resolution(s): {:?} → {} rectangular AR-preserving bucket(s) \
         (e.g. {:?} ... {:?})",
        resolutions,
        rect_buckets.len(),
        rect_buckets.first().unwrap(),
        rect_buckets.last().unwrap(),
    );
    std::fs::create_dir_all(&args.output_dir)?;
    flame_core::config::set_default_dtype(DType::BF16);
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    let device = flame_core::global_cuda_device();

    log::info!("[1/3] Loading Qwen3-VL tokenizer from {:?}...", args.model_path);
    let tokenizer_path = args.model_path.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer::from_file({}): {e}", tokenizer_path.display()))?;

    // Validate special-token IDs against the bake-in constants. Same
    // contract as HiDreamO1Pipeline::new (pipeline.rs:110-119).
    let config = HiDreamO1Config::dev_8b();
    validate_token_id(&tokenizer, "<|tms_token|>", config.tms_token_id)?;
    validate_token_id(&tokenizer, "<|image_pad|>", config.image_token_id)?;
    validate_token_id(&tokenizer, "<|vision_start|>", config.vision_start_token_id)?;
    if tokenizer.token_to_id("<|boi_token|>").is_none() {
        anyhow::bail!("tokenizer missing special token <|boi_token|>");
    }

    log::info!("[2/3] Scanning {:?} for image+caption pairs...", args.input_dir);
    let mut pairs = Vec::new();
    for entry in std::fs::read_dir(&args.input_dir)? {
        let p = entry?.path();
        if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
            if matches!(
                ext.to_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "webp" | "bmp"
            ) {
                let stem = p.file_stem().unwrap().to_str().unwrap();
                pairs.push((p.clone(), args.input_dir.join(format!("{stem}.txt"))));
            }
        }
    }
    log::info!("Found {} image-caption pairs", pairs.len());
    if args.max_samples > 0 && pairs.len() > args.max_samples {
        pairs.truncate(args.max_samples);
    }

    let aug_cfg = eridiffusion_core::training::features::image_aug::AugConfig {
        flip: args.aug_flip,
        brightness: args.aug_brightness,
        contrast: args.aug_contrast,
    };
    if aug_cfg.is_active() {
        log::info!(
            "[image-aug] flip={} brightness={} contrast={} seed={}",
            aug_cfg.flip, aug_cfg.brightness, aug_cfg.contrast, args.aug_seed
        );
    }

    // Sentinel `_meta.json` — mirrors prepare_chroma's format string but
    // surfaces the resolution + token-stream version. Trainer reads this
    // at startup to refuse caches built at a different resolution.
    //
    // `resampler` is logged for cross-loader diagnostic clarity: Lanczos3
    // is the project-wide convention across all `prepare_*` binaries,
    // whereas the Python `HiDream-O1-Image/models/utils.py` reference path
    // uses `Image.BICUBIC`. The per-pixel divergence is within bf16 noise
    // but may produce a small persistent distribution shift; see the M2
    // skeptic review HIGH #2.
    //
    // `writer` lets the trainer (M3) distinguish caches produced by
    // different binaries that might claim the same `format` string.
    // CACHE FORMAT VERSION HISTORY:
    //   v1: patches, input_ids, position_ids, vinput_mask, image_grid
    //   v2 (2026-05-17): adds `token_types` (token_types_bin = (raw > 0)).
    //       v1 caches CANNOT be used by the v2 trainer because the attention
    //       mask construction now relies on token_types_bin to make the TMS
    //       row full-attention. See
    //       `EriDiffusion-v2/docs/hidream_o1_g0_deep_investigation.md`.
    //   v3 (2026-05-18): multi-resolution buckets. Per-sample `image_grid`
    //       can vary across the cache dir. v3 caches use the same field set
    //       as v2 (identical schema for any single sample); the version bump
    //       declares that the trainer must not assume a single global
    //       resolution. v2 caches stay readable by the v3 trainer (a v2
    //       cache is just a v3 cache with `len(resolutions) == 1`); v3
    //       caches are NOT readable by any trainer that asserts a global
    //       resolution. The trainer's per-step `load_file` already pulls
    //       per-sample shapes, so this is transparent at the trainer level.
    //       Aligns with edv2-reference `train_lora_hidream_48.yaml`
    //       `datasets.resolution: [512, 768, 1024]`.
    const CACHE_VERSION: u32 = 3;
    let resolutions_json = resolutions
        .iter()
        .map(|r| r.to_string())
        .collect::<Vec<_>>()
        .join(",");
    // Also serialise the rectangular bucket list as "WxH" strings for
    // diagnostic clarity. The trainer doesn't read this field; per-sample
    // shape comes from each `.safetensors` `image_grid`.
    let rect_buckets_json = rect_buckets
        .iter()
        .map(|(w, h)| format!("\"{}x{}\"", w, h))
        .collect::<Vec<_>>()
        .join(",");
    let meta = format!(
        r#"{{"version": {}, "format": "hidream-o1-v3", "writer": "prepare_hidream_o1", "resolutions": [{}], "rect_buckets": [{}], "patch_size": {}, "resampler": "lanczos3", "crop_mode": "ar_preserving_center_crop", "fields": ["patches", "input_ids", "position_ids", "vinput_mask", "token_types", "image_grid"]}}"#,
        CACHE_VERSION, resolutions_json, rect_buckets_json, patch_size
    );
    std::fs::write(args.output_dir.join("_meta.json"), &meta)
        .map_err(|e| anyhow::anyhow!("write _meta.json: {e}"))?;

    log::info!(
        "[3/3] Encoding {} samples ({} AR-preserving buckets)...",
        pairs.len(),
        rect_buckets.len()
    );

    let p = patch_size as usize;

    let mut cached = 0usize;
    let mut skipped = 0usize;
    for (idx, (img_path, txt_path)) in pairs.iter().enumerate() {
        let hash = format!("{:x}", md5::compute(img_path.to_string_lossy().as_bytes()));
        // Prefix with `sample_` to match `train_hidream_o1`'s glob filter
        // (only files whose name starts with `sample_` are picked up).
        // The hash suffix preserves stable per-image dedup / --skip-existing.
        let out_path = args.output_dir.join(format!("sample_{hash}.safetensors"));
        if args.skip_existing && out_path.exists() {
            skipped += 1;
            continue;
        }

        // ── Image → AR-preserving bucket fit → center crop → [-1, 1] → patchify ──
        //
        // Mirrors edv2-reference's `AspectRatioBucketMixin.setup_buckets`
        // (`dataloader_mixins.py:262-295`) + `get_bucket_for_image_size`
        // (`buckets.py:101-127`):
        //   1. Pick the rectangular bucket (bw, bh) that minimizes
        //      removed pixels under an AR-preserving "fit larger" scale.
        //   2. scale = max(bw/sw, bh/sh) → both dims ≥ bucket dims.
        //   3. Resize to (ceil(sw*scale), ceil(sh*scale)).
        //   4. Center-crop to exactly (bw, bh). (Random-crop variant is
        //      a future flag; edv2-reference defaults to center.)
        // Aspect ratio is preserved — no square stretch.
        let probe = image::open(img_path)?;
        let (src_w, src_h) = (probe.width(), probe.height());
        let (bw, bh) = pick_bucket_for_image(src_w, src_h, &rect_buckets);
        let scale_w = bw as f64 / src_w as f64;
        let scale_h = bh as f64 / src_h as f64;
        let scale = scale_w.max(scale_h);
        let scaled_w = ((src_w as f64) * scale).ceil() as u32;
        let scaled_h = ((src_h as f64) * scale).ceil() as u32;
        // Safety: ceil() may overshoot by 1 px under floating noise; clamp
        // so the crop always fits inside the resize output.
        let scaled_w = scaled_w.max(bw);
        let scaled_h = scaled_h.max(bh);
        let resized = probe
            .resize_exact(scaled_w, scaled_h, image::imageops::FilterType::Lanczos3)
            .to_rgb8();
        let crop_x = (scaled_w - bw) / 2;
        let crop_y = (scaled_h - bh) / 2;
        let cropped = image::imageops::crop_imm(&resized, crop_x, crop_y, bw, bh).to_image();
        let mut img = image::DynamicImage::ImageRgb8(cropped).to_rgb32f();
        let h_patches = (bh as usize) / p;
        let w_patches = (bw as usize) / p;
        let image_len = h_patches * w_patches;
        if idx == 0 {
            log::info!(
                "[bucket] sample 0: src={}x{} → bucket={}x{} (scaled={}x{}, crop_off=({},{}))",
                src_w, src_h, bw, bh, scaled_w, scaled_h, crop_x, crop_y
            );
        }
        if aug_cfg.is_active() {
            use rand::SeedableRng;
            // Avoid the `seed ^ idx` degenerate case where `aug_seed=0, idx=0`
            // collapses to seed 0 (and per-sample RNGs become correlated
            // under the default seed). Multiply-add through a large odd
            // constant gives a well-distributed per-sample seed.
            let seed = args
                .aug_seed
                .wrapping_add((idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut aug_rng = rand::rngs::StdRng::seed_from_u64(seed);
            eridiffusion_core::training::features::image_aug::apply_augs(
                &mut img,
                None,
                &aug_cfg,
                &mut aug_rng,
            );
        }
        let (iw, ih) = img.dimensions();
        // CHW transpose — same bug fix as prepare_klein / prepare_chroma.
        // Without explicit transpose, raw `image::pixels()` (HWC interleaved)
        // reshaped to [1, 3, H, W] (CHW) scrambles channels. Also maps
        // [0, 1] → [-1, 1] (the pixel range HiDream-O1 expects per
        // pipeline.py:390 `(z + 1) / 2`).
        let (wu, hu) = (iw as usize, ih as usize);
        let mut pixels = vec![0f32; 3 * hu * wu];
        for (x, y, px) in img.enumerate_pixels() {
            let (xu, yu) = (x as usize, y as usize);
            for c in 0..3 {
                pixels[c * hu * wu + yu * wu + xu] = px.0[c] * 2.0 - 1.0;
            }
        }
        let img_t = Tensor::from_vec(pixels, Shape::from_dims(&[1, 3, hu, wu]), device.clone())?
            .to_dtype(DType::BF16)?;
        // [1, 3, H, W] → [1, L, 3072] where 3072 = 3*32*32.
        let patches = BottleneckPatchEmbed::patchify(&img_t, p)?;

        // ── Caption → Qwen3-VL chat-template tokens ──
        // A missing `.txt` is almost always a dataset bug rather than an
        // intentional CFG-uncond signal. Warn so it shows up in the prep
        // log instead of silently becoming an empty-prompt cache.
        let caption_raw = match std::fs::read_to_string(txt_path) {
            Ok(s) => s,
            Err(e) => {
                log::warn!(
                    "caption missing for {:?} ({}); caching as empty prompt",
                    txt_path,
                    e
                );
                String::new()
            }
        };
        // NOTE: `.trim()` matches the project-wide `prepare_*` convention
        // (see `prepare_chroma.rs:197`). Python `pipeline.py` does NOT
        // trim user prompts; for normal captions the divergence is zero
        // tokens, but a caption like "  a cat  " will tokenize one or
        // two tokens shorter here than at Python inference. The inference
        // sample loop should apply the same trim if added.
        // TODO(O1-M2.1): decide whether to drop `.trim()` for byte-exact
        // Python parity, or keep and propagate to inference sample loops.
        let caption = caption_raw.trim();
        if caption.is_empty() {
            log::warn!(
                "caption file {:?} is empty; building empty-prompt cache",
                txt_path
            );
        }
        let template = apply_chat_template_t2i(caption);

        // `add_special_tokens=false` — the chat template already includes
        // every special marker the model expects. Matches pipeline.rs:241-244.
        let enc = tokenizer
            .encode(template.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenize failed: {e}"))?;
        let text_ids: Vec<u32> = enc.get_ids().to_vec();
        let txt_seq_len = text_ids.len();

        // ── Build full id stream (text + vision-start + image-pad slots) ──
        // Matches pipeline.rs:251-257. The first image slot is
        // vision_start_token_id (edge case A2 / pipeline.py:51-58); the
        // remaining `image_len - 1` slots are image_token_id.
        //
        // TODO(O1-M2.1): verify end-to-end through `model.forward` at
        // TRAINING time that the noise patch placed in the
        // vision-start slot (vmask=1 there) is consumed correctly. The
        // bottleneck patch embed writes all L slots; the scatter happens
        // in `vinput_mask`-gated regions of the transformer — looks OK
        // in theory but no training-time test confirms it (Q2).
        //
        // TODO(O1-M2.1): captions of varying length produce varying
        // `S_text` (and therefore `S_total`); M3 trainer batching needs
        // per-sample sequence handling or padding — confirm M3 plan
        // anticipates this (Q3).
        let mut full_ids: Vec<u32> = Vec::with_capacity(txt_seq_len + image_len);
        full_ids.extend_from_slice(&text_ids);
        full_ids.push(config.vision_start_token_id);
        for _ in 1..image_len {
            full_ids.push(config.image_token_id);
        }
        let all_seq_len = full_ids.len();

        // ── 3D MRoPE positions over the FULL stream ──
        // skip_vision_start_token = [1] for T2I (pipeline.py:58 / pipeline.rs:269).
        let (t_pos, h_pos, w_pos) = build_mrope_positions(
            &full_ids,
            config.image_token_id,
            config.video_token_id,
            config.vision_start_token_id,
            &[(1, h_patches, w_patches)],
            &[1],
            Some(config.fix_point),
        );

        // ── vinput_mask: 1.0 at the L image-patch slots (tail), 0.0 elsewhere ──
        // Matches pipeline.rs:287-290.
        let mut vmask = vec![0.0_f32; all_seq_len];
        for i in txt_seq_len..(txt_seq_len + image_len) {
            vmask[i] = 1.0;
        }

        // ── Pack into tensors. F32 throughout — flame-core's safetensors
        //    writer dtype-erases to F32 anyway (see serialization.rs:391-394
        //    + load skip of I32/I64 at :504-505). Token ids ≤ 152k < 2^24 so
        //    F32 round-trips exactly. ──
        let input_ids_f: Vec<f32> = text_ids.iter().map(|&id| id as f32).collect();
        let input_ids = Tensor::from_vec(
            input_ids_f,
            Shape::from_dims(&[1, txt_seq_len]),
            device.clone(),
        )?;

        // Stack T/H/W into [3, S_total] row-major.
        let mut pos_f: Vec<f32> = Vec::with_capacity(3 * all_seq_len);
        pos_f.extend(t_pos.iter().map(|&v| v as f32));
        pos_f.extend(h_pos.iter().map(|&v| v as f32));
        pos_f.extend(w_pos.iter().map(|&v| v as f32));
        let position_ids = Tensor::from_vec(
            pos_f,
            Shape::from_dims(&[3, all_seq_len]),
            device.clone(),
        )?;

        let vinput_mask = Tensor::from_vec(
            vmask,
            Shape::from_dims(&[1, all_seq_len]),
            device.clone(),
        )?;

        // ── token_types: (token_types > 0) — image rows + TMS row ──
        // Matches `pipeline.py:64-70` (token_types_bin) and
        // `inference-flame/src/models/hidream_o1/pipeline.rs::build_t2i_input`.
        // True over the image slots PLUS the TMS slot at `txt_seq_len - 1`.
        // Used by the v2 trainer for `model.forward_lora(..., token_types,
        // ...)`; the attention mask makes the TMS row full-attention.
        let mut token_types = vec![0.0_f32; all_seq_len];
        for i in txt_seq_len..(txt_seq_len + image_len) {
            token_types[i] = 1.0;
        }
        if txt_seq_len > 0 {
            token_types[txt_seq_len - 1] = 1.0;
        }
        let token_types_tensor = Tensor::from_vec(
            token_types,
            Shape::from_dims(&[1, all_seq_len]),
            device.clone(),
        )?;

        let image_grid = Tensor::from_vec(
            vec![1.0_f32, h_patches as f32, w_patches as f32],
            Shape::from_dims(&[3]),
            device.clone(),
        )?;

        // TODO(O1-M2.1): `save_file` internally does `tensor.to_vec()` on
        // a `[1, L=256, 3072]` F32 tensor (~3 MB at 512²). Worth a
        // memory check over the first ~100 samples to confirm
        // `FLAME_ALLOC_POOL=0` is actually keeping host RSS flat (Q4).
        let mut tensors = HashMap::new();
        tensors.insert("patches".to_string(), patches.to_dtype(DType::F32)?);
        tensors.insert("input_ids".to_string(), input_ids);
        tensors.insert("position_ids".to_string(), position_ids);
        tensors.insert("vinput_mask".to_string(), vinput_mask);
        tensors.insert("token_types".to_string(), token_types_tensor);
        tensors.insert("image_grid".to_string(), image_grid);
        save_file(&tensors, &out_path)?;
        cached += 1;

        // Include skips in the cadence so a resume-on-mostly-complete-cache
        // run still emits progress lines during the skip scan.
        if (cached + skipped) % 50 == 0 || cached + skipped == pairs.len() {
            log::info!(
                "Cached {}/{} (skipped {}, S_text={}, S_total={}, L={})",
                cached,
                pairs.len(),
                skipped,
                txt_seq_len,
                all_seq_len,
                image_len
            );
        }
    }

    log::info!(
        "Done. {} samples cached, {} skipped, output dir {:?}",
        cached, skipped, args.output_dir
    );
    Ok(())
}
