//! prepare_asymflow — image+caption → dual-cache (Oklab + Klein VAE) + Qwen3
//! embeddings for AsymFLUX.2 Klein 9B LoRA training.
//!
//! Fork of `prepare_klein.rs`. Adds a third saved tensor — `image_oklab` —
//! holding the deterministic Oklab-encoded RGB image alongside the existing
//! `latent` (low-rank Klein VAE) and `text_embedding` (Qwen3 extract layers).
//!
//! AsymFLUX.2 trains a student → teacher distillation loop where:
//!   - `latent` (128-ch packed Klein VAE)   → low-rank student target
//!   - `image_oklab` ([1,3,H,W] BF16)       → full-rank teacher target (no
//!     learned weights — pure sRGB↔Oklab color-space math, see
//!     `eridiffusion_core::oklab`)
//!
//! Output per sample (one safetensors file in `--output-dir`):
//!   - latent:         BF16 [1, 128, H/16, W/16]   — KleinVaeEncoder.encode
//!   - image_oklab:    BF16 [1, 3,   H,    W]      — Oklab(image) per AsymFLUX.2
//!   - text_embedding: BF16 [1, 512, joint_dim]    — Qwen3 stacked extracts
//!   - text_mask:      F32  [1, 512]
//!   - latent_mask     BF16 [1, 1, H/16, W/16]      (optional, when mask source present)
//!
//! Latent + text path is byte-identical to `prepare_klein.rs`. Bucketing,
//! caption filter, skip-existing, image-aug, and crop-style flags carry over
//! unchanged.
//!
//! TODO acceptance: 10-image Alina subset, 4-tensor outputs, latent
//! byte-identical to prepare_klein, oklab decode round-trip within epsilon
//! (deferred until GPU free).

use clap::Parser;
use eridiffusion_core::encoders::{qwen3::Qwen3Encoder, vae::KleinVaeEncoder};
use flame_core::{serialization::save_file, DType, Shape, Tensor};
use eridiffusion_core::oklab;
use std::collections::HashMap;
use std::path::PathBuf;

const KLEIN_TEMPLATE_PRE: &str = "<|im_start|>user\n";
const KLEIN_TEMPLATE_POST: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const PAD_TOKEN_ID: i32 = 151643;
const TXT_PAD_LEN: usize = 512;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    input_dir: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    /// Klein VAE safetensors (e.g. flux2-vae.safetensors). Same VAE for 4B and 9B.
    #[arg(long)]
    vae_ckpt: PathBuf,
    /// Qwen3 weights path (single file or sharded dir). qwen_3_4b for 4B, larger for 9B.
    #[arg(long)]
    qwen3: PathBuf,
    #[arg(long)]
    tokenizer_path: PathBuf,
    #[arg(long, default_value = "512")]
    resolution: u32,
    #[arg(long, default_value_t = true)]
    skip_existing: bool,
    #[arg(long, default_value_t = 0)]
    max_samples: usize,
    /// Aspect-ratio bucketing. When true, image is resized + center-cropped
    /// to the closest 64-aligned bucket whose total pixel count is
    /// ≈ resolution² and whose aspect ratio is closest to the source. This
    /// matches OneTrainer Flux2BaseDataLoader (`aspect_bucketing_quantization=64`)
    /// and avoids the ~20% vertical compression that forced-square does on
    /// 4:5 portrait datasets like Alina. Set to false to keep legacy
    /// `resize_exact(R, R)` behavior.
    #[arg(long, default_value_t = true)]
    bucketing: bool,

    // ── Phase 6 multi-feature rollout ────────────────────────────────────
    /// Per-crop style: `center` (default), `random`, `top_left`, `top_right`,
    /// `bottom_left`, `bottom_right`. `random` chooses uniformly within the
    /// loose-axis margin and adds variation for subject training. Default
    /// `center` preserves byte-invariant prep output.
    #[arg(long, default_value = "center")]
    crop_style: String,
    /// Aspect-bucket alignment in pixels. Default `64` matches OT
    /// `aspect_bucketing_quantization=64`. Smaller values (`32`, `16`) give
    /// finer aspect control at the cost of more buckets. Must be a positive
    /// multiple of 8 (VAE patch size constraint).
    #[arg(long, default_value_t = 64)]
    bucket_alignment: u32,
    /// Optional caption blocklist file. One substring per line; lines
    /// starting with `#` are comments. Any caption containing any pattern
    /// is dropped (the image is not encoded). Default: no filtering.
    #[arg(long)]
    caption_filter_list: Option<PathBuf>,
    /// Re-encode every sample even if `<hash>.safetensors` already exists.
    /// Equivalent to `--skip-existing=false`; provided as an explicit flag
    /// for cache-rebuild workflows. Default `false` (skip existing).
    #[arg(long, default_value_t = false)]
    cache_invalidate: bool,
    /// Phase 6 plumbing: `--caption-tag-shuffle` records intent to randomize
    /// tag order per training step. Cache files store ENCODED text, not raw
    /// captions, so per-step shuffle requires either pre-encoded variants or
    /// runtime re-encoding. Phase 6 ships infrastructure only — this flag is
    /// recorded in the prep log for forward-compat and otherwise unused.
    #[arg(long, default_value_t = false)]
    caption_tag_shuffle: bool,
    /// Image augmentations at prep time. All default-off → byte-identical
    /// caches. Set `--aug-flip` for 50% horizontal flip per sample (also
    /// flips the latent_mask if present). `--aug-brightness <f>` and
    /// `--aug-contrast <f>` jitter pixel values uniformly. `--aug-seed`
    /// seeds the per-sample RNG.
    #[arg(long, default_value_t = false)]
    aug_flip: bool,
    #[arg(long, default_value_t = 0.0)]
    aug_brightness: f32,
    #[arg(long, default_value_t = 0.0)]
    aug_contrast: f32,
    #[arg(long, default_value_t = 0)]
    aug_seed: u64,

    // ── AsymFlow additions ───────────────────────────────────────────────
    /// Skip Oklab encoding. With this set, output is identical to
    /// `prepare_klein` (latent + text only). Use only for cache-comparison
    /// debugging — A3 trainer will fail without `image_oklab` present.
    #[arg(long, default_value_t = false)]
    no_oklab: bool,
}

#[derive(Clone, Copy, Debug)]
enum CropStyle {
    Center,
    Random,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

impl CropStyle {
    fn parse(s: &str) -> anyhow::Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "center" => Self::Center,
            "random" => Self::Random,
            "top_left" => Self::TopLeft,
            "top_right" => Self::TopRight,
            "bottom_left" => Self::BottomLeft,
            "bottom_right" => Self::BottomRight,
            other => anyhow::bail!(
                "--crop-style must be one of center|random|top_left|top_right|bottom_left|bottom_right, got `{other}`"
            ),
        })
    }

    /// (xoff, yoff) given the resized (rw, rh) dimensions and the target
    /// (tw, th) bucket. `rng` is consumed only for `Random`. For `Center`
    /// this is bit-exact identical to the previous `(rw - tw)/2`,
    /// `(rh - th)/2` math — preserves byte invariance for the default path.
    fn pick_offset<R: rand::Rng>(
        self,
        rw: u32,
        rh: u32,
        tw: u32,
        th: u32,
        rng: &mut R,
    ) -> (u32, u32) {
        let max_x = rw.saturating_sub(tw);
        let max_y = rh.saturating_sub(th);
        match self {
            CropStyle::Center => (max_x / 2, max_y / 2),
            CropStyle::Random => {
                let xo = if max_x > 0 {
                    rng.gen_range(0..=max_x)
                } else {
                    0
                };
                let yo = if max_y > 0 {
                    rng.gen_range(0..=max_y)
                } else {
                    0
                };
                (xo, yo)
            }
            CropStyle::TopLeft => (0, 0),
            CropStyle::TopRight => (max_x, 0),
            CropStyle::BottomLeft => (0, max_y),
            CropStyle::BottomRight => (max_x, max_y),
        }
    }
}

/// 64-aligned aspect-ratio buckets. Pick the bucket whose aspect is closest
/// to `(src_w / src_h)` AND whose pixel count is closest to `target_pix`.
fn pick_bucket(src_w: u32, src_h: u32, target_res: u32, alignment: u32) -> (u32, u32) {
    let align = alignment.max(8) as f32;
    const RATIOS: &[(u32, u32)] = &[
        (1, 1),
        (4, 5),
        (5, 4),
        (3, 4),
        (4, 3),
        (9, 16),
        (16, 9),
        (2, 3),
        (3, 2),
    ];
    let target_pix = (target_res as f32) * (target_res as f32);
    let src_aspect = src_w as f32 / src_h as f32;

    let mut best: Option<(f32, f32, u32, u32)> = None;
    for &(rw, rh) in RATIOS {
        let r = rw as f32 / rh as f32;
        let h_f = (target_pix / r).sqrt();
        let w_f = r * h_f;
        let h = (((h_f / align).round() as u32) * (align as u32)).max(align as u32);
        let w = (((w_f / align).round() as u32) * (align as u32)).max(align as u32);
        let aspect_dist = (r - src_aspect).abs();
        let pix_dist = ((w * h) as f32 - target_pix).abs() / target_pix;
        let score = aspect_dist * 100.0 + pix_dist;
        match best {
            None => best = Some((score, aspect_dist, w, h)),
            Some((bs, _, _, _)) if score < bs => best = Some((score, aspect_dist, w, h)),
            _ => {}
        }
    }
    let (_, _, w, h) = best.expect("RATIOS is non-empty");
    (w, h)
}

fn main() -> anyhow::Result<()> {
    // Disable the flame_core CUDA alloc pool — same rationale as prepare_klein:
    // dataset prep is one-pass, the pool leaks ~1 GB/sample under Tensor::drop
    // refcount patterns, and at ~75 samples the box OOMs. Must be set before
    // any flame_core call (OnceLock-cached on first read).
    if std::env::var_os("FLAME_ALLOC_POOL").is_none() {
        // SAFETY: single-threaded at this point (before main's first action).
        unsafe {
            std::env::set_var("FLAME_ALLOC_POOL", "0");
        }
    }
    env_logger::init();
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir)?;

    // ── Phase 6: validate flag values up front ────────────────────────────
    if args.bucket_alignment == 0 || args.bucket_alignment % 8 != 0 {
        anyhow::bail!(
            "--bucket-alignment must be a positive multiple of 8 (VAE patch size), got {}",
            args.bucket_alignment
        );
    }
    let crop_style = CropStyle::parse(&args.crop_style)?;
    let aug_cfg = eridiffusion_core::training::features::image_aug::AugConfig {
        flip: args.aug_flip,
        brightness: args.aug_brightness,
        contrast: args.aug_contrast,
    };
    if aug_cfg.is_active() {
        log::info!(
            "[image-aug] flip={} brightness={} contrast={} seed={}",
            aug_cfg.flip,
            aug_cfg.brightness,
            aug_cfg.contrast,
            args.aug_seed
        );
    }
    if !matches!(crop_style, CropStyle::Center) {
        log::info!(
            "[crop-style] {:?} (default-off path; output bytes will differ from `center`)",
            crop_style
        );
    }
    if args.bucket_alignment != 64 {
        log::info!("[bucket-alignment] {} (default 64)", args.bucket_alignment);
    }
    if args.caption_tag_shuffle {
        log::warn!(
            "[caption-tag-shuffle] enabled — Phase 6 records intent only. Cache files store encoded text; per-step shuffle requires Phase 7+ runtime re-encoder."
        );
    }
    if args.no_oklab {
        log::warn!(
            "[no-oklab] enabled — output omits `image_oklab`. A3 train_asymflow REQUIRES this tensor and will fail to load these caches."
        );
    }
    let filter_patterns: Vec<String> = if let Some(p) = args.caption_filter_list.as_ref() {
        let pats = eridiffusion_core::training::features::caption_aug::load_filter_list(p)?;
        log::info!(
            "[caption-filter-list] loaded {} pattern(s) from {}",
            pats.len(),
            p.display()
        );
        pats
    } else {
        Vec::new()
    };
    let skip_existing = args.skip_existing && !args.cache_invalidate;
    if args.cache_invalidate {
        log::info!("[cache-invalidate] re-encoding all samples (skip_existing forced false)");
    }
    let mut crop_rng = {
        use rand::SeedableRng;
        rand::rngs::StdRng::seed_from_u64(0xC0DEFACE)
    };
    flame_core::config::set_default_dtype(DType::BF16);
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    let device = flame_core::global_cuda_device();

    log::info!("[1/3] Loading Klein VAE encoder (128-ch packed latents)...");
    let dev = flame_core::Device::from(device.clone());
    let vae_weights = flame_core::serialization::load_file(&args.vae_ckpt, &device)?;
    let vae = KleinVaeEncoder::load(&vae_weights, &dev)?;
    drop(vae_weights);

    log::info!("[2/3] Loading Qwen3 text encoder (Klein extract layers [8,17,26])...");
    let qwen_weights = load_qwen3_weights(&args.qwen3, &device)?;
    let qcfg = Qwen3Encoder::config_from_weights(&qwen_weights)?;
    let joint_dim = qcfg.extract_layers.len() * qcfg.hidden_size;
    log::info!(
        "  Qwen3 hidden={} layers={} extract={:?} → text dim {}",
        qcfg.hidden_size,
        qcfg.num_layers,
        qcfg.extract_layers,
        joint_dim,
    );
    let qwen3 = Qwen3Encoder::new(qwen_weights, qcfg, device.clone());

    let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    log::info!(
        "[3/3] Encoding samples at {}² (Oklab: {})...",
        args.resolution,
        if args.no_oklab { "OFF" } else { "ON" }
    );
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
    log::info!("Found {} (image, caption) pairs", pairs.len());

    let mut written = 0usize;
    let mut skipped = 0usize;
    let t_start = std::time::Instant::now();
    for (idx, (img_path, txt_path)) in pairs.iter().enumerate() {
        if args.max_samples > 0 && written + skipped >= args.max_samples {
            break;
        }
        let hash = format!("{:x}", md5::compute(img_path.to_string_lossy().as_bytes()));
        let out_path = args.output_dir.join(format!("{hash}.safetensors"));
        if skip_existing && out_path.exists() {
            skipped += 1;
            continue;
        }

        // Caption filter early so a discarded sample doesn't cost a VAE+Qwen3 pass.
        let caption = std::fs::read_to_string(txt_path).unwrap_or_default();
        if !filter_patterns.is_empty()
            && !eridiffusion_core::training::features::caption_aug::caption_passes(
                &caption,
                &filter_patterns,
            )
        {
            log::debug!(
                "[filter] dropped {}: caption matched blocklist",
                img_path.display()
            );
            skipped += 1;
            continue;
        }

        let mut mask_opt: Option<image::GrayImage> = None;
        let img = match image::open(img_path) {
            Ok(src) => {
                let (sw, sh) = (src.width(), src.height());
                let (tw, th) = if args.bucketing {
                    pick_bucket(sw, sh, args.resolution, args.bucket_alignment)
                } else {
                    (args.resolution, args.resolution)
                };
                let src_aspect = sw as f32 / sh as f32;
                let bucket_aspect = tw as f32 / th as f32;
                let (scaled_w, scaled_h) = if src_aspect > bucket_aspect {
                    (((th as f32) * src_aspect).round() as u32, th)
                } else {
                    (tw, ((tw as f32) / src_aspect).round() as u32)
                };
                let mask_src: Option<image::GrayImage> = if src.color().has_alpha() {
                    let rgba = src.to_rgba8();
                    let mut a = image::GrayImage::new(rgba.width(), rgba.height());
                    for (x, y, p) in rgba.enumerate_pixels() {
                        a.put_pixel(x, y, image::Luma([p.0[3]]));
                    }
                    Some(a)
                } else {
                    let companion = img_path.with_extension("mask.png");
                    if companion.exists() {
                        image::open(&companion).ok().map(|i| i.to_luma8())
                    } else {
                        None
                    }
                };
                let resized =
                    src.resize_exact(scaled_w, scaled_h, image::imageops::FilterType::Lanczos3);
                let resized_rgb = resized.to_rgb8();
                let (rw, rh) = resized_rgb.dimensions();
                let (xoff, yoff) = crop_style.pick_offset(rw, rh, tw, th, &mut crop_rng);
                if idx == 0 {
                    log::info!(
                        "[bucket] src={sw}x{sh} → bucket={tw}x{th} (resized={rw}x{rh}, crop_off=({xoff},{yoff}))"
                    );
                }
                let cropped =
                    image::imageops::crop_imm(&resized_rgb, xoff, yoff, tw, th).to_image();
                if let Some(m) = mask_src {
                    let r = image::imageops::resize(
                        &m,
                        scaled_w,
                        scaled_h,
                        image::imageops::FilterType::Lanczos3,
                    );
                    mask_opt = Some(image::imageops::crop_imm(&r, xoff, yoff, tw, th).to_image());
                }
                image::DynamicImage::ImageRgb8(cropped).to_rgb32f()
            }
            Err(e) => {
                log::warn!("[{idx}] skipping {}: {e}", img_path.display());
                continue;
            }
        };
        let mut img = img;
        if aug_cfg.is_active() {
            use rand::SeedableRng;
            let mut aug_rng = rand::rngs::StdRng::seed_from_u64(args.aug_seed ^ idx as u64);
            eridiffusion_core::training::features::image_aug::apply_augs(
                &mut img,
                mask_opt.as_mut(),
                &aug_cfg,
                &mut aug_rng,
            );
        }
        let (w, h) = img.dimensions();
        // CHW transpose: image::pixels() is HWC interleaved [R,G,B,...] but
        // Tensor::from_vec(_, [1,3,H,W]) reads CHW. Same bug as prepare_klein —
        // bisect 2026-05-05 caught a 0.85→0.96 std drop when this is wrong.
        let (wu, hu) = (w as usize, h as usize);
        let mut pixels = vec![0f32; 3 * hu * wu];
        for (x, y, p) in img.enumerate_pixels() {
            let (xu, yu) = (x as usize, y as usize);
            for c in 0..3 {
                pixels[c * hu * wu + yu * wu + xu] = p.0[c] * 2.0 - 1.0;
            }
        }

        // Oklab encode happens BEFORE the GPU tensor copy so we can reuse the
        // planar `pixels` buffer (which is already `(3, H*W)` channel-major,
        // matching the layout `oklab::encode_planar` expects). The Oklab
        // pipeline is CPU-only, allocation-free, and deterministic — no
        // dependency on `device` or autograd.
        let oklab_tensor = if args.no_oklab {
            None
        } else {
            let mut ok = vec![0f32; pixels.len()];
            oklab::encode_planar(&pixels, &mut ok);
            // BF16 [1, 3, H, W] — same layout the trainer expects for the
            // teacher target. Note: oklab values are roughly N(0, 1) after
            // affine normalize, so BF16's reduced mantissa is fine.
            let t = Tensor::from_vec(ok, Shape::from_dims(&[1, 3, hu, wu]), device.clone())?
                .to_dtype(DType::BF16)?;
            Some(t)
        };

        let img_t = Tensor::from_vec(pixels, Shape::from_dims(&[1, 3, hu, wu]), device.clone())?
            .to_dtype(DType::BF16)?;
        // KleinVaeEncoder.encode handles posterior.mode + patchify + BN → [B, 128, H/16, W/16].
        let latent = vae.encode(&img_t)?;

        let prompt = format!(
            "{KLEIN_TEMPLATE_PRE}{}{KLEIN_TEMPLATE_POST}",
            caption.trim()
        );
        let enc = tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        let valid_len = ids.len().min(TXT_PAD_LEN);
        ids.resize(TXT_PAD_LEN, PAD_TOKEN_ID);
        let text_hidden = qwen3.encode(&ids)?; // [1, TXT_PAD_LEN, joint_dim]

        let mut mask_data = vec![0.0f32; TXT_PAD_LEN];
        for slot in mask_data.iter_mut().take(valid_len) {
            *slot = 1.0;
        }
        let text_mask = Tensor::from_vec(
            mask_data,
            Shape::from_dims(&[1, TXT_PAD_LEN]),
            device.clone(),
        )?;

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        if let Some(m) = mask_opt {
            let dims = latent.shape().dims();
            let lat_h = dims[2];
            let lat_w = dims[3];
            let down = image::imageops::resize(
                &m,
                lat_w as u32,
                lat_h as u32,
                image::imageops::FilterType::Triangle,
            );
            let mut mp = vec![0f32; lat_h * lat_w];
            for (x, y, p) in down.enumerate_pixels() {
                mp[y as usize * lat_w + x as usize] = p.0[0] as f32 / 255.0;
            }
            let mask_t =
                Tensor::from_vec(mp, Shape::from_dims(&[1, 1, lat_h, lat_w]), device.clone())?
                    .to_dtype(DType::BF16)?;
            tensors.insert("latent_mask".into(), mask_t);
        }
        tensors.insert("latent".into(), latent);
        if let Some(t) = oklab_tensor {
            tensors.insert("image_oklab".into(), t);
        }
        tensors.insert("text_embedding".into(), text_hidden);
        tensors.insert("text_mask".into(), text_mask);
        save_file(&tensors, &out_path)?;

        drop(tensors);
        drop(img_t);

        written += 1;

        if written % 10 == 0 || written == 1 {
            let elapsed = t_start.elapsed().as_secs_f32();
            let rss_kb: usize = std::fs::read_to_string("/proc/self/status")
                .ok()
                .and_then(|s| {
                    s.lines()
                        .find(|l| l.starts_with("VmRSS:"))
                        .and_then(|l| l.split_whitespace().nth(1))
                        .and_then(|n| n.parse().ok())
                })
                .unwrap_or(0);
            log::info!(
                "  cached {written} (skipped {skipped}) — {:.2}/s — RSS {:.1} GB",
                written as f32 / elapsed.max(1e-3),
                rss_kb as f32 / 1024.0 / 1024.0
            );
        }
    }

    log::info!(
        "Done: wrote {written}, skipped {skipped}, total {} in {:.1}s",
        pairs.len(),
        t_start.elapsed().as_secs_f32()
    );
    Ok(())
}

fn load_qwen3_weights(
    path: &std::path::Path,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> flame_core::Result<HashMap<String, Tensor>> {
    if path.is_file() {
        return flame_core::serialization::load_file(path, device);
    }
    let mut all = HashMap::new();
    for entry in std::fs::read_dir(path)
        .map_err(|e| flame_core::Error::Io(format!("read_dir {}: {e}", path.display())))?
    {
        let p = entry
            .map_err(|e| flame_core::Error::Io(format!("entry: {e}")))?
            .path();
        if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            let part = flame_core::serialization::load_file(&p, device)?;
            all.extend(part);
        }
    }
    Ok(all)
}
