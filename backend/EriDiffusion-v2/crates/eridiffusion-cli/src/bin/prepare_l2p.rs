//! prepare_l2p — image+caption → cached pixel+embedding safetensors for L2P
//! LoRA training.
//!
//! Mirrors `prepare_zimage` but skips the VAE step entirely: L2P is
//! pixel-space, so the target tensor is the normalized RGB pixel grid, not
//! a latent. Captions are encoded with the same Qwen3 4B penultimate
//! hidden state extraction (mode="zimage", layer 34) that Z-Image uses,
//! because L2P inherits Z-Image-Turbo's text encoder verbatim.
//!
//! Output per sample (one safetensors file under `--output`):
//!   - `pixel`     : BF16 [3, H, W]            — normalized to [-1, 1]
//!   - `cap_feats` : BF16 [1, seq_real, 2560]  — Qwen3-4B penultimate hidden,
//!                                              pad positions stripped
//!
//! Critical env vars set at `main()` entry:
//!   - `FLAME_ALLOC_POOL=0`   — per CONTEXT.md prepare_* rule; pool leaks
//!                              ~1 GB/sample and OOMs around sample 75.

use clap::Parser;
use eridiffusion_core::encoders::qwen3::Qwen3Encoder;
use flame_core::{serialization::save_file, DType, Shape, Tensor};
use std::collections::HashMap;
use std::path::PathBuf;

// Chat-template + thinking mode match prepare_zimage / Z-Image's tokenizer
// expectations. Verified at prepare_zimage:25-27.
const ZIMAGE_TEMPLATE_PRE: &str = "<|im_start|>user\n";
const ZIMAGE_TEMPLATE_POST: &str = "<|im_end|>\n<|im_start|>assistant\n";
const PAD_TOKEN_ID: i32 = 151643;
const TXT_PAD_LEN: usize = 512;
/// Z-Image canonical extract layer: layer 34 of Qwen3-4B (penultimate
/// hidden, matches musubi/upstream). Validated at prepare_zimage:91-98.
const QWEN3_EXTRACT_LAYER: usize = 34;

#[derive(Parser)]
struct Args {
    /// Folder with paired `*.jpg|png|webp|bmp` + matching `*.txt` captions.
    #[arg(long)]
    dataset: PathBuf,
    /// Per-sample safetensors output directory.
    #[arg(long)]
    output: PathBuf,
    /// Square resolution. L2P trains at 512 in this preset (1024² is too
    /// tight on 24 GB activations).
    #[arg(long, default_value = "512")]
    resolution: u32,
    /// Qwen3-4B weights (single .safetensors file OR a shard directory).
    /// Default matches Z-Image / L2P's bundled text encoder path.
    #[arg(
        long,
        default_value = "/home/alex/.serenity/models/text_encoders/qwen_3_4b.safetensors"
    )]
    encoder: PathBuf,
    /// Tokenizer.json (Qwen3 BPE). L2P + Z-Image share the same one.
    #[arg(
        long,
        default_value = "/home/alex/.serenity/models/zimage_base/tokenizer/tokenizer.json"
    )]
    tokenizer: PathBuf,
    /// Skip already-cached samples (filename hash collides → resume).
    #[arg(long, default_value_t = true)]
    skip_existing: bool,
    /// Cap the run for debugging (`0` = no cap).
    #[arg(long, default_value_t = 0)]
    max_samples: usize,
}

fn main() -> anyhow::Result<()> {
    // CRITICAL: disable the flame-core CUDA alloc pool BEFORE any flame-core
    // op runs. Per CONTEXT.md "prepare_* binaries": the pool leaks ~1 GB
    // per sample during a one-pass dataset prep, OOM-killing the box
    // around sample 75 on 62 GB host RAM. Setting the env var here is the
    // single-threaded canonical pattern matching prepare_zimage:67-72.
    if std::env::var_os("FLAME_ALLOC_POOL").is_none() {
        // SAFETY: single-threaded at this point.
        unsafe {
            std::env::set_var("FLAME_ALLOC_POOL", "0");
        }
    }
    env_logger::init();
    let args = Args::parse();
    std::fs::create_dir_all(&args.output)?;
    flame_core::config::set_default_dtype(DType::BF16);
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    let device = flame_core::global_cuda_device();

    log::info!("[1/2] Loading Qwen3-4B text encoder (layer {QWEN3_EXTRACT_LAYER})...");
    let qwen_weights = load_qwen3_weights(&args.encoder, &device)?;
    let mut qcfg = Qwen3Encoder::config_from_weights(&qwen_weights)?;
    qcfg.extract_layers = vec![QWEN3_EXTRACT_LAYER];
    log::info!(
        "  config: hidden={} layers={} heads={} extract={:?}",
        qcfg.hidden_size,
        qcfg.num_layers,
        qcfg.num_heads,
        qcfg.extract_layers
    );
    let qwen3 = Qwen3Encoder::new(qwen_weights, qcfg, device.clone());

    let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    log::info!("[2/2] Encoding samples from {}...", args.dataset.display());
    let mut pairs = Vec::new();
    for entry in std::fs::read_dir(&args.dataset)? {
        let p = entry?.path();
        if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
            if matches!(
                ext.to_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "webp" | "bmp"
            ) {
                let stem = p.file_stem().unwrap().to_str().unwrap();
                let txt = args.dataset.join(format!("{stem}.txt"));
                if !txt.exists() {
                    log::warn!("missing caption {} for {} — skipping", txt.display(), p.display());
                    continue;
                }
                pairs.push((p.clone(), txt));
            }
        }
    }
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    log::info!("Found {} (image, caption) pairs", pairs.len());

    let mut written = 0usize;
    let mut skipped = 0usize;
    let t_start = std::time::Instant::now();
    for (idx, (img_path, txt_path)) in pairs.iter().enumerate() {
        if args.max_samples > 0 && written + skipped >= args.max_samples {
            break;
        }
        let stem = img_path.file_stem().unwrap().to_string_lossy();
        let out_path = args.output.join(format!("{stem}.safetensors"));
        if args.skip_existing && out_path.exists() {
            skipped += 1;
            continue;
        }

        // Load + resize. Lanczos3 matches prepare_zimage.
        let img = match image::open(img_path) {
            Ok(i) => i
                .resize_exact(
                    args.resolution,
                    args.resolution,
                    image::imageops::FilterType::Lanczos3,
                )
                .to_rgb32f(),
            Err(e) => {
                log::warn!("[{idx}] skipping {}: {e}", img_path.display());
                continue;
            }
        };
        let (w, h) = img.dimensions();
        let (wu, hu) = (w as usize, h as usize);

        // HWC (image crate native) → CHW (model expects), normalize to
        // [-1, 1]. The explicit per-channel rewrite matches
        // prepare_zimage:184-191 — the pre-2026-05-05 channel-scramble bug
        // dodge per CONTEXT.md / feedback_prepare_bins_chw_transpose.
        let mut pixels = vec![0f32; 3 * hu * wu];
        for (x, y, p) in img.enumerate_pixels() {
            let (xu, yu) = (x as usize, y as usize);
            for c in 0..3 {
                pixels[c * hu * wu + yu * wu + xu] = p.0[c] * 2.0 - 1.0;
            }
        }
        // L2P trainer reshapes back to [1, 3, H, W] inside the loop; save
        // the 3D form here. BF16 — matches the L2P resident dtype.
        let pix_t = Tensor::from_vec(
            pixels,
            Shape::from_dims(&[3, hu, wu]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        // Caption → Qwen3 penultimate hidden state.
        let caption = std::fs::read_to_string(txt_path).unwrap_or_default();
        let prompt = format!(
            "{ZIMAGE_TEMPLATE_PRE}{}{ZIMAGE_TEMPLATE_POST}",
            caption.trim()
        );
        let enc = tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        let valid_len = ids.len().min(TXT_PAD_LEN);
        ids.resize(TXT_PAD_LEN, PAD_TOKEN_ID);
        // Full [1, TXT_PAD_LEN, 2560] from the encoder, then strip pad
        // positions to keep cap_feats compact. L2P trainer feeds variable-
        // length cap_feats (no separate mask path in the inference forward
        // — `pad_to_multiple` handles padding internally with the learned
        // cap_pad_token).
        let text_hidden = qwen3.encode(&ids)?.to_dtype(DType::BF16)?;
        let cap_feats = text_hidden.narrow(1, 0, valid_len)?.contiguous()?;

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert("pixel".into(), pix_t);
        tensors.insert("cap_feats".into(), cap_feats);
        save_file(&tensors, &out_path)?;
        written += 1;

        eridiffusion_core::training::progress::log_step(
            "L2P-prep",
            idx,
            pairs.len(),
            pairs.len(),
            1,
            0.0,
            0.0,
            0.0,
            t_start,
            None,
        );
    }

    log::info!(
        "Done: wrote {written}, skipped {skipped}, total {} in {:.1}s",
        pairs.len(),
        t_start.elapsed().as_secs_f32()
    );
    Ok(())
}

/// Load Qwen3 weights from either a single safetensors file or a sharded
/// directory.
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
