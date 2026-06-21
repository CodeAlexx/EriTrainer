//! prepare_ideogram — image + caption → cached latents + Qwen3-VL embeddings for
//! Ideogram-4 LoRA training. Reuses EDv2's `KleinVaeEncoder` (Flux2 VAE) and
//! `Qwen3Encoder`, plus the two parity-verified Ideogram packing adapters:
//!   - VAE: klein patchify [ae,ph,pw] → Ideogram [ph,pw,ae] channel reorder.
//!   - Text: Qwen3-VL 13 taps [0,3,..,35], klein tap-major → Ideogram hidden-major.
//! Caption is chat-templated like the Ideogram pipeline; text is variable-length
//! (no padding), matching `IdeogramModel.encode_text`.
//!
//! Cache per sample (`<hash>.safetensors`):
//!   latent          BF16 [1, 128, H/16, W/16]   (reordered, BN-normalised)
//!   text_embedding  BF16 [1, L, 53248]          (de-interleaved 13-tap)
//!   text_mask       F32  [1, L]                 (all-ones; real tokens)

use std::collections::HashMap;
use std::path::PathBuf;

use clap::Parser;
use eridiffusion_core::encoders::{qwen3::Qwen3Config, qwen3::Qwen3Encoder, vae::KleinVaeEncoder};
use flame_core::{serialization::save_file, DType, Shape, Tensor};

const TAPS: [usize; 13] = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35];
const PACKED_CH: usize = 128;
const Z_CH: usize = 32; // patch 2x2 over ae_ch=32 → 128 packed

#[derive(Parser)]
#[command(name = "prepare_ideogram", about = "Cache Ideogram-4 training latents+embeddings")]
struct Args {
    #[arg(long)]
    input_dir: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    /// Ideogram VAE safetensors (vae/diffusion_pytorch_model.safetensors).
    #[arg(long)]
    vae_ckpt: PathBuf,
    /// Ideogram text_encoder safetensors (text_encoder/model.safetensors).
    #[arg(long)]
    text_encoder: PathBuf,
    /// tokenizer.json.
    #[arg(long)]
    tokenizer_path: PathBuf,
    #[arg(long, default_value = "256")]
    resolution: u32,
}

/// klein patchify is [ae,ph,pw] (idx=ae*4+ph*2+pw); Ideogram is [ph,pw,ae]
/// (idx=ph*64+pw*32+ae). Reorder the 128 packed channels.
fn reorder_vae_channels(latent: &Tensor, device: &std::sync::Arc<flame_core::CudaDevice>) -> anyhow::Result<Tensor> {
    let dims = latent.shape().dims().to_vec();
    let spatial = dims[2] * dims[3];
    let v = latent.to_dtype(DType::F32)?.to_vec_f32()?;
    let mut out = vec![0f32; v.len()];
    for idx_ideo in 0..PACKED_CH {
        let ph = idx_ideo / 64;
        let pw = (idx_ideo % 64) / Z_CH;
        let ae = idx_ideo % Z_CH;
        let idx_klein = ae * 4 + ph * 2 + pw;
        out[idx_ideo * spatial..(idx_ideo + 1) * spatial]
            .copy_from_slice(&v[idx_klein * spatial..(idx_klein + 1) * spatial]);
    }
    Ok(Tensor::from_vec(out, Shape::from_dims(&dims), device.clone())?.to_dtype(DType::BF16)?)
}

/// klein stacks tap-major (idx=tap*H+h); Ideogram is hidden-major (idx=h*ntaps+tap).
fn deinterleave_taps(emb: &Tensor, l: usize, od: usize) -> anyhow::Result<Tensor> {
    let ntaps = TAPS.len();
    let hsz = od / ntaps;
    Ok(emb
        .reshape(&[1, l, ntaps, hsz])?
        .permute(&[0, 1, 3, 2])?
        .contiguous()?
        .reshape(&[1, l, od])?)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir)?;
    let device = flame_core::global_cuda_device();

    log::info!("[1/3] Loading Ideogram VAE encoder…");
    let vae_w = flame_core::serialization::load_file(&args.vae_ckpt, &device)?;
    let dev = flame_core::Device::from(device.clone());
    let vae = KleinVaeEncoder::load(&vae_w, &dev)?;

    log::info!("[2/3] Loading Qwen3-VL text encoder (13 taps)…");
    let raw = flame_core::serialization::load_file(&args.text_encoder, &device)?;
    // Qwen3-VL nests the LM under `language_model.` (+ a `visual.` tower).
    let mut qw = HashMap::with_capacity(raw.len());
    for (k, t) in raw {
        if let Some(rest) = k.strip_prefix("language_model.") {
            qw.insert(format!("model.{rest}"), t);
        }
    }
    let mut qcfg = Qwen3Config::qwen3_vl_text();
    qcfg.extract_layers = TAPS.to_vec();
    let qwen = Qwen3Encoder::new(qw, qcfg, device.clone());
    let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    // gather (image, caption) pairs — caption is `.json` (Ideogram structured) or `.txt`.
    let mut pairs: Vec<(PathBuf, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(&args.input_dir)? {
        let p = entry?.path();
        if matches!(p.extension().and_then(|e| e.to_str()), Some("jpg" | "jpeg" | "png")) {
            let stem = p.file_stem().unwrap().to_string_lossy().to_string();
            let cj = args.input_dir.join(format!("{stem}.json"));
            let ct = args.input_dir.join(format!("{stem}.txt"));
            pairs.push((p.clone(), if cj.exists() { cj } else { ct }));
        }
    }
    log::info!("[3/3] {} pairs → encoding", pairs.len());

    // Caching is pure inference — disable autograd so the VAE + Qwen3-VL encode
    // ops don't accumulate on the global tape (saved activations across images
    // OOM'd after ~14 without this; pool-clear alone didn't help). Mirrors prepare_klein.
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();

    let res = args.resolution;
    let mut written = 0usize;
    for (img_path, cap_path) in &pairs {
        // image → [1,3,res,res] in [-1,1], bf16 (BICUBIC ≈ CatmullRom).
        let img = image::open(img_path)?
            .resize_exact(res, res, image::imageops::FilterType::CatmullRom)
            .to_rgb8();
        let (wu, hu) = (res as usize, res as usize);
        let mut pixels = vec![0f32; 3 * hu * wu];
        for (x, y, p) in img.enumerate_pixels() {
            for c in 0..3 {
                pixels[c * hu * wu + y as usize * wu + x as usize] = p.0[c] as f32 / 255.0 * 2.0 - 1.0;
            }
        }
        let img_t = Tensor::from_vec(pixels, Shape::from_dims(&[1, 3, hu, wu]), device.clone())?
            .to_dtype(DType::BF16)?;
        let latent = reorder_vae_channels(&vae.encode(&img_t)?, &device)?;

        // caption → chat template → tokens (no special tokens, no padding).
        let raw_cap = std::fs::read_to_string(cap_path).unwrap_or_default();
        let rendered = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            raw_cap.trim()
        );
        let enc = tokenizer
            .encode(rendered.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        let l = ids.len();
        let text_embedding = deinterleave_taps(&qwen.encode(&ids)?, l, qwen.output_dim())?;
        let text_mask = Tensor::from_vec(vec![1.0f32; l], Shape::from_dims(&[1, l]), device.clone())?;

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert("latent".into(), latent);
        tensors.insert("text_embedding".into(), text_embedding);
        tensors.insert("text_mask".into(), text_mask);
        let stem = img_path.file_stem().unwrap().to_string_lossy();
        save_file(&tensors, &args.output_dir.join(format!("{stem}.safetensors")))?;
        written += 1;
        // Free per-image GPU tensors + return the flame mempool's cached blocks
        // to the driver. Without this the Qwen3-VL encode activations accumulate
        // in the infinite-caching pool and OOM after ~15 images (measured).
        drop(tensors);
        drop(img_t);
        flame_core::cuda_alloc_pool::clear_pool_cache();
        log::info!("  cached {stem} (L={l})");
    }
    log::info!("Done: wrote {written}");
    Ok(())
}
