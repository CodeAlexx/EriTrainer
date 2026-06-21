//! sample_qwenimage_edit - Qwen-Image-Edit Stage 2/3 sampler.
//!
//! This consumes a pre-encoded Stage-1 safetensors file:
//!
//!   cond, uncond, image_latents, image_h, image_w
//!
//! The Rust path then runs the Qwen Edit transformer with target/reference
//! multi-region latents and decodes the target latent through the Qwen VAE.

use clap::Parser;
use eridiffusion_cli::trainer_common;
use eridiffusion_core::encoders::wan21_decoder::Wan21VaeDecoder;
use eridiffusion_core::lycoris::{LycorisAlgo, LycorisBundleConfig};
use eridiffusion_core::models::qwenimage::{
    self as qwen_model, QwenImageLoraBundle, OUT_CHANNELS,
};
use eridiffusion_core::models::QwenImageTrainingModel;
use flame_core::{autograd::AutogradContext, DType, Result as FlameResult, Shape, Tensor};
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const VAE_SCALE_FACTOR: usize = 8;
const PATCH_SIZE: usize = 2;
const PACKED_CHANNELS: usize = OUT_CHANNELS * PATCH_SIZE * PATCH_SIZE;

#[derive(Parser)]
struct Args {
    /// Stage-1 Qwen Edit embeddings safetensors.
    #[arg(long, default_value = "output/qwenimage_edit_embeds.safetensors")]
    embeds: PathBuf,
    /// Qwen-Image-Edit transformer dir or single safetensors checkpoint.
    #[arg(long)]
    model: PathBuf,
    /// qwen_image_vae.safetensors.
    #[arg(long)]
    vae_path: PathBuf,
    /// Output PNG path.
    #[arg(long, default_value = "output/qwenimage_edit.png")]
    output: PathBuf,
    /// Optional packed target latent safetensors output.
    #[arg(long)]
    latents_output: Option<PathBuf>,
    #[arg(long, default_value = "50")]
    steps: usize,
    /// True CFG scale. Set to 1.0 for no uncond forward.
    #[arg(long, default_value = "4.0")]
    cfg: f32,
    #[arg(long, default_value = "42")]
    seed: u64,
    /// Set QWEN_BLOCK_STREAMING=1 before model load.
    #[arg(long, default_value_t = false)]
    streaming: bool,
    /// Use legacy edit modulation instead of Qwen-Image-Edit-2511 zero_cond_t.
    #[arg(long, default_value_t = false)]
    no_zero_cond_t: bool,
    /// Optional trained LoRA/LyCORIS safetensors.
    #[arg(long)]
    lora_path: Option<PathBuf>,
    #[arg(long, default_value = "16")]
    lora_rank: usize,
    #[arg(long, default_value = "16.0")]
    lora_alpha: f32,
    /// Adapter algo for --lora-path: lora | locon | loha | lokr.
    #[arg(long, default_value = "lora")]
    algo: String,
    /// LoKr Kronecker split factor.
    #[arg(long, default_value_t = 16)]
    lokr_factor: i32,
    /// OFT block size.
    #[arg(long, default_value_t = 32)]
    oft_block_size: usize,
    /// OFT Cayley-Neumann series term count.
    #[arg(long, default_value_t = 5)]
    oft_neumann_terms: usize,
    /// LoCon / LoHa / LoKr Tucker decomposition flag.
    #[arg(long, default_value_t = false)]
    use_tucker: bool,
    /// LoKr only: factorize both W1 and W2.
    #[arg(long, default_value_t = false)]
    decompose_both: bool,
    /// Enable DoRA reconstruction when the sampled checkpoint used DoRA.
    #[arg(long, default_value_t = false)]
    dora: bool,
    #[arg(long, default_value_t = true)]
    dora_wd_on_out: bool,
    #[arg(long, default_value_t = 1e-6)]
    dora_eps: f32,
    /// Conv rank override. Inert for current Qwen linear targets.
    #[arg(long, default_value_t = 0)]
    conv_rank: usize,
    /// Conv alpha override. Inert for current Qwen linear targets.
    #[arg(long, default_value_t = 0.0)]
    conv_alpha: f32,
    /// Optional SimpleTuner-style LyCORIS preset used during training.
    #[arg(long)]
    lycoris_config: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    trainer_common::init_logging();
    let args = Args::parse();
    if args.steps == 0 {
        anyhow::bail!("--steps must be > 0");
    }
    if args.streaming {
        std::env::set_var("QWEN_BLOCK_STREAMING", "1");
    }

    let device = flame_core::global_cuda_device();
    let _no_grad = AutogradContext::no_grad();
    flame_core::config::set_default_dtype(DType::BF16);
    let prev_ckpt = std::env::var("FLAME_CHECKPOINT").ok();
    std::env::set_var("FLAME_CHECKPOINT", "0");

    let result = run(args, &device);
    if let Some(v) = prev_ckpt {
        std::env::set_var("FLAME_CHECKPOINT", v);
    } else {
        std::env::set_var("FLAME_CHECKPOINT", "1");
    }
    result
}

fn run(
    args: Args,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<()> {
    let algo = parse_adapter_algo(&args.algo)?;
    let lyc_config = LycorisBundleConfig {
        algo,
        rank: args.lora_rank,
        alpha: args.lora_alpha,
        factor: args.lokr_factor,
        conv_rank: args.conv_rank,
        conv_alpha: args.conv_alpha,
        block_size: args.oft_block_size,
        neumann_terms: args.oft_neumann_terms,
        use_tucker: args.use_tucker,
        decompose_both: args.decompose_both,
        use_scalar: false,
        dora: args.dora,
        dora_wd_on_out: args.dora_wd_on_out,
        dora_eps: args.dora_eps,
        ..LycorisBundleConfig::default()
    }
    .with_optional_lycoris_config_file(args.lycoris_config.as_deref())?;

    log::info!("[1/4] Loading cached Qwen Edit embeddings...");
    let tensors = flame_core::serialization::load_file(&args.embeds, device)?;
    let cond = tensor_bf16(&tensors, "cond")?;
    let uncond = tensor_bf16(&tensors, "uncond")?;
    let image_latents = tensor_bf16(&tensors, "image_latents")?;
    let height = tensor_scalar_usize(&tensors, "image_h")?;
    let width = tensor_scalar_usize(&tensors, "image_w")?;
    drop(tensors);

    if height % (VAE_SCALE_FACTOR * PATCH_SIZE) != 0
        || width % (VAE_SCALE_FACTOR * PATCH_SIZE) != 0
    {
        anyhow::bail!(
            "image dimensions {}x{} must be multiples of {}",
            width,
            height,
            VAE_SCALE_FACTOR * PATCH_SIZE
        );
    }

    let latent_h = height / VAE_SCALE_FACTOR;
    let latent_w = width / VAE_SCALE_FACTOR;
    let h_patched = latent_h / PATCH_SIZE;
    let w_patched = latent_w / PATCH_SIZE;
    let target_seq_len = h_patched * w_patched;
    validate_embeddings(&cond, &uncond, &image_latents, target_seq_len)?;

    log::info!(
        "  cond={:?} uncond={:?} reference={:?} target={}x{} seq={}",
        cond.shape().dims(),
        uncond.shape().dims(),
        image_latents.shape().dims(),
        width,
        height,
        target_seq_len
    );

    log::info!("[2/4] Loading Qwen-Image-Edit transformer...");
    let mut model = QwenImageTrainingModel::load(
        &args.model,
        args.lora_rank,
        args.lora_alpha,
        false,
        device.clone(),
        args.seed,
    )?;
    if algo != LycorisAlgo::None {
        if matches!(algo, LycorisAlgo::Full | LycorisAlgo::Oft) {
            anyhow::bail!(
                "--algo {} is not sample-safe in the Qwen residual adapter path yet",
                algo.as_str()
            );
        }
        model.bundle = QwenImageLoraBundle::new_with_config(&lyc_config, device.clone(), args.seed)
            .map_err(|e| anyhow::anyhow!("LyCORIS bundle construction: {e}"))?;
        log::info!(
            "  using LyCORIS algo={} rank={} alpha={}",
            algo.as_str(),
            lyc_config.rank,
            lyc_config.alpha
        );
    }
    if let Some(lp) = &args.lora_path {
        model.bundle.load(lp, device)?;
        log::info!("  loaded adapter from {}", lp.display());
    }

    log::info!(
        "[3/4] Denoising edit target ({} steps, cfg={}, zero_cond_t={})...",
        args.steps,
        args.cfg,
        !args.no_zero_cond_t
    );
    let mut latents = seeded_noise_packed(
        args.seed,
        latent_h,
        latent_w,
        h_patched,
        w_patched,
        device,
    )?;
    let sigmas = qwen_sigmas(args.steps, target_seq_len);
    let regions = vec![(h_patched, w_patched), (h_patched, w_patched)];
    let t_denoise = std::time::Instant::now();

    for step in 0..args.steps {
        let sigma_curr = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let dt = sigma_next - sigma_curr;
        let t_vec = Tensor::from_vec(vec![sigma_curr], Shape::from_dims(&[1]), device.clone())?
            .to_dtype(DType::BF16)?;

        let cond_pred = if args.no_zero_cond_t {
            model.forward_edit(&latents, &image_latents, &t_vec, &cond, &regions)?
        } else {
            model.forward_edit_2511(&latents, &image_latents, &t_vec, &cond, &regions)?
        };

        let noise_pred = if args.cfg > 1.0 {
            let uncond_pred = if args.no_zero_cond_t {
                model.forward_edit(&latents, &image_latents, &t_vec, &uncond, &regions)?
            } else {
                model.forward_edit_2511(&latents, &image_latents, &t_vec, &uncond, &regions)?
            };
            let diff = cond_pred.sub(&uncond_pred)?;
            let scaled = diff.mul_scalar(args.cfg)?;
            let comb = uncond_pred.add(&scaled)?;
            norm_rescale_cfg(&cond_pred, &comb).unwrap_or(comb)
        } else {
            cond_pred
        };

        latents = latents.add(&noise_pred.mul_scalar(dt)?)?;

        if (step + 1) % 5 == 0 || step == 0 || step + 1 == args.steps {
            log::info!(
                "  step {}/{} sigma={:.4} ({:.1}s)",
                step + 1,
                args.steps,
                sigma_curr,
                t_denoise.elapsed().as_secs_f32()
            );
        }
    }
    log::info!("  denoised in {:.1}s", t_denoise.elapsed().as_secs_f32());

    if let Some(path) = &args.latents_output {
        save_latents(&latents, height, width, path, device)?;
        log::info!("  saved packed latents {}", path.display());
    }

    log::info!("[4/4] Decoding Qwen VAE...");
    let latent = qwen_model::unpack_latents(&latents, latent_h, latent_w)?;
    drop(model);
    drop(cond);
    drop(uncond);
    drop(image_latents);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::trim_cuda_mempool(0);

    let vae = Wan21VaeDecoder::from_safetensors(&args.vae_path.to_string_lossy(), device)?;
    let rgb = vae.decode_image_normalized(&latent)?;
    save_tensor_as_png(&rgb, &args.output)?;
    log::info!("  saved {}", args.output.display());
    Ok(())
}

fn tensor_bf16(tensors: &HashMap<String, Tensor>, key: &str) -> anyhow::Result<Tensor> {
    let t = tensors
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("missing tensor `{key}`"))?
        .clone();
    if t.dtype() == DType::BF16 {
        Ok(t)
    } else {
        Ok(t.to_dtype(DType::BF16)?)
    }
}

fn tensor_scalar_usize(tensors: &HashMap<String, Tensor>, key: &str) -> anyhow::Result<usize> {
    let values = tensors
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("missing tensor `{key}`"))?
        .clone()
        .to_dtype(DType::F32)?
        .to_vec()?;
    let value = *values
        .first()
        .ok_or_else(|| anyhow::anyhow!("tensor `{key}` is empty"))?;
    if !value.is_finite() || value < 1.0 || value > 8192.0 {
        anyhow::bail!("tensor `{key}` out of valid range [1, 8192]: {value}");
    }
    Ok(value as usize)
}

fn validate_embeddings(
    cond: &Tensor,
    uncond: &Tensor,
    image_latents: &Tensor,
    target_seq_len: usize,
) -> anyhow::Result<()> {
    for (name, t) in [("cond", cond), ("uncond", uncond)] {
        let d = t.shape().dims();
        if d.len() != 3 || d[0] != 1 || d[2] != 3584 {
            anyhow::bail!("{name} shape {:?}, expected [1, seq, 3584]", d);
        }
    }
    let d = image_latents.shape().dims();
    if d.len() != 3 || d[0] != 1 || d[1] != target_seq_len || d[2] != PACKED_CHANNELS {
        anyhow::bail!(
            "image_latents shape {:?}, expected [1, {}, {}]",
            d,
            target_seq_len,
            PACKED_CHANNELS
        );
    }
    Ok(())
}

fn seeded_noise_packed(
    seed: u64,
    latent_h: usize,
    latent_w: usize,
    h_patched: usize,
    w_patched: usize,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> FlameResult<Tensor> {
    let numel = OUT_CHANNELS * latent_h * latent_w;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(numel);
    for _ in 0..(numel / 2) {
        let u1: f32 = rng.gen::<f32>().max(1e-10);
        let u2: f32 = rng.gen::<f32>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        data.push(r * theta.cos());
        data.push(r * theta.sin());
    }
    if numel % 2 == 1 {
        let u1: f32 = rng.gen::<f32>().max(1e-10);
        let u2: f32 = rng.gen::<f32>();
        data.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos());
    }
    let unpacked = Tensor::from_vec(
        data,
        Shape::from_dims(&[1, OUT_CHANNELS, latent_h, latent_w]),
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let packed = qwen_model::pack_latents(&unpacked)?;
    let d = packed.shape().dims();
    debug_assert_eq!(d, &[1, h_patched * w_patched, PACKED_CHANNELS]);
    Ok(packed)
}

fn qwen_sigmas(num_steps: usize, seq_len: usize) -> Vec<f32> {
    let base_shift: f32 = 0.5;
    let max_shift: f32 = 0.9;
    let base_seq_len: f32 = 256.0;
    let max_seq_len_shift: f32 = 8192.0;
    let shift_terminal: f32 = 0.02;

    let m = (max_shift - base_shift) / (max_seq_len_shift - base_seq_len);
    let bb = base_shift - m * base_seq_len;
    let mu = (seq_len as f32) * m + bb;
    let exp_mu = mu.exp();

    let mut sigmas: Vec<f32> = (0..num_steps)
        .map(|i| {
            let t = i as f32 / (num_steps - 1).max(1) as f32;
            1.0 - t * (1.0 - 1.0 / num_steps as f32)
        })
        .collect();
    for s in sigmas.iter_mut() {
        let denom = exp_mu + (1.0 / *s - 1.0);
        *s = exp_mu / denom;
    }
    let last = *sigmas.last().unwrap();
    let one_minus_last = 1.0 - last;
    if one_minus_last.abs() > 1e-12 {
        let scale = one_minus_last / (1.0 - shift_terminal);
        for s in sigmas.iter_mut() {
            let o = 1.0 - *s;
            *s = 1.0 - o / scale;
        }
    }
    sigmas.push(0.0);
    sigmas
}

fn norm_rescale_cfg(cond: &Tensor, comb: &Tensor) -> Option<Tensor> {
    let last_dim = cond.shape().dims().len().saturating_sub(1);
    if last_dim == 0 {
        return None;
    }
    let cond_sq = cond.mul(cond).ok()?;
    let comb_sq = comb.mul(comb).ok()?;
    let cond_sum = cond_sq.sum_dim_keepdim(last_dim).ok()?;
    let comb_sum = comb_sq.sum_dim_keepdim(last_dim).ok()?;
    let cond_norm = cond_sum.sqrt().ok()?;
    let comb_norm = comb_sum.sqrt().ok()?;
    let ratio = cond_norm.div(&comb_norm).ok()?;
    comb.mul(&ratio).ok()
}

fn save_latents(
    latents: &Tensor,
    height: usize,
    width: usize,
    path: &Path,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> FlameResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| flame_core::Error::Io(format!("mkdir {}: {e}", parent.display())))?;
    }
    let mut output = HashMap::new();
    output.insert("packed_latent".to_string(), latents.clone());
    output.insert(
        "height".to_string(),
        Tensor::from_vec(vec![height as f32], Shape::from_dims(&[1]), device.clone())?
            .to_dtype(DType::BF16)?,
    );
    output.insert(
        "width".to_string(),
        Tensor::from_vec(vec![width as f32], Shape::from_dims(&[1]), device.clone())?
            .to_dtype(DType::BF16)?,
    );
    flame_core::serialization::save_file(&output, path)
}

fn save_tensor_as_png(rgb: &Tensor, path: &Path) -> FlameResult<()> {
    let dims = rgb.shape().dims();
    let (_b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec()?;

    let mut pixels = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3.min(c) {
                let val = data[ch * h * w + y * w + x];
                pixels[(y * w + x) * 3 + ch] =
                    ((val.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8;
            }
        }
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| flame_core::Error::Io(format!("mkdir {}: {e}", parent.display())))?;
    }
    let file = std::fs::File::create(path)
        .map_err(|e| flame_core::Error::Io(format!("create {}: {e}", path.display())))?;
    let mut writer = std::io::BufWriter::new(file);
    let encoder = image::codecs::png::PngEncoder::new(&mut writer);
    encoder
        .encode(&pixels, w as u32, h as u32, image::ColorType::Rgb8)
        .map_err(|e| flame_core::Error::Io(format!("PNG encode: {e}")))?;
    Ok(())
}

fn parse_adapter_algo(raw: &str) -> anyhow::Result<LycorisAlgo> {
    let algo_str = raw.trim().to_ascii_lowercase();
    if algo_str == "lora" || algo_str == "none" || algo_str.is_empty() {
        Ok(LycorisAlgo::None)
    } else {
        LycorisAlgo::parse(raw).map_err(|e| anyhow::anyhow!("--algo: {e}"))
    }
}
