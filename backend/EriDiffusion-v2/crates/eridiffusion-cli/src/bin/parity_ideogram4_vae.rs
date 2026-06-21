//! parity_ideogram4_vae — gate the EDv2 `KleinVaeEncoder` (reused) vs the torch
//! oracle `latents` [1,128,16,16] in `ideogram4_fx_vae_encode.safetensors`.
//! Feeds the fixture's `image` [1,3,256,256]; encode = posterior.mode + 2×patchify
//! + BN-normalise (= Ideogram's latent_norm). Confirms stage 6 = reuse.
//!
//! Run (GPU):
//!   LIBTORCH=/home/alex/libs/libtorch LD_LIBRARY_PATH=$LIBTORCH/lib \
//!     cargo run --release --bin parity_ideogram4_vae

use std::process::ExitCode;

use eridiffusion_core::encoders::vae::KleinVaeEncoder;
use flame_core::parity::{ParityHarness, ParityTolerance};
use flame_core::{DType, Shape, Tensor};

const VAE_FX: &str =
    "/home/alex/mojodiffusion/serenitymojo/models/vae/parity/ideogram4_fx_vae_encode.safetensors";
const IDEOGRAM_VAE: &str =
    "/home/alex/.serenity/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors";
const MIN_COS: f32 = 0.999;

fn run() -> anyhow::Result<bool> {
    let device = flame_core::global_cuda_device();

    let fx = flame_core::serialization::load_file(std::path::Path::new(VAE_FX), &device)
        .map_err(|e| anyhow::anyhow!("load vae fixture: {e}"))?;
    let img = fx
        .get("image")
        .ok_or_else(|| anyhow::anyhow!("fixture missing image"))?
        .to_dtype(DType::BF16)?;

    let vae_w = flame_core::serialization::load_file(std::path::Path::new(IDEOGRAM_VAE), &device)
        .map_err(|e| anyhow::anyhow!("load vae weights: {e}"))?;
    let dev = flame_core::Device::from(device.clone());
    let vae = KleinVaeEncoder::load(&vae_w, &dev)?;

    let latents = vae.encode(&img)?.to_dtype(DType::F32)?; // [1,128,16,16] klein channel order
    let dims = latents.shape().dims().to_vec();
    println!("[vae] latents {dims:?} (klein order)");

    // Reorder the 128 packed channels: klein patchify is [ae_ch, ph, pw]
    // (idx = ae*4 + ph*2 + pw); Ideogram is [ph, pw, ae_ch] (idx = ph*64 + pw*32 + ae).
    let spatial = dims[2] * dims[3];
    let v = latents.to_vec_f32()?;
    let mut reordered = vec![0f32; v.len()];
    for idx_ideo in 0..128usize {
        let ph = idx_ideo / 64;
        let rem = idx_ideo % 64;
        let pw = rem / 32;
        let ae = rem % 32;
        let idx_klein = ae * 4 + ph * 2 + pw;
        reordered[idx_ideo * spatial..(idx_ideo + 1) * spatial]
            .copy_from_slice(&v[idx_klein * spatial..(idx_klein + 1) * spatial]);
    }
    let latents = Tensor::from_vec(reordered, Shape::from_dims(&dims), device.clone())?;

    let mut h = ParityHarness::load(VAE_FX, device.clone())
        .map_err(|e| anyhow::anyhow!("harness: {e}"))?
        .with_tolerance(ParityTolerance {
            min_cos: MIN_COS,
            max_abs_ratio: 1.0,
        });
    let r = h
        .compare("latents", &latents)
        .map_err(|e| anyhow::anyhow!("compare latents: {e}"))?;
    let ok = r.cos >= MIN_COS;
    println!(
        "{:<14} {:>10.6} {:>12.4e} {:>12.4e}  {}",
        "latents",
        r.cos,
        r.max_abs,
        r.mean_abs,
        if ok { "OK" } else { "FAIL" }
    );
    Ok(ok)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::from(0),
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("[parity] error: {e:#}");
            ExitCode::from(2)
        }
    }
}
