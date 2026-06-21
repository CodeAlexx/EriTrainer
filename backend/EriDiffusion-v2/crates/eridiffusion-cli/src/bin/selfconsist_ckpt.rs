//! Isolate gradient-checkpoint vs composition for the klein backward defect.
//! klein forward_train self-consistency (our backward vs finite-diff of our own
//! forward) on a SMALL synthetic latent that fits WITHOUT checkpoint. Run twice:
//!   KLEIN_GRAD_CHECKPOINT=1 ./selfconsist_ckpt   (composed test uses this)
//!   KLEIN_GRAD_CHECKPOINT=0 ./selfconsist_ckpt   (only fits because latent is tiny)
//! ratio ~0.77 with ckpt and ~1.0 without => checkpoint recompute is the defect.
//! ratio equal both ways => NOT checkpoint (multi-block residual / nonlinear op).
//! No fixture, no diffusers — self-consistency only.

use std::path::PathBuf;
use flame_core::autograd::AutogradContext;
use flame_core::{DType, Shape, Tensor};
use eridiffusion_core::config::TrainConfig;
use eridiffusion_core::models::klein::KleinModel;

const TRANSFORMER: &str = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-4b.safetensors";

fn main() -> anyhow::Result<()> {
    std::env::set_var("FLAME_ALLOC_POOL", "0");
    let device = flame_core::global_cuda_device();
    let config = TrainConfig::from_json_path("configs/klein4b_eri2_baseline_diff.json")?;
    let mut model = KleinModel::load(&[PathBuf::from(TRANSFORMER)], &config, device.clone())?;

    // Small synthetic input: fixture is [1,128,40,28]+txt[1,512,12288]; shrink to fit no-ckpt.
    let (lh, lw, n_txt) = (8usize, 8usize, 64usize);
    let x_t = Tensor::randn(Shape::from_dims(&[1, 128, lh, lw]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;
    let txt = Tensor::randn(Shape::from_dims(&[1, n_txt, 7680]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;
    let timestep = Tensor::from_vec(vec![0.5f32], Shape::from_dims(&[1]), device.clone())?;
    let ckpt = std::env::var("KLEIN_GRAD_CHECKPOINT").unwrap_or_else(|_| "1".into());
    println!("synth self-consistency: x_t[1,128,{lh},{lw}] txt[1,{n_txt},7680] KLEIN_GRAD_CHECKPOINT={ckpt}");

    AutogradContext::clear();
    AutogradContext::set_enabled(true);
    let x_leaf = x_t.clone().requires_grad_(true);
    let vel = model.forward_train(&x_leaf, &txt, &timestep, None)?;
    let loss = vel.to_dtype(DType::F32)?.square()?.mean()?;
    println!("loss = {:.6e}", loss.to_vec()?[0]);
    let grads = loss.backward()?;
    let dx = grads.get(x_leaf.id()).ok_or_else(|| anyhow::anyhow!("no dL/dx (dead backward)"))?.clone();
    let dv = dx.to_dtype(DType::F32)?.to_vec()?;

    let mut loss_at = |xt: &Tensor| -> anyhow::Result<f64> {
        let _g = AutogradContext::no_grad();
        let v = model.forward_train(xt, &txt, &timestep, None)?;
        Ok(v.to_dtype(DType::F32)?.square()?.mean()?.to_vec()?[0] as f64)
    };
    let x0 = x_t.to_dtype(DType::F32)?.to_vec()?;
    let shp = x_t.shape().clone();
    let gn: f64 = dv.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let vunit: Vec<f32> = dv.iter().map(|x| (*x as f64 / (gn + 1e-30)) as f32).collect();
    let ana = gn;
    println!("--- self-consistency (ratio ~1.0 = backward is true gradient of forward) ---");
    for eps in [0.4f32, 1.0, 2.0, 4.0, 8.0] {
        let plus: Vec<f32> = x0.iter().zip(&vunit).map(|(a, b)| a + eps * b).collect();
        let minus: Vec<f32> = x0.iter().zip(&vunit).map(|(a, b)| a - eps * b).collect();
        let xp = Tensor::from_vec(plus, shp.clone(), device.clone())?.to_dtype(DType::BF16)?;
        let xm = Tensor::from_vec(minus, shp.clone(), device.clone())?.to_dtype(DType::BF16)?;
        let lp = loss_at(&xp)?;
        let lm = loss_at(&xm)?;
        let num = (lp - lm) / (2.0 * eps as f64);
        println!("  eps={eps:.2} numeric={num:.4e} analytic={ana:.4e} ratio={:.3}", num / (ana + 1e-30));
    }
    Ok(())
}
