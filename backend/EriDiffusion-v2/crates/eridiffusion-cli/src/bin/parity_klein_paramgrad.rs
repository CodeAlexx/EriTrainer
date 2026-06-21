//! Klein TRAINING-gradient self-consistency check (the test that actually
//! matters: are the gradients w.r.t. the trainable LoRA params correct?).
//!
//! This does NOT compare to diffusers (the 4% forward gap poisons any
//! cross-impl grad comparison). Instead it checks our analytic param-gradient
//! against the finite-difference directional derivative of OUR OWN forward:
//!
//!   analytic  = ||g||_2                         (g = dL/d(all lora params))
//!   numeric   = (L(theta + eps*g/||g||) - L(theta - eps*g/||g||)) / (2 eps)
//!   ratio     = numeric / ||g||_2
//!
//! If our backward IS the true gradient of our forward, ratio -> 1.0 as eps
//! enters the regime where the perturbation survives BF16 quantization (and
//! before curvature kicks in). We sweep eps and look for a PLATEAU:
//!   plateau ~1.0      => training gradient is CORRECT (proof)
//!   plateau ~constant!=1 => backward has a scale/direction error (proof of bug)
//!   monotonic climb   => BF16-confounded, INCONCLUSIVE (do not conclude)
//!
//!   cargo run --release --bin parity_klein_paramgrad -- parity/klein_fwd
//!   (env: LD_LIBRARY_PATH=libtorch, FLAME_ALLOC_POOL=0)

use std::path::PathBuf;
use flame_core::autograd::AutogradContext;
use flame_core::{DType, Shape, Tensor};
use eridiffusion_core::config::TrainConfig;
use eridiffusion_core::models::klein::KleinModel;
use eridiffusion_core::models::TrainableModel;

const TRANSFORMER: &str = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors";

fn main() -> anyhow::Result<()> {
    let dir: PathBuf = std::env::args().nth(1).unwrap_or_else(|| "parity/klein_fwd".into()).into();
    std::env::set_var("FLAME_ALLOC_POOL", "0");
    let device = flame_core::global_cuda_device();

    let config = TrainConfig::from_json_path("configs/klein9b_alina.json")?;
    let mut model = KleinModel::load(&[PathBuf::from(TRANSFORMER)], &config, device.clone())?;

    // TRAP 3: optional trained-LoRA checkpoint (arg 2). Loads LoRA weights over the init
    // zeros so the finite-diff gradient check runs at a TRAINED state (e.g. near the
    // ~step-600 bifurcation). No ckpt ⇒ init (zero LoRA), the original self-consistency test.
    if let Some(lora) = std::env::args().nth(2) {
        eprintln!("[paramgrad] loading trained LoRA from {lora}");
        model.load_weights(&lora)?;
    }

    let fix = flame_core::serialization::load_file(&dir.join("fwd_fixture.safetensors"), &device)?;
    let x_t = fix.get("x_t").unwrap().to_dtype(DType::BF16)?;
    let txt = fix.get("text_embedding").unwrap().to_dtype(DType::BF16)?;
    let target = fix.get("target").unwrap().to_dtype(DType::F32)?;
    let ts = fix.get("timestep").unwrap().to_vec()?;
    let timestep = Tensor::from_vec(vec![ts[0]], Shape::from_dims(&[1]), device.clone())?;

    // ---- forward + loss + backward ----
    flame_core::autograd::AutogradContext::clear();
    flame_core::autograd::AutogradContext::set_enabled(true);
    let vel = model.forward_train(&x_t, &txt, &timestep, None)?;
    let loss = vel.to_dtype(DType::F32)?.sub(&target)?.square()?.mean()?;
    let loss0 = loss.to_vec()?[0] as f64;
    println!("baseline loss = {loss0:.8e}");
    let grads = loss.backward()?;

    // ---- collect (param, value, grad) for every trainable param with a grad ----
    let params = model.parameters();
    struct P { value: Vec<f32>, grad: Vec<f32>, shape: Shape, idx: usize }
    let mut ps: Vec<P> = Vec::new();
    let mut g2: f64 = 0.0;          // ||g||_2^2
    let mut g1: f64 = 0.0;          // ||g||_1
    let mut n_elem: usize = 0;
    for (idx, p) in params.iter().enumerate() {
        let g = match grads.get(p.id()) {
            Some(g) => g.to_dtype(DType::F32)?.to_vec()?,
            None => continue,
        };
        let v = p.tensor()?.to_dtype(DType::F32)?.to_vec()?;
        assert_eq!(v.len(), g.len());
        for &gi in &g { g2 += (gi as f64) * (gi as f64); g1 += (gi as f64).abs(); }
        n_elem += g.len();
        ps.push(P { value: v, grad: g, shape: p.tensor()?.shape().clone(), idx });
    }
    let gnorm = g2.sqrt();
    println!("trainable params with grad: {} tensors, {} elements", ps.len(), n_elem);
    println!("||g||_2 = {gnorm:.6e}   ||g||_1 = {g1:.6e}   (mean|g| = {:.3e})", g1 / n_elem.max(1) as f64);

    // unit L2 direction d = g / ||g||_2 ; per-tensor slices stored in ps.grad / gnorm
    let dev = device.clone();
    let perturb = |ps: &Vec<P>, model: &mut KleinModel, scale: f64| -> anyhow::Result<()> {
        for p in ps {
            let nv: Vec<f32> = p.value.iter().zip(&p.grad)
                .map(|(&v, &gi)| (v as f64 + scale * gi as f64 / gnorm) as f32)
                .collect();
            let t = Tensor::from_vec(nv, p.shape.clone(), dev.clone())?
                .to_dtype(DType::F32)?.requires_grad_(true);
            model.parameters()[p.idx].set_data(t)?;
        }
        Ok(())
    };
    let restore = |ps: &Vec<P>, model: &mut KleinModel| -> anyhow::Result<()> {
        for p in ps {
            let t = Tensor::from_vec(p.value.clone(), p.shape.clone(), dev.clone())?
                .to_dtype(DType::F32)?.requires_grad_(true);
            model.parameters()[p.idx].set_data(t)?;
        }
        Ok(())
    };
    let loss_now = |model: &mut KleinModel| -> anyhow::Result<f64> {
        let _g = AutogradContext::no_grad();
        let v = model.forward_train(&x_t, &txt, &timestep, None)?;
        Ok(v.to_dtype(DType::F32)?.sub(&target)?.square()?.mean()?.to_vec()?[0] as f64)
    };

    println!("\n--- PARAM-GRADIENT self-consistency (numeric vs analytic ||g||) ---");
    println!("  eps        numeric        analytic       ratio     (L+ , L-)");
    for eps in [0.003f64, 0.01, 0.03, 0.1, 0.3, 1.0] {
        perturb(&ps, &mut model, eps)?;
        let lp = loss_now(&mut model)?;
        perturb(&ps, &mut model, -eps)?;
        let lm = loss_now(&mut model)?;
        restore(&ps, &mut model)?;
        let numeric = (lp - lm) / (2.0 * eps);
        println!("  {eps:<8.3}  {numeric:.6e}   {gnorm:.6e}   {:.4}   ({lp:.7e} , {lm:.7e})",
            numeric / (gnorm + 1e-30));
    }
    println!("  ratio plateau ~1.0 => our backward = true gradient of our forward (CORRECT)");
    println!("  ratio plateau !=1  => backward scale/direction error (BUG)");
    println!("  monotonic climb    => BF16-confounded, INCONCLUSIVE");
    Ok(())
}
