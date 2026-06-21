//! Klein forward-prediction parity: our `forward_train` vs diffusers klein DiT.
//!
//! Loads the shared fixture (parity/klein_fwd/fwd_fixture.safetensors: x_t,
//! timestep, text_embedding, velocity_ref) produced by `ref.py`, runs our
//! KleinModel.forward_train on the IDENTICAL inputs, and compares the predicted
//! velocity (cosine + relL2). Isolates our packing + RoPE ids + 32-block DiT
//! against the reference DiT on identical input.
//!
//!   cargo run --release --bin parity_klein_fwd -- parity/klein_fwd
//!   (env: LD_LIBRARY_PATH=libtorch, FLAME_ALLOC_POOL=0)

use std::path::PathBuf;
use flame_core::autograd::AutogradContext;
use flame_core::{DType, Shape, Tensor};
use eridiffusion_core::config::TrainConfig;
use eridiffusion_core::models::klein::KleinModel;

const TRANSFORMER: &str = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors";

fn cos_rel(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb, mut diff) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y; na += x * x; nb += y * y; diff += (x - y) * (x - y);
    }
    (dot / (na.sqrt() * nb.sqrt() + 1e-30), diff.sqrt() / (nb.sqrt() + 1e-30))
}

fn main() -> anyhow::Result<()> {
    let dir: PathBuf = std::env::args().nth(1).unwrap_or_else(|| "parity/klein_fwd".into()).into();
    std::env::set_var("FLAME_ALLOC_POOL", "0");
    std::env::set_var("FLAME_KLEIN_PROBE", "1");
    eridiffusion_core::models::klein::KLEIN_GRAD_PROBE.lock().unwrap().clear();
    eridiffusion_core::models::klein::KLEIN_FWD_VALS.lock().unwrap().clear();
    let device = flame_core::global_cuda_device();

    let config = TrainConfig::from_json_path("configs/klein9b_alina.json")?;
    let mut model = KleinModel::load(&[PathBuf::from(TRANSFORMER)], &config, device.clone())?;

    let fix = flame_core::serialization::load_file(&dir.join("fwd_fixture.safetensors"), &device)?;
    let x_t = fix.get("x_t").unwrap().to_dtype(DType::BF16)?;
    let txt = fix.get("text_embedding").unwrap().to_dtype(DType::BF16)?;
    let ref_v = fix.get("velocity_ref").unwrap().to_dtype(DType::F32)?;
    let ts = fix.get("timestep").unwrap().to_vec()?;
    let timestep = Tensor::from_vec(vec![ts[0]], Shape::from_dims(&[1]), device.clone())?;
    println!("x_t{:?} txt{:?} timestep={:.4} ref_v{:?}",
        x_t.shape().dims(), txt.shape().dims(), ts[0], ref_v.shape().dims());

    // Full-step BACKWARD parity: x_t as a leaf requiring grad, run forward_train
    // -> MSE(pred, target) -> backward, read dL/dx_t, compare DIRECTION to the
    // diffusers reference. This exercises the COMPOSED 32-block backward in the
    // real training path — the one thing isolated-op parity never tested.
    let target = fix.get("target").unwrap().to_dtype(DType::F32)?;
    let dldx_ref = fix.get("dLdx_ref").unwrap().to_dtype(DType::F32)?;
    let x_t_leaf = x_t.clone().requires_grad_(true);

    flame_core::autograd::AutogradContext::clear();
    flame_core::autograd::AutogradContext::set_enabled(true);
    let vel = model.forward_train(&x_t_leaf, &txt, &timestep, None)?;
    let loss = vel.to_dtype(DType::F32)?.sub(&target)?.square()?.mean()?;
    println!("our loss = {:.6e}", loss.to_vec()?[0]);
    let grads = loss.backward()?;

    // Per-block probe grads → save for comparison vs diffusers per-block grads.
    {
        let probes = eridiffusion_core::models::klein::KLEIN_GRAD_PROBE.lock().unwrap().clone();
        let retained = AutogradContext::take_retained_intermediate_grads();
        let mut out = std::collections::HashMap::new();
        for (label, id) in &probes {
            if let Some(g) = retained.get(id) {
                out.insert(label.clone(), g.to_dtype(DType::F32)?);
            } else {
                println!("  [probe] {label}: NO retained grad");
            }
        }
        println!("per-block probes captured: {}/{}", out.len(), probes.len());
        if !out.is_empty() {
            flame_core::serialization::save_file(&out, dir.join("our_block_grads.safetensors"))?;
        }

        // Per-block FORWARD activations (our side) → compare vs diffusers
        // forward hooks. forward cos high + backward cos low = backward bug.
        let fwd = eridiffusion_core::models::klein::KLEIN_FWD_VALS.lock().unwrap().clone();

        // === RECOMPUTE FIDELITY: initial forward (#0) vs checkpoint recompute (#1) ===
        // If recompute diverges from the initial forward, the backward operates on
        // activations that differ from those that produced the loss → corrupted grad.
        {
            use std::collections::HashMap;
            let mut by_base: HashMap<String, HashMap<usize, Vec<f32>>> = HashMap::new();
            for (label, vals) in &fwd {
                if let Some((base, nstr)) = label.rsplit_once('#') {
                    if let Ok(n) = nstr.parse::<usize>() {
                        by_base.entry(base.to_string()).or_default().insert(n, vals.clone());
                    }
                }
            }
            let mut keys: Vec<_> = by_base.keys().cloned().collect();
            keys.sort();
            println!("--- RECOMPUTE FIDELITY: initial(#0) vs recompute(#1) ---");
            let mut worst = 1.0f64;
            let mut pairs = 0;
            for base in &keys {
                let m = &by_base[base];
                if let (Some(a), Some(b)) = (m.get(&0), m.get(&1)) {
                    if a.len() == b.len() {
                        let (cos, rel) = cos_rel(a, b);
                        if cos < worst { worst = cos; }
                        pairs += 1;
                        println!("  {base:<26} #0 vs #1: cos={cos:.8} relL2={rel:.4e}");
                    }
                }
            }
            println!("  >>> {pairs} blocks compared; WORST recompute cos = {worst:.8}  (1.0=faithful; <1=recompute DIVERGES from forward)");
        }

        let mut facts = std::collections::HashMap::new();
        for (label, vals) in &fwd {
            let n = vals.len();
            facts.insert(label.clone(),
                Tensor::from_vec(vals.clone(), Shape::from_dims(&[n]), device.clone())?);
        }
        println!("per-block forward acts captured: {}", facts.len());
        if !facts.is_empty() {
            flame_core::serialization::save_file(&facts, dir.join("our_block_acts.safetensors"))?;
        }
    }

    let ov = vel.to_dtype(DType::F32)?.to_vec()?;
    let rv = ref_v.to_vec()?;
    let (fcos, frel) = cos_rel(&ov, &rv);
    println!("--- FORWARD parity ---");
    println!("  velocity: cos={fcos:.6}  relL2={frel:.4e}");

    let dx = grads.get(x_t_leaf.id()).cloned()
        .ok_or_else(|| anyhow::anyhow!("NO dL/dx_t gradient (dead backward)"))?;
    let dv = dx.to_dtype(DType::F32)?.to_vec()?;
    let rd = dldx_ref.to_vec()?;
    let (cos, rel) = cos_rel(&dv, &rd);
    println!("--- BACKWARD parity vs diffusers (dL/dx_t) ---");
    println!("  dL/dx_t vs diffusers:  cos={cos:.6}  relL2={rel:.4e}");

    // SELF-CONSISTENCY: is our analytic dL/dx_t the true gradient of OUR forward?
    // Directional finite difference: numeric <grad,v> = (loss(x+ev)-loss(x-ev))/(2e)
    // vs analytic <our_grad,v>. If they match, our backward is CORRECT for our
    // forward (and the diffusers gap is a forward difference, not a backward bug).
    // If they diverge, our backward is genuinely wrong.
    println!("--- SELF-CONSISTENCY (our backward vs finite-diff of OUR forward) ---");
    let mut loss_at = |xt: &Tensor| -> anyhow::Result<f64> {
        let _g = AutogradContext::no_grad();
        let v = model.forward_train(xt, &txt, &timestep, None)?;
        Ok(v.to_dtype(DType::F32)?.sub(&target)?.square()?.mean()?.to_vec()?[0] as f64)
    };
    let x0 = x_t.to_dtype(DType::F32)?.to_vec()?;
    let shp = x_t.shape().clone();
    let gn: f64 = dv.iter().map(|x|(*x as f64).powi(2)).sum::<f64>().sqrt();
    let vunit: Vec<f32> = dv.iter().map(|x| (*x as f64 / (gn+1e-30)) as f32).collect();
    let ana: f64 = dv.iter().zip(&vunit).map(|(g,v)| *g as f64 * *v as f64).sum(); // = |grad|
    // NOTE: xp/xm are cast to BF16 below, so perturbations below BF16 input
    // resolution (~0.004 near |x|~1) get rounded away — this UNDER-counts
    // `numeric` at small eps. The unit grad dir spreads ~0.0026/component over
    // ~143k dims, so eps must be ≳2 for the per-element step to clear BF16
    // resolution. Sweep into the large-eps regime to separate genuine backward
    // over-magnitude (ratio plateaus <1) from BF16-input-quantization artifact
    // (ratio → 1.0 as eps grows).
    for eps in [0.15f32, 0.25, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0] {
        let plus: Vec<f32> = x0.iter().zip(&vunit).map(|(a,b)| a + eps*b).collect();
        let minus: Vec<f32> = x0.iter().zip(&vunit).map(|(a,b)| a - eps*b).collect();
        let xp = Tensor::from_vec(plus, shp.clone(), device.clone())?.to_dtype(DType::BF16)?;
        let xm = Tensor::from_vec(minus, shp.clone(), device.clone())?.to_dtype(DType::BF16)?;
        let lp = loss_at(&xp)?; let lm = loss_at(&xm)?;
        let num = (lp - lm) / (2.0 * eps as f64);
        println!("  eps={eps:.2} grad-dir: numeric={num:.4e} analytic(|grad|)={ana:.4e} ratio={:.3}  (l+={lp:.5} l-={lm:.5})", num/(ana+1e-30));
    }
    println!("  (ratios ~1.0 => our backward is the true gradient of our forward;");
    println!("   ratios far from 1.0 => our backward is genuinely wrong)");
    Ok(())
}
