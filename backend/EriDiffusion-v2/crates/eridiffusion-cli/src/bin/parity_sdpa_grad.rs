//! SDPA backward direction parity: our `attention::sdpa` backward vs PyTorch.
//!
//! Injects a fixed upstream grad G via loss = sum(out * G) so dOut = G exactly,
//! then compares dQ/dK/dV direction (cosine) to PyTorch's SDPA backward.
//! Klein attention shape: [B=1, H=32, S=256, D=128] BF16.
//!
//!   cargo run --release --bin parity_sdpa_grad -- parity/klein_sdpa_grad

use std::path::PathBuf;
use flame_core::autograd::AutogradContext;
use flame_core::{DType, Tensor};

fn cos_and_rell2(ours: &[f32], reference: &[f32]) -> (f64, f64) {
    assert_eq!(ours.len(), reference.len());
    let (mut dot, mut na, mut nb, mut diff) = (0f64, 0f64, 0f64, 0f64);
    for (&a, &b) in ours.iter().zip(reference.iter()) {
        let (a, b) = (a as f64, b as f64);
        dot += a * b; na += a * a; nb += b * b; diff += (a - b) * (a - b);
    }
    (dot / (na.sqrt() * nb.sqrt() + 1e-30), diff.sqrt() / (nb.sqrt() + 1e-30))
}

fn report(name: &str, ours: &Tensor, refs: &std::collections::HashMap<String, Tensor>) -> anyhow::Result<()> {
    let ov = ours.to_dtype(DType::F32)?.to_vec()?;
    let on: f64 = ov.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    println!("  ours ||{}|| = {:.4e}", name, on);
    for regime in ["f32", "bf16mirror"] {
        let r = refs.get(&format!("{name}.{regime}")).ok_or_else(|| anyhow::anyhow!("missing {name}.{regime}"))?;
        let rv = r.to_dtype(DType::F32)?.to_vec()?;
        let (cos, rel) = cos_and_rell2(&ov, &rv);
        println!("    vs {regime:10}: cos={cos:.6}  relL2={rel:.4e}");
    }
    Ok(())
}

fn leaf_bf16(t: &Tensor) -> anyhow::Result<Tensor> {
    Ok(t.to_dtype_no_grad(DType::BF16)?.requires_grad_(true))
}

fn main() -> anyhow::Result<()> {
    let dir: PathBuf = std::env::args().nth(1).unwrap_or_else(|| "parity/klein_sdpa_grad".into()).into();
    let device = flame_core::global_cuda_device();
    let fix = flame_core::serialization::load_file(&dir.join("fixture.safetensors"), &device)?;
    let refs = flame_core::serialization::load_file(&dir.join("ref.safetensors"), &device)?;

    let q = leaf_bf16(fix.get("Q").unwrap())?;
    let k = leaf_bf16(fix.get("K").unwrap())?;
    let v = leaf_bf16(fix.get("V").unwrap())?;
    let g = fix.get("G").unwrap().to_dtype(DType::BF16)?;
    println!("Q{:?} H/D from shape", q.shape().dims());

    AutogradContext::clear();
    AutogradContext::set_enabled(true);

    let out = flame_core::attention::sdpa(&q, &k, &v, None)?;
    let loss = out.mul(&g)?.sum()?;
    println!("loss(sum out*G) = {:.6e}", loss.to_vec()?[0]);
    let grads = loss.backward()?;

    let dq = grads.get(q.id()).ok_or_else(|| anyhow::anyhow!("no dQ"))?.clone();
    let dk = grads.get(k.id()).ok_or_else(|| anyhow::anyhow!("no dK"))?.clone();
    let dv = grads.get(v.id()).ok_or_else(|| anyhow::anyhow!("no dV"))?.clone();

    println!("--- forward parity ---");
    report("out", &out.to_dtype(DType::F32)?, &refs)?;
    println!("--- SDPA backward direction parity ---");
    report("dQ", &dq, &refs)?;
    report("dK", &dk, &refs)?;
    report("dV", &dv, &refs)?;
    Ok(())
}
