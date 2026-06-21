//! Single-step LoRA gradient parity: our autograd vs PyTorch reference.
//!
//! Loads the shared fixture (parity/klein_lora_grad/fixture.safetensors) and
//! the PyTorch reference grads (ref.safetensors), runs our
//! `LoRALinear::forward_delta` + backward on the IDENTICAL inputs, and diffs
//! the gradient on lora_A / lora_B against PyTorch.
//!
//! Run:
//!   cargo run --release --bin parity_lora_grad -- parity/klein_lora_grad
//!
//! No GPU model load, no offload — just one LoRA module forward/backward.

use std::path::PathBuf;

use flame_core::autograd::AutogradContext;
use flame_core::{DType, Tensor};
use eridiffusion_core::lora::LoRALinear;

const IN: usize = 4096;
const OUT: usize = 4096;
const RANK: usize = 16;
const ALPHA: f32 = 16.0;

fn cos_and_rell2(ours: &[f32], reference: &[f32]) -> (f64, f64) {
    assert_eq!(ours.len(), reference.len(), "length mismatch");
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut diff = 0f64;
    for (&a, &b) in ours.iter().zip(reference.iter()) {
        let (a, b) = (a as f64, b as f64);
        dot += a * b;
        na += a * a;
        nb += b * b;
        diff += (a - b) * (a - b);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    let rel = diff.sqrt() / (nb.sqrt() + 1e-30);
    (cos, rel)
}

fn report(name: &str, ours: &Tensor, refs: &std::collections::HashMap<String, Tensor>) -> anyhow::Result<()> {
    let ov = ours.to_dtype(DType::F32)?.to_vec()?;
    let on: f64 = ov.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    println!("  ours ||{}|| = {:.6e}", name, on);
    for regime in ["f32", "bf16mirror"] {
        let key = format!("{name}.{regime}");
        let r = refs.get(&key).ok_or_else(|| anyhow::anyhow!("missing ref {key}"))?;
        let rv = r.to_dtype(DType::F32)?.to_vec()?;
        let (cos, rel) = cos_and_rell2(&ov, &rv);
        let rn: f64 = rv.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        println!(
            "    vs {regime:10}: cos={cos:.6}  relL2={rel:.4e}  (ref ||{name}||={rn:.6e})"
        );
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "parity/klein_lora_grad".to_string())
        .into();
    let device = flame_core::global_cuda_device();

    let fix = flame_core::serialization::load_file(&dir.join("fixture.safetensors"), &device)?;
    let refs = flame_core::serialization::load_file(&dir.join("ref.safetensors"), &device)?;

    let x = fix.get("X").ok_or_else(|| anyhow::anyhow!("missing X"))?.clone();
    let t = fix.get("T").ok_or_else(|| anyhow::anyhow!("missing T"))?.clone();
    let a = fix.get("A").ok_or_else(|| anyhow::anyhow!("missing A"))?.clone();
    let b = fix.get("B").ok_or_else(|| anyhow::anyhow!("missing B"))?.clone();
    println!(
        "fixture: X{:?} T{:?} A{:?} B{:?}",
        x.shape().dims(), t.shape().dims(), a.shape().dims(), b.shape().dims()
    );

    // Build our LoRA module and overwrite A/B with the fixture values.
    let lora = LoRALinear::new(IN, OUT, RANK, ALPHA, device.clone(), 0)?;
    lora.lora_a.set_data(a.to_dtype(DType::F32)?.requires_grad_(true))?;
    lora.lora_b.set_data(b.to_dtype(DType::F32)?.requires_grad_(true))?;

    // Mirror training: activations enter the adapter in BF16.
    let x_bf16 = x.to_dtype(DType::BF16)?;

    AutogradContext::clear();
    AutogradContext::set_enabled(true);

    let pred = lora.forward_delta(&x_bf16)?; // [1, SEQ, OUT], bf16
    let pred_f32 = pred.to_dtype(DType::F32)?;
    let t_f32 = t.to_dtype(DType::F32)?;
    let loss = pred_f32.sub(&t_f32)?.square()?.mean()?;
    let loss_val = loss.to_vec()?[0];
    println!("our loss = {:.6e}", loss_val);

    let grads = loss.backward()?;

    let ga = grads
        .get(lora.lora_a.id())
        .ok_or_else(|| anyhow::anyhow!("NO GRADIENT for lora_A (id not in map) — dead path"))?
        .clone();
    let gb = grads
        .get(lora.lora_b.id())
        .ok_or_else(|| anyhow::anyhow!("NO GRADIENT for lora_B (id not in map) — dead path"))?
        .clone();

    println!("--- forward (delta) parity ---");
    report("delta", &pred_f32, &refs)?;
    println!("--- gradient parity ---");
    report("grad_A", &ga, &refs)?;
    report("grad_B", &gb, &refs)?;
    Ok(())
}
