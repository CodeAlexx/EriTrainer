//! RMSNorm + LayerNorm backward parity vs PyTorch.
//!   cargo run --release --bin parity_norm_grad -- parity/klein_norm_grad
use std::path::PathBuf;
use flame_core::autograd::AutogradContext;
use flame_core::{DType, Tensor};

const EPS: f32 = 1e-6;

fn cos_rel(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut diff) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (x as f64, y as f64);
        dot += x*y; na += x*x; nb += y*y; diff += (x-y)*(x-y);
    }
    (dot/(na.sqrt()*nb.sqrt()+1e-30), diff.sqrt()/(nb.sqrt()+1e-30))
}
fn cmp(name: &str, ours: &Tensor, refv: &Tensor) -> anyhow::Result<()> {
    let o = ours.to_dtype(DType::F32)?.to_vec()?;
    let r = refv.to_dtype(DType::F32)?.to_vec()?;
    let on: f64 = o.iter().map(|x|(*x as f64).powi(2)).sum::<f64>().sqrt();
    let rn: f64 = r.iter().map(|x|(*x as f64).powi(2)).sum::<f64>().sqrt();
    let (c, rel) = cos_rel(&o, &r);
    let flag = if c > 0.99 { "OK" } else { "*** DIVERGES ***" };
    println!("  {name:10} cos={c:.6} relL2={rel:.4e}  ours||{:.4e}|| ref||{:.4e}||  {flag}", on, rn);
    Ok(())
}
fn leaf(t: &Tensor) -> anyhow::Result<Tensor> { Ok(t.to_dtype(DType::BF16)?.requires_grad_(true)) }

fn main() -> anyhow::Result<()> {
    let dir: PathBuf = std::env::args().nth(1).unwrap_or_else(|| "parity/klein_norm_grad".into()).into();
    let dev = flame_core::global_cuda_device();
    let f = flame_core::serialization::load_file(&dir.join("fixture.safetensors"), &dev)?;

    // ---- RMSNorm (QK norm) ----
    let d = *f.get("rms_x").unwrap().shape().dims().last().unwrap();
    let x = leaf(f.get("rms_x").unwrap())?;
    let w = leaf(f.get("rms_w").unwrap())?;
    let g = f.get("rms_G").unwrap().to_dtype(DType::BF16)?;
    AutogradContext::clear(); AutogradContext::set_enabled(true);
    let out = flame_core::norm::rms_norm(&x, &[d], Some(&w), EPS)?;
    let loss = out.mul(&g)?.sum()?;
    let grads = loss.backward()?;
    println!("=== RMSNorm backward (QK norm, per-block on q&k) ===");
    cmp("rms_out", &out.to_dtype(DType::F32)?, f.get("rms_out").unwrap())?;
    match grads.get(x.id()) { Some(dx)=>cmp("rms_dx", dx, f.get("rms_dx").unwrap())?, None=>println!("  rms_dx: NONE") }
    match grads.get(w.id()) { Some(dw)=>cmp("rms_dw", dw, f.get("rms_dw").unwrap())?, None=>println!("  rms_dw: NONE") }

    // ---- LayerNorm (modulation, no affine) ----
    let dim = *f.get("ln_x").unwrap().shape().dims().last().unwrap();
    let y = leaf(f.get("ln_x").unwrap())?;
    let gl = f.get("ln_G").unwrap().to_dtype(DType::BF16)?;
    AutogradContext::clear(); AutogradContext::set_enabled(true);
    let lout = flame_core::layer_norm::layer_norm(&y, &[dim], None, None, EPS)?;
    let lloss = lout.mul(&gl)?.sum()?;
    let lgrads = lloss.backward()?;
    println!("=== LayerNorm backward (modulation) ===");
    cmp("ln_out", &lout.to_dtype(DType::F32)?, f.get("ln_out").unwrap())?;
    match lgrads.get(y.id()) { Some(dx)=>cmp("ln_dx", dx, f.get("ln_dx").unwrap())?, None=>println!("  ln_dx: NONE") }

    // ---- SwiGLU split-lastdim (MLP activation, every block) ----
    let sx = leaf(f.get("sw_x").unwrap())?;
    let gs = f.get("sw_G").unwrap().to_dtype(DType::BF16)?;
    AutogradContext::clear(); AutogradContext::set_enabled(true);
    let sout = flame_core::bf16_ops::swiglu_split_lastdim_bf16(&sx)?;
    let sloss = sout.mul(&gs)?.sum()?;
    let sgrads = sloss.backward()?;
    println!("=== SwiGLU backward (MLP activation, per-block) ===");
    cmp("sw_out", &sout.to_dtype(DType::F32)?, f.get("sw_out").unwrap())?;
    match sgrads.get(sx.id()) { Some(dx)=>cmp("sw_dx", dx, f.get("sw_dx").unwrap())?, None=>println!("  sw_dx: NONE *** DEAD ***") }
    Ok(())
}
