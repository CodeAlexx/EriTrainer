//! parity_ideogram4_block_adjoint — confound-free adjoint self-test of the LINEAR
//! plumbing in ideogram's block backward (the ops `adaln_modulation`'s LoRA grad
//! flows through, which the FD self-consistency flagged).
//!
//! For any LINEAR op f, the true backward IS the adjoint: <f(x),G> == <x, fᵀ(G)>.
//! We run f(x) with autograd, inject upstream grad G via loss=<f(x),G>, backward
//! to get fᵀ(G)=grad_x, and check the two inner products match. EXACT in F32 —
//! no finite-diff, no bf16 param-cast, no cross-impl drift (the confounds that
//! muddied the oracle + FD tests). A large rel gap ⇒ that op's backward is NOT
//! the adjoint of its forward = a gradient-wiring bug.
//!
//! Run (GPU):
//!   LIBTORCH=/home/alex/libs/libtorch LD_LIBRARY_PATH=$LIBTORCH/lib \
//!     cargo run --release --bin parity_ideogram4_block_adjoint

use flame_core::autograd::AutogradContext;
use flame_core::{DType, Shape, Tensor};

fn dot_f32(a: &Tensor, b: &Tensor) -> anyhow::Result<f64> {
    let a = a.to_dtype(DType::F32)?.to_vec_f32()?;
    let b = b.to_dtype(DType::F32)?.to_vec_f32()?;
    assert_eq!(a.len(), b.len(), "dot length mismatch");
    Ok(a.iter().zip(&b).map(|(&x, &y)| x as f64 * y as f64).sum())
}

fn randn(shape: &[usize], dev: &std::sync::Arc<flame_core::CudaDevice>) -> anyhow::Result<Tensor> {
    Ok(Tensor::randn(Shape::from_dims(shape), 0.0, 1.0, dev.clone())?)
}

fn report(name: &str, lhs: f64, rhs: f64) -> bool {
    let rel = (lhs - rhs).abs() / (rhs.abs().max(lhs.abs()) + 1e-30);
    let ok = rel < 0.01;
    let v = if ok { "OK (adjoint holds)" } else { "*** BUG: backward != adjoint ***" };
    println!("  {name:<34} <f,G>={lhs:+.6e}  <x,fᵀG>={rhs:+.6e}  rel={rel:.2e}  {v}");
    ok
}

fn main() -> anyhow::Result<()> {
    std::env::set_var("FLAME_ALLOC_POOL", "0");
    let dev = flame_core::global_cuda_device();
    let h = 64usize; // chunk size (stands in for hidden); adjoint is size-independent
    let mut all_ok = true;

    // ---- POSITIVE CONTROL: contiguous reshape (known-correct) must report OK ----
    {
        AutogradContext::clear();
        AutogradContext::set_enabled(true);
        let x = randn(&[1, 8, 4 * h], &dev)?.requires_grad_(true);
        let y = x.reshape(&[1, 8 * 4 * h])?;
        let g = randn(y.shape().dims(), &dev)?;
        let loss = y.mul(&g)?.sum()?;
        let lhs = loss.to_vec_f32()?[0] as f64;
        let grads = loss.backward()?;
        let gx = grads.get(x.id()).ok_or_else(|| anyhow::anyhow!("no grad_x"))?.clone();
        let rhs = dot_f32(&x, &gx)?;
        println!("CONTROL contiguous reshape (must be OK):");
        all_ok &= report("reshape", lhs, rhs);
    }

    // ---- adaln_modulation backward path: m[1,L,4h] -> 4× narrow(dim2, k·h, h) ----
    //      (scale_msa/gate_msa/scale_mlp/gate_mlp split). The grad must scatter-add
    //      the 4 slices back into m at the right offsets.
    {
        AutogradContext::clear();
        AutogradContext::set_enabled(true);
        let l = 2usize;
        let m = randn(&[1, l, 4 * h], &dev)?.requires_grad_(true);
        let mut loss: Option<Tensor> = None;
        for k in 0..4 {
            let s = m.narrow(2, k * h, h)?;
            let g = randn(s.shape().dims(), &dev)?;
            let term = s.mul(&g)?.sum()?;
            loss = Some(match loss {
                None => term,
                Some(acc) => acc.add(&term)?,
            });
        }
        let loss = loss.unwrap();
        let lhs = loss.to_vec_f32()?[0] as f64;
        let grads = loss.backward()?;
        let gm = grads.get(m.id()).ok_or_else(|| anyhow::anyhow!("no grad_m"))?.clone();
        let rhs = dot_f32(&m, &gm)?;
        println!("adaln_modulation 4-way last-dim narrow:");
        all_ok &= report("narrow x4 (scatter-add)", lhs, rhs);
    }

    // ---- qkv split as block_standalone does it: reshape->narrow(MIDDLE dim 2)->reshape ----
    {
        AutogradContext::clear();
        AutogradContext::set_enabled(true);
        let (nh, dh, lq) = (4usize, 16usize, 8usize);
        let inner = nh * dh;
        let qkv = randn(&[1, lq, 3 * inner], &dev)?.requires_grad_(true);
        let qkv5 = qkv.reshape(&[1, lq, 3, nh, dh])?;
        let mut loss: Option<Tensor> = None;
        for i in 0..3 {
            let part = qkv5.narrow(2, i, 1)?.reshape(&[1, lq, nh, dh])?;
            let g = randn(part.shape().dims(), &dev)?;
            let term = part.mul(&g)?.sum()?;
            loss = Some(match loss {
                None => term,
                Some(acc) => acc.add(&term)?,
            });
        }
        let loss = loss.unwrap();
        let lhs = loss.to_vec_f32()?[0] as f64;
        let grads = loss.backward()?;
        let gq = grads.get(qkv.id()).ok_or_else(|| anyhow::anyhow!("no grad_qkv"))?.clone();
        let rhs = dot_f32(&qkv, &gq)?;
        println!("qkv reshape->narrow(mid dim2)->reshape:");
        all_ok &= report("qkv split", lhs, rhs);
    }

    // ---- attention merge: attn[1,nh,L,dh] -> permute([0,2,1,3]).reshape[1,L,inner] ----
    {
        AutogradContext::clear();
        AutogradContext::set_enabled(true);
        let (nh, dh, lq) = (4usize, 16usize, 8usize);
        let attn = randn(&[1, nh, lq, dh], &dev)?.requires_grad_(true);
        let merged = attn.permute(&[0, 2, 1, 3])?.reshape(&[1, lq, nh * dh])?;
        let g = randn(merged.shape().dims(), &dev)?;
        let loss = merged.mul(&g)?.sum()?;
        let lhs = loss.to_vec_f32()?[0] as f64;
        let grads = loss.backward()?;
        let ga = grads.get(attn.id()).ok_or_else(|| anyhow::anyhow!("no grad_attn"))?.clone();
        let rhs = dot_f32(&attn, &ga)?;
        println!("attention permute([0,2,1,3]).reshape:");
        all_ok &= report("permute.reshape", lhs, rhs);
    }

    println!(
        "\n[VERDICT] {}",
        if all_ok {
            "all linear plumbing ops are adjoint-correct (the FD anomaly is NOT here)"
        } else {
            "a plumbing op's backward != its adjoint — gradient-wiring bug localized above"
        }
    );
    Ok(())
}
