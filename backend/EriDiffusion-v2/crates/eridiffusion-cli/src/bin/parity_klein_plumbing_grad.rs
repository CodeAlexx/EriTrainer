//! Adjoint (transpose) self-test for klein's LINEAR fused tensor-plumbing ops.
//!
//! For any LINEAR op f, the true backward is the adjoint: <f(x), G> == <x, f^T(G)>.
//! We run f(x) with autograd, inject upstream grad G via loss=<f(x),G>, backward
//! to get f^T(G)=grad_x, and check <f(x),G> == <x, grad_x>. Exact (no finite-diff,
//! no BF16 perturbation confound, no diffusers). A gross mismatch => the fused
//! backward is NOT the adjoint of its forward = a graph-wiring bug. (BF16 rounding
//! gives ~<1% agreement when correct; a wiring bug gives a large relative gap.)
//!
//!   cargo run --release --bin parity_klein_plumbing_grad

use flame_core::autograd::AutogradContext;
use flame_core::{DType, Shape, Tensor};

fn dot_f32(a: &Tensor, b: &Tensor) -> anyhow::Result<f64> {
    let a = a.to_dtype(DType::F32)?.to_vec()?;
    let b = b.to_dtype(DType::F32)?.to_vec()?;
    assert_eq!(a.len(), b.len(), "dot length mismatch");
    Ok(a.iter().zip(&b).map(|(&x, &y)| x as f64 * y as f64).sum())
}

fn randn(shape: &[usize], dev: &std::sync::Arc<flame_core::CudaDevice>) -> anyhow::Result<Tensor> {
    Ok(Tensor::randn(Shape::from_dims(shape), 0.0, 1.0, dev.clone())?.to_dtype(DType::BF16)?)
}

fn report(name: &str, lhs: f64, rhs: f64) {
    let rel = (lhs - rhs).abs() / (rhs.abs().max(lhs.abs()) + 1e-30);
    let verdict = if rel < 0.02 { "OK (adjoint holds)" } else { "*** BUG: backward != adjoint ***" };
    println!("  {name:<28} <f(x),G>={lhs:+.6e}  <x,f^T(G)>={rhs:+.6e}  rel={rel:.3e}  {verdict}");
}

fn main() -> anyhow::Result<()> {
    std::env::set_var("FLAME_ALLOC_POOL", "0");
    let dev = flame_core::global_cuda_device();
    let (b, n_txt, n_img, h, d) = (1usize, 256, 512, 24, 128);
    let inner = h * d; // 3072
    let n = n_txt + n_img;
    println!("config: b={b} n_txt={n_txt} n_img={n_img} h={h} d={d} inner={inner}");

    // ---- qkv_split_permute_bf16: [b,n,3*inner] -> (q,k,v) each [b,h,n,d] ----
    {
        AutogradContext::clear();
        AutogradContext::set_enabled(true);
        let qkv = randn(&[b, n, 3 * inner], &dev)?.requires_grad_(true);
        let (q, k, v) = flame_core::bf16_ops::qkv_split_permute_bf16(&qkv, h, d)?;
        let gq = randn(q.shape().dims(), &dev)?;
        let gk = randn(k.shape().dims(), &dev)?;
        let gv = randn(v.shape().dims(), &dev)?;
        let loss = q.to_dtype(DType::F32)?.mul(&gq.to_dtype(DType::F32)?)?.sum()?
            .add(&k.to_dtype(DType::F32)?.mul(&gk.to_dtype(DType::F32)?)?.sum()?)?
            .add(&v.to_dtype(DType::F32)?.mul(&gv.to_dtype(DType::F32)?)?.sum()?)?;
        let lhs = loss.to_vec()?[0] as f64;
        let grads = loss.backward()?;
        let grad_qkv = grads.get(qkv.id()).ok_or_else(|| anyhow::anyhow!("no grad_qkv"))?.clone();
        let rhs = dot_f32(&qkv, &grad_qkv)?;
        println!("qkv_split_permute_bf16:");
        report("adjoint", lhs, rhs);
    }

    // ---- attn_split_txt_img_bf16: [b,h,n,d] -> (txt[b,n_txt,h*d], img[b,n_img,h*d]) ----
    {
        AutogradContext::clear();
        AutogradContext::set_enabled(true);
        let attn = randn(&[b, h, n, d], &dev)?.requires_grad_(true);
        let (txt_out, img_out) = flame_core::bf16_ops::attn_split_txt_img_bf16(&attn, n_txt, n_img)?;
        let gt = randn(txt_out.shape().dims(), &dev)?;
        let gi = randn(img_out.shape().dims(), &dev)?;
        let loss = txt_out.to_dtype(DType::F32)?.mul(&gt.to_dtype(DType::F32)?)?.sum()?
            .add(&img_out.to_dtype(DType::F32)?.mul(&gi.to_dtype(DType::F32)?)?.sum()?)?;
        let lhs = loss.to_vec()?[0] as f64;
        let grads = loss.backward()?;
        let grad_attn = grads.get(attn.id()).ok_or_else(|| anyhow::anyhow!("no grad_attn"))?.clone();
        let rhs = dot_f32(&attn, &grad_attn)?;
        println!("attn_split_txt_img_bf16:");
        report("adjoint", lhs, rhs);
    }

    // ---- Tensor::cat along seq (dim 2 of [b,h,n,d]) -> backward should split ----
    {
        AutogradContext::clear();
        AutogradContext::set_enabled(true);
        let a = randn(&[b, h, n_txt, d], &dev)?.requires_grad_(true);
        let bb = randn(&[b, h, n_img, d], &dev)?.requires_grad_(true);
        let c = Tensor::cat(&[&a, &bb], 2)?;
        let gc = randn(c.shape().dims(), &dev)?;
        let loss = c.to_dtype(DType::F32)?.mul(&gc.to_dtype(DType::F32)?)?.sum()?;
        let lhs = loss.to_vec()?[0] as f64;
        let grads = loss.backward()?;
        let ga = grads.get(a.id()).ok_or_else(|| anyhow::anyhow!("no grad_a"))?.clone();
        let gb = grads.get(bb.id()).ok_or_else(|| anyhow::anyhow!("no grad_b"))?.clone();
        let rhs = dot_f32(&a, &ga)? + dot_f32(&bb, &gb)?;
        println!("Tensor::cat(dim=2):");
        report("adjoint", lhs, rhs);
    }

    // ---- SINGLE-BLOCK pattern: qkv_split_permute on a NARROWED (strided last-dim)
    //      input. Real single block: qkv = qkv_mlp.narrow(2,0,qkv_dim) -> qkv_split.
    //      (earlier qkv_split test used a CONTIGUOUS input; this is the untested case) ----
    {
        AutogradContext::clear();
        AutogradContext::set_enabled(true);
        let mlp = 512usize;
        let qkv_mlp = randn(&[b, n, 3 * inner + 2 * mlp], &dev)?.requires_grad_(true);
        let qkv = qkv_mlp.narrow(2, 0, 3 * inner)?; // STRIDED last-dim narrow
        let (q, k, v) = flame_core::bf16_ops::qkv_split_permute_bf16(&qkv, h, d)?;
        let gq = randn(q.shape().dims(), &dev)?;
        let gk = randn(k.shape().dims(), &dev)?;
        let gv = randn(v.shape().dims(), &dev)?;
        let loss = q.to_dtype(DType::F32)?.mul(&gq.to_dtype(DType::F32)?)?.sum()?
            .add(&k.to_dtype(DType::F32)?.mul(&gk.to_dtype(DType::F32)?)?.sum()?)?
            .add(&v.to_dtype(DType::F32)?.mul(&gv.to_dtype(DType::F32)?)?.sum()?)?;
        let lhs = loss.to_vec()?[0] as f64;
        let grads = loss.backward()?;
        let g = grads.get(qkv_mlp.id()).ok_or_else(|| anyhow::anyhow!("no g"))?.clone();
        let rhs = dot_f32(&qkv_mlp, &g)?;
        println!("qkv_split_permute on NARROWED (strided last-dim) input:");
        report("adjoint", lhs, rhs);
    }

    // ---- SINGLE-BLOCK cat([attn_out, mlp_act], 2): cat along LAST dim, 3D, unequal ----
    {
        AutogradContext::clear();
        AutogradContext::set_enabled(true);
        let mlp = 512usize;
        let a = randn(&[b, n, inner], &dev)?.requires_grad_(true);  // attn_out [b,n,h*d]
        let bb = randn(&[b, n, mlp], &dev)?.requires_grad_(true);   // mlp_act [b,n,mlp]
        let c = Tensor::cat(&[&a, &bb], 2)?;
        let gc = randn(c.shape().dims(), &dev)?;
        let loss = c.to_dtype(DType::F32)?.mul(&gc.to_dtype(DType::F32)?)?.sum()?;
        let lhs = loss.to_vec()?[0] as f64;
        let grads = loss.backward()?;
        let ga = grads.get(a.id()).ok_or_else(|| anyhow::anyhow!("no ga"))?.clone();
        let gb = grads.get(bb.id()).ok_or_else(|| anyhow::anyhow!("no gb"))?.clone();
        let rhs = dot_f32(&a, &ga)? + dot_f32(&bb, &gb)?;
        println!("Tensor::cat(dim=2, LAST dim, 3D unequal) [single-block attn|mlp]:");
        report("adjoint", lhs, rhs);
    }

    // ---- narrow MID dim, used DIRECTLY (klein block_out img/txt split @1751-2,
    //      img_only @2059): x[b,seq,dim] -> narrow(1,0,k) + narrow(1,k,seq-k) ----
    {
        AutogradContext::clear();
        AutogradContext::set_enabled(true);
        let seq = n; let dim = h * d; let k = n_txt;
        let x = randn(&[b, seq, dim], &dev)?.requires_grad_(true);
        let a = x.narrow(1, 0, k)?;
        let bb = x.narrow(1, k, seq - k)?;
        let ga = randn(a.shape().dims(), &dev)?;
        let gb = randn(bb.shape().dims(), &dev)?;
        let loss = a.to_dtype(DType::F32)?.mul(&ga.to_dtype(DType::F32)?)?.sum()?
            .add(&bb.to_dtype(DType::F32)?.mul(&gb.to_dtype(DType::F32)?)?.sum()?)?;
        let lhs = loss.to_vec()?[0] as f64;
        let grads = loss.backward()?;
        let gx = grads.get(x.id()).ok_or_else(|| anyhow::anyhow!("no gx"))?.clone();
        let rhs = dot_f32(&x, &gx)?;
        println!("narrow MID-dim (dim=1) x2, used directly (block_out / img_only split):");
        report("adjoint", lhs, rhs);
    }

    // ---- ISOLATION: attn_split is narrow(mid dim) -> permute([0,2,1,3]) -> reshape ----
    let adj_single = |label: &str, contig: bool| -> anyhow::Result<()> {
        AutogradContext::clear();
        AutogradContext::set_enabled(true);
        let x = randn(&[b, h, n, d], &dev)?.requires_grad_(true);
        let narrowed = x.narrow(2, n_txt, n_img)?; // img slice, OFFSET along mid dim
        let narrowed = if contig { narrowed.contiguous()? } else { narrowed };
        let y = narrowed.permute(&[0, 2, 1, 3])?.reshape(&[b, n_img, h * d])?;
        let g = randn(y.shape().dims(), &dev)?;
        let loss = y.to_dtype(DType::F32)?.mul(&g.to_dtype(DType::F32)?)?.sum()?;
        let lhs = loss.to_vec()?[0] as f64;
        let grads = loss.backward()?;
        let gx = grads.get(x.id()).ok_or_else(|| anyhow::anyhow!("no grad"))?.clone();
        let rhs = dot_f32(&x, &gx)?;
        report(label, lhs, rhs);
        Ok(())
    };
    println!("isolation (narrow mid-dim -> permute -> reshape):");
    adj_single("  narrow.permute.reshape", false)?;
    adj_single("  narrow.CONTIGUOUS.permute.reshape", true)?;

    // ---- PIN THE EXACT OP: isolate each op applied to the offset (mid-narrowed) view ----
    // adj_op(name, f): adjoint of f on a fresh mid-narrowed (offset) view.
    let adj_op = |label: &str, f: &dyn Fn(&Tensor) -> anyhow::Result<Tensor>| -> anyhow::Result<()> {
        AutogradContext::clear();
        AutogradContext::set_enabled(true);
        let x = randn(&[b, h, n, d], &dev)?.requires_grad_(true);
        let narrowed = x.narrow(2, n_txt, n_img)?; // OFFSET along mid dim
        let y = f(&narrowed)?;
        let g = randn(y.shape().dims(), &dev)?;
        let loss = y.to_dtype(DType::F32)?.mul(&g.to_dtype(DType::F32)?)?.sum()?;
        let lhs = loss.to_vec()?[0] as f64;
        let grads = loss.backward()?;
        let gx = grads.get(x.id()).ok_or_else(|| anyhow::anyhow!("no grad"))?.clone();
        report(label, lhs, dot_f32(&x, &gx)?);
        Ok(())
    };
    println!("isolation (single op on the OFFSET mid-narrowed view):");
    adj_op("  narrow -> contiguous", &|t| Ok(t.contiguous()?))?;
    adj_op("  narrow -> permute (views only)", &|t| Ok(t.permute(&[0, 2, 1, 3])?))?;
    adj_op("  narrow -> permute -> contiguous", &|t| Ok(t.permute(&[0, 2, 1, 3])?.contiguous()?))?;

    // ---- COMPOSED joint-attention linear skeleton (the REAL double-block backward chain):
    //      qkv_split_permute -> cat([txt,img], dim=2) -> [sdpa as IDENTITY] -> attn_split.
    //      Each piece passes adjoint alone; this tests whether the CHAIN does. The backward
    //      here is: attn_split^T -> cat^T (mid-dim narrow into txt/img) -> qkv_split_permute^T. ----
    {
        AutogradContext::clear();
        AutogradContext::set_enabled(true);
        let img_qkv = randn(&[b, n_img, 3 * inner], &dev)?.requires_grad_(true);
        let txt_qkv = randn(&[b, n_txt, 3 * inner], &dev)?.requires_grad_(true);
        let (img_q, _ik, _iv) = flame_core::bf16_ops::qkv_split_permute_bf16(&img_qkv, h, d)?;
        let (txt_q, _tk, _tv) = flame_core::bf16_ops::qkv_split_permute_bf16(&txt_qkv, h, d)?;
        let q = Tensor::cat(&[&txt_q, &img_q], 2)?; // [b,h,n,d], cat of permuted views (txt first)
        let attn_out = q; // sdpa replaced by identity to keep it linear
        let (txt_out, img_out) =
            flame_core::bf16_ops::attn_split_txt_img_bf16(&attn_out, n_txt, n_img)?;
        let gt = randn(txt_out.shape().dims(), &dev)?;
        let gi = randn(img_out.shape().dims(), &dev)?;
        let loss = txt_out.to_dtype(DType::F32)?.mul(&gt.to_dtype(DType::F32)?)?.sum()?
            .add(&img_out.to_dtype(DType::F32)?.mul(&gi.to_dtype(DType::F32)?)?.sum()?)?;
        let lhs = loss.to_vec()?[0] as f64;
        let grads = loss.backward()?;
        let gimg = grads.get(img_qkv.id()).ok_or_else(|| anyhow::anyhow!("no g img_qkv"))?.clone();
        let gtxt = grads.get(txt_qkv.id()).ok_or_else(|| anyhow::anyhow!("no g txt_qkv"))?.clone();
        let rhs = dot_f32(&img_qkv, &gimg)? + dot_f32(&txt_qkv, &gtxt)?;
        println!("COMPOSED joint-attn skeleton (qkv_split->cat(dim2)->identity->attn_split):");
        report("  qkv->cat->attn_split", lhs, rhs);
    }

    // permute([0,2,1,3]) alone on a fresh contiguous tensor
    {
        AutogradContext::clear();
        AutogradContext::set_enabled(true);
        let x = randn(&[b, h, n, d], &dev)?.requires_grad_(true);
        let y = x.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?;
        let g = randn(y.shape().dims(), &dev)?;
        let loss = y.to_dtype(DType::F32)?.mul(&g.to_dtype(DType::F32)?)?.sum()?;
        let lhs = loss.to_vec()?[0] as f64;
        let grads = loss.backward()?;
        let gx = grads.get(x.id()).ok_or_else(|| anyhow::anyhow!("no grad"))?.clone();
        report("  permute.reshape (no narrow)", lhs, dot_f32(&x, &gx)?);
    }

    // ---- STRIDE-SENSITIVITY: op backward must give the SAME input-grad whether
    // the input is contiguous or a strided view with IDENTICAL values. q/k feed
    // head_rms_norm and rope as PERMUTED (strided) views out of qkv_split_permute.
    let grad_of = |build: &dyn Fn(&Tensor) -> anyhow::Result<Tensor>, x: &Tensor| -> anyhow::Result<(Tensor, Vec<f32>)> {
        let g_seed = randn(x.shape().dims(), &dev)?; // same seed reused by caller via closure capture? no — pass in
        let y = build(x)?;
        let loss = y.to_dtype(DType::F32)?.mul(&g_seed.to_dtype(DType::F32)?)?.sum()?;
        let grads = loss.backward()?;
        let gx = grads.get(x.id()).ok_or_else(|| anyhow::anyhow!("no grad"))?.to_dtype(DType::F32)?;
        Ok((g_seed, gx.to_vec()?))
    };
    let _ = grad_of; // (kept simple below instead)

    fn cos(a: &[f32], b: &[f32]) -> f64 {
        let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) { d += x as f64 * y as f64; na += (x as f64).powi(2); nb += (y as f64).powi(2); }
        d / (na.sqrt() * nb.sqrt() + 1e-30)
    }

    // Build a base [b,n,h,d] and a strided q=[b,h,n,d] view (permuted) + its
    // contiguous twin (identical values). Reuse ONE grad seed G for both.
    let base = randn(&[b, n, h, d], &dev)?;
    let g_seed = randn(&[b, h, n, d], &dev)?.to_dtype(DType::F32)?.to_vec()?;

    let run_strided_vs_contig = |label: &str, op: &dyn Fn(&Tensor) -> anyhow::Result<Tensor>| -> anyhow::Result<()> {
        // strided input: permute view of base
        AutogradContext::clear(); AutogradContext::set_enabled(true);
        let bs = base.clone().requires_grad_(true);
        let q_strided = bs.permute(&[0, 2, 1, 3])?; // [b,h,n,d], strided view
        let y_s = op(&q_strided)?;
        let gtail: Vec<f32> = g_seed[..y_s.to_dtype(DType::F32)?.to_vec()?.len()].to_vec();
        let gt = Tensor::from_vec(gtail.clone(), y_s.shape().clone(), dev.clone())?;
        let loss_s = y_s.to_dtype(DType::F32)?.mul(&gt.to_dtype(DType::F32)?)?.sum()?;
        let grads_s = loss_s.backward()?;
        let gb_s = grads_s.get(bs.id()).ok_or_else(|| anyhow::anyhow!("no grad strided"))?.to_dtype(DType::F32)?.to_vec()?;

        // contiguous twin: materialize q, run op, then we need grad w.r.t the SAME base layout.
        AutogradContext::clear(); AutogradContext::set_enabled(true);
        let bc = base.clone().requires_grad_(true);
        let q_contig = bc.permute(&[0, 2, 1, 3])?.contiguous()?; // identical values, contiguous
        let y_c = op(&q_contig)?;
        let gt2 = Tensor::from_vec(gtail, y_c.shape().clone(), dev.clone())?;
        let loss_c = y_c.to_dtype(DType::F32)?.mul(&gt2.to_dtype(DType::F32)?)?.sum()?;
        let grads_c = loss_c.backward()?;
        let gb_c = grads_c.get(bc.id()).ok_or_else(|| anyhow::anyhow!("no grad contig"))?.to_dtype(DType::F32)?.to_vec()?;

        let c = cos(&gb_s, &gb_c);
        let verdict = if c > 0.999 { "OK (stride-invariant)" } else { "*** BUG: backward stride-sensitive ***" };
        println!("  {label:<26} cos(strided_grad, contig_grad)={c:.6}  {verdict}");
        Ok(())
    };

    // ---- cat of STRIDED (permuted) inputs — real double-block does cat([txt_q,img_q])
    // where txt_q,img_q are permuted views out of qkv_split_permute. ----
    {
        AutogradContext::clear();
        AutogradContext::set_enabled(true);
        let base_t = randn(&[b, n_txt, h, d], &dev)?.requires_grad_(true);
        let base_i = randn(&[b, n_img, h, d], &dev)?.requires_grad_(true);
        let a = base_t.permute(&[0, 2, 1, 3])?; // [b,h,n_txt,d] strided
        let bb = base_i.permute(&[0, 2, 1, 3])?; // [b,h,n_img,d] strided
        let c = Tensor::cat(&[&a, &bb], 2)?;
        let gc = randn(c.shape().dims(), &dev)?;
        let loss = c.to_dtype(DType::F32)?.mul(&gc.to_dtype(DType::F32)?)?.sum()?;
        let lhs = loss.to_vec()?[0] as f64;
        let grads = loss.backward()?;
        let ga = grads.get(base_t.id()).ok_or_else(|| anyhow::anyhow!("no gt"))?.clone();
        let gb = grads.get(base_i.id()).ok_or_else(|| anyhow::anyhow!("no gi"))?.clone();
        let rhs = dot_f32(&base_t, &ga)? + dot_f32(&base_i, &gb)?;
        println!("Tensor::cat(dim=2) of STRIDED (permuted) inputs:");
        report("adjoint", lhs, rhs);
    }

    println!("stride-sensitivity (input = permuted/strided view, q/k path):");
    // head_rms_norm: reshape([b*h*n,d]) -> rms_norm -> reshape back
    let scale = randn(&[d], &dev)?;
    run_strided_vs_contig("head_rms_norm", &|x: &Tensor| {
        let dm = x.shape().dims().to_vec(); let (b,h,n,d)=(dm[0],dm[1],dm[2],dm[3]);
        let flat = x.reshape(&[b*h*n, d])?;
        let normed = flame_core::norm::rms_norm(&flat, &[d], Some(&scale), 1e-6)?;
        Ok(normed.reshape(&[b,h,n,d])?)
    })?;
    // rope_fused_bf16 on strided q
    let cosr = randn(&[1,1,n,d/2], &dev)?;
    let sinr = randn(&[1,1,n,d/2], &dev)?;
    run_strided_vs_contig("rope_fused_bf16", &|x: &Tensor| {
        Ok(flame_core::bf16_ops::rope_fused_bf16(x, &cosr, &sinr)?)
    })?;

    // swiglu_split_lastdim_bf16: real input is a LAST-DIM narrow (strided) view of
    // qkv_mlp. Test contiguous vs narrowed-strided input with identical values.
    {
        let mlp = 512usize;
        let base2 = randn(&[b, n, 2 * mlp], &dev)?; // the gate_up values
        let g_sw = randn(&[b, n, mlp], &dev)?.to_dtype(DType::F32)?.to_vec()?;
        // contiguous
        AutogradContext::clear(); AutogradContext::set_enabled(true);
        let bc = base2.clone().requires_grad_(true);
        let yc = flame_core::bf16_ops::swiglu_split_lastdim_bf16(&bc)?;
        let gtc = Tensor::from_vec(g_sw.clone(), yc.shape().clone(), dev.clone())?;
        let lc = yc.to_dtype(DType::F32)?.mul(&gtc.to_dtype(DType::F32)?)?.sum()?;
        let gc = lc.backward()?.get(bc.id()).ok_or_else(|| anyhow::anyhow!("no gc"))?.to_dtype(DType::F32)?.to_vec()?;
        // strided: pad with extra cols, narrow back to [.., 2*mlp]
        AutogradContext::clear(); AutogradContext::set_enabled(true);
        let padded = {
            // build [b,n,2*mlp + 64] then narrow last dim to first 2*mlp, but with
            // base2 values in the front so values match the contiguous case.
            let extra = randn(&[b, n, 64], &dev)?;
            Tensor::cat(&[&base2, &extra], 2)?
        }.requires_grad_(true);
        let strided_in = padded.narrow(2, 0, 2 * mlp)?; // strided view (stride 2*mlp+64)
        let ys = flame_core::bf16_ops::swiglu_split_lastdim_bf16(&strided_in)?;
        let gts = Tensor::from_vec(g_sw, ys.shape().clone(), dev.clone())?;
        let ls = ys.to_dtype(DType::F32)?.mul(&gts.to_dtype(DType::F32)?)?.sum()?;
        let gpad = ls.backward()?.get(padded.id()).ok_or_else(|| anyhow::anyhow!("no gpad"))?.to_dtype(DType::F32)?.to_vec()?;
        // compare grad on the first 2*mlp cols (the part that matches base2)
        let cols = 2 * mlp; let stride = 2 * mlp + 64;
        let mut gs_front = Vec::with_capacity(b * n * cols);
        for row in 0..(b * n) { gs_front.extend_from_slice(&gpad[row * stride..row * stride + cols]); }
        let c = cos(&gs_front, &gc);
        let verdict = if c > 0.999 { "OK (stride-invariant)" } else { "*** BUG: backward stride-sensitive ***" };
        println!("  {:<26} cos(strided_grad, contig_grad)={c:.6}  {verdict}", "swiglu_split_lastdim");
    }

    // ---- NONLINEAR backward CORRECTNESS via clean F32 directional finite-diff.
    // adjoint (linear-only) + stride/F32-invariance can't catch a consistently-wrong
    // formula. numeric <Jv,G> = (L(x+e d)-L(x-e d))/2e along d=grad/|grad| should = |grad|.
    // Small F32 tensors -> no BF16 confound. ratio !=1 => backward formula wrong.
    fn fd_check(label: &str, dev: &std::sync::Arc<flame_core::CudaDevice>,
                shape: &[usize], build: &dyn Fn(&Tensor) -> anyhow::Result<Tensor>) -> anyhow::Result<()> {
        AutogradContext::clear(); AutogradContext::set_enabled(true);
        let n: usize = shape.iter().product();
        let x0: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.7).sin())).collect();
        let x = Tensor::from_vec(x0.clone(), Shape::from_dims(shape), dev.clone())?.requires_grad_(true);
        let y = build(&x)?;
        let m = y.to_dtype(DType::F32)?.to_vec()?.len();
        let g: Vec<f32> = (0..m).map(|i| ((i as f32 * 1.3).cos())).collect();
        let gt = Tensor::from_vec(g.clone(), y.shape().clone(), dev.clone())?;
        let loss = y.to_dtype(DType::F32)?.mul(&gt)?.sum()?;
        let grads = loss.backward()?;
        let gx = grads.get(x.id()).ok_or_else(|| anyhow::anyhow!("no grad"))?.to_dtype(DType::F32)?.to_vec()?;
        let gn: f64 = gx.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let d: Vec<f32> = gx.iter().map(|v| (*v as f64 / (gn+1e-30)) as f32).collect();
        let loss_at = |xv: &[f32]| -> anyhow::Result<f64> {
            let _ng = AutogradContext::no_grad();
            let xt = Tensor::from_vec(xv.to_vec(), Shape::from_dims(shape), dev.clone())?;
            let yy = build(&xt)?;
            Ok(yy.to_dtype(DType::F32)?.to_vec()?.iter().zip(&g).map(|(a,b)| *a as f64 * *b as f64).sum())
        };
        let eps = 1e-2f32;
        let xp: Vec<f32> = x0.iter().zip(&d).map(|(a,b)| a + eps*b).collect();
        let xm: Vec<f32> = x0.iter().zip(&d).map(|(a,b)| a - eps*b).collect();
        let num = (loss_at(&xp)? - loss_at(&xm)?) / (2.0*eps as f64);
        let ratio = num / (gn + 1e-30);
        let verdict = if (ratio-1.0).abs() < 0.05 { "OK (backward correct)" } else { "*** BUG: backward formula wrong ***" };
        println!("  {label:<22} numeric={num:.5e} analytic|grad|={gn:.5e} ratio={ratio:.4}  {verdict}");
        Ok(())
    }

    let _ = fd_check; // (rms_norm/swiglu are BF16-only; F32 fd path errors — use analytic below)

    // ---- rms_norm backward vs HAND-DERIVED analytic formula (F32 ground truth) ----
    // y_i = x_i * r * w_i,  r = 1/sqrt(mean_j x_j^2 + eps)   (over last dim, per row)
    // grad_x_i = r*w_i*g_i - (r^3 * x_i / d) * sum_j(g_j*w_j*x_j),  g = dL/dy
    // Tests the backward FORMULA (adjoint can't, op is nonlinear). klein head dim d=128.
    {
        let (rows, dd) = (8usize, 128usize);
        let xv: Vec<f32> = (0..rows*dd).map(|i| ((i as f32*0.37).sin()*1.7)).collect();
        let wv: Vec<f32> = (0..dd).map(|i| (1.0 + 0.1*((i as f32*0.21).cos()))).collect();
        let gv: Vec<f32> = (0..rows*dd).map(|i| ((i as f32*0.13).cos())).collect();
        AutogradContext::clear(); AutogradContext::set_enabled(true);
        let x = Tensor::from_vec(xv.clone(), Shape::from_dims(&[rows, dd]), dev.clone())?
            .to_dtype(DType::BF16)?.requires_grad_(true);
        let w = Tensor::from_vec(wv.clone(), Shape::from_dims(&[dd]), dev.clone())?.to_dtype(DType::BF16)?;
        let g = Tensor::from_vec(gv.clone(), Shape::from_dims(&[rows, dd]), dev.clone())?.to_dtype(DType::BF16)?;
        let y = flame_core::norm::rms_norm(&x, &[dd], Some(&w), 1e-6)?;
        let loss = y.to_dtype(DType::F32)?.mul(&g.to_dtype(DType::F32)?)?.sum()?;
        let grads = loss.backward()?;
        let ours = grads.get(x.id()).ok_or_else(|| anyhow::anyhow!("no grad_x"))?.to_dtype(DType::F32)?.to_vec()?;
        // analytic in F32
        let eps = 1e-6f64;
        let mut analytic = vec![0f32; rows*dd];
        for r in 0..rows {
            let off = r*dd;
            let ms: f64 = (0..dd).map(|j| (xv[off+j] as f64).powi(2)).sum::<f64>() / dd as f64;
            let rr = 1.0/(ms+eps).sqrt();
            let s: f64 = (0..dd).map(|j| gv[off+j] as f64 * wv[j] as f64 * xv[off+j] as f64).sum();
            for i in 0..dd {
                analytic[off+i] = (rr*wv[i] as f64*gv[off+i] as f64
                    - rr.powi(3)*xv[off+i] as f64/dd as f64 * s) as f32;
            }
        }
        let c = cos(&ours, &analytic);
        let on: f64 = ours.iter().map(|v|(*v as f64).powi(2)).sum::<f64>().sqrt();
        let an: f64 = analytic.iter().map(|v|(*v as f64).powi(2)).sum::<f64>().sqrt();
        let verdict = if c > 0.99 { "OK (backward correct)" } else { "*** BUG: rms_norm backward formula wrong ***" };
        println!("\nrms_norm backward vs analytic (F32 ground truth, d=128):");
        println!("  cos={c:.6}  |ours|={on:.4e} |analytic|={an:.4e}  ratio={:.4}  {verdict}", on/(an+1e-30));
    }

    // ---- swiglu_split_lastdim backward vs analytic (both gate/up orderings) ----
    // silu(z)=z*sigmoid(z); silu'(z)=sig(z)*(1+z*(1-sig(z))).
    // order A: [gate,up] y=silu(gate)*up ; order B: [up,gate] y=up*silu(gate).
    {
        let (rows, hh) = (8usize, 64usize);
        let n = rows*2*hh;
        let xv: Vec<f32> = (0..n).map(|i| ((i as f32*0.29).sin()*1.3)).collect();
        let gv: Vec<f32> = (0..rows*hh).map(|i| ((i as f32*0.17).cos())).collect();
        AutogradContext::clear(); AutogradContext::set_enabled(true);
        let x = Tensor::from_vec(xv.clone(), Shape::from_dims(&[rows, 2*hh]), dev.clone())?
            .to_dtype(DType::BF16)?.requires_grad_(true);
        let y = flame_core::bf16_ops::swiglu_split_lastdim_bf16(&x)?;
        let yv = y.to_dtype(DType::F32)?.to_vec()?;
        let g = Tensor::from_vec(gv.clone(), y.shape().clone(), dev.clone())?.to_dtype(DType::BF16)?;
        let loss = y.to_dtype(DType::F32)?.mul(&g.to_dtype(DType::F32)?)?.sum()?;
        let grads = loss.backward()?;
        let ours = grads.get(x.id()).ok_or_else(|| anyhow::anyhow!("no g"))?.to_dtype(DType::F32)?.to_vec()?;
        let sig = |z: f64| 1.0/(1.0+(-z).exp());
        let silu = |z: f64| z*sig(z);
        let silup = |z: f64| { let s=sig(z); s*(1.0+z*(1.0-s)) };
        // determine order by matching forward: A => y=silu(x[:h])*x[h:]
        let (mut fa, mut fb) = (0f64,0f64);
        for r in 0..rows { for i in 0..hh {
            let a = silu(xv[r*2*hh+i] as f64)*xv[r*2*hh+hh+i] as f64;
            let b = xv[r*2*hh+i] as f64*silu(xv[r*2*hh+hh+i] as f64);
            fa += (a - yv[r*hh+i] as f64).powi(2); fb += (b - yv[r*hh+i] as f64).powi(2);
        }}
        let order_a = fa < fb;
        let mut ana = vec![0f32; n];
        for r in 0..rows { for i in 0..hh {
            let (gi,ui) = if order_a {(xv[r*2*hh+i] as f64, xv[r*2*hh+hh+i] as f64)} else {(xv[r*2*hh+hh+i] as f64, xv[r*2*hh+i] as f64)};
            let gg = gv[r*hh+i] as f64;
            let dgate = gg*ui*silup(gi);
            let dup = gg*silu(gi);
            if order_a { ana[r*2*hh+i]=dgate as f32; ana[r*2*hh+hh+i]=dup as f32; }
            else { ana[r*2*hh+hh+i]=dgate as f32; ana[r*2*hh+i]=dup as f32; }
        }}
        let c = cos(&ours, &ana);
        println!("\nswiglu backward vs analytic (order {}):", if order_a {"A=[gate,up]"} else {"B=[up,gate]"});
        println!("  cos={c:.6}  {}", if c>0.99 {"OK (backward correct)"} else {"*** BUG: swiglu backward formula wrong ***"});
    }

    Ok(())
}
