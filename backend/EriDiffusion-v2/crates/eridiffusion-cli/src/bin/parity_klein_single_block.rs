//! STANDALONE single-block forward+backward parity (DIAGNOSTIC, uncommitted).
//!
//! diffusers Flux2 single_transformer_blocks[0]  vs  our
//! `single_block_forward_standalone` on IDENTICAL controlled inputs/weights.
//!
//! Validity gate: FORWARD cos (our out vs diffusers out, identical input) MUST
//! be > 0.999 before trusting anything about the backward. Only then do we
//! interpret the backward grad_input cos:
//!   bwd cos ~0.99  => single-block backward CORRECT  => runaway is (c) fwd divergence
//!   bwd cos ~0.5-0.8 => single-block backward BUG (b) => bisect sub-op
//!
//! Reads parity/klein_sb/sb_fixture.safetensors (from ref.py).
//!
//!   cargo build --release --bin parity_klein_single_block
//!   LD_LIBRARY_PATH=libtorch FLAME_ALLOC_POOL=0 ./target/release/parity_klein_single_block
//!
//! Sub-op bisection (only meaningful AFTER forward cos > 0.999): set
//!   FLAME_SB_DISABLE_MLP=1   zero the swiglu/mlp branch on BOTH sides
//!   FLAME_SB_DISABLE_ROPE=1  skip rope on BOTH sides
//! (the ref.py companion must be re-run with the same toggle to stay matched).

use std::collections::HashMap;
use std::path::PathBuf;

use flame_core::autograd::AutogradContext;
use flame_core::{DType, Shape, Tensor};
use eridiffusion_core::models::klein;

const TRANSFORMER: &str =
    "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors";

// Klein 9B single-block dims.
const DIM: usize = 4096;
const HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const MLP_HIDDEN: usize = 12288;
const THETA: f32 = 2000.0;
const AXES: [usize; 4] = [32, 32, 32, 32];

fn cos_rel(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb, mut diff) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        diff += (x - y) * (x - y);
    }
    (
        dot / (na.sqrt() * nb.sqrt() + 1e-30),
        diff.sqrt() / (nb.sqrt() + 1e-30),
    )
}

fn main() -> anyhow::Result<()> {
    std::env::set_var("FLAME_ALLOC_POOL", "0");
    let dir: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "parity/klein_sb".into())
        .into();
    let device = flame_core::global_cuda_device();

    // ---- load fixture ----
    let fix = flame_core::serialization::load_file(&dir.join("sb_fixture.safetensors"), &device)?;
    let get = |k: &str| -> anyhow::Result<Tensor> {
        fix.get(k)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("fixture missing {k}"))
    };
    let hidden_f32 = get("hidden_states")?.to_dtype(DType::F32)?; // [1,N,4096]
    // mods stored [1,1,4096] in the fixture; the model passes them 2D [B,dim]
    // (modulate_pre_fused does `scale.unsqueeze(1)` -> [B,1,dim]). Reshape to [1,dim].
    let mod_shift = get("mod_shift")?.to_dtype(DType::BF16)?.reshape(&[1, DIM])?; // [1,4096]
    let mod_scale = get("mod_scale")?.to_dtype(DType::BF16)?.reshape(&[1, DIM])?;
    let mod_gate = get("mod_gate")?.to_dtype(DType::BF16)?.reshape(&[1, DIM])?;
    let img_ids = get("img_ids")?.to_dtype(DType::F32)?; // [N,4]
    let g_up = get("G")?.to_dtype(DType::F32)?; // [1,N,4096]
    let ref_out = get("output")?.to_dtype(DType::F32)?;
    let ref_grad = get("grad_hidden_states")?.to_dtype(DType::F32)?;

    let n = hidden_f32.shape().dims()[1];
    println!(
        "hidden{:?} img_ids{:?} N={n}  |ref_out|={:.4e} |ref_grad|={:.4e}",
        hidden_f32.shape().dims(),
        img_ids.shape().dims(),
        ref_out.to_vec()?.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt(),
        ref_grad.to_vec()?.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt(),
    );

    // ---- load block-0 weights from the checkpoint (BF16, [out,in] orientation) ----
    let all = flame_core::serialization::load_file(&PathBuf::from(TRANSFORMER), &device)?;
    let mut lw: HashMap<String, Tensor> = HashMap::new();
    for key in [
        "single_blocks.0.linear1.weight",
        "single_blocks.0.linear2.weight",
        "single_blocks.0.norm.query_norm.scale",
        "single_blocks.0.norm.key_norm.scale",
    ] {
        let t = all
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("checkpoint missing {key}"))?
            .to_dtype(DType::BF16)?;
        lw.insert(key.to_string(), t);
    }
    println!("loaded {} block-0 weights", lw.len());

    // ---- build rope our way from the same img_ids (txt_ids empty: N is all img) ----
    let txt_ids = Tensor::zeros_dtype(Shape::from_dims(&[0, 4]), DType::F32, device.clone())?;
    let (pe_cos, pe_sin) =
        klein::parity_build_rope(&img_ids, &txt_ids, &AXES, THETA, &device)?;
    println!(
        "our pe_cos{:?} pe_sin{:?}",
        pe_cos.shape().dims(),
        pe_sin.shape().dims()
    );

    // ---- forward ----
    let _ = (DIM, MLP_HIDDEN); // kept for clarity / asserts below
    AutogradContext::clear();
    AutogradContext::set_enabled(true);
    let hidden_leaf = hidden_f32.to_dtype(DType::BF16)?.requires_grad_(true);

    let out = klein::parity_single_block_forward(
        hidden_leaf.clone(),
        [mod_shift.clone(), mod_scale.clone(), mod_gate.clone()],
        pe_cos.clone(),
        pe_sin.clone(),
        lw.clone(),
        0,
        HEADS,
        HEAD_DIM,
        DIM,
        MLP_HIDDEN,
    )?;

    let ov = out.to_dtype(DType::F32)?.to_vec()?;
    let rv = ref_out.to_vec()?;
    let (fcos, frel) = cos_rel(&ov, &rv);
    println!("=== FORWARD parity (our out vs diffusers out, IDENTICAL input) ===");
    println!("  output: cos={fcos:.6}  relL2={frel:.4e}");
    let valid = fcos > 0.999;
    if !valid {
        println!("  *** FORWARD cos <= 0.999 — HARNESS or FORWARD diverges. Backward result is NOT trustworthy. ***");
    } else {
        println!("  forward parity OK (> 0.999) — backward result below is trustworthy.");
    }

    // ---- backward: loss = sum(out * G); compare dL/d(hidden) to diffusers ----
    let loss = out
        .to_dtype(DType::F32)?
        .mul(&g_up)?
        .sum()?;
    println!("  our loss = {:.6e}  (ref loss in ref.py log)", loss.to_vec()?[0]);
    let grads = loss.backward()?;
    let dx = grads
        .get(hidden_leaf.id())
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("NO dL/d(hidden) gradient (dead backward)"))?;
    let dv = dx.to_dtype(DType::F32)?.to_vec()?;
    let rd = ref_grad.to_vec()?;
    let (bcos, brel) = cos_rel(&dv, &rd);
    let our_gn: f64 = dv.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let ref_gn: f64 = rd.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    println!("=== BACKWARD parity (dL/d(hidden) vs diffusers) ===");
    println!(
        "  grad_input: cos={bcos:.6}  relL2={brel:.4e}  |ours|={our_gn:.4e} |ref|={ref_gn:.4e} ratio={:.4}",
        our_gn / (ref_gn + 1e-30)
    );

    // ---- SELF-CONSISTENCY (POSITIVE CONTROL for the 9B full-model harness) ----
    // This block's backward is known-correct (cos>0.99 vs diffusers above). Run
    // the SAME finite-diff harness used on the 9B full model. ratio~1.0 here =>
    // harness SOUND => the 9B full-model plateau (0.67) is a real composition/
    // checkpoint defect. ratio~0.67 here => harness systematically under-reads.
    {
        let loss_at = |h: &Tensor| -> anyhow::Result<f64> {
            let _g = AutogradContext::no_grad();
            let o = klein::parity_single_block_forward(
                h.clone(),
                [mod_shift.clone(), mod_scale.clone(), mod_gate.clone()],
                pe_cos.clone(), pe_sin.clone(), lw.clone(),
                0, HEADS, HEAD_DIM, DIM, MLP_HIDDEN,
            )?;
            Ok(o.to_dtype(DType::F32)?.mul(&g_up)?.sum()?.to_vec()?[0] as f64)
        };
        let x0 = hidden_f32.to_dtype(DType::F32)?.to_vec()?;
        let shp = hidden_leaf.shape().clone();
        let gnv: f64 = dv.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let vunit: Vec<f32> = dv.iter().map(|x| (*x as f64 / (gnv + 1e-30)) as f32).collect();
        let ana: f64 = gnv; // |grad|
        println!("=== SELF-CONSISTENCY (single-block, POSITIVE CONTROL) ===");
        for eps in [0.4f32, 1.0, 2.0, 4.0, 8.0] {
            let plus: Vec<f32> = x0.iter().zip(&vunit).map(|(a, b)| a + eps * b).collect();
            let minus: Vec<f32> = x0.iter().zip(&vunit).map(|(a, b)| a - eps * b).collect();
            let xp = Tensor::from_vec(plus, shp.clone(), device.clone())?.to_dtype(DType::BF16)?;
            let xm = Tensor::from_vec(minus, shp.clone(), device.clone())?.to_dtype(DType::BF16)?;
            let lp = loss_at(&xp)?;
            let lm = loss_at(&xm)?;
            let num = (lp - lm) / (2.0 * eps as f64);
            println!("  eps={eps:.2} numeric={num:.4e} analytic(|grad|)={ana:.4e} ratio={:.3}", num / (ana + 1e-30));
        }
        println!("  (ratio~1.0 => harness SOUND, 9B 0.67 is a REAL defect; ratio~0.67 => harness under-reads)");
    }

    // ---- verdict ----
    println!("=== VERDICT ===");
    if !valid {
        println!("  INVALID: forward cos {fcos:.6} <= 0.999. Cannot judge backward. Forward diverges => localizes a FORWARD bug (c) or harness mismatch.");
    } else if bcos > 0.98 {
        println!("  (c): single-block BACKWARD is CORRECT (grad_input cos {bcos:.4}). Runaway is FORWARD divergence amplified, NOT a backward-wiring bug.");
    } else if bcos < 0.85 {
        println!("  (b): single-block BACKWARD BUG CONFIRMED (grad_input cos {bcos:.4}, forward {fcos:.6} clean). Bisect the sub-op.");
    } else {
        println!("  AMBIGUOUS: backward cos {bcos:.4} between thresholds — investigate.");
    }
    Ok(())
}
