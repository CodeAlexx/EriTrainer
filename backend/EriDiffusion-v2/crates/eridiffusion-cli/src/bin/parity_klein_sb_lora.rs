//! TASK A (DIAGNOSTIC, uncommitted): single-block LoRA-PATH backward parity at
//! LARGE ‖B‖, de-conflated.
//!
//! Injects the TRAINED block-N LoRA into BOTH diffusers Flux2 single block
//! (ref_lora.py, parallel low-rank branch into to_qkv_mlp_proj=linear1 and
//! to_out=linear2) AND our `single_block_forward_standalone` (legacy
//! adapters), on IDENTICAL controlled inputs. Compares dL/d(lora_B) cos for
//! linear1 and linear2.
//!
//! Validity gate: FORWARD cos (LoRA injected, identical input) MUST be > 0.999
//! before trusting backward. If <0.999 the injection mapping is wrong.
//!
//! Reads parity/klein_sb/sb_lora_fixture_b{BLK}.safetensors (from ref_lora.py).
//!
//!   BLK=<n> cargo build --release --bin parity_klein_sb_lora
//!   LD_LIBRARY_PATH=libtorch FLAME_ALLOC_POOL=0 ./target/release/parity_klein_sb_lora <BLK>

use std::collections::HashMap;
use std::path::PathBuf;

use flame_core::autograd::AutogradContext;
use flame_core::{DType, Shape, Tensor};
use eridiffusion_core::lora::LoRALinear;
use eridiffusion_core::models::klein;

const TRANSFORMER: &str =
    "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors";

const DIM: usize = 4096;
const HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const MLP_HIDDEN: usize = 12288;
const THETA: f32 = 2000.0;
const AXES: [usize; 4] = [32, 32, 32, 32];
const RANK: usize = 16;
const ALPHA: f32 = 16.0;

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

fn l2(v: &[f32]) -> f64 {
    v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt()
}

/// Build a LoRALinear with the given A[rank,in] / B[out,rank] weights injected.
fn build_lora(
    a: &Tensor,
    b: &Tensor,
    in_features: usize,
    out_features: usize,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> anyhow::Result<LoRALinear> {
    let lin = LoRALinear::new(in_features, out_features, RANK, ALPHA, device.clone(), 0)?;
    let mut src: HashMap<String, Tensor> = HashMap::new();
    src.insert("x.lora_A.weight".to_string(), a.clone());
    src.insert("x.lora_B.weight".to_string(), b.clone());
    lin.load_tensors("x", &src)?;
    Ok(lin)
}

fn main() -> anyhow::Result<()> {
    std::env::set_var("FLAME_ALLOC_POOL", "0");
    let blk: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let dir = PathBuf::from("parity/klein_sb");
    let device = flame_core::global_cuda_device();

    let fixture = dir.join(format!("sb_lora_fixture_b{blk}.safetensors"));
    let fix = flame_core::serialization::load_file(&fixture, &device)?;
    let get = |k: &str| -> anyhow::Result<Tensor> {
        fix.get(k)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("fixture missing {k}"))
    };
    let hidden_f32 = get("hidden_states")?.to_dtype(DType::F32)?;
    let mod_shift = get("mod_shift")?.to_dtype(DType::BF16)?.reshape(&[1, DIM])?;
    let mod_scale = get("mod_scale")?.to_dtype(DType::BF16)?.reshape(&[1, DIM])?;
    let mod_gate = get("mod_gate")?.to_dtype(DType::BF16)?.reshape(&[1, DIM])?;
    let img_ids = get("img_ids")?.to_dtype(DType::F32)?;
    let g_up = get("G")?.to_dtype(DType::F32)?;
    let ref_out = get("output")?.to_dtype(DType::F32)?;

    let lin1_a = get("lin1_A")?.to_dtype(DType::F32)?; // [rank,4096]
    let lin1_b = get("lin1_B")?.to_dtype(DType::F32)?; // [36864,rank]
    let lin2_a = get("lin2_A")?.to_dtype(DType::F32)?; // [rank,16384]
    let lin2_b = get("lin2_B")?.to_dtype(DType::F32)?; // [4096,rank]
    let ref_g_lin1_b = get("grad_lin1_B")?.to_dtype(DType::F32)?;
    let ref_g_lin2_b = get("grad_lin2_B")?.to_dtype(DType::F32)?;
    let ref_g_lin1_a = get("grad_lin1_A")?.to_dtype(DType::F32)?;
    let ref_g_lin2_a = get("grad_lin2_A")?.to_dtype(DType::F32)?;

    let n = hidden_f32.shape().dims()[1];
    println!(
        "block {blk}: hidden{:?} N={n}  |lin1_B|={:.3} |lin2_B|={:.3}  |ref_out|={:.4e}",
        hidden_f32.shape().dims(),
        l2(&lin1_b.to_vec()?),
        l2(&lin2_b.to_vec()?),
        l2(&ref_out.to_vec()?),
    );

    // base block weights
    let all = flame_core::serialization::load_file(&PathBuf::from(TRANSFORMER), &device)?;
    let mut lw: HashMap<String, Tensor> = HashMap::new();
    for key in [
        format!("single_blocks.{blk}.linear1.weight"),
        format!("single_blocks.{blk}.linear2.weight"),
        format!("single_blocks.{blk}.norm.query_norm.scale"),
        format!("single_blocks.{blk}.norm.key_norm.scale"),
    ] {
        let t = all
            .get(&key)
            .ok_or_else(|| anyhow::anyhow!("checkpoint missing {key}"))?
            .to_dtype(DType::BF16)?;
        lw.insert(key, t);
    }

    // rope our way
    let txt_ids = Tensor::zeros_dtype(Shape::from_dims(&[0, 4]), DType::F32, device.clone())?;
    let (pe_cos, pe_sin) = klein::parity_build_rope(&img_ids, &txt_ids, &AXES, THETA, &device)?;

    // adapters: linear1 in=4096 out=36864 ; linear2 in=16384 out=4096
    let lin1_lora = build_lora(&lin1_a, &lin1_b, DIM, 3 * DIM + 2 * MLP_HIDDEN, &device)?;
    let lin2_lora = build_lora(&lin2_a, &lin2_b, DIM + MLP_HIDDEN, DIM, &device)?;
    let lin1_b_id = lin1_lora.lora_b().id();
    let lin2_b_id = lin2_lora.lora_b().id();
    let lin1_a_id = lin1_lora.lora_a().id();
    let lin2_a_id = lin2_lora.lora_a().id();

    AutogradContext::clear();
    AutogradContext::set_enabled(true);
    let hidden_in = hidden_f32.to_dtype(DType::BF16)?; // not a leaf; only param grads matter

    let out = klein::parity_single_block_forward_lora(
        hidden_in,
        [mod_shift, mod_scale, mod_gate],
        pe_cos,
        pe_sin,
        lw,
        vec![lin1_lora.clone(), lin2_lora.clone()],
        blk,
        HEADS,
        HEAD_DIM,
        DIM,
        MLP_HIDDEN,
    )?;

    let ov = out.to_dtype(DType::F32)?.to_vec()?;
    let rv = ref_out.to_vec()?;
    let (fcos, frel) = cos_rel(&ov, &rv);
    println!("=== FORWARD parity (LoRA injected, IDENTICAL input) ===");
    println!("  output: cos={fcos:.6}  relL2={frel:.4e}");
    let valid = fcos > 0.999;
    if !valid {
        println!("  *** FORWARD cos <= 0.999 — injection mapping WRONG. Fix before trusting backward. ***");
    } else {
        println!("  forward parity OK (> 0.999) — backward (dL/d lora_B) below is trustworthy.");
    }

    // backward
    let loss = out.to_dtype(DType::F32)?.mul(&g_up)?.sum()?;
    println!("  our loss = {:.6e}", loss.to_vec()?[0]);
    let grads = loss.backward()?;

    let fetch = |id, name: &str| -> anyhow::Result<Vec<f32>> {
        let g = grads
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("NO grad for {name} (dead backward)"))?;
        let gf = g.to_dtype(DType::F32)?;
        let v = gf.to_vec()?;
        Ok(v)
    };

    let g_lin1_b = fetch(lin1_b_id, "lin1_B")?;
    let g_lin2_b = fetch(lin2_b_id, "lin2_B")?;
    let g_lin1_a = fetch(lin1_a_id, "lin1_A")?;
    let g_lin2_a = fetch(lin2_a_id, "lin2_A")?;

    let r1b = ref_g_lin1_b.to_vec()?;
    let r2b = ref_g_lin2_b.to_vec()?;
    let r1a = ref_g_lin1_a.to_vec()?;
    let r2a = ref_g_lin2_a.to_vec()?;

    let (c1b, rel1b) = cos_rel(&g_lin1_b, &r1b);
    let (c2b, rel2b) = cos_rel(&g_lin2_b, &r2b);
    let (c1a, rel1a) = cos_rel(&g_lin1_a, &r1a);
    let (c2a, rel2a) = cos_rel(&g_lin2_a, &r2a);

    println!("=== BACKWARD parity (dL/d param vs diffusers) ===");
    println!(
        "  linear1.lora_B: cos={c1b:.6} relL2={rel1b:.4e}  |ours|={:.4e} |ref|={:.4e} ratio={:.4}",
        l2(&g_lin1_b), l2(&r1b), l2(&g_lin1_b) / (l2(&r1b) + 1e-30)
    );
    println!(
        "  linear2.lora_B: cos={c2b:.6} relL2={rel2b:.4e}  |ours|={:.4e} |ref|={:.4e} ratio={:.4}",
        l2(&g_lin2_b), l2(&r2b), l2(&g_lin2_b) / (l2(&r2b) + 1e-30)
    );
    println!(
        "  linear1.lora_A: cos={c1a:.6} relL2={rel1a:.4e}  |ours|={:.4e} |ref|={:.4e} ratio={:.4}",
        l2(&g_lin1_a), l2(&r1a), l2(&g_lin1_a) / (l2(&r1a) + 1e-30)
    );
    println!(
        "  linear2.lora_A: cos={c2a:.6} relL2={rel2a:.4e}  |ours|={:.4e} |ref|={:.4e} ratio={:.4}",
        l2(&g_lin2_a), l2(&r2a), l2(&g_lin2_a) / (l2(&r2a) + 1e-30)
    );

    println!("=== VERDICT (block {blk}) ===");
    if !valid {
        println!("  INVALID forward cos {fcos:.6} <= 0.999 — fix injection.");
    } else {
        let worst = c1b.min(c2b);
        if worst > 0.99 {
            println!("  (A) LoRA-path BACKWARD CORRECT at scale (lora_B cos lin1={c1b:.4} lin2={c2b:.4} both >0.99). Root = forward-divergence.");
        } else {
            println!("  (A) LoRA-path BACKWARD BUG at scale (lora_B cos lin1={c1b:.4} lin2={c2b:.4}, min<0.99, forward clean {fcos:.6}). Localize the path.");
        }
    }
    Ok(())
}
