//! TASK B (DIAGNOSTIC, uncommitted): full-model LoRA param-grad CROSS-IMPL cos
//! vs diffusers. Loads the trained LoRA into our full klein model, runs full
//! fwd+bwd on the SAME fwd_fixture, and compares dL/d(lora_{A,B}) per module
//! against the diffusers reference (parity/klein_fwd/lora_paramgrad_ref.safetensors
//! from ref_lora_paramgrad.py).
//!
//! Conflated by the ~4.2% forward divergence — interpret per the campaign
//! caveat. Reports per-module-family cos distribution + which families are
//! lowest. The 2026-04-28 signature is ~0.2-0.6 cos with per-block-acts ~0.99.
//!
//!   cargo build --release --bin parity_klein_lora_paramgrad_xcmp
//!   LD_LIBRARY_PATH=libtorch FLAME_ALLOC_POOL=0 ./target/release/parity_klein_lora_paramgrad_xcmp \
//!       output/klein_ckpts/klein_lora_700steps.safetensors

use std::collections::HashMap;
use std::path::PathBuf;

use flame_core::autograd::AutogradContext;
use flame_core::{DType, Shape, Tensor};
use eridiffusion_core::config::TrainConfig;
use eridiffusion_core::models::klein::KleinModel;
use eridiffusion_core::models::TrainableModel;

const TRANSFORMER: &str = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors";

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

/// Coarse family bucket for a LoRA key.
fn family(key: &str) -> &'static str {
    if key.starts_with("single_blocks") {
        if key.contains("linear1") {
            "single.linear1"
        } else {
            "single.linear2"
        }
    } else if key.contains("img_attn") {
        "double.img_attn"
    } else if key.contains("txt_attn") {
        "double.txt_attn"
    } else if key.contains("img_mlp") {
        "double.img_mlp"
    } else if key.contains("txt_mlp") {
        "double.txt_mlp"
    } else {
        "other"
    }
}

fn main() -> anyhow::Result<()> {
    std::env::set_var("FLAME_ALLOC_POOL", "0");
    let lora_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "output/klein_ckpts/klein_lora_700steps.safetensors".into());
    let dir = PathBuf::from("parity/klein_fwd");
    let device = flame_core::global_cuda_device();

    let config = TrainConfig::from_json_path("configs/klein9b_alina.json")?;
    let mut model = KleinModel::load(&[PathBuf::from(TRANSFORMER)], &config, device.clone())?;
    eprintln!("[xcmp] loading trained LoRA from {lora_path}");
    model.load_weights(&lora_path)?;

    let fix = flame_core::serialization::load_file(&dir.join("fwd_fixture.safetensors"), &device)?;
    let x_t = fix.get("x_t").unwrap().to_dtype(DType::BF16)?;
    let txt = fix.get("text_embedding").unwrap().to_dtype(DType::BF16)?;
    let target = fix.get("target").unwrap().to_dtype(DType::F32)?;
    let ts = fix.get("timestep").unwrap().to_vec()?;
    let timestep = Tensor::from_vec(vec![ts[0]], Shape::from_dims(&[1]), device.clone())?;

    // reference grads
    let refg = flame_core::serialization::load_file(&dir.join("lora_paramgrad_ref.safetensors"), &device)?;

    AutogradContext::clear();
    AutogradContext::set_enabled(true);
    let vel = model.forward_train(&x_t, &txt, &timestep, None)?;
    // forward velocity cos vs the LoRA-injected diffusers velocity (validity context)
    if let Some(vref) = refg.get("velocity_lora_ref") {
        let ov = vel.to_dtype(DType::F32)?.to_vec()?;
        let rv = vref.to_dtype(DType::F32)?.to_vec()?;
        if ov.len() == rv.len() {
            let (vc, vr) = cos_rel(&ov, &rv);
            println!("FORWARD velocity cos (ours vs diffusers, LoRA injected) = {vc:.6}  relL2={vr:.4e}");
        } else {
            println!("velocity len mismatch: ours {} ref {}", ov.len(), rv.len());
        }
    }
    let loss = vel.to_dtype(DType::F32)?.sub(&target)?.square()?.mean()?;
    let loss0 = loss.to_vec()?[0] as f64;
    println!("our full-model loss = {loss0:.6e}");
    let grads = loss.backward()?;

    // map name -> our grad, compare to ref
    let named = model.named_parameters();
    println!("named params: {}", named.len());

    struct Row {
        key: String,
        is_b: bool,
        cos: f64,
        rel: f64,
        our_n: f64,
        ref_n: f64,
    }
    let mut rows: Vec<Row> = Vec::new();
    let mut missing_ours = 0usize;
    let mut missing_ref = 0usize;
    for (name, param) in &named {
        // name is e.g. "single_blocks.0.linear1.lora_A.weight"
        // ref key is "grad_single_blocks.0.linear1.lora_A"
        let stripped = name.trim_end_matches(".weight"); // "...lora_A"
        let refkey = format!("grad_{stripped}");
        let our = match grads.get(param.id()) {
            Some(g) => g.to_dtype(DType::F32)?.to_vec()?,
            None => {
                missing_ours += 1;
                continue;
            }
        };
        let rg = match refg.get(&refkey) {
            Some(t) => t.to_dtype(DType::F32)?.to_vec()?,
            None => {
                missing_ref += 1;
                continue;
            }
        };
        if our.len() != rg.len() {
            eprintln!("len mismatch {name}: ours {} ref {}", our.len(), rg.len());
            continue;
        }
        let (c, r) = cos_rel(&our, &rg);
        rows.push(Row {
            key: stripped.to_string(),
            is_b: stripped.ends_with("lora_B"),
            cos: c,
            rel: r,
            our_n: l2(&our),
            ref_n: l2(&rg),
        });
    }
    println!("compared {} params (missing ours={missing_ours}, missing ref={missing_ref})", rows.len());

    // overall distribution for lora_B (the runaway-relevant params)
    let b_cos: Vec<f64> = rows.iter().filter(|r| r.is_b).map(|r| r.cos).collect();
    let a_cos: Vec<f64> = rows.iter().filter(|r| !r.is_b).map(|r| r.cos).collect();
    let stat = |v: &[f64]| -> (f64, f64, f64) {
        if v.is_empty() {
            return (f64::NAN, f64::NAN, f64::NAN);
        }
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = s.iter().sum::<f64>() / s.len() as f64;
        (s[0], mean, s[s.len() - 1])
    };
    let (bmin, bmean, bmax) = stat(&b_cos);
    let (amin, amean, amax) = stat(&a_cos);
    println!("\n=== dL/d(lora_B) cos vs diffusers ===  min={bmin:.4} mean={bmean:.4} max={bmax:.4}  (n={})", b_cos.len());
    println!("=== dL/d(lora_A) cos vs diffusers ===  min={amin:.4} mean={amean:.4} max={amax:.4}  (n={})", a_cos.len());

    // per-family (lora_B only)
    let fams = [
        "single.linear1",
        "single.linear2",
        "double.img_attn",
        "double.txt_attn",
        "double.img_mlp",
        "double.txt_mlp",
    ];
    println!("\n=== per-family dL/d(lora_B) cos (sorted, lowest family = the lead) ===");
    let mut fam_rows: Vec<(&str, f64, f64, f64, usize)> = Vec::new();
    for f in fams {
        let v: Vec<f64> = rows
            .iter()
            .filter(|r| r.is_b && family(&r.key) == f)
            .map(|r| r.cos)
            .collect();
        if v.is_empty() {
            continue;
        }
        let (mn, me, mx) = stat(&v);
        fam_rows.push((f, mn, me, mx, v.len()));
    }
    fam_rows.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap()); // by mean cos
    for (f, mn, me, mx, n) in &fam_rows {
        println!("  {f:18} cos min={mn:.4} mean={me:.4} max={mx:.4}  (n={n})");
    }

    // depth ordering: cos vs block for single.linear2 (nearest loss) and
    // single.linear1, to test forward-divergence-amplification (cos high near
    // loss / late single blocks, collapsing toward deep/early double blocks).
    println!("\n=== single.linear2 lora_B cos by block (24=nearest loss, 0=deepest single) ===");
    let mut by_blk: Vec<(usize, f64)> = rows
        .iter()
        .filter(|r| r.is_b && r.key.starts_with("single_blocks") && r.key.contains("linear2"))
        .map(|r| {
            let n: usize = r.key.split('.').nth(1).unwrap().parse().unwrap();
            (n, r.cos)
        })
        .collect();
    by_blk.sort_by_key(|x| x.0);
    for (n, c) in &by_blk {
        println!("  single_blocks.{n}.linear2.lora_B  cos={c:.4}");
    }

    // lowest-cos individual lora_B params
    let mut bsorted: Vec<&Row> = rows.iter().filter(|r| r.is_b).collect();
    bsorted.sort_by(|a, b| a.cos.partial_cmp(&b.cos).unwrap());
    println!("\n=== 12 lowest-cos lora_B params ===");
    for r in bsorted.iter().take(12) {
        println!(
            "  {:42} cos={:.4} relL2={:.3e}  |ours|={:.3e} |ref|={:.3e} ratio={:.3}",
            r.key, r.cos, r.rel, r.our_n, r.ref_n, r.our_n / (r.ref_n + 1e-30)
        );
    }

    Ok(())
}
