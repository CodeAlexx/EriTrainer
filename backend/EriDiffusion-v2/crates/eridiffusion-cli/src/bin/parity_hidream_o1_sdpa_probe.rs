use clap::Parser;
use flame_core::{AutogradContext, DType, Tensor};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "parity_hidream_o1_sdpa_probe")]
struct Args {
    #[arg(long, default_value = "/tmp/hidream_o1_train_step_ref.safetensors")]
    ref_path: PathBuf,
    #[arg(long, default_value_t = 0.005)]
    max_abs: f32,
    #[arg(long, default_value_t = 0.99999)]
    min_cos: f32,
}

#[derive(Debug, Clone)]
struct Metrics {
    cos: f32,
    max_abs: f32,
    mean_abs: f32,
    rel: f32,
    nonzero: usize,
}

fn compare(name: &str, ours: &Tensor, reference: &Tensor) -> anyhow::Result<Metrics> {
    let a = ours.to_dtype(DType::F32)?.to_vec_f32()?;
    let b = reference.to_dtype(DType::F32)?.to_vec_f32()?;
    if a.len() != b.len() {
        anyhow::bail!("{name}: len mismatch ours={} ref={}", a.len(), b.len());
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut max_ref = 0.0f32;
    let mut nonzero = 0usize;
    let mut top: Vec<(f32, usize, f32, f32)> = Vec::new();
    for (idx, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        dot += (x as f64) * (y as f64);
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
        let abs = (x - y).abs();
        if abs != 0.0 {
            nonzero += 1;
        }
        max_abs = max_abs.max(abs);
        max_ref = max_ref.max(y.abs());
        sum_abs += abs as f64;
        if top.len() < 8 {
            top.push((abs, idx, x, y));
            top.sort_by(|p, q| q.0.partial_cmp(&p.0).unwrap_or(std::cmp::Ordering::Equal));
        } else if abs > top.last().map(|v| v.0).unwrap_or(0.0) {
            *top.last_mut().unwrap() = (abs, idx, x, y);
            top.sort_by(|p, q| q.0.partial_cmp(&p.0).unwrap_or(std::cmp::Ordering::Equal));
        }
    }
    let cos = if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        (dot / (na.sqrt() * nb.sqrt())) as f32
    };
    let m = Metrics {
        cos,
        max_abs,
        mean_abs: (sum_abs / a.len().max(1) as f64) as f32,
        rel: max_abs / max_ref.max(1.0e-12),
        nonzero,
    };
    println!(
        "{name:<24}: cos={:.8} max_abs={:.9e} mean_abs={:.9e} rel={:.9e} nonzero={}/{}",
        m.cos,
        m.max_abs,
        m.mean_abs,
        m.rel,
        m.nonzero,
        a.len()
    );
    let shape = ours.shape().dims().to_vec();
    for (rank, (abs, idx, x, y)) in top.into_iter().enumerate() {
        if abs == 0.0 {
            continue;
        }
        let coord = unravel_index(idx, &shape);
        println!(
            "  top#{:<2} flat={} coord={coord:?} ours={:.9e} ref={:.9e} delta={:.9e} abs={:.9e}",
            rank + 1,
            idx,
            x,
            y,
            x - y,
            abs
        );
    }
    Ok(m)
}

fn unravel_index(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut out = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        let dim = shape[axis].max(1);
        out[axis] = idx % dim;
        idx /= dim;
    }
    out
}

fn prefix_len(token_types: &Tensor) -> anyhow::Result<usize> {
    let values = token_types.to_dtype(DType::F32)?.to_vec_f32()?;
    Ok(values.iter().take_while(|&&v| v == 0.0).count())
}

fn has_key(tensors: &std::collections::HashMap<String, Tensor>, key: &str) -> bool {
    tensors.contains_key(key)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    let device = flame_core::global_cuda_device();
    let tensors = flame_core::serialization::load_file(&args.ref_path, &device)?;
    let get = |key: &str| -> anyhow::Result<&Tensor> {
        tensors
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("ref missing key {key:?}"))
    };

    if has_key(&tensors, "q") && has_key(&tensors, "dout") {
        AutogradContext::reset();
        flame_core::config::set_default_dtype(DType::BF16);
        let q = get("q")?.to_dtype(DType::BF16)?.requires_grad_(true);
        let k = get("k")?.to_dtype(DType::BF16)?.requires_grad_(true);
        let v = get("v")?.to_dtype(DType::BF16)?.requires_grad_(true);
        let dout = get("dout")?.to_dtype(DType::BF16)?;
        let prefix = prefix_len(get("token_types")?)?;
        println!(
            "sdpa bwd probe: q={:?} k={:?} v={:?} dout={:?} prefix={}",
            q.shape().dims(),
            k.shape().dims(),
            v.shape().dims(),
            dout.shape().dims(),
            prefix
        );

        let out = flame_core::attention::sdpa_prefix_causal_full(&q, &k, &v, prefix)?;
        if let Ok(target_out) = get("out") {
            compare("sdpa.out", &out, target_out)?;
        }
        let loss = out
            .to_dtype(DType::F32)?
            .mul(&dout.to_dtype(DType::F32)?)?
            .sum()?;
        let grads = loss.backward()?;
        let dq = grads
            .get(q.id())
            .ok_or_else(|| anyhow::anyhow!("missing dq for q id {:?}", q.id()))?;
        let dk = grads
            .get(k.id())
            .ok_or_else(|| anyhow::anyhow!("missing dk for k id {:?}", k.id()))?;
        let dv = grads
            .get(v.id())
            .ok_or_else(|| anyhow::anyhow!("missing dv for v id {:?}", v.id()))?;
        let mq = compare("sdpa.dq", dq, get("dq")?)?;
        let mk = compare("sdpa.dk", dk, get("dk")?)?;
        let mv = compare("sdpa.dv", dv, get("dv")?)?;
        let ok = [mq, mk, mv]
            .iter()
            .all(|m| m.cos >= args.min_cos && m.max_abs <= args.max_abs);
        if ok {
            println!("PASS");
            return Ok(());
        }
        anyhow::bail!(
            "FAIL thresholds: min_cos={} max_abs={}",
            args.min_cos,
            args.max_abs
        );
    }

    let q = get("layer00.q_rope")?.to_dtype(DType::BF16)?;
    let k = get("layer00.k_repeat")?.to_dtype(DType::BF16)?;
    let v = get("layer00.v_repeat")?.to_dtype(DType::BF16)?;
    let target = get("layer00.sdpa_out")?;
    let prefix = prefix_len(get("token_types")?)?;
    println!(
        "sdpa probe: q={:?} k={:?} v={:?} prefix={}",
        q.shape().dims(),
        k.shape().dims(),
        v.shape().dims(),
        prefix
    );

    let out = flame_core::attention::sdpa_prefix_causal_full(&q, &k, &v, prefix)?;
    let m = compare("layer00.sdpa_out", &out, target)?;
    if m.cos >= args.min_cos && m.max_abs <= args.max_abs {
        println!("PASS");
        Ok(())
    } else {
        anyhow::bail!(
            "FAIL thresholds: min_cos={} max_abs={}",
            args.min_cos,
            args.max_abs
        )
    }
}
