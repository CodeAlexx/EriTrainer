//! parity_ideogram4_lora_grad — BACKWARD parity oracle (the row no prior test covered).
//!
//! All earlier ideogram oracle tests (vae / encoder / predict / sampler_chain /
//! lora_b0) verified FORWARD numerics only; lora_b0's backward section was
//! "diagnostic, not gating" (finite + B-nonzero, never direction). This bin is
//! the gradient-DIRECTION test: it loads ai-toolkit's TRUE per-module LoRA grads
//! (PyTorch autograd through its tested ideogram4 transformer — gen_oracle.py),
//! computed on byte-identical inputs at t=0.5, runs OUR forward+backward on the
//! SAME inputs with the SAME A/B, and compares.
//!
//! Stage 1 (forward, re-measured here): our velocity vs oracle pred_v cosine.
//! Stage 2 (backward, THE test): per-module grad_A / grad_B cosine + |g| ratio.
//!
//! Bar note (numeric-parity-testing skill): both sides share the SAME fp8→bf16
//! base weights, so this is a CROSS-IMPL comparison whose floor is set by the
//! forward bf16-accumulation drift (~the predict-parity relL2), not a clean
//! 0.999. A uniform high-but-<1 grad cosine = drift; a per-layer COLLAPSE
//! (cos≪, or negative) = a real backward bug (klein's failure mode). The bin
//! PRINTS the numbers and localizes the worst modules; interpretation follows.
//!
//! Run (GPU):
//!   LIBTORCH=/home/alex/libs/libtorch LD_LIBRARY_PATH=$LIBTORCH/lib \
//!     cargo run --release --bin parity_ideogram4_lora_grad

use std::process::ExitCode;

use eridiffusion_core::models::ideogram;
use eridiffusion_core::models::ideogram_dit::IdeogramDit;
use eridiffusion_core::training::accumulate_parameter_grads;
use flame_core::{AutogradContext, DType, Shape, Tensor};

const ORACLE: &str =
    "/home/alex/EriTrainer/trainer/parity/ideogram_lora_grad/oracle_dump.safetensors";
const TRANSFORMER: &str =
    "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors";
// Same order as IdeogramDit::attach_block_loras and gen_oracle.py TARGETS.
const TARGETS: [&str; 6] = [
    "attention.qkv",
    "attention.o",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
    "adaln_modulation",
];
const NUM_LAYERS: usize = 34;
const RANK: usize = 16;
const ALPHA: f32 = 16.0;
const FWD_BAR: f32 = 0.999;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn l2(a: &[f32]) -> f64 {
    a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt()
}

fn velocity(out: &Tensor, nt: usize, gh: usize, gw: usize) -> anyhow::Result<Tensor> {
    let nimg = gh * gw;
    Ok(out
        .narrow(1, nt, nimg)?
        .contiguous()?
        .reshape(&[1, gh, gw, 128])?
        .permute(&[0, 3, 1, 2])?
        .contiguous()?
        .mul_scalar(-1.0)?)
}

fn run() -> anyhow::Result<bool> {
    let device = flame_core::global_cuda_device();
    let fx = flame_core::serialization::load_file(std::path::Path::new(ORACLE), &device)
        .map_err(|e| anyhow::anyhow!("load oracle: {e}"))?;

    let noisy = fx
        .get("noisy")
        .ok_or_else(|| anyhow::anyhow!("oracle missing noisy"))?
        .to_dtype(DType::F32)?;
    let target = fx
        .get("target")
        .ok_or_else(|| anyhow::anyhow!("oracle missing target"))?
        .to_dtype(DType::F32)?;
    let text = fx
        .get("text")
        .ok_or_else(|| anyhow::anyhow!("oracle missing text"))?;
    let t = fx
        .get("t")
        .ok_or_else(|| anyhow::anyhow!("oracle missing t"))?
        .to_vec_f32()?[0];
    let oracle_pred = fx
        .get("pred_v")
        .ok_or_else(|| anyhow::anyhow!("oracle missing pred_v"))?
        .to_dtype(DType::F32)?;

    let ld = noisy.shape().dims().to_vec();
    let (gh, gw) = (ld[2], ld[3]);
    println!(
        "[oracle] t={t}  noisy={:?}  target={:?}  text={:?}",
        ld,
        target.shape().dims(),
        text.shape().dims()
    );

    let mut dit = IdeogramDit::load(TRANSFORMER, device.clone())?;
    let mut params = dit.attach_block_loras(RANK, ALPHA)?;
    eridiffusion_cli::trainer_pipeline::apply_autograd_v2_grad_policy(&mut params, false, "params");

    // Overwrite each adapter's A/B with the oracle's byte-identical params, so
    // the ONLY variable is our forward+backward computation.
    for li in 0..NUM_LAYERS {
        for (ti, suf) in TARGETS.iter().enumerate() {
            let key = format!("{li}.{suf}");
            let a = fx
                .get(&format!("A.{key}"))
                .ok_or_else(|| anyhow::anyhow!("oracle missing A.{key}"))?
                .to_dtype(DType::F32)?;
            let b = fx
                .get(&format!("B.{key}"))
                .ok_or_else(|| anyhow::anyhow!("oracle missing B.{key}"))?
                .to_dtype(DType::F32)?;
            let idx = li * 6 + ti;
            params[2 * idx].set_data(a)?;
            params[2 * idx + 1].set_data(b)?;
        }
    }
    println!("[setup] {} adapters, A/B set from oracle (byte-identical)", params.len() / 2);

    // Mirror train_ideogram::step_loss exactly, fed the oracle's noisy + text.
    let packed = ideogram::build_packed_inputs(&noisy, text, gh, gw, device.clone())?;
    let (cos, sin) = ideogram::build_mrope(
        &packed.position_ids,
        ideogram::HEAD_DIM,
        ideogram::MROPE_SECTION,
        ideogram::MROPE_THETA,
        device.clone(),
    )?;
    let seq = packed.x.shape().dims()[1];
    let nt = seq - gh * gw;
    let model_t = Tensor::from_vec(vec![1.0 - t], Shape::from_dims(&[1]), device.clone())?;
    let x_bf = packed.x.to_dtype(DType::BF16)?;
    let llm_bf = packed.llm_full.to_dtype(DType::BF16)?;

    let out = dit.forward(&x_bf, &llm_bf, &model_t, &packed.indicator, &cos, &sin, None, 0)?;
    let vel = velocity(&out, nt, gh, gw)?;

    // STAGE 1 — forward (re-measured): our velocity vs oracle pred_v.
    let vc = cosine(
        &vel.to_dtype(DType::F32)?.to_vec_f32()?,
        &oracle_pred.to_vec_f32()?,
    );
    println!(
        "\n[stage1 FORWARD]  velocity vs oracle pred_v  cos {vc:.6}  ({})",
        if vc >= FWD_BAR { "OK" } else { "<0.999 — bf16 cross-impl drift" }
    );

    // STAGE 2 — backward (THE test).
    let loss = vel.sub(&target)?.square()?.mean()?;
    let loss_val = loss.to_vec_f32()?[0];
    let grads = AutogradContext::backward_v2(&loss)?;
    accumulate_parameter_grads(&params, &grads)?;
    println!("[stage2 BACKWARD] loss {loss_val:.6e}");

    let (mut a_cos, mut b_cos) = (Vec::new(), Vec::new());
    let (mut a_ratio, mut b_ratio) = (Vec::new(), Vec::new());
    // worst by min(cosA,cosB); also per-target-kind mean to spot a structural drop.
    let mut worst: Vec<(f32, String, f32, f32)> = Vec::new();
    let mut by_kind: std::collections::BTreeMap<&str, (f64, usize)> =
        std::collections::BTreeMap::new();

    for li in 0..NUM_LAYERS {
        for (ti, suf) in TARGETS.iter().enumerate() {
            let key = format!("{li}.{suf}");
            let idx = li * 6 + ti;
            let oga = fx
                .get(&format!("gA.{key}"))
                .ok_or_else(|| anyhow::anyhow!("oracle missing gA.{key}"))?
                .to_vec_f32()?;
            let ogb = fx
                .get(&format!("gB.{key}"))
                .ok_or_else(|| anyhow::anyhow!("oracle missing gB.{key}"))?
                .to_vec_f32()?;
            let ga = params[2 * idx]
                .grad()
                .ok_or_else(|| anyhow::anyhow!("no grad A {key}"))?
                .to_dtype(DType::F32)?
                .to_vec_f32()?;
            let gb = params[2 * idx + 1]
                .grad()
                .ok_or_else(|| anyhow::anyhow!("no grad B {key}"))?
                .to_dtype(DType::F32)?
                .to_vec_f32()?;
            let (ca, cb) = (cosine(&ga, &oga), cosine(&gb, &ogb));
            let ra = (l2(&ga) / l2(&oga).max(1e-30)) as f32;
            let rb = (l2(&gb) / l2(&ogb).max(1e-30)) as f32;
            a_cos.push(ca);
            b_cos.push(cb);
            a_ratio.push(ra);
            b_ratio.push(rb);
            worst.push((ca.min(cb), key.clone(), ca, cb));
            let e = by_kind.entry(suf).or_insert((0.0, 0));
            e.0 += ca.min(cb) as f64;
            e.1 += 1;
        }
    }

    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let min = |v: &[f32]| v.iter().cloned().fold(f32::INFINITY, f32::min);

    println!(
        "\n[grad cosine]  A: mean {:.4} min {:.4}   B: mean {:.4} min {:.4}",
        mean(&a_cos),
        min(&a_cos),
        mean(&b_cos),
        min(&b_cos)
    );
    println!(
        "[grad |ratio|] A: mean {:.3}              B: mean {:.3}",
        mean(&a_ratio),
        mean(&b_ratio)
    );
    println!("\n[mean min-cos by target kind]");
    for (k, (s, n)) in &by_kind {
        println!("  {k:<20} {:.4}  (n={n})", s / *n as f64);
    }
    worst.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    println!("\n[worst 10 modules by min(cosA,cosB)]");
    for (m, k, ca, cb) in worst.iter().take(10) {
        println!("  {k:<26} cosA {ca:.4}  cosB {cb:.4}  (min {m:.4})");
    }

    // ---------------------------------------------------------------------
    // Gate C — SELF-CONSISTENCY (no toolkit, no cross-impl drift).
    // The loss is quadratic in B alone (and in A alone), so a CENTRAL finite
    // difference along the analytic-grad direction is EXACT up to bf16 forward
    // noise: (L(p+αĝ) − L(p−αĝ)) / (2α) must equal ‖g‖. ratio≈1 ⇒ OUR backward
    // is self-consistent with OUR forward ⇒ the cross-impl 0.30 above was drift,
    // backward is fine. ratio≠1 ⇒ OUR backward is genuinely wrong at that module.
    // ---------------------------------------------------------------------
    println!("\n[Gate C] self-consistency: directional FD of our backward vs our forward");
    println!("  (loss is quadratic in B ⇒ central FD is exact; only bf16 forward noise limits it,");
    println!("   so grow α until Δloss ≫ measured noise floor, then ratio=FD/‖g‖ is trustworthy)");
    let fwd_loss = |dit: &IdeogramDit| -> anyhow::Result<f32> {
        let _g = AutogradContext::no_grad();
        let out = dit.forward(&x_bf, &llm_bf, &model_t, &packed.indicator, &cos, &sin, None, 0)?;
        let vel = velocity(&out, nt, gh, gw)?;
        Ok(vel.sub(&target)?.square()?.mean()?.to_vec_f32()?[0])
    };
    // forward noise floor: two unperturbed evals (bf16 + any kernel nondeterminism).
    let nf0 = fwd_loss(&dit)?;
    let nf1 = fwd_loss(&dit)?;
    let noise = ((nf0 - nf1).abs() as f64).max(1e-9);
    println!("  forward noise floor |Δloss| = {noise:.3e}  (loss {nf0:.4})");
    // worst flagged (adaln/w3/qkv) + controls (o, w2) to compare ratios.
    let probe: [(usize, &str); 7] = [
        (0, "adaln_modulation"),
        (2, "adaln_modulation"),
        (16, "adaln_modulation"),
        (0, "attention.qkv"),
        (8, "feed_forward.w3"),
        (17, "attention.o"),
        (20, "feed_forward.w2"),
    ];
    println!("  module                    ratio@α.02  ratio@α.3   SNR@.3    verdict");
    for (li, suf) in probe {
        let ti = TARGETS.iter().position(|s| *s == suf).unwrap();
        let idx = li * 6 + ti;
        let g = params[2 * idx + 1].grad().unwrap().to_dtype(DType::F32)?;
        let gn = l2(&g.to_vec_f32()?);
        if gn < 1e-30 {
            println!("  {li}.{suf}  ‖g‖≈0 (skip)");
            continue;
        }
        let b0 = params[2 * idx + 1].tensor()?.to_dtype(DType::F32)?;
        let bn = l2(&b0.to_vec_f32()?).max(1e-12);
        let ghat = g.affine((1.0 / gn) as f32, 0.0)?; // unit dir; per-elem step ∝ α
        // Small α: per-element step near the bf16 param-cast ulp (rounding depresses
        // the ratio). Large α (0.3·‖B‖, step ≫ ulp): cast-safe. Loss is quadratic in
        // B so central FD is exact at either α. ratio→1 at large α ⇒ small-α was the
        // cast artifact (backward OK). Stays <1 at large α ⇒ genuine backward error.
        let mut ratios = [0.0f64; 2];
        let mut snr_large = 0.0f64;
        for (j, frac) in [0.02f64, 0.3f64].iter().enumerate() {
            let alpha = *frac * bn;
            let step = ghat.affine(alpha as f32, 0.0)?;
            params[2 * idx + 1].set_data(b0.add(&step)?)?;
            let lp = fwd_loss(&dit)?;
            params[2 * idx + 1].set_data(b0.sub(&step)?)?;
            let lm = fwd_loss(&dit)?;
            ratios[j] = ((lp - lm) as f64 / (2.0 * alpha)) / gn;
            if j == 1 {
                snr_large = ((lp - lm).abs() as f64) / noise;
            }
        }
        params[2 * idx + 1].set_data(b0.clone())?; // restore
        let verdict = if snr_large < 8.0 {
            "SWAMPED"
        } else if (ratios[1] - 1.0).abs() < 0.15 {
            "≈1 backward OK (small-α was cast artifact)"
        } else {
            "≠1 BACKWARD WRONG (cast-safe α)"
        };
        println!(
            "  {:<24} {:>9.3}  {:>9.3}  {:>9.0}   {}",
            format!("{li}.{suf}"),
            ratios[0],
            ratios[1],
            snr_large,
            verdict
        );
    }

    Ok(vc >= FWD_BAR)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::from(0),
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("[parity] error: {e:#}");
            ExitCode::from(2)
        }
    }
}
