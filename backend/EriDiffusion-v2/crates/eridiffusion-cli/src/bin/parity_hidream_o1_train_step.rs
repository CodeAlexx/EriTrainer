//! HiDream-O1 exact training-step parity.
//!
//! Consumes `tests/parity/hidream_o1_train_step_ref.py` output and replays the
//! same cached sample, fixed noisy patches, timestep, velocity target, and loss
//! in Rust. This isolates data/noise/timestep/objective mismatches from
//! stochastic multi-step training noise.

use clap::Parser;
use eridiffusion_core::training::offload::setup_grow_activation_cache;
use eridiffusion_core::training::training_features::{Optimizer, OptimizerKind};
use flame_core::parameter::Parameter;
use flame_core::{CudaDevice, DType, Shape, Tensor};
use eridiffusion_core::models::hidream_o1::{
    default_target_suffixes, BottleneckPatchEmbed, HiDreamO1Config, HiDreamO1WeightLoader,
    LoraRegistry, MRopePositions, TimestepEmbedder,
};
use safetensors::{Dtype as StDtype, SafeTensors};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

const DEFAULT_MODEL_PATH: &str = "/home/alex/HiDream-O1-Image-Full-weights";
const DEFAULT_REF_PATH: &str = "/tmp/hidream_o1_train_step_ref.safetensors";
const DEFAULT_LORA_REF_PATH: &str = "/tmp/hidream_o1_lora_step_ref.safetensors";
const LORA_RANK: usize = 32;
const LORA_ALPHA: f32 = 32.0;
// 252 decoder + 5 resident heads = 257 (matches Python ref with transformer_only=False).
const LORA_ADAPTERS: usize = 257;
const CLIP_GRAD_NORM: f32 = 1.0;
const FORWARD_PROBE_KEYS: &[&str] = &[
    "normed",
    "q_proj",
    "k_proj",
    "v_proj",
    "q_heads",
    "k_heads",
    "v_heads",
    "cos_half",
    "sin_half",
    "q_mean_sq",
    "k_mean_sq",
    "q_inv",
    "k_inv",
    "q_unit",
    "k_unit",
    "q_normed",
    "k_normed",
    "q_rope",
    "k_rope",
    "k_repeat",
    "v_repeat",
    "sdpa_out",
    "o_proj_in",
    "attn_out",
    "after_attn",
    "normed2",
    "gate",
    "up",
    "mlp_inner",
    "mlp_out",
    "hidden_out",
];
const FORWARD_TRAP_KEYS: &[&str] = &[
    "sdpa_out",
    "o_proj_in",
    "attn_out",
    "after_attn",
    "normed2",
    "gate",
    "up",
    "mlp_inner",
    "mlp_out",
    "hidden_out",
];

#[derive(Parser)]
#[command(name = "parity_hidream_o1_train_step")]
struct Args {
    /// HiDream-O1 weights dir.
    #[arg(long, default_value = DEFAULT_MODEL_PATH)]
    model_path: PathBuf,
    /// PyTorch training-step reference dump.
    #[arg(long, default_value = DEFAULT_REF_PATH)]
    ref_path: PathBuf,
    /// Forward-output cosine floor.
    #[arg(long, default_value_t = 0.99999)]
    min_cos: f32,
    /// Forward-output absolute error cap.
    #[arg(long, default_value_t = 0.005)]
    max_abs: f32,
    /// Forward-output relative error cap.
    #[arg(long, default_value_t = 0.01)]
    max_rel: f32,
    /// Velocity-loss relative-difference cap.
    #[arg(long, default_value_t = 0.00001)]
    max_loss_rel: f32,
    /// Dump Rust per-layer hidden states, then compare them against keys in --ref-path.
    #[arg(long, default_value = "")]
    per_layer_dump: String,
    /// Only replay and compare the pre-decoder layer-0 input assembly.
    #[arg(long)]
    predecoder_only: bool,
    /// Replay LoRA init + backward + clipping + one AdamW8bit step.
    #[arg(long)]
    lora_step: bool,
    /// PyTorch LoRA-step reference dump.
    #[arg(long, default_value = DEFAULT_LORA_REF_PATH)]
    lora_ref_path: PathBuf,
    /// LoRA tensor cosine floor.
    #[arg(long, default_value_t = 0.999)]
    lora_min_cos: f32,
    /// LoRA tensor absolute error cap.
    #[arg(long, default_value_t = 0.005)]
    lora_max_abs: f32,
    /// LoRA tensor mean absolute error cap.
    #[arg(long, default_value_t = 1.0e-5)]
    lora_max_mean_abs: f32,
    /// LoRA tensor relative error cap.
    #[arg(long, default_value_t = 0.05)]
    lora_max_rel: f32,
}

fn decode_position_ids(pos: &Tensor) -> anyhow::Result<(Vec<u32>, Vec<u32>, Vec<u32>)> {
    let dims = pos.shape().dims().to_vec();
    if dims.len() != 2 || dims[0] != 3 {
        anyhow::bail!("position_ids: expected [3, S_total], got {:?}", dims);
    }
    let s_total = dims[1];
    let flat = pos.to_dtype(DType::F32)?.to_vec_f32()?;
    let mut t = Vec::with_capacity(s_total);
    let mut h = Vec::with_capacity(s_total);
    let mut w = Vec::with_capacity(s_total);
    for i in 0..s_total {
        t.push(flat[i] as u32);
        h.push(flat[s_total + i] as u32);
        w.push(flat[2 * s_total + i] as u32);
    }
    Ok((t, h, w))
}

fn gather_image_rows(x_pred: &Tensor, vinput_mask: &Tensor) -> anyhow::Result<Tensor> {
    let xd = x_pred.shape().dims().to_vec();
    if xd.len() != 3 {
        anyhow::bail!("gather_image_rows: expected [B,S,C], got {:?}", xd);
    }
    let (_b, s_total, _c) = (xd[0], xd[1], xd[2]);
    let host = vinput_mask.to_dtype(DType::F32)?.to_vec_f32()?;
    let mut first: Option<usize> = None;
    let mut last: Option<usize> = None;
    for i in 0..s_total {
        if host[i] != 0.0 {
            first.get_or_insert(i);
            last = Some(i);
        }
    }
    let (first, last) = (
        first.ok_or_else(|| anyhow::anyhow!("vinput_mask has no image slots"))?,
        last.unwrap(),
    );
    let len = last - first + 1;
    let count = host.iter().filter(|&&x| x != 0.0).count();
    if count != len {
        anyhow::bail!(
            "non-contiguous image slots not supported (count {} != span {})",
            count,
            len
        );
    }
    Ok(x_pred.narrow(1, first, len)?)
}

#[derive(Debug, Clone, Copy)]
struct Metrics {
    cos: f32,
    max_abs: f32,
    mean_abs: f32,
    rel: f32,
}

fn parity_trap_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HIDREAM_O1_PARITY_TRAP")
            .ok()
            .as_deref()
            == Some("1")
    })
}

fn parity_trap_key(name: &str) -> bool {
    if let Ok(patterns) = std::env::var("HIDREAM_O1_DIFF_TRAP") {
        if patterns
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .any(|part| name.contains(part))
        {
            return true;
        }
    }

    let Some((prefix, suffix)) = name.split_once('.') else {
        return false;
    };
    prefix.starts_with("layer") && FORWARD_TRAP_KEYS.contains(&suffix)
}

fn dump_probe_layers_from_env() -> Vec<usize> {
    let mut out: Vec<usize> = std::env::var("HIDREAM_DUMP_PROBE_LAYERS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![0]);
    if out.is_empty() {
        out.push(0);
    }
    out.sort_unstable();
    out.dedup();
    out
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

fn print_offender_trap(name: &str, shape: &[usize], ours: &[f32], reference: &[f32]) {
    if !(parity_trap_enabled() || std::env::var("HIDREAM_O1_DIFF_TRAP").is_ok())
        || !parity_trap_key(name)
    {
        return;
    }

    let mut nonzero = 0usize;
    let mut top: Vec<(f32, usize, f32, f32)> = Vec::with_capacity(8);
    for (idx, (&x, &y)) in ours.iter().zip(reference.iter()).enumerate() {
        let abs = (x - y).abs();
        if abs == 0.0 {
            continue;
        }
        nonzero += 1;
        if top.len() < 8 {
            top.push((abs, idx, x, y));
            top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        } else if abs > top.last().map(|v| v.0).unwrap_or(0.0) {
            *top.last_mut().unwrap() = (abs, idx, x, y);
            top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        }
    }

    eprintln!(
        "[o1-trap] {name}: shape={shape:?} nonzero_diff={} / {}",
        nonzero,
        ours.len()
    );
    for (rank, (abs, idx, x, y)) in top.iter().enumerate() {
        let coord = unravel_index(*idx, shape);
        eprintln!(
            "[o1-trap] {name}: top#{:<2} flat={} coord={coord:?} ours={:.9e} ref={:.9e} delta={:.9e} abs={:.9e}",
            rank + 1,
            idx,
            x,
            y,
            x - y,
            abs
        );
    }
}

fn compare_slices(name: &str, a: &[f32], b: &[f32]) -> anyhow::Result<Metrics> {
    if a.len() != b.len() {
        anyhow::bail!("{name}: len mismatch ours={} ref={}", a.len(), b.len());
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut max_ref = 0.0f32;
    let mut sum_abs = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let xf = *x as f64;
        let yf = *y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        let y_abs = y.abs();
        if y_abs > max_ref {
            max_ref = y_abs;
        }
        sum_abs += d as f64;
    }
    let cos = if na == 0.0 && nb == 0.0 {
        if max_abs == 0.0 { 1.0 } else { 0.0 }
    } else {
        (dot / (na.sqrt() * nb.sqrt() + 1e-30)) as f32
    };
    Ok(Metrics {
        cos,
        max_abs,
        mean_abs: (sum_abs / a.len() as f64) as f32,
        rel: max_abs / max_ref.max(1.0e-12),
    })
}

fn compare_vec(name: &str, ours: &Tensor, reference: &Tensor) -> anyhow::Result<Metrics> {
    if ours.shape().dims() != reference.shape().dims() {
        anyhow::bail!(
            "{name}: shape mismatch ours={:?} ref={:?}",
            ours.shape().dims(),
            reference.shape().dims()
        );
    }
    let a = ours.to_dtype(DType::F32)?.to_vec_f32()?;
    let b = reference.to_dtype(DType::F32)?.to_vec_f32()?;
    print_offender_trap(name, ours.shape().dims(), &a, &b);
    compare_slices(name, &a, &b)
}

fn scalar1(t: &Tensor) -> anyhow::Result<f32> {
    let v = t.to_dtype(DType::F32)?.to_vec_f32()?;
    v.first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("scalar tensor is empty"))
}

fn st_view<'a>(st: &'a SafeTensors<'a>, key: &str) -> anyhow::Result<safetensors::tensor::TensorView<'a>> {
    st.tensor(key)
        .map_err(|e| anyhow::anyhow!("lora ref missing key {key:?}: {e}"))
}

fn st_f32(st: &SafeTensors, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
    let t = st_view(st, key)?;
    if t.dtype() != StDtype::F32 {
        anyhow::bail!("{key}: expected F32, got {:?}", t.dtype());
    }
    let bytes = t.data();
    if bytes.len() % 4 != 0 {
        anyhow::bail!("{key}: byte len {} not divisible by 4", bytes.len());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok((out, t.shape().to_vec()))
}

fn st_scalar(st: &SafeTensors, key: &str) -> anyhow::Result<f32> {
    let (v, _) = st_f32(st, key)?;
    v.first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("{key}: scalar tensor is empty"))
}

fn tensor_from_st(
    st: &SafeTensors,
    key: &str,
    device: &Arc<CudaDevice>,
) -> anyhow::Result<Tensor> {
    let (data, shape) = st_f32(st, key)?;
    Tensor::from_vec(data, Shape::from_dims(&shape), device.clone())
        .map_err(|e| anyhow::anyhow!("{key}: Tensor::from_vec: {e}"))
}

fn zero_like_param(p: &Parameter, device: &Arc<CudaDevice>) -> anyhow::Result<Tensor> {
    Tensor::zeros_dtype(p.shape(), DType::F32, device.clone())
        .map_err(|e| anyhow::anyhow!("zero_like_param: {e}"))
}

fn compare_tensor_to_st(
    key: &str,
    ours: &Tensor,
    st: &SafeTensors,
) -> anyhow::Result<Metrics> {
    let (reference, shape) = st_f32(st, key)?;
    if ours.shape().dims() != shape.as_slice() {
        anyhow::bail!(
            "{key}: shape mismatch ours={:?} ref={:?}",
            ours.shape().dims(),
            shape
        );
    }
    let ours = ours.to_dtype(DType::F32)?.to_vec_f32()?;
    print_offender_trap(key, &shape, &ours, &reference);
    compare_slices(key, &ours, &reference)
}

fn lora_pass(m: &Metrics, args: &Args) -> bool {
    m.cos >= args.lora_min_cos
        && m.max_abs <= args.lora_max_abs
        && m.mean_abs <= args.lora_max_mean_abs
        && m.rel <= args.lora_max_rel
}

fn forward_pass(m: &Metrics, args: &Args) -> bool {
    m.cos >= args.min_cos && m.max_abs <= args.max_abs && m.rel <= args.max_rel
}

fn print_metric(label: &str, key: &str, m: &Metrics, ok: bool) {
    println!(
        "{label:<12} {key:<28} cos={:.8} max_abs={:.6e} mean_abs={:.6e} rel={:.6e} {}",
        m.cos,
        m.max_abs,
        m.mean_abs,
        m.rel,
        if ok { "OK" } else { "FAIL" },
    );
}

fn compare_ref_tensor(
    label: &str,
    key: &str,
    ours: &Tensor,
    ref_tensors: &std::collections::HashMap<String, Tensor>,
    args: &Args,
) -> anyhow::Result<bool> {
    let reference = ref_tensors
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("ref missing key {key:?}"))?
        .to_dtype(DType::F32)?;
    let m = compare_vec(key, ours, &reference)?;
    let ok = forward_pass(&m, args);
    print_metric(label, key, &m, ok);
    Ok(ok)
}

fn compare_per_layer_dump(
    path: &str,
    ref_tensors: &std::collections::HashMap<String, Tensor>,
    cfg: &HiDreamO1Config,
    device: &Arc<CudaDevice>,
    args: &Args,
) -> anyhow::Result<bool> {
    let ours = flame_core::serialization::load_file(std::path::Path::new(path), device)
        .map_err(|e| anyhow::anyhow!("load Rust per-layer dump {path}: {e}"))?;
    let probe_layers = dump_probe_layers_from_env();
    let mut keys = Vec::with_capacity(cfg.num_layers + 2 + probe_layers.len() * FORWARD_PROBE_KEYS.len());
    keys.push("hidden_input_layer_00".to_string());
    for layer_idx in probe_layers {
        for suffix in FORWARD_PROBE_KEYS {
            keys.push(format!("layer{layer_idx:02}.{suffix}"));
        }
    }
    for i in 0..cfg.num_layers {
        keys.push(format!("hidden_layer_{i:02}"));
    }
    keys.push("hidden_final_norm".to_string());

    println!();
    println!("per-layer forward parity (all layers, no early exit):");
    let mut all_ok = true;
    let mut first_fail: Option<String> = None;
    for key in keys {
        let ours_t = ours
            .get(&key)
            .ok_or_else(|| anyhow::anyhow!("Rust per-layer dump missing key {key:?}"))?;
        let ref_t = ref_tensors
            .get(&key)
            .ok_or_else(|| anyhow::anyhow!("ref missing per-layer key {key:?}; rerun Python ref with --dump-layers"))?;
        let m = compare_vec(&key, ours_t, ref_t)?;
        let ok = forward_pass(&m, args);
        print_metric("forward", &key, &m, ok);
        if !ok {
            all_ok = false;
            if first_fail.is_none() {
                first_fail = Some(key.clone());
            }
        }
    }
    if let Some(k) = first_fail {
        println!("[o1-train-step] first failing per-layer stage: forward::{k}");
    }
    Ok(all_ok)
}

fn scatter_tms_token_for_predecoder(
    cfg: &HiDreamO1Config,
    text_emb: &Tensor,
    input_ids: &Tensor,
    t_emb: &Tensor,
    device: &Arc<CudaDevice>,
) -> anyhow::Result<Tensor> {
    let dims = text_emb.shape().dims().to_vec();
    if dims.len() != 3 {
        anyhow::bail!("predecoder scatter: text_emb must be [B,S,H], got {:?}", dims);
    }
    let (b, s_text, h) = (dims[0], dims[1], dims[2]);
    let id_dims = input_ids.shape().dims();
    if id_dims.len() != 2 || id_dims[0] != b || id_dims[1] != s_text {
        anyhow::bail!(
            "predecoder scatter: input_ids shape {:?} does not match [{},{}]",
            id_dims,
            b,
            s_text
        );
    }
    let t_dims = t_emb.shape().dims();
    if t_dims.len() != 2 || t_dims[0] != b || t_dims[1] != h {
        anyhow::bail!(
            "predecoder scatter: t_emb shape {:?} does not match [{},{}]",
            t_dims,
            b,
            h
        );
    }

    let ids = if input_ids.dtype() == DType::I32 {
        input_ids.to_vec_f32()?
    } else {
        input_ids.to_dtype(DType::F32)?.to_vec_f32()?
    };
    let mut mask = vec![0.0f32; b * s_text * h];
    let tms = cfg.tms_token_id as f32;
    for bi in 0..b {
        for si in 0..s_text {
            if ids[bi * s_text + si] == tms {
                let off = (bi * s_text + si) * h;
                for d in 0..h {
                    mask[off + d] = 1.0;
                }
            }
        }
    }
    let mask = Tensor::from_vec_dtype(
        mask,
        Shape::from_dims(&[b, s_text, h]),
        device.clone(),
        DType::BF16,
    )?;
    let t_expanded = t_emb
        .reshape(&[b, 1, h])?
        .broadcast_to(&Shape::from_dims(&[b, s_text, h]))?;
    Ok(Tensor::where_mask(&mask, &t_expanded, text_emb)?)
}

fn run_predecoder_only(
    args: &Args,
    ref_tensors: &std::collections::HashMap<String, Tensor>,
    input_ids: &Tensor,
    timestep: &Tensor,
    noisy: &Tensor,
    device: &Arc<CudaDevice>,
) -> anyhow::Result<bool> {
    let cfg = HiDreamO1Config::dev_8b();
    let loader = HiDreamO1WeightLoader::from_dir(&args.model_path)
        .map_err(|e| anyhow::anyhow!("HiDreamO1WeightLoader: {e}"))?;
    let mut weights = loader
        .load_selected_resident_weights_bf16(
            &[
                "model.x_embedder.proj1.weight",
                "model.x_embedder.proj2.weight",
                "model.x_embedder.proj2.bias",
                "model.t_embedder1.mlp.0.weight",
                "model.t_embedder1.mlp.0.bias",
                "model.t_embedder1.mlp.2.weight",
                "model.t_embedder1.mlp.2.bias",
            ],
            device,
        )
        .map_err(|e| anyhow::anyhow!("load resident weights: {e}"))?;
    let take = |m: &mut std::collections::HashMap<String, Tensor>, k: &str| -> anyhow::Result<Tensor> {
        m.remove(k)
            .ok_or_else(|| anyhow::anyhow!("missing resident weight key {k}"))
    };

    let text_emb_ref = ref_tensors
        .get("pre.text_emb")
        .ok_or_else(|| anyhow::anyhow!("ref missing key \"pre.text_emb\"; rerun Python ref with --dump-layers"))?
        .to_dtype(DType::BF16)?;

    let mut timestep_embedder = TimestepEmbedder::new(&cfg, device)?;
    timestep_embedder
        .mlp_in
        .copy_weight_from(&take(&mut weights, "model.t_embedder1.mlp.0.weight")?)?;
    timestep_embedder
        .mlp_in
        .copy_bias_from(&take(&mut weights, "model.t_embedder1.mlp.0.bias")?)?;
    timestep_embedder
        .mlp_out
        .copy_weight_from(&take(&mut weights, "model.t_embedder1.mlp.2.weight")?)?;
    timestep_embedder
        .mlp_out
        .copy_bias_from(&take(&mut weights, "model.t_embedder1.mlp.2.bias")?)?;

    let mut patch_embed = BottleneckPatchEmbed::new(&cfg, device)?;
    patch_embed
        .proj1
        .copy_weight_from(&take(&mut weights, "model.x_embedder.proj1.weight")?)?;
    patch_embed
        .proj2
        .copy_weight_from(&take(&mut weights, "model.x_embedder.proj2.weight")?)?;
    patch_embed
        .proj2
        .copy_bias_from(&take(&mut weights, "model.x_embedder.proj2.bias")?)?;

    // bisect: also expose the sinusoid output and MLP intermediate stages so we
    // can localize the 1-ULP delta inside the 2-layer MLP (Linear → SiLU →
    // Linear). The Python ref dumps the same intermediates with matching names.
    let t_freq_bf16 = TimestepEmbedder::timestep_embedding(
        timestep,
        cfg.timestep_freq_dim,
        10_000.0,
        1000.0,
        device,
    )?;
    let batch = t_freq_bf16.shape().dims()[0];
    let t_freq_3d = t_freq_bf16.reshape(&[batch, 1, cfg.timestep_freq_dim])?;
    let t_mlp0_out_3d = flame_core::ops::fused_inference::fused_linear3d_native_pytorch_parity(
        &t_freq_3d,
        &timestep_embedder.mlp_in.weight,
        timestep_embedder.mlp_in.bias.as_ref(),
    )?;
    let t_mlp0_out = t_mlp0_out_3d.reshape(&[batch, cfg.hidden_size])?;
    let t_silu_out = t_mlp0_out.silu()?;
    // Bias-add isolation: run the second Linear's GEMM WITHOUT bias and dump it,
    // to separate epilogue/bias precision from the GEMM kernel precision.
    let t_silu_3d = t_silu_out.reshape(&[batch, 1, cfg.hidden_size])?;
    let t_mlp2_nobias_3d = flame_core::ops::fused_inference::fused_linear3d_native_pytorch_parity(
        &t_silu_3d,
        &timestep_embedder.mlp_out.weight,
        None,
    )?;
    let t_mlp2_nobias = t_mlp2_nobias_3d.reshape(&[batch, cfg.hidden_size])?;
    let t_emb = timestep_embedder.forward_lora(timestep, None)?;
    let text_emb_with_t =
        scatter_tms_token_for_predecoder(&cfg, &text_emb_ref, input_ids, &t_emb, device)?;
    let patch_proj1 = flame_core::ops::fused_inference::fused_linear3d_native_pytorch_parity(
        noisy,
        &patch_embed.proj1.weight,
        patch_embed.proj1.bias.as_ref(),
    )?;
    let patch_proj1_native = flame_core::ops::fused_inference::fused_linear3d_native(
        noisy,
        &patch_embed.proj1.weight,
        patch_embed.proj1.bias.as_ref(),
    )?;
    let patch_proj1_matmul = {
        let wt = patch_embed.proj1.weight.transpose()?;
        noisy.matmul(&wt)?
    };
    let patch_proj2_nobias =
        flame_core::ops::fused_inference::fused_linear3d_native_pytorch_parity(
            &patch_proj1,
            &patch_embed.proj2.weight,
            None,
        )?;
    let patch_emb = patch_embed.forward_lora(noisy, None)?;
    let inputs_embeds = Tensor::cat(&[&text_emb_with_t, &patch_emb], 1)?;

    println!();
    println!("pre-decoder parity:");
    for (label, tensor) in [
        ("pre.native", &patch_proj1_native),
        ("pre.matmul", &patch_proj1_matmul),
    ] {
        let _ = compare_ref_tensor(label, "pre.patch_proj1", &tensor.to_dtype(DType::F32)?, ref_tensors, args)?;
    }
    for (key, tensor) in [
        ("pre.t_freq_bf16", &t_freq_bf16),
        ("pre.t_mlp0_out", &t_mlp0_out),
        ("pre.t_silu_out", &t_silu_out),
        ("pre.t_mlp2_nobias", &t_mlp2_nobias),
        ("pre.t_emb", &t_emb),
        ("pre.text_emb_with_t", &text_emb_with_t),
        ("pre.patch_proj1", &patch_proj1),
        ("pre.patch_proj2_nobias", &patch_proj2_nobias),
        ("pre.patch_emb", &patch_emb),
        ("pre.inputs_embeds", &inputs_embeds),
        ("hidden_input_layer_00", &inputs_embeds),
    ] {
        let ok = compare_ref_tensor("pre", key, &tensor.to_dtype(DType::F32)?, ref_tensors, args)?;
        if !ok {
            println!("[o1-train-step] first failing stage: predecoder::{key}");
            return Ok(false);
        }
    }
    Ok(true)
}

fn load_lora_init(
    lora: &LoraRegistry,
    st: &SafeTensors,
    device: &Arc<CudaDevice>,
) -> anyhow::Result<()> {
    let named = lora.named_parameters();
    if named.len() != LORA_ADAPTERS * 2 {
        anyhow::bail!(
            "expected {} LoRA tensors, got {}",
            LORA_ADAPTERS * 2,
            named.len()
        );
    }
    for (name, p) in named {
        let key = format!("init.{name}");
        let t = tensor_from_st(st, &key, device)?.requires_grad_(true);
        p.set_data(t)?;
    }
    Ok(())
}

fn report_lora_group(
    label: &str,
    named: &[(String, Parameter)],
    tensor_for: impl Fn(&Parameter) -> anyhow::Result<Tensor>,
    st: &SafeTensors,
    args: &Args,
) -> anyhow::Result<bool> {
    let mut first_fail: Option<(String, Metrics)> = None;
    let mut worst_abs: Option<(String, Metrics)> = None;
    for (name, p) in named {
        let ours = tensor_for(p)?;
        let key = format!("{label}.{name}");
        let m = compare_tensor_to_st(&key, &ours, st)?;
        if !lora_pass(&m, args) && first_fail.is_none() {
            first_fail = Some((key.clone(), m));
        }
        match &worst_abs {
            Some((_, w)) if w.max_abs >= m.max_abs => {}
            _ => worst_abs = Some((key, m)),
        }
    }
    if let Some((key, m)) = &worst_abs {
        println!(
            "{label:<10}: checked {:>3} tensors | worst_abs {key} cos={:.8} max_abs={:.6e} mean_abs={:.6e} rel={:.6e}",
            named.len(),
            m.cos,
            m.max_abs,
            m.mean_abs,
            m.rel
        );
    }
    if let Some((key, m)) = first_fail {
        println!(
            "{label:<10}: FAIL first={key} cos={:.8} max_abs={:.6e} mean_abs={:.6e} rel={:.6e}",
            m.cos,
            m.max_abs,
            m.mean_abs,
            m.rel
        );
        Ok(false)
    } else {
        println!("{label:<10}: OK");
        Ok(true)
    }
}

fn report_lora_selected(
    label: &str,
    keys: &[&str],
    named: &[(String, Parameter)],
    tensor_for: impl Fn(&Parameter) -> anyhow::Result<Tensor>,
    st: &SafeTensors,
) -> anyhow::Result<()> {
    println!("{label:<10}: selected tensors");
    for name in keys {
        let p = named
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p)
            .ok_or_else(|| anyhow::anyhow!("{label}: missing selected LoRA key {name}"))?;
        let key = format!("{label}.{name}");
        let ours = tensor_for(p)?;
        let m = compare_tensor_to_st(&key, &ours, st)?;
        println!(
            "  {key:<48} cos={:.8} max_abs={:.6e} mean_abs={:.6e} rel={:.6e}",
            m.cos,
            m.max_abs,
            m.mean_abs,
            m.rel
        );
    }
    Ok(())
}

fn run() -> anyhow::Result<bool> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
    let args = Args::parse();

    if !args.ref_path.exists() {
        anyhow::bail!(
            "ref dump {} not found; run tests/parity/hidream_o1_train_step_ref.py first",
            args.ref_path.display()
        );
    }
    if !args.model_path.exists() {
        anyhow::bail!("model path does not exist: {}", args.model_path.display());
    }
    if args.lora_step && !args.lora_ref_path.exists() {
        anyhow::bail!(
            "lora ref dump {} not found; run tests/parity/hidream_o1_train_step_ref.py --lora-step first",
            args.lora_ref_path.display()
        );
    }

    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();
    let disable_grow_activation_cache =
        std::env::var("FLAME_DISABLE_GROW_ACTIVATION_CACHE").ok().as_deref() == Some("1");
    let activation_cache = if args.lora_step && !disable_grow_activation_cache {
        let slab_bytes = 1usize << 30;
        let cache = setup_grow_activation_cache(&device, slab_bytes)
            .map_err(|e| anyhow::anyhow!("setup_grow_activation_cache: {e}"))?;
        log::info!(
            "[o1-train-step] grow activation cache installed: {} MB",
            slab_bytes / (1024 * 1024)
        );
        Some(cache)
    } else {
        if args.lora_step {
            log::info!("[o1-train-step] grow activation cache disabled by env");
        }
        None
    };
    let _no_grad = if args.lora_step {
        None
    } else {
        Some(flame_core::autograd::AutogradContext::no_grad())
    };

    log::info!("[o1-train-step] ref: {}", args.ref_path.display());
    let ref_tensors = flame_core::serialization::load_file(&args.ref_path, &device)
        .map_err(|e| anyhow::anyhow!("load ref {}: {e}", args.ref_path.display()))?;
    let get = |k: &str| -> anyhow::Result<&Tensor> {
        ref_tensors
            .get(k)
            .ok_or_else(|| anyhow::anyhow!("ref missing key {k:?}"))
    };

    let noisy = get("noisy")?.to_dtype(DType::BF16)?;
    let input_ids = get("input_ids")?.to_dtype(DType::I32)?;
    let position_ids = get("position_ids")?.clone();
    let vinput_mask = get("vinput_mask")?.to_dtype(DType::BF16)?;
    let token_types_bin = get("token_types")?.to_dtype(DType::BF16)?;
    let timestep = get("timestep")?.to_dtype(DType::BF16)?;
    let t_scalar = scalar1(get("t_scalar")?)?;
    let target_velocity = get("target_velocity")?.to_dtype(DType::F32)?;
    let x_pred_rows_ref = get("x_pred_rows")?.to_dtype(DType::F32)?;
    let pred_velocity_ref = get("pred_velocity")?.to_dtype(DType::F32)?;
    let loss_velocity_ref = scalar1(get("loss_velocity")?)?;

    log::info!(
        "[o1-train-step] shapes: noisy={:?} input_ids={:?} pos={:?} vmask={:?} timestep={:?}",
        noisy.shape().dims(),
        input_ids.shape().dims(),
        position_ids.shape().dims(),
        vinput_mask.shape().dims(),
        timestep.shape().dims(),
    );
    log::info!(
        "[o1-train-step] pinned t_scalar={:.6} ref_loss={:.9}",
        t_scalar,
        loss_velocity_ref
    );
    if args.lora_step {
        log::info!(
            "[o1-train-step] lora ref: {}",
            args.lora_ref_path.display()
        );
    }
    if args.predecoder_only {
        return run_predecoder_only(
            &args,
            &ref_tensors,
            &input_ids,
            &timestep,
            &noisy,
            &device,
        );
    }

    let lora_ref_buf = if args.lora_step {
        Some(std::fs::read(&args.lora_ref_path)?)
    } else {
        None
    };
    let lora_ref = match lora_ref_buf.as_ref() {
        Some(buf) => Some(SafeTensors::deserialize(buf)?),
        None => None,
    };
    let sdpa_bwd_ref_path = std::env::var("HIDREAM_SDPA_BWD_REF")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from);
    let sdpa_bwd_ref_buf = match sdpa_bwd_ref_path.as_ref() {
        Some(path) => Some(
            std::fs::read(path)
                .map_err(|e| anyhow::anyhow!("read HIDREAM_SDPA_BWD_REF {}: {e}", path.display()))?,
        ),
        None => None,
    };
    let sdpa_bwd_ref = match sdpa_bwd_ref_buf.as_ref() {
        Some(buf) => Some(SafeTensors::deserialize(buf)?),
        None => None,
    };

    let (t_pos, h_pos, w_pos) = decode_position_ids(&position_ids)?;
    let pos_view = MRopePositions {
        t: &t_pos,
        h: &h_pos,
        w: &w_pos,
    };

    let cfg = HiDreamO1Config::dev_8b();
    log::info!(
        "[o1-train-step] loading model from {}",
        args.model_path.display()
    );
    let loader = HiDreamO1WeightLoader::from_dir(&args.model_path)
        .map_err(|e| anyhow::anyhow!("HiDreamO1WeightLoader: {e}"))?;
    let mut model = loader
        .load_model(&cfg, &device)
        .map_err(|e| anyhow::anyhow!("load_model: {e}"))?;

    let lora = if let Some(st) = lora_ref.as_ref() {
        let lora = LoraRegistry::new_with_dtype_and_resident(
            &cfg,
            LORA_RANK,
            LORA_ALPHA,
            default_target_suffixes(),
            0,
            &device,
            DType::F32,
            true, // include 5 resident O1 head adapters (matches Python ref)
        )
        .map_err(|e| anyhow::anyhow!("LoraRegistry::new: {e}"))?;
        load_lora_init(&lora, st, &device)?;
        log::info!(
            "[o1-train-step] LoRA attached: {} adapters, rank={}, alpha={}",
            lora.len(),
            lora.rank,
            lora.alpha
        );
        Some(lora)
    } else {
        None
    };

    if !args.per_layer_dump.is_empty() {
        std::env::set_var("HIDREAM_DUMP_LAYERS", &args.per_layer_dump);
        log::info!(
            "[o1-train-step] per-layer dump enabled: {}",
            args.per_layer_dump
        );
    }

    // Diagnostic trap: only arm/retain internal backward probes when explicitly
    // requested. Leaving this on for normal parity changes the retained autograd
    // set and can perturb tiny BF16 backward roundoff enough for Adam's first
    // step to amplify it.
    let bwd_trap_enabled = parity_trap_enabled() || sdpa_bwd_ref.is_some();
    if bwd_trap_enabled {
        eridiffusion_core::models::hidream_o1::trap::arm_probes();
    } else {
        eridiffusion_core::models::hidream_o1::trap::disarm_probes();
    }

    log::info!("[o1-train-step] running Rust forward ...");
    let start = std::time::Instant::now();
    let x_pred_full = model
        .forward_lora(
            &input_ids,
            &timestep,
            &noisy,
            &pos_view,
            &vinput_mask,
            &token_types_bin,
            None,
            lora.as_ref(),
        )
        .map_err(|e| anyhow::anyhow!("forward_lora: {e}"))?;
    log::info!(
        "[o1-train-step] forward done in {:.2}s",
        start.elapsed().as_secs_f32()
    );

    let x_pred_rows = gather_image_rows(&x_pred_full, &vinput_mask)?;
    let sigma = t_scalar.max(1.0e-3);
    let pred_velocity = noisy
        .to_dtype(DType::F32)?
        .sub(&x_pred_rows.to_dtype(DType::F32)?)?
        .mul_scalar(1.0 / sigma)?
        .to_dtype(DType::BF16)?
        .to_dtype(DType::F32)?;
    let loss_velocity = pred_velocity
        .sub(&target_velocity)?
        .square()?
        .mean()?;
    let loss_velocity_val = scalar1(&loss_velocity)?;

    let x_metrics = compare_vec("x_pred_rows", &x_pred_rows, &x_pred_rows_ref)?;
    let v_metrics = compare_vec("pred_velocity", &pred_velocity, &pred_velocity_ref)?;
    let loss_abs = (loss_velocity_val - loss_velocity_ref).abs();
    let loss_rel = loss_abs / loss_velocity_ref.abs().max(1.0e-12);

    println!();
    let mut per_layer_ok = true;
    if !args.per_layer_dump.is_empty() {
        per_layer_ok = compare_per_layer_dump(
            &args.per_layer_dump,
            &ref_tensors,
            &cfg,
            &device,
            &args,
        )?;
        if !per_layer_ok {
            println!("[o1-train-step] per-layer dump showed drift (continuing for backward diag)");
        }
    }

    // DIAGNOSTIC MODE 2026-05-21: continue past forward failure so we can
    // see LoRA backward / grad parity numbers even when forward drift exceeds
    // tolerance. Original early-return logic preserved as "tag noting which
    // gate would have failed" without bailing.
    let x_full_ok = compare_ref_tensor(
        "forward",
        "x_pred_full",
        &x_pred_full.to_dtype(DType::F32)?,
        &ref_tensors,
        &args,
    )?;
    if !x_full_ok {
        println!("[o1-train-step] forward::x_pred_full FAILED (continuing for backward diag)");
    }

    let x_rows_ok = forward_pass(&x_metrics, &args);
    print_metric("forward", "x_pred_rows", &x_metrics, x_rows_ok);
    if !x_rows_ok {
        println!("[o1-train-step] forward::x_pred_rows FAILED (continuing for backward diag)");
    }

    let pred_velocity_ok = forward_pass(&v_metrics, &args);
    print_metric("objective", "pred_velocity", &v_metrics, pred_velocity_ok);
    if !pred_velocity_ok {
        println!("[o1-train-step] objective::pred_velocity FAILED (continuing for backward diag)");
    }

    println!(
        "loss_velocity: ours={:.9} ref={:.9} abs={:.6e} rel={:.6e}",
        loss_velocity_val, loss_velocity_ref, loss_abs, loss_rel
    );
    let loss_ok = loss_rel <= args.max_loss_rel;
    if !loss_ok {
        println!("[o1-train-step] objective::loss_velocity FAILED (continuing for backward diag)");
    }

    let lora_passed = if args.lora_step {
        let st = lora_ref.as_ref().unwrap();
        let lora = lora.as_ref().unwrap();
        let named = lora.named_parameters();
        let params = lora.parameters();
        let init_ok = report_lora_group("init", &named, |p| Ok(p.tensor()?), st, &args)?;
        let loss_ref_lora = st_scalar(st, "loss_velocity")?;
        let loss_lora_abs = (loss_velocity_val - loss_ref_lora).abs();
        let loss_lora_rel = loss_lora_abs / loss_ref_lora.abs().max(1.0e-12);
        println!(
            "lora loss_velocity: ours={:.9} ref={:.9} abs={:.6e} rel={:.6e}",
            loss_velocity_val, loss_ref_lora, loss_lora_abs, loss_lora_rel
        );

        let mut retain_ids = std::collections::HashSet::new();
        retain_ids.insert(pred_velocity.id());
        retain_ids.insert(x_pred_rows.id());
        retain_ids.insert(x_pred_full.id());
        // Soul.md trap: with gradient checkpointing, decoder.rs registers the
        // recompute-pass IDs into the retain set directly via
        // `retain_intermediate_grads_add` from inside the recompute closure.
        // We DON'T call `take_probes()` here — keep the trap registry armed
        // so the recompute pass overwrites whatever was captured in the no-
        // autograd first forward. We pull the (name -> id) map AFTER backward.
        flame_core::autograd::AutogradContext::retain_intermediate_grads(retain_ids);
        let grads = loss_velocity.backward()?;
        let retained = flame_core::autograd::AutogradContext::take_retained_intermediate_grads();
        let trap_probe_ids = eridiffusion_core::models::hidream_o1::trap::take_probes()
            .unwrap_or_default();
        println!("[trap] captured {} probe(s) (last-writer wins, from recompute):", trap_probe_ids.len());
        for (name, id) in trap_probe_ids.iter() {
            println!("[trap]   {name} -> TensorId({})  grad_present={}", id.0, retained.contains_key(id));
        }
        let mut mid_ok = true;
        for (key, tensor) in [
            ("grad_mid.pred_velocity", &pred_velocity),
            ("grad_mid.x_pred_rows", &x_pred_rows),
            ("grad_mid.x_pred_full", &x_pred_full),
        ] {
            let grad = retained
                .get(&tensor.id())
                .or_else(|| grads.get(tensor.id()))
                .ok_or_else(|| anyhow::anyhow!("missing intermediate gradient for {key}"))?;
            let m = compare_tensor_to_st(key, grad, st)?;
            let ok = lora_pass(&m, &args);
            println!(
                "{key:<24}: cos={:.8} max_abs={:.6e} mean_abs={:.6e} rel={:.6e} {}",
                m.cos,
                m.max_abs,
                m.mean_abs,
                m.rel,
                if ok { "OK" } else { "FAIL" }
            );
            mid_ok &= ok;
        }
        // Soul.md trap: layer-0 attention chain probes. Walk from o_proj
        // backward (attn_out) to v_proj output to find the first cos-collapse.
        // Some probes are at a different shape than the Python ref; for those
        // we rearrange Rust's grad to match the Python layout before compare.
        let n_q = cfg.num_attention_heads;
        let n_kv = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let n_rep = n_q / n_kv;
        let probe_layer_idx = std::env::var("HIDREAM_BWD_PROBE_LAYER")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|&idx| idx < cfg.num_layers)
            .unwrap_or(cfg.num_layers - 1);
        for (name, id) in trap_probe_ids.iter() {
            let key = format!("grad_probe.layers.{probe_layer_idx:02}.{name}");
            let grad = match retained.get(id) {
                Some(g) => g,
                None => {
                    println!("{key:<40}: SKIPPED (no grad retained — tensor may not have been on backward path)");
                    continue;
                }
            };
            // Reshape Rust grad to match the Python ref's hook-captured layout.
            // Python hook locations:
            //   attn_out  ←  o_proj's INPUT  =  [B, S, Hq*D]  (post-permute, post-reshape)
            //   v_proj_out ←  v_proj OUTPUT   =  [B, S, Hkv*D]
            //   v_post_repeat_kv (no Python equivalent — compare vs v_proj_out
            //     after sum-reducing over n_rep)
            let (rearranged_grad, compare_key): (Tensor, String) = match name.as_str() {
                "attn_out" => {
                    // Rust: [B, Hq, S, D]  →  permute (0,2,1,3) → [B, S, Hq, D] → reshape [B, S, Hq*D]
                    let dims = grad.shape().dims().to_vec();
                    let (b, hq, s, d) = (dims[0], dims[1], dims[2], dims[3]);
                    let g = grad.permute(&[0, 2, 1, 3])?.contiguous()?.reshape(&[b, s, hq * d])?;
                    (g, key.clone())
                }
                "v_post_repeat_kv" => {
                    // Rust: [B, Hq, S, D]  →  view [B, Hkv, n_rep, S, D] → sum dim=2 →
                    //   [B, Hkv, S, D] → permute (0,2,1,3) → [B, S, Hkv, D] → reshape [B, S, Hkv*D]
                    // Then compare to Python's `grad_probe.layers.{probe_layer_idx:02}.v_proj_out`
                    // (same logical point: grad at V right after the projection).
                    let dims = grad.shape().dims().to_vec();
                    let (b, hq, s, d) = (dims[0], dims[1], dims[2], dims[3]);
                    assert_eq!(hq, n_q, "v_post_repeat_kv probe shape mismatch");
                    let g = grad
                        .reshape(&[b, n_kv, n_rep, s, d])?
                        .sum_dim_keepdim(2)?  // [B, Hkv, 1, S, D]
                        .reshape(&[b, n_kv, s, d])?
                        .permute(&[0, 2, 1, 3])?
                        .contiguous()?
                        .reshape(&[b, s, n_kv * head_dim])?;
                    // Compare against v_proj_out (same logical point post-reduction).
                    (g, format!("grad_probe.layers.{probe_layer_idx:02}.v_proj_out"))
                }
                _ => (grad.clone(), key.clone()),
            };
            let fixture_key = match name.as_str() {
                "q_sdpa_in" => Some("dq"),
                "k_sdpa_in" => Some("dk"),
                "v_sdpa_in" => Some("dv"),
                _ => None,
            };
            let (m, note_extra) = if let (Some(sdpa_st), Some(fixture_key)) =
                (sdpa_bwd_ref.as_ref(), fixture_key)
            {
                match tensor_from_st(sdpa_st, fixture_key, &device)
                    .and_then(|reference| compare_vec(&compare_key, &rearranged_grad, &reference))
                {
                    Ok(m) => (m, format!(" (vs HIDREAM_SDPA_BWD_REF::{fixture_key})")),
                    Err(e) => {
                        println!("{key:<40}: SKIPPED ({e})");
                        continue;
                    }
                }
            } else {
                match compare_tensor_to_st(&compare_key, &rearranged_grad, st) {
                    Ok(m) => (m, String::new()),
                    Err(e) => {
                        println!("{key:<40}: SKIPPED ({e})");
                        continue;
                    }
                }
            };
            let note = if compare_key != key { format!(" (vs {compare_key})") } else { String::new() };
            println!(
                "{key:<40}: cos={:.8} max_abs={:.6e} mean_abs={:.6e} rel={:.6e}{note}{note_extra}",
                m.cos, m.max_abs, m.mean_abs, m.rel
            );
        }
        let grad_pre_ok = report_lora_group(
            "grad_pre",
            &named,
            |p| {
                if let Some(g) = grads.get(p.id()) {
                    Ok(g.clone())
                } else {
                    zero_like_param(p, &device)
                }
            },
            st,
            &args,
        )?;
        report_lora_selected(
            "grad_pre",
            &[
                "layers.35.mlp.down_proj.lora_B",
                "layers.35.mlp.up_proj.lora_B",
                "layers.35.mlp.gate_proj.lora_B",
                "layers.35.self_attn.o_proj.lora_B",
                "layers.35.self_attn.q_proj.lora_B",
                "layers.35.self_attn.k_proj.lora_B",
                "layers.35.self_attn.v_proj.lora_B",
            ],
            &named,
            |p| {
                if let Some(g) = grads.get(p.id()) {
                    Ok(g.clone())
                } else {
                    zero_like_param(p, &device)
                }
            },
            st,
        )?;

        let grad_refs: Vec<&Tensor> = params.iter().filter_map(|p| grads.get(p.id())).collect();
        let total_norm = flame_core::ops::grad_norm::global_l2_norm(&grad_refs)?.item()? as f32;
        let ref_norm = st_scalar(st, "global_grad_norm_pre")?;
        let norm_abs = (total_norm - ref_norm).abs();
        let norm_rel = norm_abs / ref_norm.abs().max(1.0e-12);
        println!(
            "grad_norm_pre: ours={:.9} ref={:.9} abs={:.6e} rel={:.6e}",
            total_norm, ref_norm, norm_abs, norm_rel
        );
        if !total_norm.is_finite() {
            anyhow::bail!("NaN/Inf grad_norm: {total_norm}");
        }
        let scale = if total_norm > CLIP_GRAD_NORM {
            CLIP_GRAD_NORM / total_norm
        } else {
            1.0
        };
        let ref_scale = st_scalar(st, "clip_scale")?;
        println!(
            "clip_scale   : ours={:.9} ref={:.9} abs={:.6e}",
            scale,
            ref_scale,
            (scale - ref_scale).abs()
        );
        for p in &params {
            if let Some(g) = grads.get(p.id()) {
                let g_scaled = if scale < 1.0 {
                    g.mul_scalar(scale)?
                } else {
                    g.clone()
                };
                p.set_grad(g_scaled)?;
            }
        }
        let grad_post_ok = report_lora_group(
            "grad_post",
            &named,
            |p| {
                if let Some(g) = p.grad() {
                    Ok(g)
                } else {
                    zero_like_param(p, &device)
                }
            },
            st,
            &args,
        )?;

        if std::env::var("HIDREAM_USE_REF_GRADS_FOR_OPT")
            .ok()
            .as_deref()
            == Some("1")
        {
            println!("[o1-train-step] diagnostic: using reference grad_post tensors for optimizer step");
            for (name, p) in &named {
                let key = format!("grad_post.{name}");
                p.set_grad(tensor_from_st(st, &key, &device)?)?;
            }
        }

        let lr = st_scalar(st, "adamw_lr")?;
        let wd = st_scalar(st, "adamw_weight_decay")?;
        let mut opt = Optimizer::new(OptimizerKind::AdamW8bit, lr, 0.9, 0.999, 1.0e-6, wd);
        {
            let _g = flame_core::autograd::AutogradContext::no_grad();
            opt.step(&params)?;
            opt.zero_grad(&params);
        }
        let post_ok = report_lora_group("post", &named, |p| Ok(p.tensor()?), st, &args)?;
        if let Some(cache) = &activation_cache {
            let mut cache = cache
                .lock()
                .map_err(|_| anyhow::anyhow!("grow activation cache mutex poisoned"))?;
            cache.reset();
        }
        init_ok
            && mid_ok
            && grad_pre_ok
            && grad_post_ok
            && post_ok
            && loss_lora_rel <= args.max_loss_rel
            && norm_rel <= args.lora_max_rel
    } else {
        true
    };

    let forward_passed = per_layer_ok && x_full_ok && x_rows_ok;
    let objective_passed = pred_velocity_ok && loss_ok;
    let passed = forward_passed && objective_passed && lora_passed;
    println!(
        "[o1-train-step] gates: per_layer={} x_pred_full={} x_pred_rows={} pred_velocity={} loss={} lora={}",
        per_layer_ok, x_full_ok, x_rows_ok, pred_velocity_ok, loss_ok, lora_passed
    );

    if passed {
        println!(
            "[o1-train-step] PASS — fixed input, target, prediction and loss are in parity"
        );
        Ok(true)
    } else {
        println!(
            "[o1-train-step] FAIL — thresholds: min_cos={} max_abs={} max_rel={} max_loss_rel={}",
            args.min_cos, args.max_abs, args.max_rel, args.max_loss_rel
        );
        Ok(false)
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::from(0),
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("[o1-train-step] setup error: {e:#}");
            ExitCode::from(2)
        }
    }
}
