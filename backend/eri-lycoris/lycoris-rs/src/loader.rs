//! safetensors loader for LyCORIS / Kohya LoRA checkpoints.
//!
//! Reads a single `.safetensors` file, groups tensors by adapter prefix, and
//! auto-detects the adapter type (LoCon / LoHa / LoKr / Full) from the key
//! suffixes. Produces a [`LycorisCollection`] ready for weight-merge mode.
//!
//! **On-disk conventions** (Kohya / LyCORIS):
//! - Linear `down`: `[rank, in_features]`
//! - Linear `up`:   `[out_features, rank]`
//! - Conv   `down`: `[rank, in_channels, kh, kw]` (or `[rank, in, 1, 1]` for 1×1)
//! - Conv   `up`:   `[out_channels, rank, 1, 1]`
//! - `mid` (LoCon Tucker): `[rank, rank, kh, kw]`
//! - Alpha: 0-d or 1-elem scalar.
//!
//! We transpose on load so the internal algorithm modules — which follow the
//! Flame convention `[IN, OUT]` / `[KH, KW, IC, OC]` — work unchanged. ΔW is
//! emitted in Flame convention and reshaped at merge time if the base weight
//! uses a different layout.
//!
//! Training-only fields (`dora_scale`, etc.) are silently ignored.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context};
use cudarc::driver::CudaDevice;
use flame_core::{DType, Tensor};

use crate::algorithms::{
    full::FullAdapter,
    locon::LoConModule,
    loha::LoHaModule,
    lokr::LoKrModule,
};
use crate::{LycorisAdapter, LycorisCollection};

// ---------------------------------------------------------------------------
// Public entry points (called from `lib.rs`).
// ---------------------------------------------------------------------------

/// Load a LyCORIS safetensors file from disk and group keys into adapters.
pub fn load(path: &Path, device: Arc<CudaDevice>) -> anyhow::Result<LycorisCollection> {
    let raw = flame_core::serialization::load_file(path, &device)
        .map_err(|e| anyhow!("failed to read safetensors {:?}: {:?}", path, e))?;

    build_collection(raw, device)
}

/// Weight-merge mode. Compute ΔW for each adapter, reshape to base shape if
/// needed, and replace `weights[mapped_key]` with `base + strength * ΔW`.
pub fn apply_collection(
    coll: &LycorisCollection,
    weights: &mut HashMap<String, Tensor>,
    strength: f32,
    name_mapper: impl Fn(&str) -> Option<String>,
) -> anyhow::Result<()> {
    for (prefix, adapter) in &coll.adapters {
        let Some(base_key) = name_mapper(prefix) else {
            continue;
        };
        let Some(base) = weights.get(&base_key) else {
            // Caller can legitimately not have every mapped key present
            // (e.g. an adapter on the text encoder when only UNet weights
            // are supplied). Skip silently.
            continue;
        };

        // Adapter ΔW in Flame convention. Inner algorithm modules already
        // include the alpha/rank scale — we only multiply by `strength`.
        let mut delta = adapter
            .delta_weight()
            .map_err(|e| anyhow!("{}: delta_weight failed: {:?}", prefix, e))?;

        if strength != 1.0 {
            delta = delta
                .mul_scalar(strength)
                .map_err(|e| anyhow!("{}: mul_scalar failed: {:?}", prefix, e))?;
        }

        // Reshape / reorder ΔW so it broadcasts cleanly onto the base.
        let aligned = align_delta_to_base(&delta, base).with_context(|| {
            format!("{}: cannot align ΔW shape {:?} to base {:?}", prefix, delta.dims(), base.dims())
        })?;

        // Cast ΔW to base dtype so the add doesn't blow up on mixed precision.
        let aligned = if aligned.dtype() != base.dtype() {
            aligned
                .to_dtype(base.dtype())
                .map_err(|e| anyhow!("{}: dtype cast failed: {:?}", prefix, e))?
        } else {
            aligned
        };

        let merged = base
            .add(&aligned)
            .map_err(|e| anyhow!("{}: base + ΔW failed: {:?}", prefix, e))?;

        weights.insert(base_key.clone(), merged);

        // P0-7: Full adapters can carry an optional bias delta `.diff_b`.
        // Apply it to the matching `.bias` key when both are present.
        // Convention: if base_key ends in ".weight", the bias lives at the
        // same path with ".bias". For non-Full adapters this is a no-op.
        if let LycorisAdapter::Full(full) = adapter {
            if let Some(bias_delta) = full
                .delta_bias(strength)
                .map_err(|e| anyhow!("{}: delta_bias failed: {:?}", prefix, e))?
            {
                let bias_key = if let Some(stem) = base_key.strip_suffix(".weight") {
                    format!("{}.bias", stem)
                } else {
                    // No `.weight` suffix to swap; assume the mapper already
                    // returned a bias-style key or the caller named keys
                    // without `.weight`. Fall back to appending `.bias` to
                    // the prefix-mapped key only if the literal key exists.
                    format!("{}.bias", base_key)
                };
                if let Some(base_bias) = weights.get(&bias_key) {
                    let aligned_b = if bias_delta.dtype() != base_bias.dtype() {
                        bias_delta
                            .to_dtype(base_bias.dtype())
                            .map_err(|e| anyhow!("{}: bias dtype cast failed: {:?}", prefix, e))?
                    } else {
                        bias_delta
                    };
                    let merged_b = base_bias
                        .add(&aligned_b)
                        .map_err(|e| anyhow!("{}: base_bias + Δb failed: {:?}", prefix, e))?;
                    weights.insert(bias_key, merged_b);
                } else {
                    eprintln!(
                        "lycoris-rs: Full adapter '{}' has .diff_b but no '{}' key in weights — bias delta dropped.",
                        prefix, bias_key
                    );
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Prefix grouping + type detection.
// ---------------------------------------------------------------------------

/// Known LyCORIS / Kohya suffixes. The first match wins when splitting a key
/// into `(prefix, suffix)`.
const KNOWN_SUFFIXES: &[&str] = &[
    // LoCon / standard LoRA
    ".lora_up.weight",
    ".lora_down.weight",
    ".lora_mid.weight",
    // LoHa (Hadamard)
    ".hada_w1_a",
    ".hada_w1_b",
    ".hada_w2_a",
    ".hada_w2_b",
    ".hada_t1",
    ".hada_t2",
    // LoKr (Kronecker)
    ".lokr_w1",
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w2",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_t2",
    // Full
    ".diff",
    // Shared
    ".alpha",
    // Training-only — recognised so we ignore them without warning.
    ".dora_scale",
    ".diff_b",
];

fn split_prefix_suffix(key: &str) -> Option<(&str, &str)> {
    for sfx in KNOWN_SUFFIXES {
        if let Some(pfx) = key.strip_suffix(sfx) {
            return Some((pfx, sfx));
        }
    }
    None
}

fn build_collection(
    raw: HashMap<String, Tensor>,
    device: Arc<CudaDevice>,
) -> anyhow::Result<LycorisCollection> {
    // Group tensors by adapter prefix (BTreeMap for stable iteration order).
    let mut groups: BTreeMap<String, HashMap<String, Tensor>> = BTreeMap::new();
    let mut unknown: Vec<String> = Vec::new();

    for (key, tensor) in raw {
        match split_prefix_suffix(&key) {
            Some((prefix, suffix)) => {
                groups
                    .entry(prefix.to_string())
                    .or_default()
                    .insert(suffix.to_string(), tensor);
            }
            None => {
                unknown.push(key);
            }
        }
    }

    if !unknown.is_empty() {
        eprintln!(
            "lycoris-rs: {} key(s) did not match any known LyCORIS suffix and were skipped (first: {})",
            unknown.len(),
            unknown[0]
        );
    }

    let mut adapters: HashMap<String, LycorisAdapter> = HashMap::new();
    let mut dora_skipped: usize = 0;
    for (prefix, entries) in groups {
        // P0-6: DoRA-trained adapters store an extra `.dora_scale` tensor
        // (`lycoris.functional.general.apply_dora_scale`, general.py:95-108)
        // that renormalises the merged weight per-output-channel. Without
        // applying it, plain `base + alpha/rank * down @ up` is silently
        // wrong by O(10–50%) per channel. Rather than pretend we handle it,
        // refuse to load the adapter with a loud message — partial-correct
        // is worse than missing.
        if entries.contains_key(".dora_scale") {
            eprintln!(
                "lycoris-rs: skipping DoRA adapter '{}' — .dora_scale present \
                 but DoRA correction is not yet implemented in lycoris-rs. \
                 Loading this adapter without DoRA would silently corrupt \
                 the merged weight (per-channel norm renormalisation skipped). \
                 See lycoris/functional/general.py:apply_dora_scale.",
                prefix
            );
            dora_skipped += 1;
            continue;
        }
        match classify(&entries) {
            AdapterKind::LoCon => {
                let a = build_locon(&prefix, entries, device.clone())?;
                adapters.insert(prefix, LycorisAdapter::LoCon(a));
            }
            AdapterKind::LoHa => {
                let a = build_loha(&prefix, entries, device.clone())?;
                adapters.insert(prefix, LycorisAdapter::LoHa(a));
            }
            AdapterKind::LoKr => {
                let a = build_lokr(&prefix, entries, device.clone())?;
                adapters.insert(prefix, LycorisAdapter::LoKr(a));
            }
            AdapterKind::Full => {
                let a = build_full(&prefix, entries)?;
                adapters.insert(prefix, LycorisAdapter::Full(a));
            }
            AdapterKind::Unknown => {
                let suffixes: Vec<&String> = entries.keys().collect();
                eprintln!(
                    "lycoris-rs: skipping prefix '{}' — unrecognised combination of suffixes: {:?}",
                    prefix, suffixes
                );
            }
        }
    }
    if dora_skipped > 0 {
        eprintln!(
            "lycoris-rs: skipped {} DoRA adapter(s) total. The merge will produce results \
             that DIFFER from a DoRA-aware loader on those layers.",
            dora_skipped
        );
    }

    Ok(LycorisCollection { adapters })
}

#[derive(Debug)]
enum AdapterKind {
    LoCon,
    LoHa,
    LoKr,
    Full,
    Unknown,
}

fn classify(entries: &HashMap<String, Tensor>) -> AdapterKind {
    let keys: HashSet<&str> = entries.keys().map(String::as_str).collect();
    let has = |k: &str| keys.contains(k);

    if has(".lora_up.weight") && has(".lora_down.weight") {
        AdapterKind::LoCon
    } else if has(".hada_w1_a") && has(".hada_w1_b") && has(".hada_w2_a") && has(".hada_w2_b") {
        AdapterKind::LoHa
    } else if has(".lokr_w1") || (has(".lokr_w1_a") && has(".lokr_w1_b")) {
        AdapterKind::LoKr
    } else if has(".diff") {
        AdapterKind::Full
    } else {
        AdapterKind::Unknown
    }
}

// ---------------------------------------------------------------------------
// Helpers: alpha scalar, dtype coercion, shape reasoning.
// ---------------------------------------------------------------------------

fn read_alpha(entries: &HashMap<String, Tensor>, default: f32) -> anyhow::Result<f32> {
    let Some(t) = entries.get(".alpha") else {
        return Ok(default);
    };
    let v = t
        .to_dtype(DType::F32)
        .and_then(|x| x.to_vec())
        .map_err(|e| anyhow!("alpha to_vec failed: {:?}", e))?;
    if v.is_empty() {
        Ok(default)
    } else {
        Ok(v[0])
    }
}

/// Ensure a tensor is BF16 storage — our algorithm modules enforce BF16.
fn ensure_bf16(t: Tensor) -> anyhow::Result<Tensor> {
    if t.dtype() == DType::BF16 {
        Ok(t)
    } else {
        t.to_dtype(DType::BF16)
            .map_err(|e| anyhow!("cast to BF16 failed: {:?}", e))
    }
}

/// Transpose a 2D tensor.
fn transpose_2d(t: &Tensor) -> anyhow::Result<Tensor> {
    t.transpose().map_err(|e| anyhow!("transpose_2d failed: {:?}", e))
}

// ---------------------------------------------------------------------------
// Per-adapter builders.
// ---------------------------------------------------------------------------

fn build_locon(
    prefix: &str,
    mut entries: HashMap<String, Tensor>,
    device: Arc<CudaDevice>,
) -> anyhow::Result<LoConModule> {
    let down_k = entries
        .remove(".lora_down.weight")
        .ok_or_else(|| anyhow!("{}: missing lora_down.weight", prefix))?;
    let up_k = entries
        .remove(".lora_up.weight")
        .ok_or_else(|| anyhow!("{}: missing lora_up.weight", prefix))?;
    let mid_k = entries.remove(".lora_mid.weight");

    // Kohya convention:
    //   Linear:  down [R, IN],       up [OUT, R]
    //   Conv:    down [R, IN, KH,KW], up [OUT, R, 1, 1], mid [R, R, KH, KW]
    let down_dims = down_k.dims().to_vec();
    let (is_conv, rank) = match down_dims.len() {
        2 => (false, down_dims[0]),
        4 => (true, down_dims[0]),
        other => return Err(anyhow!("{}: unexpected down rank {}D", prefix, other)),
    };

    let alpha = read_alpha(&entries, rank as f32)?;

    // Convert to Flame convention: down [IN, R] / [KH,KW,IC,R], up [R, OUT] / [KH,KW,R,OC].
    let (down, up, mid) = if !is_conv {
        // Linear: transpose both.
        let down = ensure_bf16(transpose_2d(&down_k)?)?; // [IN, R]
        let up = ensure_bf16(transpose_2d(&up_k)?)?; // [R, OUT]
        (down, up, None)
    } else {
        // Conv: permute PyTorch [O,I,KH,KW] → Flame [KH,KW,I,O].
        // Here O is either rank (down) or out_channels (up).
        let down = ensure_bf16(
            down_k
                .permute(&[2, 3, 1, 0])
                .map_err(|e| anyhow!("down permute: {:?}", e))?,
        )?; // [KH, KW, IN, R]
        let up = ensure_bf16(
            up_k.permute(&[2, 3, 1, 0])
                .map_err(|e| anyhow!("up permute: {:?}", e))?,
        )?; // [1, 1, R, OUT]
        let mid = if let Some(m) = mid_k {
            // mid kohya: [R_out, R_in, KH, KW] → Flame [KH, KW, R_in, R_out].
            let m = ensure_bf16(
                m.permute(&[2, 3, 1, 0])
                    .map_err(|e| anyhow!("mid permute: {:?}", e))?,
            )?;
            Some(m)
        } else {
            None
        };
        (down, up, mid)
    };

    Ok(LoConModule {
        down: flame_core::parameter::Parameter::new(down),
        up: flame_core::parameter::Parameter::new(up),
        mid: mid.map(flame_core::parameter::Parameter::new),
        rank,
        alpha,
        device,
        is_conv,
    })
}

fn build_loha(
    prefix: &str,
    mut entries: HashMap<String, Tensor>,
    device: Arc<CudaDevice>,
) -> anyhow::Result<LoHaModule> {
    let take = |e: &mut HashMap<String, Tensor>, key: &str| -> anyhow::Result<Tensor> {
        e.remove(key)
            .ok_or_else(|| anyhow!("{}: missing {}", prefix, key))
    };

    let w1a_k = take(&mut entries, ".hada_w1_a")?;
    let w1b_k = take(&mut entries, ".hada_w1_b")?;
    let w2a_k = take(&mut entries, ".hada_w2_a")?;
    let w2b_k = take(&mut entries, ".hada_w2_b")?;
    let t1_k = entries.remove(".hada_t1");
    let t2_k = entries.remove(".hada_t2");

    // LyCORIS LoHa on-disk convention (matches Kohya LoRA):
    //   Linear (no t):       w1a/w2a [R, IN], w1b/w2b [OUT, R]
    //   Conv non-Tucker:     w1a/w2a [R, IN, KH, KW], w1b/w2b [OUT, R, 1, 1]
    //                        (or w1a [R, IN, 1, 1] for 1×1)
    //   Conv Tucker:         t1/t2 [R, R, KH, KW]
    //                        w1d (= hada_w1_a) [R, IN]   ← 2D!
    //                        w1u (= hada_w1_b) [R, OUT]  ← 2D, NOTE: differs
    //                        from non-Tucker linear where w1u is [OUT, R].
    //
    // P0-3: detect Tucker by presence of .hada_t1/.hada_t2, NOT by w1a rank.
    // Previously the loader inferred is_conv from `w1a.dims().len()`, so a
    // Tucker LoHa (2D w1a) was silently classified as Linear, dropping
    // t1/t2 entirely and producing wrong ΔW.
    let is_tucker = t1_k.is_some() || t2_k.is_some();
    let w1a_dims = w1a_k.dims().to_vec();
    let (is_conv, rank) = if is_tucker {
        // Tucker is always conv. Rank comes from t1/t2 (or w1a[0]).
        let rank = w1a_dims[0];
        (true, rank)
    } else {
        match w1a_dims.len() {
            2 => (false, w1a_dims[0]),
            4 => (true, w1a_dims[0]),
            other => return Err(anyhow!("{}: unexpected hada_w1_a rank {}D", prefix, other)),
        }
    };

    let alpha = read_alpha(&entries, rank as f32)?;

    let (w1a, w1b, w2a, w2b, t1, t2) = if is_tucker {
        // Tucker: w1a/w2a are 2D [R, IN], w1b/w2b are 2D [R, OUT] on disk.
        // Internal layout: w1a/w2a [IN, R], w1b/w2b [R, OUT].
        // t1/t2 on disk [R, R, KH, KW] → permute to [KH, KW, R, R].
        for (name, w) in [(".hada_w1_a", &w1a_k), (".hada_w1_b", &w1b_k),
                           (".hada_w2_a", &w2a_k), (".hada_w2_b", &w2b_k)] {
            if w.dims().len() != 2 {
                return Err(anyhow!(
                    "{}: Tucker LoHa expects 2D {} on disk, got {}D",
                    prefix, name, w.dims().len()
                ));
            }
        }
        let w1a = ensure_bf16(transpose_2d(&w1a_k)?)?; // [IN, R]
        let w1b = ensure_bf16(w1b_k); // [R, OUT] — already correct
        let w1b = w1b?;
        let w2a = ensure_bf16(transpose_2d(&w2a_k)?)?;
        let w2b = ensure_bf16(w2b_k)?;
        let t1 = match t1_k {
            Some(t) => Some(ensure_bf16(
                t.permute(&[2, 3, 0, 1])
                    .map_err(|e| anyhow!("t1 permute: {:?}", e))?,
            )?),
            None => return Err(anyhow!(
                "{}: Tucker LoHa requires both .hada_t1 and .hada_t2; .hada_t1 missing", prefix
            )),
        };
        let t2 = match t2_k {
            Some(t) => Some(ensure_bf16(
                t.permute(&[2, 3, 0, 1])
                    .map_err(|e| anyhow!("t2 permute: {:?}", e))?,
            )?),
            None => return Err(anyhow!(
                "{}: Tucker LoHa requires both .hada_t1 and .hada_t2; .hada_t2 missing", prefix
            )),
        };
        (w1a, w1b, w2a, w2b, t1, t2)
    } else if !is_conv {
        (
            ensure_bf16(transpose_2d(&w1a_k)?)?, // [IN, R]
            ensure_bf16(transpose_2d(&w1b_k)?)?, // [R, OUT]
            ensure_bf16(transpose_2d(&w2a_k)?)?,
            ensure_bf16(transpose_2d(&w2b_k)?)?,
            None,
            None,
        )
    } else {
        let w1a = ensure_bf16(
            w1a_k
                .permute(&[2, 3, 1, 0])
                .map_err(|e| anyhow!("w1a permute: {:?}", e))?,
        )?;
        let w1b = ensure_bf16(
            w1b_k
                .permute(&[2, 3, 1, 0])
                .map_err(|e| anyhow!("w1b permute: {:?}", e))?,
        )?;
        let w2a = ensure_bf16(
            w2a_k
                .permute(&[2, 3, 1, 0])
                .map_err(|e| anyhow!("w2a permute: {:?}", e))?,
        )?;
        let w2b = ensure_bf16(
            w2b_k
                .permute(&[2, 3, 1, 0])
                .map_err(|e| anyhow!("w2b permute: {:?}", e))?,
        )?;
        // Non-tucker conv has no t1/t2 by definition.
        (w1a, w1b, w2a, w2b, None, None)
    };

    Ok(LoHaModule {
        w1a: flame_core::parameter::Parameter::new(w1a),
        w1b: flame_core::parameter::Parameter::new(w1b),
        w2a: flame_core::parameter::Parameter::new(w2a),
        w2b: flame_core::parameter::Parameter::new(w2b),
        t1: t1.map(flame_core::parameter::Parameter::new),
        t2: t2.map(flame_core::parameter::Parameter::new),
        rank,
        alpha,
        device,
        is_conv,
    })
}

fn build_lokr(
    prefix: &str,
    mut entries: HashMap<String, Tensor>,
    device: Arc<CudaDevice>,
) -> anyhow::Result<LoKrModule> {
    // LoKr variants (Kohya / LyCORIS):
    //   w1 full:         lokr_w1 [OL, IM]
    //   w1 factorized:   lokr_w1_a [OL, R], lokr_w1_b [R, IM]
    //   w2 full (2D):    lokr_w2 [OK, IN]        (linear)
    //   w2 full (4D):    lokr_w2 [OK, IN, KH, KW] (conv)
    //   w2 factorized:   lokr_w2_a [OK, R], lokr_w2_b [R, IN] or [R, IN, KH, KW]
    //   w2 tucker:       lokr_t2 [R, R, KH, KW], lokr_w2_a [OK, R], lokr_w2_b [R, IN]
    //
    // All of these are native to the `LoKrModule` layout defined in
    // `algorithms/lokr.rs`, except the Tucker core (needs a permute) and
    // w2_b conv (needs to be kept as [R, IN, KH, KW] per LokrModule contract).

    let w1 = entries
        .remove(".lokr_w1")
        .map(|t| ensure_bf16(t))
        .transpose()?;
    let w1a = entries
        .remove(".lokr_w1_a")
        .map(|t| ensure_bf16(t))
        .transpose()?;
    let w1b = entries
        .remove(".lokr_w1_b")
        .map(|t| ensure_bf16(t))
        .transpose()?;
    let w2 = entries
        .remove(".lokr_w2")
        .map(|t| ensure_bf16(t))
        .transpose()?;
    let w2a = entries
        .remove(".lokr_w2_a")
        .map(|t| ensure_bf16(t))
        .transpose()?;
    let w2b = entries
        .remove(".lokr_w2_b")
        .map(|t| ensure_bf16(t))
        .transpose()?;

    // Tucker core on disk is [R_out, R_in, KH, KW]; LokrModule wants the
    // `rebuild_conv_tucker` convention [KH, KW, R_in, R_out].
    let t2 = if let Some(t) = entries.remove(".lokr_t2") {
        let permuted = t
            .permute(&[2, 3, 1, 0])
            .map_err(|e| anyhow!("{}: lokr_t2 permute: {:?}", prefix, e))?;
        Some(ensure_bf16(permuted)?)
    } else {
        None
    };

    // Determine is_conv from the W2 shape we actually have.
    let is_conv = if let Some(w) = &w2 {
        w.dims().len() == 4 && !(w.dims()[2] == 1 && w.dims()[3] == 1)
    } else if let Some(w) = &w2b {
        w.dims().len() == 4
    } else if t2.is_some() {
        true
    } else {
        false
    };

    // Rank: prefer the factorised side that actually gives us an r.
    let rank = w1a
        .as_ref()
        .map(|w| w.dims()[1])
        .or_else(|| w1b.as_ref().map(|w| w.dims()[0]))
        .or_else(|| w2a.as_ref().map(|w| w.dims()[1]))
        .or_else(|| w2b.as_ref().map(|w| w.dims()[0]))
        .or_else(|| t2.as_ref().map(|w| w.dims()[3]))
        .unwrap_or(0);

    let alpha = read_alpha(&entries, rank.max(1) as f32)?;

    // Shape metadata — populate sensibly so downstream ops have a value,
    // even though LokrModule only uses it for linear shape math.
    let shape = ((0usize, 0usize), (0usize, 0usize));

    Ok(LoKrModule {
        w1: w1.map(flame_core::parameter::Parameter::new),
        w1a: w1a.map(flame_core::parameter::Parameter::new),
        w1b: w1b.map(flame_core::parameter::Parameter::new),
        w2: w2.map(flame_core::parameter::Parameter::new),
        w2a: w2a.map(flame_core::parameter::Parameter::new),
        w2b: w2b.map(flame_core::parameter::Parameter::new),
        t2: t2.map(flame_core::parameter::Parameter::new),
        rank,
        alpha,
        device,
        shape,
        is_conv,
    })
}

fn build_full(prefix: &str, mut entries: HashMap<String, Tensor>) -> anyhow::Result<FullAdapter> {
    let diff = entries
        .remove(".diff")
        .ok_or_else(|| anyhow!("{}: missing .diff", prefix))?;
    // P0-7: pick up `.diff_b` (bias delta) if present. Upstream Full saves
    // it whenever the original layer has a bias (modules/full.py:128-132).
    let diff_b = entries.remove(".diff_b");
    Ok(FullAdapter {
        diff: flame_core::parameter::Parameter::new(diff),
        diff_b: diff_b.map(flame_core::parameter::Parameter::new),
    })
}

// ---------------------------------------------------------------------------
// Shape alignment at merge time.
// ---------------------------------------------------------------------------

/// Best-effort reshape / transpose of ΔW to match the base weight's shape.
///
/// Supported cases:
/// - Shapes already match → no-op.
/// - Both 2D with transposed dims → apply `transpose()`.
/// - Same number of elements in a different valid layout → `reshape`.
/// - Fall through to an error with full context.
fn align_delta_to_base(delta: &Tensor, base: &Tensor) -> anyhow::Result<Tensor> {
    let d = delta.dims();
    let b = base.dims();

    if d == b {
        return Ok(delta.clone());
    }

    // Pure transpose (2D) — the most common mismatch between Flame [IN,OUT]
    // ΔW and PyTorch-style [OUT,IN] base weights.
    if d.len() == 2 && b.len() == 2 && d[0] == b[1] && d[1] == b[0] {
        return delta
            .transpose()
            .map_err(|e| anyhow!("transpose align failed: {:?}", e));
    }

    // 4D Flame [KH,KW,IC,OC] → PyTorch [OC,IC,KH,KW].
    if d.len() == 4 && b.len() == 4 {
        if d[0] == b[2] && d[1] == b[3] && d[2] == b[1] && d[3] == b[0] {
            return delta
                .permute(&[3, 2, 0, 1])
                .map_err(|e| anyhow!("permute align failed: {:?}", e));
        }
        // Exact shape match — try a plain reshape.
        if d.iter().product::<usize>() == b.iter().product::<usize>() {
            return delta
                .reshape(b)
                .map_err(|e| anyhow!("reshape align failed: {:?}", e));
        }
    }

    // Element-count match with a compatible layout — last-ditch reshape.
    if d.iter().product::<usize>() == b.iter().product::<usize>() {
        return delta
            .reshape(b)
            .map_err(|e| anyhow!("fallback reshape failed: {:?}", e));
    }

    Err(anyhow!(
        "no alignment rule: delta {:?} vs base {:?}",
        d,
        b
    ))
}
