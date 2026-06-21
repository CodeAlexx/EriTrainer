//! Loader round-trip smoke tests.
//!
//! Constructs synthetic LyCORIS safetensors files (linear-only, F32 on
//! disk — the loader's `ensure_bf16` casts on load) for each adapter
//! variant and asserts:
//!
//! 1. `LycorisCollection::load` detects the right adapter type.
//! 2. The loaded module's `delta_weight()` produces the expected ΔW values
//!    against a hand-computed reference.
//! 3. Loader-side fixes from this round trigger correctly:
//!    - P0-3: Tucker LoHa detected via `.hada_t1` / `.hada_t2`, not by w1a rank.
//!    - P0-6: `.dora_scale`-bearing adapters are skipped with a warning.
//!    - P0-7: `.diff_b` is propagated through `apply_to`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::{DType, Shape, Tensor};

use lycoris_rs::tensor_utils;
use lycoris_rs::{LycorisAdapter, LycorisCollection};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn try_device() -> Option<Arc<CudaDevice>> {
    CudaDevice::new(0).ok()
}

fn f32_tensor(data: Vec<f32>, dims: &[usize], device: Arc<CudaDevice>) -> Tensor {
    Tensor::from_vec(data, Shape::from_dims(dims), device).expect("from_vec")
}

fn bf16_tensor(data: Vec<f32>, dims: &[usize], device: Arc<CudaDevice>) -> Tensor {
    tensor_utils::from_vec_bf16(data, Shape::from_dims(dims), device)
        .expect("from_vec_bf16")
}

fn tmp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    p.push(format!("lycoris_rs_loader_{}_{}.safetensors", name, nonce));
    p
}

// BF16 has ~7-bit mantissa, so step size at magnitude 10 is ~0.0625 and at 100 is ~0.5.
// 5e-2 covers both round-trip noise and the actual values the regression check needs to
// distinguish (e.g. diff_b dropped → bias stays at 100 vs +1.0 update; way bigger than 5e-2).
const BF16_EPS: f32 = 5e-2;

fn assert_close(got: f32, expected: f32, label: &str) {
    assert!(
        (got - expected).abs() < BF16_EPS,
        "{}: got {}, expected {} (diff {})",
        label,
        got,
        expected,
        (got - expected).abs()
    );
}

// ---------------------------------------------------------------------------
// Multi-adapter loader test: LoCon Linear + LoHa Linear + LoKr Linear + Full
// ---------------------------------------------------------------------------

#[test]
fn loader_round_trip_all_four_adapter_kinds() {
    let Some(device) = try_device() else {
        eprintln!("loader::all_four: no CUDA device, skipping");
        return;
    };

    // --- LoCon Linear: down [R=2, IN=4] ones, up [OUT=8, R=2] all 1s.
    // Internal: scale=alpha/rank=2/2=1, ΔW[i,o] = sum_r 1*1 = 2.
    let locon_prefix = "lora_unet_block_0_attn_to_q";
    let locon_down = f32_tensor(vec![1.0; 2 * 4], &[2, 4], device.clone());
    let locon_up = f32_tensor(vec![1.0; 8 * 2], &[8, 2], device.clone());
    let locon_alpha = f32_tensor(vec![2.0], &[1], device.clone());

    // --- LoHa Linear: w1a, w1b, w2a, w2b all ones, alpha=2, rank=2.
    // Per smoke: w1[i,o]=2, w2[i,o]=2, ΔW=4*scale=4.
    let loha_prefix = "lora_unet_block_0_attn_to_k";
    let loha_w1a = f32_tensor(vec![1.0; 2 * 4], &[2, 4], device.clone()); // [R, IN]
    let loha_w1b = f32_tensor(vec![1.0; 8 * 2], &[8, 2], device.clone()); // [OUT, R]
    let loha_w2a = f32_tensor(vec![1.0; 2 * 4], &[2, 4], device.clone());
    let loha_w2b = f32_tensor(vec![1.0; 8 * 2], &[8, 2], device.clone());
    let loha_alpha = f32_tensor(vec![2.0], &[1], device.clone());

    // --- LoKr Linear with full W1 + full W2 (P0-4 path).
    // W1=[[1,2],[3,4]], W2=[[5,6],[7,8]], scale=1.
    let lokr_prefix = "lora_unet_block_0_attn_to_v";
    let lokr_w1 = f32_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], device.clone());
    let lokr_w2 = f32_tensor(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], device.clone());
    // No alpha → defaults to rank.max(1)=1.

    // --- Full Linear with diff and diff_b.
    let full_prefix = "lora_unet_block_0_proj_out";
    let full_diff = f32_tensor(vec![0.5, 0.0, 0.0, 0.5], &[2, 2], device.clone());
    let full_diff_b = f32_tensor(vec![1.0, 2.0], &[2], device.clone());

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert(format!("{}.lora_down.weight", locon_prefix), locon_down);
    tensors.insert(format!("{}.lora_up.weight", locon_prefix), locon_up);
    tensors.insert(format!("{}.alpha", locon_prefix), locon_alpha);

    tensors.insert(format!("{}.hada_w1_a", loha_prefix), loha_w1a);
    tensors.insert(format!("{}.hada_w1_b", loha_prefix), loha_w1b);
    tensors.insert(format!("{}.hada_w2_a", loha_prefix), loha_w2a);
    tensors.insert(format!("{}.hada_w2_b", loha_prefix), loha_w2b);
    tensors.insert(format!("{}.alpha", loha_prefix), loha_alpha);

    tensors.insert(format!("{}.lokr_w1", lokr_prefix), lokr_w1);
    tensors.insert(format!("{}.lokr_w2", lokr_prefix), lokr_w2);

    tensors.insert(format!("{}.diff", full_prefix), full_diff);
    tensors.insert(format!("{}.diff_b", full_prefix), full_diff_b);

    let path = tmp_path("all_four");
    flame_core::serialization::save_file(&tensors, &path).expect("save safetensors");

    let coll = LycorisCollection::load(&path, device.clone()).expect("load collection");

    // Cleanup tmp file.
    let _ = std::fs::remove_file(&path);

    // 1) All 4 prefixes recognised, right kinds.
    assert_eq!(coll.adapters.len(), 4, "expected 4 adapters");
    match coll.adapters.get(locon_prefix) {
        Some(LycorisAdapter::LoCon(_)) => {}
        other => panic!("LoCon prefix not detected; got {:?}", other.is_some()),
    }
    match coll.adapters.get(loha_prefix) {
        Some(LycorisAdapter::LoHa(_)) => {}
        _ => panic!("LoHa prefix not detected"),
    }
    match coll.adapters.get(lokr_prefix) {
        Some(LycorisAdapter::LoKr(_)) => {}
        _ => panic!("LoKr prefix not detected"),
    }
    match coll.adapters.get(full_prefix) {
        Some(LycorisAdapter::Full(_)) => {}
        _ => panic!("Full prefix not detected"),
    }

    // 2) ΔW values match the hand-computed expectations.
    // LoCon: every element = 2.0, shape [IN=4, OUT=8] (Flame convention).
    let locon_dw = coll
        .adapters
        .get(locon_prefix)
        .unwrap()
        .delta_weight()
        .unwrap();
    assert_eq!(locon_dw.dims(), &[4, 8], "LoCon ΔW shape");
    let v = locon_dw.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    assert_close(v[0], 2.0, "LoCon ΔW[0,0]");
    assert_close(v[v.len() - 1], 2.0, "LoCon ΔW[last]");

    // LoHa: every element = 4.0, shape [IN=4, OUT=8].
    let loha_dw = coll
        .adapters
        .get(loha_prefix)
        .unwrap()
        .delta_weight()
        .unwrap();
    assert_eq!(loha_dw.dims(), &[4, 8], "LoHa ΔW shape");
    let v = loha_dw.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    assert_close(v[0], 4.0, "LoHa ΔW[0,0]");
    assert_close(v[v.len() - 1], 4.0, "LoHa ΔW[last]");

    // LoKr: kron(W1, W2) [4, 4]. Spot checks per upstream Kronecker semantics.
    let lokr_dw = coll
        .adapters
        .get(lokr_prefix)
        .unwrap()
        .delta_weight()
        .unwrap();
    assert_eq!(lokr_dw.dims(), &[4, 4], "LoKr ΔW shape");
    let v = lokr_dw.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    // Reference kron table:
    //   [[ 5, 6,10,12],
    //    [ 7, 8,14,16],
    //    [15,18,20,24],
    //    [21,24,28,32]]
    assert_close(v[0 * 4 + 0], 5.0, "LoKr ΔW[0,0]");
    assert_close(v[3 * 4 + 3], 32.0, "LoKr ΔW[3,3]");

    // Full: diff is loaded as-is (no transposes), shape [2, 2].
    let full_dw = coll
        .adapters
        .get(full_prefix)
        .unwrap()
        .delta_weight()
        .unwrap();
    assert_eq!(full_dw.dims(), &[2, 2], "Full ΔW shape");
    let v = full_dw.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    assert_close(v[0], 0.5, "Full ΔW[0,0]");
    assert_close(v[3], 0.5, "Full ΔW[1,1]");
}

// ---------------------------------------------------------------------------
// P0-7: end-to-end .diff_b apply through apply_to / apply_collection.
// ---------------------------------------------------------------------------

#[test]
fn loader_full_diff_b_propagates_through_apply_to() {
    let Some(device) = try_device() else {
        eprintln!("loader::diff_b: no CUDA device, skipping");
        return;
    };

    let prefix = "lora_unet_block_0_proj_out";
    let diff = f32_tensor(vec![0.1, 0.2, 0.3, 0.4], &[2, 2], device.clone());
    let diff_b = f32_tensor(vec![1.0, 2.0], &[2], device.clone());

    let mut entries: HashMap<String, Tensor> = HashMap::new();
    entries.insert(format!("{}.diff", prefix), diff);
    entries.insert(format!("{}.diff_b", prefix), diff_b);

    let path = tmp_path("full_diff_b");
    flame_core::serialization::save_file(&entries, &path).expect("save");

    let coll = LycorisCollection::load(&path, device.clone()).expect("load");
    let _ = std::fs::remove_file(&path);

    // Build a base weights map containing both .weight and .bias for the
    // mapped key, plus a few unrelated entries to make sure they're left alone.
    let base_w = bf16_tensor(vec![10.0; 4], &[2, 2], device.clone());
    let base_b = bf16_tensor(vec![100.0; 2], &[2], device.clone());
    let unrelated = bf16_tensor(vec![42.0; 4], &[2, 2], device.clone());

    let mut weights: HashMap<String, Tensor> = HashMap::new();
    weights.insert("model.layer.weight".to_string(), base_w);
    weights.insert("model.layer.bias".to_string(), base_b);
    weights.insert("model.unrelated.weight".to_string(), unrelated);

    coll.apply_to(&mut weights, 1.0, |p| {
        if p == prefix {
            Some("model.layer.weight".to_string())
        } else {
            None
        }
    })
    .expect("apply_to");

    // weight: 10 + diff
    let w = weights
        .get("model.layer.weight")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec()
        .unwrap();
    assert_close(w[0], 10.1, "merged W[0]");
    assert_close(w[3], 10.4, "merged W[3]");

    // bias: 100 + diff_b. P0-7 regression: previously this stayed at 100.
    let b = weights
        .get("model.layer.bias")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec()
        .unwrap();
    assert_close(b[0], 101.0, "merged b[0] (P0-7)");
    assert_close(b[1], 102.0, "merged b[1] (P0-7)");

    // Unrelated key untouched.
    let u = weights
        .get("model.unrelated.weight")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec()
        .unwrap();
    assert_close(u[0], 42.0, "unrelated untouched");
}

// ---------------------------------------------------------------------------
// P0-6: DoRA detection. Adapter with `.dora_scale` must be skipped with a
// loud message rather than silently merged with wrong math.
// ---------------------------------------------------------------------------

#[test]
fn loader_skips_dora_adapter_loudly() {
    let Some(device) = try_device() else {
        eprintln!("loader::dora: no CUDA device, skipping");
        return;
    };

    let prefix = "lora_unet_block_0_attn_to_q";
    // A normal LoCon adapter, but with a .dora_scale entry.
    let down = f32_tensor(vec![1.0; 2 * 4], &[2, 4], device.clone());
    let up = f32_tensor(vec![1.0; 8 * 2], &[8, 2], device.clone());
    let alpha = f32_tensor(vec![2.0], &[1], device.clone());
    let dora_scale = f32_tensor(vec![1.0; 8], &[8], device.clone());

    let mut entries: HashMap<String, Tensor> = HashMap::new();
    entries.insert(format!("{}.lora_down.weight", prefix), down);
    entries.insert(format!("{}.lora_up.weight", prefix), up);
    entries.insert(format!("{}.alpha", prefix), alpha);
    entries.insert(format!("{}.dora_scale", prefix), dora_scale);

    // Add a clean adapter to verify mixed loads still work.
    let clean_prefix = "lora_unet_block_1_attn_to_q";
    let clean_down = f32_tensor(vec![1.0; 2 * 4], &[2, 4], device.clone());
    let clean_up = f32_tensor(vec![1.0; 8 * 2], &[8, 2], device.clone());
    entries.insert(format!("{}.lora_down.weight", clean_prefix), clean_down);
    entries.insert(format!("{}.lora_up.weight", clean_prefix), clean_up);

    let path = tmp_path("dora");
    flame_core::serialization::save_file(&entries, &path).expect("save");
    let coll = LycorisCollection::load(&path, device.clone()).expect("load");
    let _ = std::fs::remove_file(&path);

    // DoRA prefix must NOT be in the collection.
    assert!(
        !coll.adapters.contains_key(prefix),
        "P0-6: DoRA adapter '{}' must be skipped, not loaded",
        prefix
    );
    // Clean prefix MUST be loaded.
    assert!(
        coll.adapters.contains_key(clean_prefix),
        "clean adapter '{}' must still be loaded alongside skipped DoRA",
        clean_prefix
    );
    assert_eq!(coll.adapters.len(), 1, "only the clean adapter survives");
}

// ---------------------------------------------------------------------------
// P0-3: Tucker LoHa loader detection. Synthetic Tucker LoHa with 2D w1a/w2a.
// Before fix, w1a being 2D set is_conv=false and t1/t2 were dropped silently.
// After fix, presence of .hada_t1/.hada_t2 is_conv flips to Tucker+conv path.
// ---------------------------------------------------------------------------

#[test]
fn loader_detects_tucker_loha_via_t1_t2() {
    let Some(device) = try_device() else {
        eprintln!("loader::tucker_loha: no CUDA device, skipping");
        return;
    };

    // Setup mirrors the in-memory smoke test for Tucker LoHa, but built
    // from on-disk Python layout:
    //   w1d, w2d : [R, IN]   (2D)
    //   w1u, w2u : [R, OUT]  (2D)
    //   t1, t2   : [R, R, KH, KW]
    let prefix = "lora_unet_conv_block_0";
    let kh = 2usize;
    let kw = 2usize;
    let inn = 3usize;
    let out = 4usize;
    let r = 2usize;

    let w1d = f32_tensor(vec![1.0; r * inn], &[r, inn], device.clone());
    let w1u = f32_tensor(vec![1.0; r * out], &[r, out], device.clone());
    let w2d = f32_tensor(vec![1.0; r * inn], &[r, inn], device.clone());
    let mut w2u_data = vec![0.0f32; r * out];
    for o in 0..out {
        w2u_data[0 * out + o] = 1.0;
    }
    let w2u = f32_tensor(w2u_data, &[r, out], device.clone());
    let t1 = f32_tensor(vec![1.0; r * r * kh * kw], &[r, r, kh, kw], device.clone());
    let t2 = f32_tensor(vec![1.0; r * r * kh * kw], &[r, r, kh, kw], device.clone());
    let alpha = f32_tensor(vec![r as f32], &[1], device.clone());

    let mut entries: HashMap<String, Tensor> = HashMap::new();
    entries.insert(format!("{}.hada_w1_a", prefix), w1d);
    entries.insert(format!("{}.hada_w1_b", prefix), w1u);
    entries.insert(format!("{}.hada_w2_a", prefix), w2d);
    entries.insert(format!("{}.hada_w2_b", prefix), w2u);
    entries.insert(format!("{}.hada_t1", prefix), t1);
    entries.insert(format!("{}.hada_t2", prefix), t2);
    entries.insert(format!("{}.alpha", prefix), alpha);

    let path = tmp_path("tucker_loha");
    flame_core::serialization::save_file(&entries, &path).expect("save");
    let coll = LycorisCollection::load(&path, device.clone()).expect("load");
    let _ = std::fs::remove_file(&path);

    let adapter = coll.adapters.get(prefix).expect("Tucker LoHa loaded");
    let dw = adapter.delta_weight().expect("Tucker LoHa delta");
    // P0-3 fix: should be a 4D conv kernel [KH, KW, IN, OUT], NOT a 2D matrix.
    assert_eq!(
        dw.dims(),
        &[kh, kw, inn, out],
        "P0-3: Tucker LoHa must yield [KH,KW,IN,OUT] shape (conv), not 2D"
    );
    let v = dw.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    for &x in &v {
        assert!(x.is_finite(), "Tucker LoHa output contains non-finite");
    }
    // Reference value: branch1 = 4 (R*R), branch2 = 2 (R*1 + R*0 = 2), product=8.
    assert_close(v[0], 8.0, "Tucker LoHa ΔW[0,0,0,0] from loader");
    assert_close(v[v.len() - 1], 8.0, "Tucker LoHa ΔW[last] from loader");
}
