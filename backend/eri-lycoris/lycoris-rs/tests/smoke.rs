//! Smoke tests for the four LyCORIS adapter variants.
//!
//! Each test constructs an adapter in memory (no file IO), calls
//! `delta_weight()` / `get_diff_weight()`, asserts the resulting shape, and
//! spot-checks one or two elements against a hand-computed value.
//!
//! These are CUDA tests — they require a GPU to run. They are skipped when
//! `CudaDevice::new(0)` fails.

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::parameter::Parameter;
use flame_core::{DType, Shape, Tensor};

use lycoris_rs::algorithms::{
    full::FullAdapter,
    locon::LoConModule,
    loha::LoHaModule,
    lokr::LoKrModule,
};
use lycoris_rs::tensor_utils;
use lycoris_rs::LycorisModule;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn try_device() -> Option<Arc<CudaDevice>> {
    CudaDevice::new(0).ok()
}

fn bf16(data: Vec<f32>, dims: &[usize], device: Arc<CudaDevice>) -> Tensor {
    tensor_utils::from_vec_bf16(data, Shape::from_dims(dims), device)
        .expect("build BF16 tensor")
}

/// BF16 precision for spot-checks: ~3 decimal digits of mantissa.
const BF16_EPS: f32 = 1e-2;

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
// LoCon: linear 4 → 8, rank=2, alpha=2. Scale = alpha/rank = 1.0.
//
//   down = 1s everywhere  shape [IN=4, R=2]
//   up[0, :] = 1s, up[1, :] = 0s  shape [R=2, OUT=8]
//   ΔW = down @ up — every element = 1*1 + 1*0 = 1
//   scale = 1  →  ΔW[i,j] = 1.0  for all (i,j)
// ---------------------------------------------------------------------------

#[test]
fn locon_linear_delta_shape_and_value() {
    let Some(device) = try_device() else {
        eprintln!("smoke::locon: no CUDA device, skipping");
        return;
    };

    let down = bf16(vec![1.0; 4 * 2], &[4, 2], device.clone());
    let up_data: Vec<f32> = (0..2 * 8)
        .map(|i| if i < 8 { 1.0 } else { 0.0 })
        .collect();
    let up = bf16(up_data, &[2, 8], device.clone());

    let m = LoConModule {
        down: Parameter::new(down),
        up: Parameter::new(up),
        mid: None,
        rank: 2,
        alpha: 2.0,
        device: device.clone(),
        is_conv: false,
    };

    let dw = m.get_diff_weight().expect("delta weight");
    assert_eq!(dw.dims(), &[4, 8], "LoCon linear ΔW shape");

    let dw_f32 = dw.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    // Every element should be 1.0
    assert_close(dw_f32[0], 1.0, "LoCon ΔW[0,0]");
    assert_close(dw_f32[15], 1.0, "LoCon ΔW[1,7]");
    assert_close(dw_f32[31], 1.0, "LoCon ΔW[3,7]");
}

// ---------------------------------------------------------------------------
// LoHa: 4x4 linear. Both branches rank=2, alpha=2, scale=1.0.
//
//   w1a = all 1s [4,2], w1b = all 1s [2,4]  →  w1 = all 2s [4,4]
//   w2a = all 1s [4,2], w2b[0,:]=1, w2b[1,:]=0 → w2[i,j] = 1*1 + 1*0 = 1
//   diff = w1 ⊙ w2 = 2  (for every element)
//   scale = 1  →  ΔW = 2 everywhere
// ---------------------------------------------------------------------------

#[test]
fn loha_linear_delta_shape_and_value() {
    let Some(device) = try_device() else {
        eprintln!("smoke::loha: no CUDA device, skipping");
        return;
    };

    let w1a = bf16(vec![1.0; 4 * 2], &[4, 2], device.clone());
    let w1b = bf16(vec![1.0; 2 * 4], &[2, 4], device.clone());
    let w2a = bf16(vec![1.0; 4 * 2], &[4, 2], device.clone());
    let w2b_data: Vec<f32> = (0..2 * 4)
        .map(|i| if i < 4 { 1.0 } else { 0.0 })
        .collect();
    let w2b = bf16(w2b_data, &[2, 4], device.clone());

    let m = LoHaModule {
        w1a: Parameter::new(w1a),
        w1b: Parameter::new(w1b),
        w2a: Parameter::new(w2a),
        w2b: Parameter::new(w2b),
        t1: None,
        t2: None,
        rank: 2,
        alpha: 2.0,
        device: device.clone(),
        is_conv: false,
    };

    let dw = m.get_diff_weight().expect("delta weight");
    assert_eq!(dw.dims(), &[4, 4], "LoHa linear ΔW shape");

    let dw_f32 = dw.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    // Every element should be 2.0
    assert_close(dw_f32[0], 2.0, "LoHa ΔW[0,0]");
    assert_close(dw_f32[5], 2.0, "LoHa ΔW[1,1]");
    assert_close(dw_f32[15], 2.0, "LoHa ΔW[3,3]");
}

// ---------------------------------------------------------------------------
// LoKr: linear 4×4 formed as kron(W1:[2,2], W2:[2,2]).
//
//   W1 = [[1,2],[3,4]]
//   W2 = [[5,6],[7,8]]
//   kron:
//     [ 1·W2   2·W2 ]      [[ 5,  6, 10, 12],
//     [ 3·W2   4·W2 ]  →    [ 7,  8, 14, 16],
//                           [15, 18, 20, 24],
//                           [21, 24, 28, 32]]
//   Spot-check [0,0] = 5 and [3,3] = 32.
// ---------------------------------------------------------------------------

#[test]
fn lokr_linear_delta_shape_and_value() {
    let Some(device) = try_device() else {
        eprintln!("smoke::lokr: no CUDA device, skipping");
        return;
    };

    // In LoKr the linear path is `kron(W1, W2) * scale` with W1:[OL,IM], W2:[OK,IN].
    // Here OL=OK=IM=IN=2 ⇒ out shape [4,4].
    let w1 = bf16(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], device.clone());
    let w2 = bf16(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], device.clone());

    // rank=1, alpha=1 ⇒ scale = 1.0 (so the kron values aren't scaled away).
    let m = LoKrModule {
        w1: Some(Parameter::new(w1)),
        w1a: None,
        w1b: None,
        w2: Some(Parameter::new(w2)),
        w2a: None,
        w2b: None,
        t2: None,
        rank: 1,
        alpha: 1.0,
        device: device.clone(),
        shape: ((2, 2), (2, 2)),
        is_conv: false,
    };

    let dw = m.get_diff_weight().expect("delta weight");
    assert_eq!(dw.dims(), &[4, 4], "LoKr linear ΔW shape");

    let v = dw.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    // Row-major indexing into [4,4]: idx = row*4 + col.
    assert_close(v[0 * 4 + 0], 5.0, "LoKr ΔW[0,0]");
    assert_close(v[0 * 4 + 1], 6.0, "LoKr ΔW[0,1]");
    assert_close(v[1 * 4 + 0], 7.0, "LoKr ΔW[1,0]");
    assert_close(v[2 * 4 + 2], 20.0, "LoKr ΔW[2,2]");
    assert_close(v[3 * 4 + 3], 32.0, "LoKr ΔW[3,3]");
}

// ---------------------------------------------------------------------------
// Full: diff = [[1,0],[0,1]], strength = 0.5 → [[0.5, 0], [0, 0.5]].
// ---------------------------------------------------------------------------

#[test]
fn full_delta_shape_and_value() {
    let Some(device) = try_device() else {
        eprintln!("smoke::full: no CUDA device, skipping");
        return;
    };

    let diff = bf16(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], device.clone());
    let m = FullAdapter {
        diff: Parameter::new(diff),
        diff_b: None,
    };

    let dw = m.delta_weight(0.5).expect("delta weight");
    assert_eq!(dw.dims(), &[2, 2], "Full ΔW shape");

    let v = dw.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    assert_close(v[0], 0.5, "Full ΔW[0,0]");
    assert_close(v[1], 0.0, "Full ΔW[0,1]");
    assert_close(v[2], 0.0, "Full ΔW[1,0]");
    assert_close(v[3], 0.5, "Full ΔW[1,1]");

    // strength = 1.0 should short-circuit to returning the tensor as-is.
    let dw1 = m.delta_weight(1.0).expect("delta 1.0");
    let v1 = dw1.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    assert_close(v1[0], 1.0, "Full strength=1 ΔW[0,0]");
    assert_close(v1[3], 1.0, "Full strength=1 ΔW[1,1]");
}

// ---------------------------------------------------------------------------
// LoCon Conv2d (non-Tucker spatial). down [KH,KW,IC,R], up [KH,KW,R,OC].
// IC=8, OC=16, KH=KW=3, R=4, alpha=4 → scale=1.
//
// down = ones [3,3,8,4]
// up   = [3,3,4,16] with up[h,w,0,:]=1 and up[h,w,1:,:]=0
//
// per spatial pos (h,w):  ΔW[h,w,i,o] = sum_r down[h,w,i,r]*up[h,w,r,o]
//                                     = 1*1 + 1*0 + 1*0 + 1*0 = 1
// ---------------------------------------------------------------------------

#[test]
fn locon_conv2d_delta_shape_and_value() {
    let Some(device) = try_device() else {
        eprintln!("smoke::locon_conv: no CUDA device, skipping");
        return;
    };

    let kh = 3usize;
    let kw = 3usize;
    let ic = 8usize;
    let oc = 16usize;
    let r = 4usize;

    let down = bf16(vec![1.0; kh * kw * ic * r], &[kh, kw, ic, r], device.clone());
    // up_data[h, w, rr, o] = 1 if rr == 0 else 0
    let mut up_data = vec![0.0f32; kh * kw * r * oc];
    for h in 0..kh {
        for w in 0..kw {
            for o in 0..oc {
                // rr = 0
                let idx = ((h * kw + w) * r + 0) * oc + o;
                up_data[idx] = 1.0;
            }
        }
    }
    let up = bf16(up_data, &[kh, kw, r, oc], device.clone());

    let m = LoConModule {
        down: Parameter::new(down),
        up: Parameter::new(up),
        mid: None,
        rank: r,
        alpha: r as f32, // scale = 1.0
        device: device.clone(),
        is_conv: true,
    };

    let dw = m.get_diff_weight().expect("conv delta weight");
    assert_eq!(
        dw.dims(),
        &[kh, kw, ic, oc],
        "LoCon conv non-Tucker ΔW shape"
    );

    let v = dw.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    // Spot-check a few positions: every element should be 1.0.
    let center = ((1 * kw + 1) * ic + 4) * oc + 7;
    let corner = 0;
    let last = kh * kw * ic * oc - 1;
    assert_close(v[center], 1.0, "LoCon conv ΔW center [1,1,4,7]");
    assert_close(v[corner], 1.0, "LoCon conv ΔW corner [0,0,0,0]");
    assert_close(v[last], 1.0, "LoCon conv ΔW last [2,2,7,15]");
}

// ---------------------------------------------------------------------------
// LoCon Conv2d Tucker (P0-1 regression test).
//
// Before fix: returned Err("Tucker conv decomposition requires full tensor
// contraction implementation"). After fix: dispatches to rebuild_conv_tucker
// in src/ops/tucker.rs and returns a real ΔW.
//
// Setup: KH=KW=2, IC=4, OC=4, R=2.
//   mid  = ones [2, 2, 2, 2]
//   down = ones [1, 1, 4, 2]  (IC=4, R=2)
//   up   = [1, 1, 2, 4] with up[0,0,0,:]=1, up[0,0,1,:]=0
//
// Per spatial (h,w): mid[h,w,:,:] is ones[2,2].
//   down[IC=4, R=2] @ mid_hw[R=2, R=2] → all 2s [4, 2]
//   [4, 2] @ up_2d[R=2, OC=4] (row 0 ones, row 1 zeros) → all 2s [4, 4]
// alpha=R=2 → scale=1, result element = 2.0 everywhere.
// ---------------------------------------------------------------------------

#[test]
fn locon_conv2d_tucker_works_after_p0_1_fix() {
    let Some(device) = try_device() else {
        eprintln!("smoke::locon_tucker: no CUDA device, skipping");
        return;
    };

    let kh = 2usize;
    let kw = 2usize;
    let ic = 4usize;
    let oc = 4usize;
    let r = 2usize;

    let mid = bf16(vec![1.0; kh * kw * r * r], &[kh, kw, r, r], device.clone());
    let down = bf16(vec![1.0; 1 * 1 * ic * r], &[1, 1, ic, r], device.clone());
    // up: only row 0 of R is ones, row 1 is zeros (per OC).
    let mut up_data = vec![0.0f32; 1 * 1 * r * oc];
    for o in 0..oc {
        up_data[0 * oc + o] = 1.0; // rr=0
    }
    let up = bf16(up_data, &[1, 1, r, oc], device.clone());

    let m = LoConModule {
        down: Parameter::new(down),
        up: Parameter::new(up),
        mid: Some(Parameter::new(mid)),
        rank: r,
        alpha: r as f32, // scale = 1.0
        device: device.clone(),
        is_conv: true,
    };

    let dw = m
        .get_diff_weight()
        .expect("Tucker LoCon delta weight (P0-1 fix)");
    assert_eq!(dw.dims(), &[kh, kw, ic, oc], "LoCon Tucker ΔW shape");

    let v = dw.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    // No NaN / Inf
    for (i, &x) in v.iter().enumerate() {
        assert!(x.is_finite(), "non-finite at idx {}: {}", i, x);
    }
    // Hand-computed: every element = 2.0
    assert_close(v[0], 2.0, "LoCon Tucker ΔW[0,0,0,0]");
    let mid_idx = ((1 * kw + 1) * ic + 2) * oc + 3;
    assert_close(v[mid_idx], 2.0, "LoCon Tucker ΔW[1,1,2,3]");
    assert_close(v[v.len() - 1], 2.0, "LoCon Tucker ΔW[last]");
}

// ---------------------------------------------------------------------------
// LoHa Conv2d (non-Tucker spatial 1×1).
//
// Internal layout matches loader: w1a/w2a [1,1,IC,R], w1b/w2b [1,1,R,OC].
// IC=4, OC=8, R=2, alpha=2 → scale=1.
//
//   w1a = ones [1,1,4,2], w1b = ones [1,1,2,8]   →  w1[i,o] = 2
//   w2a = ones [1,1,4,2], w2b[0,0,0,:]=1, w2b[0,0,1,:]=0  →  w2[i,o] = 1
//   ΔW[0,0,i,o] = w1*w2*scale = 2*1*1 = 2
// ---------------------------------------------------------------------------

#[test]
fn loha_conv2d_1x1_delta_shape_and_value() {
    let Some(device) = try_device() else {
        eprintln!("smoke::loha_conv: no CUDA device, skipping");
        return;
    };

    let ic = 4usize;
    let oc = 8usize;
    let r = 2usize;

    let w1a = bf16(vec![1.0; ic * r], &[1, 1, ic, r], device.clone());
    let w1b = bf16(vec![1.0; r * oc], &[1, 1, r, oc], device.clone());
    let w2a = bf16(vec![1.0; ic * r], &[1, 1, ic, r], device.clone());
    let mut w2b_data = vec![0.0f32; r * oc];
    for o in 0..oc {
        w2b_data[0 * oc + o] = 1.0;
    }
    let w2b = bf16(w2b_data, &[1, 1, r, oc], device.clone());

    let m = LoHaModule {
        w1a: Parameter::new(w1a),
        w1b: Parameter::new(w1b),
        w2a: Parameter::new(w2a),
        w2b: Parameter::new(w2b),
        t1: None,
        t2: None,
        rank: r,
        alpha: r as f32, // scale=1.0
        device: device.clone(),
        is_conv: true,
    };

    let dw = m.get_diff_weight().expect("LoHa conv 1x1 delta");
    assert_eq!(dw.dims(), &[1, 1, ic, oc], "LoHa conv 1x1 ΔW shape");

    let v = dw.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    assert_close(v[0], 2.0, "LoHa conv ΔW[0,0,0,0]");
    assert_close(v[ic * oc - 1], 2.0, "LoHa conv ΔW[last]");
}

// ---------------------------------------------------------------------------
// LoKr Conv2d with full W2 (1×1 kernel detected as conv via lokr.rs is_conv).
//
// We exercise the make_kronecker_conv_kernel path by giving W2 a real
// spatial dim. Actually the easier-to-verify case: kron(W1:[OL,IM], W2:[OK,IN,KH,KW])
// with OL=OK=IM=IN=2, KH=KW=2. Just verify the result is finite, has the
// right shape, and one corner element checks out.
//
// W1 = [[1, 2], [3, 4]]
// W2 = ones [2, 2, 2, 2]
// scale = 1 (rank=0 → P0-4 fix returns 1.0 default)
//
// kernel layout out: [KH=2, KW=2, IC=IM*IN=4, OC=OL*OK=4]
// At any [kh, kw], kernel[kh, kw, im_in, ol_ok] = w1[ol, im] * w2[ok, in, kh, kw]
// where im_in = im*IN + in_idx, ol_ok = ol*OK + ok.
// W2 is all ones so the spatial pattern is uniform.
// At (im=0,in=0,ol=0,ok=0): kernel = w1[0,0] * 1 = 1
// At (im=1,in=1,ol=1,ok=1): kernel = w1[1,1] * 1 = 4
// ---------------------------------------------------------------------------

#[test]
fn lokr_conv2d_full_w2_delta_shape_and_value() {
    let Some(device) = try_device() else {
        eprintln!("smoke::lokr_conv: no CUDA device, skipping");
        return;
    };

    let w1 = bf16(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], device.clone());
    let w2 = bf16(vec![1.0; 2 * 2 * 2 * 2], &[2, 2, 2, 2], device.clone());

    // Note: rank=0, alpha=1 — P0-4 means scale=1.0 instead of 0.0.
    let m = LoKrModule {
        w1: Some(Parameter::new(w1)),
        w1a: None,
        w1b: None,
        w2: Some(Parameter::new(w2)),
        w2a: None,
        w2b: None,
        t2: None,
        rank: 0,
        alpha: 1.0,
        device: device.clone(),
        shape: ((2, 2), (2, 2)),
        is_conv: true,
    };

    let dw = m.get_diff_weight().expect("LoKr conv full W2 delta");
    // Shape: [KH=2, KW=2, IC=IM*IN=4, OC=OL*OK=4]
    assert_eq!(dw.dims(), &[2, 2, 4, 4], "LoKr conv ΔW shape");

    let v = dw.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    for (i, &x) in v.iter().enumerate() {
        assert!(x.is_finite(), "non-finite at idx {}: {}", i, x);
    }
    // The output layout is [KH, KW, IC, OC]. At any (kh, kw), the IC×OC slab
    // is kron(W1:[OL,IM], W2_slab:[OK,IN]) where W2_slab is ones[OK,IN].
    // kron(W1, ones2) puts W1[i,j] in a 2×2 block: every position [i*2+k, j*2+l] = W1[i,j].
    // Let's check the [0,0] spatial slab: should mirror this kron pattern.
    // dw at (kh=0, kw=0, ic=0, oc=0): block=(im=0,in=0,ol=0,ok=0) → w1[0,0]=1
    let i00 = ((0 * 2 + 0) * 4 + 0) * 4 + 0;
    assert_close(v[i00], 1.0, "LoKr conv ΔW[0,0,0,0]");
    // (kh=0, kw=0, ic=3 [im=1,in=1], oc=3 [ol=1,ok=1]): w1[1,1]=4
    let i33 = ((0 * 2 + 0) * 4 + 3) * 4 + 3;
    assert_close(v[i33], 4.0, "LoKr conv ΔW[0,0,3,3]");
}

// ---------------------------------------------------------------------------
// P0-4 regression: LoKr Linear with full W1 + full W2 (rank=0). Before
// the scale_from fix, scale was 0.0 → make_kronecker early-exits to a
// zeros tensor, silently no-op-ing the adapter. After fix, scale=1.0.
// ---------------------------------------------------------------------------

#[test]
fn lokr_linear_full_w1_full_w2_not_zero_after_p0_4_fix() {
    let Some(device) = try_device() else {
        eprintln!("smoke::lokr_p0_4: no CUDA device, skipping");
        return;
    };

    // Same as the existing lokr_linear test but with rank=0, simulating
    // a loader-built LoKrModule that couldn't infer a rank.
    let w1 = bf16(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], device.clone());
    let w2 = bf16(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], device.clone());

    let m = LoKrModule {
        w1: Some(Parameter::new(w1)),
        w1a: None,
        w1b: None,
        w2: Some(Parameter::new(w2)),
        w2a: None,
        w2b: None,
        t2: None,
        rank: 0,    // ← P0-4 trigger
        alpha: 1.0,
        device: device.clone(),
        shape: ((2, 2), (2, 2)),
        is_conv: false,
    };

    let dw = m.get_diff_weight().expect("LoKr P0-4 delta");
    assert_eq!(dw.dims(), &[4, 4]);
    let v = dw.to_dtype(DType::F32).unwrap().to_vec().unwrap();

    // If P0-4 weren't fixed every value would be 0.0. Verify the kron
    // values come through (same as the scale=1 reference).
    assert_close(v[0], 5.0, "P0-4 LoKr ΔW[0,0]");
    assert_close(v[3 * 4 + 3], 32.0, "P0-4 LoKr ΔW[3,3]");
}

// ---------------------------------------------------------------------------
// P0-7 regression: Full adapter must carry diff_b through delta_bias().
// ---------------------------------------------------------------------------

#[test]
fn full_adapter_diff_b_propagates_after_p0_7_fix() {
    let Some(device) = try_device() else {
        eprintln!("smoke::full_diff_b: no CUDA device, skipping");
        return;
    };

    let diff = bf16(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], device.clone());
    let diff_b = bf16(vec![10.0, 20.0], &[2], device.clone());
    let m = FullAdapter {
        diff: Parameter::new(diff),
        diff_b: Some(Parameter::new(diff_b)),
    };

    // strength=0.5
    let bias = m
        .delta_bias(0.5)
        .expect("delta_bias")
        .expect("Some(bias)");
    assert_eq!(bias.dims(), &[2]);
    let v = bias.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    assert_close(v[0], 5.0, "Full Δb[0] @ strength=0.5");
    assert_close(v[1], 10.0, "Full Δb[1] @ strength=0.5");

    // None case
    let m_no_bias = FullAdapter {
        diff: Parameter::new(bf16(vec![1.0; 4], &[2, 2], device.clone())),
        diff_b: None,
    };
    let none = m_no_bias.delta_bias(1.0).expect("delta_bias none");
    assert!(none.is_none(), "no diff_b → None");
}

// ---------------------------------------------------------------------------
// P0-2 regression: Tucker LoHa weight reconstruction.
//
// Internal layout (post-loader): t1/t2 [KH,KW,R,R], w1a/w2a [IN,R], w1b/w2b [R,OUT].
//
// Setup: KH=KW=2, IN=3, OUT=4, R=2. All tensors filled to make hand-math tractable.
//   t1 = ones[2,2,2,2], t2 = ones[2,2,2,2]
//   w1a = ones[3,2], w1b = ones[2,4]   → branch1[any] = (R*R)*1 = 4 (per math walked above)
//   w2a = ones[3,2], w2b: row 0 ones, row 1 zeros → branch2[any] = 2
//
// scale=1.0 (alpha=R=2). hadamard = 4*2 = 8.
// ---------------------------------------------------------------------------

#[test]
fn loha_conv2d_tucker_works_after_p0_2_fix() {
    let Some(device) = try_device() else {
        eprintln!("smoke::loha_tucker: no CUDA device, skipping");
        return;
    };

    let kh = 2usize;
    let kw = 2usize;
    let inn = 3usize;
    let out = 4usize;
    let r = 2usize;

    let t1 = bf16(vec![1.0; kh * kw * r * r], &[kh, kw, r, r], device.clone());
    let t2 = bf16(vec![1.0; kh * kw * r * r], &[kh, kw, r, r], device.clone());

    // Tucker layout (post-loader): w1a [IN, R], w1b [R, OUT].
    let w1a = bf16(vec![1.0; inn * r], &[inn, r], device.clone());
    let w1b = bf16(vec![1.0; r * out], &[r, out], device.clone());
    let w2a = bf16(vec![1.0; inn * r], &[inn, r], device.clone());
    let mut w2b_data = vec![0.0f32; r * out];
    for o in 0..out {
        w2b_data[0 * out + o] = 1.0;
    }
    let w2b = bf16(w2b_data, &[r, out], device.clone());

    let m = LoHaModule {
        w1a: Parameter::new(w1a),
        w1b: Parameter::new(w1b),
        w2a: Parameter::new(w2a),
        w2b: Parameter::new(w2b),
        t1: Some(Parameter::new(t1)),
        t2: Some(Parameter::new(t2)),
        rank: r,
        alpha: r as f32, // scale=1.0
        device: device.clone(),
        is_conv: true,
    };

    let dw = m.get_diff_weight().expect("Tucker LoHa delta (P0-2 fix)");
    assert_eq!(dw.dims(), &[kh, kw, inn, out], "Tucker LoHa shape");
    let v = dw.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    for (i, &x) in v.iter().enumerate() {
        assert!(x.is_finite(), "non-finite at idx {}: {}", i, x);
    }
    // All elements should equal branch1*branch2 = 4*2 = 8.
    assert_close(v[0], 8.0, "Tucker LoHa ΔW[0,0,0,0]");
    let mid_idx = ((1 * kw + 0) * inn + 1) * out + 2;
    assert_close(v[mid_idx], 8.0, "Tucker LoHa ΔW[1,0,1,2]");
    assert_close(v[v.len() - 1], 8.0, "Tucker LoHa ΔW[last]");
}
