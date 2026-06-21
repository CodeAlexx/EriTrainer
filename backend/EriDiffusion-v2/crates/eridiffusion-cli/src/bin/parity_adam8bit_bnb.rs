//! bitsandbytes 0.49.2 byte-parity test for `flame_core::adam8bit_kernel::adam8bit_step_bnb`.
//!
//! Loads the BEFORE snapshot dumped by `tests/parity/adam8bit_bnb_python_ref.py`,
//! pushes the same param/grad/state/absmax/qmap onto GPU, runs ONE step
//! through our Rust kernel, then diffs against the AFTER snapshot from the
//! same Python script.
//!
//! Gates (PASS = all true):
//!   - LUT eq:        max |qmap_rs - qmap_py|  < 1e-7 (sanity, must match exactly)
//!   - param:         max |Δ|  < 1e-5     (F32 identity arithmetic)
//!   - state1 (m):    EXACT EQUAL u8 codes (nearest-LUT-entry must agree)
//!   - state2 (v):    EXACT EQUAL u8 codes
//!   - absmax1 (m):   max |Δ| < 1e-6      (single block reduction, deterministic)
//!   - absmax2 (v):   max |Δ| < 1e-6
//!
//! If state codes differ: prints a histogram of LUT-index distance — an
//! off-by-one is acceptable only if the bnb kernel uses a different
//! tiebreak; off-by-more is a real bug.
//!
//! Exit codes:
//!   0 = PASS, 1 = FAIL, 2 = BLOCKED (missing data / CUDA / etc).

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use flame_core::adam8bit_kernel::{
    adam8bit_step_bnb, alloc_state, create_dynamic_map, upload_qmap, ADAM8BIT_BLOCK_SIZE,
};
use flame_core::{DType, Shape, Tensor};
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;

const DATA_DIR: &str =
    "/home/alex/EriDiffusion/EriDiffusion-v2/tests/parity/adam8bit_data";

const LUT_TOL: f32 = 1e-7;
const PARAM_TOL: f32 = 1e-5;
const ABSMAX_TOL: f32 = 1e-6;

// -----------------------------------------------------------------------------
// safetensors helpers (U8 + F32 ; tiny tensors, no mmap needed)
// -----------------------------------------------------------------------------

fn load_st(path: &str) -> anyhow::Result<Vec<u8>> {
    Ok(std::fs::read(path)?)
}

fn view<'a>(st: &'a SafeTensors<'a>, key: &str) -> anyhow::Result<TensorView<'a>> {
    st.tensor(key)
        .map_err(|e| anyhow::anyhow!("safetensors key '{key}' missing: {e}"))
}

fn read_f32(st: &SafeTensors, key: &str) -> anyhow::Result<Vec<f32>> {
    let t = view(st, key)?;
    if t.dtype() != safetensors::Dtype::F32 {
        anyhow::bail!("'{key}' expected F32, got {:?}", t.dtype());
    }
    let bytes = t.data();
    if bytes.len() % 4 != 0 {
        anyhow::bail!("'{key}' f32 byte len {} not divisible by 4", bytes.len());
    }
    let n = bytes.len() / 4;
    let mut v = Vec::<f32>::with_capacity(n);
    for i in 0..n {
        let b: [u8; 4] = bytes[i * 4..i * 4 + 4].try_into().unwrap();
        v.push(f32::from_le_bytes(b));
    }
    Ok(v)
}

fn read_u8(st: &SafeTensors, key: &str) -> anyhow::Result<Vec<u8>> {
    let t = view(st, key)?;
    if t.dtype() != safetensors::Dtype::U8 {
        anyhow::bail!("'{key}' expected U8, got {:?}", t.dtype());
    }
    Ok(t.data().to_vec())
}

fn shape_of(st: &SafeTensors, key: &str) -> anyhow::Result<Vec<usize>> {
    Ok(view(st, key)?.shape().to_vec())
}

// -----------------------------------------------------------------------------
// Diff helpers
// -----------------------------------------------------------------------------

struct F32Diff {
    max_abs: f32,
    mean_abs: f32,
    argmax_idx: usize,
    rs_at: f32,
    py_at: f32,
}

fn diff_f32(rs: &[f32], py: &[f32]) -> F32Diff {
    assert_eq!(rs.len(), py.len());
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut argmax_idx = 0usize;
    for i in 0..rs.len() {
        let d = (rs[i] - py[i]).abs();
        sum_abs += d as f64;
        if d > max_abs {
            max_abs = d;
            argmax_idx = i;
        }
    }
    F32Diff {
        max_abs,
        mean_abs: (sum_abs / rs.len().max(1) as f64) as f32,
        argmax_idx,
        rs_at: rs[argmax_idx],
        py_at: py[argmax_idx],
    }
}

struct U8Diff {
    mismatch: usize,
    /// Buckets: how many code-pairs differ by |Δ| in each range.
    bucket_1: usize,    // |Δ| == 1
    bucket_2_4: usize,  // |Δ| 2..=4
    bucket_5_16: usize, // |Δ| 5..=16
    bucket_gt16: usize, // |Δ| > 16
    /// First few sample mismatches (index, rs_code, py_code).
    samples: Vec<(usize, u8, u8)>,
}

fn diff_u8(rs: &[u8], py: &[u8]) -> U8Diff {
    assert_eq!(rs.len(), py.len());
    let mut d = U8Diff {
        mismatch: 0,
        bucket_1: 0,
        bucket_2_4: 0,
        bucket_5_16: 0,
        bucket_gt16: 0,
        samples: Vec::new(),
    };
    for i in 0..rs.len() {
        if rs[i] != py[i] {
            d.mismatch += 1;
            let delta = (rs[i] as i32 - py[i] as i32).unsigned_abs();
            match delta {
                1 => d.bucket_1 += 1,
                2..=4 => d.bucket_2_4 += 1,
                5..=16 => d.bucket_5_16 += 1,
                _ => d.bucket_gt16 += 1,
            }
            if d.samples.len() < 8 {
                d.samples.push((i, rs[i], py[i]));
            }
        }
    }
    d
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

fn run() -> anyhow::Result<bool> {
    println!("== adam8bit_step_bnb vs bitsandbytes 0.49.2 byte-parity ==\n");

    // ---- 0. Hyperparams ----
    let hp_path = format!("{DATA_DIR}/hyperparams.json");
    let hp: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&hp_path)?)?;
    let lr = hp["lr"].as_f64().unwrap() as f32;
    let beta1 = hp["beta1"].as_f64().unwrap() as f32;
    let beta2 = hp["beta2"].as_f64().unwrap() as f32;
    let eps = hp["eps"].as_f64().unwrap() as f32;
    let wd = hp["wd"].as_f64().unwrap() as f32;
    let step = hp["step"].as_i64().unwrap() as i32;
    let blocksize = hp["blocksize"].as_i64().unwrap() as usize;
    let numel = hp["numel"].as_i64().unwrap() as usize;
    assert_eq!(blocksize, ADAM8BIT_BLOCK_SIZE);
    let bc1 = 1.0f32 - beta1.powi(step);
    let bc2 = 1.0f32 - beta2.powi(step);
    let n_blocks = numel.div_ceil(blocksize);
    println!(
        "hyperparams: lr={lr} beta1={beta1} beta2={beta2} eps={eps} wd={wd} step={step} numel={numel} blocksize={blocksize}\n              bc1={bc1:.6} bc2={bc2:.6}"
    );

    // ---- 1. Load BEFORE + AFTER snapshots ----
    let before_buf = load_st(&format!("{DATA_DIR}/before.safetensors"))?;
    let after_buf = load_st(&format!("{DATA_DIR}/after.safetensors"))?;
    let before = SafeTensors::deserialize(&before_buf)?;
    let after = SafeTensors::deserialize(&after_buf)?;

    let param_before = read_f32(&before, "before.param")?;
    let grad_before = read_f32(&before, "before.grad")?;
    let state1_before = read_u8(&before, "before.state1")?;
    let state2_before = read_u8(&before, "before.state2")?;
    let absmax1_before = read_f32(&before, "before.absmax1")?;
    let absmax2_before = read_f32(&before, "before.absmax2")?;
    let qmap1_py = read_f32(&before, "before.qmap1")?;
    let qmap2_py = read_f32(&before, "before.qmap2")?;
    assert_eq!(param_before.len(), numel);
    assert_eq!(state1_before.len(), numel);
    assert_eq!(absmax1_before.len(), n_blocks);
    assert_eq!(qmap1_py.len(), 256);
    assert_eq!(shape_of(&before, "before.state1")?, vec![numel]);

    let param_after_py = read_f32(&after, "after.param")?;
    let state1_after_py = read_u8(&after, "after.state1")?;
    let state2_after_py = read_u8(&after, "after.state2")?;
    let absmax1_after_py = read_f32(&after, "after.absmax1")?;
    let absmax2_after_py = read_f32(&after, "after.absmax2")?;

    // ---- 2. Verify our create_dynamic_map matches bnb's qmap exactly ----
    let qmap_signed_rs = create_dynamic_map(true);
    let qmap_unsigned_rs = create_dynamic_map(false);
    let lut1_diff = diff_f32(&qmap_signed_rs, &qmap1_py);
    let lut2_diff = diff_f32(&qmap_unsigned_rs, &qmap2_py);
    println!(
        "LUT signed  : max |Δ| = {:.3e}  (idx={}, rs={:+.6e}, py={:+.6e})",
        lut1_diff.max_abs, lut1_diff.argmax_idx, lut1_diff.rs_at, lut1_diff.py_at
    );
    println!(
        "LUT unsigned: max |Δ| = {:.3e}  (idx={}, rs={:+.6e}, py={:+.6e})",
        lut2_diff.max_abs, lut2_diff.argmax_idx, lut2_diff.rs_at, lut2_diff.py_at
    );
    let lut_pass = lut1_diff.max_abs < LUT_TOL && lut2_diff.max_abs < LUT_TOL;
    if !lut_pass {
        println!("  ^^ LUT mismatch — abort, no point running the kernel.");
        // Print first 8 diverging entries.
        for i in 0..256 {
            let d = (qmap_signed_rs[i] - qmap1_py[i]).abs();
            if d > LUT_TOL {
                println!(
                    "   signed[{i}]: rs={:+.8e}  py={:+.8e}  Δ={:.3e}",
                    qmap_signed_rs[i], qmap1_py[i], d
                );
            }
        }
        return Ok(false);
    }

    // ---- 3. Move BEFORE state onto GPU and run kernel ----
    let device: Arc<CudaDevice> = CudaDevice::new(0)
        .map_err(|e| anyhow::anyhow!("CudaDevice::new(0) failed: {e:?}"))?;
    println!("\ncuda device: ordinal {}", device.ordinal());

    // Build Tensor for param and grad (F32, contiguous, on device).
    let mut param_tensor = Tensor::from_vec(
        param_before.clone(),
        Shape::from_dims(&[numel]),
        device.clone(),
    )?;
    let grad_tensor = Tensor::from_vec(
        grad_before.clone(),
        Shape::from_dims(&[numel]),
        device.clone(),
    )?;

    // Allocate the state slabs zero-init through alloc_state (matches Python
    // init), then overwrite with the BEFORE bytes (which for this step-0
    // snapshot are also all zeros — but the harness will work for multi-step
    // followups too).
    let (mut m_codes, mut v_codes, mut m_absmax, mut v_absmax) =
        alloc_state(&device, numel).map_err(|e| anyhow::anyhow!("alloc_state: {e:?}"))?;

    upload_into_u8(&device, &state1_before, &mut m_codes)?;
    upload_into_u8(&device, &state2_before, &mut v_codes)?;
    upload_into_f32(&device, &absmax1_before, &mut m_absmax)?;
    upload_into_f32(&device, &absmax2_before, &mut v_absmax)?;

    // Upload qmaps (using *our* host LUT — already proven equal to bnb's).
    let qmap_signed = upload_qmap(&device, &qmap_signed_rs)
        .map_err(|e| anyhow::anyhow!("upload qmap signed: {e:?}"))?;
    let qmap_unsigned = upload_qmap(&device, &qmap_unsigned_rs)
        .map_err(|e| anyhow::anyhow!("upload qmap unsigned: {e:?}"))?;

    // ---- 4. RUN ONE STEP ----
    adam8bit_step_bnb(
        &mut param_tensor,
        &grad_tensor,
        &mut m_codes,
        &mut v_codes,
        &mut m_absmax,
        &mut v_absmax,
        &qmap_signed,
        &qmap_unsigned,
        lr,
        beta1,
        beta2,
        eps,
        wd,
        bc1,
        bc2,
    )
    .map_err(|e| anyhow::anyhow!("adam8bit_step_bnb: {e:?}"))?;
    device
        .synchronize()
        .map_err(|e| anyhow::anyhow!("sync: {e:?}"))?;

    // ---- 5. D2H readback ----
    let param_after_rs = {
        // Param tensor is F32; pull via as_slice_f32 → sync_d2h via the tensor API.
        // Simplest cross-version-stable way: re-read via the same Tensor's
        // to_vec_f32() helper.
        param_tensor
            .to_vec_f32()
            .map_err(|e| anyhow::anyhow!("param to_vec_f32: {e:?}"))?
    };
    let state1_after_rs = device
        .dtoh_sync_copy(&m_codes)
        .map_err(|e| anyhow::anyhow!("d2h m_codes: {e:?}"))?;
    let state2_after_rs = device
        .dtoh_sync_copy(&v_codes)
        .map_err(|e| anyhow::anyhow!("d2h v_codes: {e:?}"))?;
    let absmax1_after_rs = device
        .dtoh_sync_copy(&m_absmax)
        .map_err(|e| anyhow::anyhow!("d2h m_absmax: {e:?}"))?;
    let absmax2_after_rs = device
        .dtoh_sync_copy(&v_absmax)
        .map_err(|e| anyhow::anyhow!("d2h v_absmax: {e:?}"))?;

    // ---- 6. DIFF ----
    println!("\n--- DIFF (rust vs python) ---");
    let pd = diff_f32(&param_after_rs, &param_after_py);
    println!(
        "param  : max |Δ| = {:.3e}  mean = {:.3e}  (idx={}, rs={:+.7}, py={:+.7})",
        pd.max_abs, pd.mean_abs, pd.argmax_idx, pd.rs_at, pd.py_at
    );
    let s1 = diff_u8(&state1_after_rs, &state1_after_py);
    println!(
        "state1 : mismatch = {}/{}  buckets |Δ|: 1->{}  2-4->{}  5-16->{}  >16->{}",
        s1.mismatch, numel, s1.bucket_1, s1.bucket_2_4, s1.bucket_5_16, s1.bucket_gt16
    );
    if !s1.samples.is_empty() {
        print!("  state1 samples (idx, rs_code, py_code):");
        for (i, r, p) in &s1.samples {
            print!(" ({i},{r},{p})");
        }
        println!();
    }
    let s2 = diff_u8(&state2_after_rs, &state2_after_py);
    println!(
        "state2 : mismatch = {}/{}  buckets |Δ|: 1->{}  2-4->{}  5-16->{}  >16->{}",
        s2.mismatch, numel, s2.bucket_1, s2.bucket_2_4, s2.bucket_5_16, s2.bucket_gt16
    );
    if !s2.samples.is_empty() {
        print!("  state2 samples (idx, rs_code, py_code):");
        for (i, r, p) in &s2.samples {
            print!(" ({i},{r},{p})");
        }
        println!();
    }
    let a1 = diff_f32(&absmax1_after_rs, &absmax1_after_py);
    println!(
        "absmax1: max |Δ| = {:.3e}  mean = {:.3e}  (idx={}, rs={:+.7e}, py={:+.7e})",
        a1.max_abs, a1.mean_abs, a1.argmax_idx, a1.rs_at, a1.py_at
    );
    let a2 = diff_f32(&absmax2_after_rs, &absmax2_after_py);
    println!(
        "absmax2: max |Δ| = {:.3e}  mean = {:.3e}  (idx={}, rs={:+.7e}, py={:+.7e})",
        a2.max_abs, a2.mean_abs, a2.argmax_idx, a2.rs_at, a2.py_at
    );

    println!("\n--- GATES ---");
    let g_lut = lut_pass;
    let g_param = pd.max_abs < PARAM_TOL;
    let g_state1 = s1.mismatch == 0;
    let g_state2 = s2.mismatch == 0;
    let g_absmax1 = a1.max_abs < ABSMAX_TOL;
    let g_absmax2 = a2.max_abs < ABSMAX_TOL;
    println!(
        "LUT       : {}   (signed Δ {:.3e}, unsigned Δ {:.3e}; tol {LUT_TOL:.0e})",
        gate(g_lut),
        lut1_diff.max_abs,
        lut2_diff.max_abs
    );
    println!(
        "param     : {}   (max |Δ| {:.3e}; tol {PARAM_TOL:.0e})",
        gate(g_param),
        pd.max_abs
    );
    println!(
        "state1    : {}   (mismatch {}/{}; EXACT EQ required)",
        gate(g_state1),
        s1.mismatch,
        numel
    );
    println!(
        "state2    : {}   (mismatch {}/{}; EXACT EQ required)",
        gate(g_state2),
        s2.mismatch,
        numel
    );
    println!(
        "absmax1   : {}   (max |Δ| {:.3e}; tol {ABSMAX_TOL:.0e})",
        gate(g_absmax1),
        a1.max_abs
    );
    println!(
        "absmax2   : {}   (max |Δ| {:.3e}; tol {ABSMAX_TOL:.0e})",
        gate(g_absmax2),
        a2.max_abs
    );

    let all = g_lut && g_param && g_state1 && g_state2 && g_absmax1 && g_absmax2;
    println!(
        "\n=== {} ===",
        if all { "PASS" } else { "FAIL" }
    );
    Ok(all)
}

fn gate(b: bool) -> &'static str {
    if b {
        "PASS"
    } else {
        "FAIL"
    }
}

fn upload_into_u8(
    device: &Arc<CudaDevice>,
    host: &[u8],
    dst: &mut CudaSlice<u8>,
) -> anyhow::Result<()> {
    if host.len() > dst.len() {
        anyhow::bail!(
            "u8 upload: host {} > dst {}",
            host.len(),
            dst.len()
        );
    }
    // alloc_state may over-allocate via the pool; if host is shorter, pad with 0.
    let mut padded = host.to_vec();
    if padded.len() < dst.len() {
        padded.resize(dst.len(), 0u8);
    }
    device
        .htod_copy_into(padded, dst)
        .map_err(|e| anyhow::anyhow!("htod_copy_into u8: {e:?}"))?;
    Ok(())
}

fn upload_into_f32(
    device: &Arc<CudaDevice>,
    host: &[f32],
    dst: &mut CudaSlice<f32>,
) -> anyhow::Result<()> {
    if host.len() > dst.len() {
        anyhow::bail!(
            "f32 upload: host {} > dst {}",
            host.len(),
            dst.len()
        );
    }
    let mut padded = host.to_vec();
    if padded.len() < dst.len() {
        padded.resize(dst.len(), 0.0f32);
    }
    device
        .htod_copy_into(padded, dst)
        .map_err(|e| anyhow::anyhow!("htod_copy_into f32: {e:?}"))?;
    Ok(())
}

// Suppress unused-import warning on `HashMap`/`PathBuf` if not used after refactor.
#[allow(dead_code)]
fn _touch(_: HashMap<String, Tensor>, _: PathBuf, _: DType) {}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::from(0),
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("BLOCKED: {e:#}");
            ExitCode::from(2)
        }
    }
}
