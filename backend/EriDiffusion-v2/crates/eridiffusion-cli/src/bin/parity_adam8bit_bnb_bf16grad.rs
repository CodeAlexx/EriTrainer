//! bitsandbytes 0.49.2 BF16-grad SELF-CONSISTENCY (no native BF16 in bnb).
//!
//! bnb's host dispatch upcasts BF16 grad via `g.float()` before the F32
//! kernel. So this is NOT a true bnb-vs-us comparison — it's:
//!
//!   Python: F32 raw grad -> BF16 (truncate) -> F32 (lossless) -> bnb F32 kernel
//!   Rust  : F32 raw grad -> BF16 device tensor -> our native BF16-grad kernel
//!
//! Both kernels see arithmetically equivalent grad values; the Rust path
//! does `__bfloat162float(bf16_byte)` inside the kernel, which equals the
//! F32 value Python computed via `.float()`. Result: identical math, so
//! results should match to ~BF16-precision noise.
//!
//! Tolerances:
//!   - param   : 1e-3   (BF16 grad has only ~3 decimal digits precision)
//!   - codes   : ≤ 5 mismatches per stream (LUT tiebreak shifts near boundaries)
//!   - absmax  : 1e-4
//!
//! Wider failure (param Δ > 5e-3 OR codes > 30) -> real BF16-path kernel bug.

use std::process::ExitCode;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use flame_core::adam8bit_kernel::{
    adam8bit_step_bnb, alloc_state, create_dynamic_map, upload_qmap, ADAM8BIT_BLOCK_SIZE,
};
use flame_core::{DType, Shape, Tensor};
use safetensors::SafeTensors;

const DATA_DIR: &str =
    "/home/alex/EriDiffusion/EriDiffusion-v2/tests/parity/adam8bit_data/bf16grad";

const PARAM_TOL_BF16: f32 = 1e-3;
const CODE_TOL_BF16: usize = 5;
const ABSMAX_TOL_BF16: f32 = 1e-4;

fn read_f32(st: &SafeTensors, key: &str) -> anyhow::Result<Vec<f32>> {
    let t = st.tensor(key)?;
    let bytes = t.data();
    let n = bytes.len() / 4;
    let mut v = Vec::<f32>::with_capacity(n);
    for i in 0..n {
        let b: [u8; 4] = bytes[i * 4..i * 4 + 4].try_into().unwrap();
        v.push(f32::from_le_bytes(b));
    }
    Ok(v)
}

fn read_u8(st: &SafeTensors, key: &str) -> anyhow::Result<Vec<u8>> {
    Ok(st.tensor(key)?.data().to_vec())
}

fn diff_f32_max(rs: &[f32], py: &[f32]) -> (f32, usize) {
    let mut m = 0.0f32;
    let mut a = 0usize;
    for i in 0..rs.len() {
        let d = (rs[i] - py[i]).abs();
        if d > m {
            m = d;
            a = i;
        }
    }
    (m, a)
}

fn diff_u8_count(rs: &[u8], py: &[u8]) -> usize {
    rs.iter().zip(py.iter()).filter(|(a, b)| a != b).count()
}

fn upload_u8(device: &Arc<CudaDevice>, host: &[u8], dst: &mut CudaSlice<u8>) -> anyhow::Result<()> {
    let mut padded = host.to_vec();
    if padded.len() < dst.len() {
        padded.resize(dst.len(), 0u8);
    }
    device.htod_copy_into(padded, dst).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    Ok(())
}

fn upload_f32(device: &Arc<CudaDevice>, host: &[f32], dst: &mut CudaSlice<f32>) -> anyhow::Result<()> {
    let mut padded = host.to_vec();
    if padded.len() < dst.len() {
        padded.resize(dst.len(), 0.0);
    }
    device.htod_copy_into(padded, dst).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    Ok(())
}

fn run() -> anyhow::Result<bool> {
    println!("== adam8bit_step_bnb BF16-grad SELF-CONSISTENCY vs bnb 0.49.2 ==\n");

    let hp: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{DATA_DIR}/hyperparams.json"))?)?;
    let lr = hp["lr"].as_f64().unwrap() as f32;
    let beta1 = hp["beta1"].as_f64().unwrap() as f32;
    let beta2 = hp["beta2"].as_f64().unwrap() as f32;
    let eps = hp["eps"].as_f64().unwrap() as f32;
    let wd = hp["wd"].as_f64().unwrap() as f32;
    let step = hp["step"].as_i64().unwrap() as i32;
    let blocksize = hp["blocksize"].as_i64().unwrap() as usize;
    let numel = hp["numel"].as_i64().unwrap() as usize;
    assert_eq!(blocksize, ADAM8BIT_BLOCK_SIZE);
    let bc1 = 1.0 - beta1.powi(step);
    let bc2 = 1.0 - beta2.powi(step);
    let n_blocks = numel.div_ceil(blocksize);
    println!("lr={lr} betas=({beta1},{beta2}) eps={eps} wd={wd} step={step} numel={numel}\n");

    let before_buf = std::fs::read(format!("{DATA_DIR}/before.safetensors"))?;
    let after_buf = std::fs::read(format!("{DATA_DIR}/after.safetensors"))?;
    let before = SafeTensors::deserialize(&before_buf)?;
    let after = SafeTensors::deserialize(&after_buf)?;

    let param_before = read_f32(&before, "before.param")?;
    // The F32 `grad_f32` is what bnb saw on the Python side (a BF16
    // round-trip of the raw grad). It contains values that are exactly
    // representable in BF16 (low 16 bits zero), so calling
    // `from_vec_dtype(BF16)` produces the same bit pattern Python's
    // `grad_bf16` holds.
    let grad_f32_bnb = read_f32(&before, "before.grad_f32")?;
    let state1_before = read_u8(&before, "before.state1")?;
    let state2_before = read_u8(&before, "before.state2")?;
    let absmax1_before = read_f32(&before, "before.absmax1")?;
    let absmax2_before = read_f32(&before, "before.absmax2")?;

    let param_after_py = read_f32(&after, "after.param")?;
    let state1_after_py = read_u8(&after, "after.state1")?;
    let state2_after_py = read_u8(&after, "after.state2")?;
    let absmax1_after_py = read_f32(&after, "after.absmax1")?;
    let absmax2_after_py = read_f32(&after, "after.absmax2")?;

    let qmap_signed_rs = create_dynamic_map(true);
    let qmap_unsigned_rs = create_dynamic_map(false);

    let device: Arc<CudaDevice> = CudaDevice::new(0).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let qmap_signed_dev = upload_qmap(&device, &qmap_signed_rs).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let qmap_unsigned_dev = upload_qmap(&device, &qmap_unsigned_rs).map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let mut param_tensor = Tensor::from_vec(param_before.clone(), Shape::from_dims(&[numel]), device.clone())?;
    // BF16 grad: from_vec_dtype upcasts via f32_to_bf16_u16 device kernel,
    // which truncates the low 16 bits — but our input already has those bits
    // zero (BF16 round-trip on Python side), so this produces the same
    // BF16 bytes as Python's `grad_bf16` tensor.
    let grad_tensor = Tensor::from_vec_dtype(
        grad_f32_bnb.clone(),
        Shape::from_dims(&[numel]),
        device.clone(),
        DType::BF16,
    )?;
    assert_eq!(grad_tensor.dtype(), DType::BF16);

    let (mut m_codes, mut v_codes, mut m_absmax, mut v_absmax) =
        alloc_state(&device, numel).map_err(|e| anyhow::anyhow!("alloc_state: {e:?}"))?;
    upload_u8(&device, &state1_before, &mut m_codes)?;
    upload_u8(&device, &state2_before, &mut v_codes)?;
    upload_f32(&device, &absmax1_before, &mut m_absmax)?;
    upload_f32(&device, &absmax2_before, &mut v_absmax)?;

    adam8bit_step_bnb(
        &mut param_tensor, &grad_tensor,
        &mut m_codes, &mut v_codes,
        &mut m_absmax, &mut v_absmax,
        &qmap_signed_dev, &qmap_unsigned_dev,
        lr, beta1, beta2, eps, wd, bc1, bc2,
    ).map_err(|e| anyhow::anyhow!("kernel: {e:?}"))?;
    device.synchronize().map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let param_rs = param_tensor.to_vec_f32().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let state1_rs = device.dtoh_sync_copy(&m_codes).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let state2_rs = device.dtoh_sync_copy(&v_codes).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let absmax1_rs = device.dtoh_sync_copy(&m_absmax).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let absmax2_rs = device.dtoh_sync_copy(&v_absmax).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let state1_rs = &state1_rs[..numel];
    let state2_rs = &state2_rs[..numel];
    let absmax1_rs = &absmax1_rs[..n_blocks];
    let absmax2_rs = &absmax2_rs[..n_blocks];

    let (pd, pidx) = diff_f32_max(&param_rs, &param_after_py);
    let s1 = diff_u8_count(state1_rs, &state1_after_py);
    let s2 = diff_u8_count(state2_rs, &state2_after_py);
    let (a1, _) = diff_f32_max(absmax1_rs, &absmax1_after_py);
    let (a2, _) = diff_f32_max(absmax2_rs, &absmax2_after_py);

    println!("DIFF (rust BF16-kernel vs python F32-kernel-on-BF16-roundtripped-grad):");
    println!("  param   : max |Δ| = {pd:.3e}  (idx {pidx})");
    println!("  state1  : mismatch = {s1}/{numel}");
    println!("  state2  : mismatch = {s2}/{numel}");
    println!("  absmax1 : max |Δ| = {a1:.3e}");
    println!("  absmax2 : max |Δ| = {a2:.3e}");

    let g_param = pd < PARAM_TOL_BF16;
    let g_s1 = s1 <= CODE_TOL_BF16;
    let g_s2 = s2 <= CODE_TOL_BF16;
    let g_a1 = a1 < ABSMAX_TOL_BF16;
    let g_a2 = a2 < ABSMAX_TOL_BF16;

    println!(
        "\n--- GATES (BF16-precision tolerances) ---\nparam   : {}  (tol {PARAM_TOL_BF16:.0e})\nstate1  : {}  (tol ≤{CODE_TOL_BF16})\nstate2  : {}  (tol ≤{CODE_TOL_BF16})\nabsmax1 : {}  (tol {ABSMAX_TOL_BF16:.0e})\nabsmax2 : {}  (tol {ABSMAX_TOL_BF16:.0e})",
        if g_param {"PASS"} else {"FAIL"},
        if g_s1 {"PASS"} else {"FAIL"},
        if g_s2 {"PASS"} else {"FAIL"},
        if g_a1 {"PASS"} else {"FAIL"},
        if g_a2 {"PASS"} else {"FAIL"},
    );
    // Interpretation hint
    if s1 + s2 > 0 && s1 <= CODE_TOL_BF16 && s2 <= CODE_TOL_BF16 {
        println!(
            "\nInterpretation: {} total code mismatches across both streams.\n  Likely legitimate BF16-precision LUT tiebreak drift (boundary cases\n  where requant normalization lands within 0.5 ULP of two adjacent qmap entries).",
            s1 + s2
        );
    } else if s1 == 0 && s2 == 0 {
        println!("\nInterpretation: byte-exact match — BF16 kernel arithmetic is bit-equivalent to F32-on-BF16-roundtripped path.");
    }

    let all = g_param && g_s1 && g_s2 && g_a1 && g_a2;
    println!("\n=== BF16-GRAD {} ===", if all {"PASS"} else {"FAIL"});
    Ok(all)
}

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
