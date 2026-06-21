//! bitsandbytes 0.49.2 byte-parity, MULTI-STEP variant.
//!
//! For each step i in 1..=N_STEPS:
//!   1. Load `step_i_before.safetensors` (state going INTO that step).
//!   2. Run our `adam8bit_step_bnb` once.
//!   3. Diff against `step_i_after.safetensors`.
//!
//! Tolerance is per-step:
//!   - step 1 : strict (param 1e-5, codes EXACT, absmax 1e-6) — same as baseline
//!   - steps 2..=10 : loose (param 5e-5, codes ≤ 2 mismatches, absmax 1e-5)
//!     accommodates LUT-tiebreak drift accumulating over moments.
//!
//! HARD STOP: param Δ > 1e-3 or code mismatch > 10 anywhere — abort and report
//! which step + what changed.
//!
//! Exit codes: 0 PASS, 1 FAIL, 2 BLOCKED.

use std::process::ExitCode;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use flame_core::adam8bit_kernel::{
    adam8bit_step_bnb, alloc_state, create_dynamic_map, upload_qmap, ADAM8BIT_BLOCK_SIZE,
};
use flame_core::{Shape, Tensor};
use safetensors::SafeTensors;

const DATA_DIR: &str =
    "/home/alex/EriDiffusion/EriDiffusion-v2/tests/parity/adam8bit_data/multistep";

const PARAM_TOL_STRICT: f32 = 1e-5;
const PARAM_TOL_LOOSE: f32 = 5e-5;
const ABSMAX_TOL_STRICT: f32 = 1e-6;
const ABSMAX_TOL_LOOSE: f32 = 1e-5;
const CODE_MISMATCH_LOOSE: usize = 2;

const PARAM_HARD_STOP: f32 = 1e-3;
const CODE_HARD_STOP: usize = 10;

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
    let t = st.tensor(key)?;
    Ok(t.data().to_vec())
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
    let mut c = 0;
    for i in 0..rs.len() {
        if rs[i] != py[i] {
            c += 1;
        }
    }
    c
}

fn upload_u8(device: &Arc<CudaDevice>, host: &[u8], dst: &mut CudaSlice<u8>) -> anyhow::Result<()> {
    let mut padded = host.to_vec();
    if padded.len() < dst.len() {
        padded.resize(dst.len(), 0u8);
    }
    device
        .htod_copy_into(padded, dst)
        .map_err(|e| anyhow::anyhow!("htod u8: {e:?}"))?;
    Ok(())
}

fn upload_f32(device: &Arc<CudaDevice>, host: &[f32], dst: &mut CudaSlice<f32>) -> anyhow::Result<()> {
    let mut padded = host.to_vec();
    if padded.len() < dst.len() {
        padded.resize(dst.len(), 0.0);
    }
    device
        .htod_copy_into(padded, dst)
        .map_err(|e| anyhow::anyhow!("htod f32: {e:?}"))?;
    Ok(())
}

fn run() -> anyhow::Result<bool> {
    println!("== adam8bit_step_bnb MULTI-STEP parity vs bitsandbytes 0.49.2 ==\n");

    let hp: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{DATA_DIR}/hyperparams.json"))?)?;
    let lr = hp["lr"].as_f64().unwrap() as f32;
    let beta1 = hp["beta1"].as_f64().unwrap() as f32;
    let beta2 = hp["beta2"].as_f64().unwrap() as f32;
    let eps = hp["eps"].as_f64().unwrap() as f32;
    let wd = hp["wd"].as_f64().unwrap() as f32;
    let blocksize = hp["blocksize"].as_i64().unwrap() as usize;
    let numel = hp["numel"].as_i64().unwrap() as usize;
    let n_steps = hp["n_steps"].as_i64().unwrap() as i32;
    assert_eq!(blocksize, ADAM8BIT_BLOCK_SIZE);
    let n_blocks = numel.div_ceil(blocksize);
    println!(
        "lr={lr} beta1={beta1} beta2={beta2} eps={eps} wd={wd} numel={numel} blocksize={blocksize} n_steps={n_steps}\n"
    );

    // qmaps shared across steps
    let qmaps_buf = std::fs::read(format!("{DATA_DIR}/qmaps.safetensors"))?;
    let qmaps = SafeTensors::deserialize(&qmaps_buf)?;
    let qmap1_py = read_f32(&qmaps, "qmap1")?;
    let qmap2_py = read_f32(&qmaps, "qmap2")?;
    let qmap_signed_rs = create_dynamic_map(true);
    let qmap_unsigned_rs = create_dynamic_map(false);
    // sanity: LUTs match
    let (lut1d, _) = diff_f32_max(&qmap_signed_rs, &qmap1_py);
    let (lut2d, _) = diff_f32_max(&qmap_unsigned_rs, &qmap2_py);
    assert!(lut1d < 1e-7 && lut2d < 1e-7, "LUT mismatch (signed Δ {lut1d}, unsigned Δ {lut2d})");

    let device: Arc<CudaDevice> = CudaDevice::new(0)
        .map_err(|e| anyhow::anyhow!("CudaDevice::new(0): {e:?}"))?;
    let qmap_signed_dev = upload_qmap(&device, &qmap_signed_rs).map_err(|e| anyhow::anyhow!("upload qmap_s: {e:?}"))?;
    let qmap_unsigned_dev = upload_qmap(&device, &qmap_unsigned_rs).map_err(|e| anyhow::anyhow!("upload qmap_u: {e:?}"))?;

    println!("step | param Δ      | code1 mism | code2 mism | absmax1 Δ   | absmax2 Δ   | gate");
    println!("-----+--------------+------------+------------+-------------+-------------+-----");

    let mut all_pass = true;
    for step in 1..=n_steps {
        let before_buf = std::fs::read(format!("{DATA_DIR}/step_{step}_before.safetensors"))?;
        let after_buf = std::fs::read(format!("{DATA_DIR}/step_{step}_after.safetensors"))?;
        let before = SafeTensors::deserialize(&before_buf)?;
        let after = SafeTensors::deserialize(&after_buf)?;

        let param_before = read_f32(&before, "before.param")?;
        let grad_before = read_f32(&before, "before.grad")?;
        let state1_before = read_u8(&before, "before.state1")?;
        let state2_before = read_u8(&before, "before.state2")?;
        let absmax1_before = read_f32(&before, "before.absmax1")?;
        let absmax2_before = read_f32(&before, "before.absmax2")?;

        let param_after_py = read_f32(&after, "after.param")?;
        let state1_after_py = read_u8(&after, "after.state1")?;
        let state2_after_py = read_u8(&after, "after.state2")?;
        let absmax1_after_py = read_f32(&after, "after.absmax1")?;
        let absmax2_after_py = read_f32(&after, "after.absmax2")?;

        let mut param_tensor = Tensor::from_vec(param_before.clone(), Shape::from_dims(&[numel]), device.clone())?;
        let grad_tensor = Tensor::from_vec(grad_before.clone(), Shape::from_dims(&[numel]), device.clone())?;
        let (mut m_codes, mut v_codes, mut m_absmax, mut v_absmax) =
            alloc_state(&device, numel).map_err(|e| anyhow::anyhow!("alloc_state: {e:?}"))?;
        upload_u8(&device, &state1_before, &mut m_codes)?;
        upload_u8(&device, &state2_before, &mut v_codes)?;
        upload_f32(&device, &absmax1_before, &mut m_absmax)?;
        upload_f32(&device, &absmax2_before, &mut v_absmax)?;

        let bc1 = 1.0 - beta1.powi(step);
        let bc2 = 1.0 - beta2.powi(step);
        adam8bit_step_bnb(
            &mut param_tensor, &grad_tensor,
            &mut m_codes, &mut v_codes,
            &mut m_absmax, &mut v_absmax,
            &qmap_signed_dev, &qmap_unsigned_dev,
            lr, beta1, beta2, eps, wd, bc1, bc2,
        ).map_err(|e| anyhow::anyhow!("kernel step {step}: {e:?}"))?;
        device.synchronize().map_err(|e| anyhow::anyhow!("sync: {e:?}"))?;

        let param_rs = param_tensor.to_vec_f32().map_err(|e| anyhow::anyhow!("readback: {e:?}"))?;
        let state1_rs = device.dtoh_sync_copy(&m_codes).map_err(|e| anyhow::anyhow!("d2h s1: {e:?}"))?;
        let state2_rs = device.dtoh_sync_copy(&v_codes).map_err(|e| anyhow::anyhow!("d2h s2: {e:?}"))?;
        let absmax1_rs = device.dtoh_sync_copy(&m_absmax).map_err(|e| anyhow::anyhow!("d2h a1: {e:?}"))?;
        let absmax2_rs = device.dtoh_sync_copy(&v_absmax).map_err(|e| anyhow::anyhow!("d2h a2: {e:?}"))?;

        // Trim to numel (alloc_state may over-allocate via pool).
        let state1_rs = &state1_rs[..numel];
        let state2_rs = &state2_rs[..numel];
        let absmax1_rs = &absmax1_rs[..n_blocks];
        let absmax2_rs = &absmax2_rs[..n_blocks];

        let (pd, pidx) = diff_f32_max(&param_rs, &param_after_py);
        let s1 = diff_u8_count(state1_rs, &state1_after_py);
        let s2 = diff_u8_count(state2_rs, &state2_after_py);
        let (a1, _) = diff_f32_max(absmax1_rs, &absmax1_after_py);
        let (a2, _) = diff_f32_max(absmax2_rs, &absmax2_after_py);

        // HARD STOP
        if pd > PARAM_HARD_STOP || s1 > CODE_HARD_STOP || s2 > CODE_HARD_STOP {
            println!(
                "{step:>4} | {pd:.3e}   | {s1:>10} | {s2:>10} | {a1:.3e}  | {a2:.3e}  | HARD-STOP"
            );
            println!(
                "\nHARD STOP @ step {step}: param Δ {pd:.3e} (idx {pidx} rs={:+.6} py={:+.6}) | codes1 {s1} | codes2 {s2}",
                param_rs[pidx], param_after_py[pidx]
            );
            return Ok(false);
        }

        let (param_tol, absmax_tol, code_tol) = if step == 1 {
            (PARAM_TOL_STRICT, ABSMAX_TOL_STRICT, 0usize)
        } else {
            (PARAM_TOL_LOOSE, ABSMAX_TOL_LOOSE, CODE_MISMATCH_LOOSE)
        };
        let pass = pd < param_tol && s1 <= code_tol && s2 <= code_tol && a1 < absmax_tol && a2 < absmax_tol;
        if !pass {
            all_pass = false;
        }
        println!(
            "{step:>4} | {pd:.3e}   | {s1:>10} | {s2:>10} | {a1:.3e}  | {a2:.3e}  | {}",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    println!("\n=== MULTI-STEP {} ===", if all_pass { "PASS" } else { "FAIL" });
    Ok(all_pass)
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
