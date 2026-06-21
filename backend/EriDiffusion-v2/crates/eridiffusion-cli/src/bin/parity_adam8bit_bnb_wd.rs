//! bitsandbytes 0.49.2 byte-parity, WEIGHT-DECAY variant (wd=1e-2).
//!
//! Same single-step shape as the baseline harness, but with wd=1e-2.
//! Pins down whether bnb's `optimizer_name="adam"` applies decoupled wd
//! (matches our kernel: `p -= lr*wd*p` after the adam update) or coupled wd.
//!
//! Gates are the strict ones from baseline (param 1e-5, codes EXACT,
//! absmax 1e-6). If diffs are clean -> bnb adam is decoupled. If param
//! diverges by ~lr*wd*p worth (~1e-5 magnitude) -> bnb is coupled, we are
//! decoupled, mismatch. Report loudly either way (no kernel patch).

use std::process::ExitCode;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use flame_core::adam8bit_kernel::{
    adam8bit_step_bnb, alloc_state, create_dynamic_map, upload_qmap, ADAM8BIT_BLOCK_SIZE,
};
use flame_core::{Shape, Tensor};
use safetensors::SafeTensors;

const DATA_DIR: &str =
    "/home/alex/EriDiffusion/EriDiffusion-v2/tests/parity/adam8bit_data/wd";

const PARAM_TOL: f32 = 1e-5;
const ABSMAX_TOL: f32 = 1e-6;

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
    device.htod_copy_into(padded, dst).map_err(|e| anyhow::anyhow!("htod u8: {e:?}"))?;
    Ok(())
}

fn upload_f32(device: &Arc<CudaDevice>, host: &[f32], dst: &mut CudaSlice<f32>) -> anyhow::Result<()> {
    let mut padded = host.to_vec();
    if padded.len() < dst.len() {
        padded.resize(dst.len(), 0.0);
    }
    device.htod_copy_into(padded, dst).map_err(|e| anyhow::anyhow!("htod f32: {e:?}"))?;
    Ok(())
}

fn run() -> anyhow::Result<bool> {
    println!("== adam8bit_step_bnb WD=1e-2 parity vs bitsandbytes 0.49.2 ==\n");

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
    assert!(wd > 0.0, "this harness requires wd > 0");
    let bc1 = 1.0 - beta1.powi(step);
    let bc2 = 1.0 - beta2.powi(step);
    let n_blocks = numel.div_ceil(blocksize);
    println!("lr={lr} betas=({beta1},{beta2}) eps={eps} wd={wd} step={step} numel={numel}\n");

    let before_buf = std::fs::read(format!("{DATA_DIR}/before.safetensors"))?;
    let after_buf = std::fs::read(format!("{DATA_DIR}/after.safetensors"))?;
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

    let param_after_py = read_f32(&after, "after.param")?;
    let state1_after_py = read_u8(&after, "after.state1")?;
    let state2_after_py = read_u8(&after, "after.state2")?;
    let absmax1_after_py = read_f32(&after, "after.absmax1")?;
    let absmax2_after_py = read_f32(&after, "after.absmax2")?;

    let qmap_signed_rs = create_dynamic_map(true);
    let qmap_unsigned_rs = create_dynamic_map(false);
    let (lut1d, _) = diff_f32_max(&qmap_signed_rs, &qmap1_py);
    let (lut2d, _) = diff_f32_max(&qmap_unsigned_rs, &qmap2_py);
    assert!(lut1d < 1e-7 && lut2d < 1e-7, "LUT mismatch");

    let device: Arc<CudaDevice> = CudaDevice::new(0).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let qmap_signed_dev = upload_qmap(&device, &qmap_signed_rs).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let qmap_unsigned_dev = upload_qmap(&device, &qmap_unsigned_rs).map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let mut param_tensor = Tensor::from_vec(param_before.clone(), Shape::from_dims(&[numel]), device.clone())?;
    let grad_tensor = Tensor::from_vec(grad_before.clone(), Shape::from_dims(&[numel]), device.clone())?;
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

    println!("DIFF (rust vs python):");
    println!("  param   : max |Δ| = {pd:.3e}  (idx {pidx} rs={:+.7} py={:+.7})",
        param_rs[pidx], param_after_py[pidx]);
    println!("  state1  : mismatch = {s1}/{numel}");
    println!("  state2  : mismatch = {s2}/{numel}");
    println!("  absmax1 : max |Δ| = {a1:.3e}");
    println!("  absmax2 : max |Δ| = {a2:.3e}");

    // Diagnostic: predict the magnitude of the wd-induced param shift if we
    // were COUPLED while bnb is DECOUPLED (or vice versa). Both modes give
    // |Δ| ~ lr*wd*|p| ~ 1e-3 * 1e-2 * 1.0 = 1e-5. So a diff in that range
    // is the wd-mode-mismatch signature; a diff << that means we agree.
    let expected_wd_mismatch_magnitude = lr * wd * 1.0_f32; // |p| ~ N(0,1)
    println!(
        "\nDiagnostic: if our kernel + bnb disagreed on coupled-vs-decoupled wd,\n             |param Δ| would be ~lr*wd*|p| = {:.3e}.",
        expected_wd_mismatch_magnitude
    );
    if pd > 1e-6 && pd < 5e-5 {
        println!("             observed Δ = {pd:.3e} sits in that range — suggests wd-mode mismatch.");
    } else if pd < PARAM_TOL {
        println!("             observed Δ = {pd:.3e} ≪ that → wd application AGREES (both decoupled).");
    }

    let g_param = pd < PARAM_TOL;
    let g_s1 = s1 == 0;
    let g_s2 = s2 == 0;
    let g_a1 = a1 < ABSMAX_TOL;
    let g_a2 = a2 < ABSMAX_TOL;
    println!(
        "\n--- GATES ---\nparam   : {}\nstate1  : {}\nstate2  : {}\nabsmax1 : {}\nabsmax2 : {}",
        if g_param {"PASS"} else {"FAIL"},
        if g_s1 {"PASS"} else {"FAIL"},
        if g_s2 {"PASS"} else {"FAIL"},
        if g_a1 {"PASS"} else {"FAIL"},
        if g_a2 {"PASS"} else {"FAIL"},
    );
    let all = g_param && g_s1 && g_s2 && g_a1 && g_a2;
    println!("\n=== WD {} ===", if all {"PASS"} else {"FAIL"});
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
