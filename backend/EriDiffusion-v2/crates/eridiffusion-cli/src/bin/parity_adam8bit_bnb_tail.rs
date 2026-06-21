//! bitsandbytes 0.49.2 byte-parity, TAIL-BLOCK variant (numel=2050).
//!
//! 9 blocks total: 8 full + last block has 2 active lanes + 254 inactive.
//! Exercises the kernel's inactive-lane guard against polluting the block
//! absmax reduction.
//!
//! Critical question: does the last block's absmax match Python? If yes,
//! inactive lanes correctly contribute 0 to the max-reduction. If no,
//! garbage from idx>=n is bleeding in.
//!
//! Gates: same strict baseline (param 1e-5, codes EXACT, absmax 1e-6).
//! Additionally: explicit per-block absmax dump for the tail block.

use std::process::ExitCode;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use flame_core::adam8bit_kernel::{
    adam8bit_step_bnb, alloc_state, create_dynamic_map, upload_qmap, ADAM8BIT_BLOCK_SIZE,
};
use flame_core::{Shape, Tensor};
use safetensors::SafeTensors;

const DATA_DIR: &str =
    "/home/alex/EriDiffusion/EriDiffusion-v2/tests/parity/adam8bit_data/tail";

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
    println!("== adam8bit_step_bnb TAIL-BLOCK (numel=2050) parity vs bnb 0.49.2 ==\n");

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
    let n_blocks = numel.div_ceil(blocksize);
    let tail_active = numel - (n_blocks - 1) * blocksize;
    let bc1 = 1.0 - beta1.powi(step);
    let bc2 = 1.0 - beta2.powi(step);
    println!("numel={numel} n_blocks={n_blocks} tail_active={tail_active} (inactive={})\n", blocksize - tail_active);

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
    let (a1, a1_idx) = diff_f32_max(absmax1_rs, &absmax1_after_py);
    let (a2, a2_idx) = diff_f32_max(absmax2_rs, &absmax2_after_py);

    println!("DIFF (rust vs python):");
    println!("  param   : max |Δ| = {pd:.3e}  (idx {pidx})");
    println!("  state1  : mismatch = {s1}/{numel}");
    println!("  state2  : mismatch = {s2}/{numel}");
    println!("  absmax1 : max |Δ| = {a1:.3e}  (block {a1_idx})");
    println!("  absmax2 : max |Δ| = {a2:.3e}  (block {a2_idx})");

    let tail_blk = n_blocks - 1;
    println!(
        "\nTail block ({tail_blk}, {tail_active} active lanes):\n  absmax1_rs={:.6e}  absmax1_py={:.6e}  Δ={:.3e}\n  absmax2_rs={:.6e}  absmax2_py={:.6e}  Δ={:.3e}",
        absmax1_rs[tail_blk], absmax1_after_py[tail_blk],
        (absmax1_rs[tail_blk] - absmax1_after_py[tail_blk]).abs(),
        absmax2_rs[tail_blk], absmax2_after_py[tail_blk],
        (absmax2_rs[tail_blk] - absmax2_after_py[tail_blk]).abs(),
    );

    let tail_absmax_match =
        (absmax1_rs[tail_blk] - absmax1_after_py[tail_blk]).abs() < ABSMAX_TOL
            && (absmax2_rs[tail_blk] - absmax2_after_py[tail_blk]).abs() < ABSMAX_TOL;
    println!(
        "Inactive-lane guard: {} (tail-block absmax {} bnb)",
        if tail_absmax_match { "OK" } else { "BROKEN" },
        if tail_absmax_match { "matches" } else { "DIVERGES from" }
    );

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
    println!("\n=== TAIL {} ===", if all {"PASS"} else {"FAIL"});
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
