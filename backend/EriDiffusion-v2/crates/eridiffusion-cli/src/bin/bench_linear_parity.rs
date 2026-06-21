//! Micro-bench: `fused_linear3d_native` vs `fused_linear3d_native_pytorch_parity`.
//!
//! Times both variants on the HiDream-O1 TimestepEmbedder shapes (256→4096
//! and 4096→4096) plus a typical decoder shape (4096→4096 with seq=497) so we
//! can decide whether the parity variant is fast enough to swap in.
//!
//! Run:
//!   cargo run --release --bin bench_linear_parity

use flame_core::ops::fused_inference::{
    fused_linear3d_native, fused_linear3d_native_pytorch_parity,
};
use flame_core::{DType, Shape, Tensor};
use std::sync::Arc;
use std::time::Instant;

fn bench_shape(label: &str, batch: usize, seq: usize, k: usize, m: usize, iters: usize) -> anyhow::Result<()> {
    let device = flame_core::global_cuda_device();

    // Build random BF16 tensors. Values don't matter for perf, only shapes.
    let input = Tensor::randn(
        Shape::from_dims(&[batch, seq, k]),
        0.0,
        1.0,
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let weight = Tensor::randn(
        Shape::from_dims(&[m, k]),
        0.0,
        0.02,
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let bias = Tensor::randn(
        Shape::from_dims(&[m]),
        0.0,
        0.01,
        device.clone(),
    )?
    .to_dtype(DType::BF16)?;

    // Warm both kernels (cache desc + algo lookup).
    for _ in 0..10 {
        let _ = fused_linear3d_native(&input, &weight, Some(&bias))?;
        let _ = fused_linear3d_native_pytorch_parity(&input, &weight, Some(&bias))?;
    }
    device.synchronize()?;

    // Time native.
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = fused_linear3d_native(&input, &weight, Some(&bias))?;
    }
    device.synchronize()?;
    let native_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // Time parity.
    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = fused_linear3d_native_pytorch_parity(&input, &weight, Some(&bias))?;
    }
    device.synchronize()?;
    let parity_us = t1.elapsed().as_micros() as f64 / iters as f64;

    let ratio = parity_us / native_us;
    println!(
        "{label:<32}  B={batch} N={seq} K={k} M={m}  native={native_us:>8.2}us  parity={parity_us:>8.2}us  ratio={ratio:.3}x"
    );

    Ok(())
}

fn main() -> anyhow::Result<()> {
    println!("fused_linear3d native vs pytorch_parity micro-bench");
    println!("====================================================");

    // TimestepEmbedder shapes (called once per forward pass; tiny tensors).
    bench_shape("TimestepEmbedder mlp_in",  1, 1,    256, 4096, 1000)?;
    bench_shape("TimestepEmbedder mlp_out", 1, 1,   4096, 4096, 1000)?;

    // Typical decoder shapes (called 36x per forward pass).
    bench_shape("Decoder q_proj",          1, 497, 4096, 4096,  500)?;
    bench_shape("Decoder kv_proj",         1, 497, 4096, 1024,  500)?;
    bench_shape("Decoder o_proj",          1, 497, 4096, 4096,  500)?;
    bench_shape("Decoder mlp.gate",        1, 497, 4096, 12288, 500)?;
    bench_shape("Decoder mlp.down",        1, 497, 12288, 4096, 500)?;

    Ok(())
}
