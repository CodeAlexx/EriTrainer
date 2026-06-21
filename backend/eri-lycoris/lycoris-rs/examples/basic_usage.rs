//! Basic usage example for LyCORIS-RS
//!
//! Demonstrates constructing LoCon / LoHa modules in memory (no file IO) and
//! printing the resulting ΔW shape. LoKr and Full construction also live in
//! the library — see `tests/smoke.rs` for a full exercise of all four.

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use lycoris_rs::algorithms::{LoConModule, LoHaModule};
use lycoris_rs::{LycorisModule, Result};

fn main() -> Result<()> {
    println!("LyCORIS-RS Basic Usage Example\n");

    // Initialize CUDA device
    let device: Arc<CudaDevice> =
        CudaDevice::new(0).expect("CUDA device required");
    println!("Using device: CUDA:0");

    // Example 1: LoRA (LoCon) for Linear Layer
    println!("\n1. Creating LoRA (LoCon) module for linear layer...");
    let locon = LoConModule::new_linear(
        512,       // in_features
        512,       // out_features
        8,         // rank
        Some(8.0), // alpha
        device.clone(),
    )?;
    println!(
        "   Created LoConModule: rank={}, alpha={}",
        locon.rank, locon.alpha
    );
    let delta_w_locon = locon.get_diff_weight()?;
    println!("   ΔW shape: {:?}", delta_w_locon.dims());

    // Example 2: LoHa (Hadamard) for Linear Layer
    println!("\n2. Creating LoHa module for linear layer...");
    let loha = LoHaModule::new_linear(512, 512, 8, Some(8.0), device.clone())?;
    println!(
        "   Created LoHaModule: rank={}, alpha={}",
        loha.rank, loha.alpha
    );
    let delta_w_loha = loha.get_diff_weight()?;
    println!("   ΔW shape: {:?}", delta_w_loha.dims());

    println!("\n✓ All examples completed successfully!");
    Ok(())
}
