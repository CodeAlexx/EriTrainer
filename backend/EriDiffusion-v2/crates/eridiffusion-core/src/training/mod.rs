pub mod block_offload;
pub mod board;
pub mod checkpoint;
pub mod ema;
pub mod features;
pub mod grad_coverage;
pub mod levers;
pub mod logging;
pub mod offload;
pub mod progress;
pub mod save_direct;
pub mod schedule;
pub mod training_features;
pub mod training_offload;

use flame_core::gradient_clip::GradientClipper;
use flame_core::{parameter::Parameter, DType, Tensor};

use crate::Result;


/// Gradient accumulation: copy grads from GradientMap to Parameters
pub fn accumulate_parameter_grads(
    params: &[Parameter],
    grads: &flame_core::gradient::GradientMap,
) -> Result<()> {
    for param in params {
        if let Some(g) = grads.get(param.id()) {
            let g = if g.dtype() == DType::F32 {
                g.clone()
            } else {
                g.to_dtype(DType::F32)?
            };
            param.set_grad(g)?;
        }
    }
    Ok(())
}

/// Clip parameter gradients by norm, return norm
pub fn clip_parameter_grads(params: &[Parameter], clipper: &GradientClipper) -> Result<f32> {
    let mut grads: Vec<Tensor> = Vec::new();
    let mut owners: Vec<usize> = Vec::new();
    for (idx, param) in params.iter().enumerate() {
        if let Some(g) = param.grad() {
            grads.push(g);
            owners.push(idx);
        }
    }
    if grads.is_empty() {
        return Ok(0.0);
    }
    let mut grad_refs: Vec<&mut Tensor> = grads.iter_mut().collect();
    let norm = clipper.clip_grads(&mut grad_refs)?;
    for (owner, grad) in owners.into_iter().zip(grads.into_iter()) {
        params[owner].set_grad(grad)?;
    }
    Ok(norm)
}
