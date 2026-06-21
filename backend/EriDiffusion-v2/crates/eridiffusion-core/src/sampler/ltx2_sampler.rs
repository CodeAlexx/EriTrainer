//! LTX-2 rectified-flow Euler sampler with sequence-length-dependent
//! shift schedule. Mirrors `LTX2Scheduler` (musubi / official Lightricks trainer).
//!
//! ## Math (T2V):
//! - Forward noising at train time: `noisy = (1 - sigma) * clean + sigma * noise`
//! - Velocity target: `noise - clean`.
//! - At sample time: Euler step `x_next = x + (sigma_next - sigma) * pred`,
//!   identical to ERNIE/Z-Image schedulers.
//!
//! ## Shift schedule (inference)
//! The sigma curve for inference is "shifted" using the exponential transform:
//!   `mu = base_shift + (max_shift - base_shift) * (n_tokens - 1024) / (4096 - 1024)`
//!   `mu = mu.clamp(base_shift, max_shift)`
//!   `sigma_shifted = exp(mu) * sigma / (1 + (exp(mu) - 1) * sigma)`
//!
//! ## Training timestep sampling: shifted logit-normal (musubi "stretched" mode)
//! Official Lightricks formula (serenity::training::noise::sample_shifted_logit_normal):
//!   1. `shift = lininterp(seq_len, base_tokens=1024, max_tokens=4096, base_shift=0.95, max_shift=2.05)`
//!   2. `z ~ N(shift, std=1.0)`                 // shift used as the normal MEAN
//!   3. `sigma_raw = sigmoid(z)`
//!   4. Stretch between percentile bounds:
//!      `p_hi = sigmoid(shift + 3.0902)`, `p_lo = sigmoid(shift - 2.5758)`
//!      `stretched = (sigma_raw - p_lo) / (p_hi - p_lo)`
//!      Reflect: `stretched = 2*eps - stretched` where `stretched < eps`
//!      Clamp to [0, 1]
//!   5. Mix 10% uniform from Uniform(eps, 1): `if rand() < 0.1 { uniform } else { stretched }`
//!
//! This differs from the old formula which applied an exp(mu) rational transform
//! AFTER sigmoid(N(0,1)). See serenity/tests/test_ltx2_shifted_logit_normal.py
//! `TestComparisonWithOldMethod` for a proof that the distributions differ.

use flame_core::{Result, Tensor};

const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;
const BASE_SHIFT: f32 = 0.95;
const MAX_SHIFT: f32 = 2.05;
const BASE_TOKEN_COUNT: f32 = 1024.0;
const MAX_TOKEN_COUNT: f32 = 4096.0;

/// Compute `mu` (the shift exponent) for a given video token count.
/// Token count = `n_video_tokens` after patchify (B=1 case: F * H_lat * W_lat).
pub fn shift_for_token_count(n_tokens: usize) -> f32 {
    let nt = n_tokens as f32;
    let raw = BASE_SHIFT
        + (MAX_SHIFT - BASE_SHIFT) * (nt - BASE_TOKEN_COUNT) / (MAX_TOKEN_COUNT - BASE_TOKEN_COUNT);
    raw.clamp(BASE_SHIFT, MAX_SHIFT)
}

/// Apply the LTX-2 exponential time shift to a raw sigma in [0, 1].
fn apply_shift(sigma: f32, mu: f32) -> f32 {
    if sigma <= 0.0 || sigma >= 1.0 {
        return sigma;
    }
    let e = mu.exp();
    e * sigma / (1.0 + (e - 1.0) * sigma)
}

/// Build a shifted sigma schedule: `num_steps + 1` values from 1.0 down to 0.0.
/// `n_tokens` = number of patchified video tokens (used to pick the shift).
pub fn schedule(num_steps: usize, n_tokens: usize) -> Vec<f32> {
    let mu = shift_for_token_count(n_tokens);
    let mut sigmas = Vec::with_capacity(num_steps + 1);
    for i in 0..=num_steps {
        let raw = 1.0 - i as f32 / num_steps as f32;
        sigmas.push(apply_shift(raw, mu));
    }
    sigmas
}

/// Convert sigma in [0, 1] to the timestep value the DiT consumes.
/// LTX-2 multiplies by `timestep_scale_multiplier = 1000`.
pub fn sigma_to_timestep(sigma: f32) -> f32 {
    sigma * NUM_TRAIN_TIMESTEPS
}

/// Single Euler ODE step: `x_next = x + (sigma_next - sigma) * pred`.
pub fn euler_step(x: &Tensor, pred: &Tensor, sigma: f32, sigma_next: f32) -> Result<Tensor> {
    let dt = sigma_next - sigma;
    x.add(&pred.mul_scalar(dt)?)
}

/// Logit-normal timestep sampler for training — official Lightricks "stretched" mode.
///
/// Matches `serenity::training::noise::sample_shifted_logit_normal` and
/// musubi's `_sample_shifted_logit_normal_sigmas` in "stretched" mode
/// (default for LTX-2.3, and confirmed correct for LTX-2.0 as well by
/// serenity/tests/test_ltx2_shifted_logit_normal.py).
///
/// Returns a continuous sigma in [0, 1].
///
/// Parameters mirror serenity defaults: std=1.0, eps=1e-3, uniform_prob=0.1.
pub fn sample_timestep_logit_normal(rng: &mut rand::rngs::StdRng, shift: f32) -> f32 {
    use rand::Rng;
    use rand_distr::{Distribution, Normal};

    const STD: f32 = 1.0;
    const EPS: f32 = 1e-3;
    const UNIFORM_PROB: f32 = 0.1;

    // Step 1: sample N(shift, std) → sigmoid → raw logit-normal
    let normal = Normal::new(shift, STD).unwrap();
    let z = normal.sample(rng);
    let sigma_raw = 1.0 / (1.0 + (-z).exp()); // sigmoid(z)

    // Step 2: percentile bounds for stretching
    let p_hi = 1.0 / (1.0 + (-(shift + 3.0902 * STD)).exp()); // sigmoid(shift + 3.0902*std)
    let p_lo = 1.0 / (1.0 + (-(shift - 2.5758 * STD)).exp()); // sigmoid(shift - 2.5758*std)
    let denom = (p_hi - p_lo).max(1e-6);

    // Step 3: stretch to cover [0, 1] more evenly
    let mut stretched = (sigma_raw - p_lo) / denom;
    // Reflect values below eps (equivalent to stretched = 2*eps - stretched)
    if stretched < EPS {
        stretched = 2.0 * EPS - stretched;
    }
    stretched = stretched.clamp(0.0, 1.0);

    // Step 4: 10% uniform mix (prevents collapse at extreme token counts)
    let selector: f32 = rng.gen();
    if selector < UNIFORM_PROB {
        // Uniform(eps, 1.0)
        let u: f32 = rng.gen();
        (1.0 - EPS) * u + EPS
    } else {
        stretched
    }
}
