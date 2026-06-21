//! `levers` — the single dispatch home for runtime-config training levers
//! (Rust counterpart of the Mojo `levers.mojo` precedent).
//!
//! The lever MATH lives in `training/features/*` and `training/training_features/*`.
//! This module is *dispatch* + the `cfg → args` binding that each trainer
//! currently inlines. The goal is one call per lever, applied identically by
//! every trainer, so a lever fixed once reaches all of them.
//!
//! **C13 default-off contract:** with config keys at their defaults, each
//! `levers::*` is **bit-identical** to the legacy inline call it replaces — it
//! forwards to the same `features::*` function with the same arguments. New
//! levers are fanned out one trainer at a time (klein first), each gated
//! bit-identical-on-default against the prior inline block before the next
//! trainer migrates.

use crate::config::TrainConfig;
use crate::training::features::lr_schedule;

/// Learning rate for `step`: linear warmup → the config's scheduler.
///
/// Centralizes the `cfg → dispatch_lr` argument binding that every trainer
/// inlines. Bit-identical to the legacy call:
/// `lr_schedule::dispatch_lr(&cfg.learning_rate_scheduler, base_lr, step,
/// total_steps, warmup_steps, cfg.lr_min_factor, cfg.learning_rate_cycles as f32)`.
pub fn lr(
    cfg: &TrainConfig,
    base_lr: f32,
    step: usize,
    total_steps: usize,
    warmup_steps: usize,
) -> f32 {
    lr_schedule::dispatch_lr(
        &cfg.learning_rate_scheduler,
        base_lr,
        step,
        total_steps,
        warmup_steps,
        cfg.lr_min_factor,
        cfg.learning_rate_cycles as f32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{LrScheduler, TrainConfig};
    use crate::training::features::lr_schedule;

    // C13: levers::lr must be bit-identical to the inline dispatch_lr call it
    // replaces. Pure f32 — no device needed. (dispatch_lr's per-scheduler
    // correctness is covered by lr_schedule's own tests; this proves the
    // lever forwards the cfg-derived args unchanged.)
    #[test]
    fn levers_lr_is_bit_identical_to_dispatch_lr() {
        let mut cfg = TrainConfig::default();
        cfg.lr_min_factor = 0.1;
        cfg.learning_rate_cycles = 3.0;
        cfg.learning_rate_scheduler = LrScheduler::Cosine;
        let base = 3e-4_f32;
        let (total, warmup) = (1000usize, 100usize);
        for &step in &[0usize, 1, 50, 99, 100, 500, 999] {
            let via_lever = lr(&cfg, base, step, total, warmup);
            let inline = lr_schedule::dispatch_lr(
                &cfg.learning_rate_scheduler,
                base,
                step,
                total,
                warmup,
                cfg.lr_min_factor,
                cfg.learning_rate_cycles as f32,
            );
            assert_eq!(via_lever.to_bits(), inline.to_bits(), "step={step}");
        }
    }
}
