//! EMA (advanced) — power-decay schedule with warmup, min/max decay clamps,
//! and validation-time parameter swap.
//!
//! Phase 3.
//!
//! Decay formula (matches diffusers `EMAModel` and SimpleTuner
//! `helpers/training/ema.py`):
//!
//! ```text
//!   step_eff = step - update_after_step
//!   decay    = 1 - (1 + step_eff / inv_gamma)^(-power)
//!   decay    = clamp(decay, min_decay, max_decay)
//!   decay    = 0.0   if step <= update_after_step
//! ```
//!
//! `decay = 0.0` means "skip this update" (the consumer short-circuits when
//! the schedule returns 0).
//!
//! Default-off invariance: leaving `inv_gamma=1.0`, `power=0.6667`,
//! `update_after_step=0`, `min_decay=0.0`, `max_decay=0.999` yields the
//! standard warmup curve. Existing trainers that don't construct EMA at all
//! are unaffected.
//!
//! Reference: diffusers `EMAModel`, SimpleTuner `helpers/training/ema.py`.

/// Knobs for EMA decay schedule.
#[derive(Debug, Clone, Copy)]
pub struct EmaConfig {
    pub inv_gamma: f32,
    pub power: f32,
    pub update_after_step: u64,
    pub min_decay: f32,
    pub max_decay: f32,
}

impl Default for EmaConfig {
    fn default() -> Self {
        Self {
            inv_gamma: 1.0,
            power: 0.6667,
            update_after_step: 0,
            min_decay: 0.0,
            max_decay: 0.9999,
        }
    }
}

/// Decay multiplier `α(step)` for the EMA update
/// `shadow = α·shadow + (1-α)·param`.
///
/// Returns `0.0` when `step <= update_after_step` (caller treats this as
/// "skip"). After warmup, returns the diffusers-compatible power-decay
/// curve clamped to `[min_decay, max_decay]`.
pub fn decay_at_step(cfg: &EmaConfig, step: u64) -> f32 {
    if step <= cfg.update_after_step {
        return 0.0;
    }
    let effective_step = (step - cfg.update_after_step) as f32;
    let inv_gamma = cfg.inv_gamma.max(1e-8);
    let value = 1.0 - (1.0 + effective_step / inv_gamma).powf(-cfg.power);
    value.max(cfg.min_decay).min(cfg.max_decay)
}

// ── Karras momentum policy — A3 (AsymFlow trainer) ───────────────────────
//
// LakonLab's AsymFlow training (`asymflux2_klein_32gpus.py:204-213`) uses a
// distinct curve from the diffusers default:
//   ```python
//   def karras(self, runner, gamma=7.0, max_momentum=1.0):
//       t = max(runner.iter + 1 - self.start_iter, 1)
//       ema_beta = min((1 - 1 / t) ** (gamma + 1), max_momentum)
//       return dict(momentum=ema_beta)
//   ```
//
// The plan (asymflow_milestone_plan.md §1, TBD #2) called for adding a
// `Karras { gamma, max_momentum }` variant to `EmaConfig`. The 12 existing
// EmaConfig literal constructions in EDv2 trainers (`train_klein.rs:649`,
// `train_chroma.rs:609`, etc.) use field-by-field syntax without
// `..Default::default()`, so adding a non-default field would break the
// 11 other trainers we're not allowed to touch in A3. Solution: ship Karras
// as a parallel free function consuming dedicated `KarrasConfig`; the
// trainer chooses which decay function to call at the EMA update site.
// This keeps `EmaConfig` byte-identical for the 12 existing call sites.

/// Karras EMA momentum config — orthogonal to [`EmaConfig`]. Used by
/// `train_asymflow` (asymflux2 config: `gamma=7.0, start_iter=100,
/// max_momentum=1.0`).
#[derive(Debug, Clone, Copy)]
pub struct KarrasConfig {
    pub gamma: f32,
    /// Maps to LakonLab `start_iter` (`asymflux2_klein_32gpus.py:210`).
    /// Mirrors the pre-`start_iter` behavior of the upstream hook: when
    /// `step <= start_iter` the trainer should COPY `param → shadow` (the
    /// hook does `p_ema.data.copy_(p_net.data)`); past `start_iter` the
    /// returned decay drives `lerp(param, shadow, momentum)`.
    pub start_iter: u64,
    pub max_momentum: f32,
}

impl Default for KarrasConfig {
    fn default() -> Self {
        // asymflux2_klein_32gpus.py defaults.
        Self {
            gamma: 7.0,
            start_iter: 100,
            max_momentum: 1.0,
        }
    }
}

/// Karras decay `(1 - 1/t)^(gamma+1)` capped at `max_momentum`, where
/// `t = max(step - start_iter, 1)`. Mirror of LakonLab `ema_hook.py:135-138`.
///
/// Trainer-side convention: the caller passes `step = iter + 1` (1-based
/// completed-step counter, matching `train_klein.rs:1552`
/// `e.update_with_schedule(... (step + 1) as u64)`). With `start_iter` set to
/// the same numeric value used in the Python config, this matches the
/// upstream `runner.iter + 1 - self.start_iter` exactly.
///
/// Returns `0.0` for `step <= start_iter` (pre-warmup); the caller mirrors
/// the upstream hook by either short-circuiting the update or by copying the
/// live param into the shadow when the decay is `0.0`.
pub fn decay_karras(cfg: &KarrasConfig, step: u64) -> f32 {
    let t_raw = (step as i64) - (cfg.start_iter as i64);
    if t_raw <= 0 {
        // Pre-warmup: upstream hook copies param→shadow. Returning 0.0 lets
        // the consumer treat this as "skip / hard-set"; an alternative
        // shaped-API would be `Option<f32>` but the existing `decay_at_step`
        // returns `0.0` for the same semantic, so we mirror that.
        return 0.0;
    }
    let t = t_raw.max(1) as f32;
    let base = (1.0 - 1.0 / t).max(0.0);
    let raw = base.powf(cfg.gamma + 1.0);
    raw.min(cfg.max_momentum).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn pre_warmup_returns_zero() {
        let cfg = EmaConfig {
            update_after_step: 100,
            ..Default::default()
        };
        assert_eq!(decay_at_step(&cfg, 0), 0.0);
        assert_eq!(decay_at_step(&cfg, 50), 0.0);
        assert_eq!(decay_at_step(&cfg, 100), 0.0);
        assert!(decay_at_step(&cfg, 101) > 0.0);
    }

    #[test]
    fn diffusers_formula_step1() {
        // With defaults inv_gamma=1.0, power=0.6667, update_after_step=0:
        // step=1: 1 - (1 + 1/1)^(-0.6667) = 1 - 2^(-0.6667) ≈ 1 - 0.6300 = 0.3700
        let cfg = EmaConfig::default();
        let d = decay_at_step(&cfg, 1);
        let expected = 1.0 - 2.0_f32.powf(-0.6667);
        assert!(approx(d, expected, 1e-5), "got {} expected {}", d, expected);
    }

    #[test]
    fn approaches_max_decay_at_high_step() {
        let cfg = EmaConfig::default();
        let d = decay_at_step(&cfg, 1_000_000);
        // Should be very close to max_decay = 0.9999.
        assert!(d <= cfg.max_decay + 1e-7);
        assert!(d > 0.999, "expected near max_decay, got {}", d);
    }

    #[test]
    fn min_decay_floors_early_warmup() {
        let cfg = EmaConfig {
            min_decay: 0.5,
            ..Default::default()
        };
        // step=1 raw value ≈ 0.37, should be floored to 0.5.
        let d = decay_at_step(&cfg, 1);
        assert!(approx(d, 0.5, 1e-6), "got {}", d);
    }

    #[test]
    fn max_decay_clamps_late_curve() {
        let cfg = EmaConfig {
            max_decay: 0.95,
            ..Default::default()
        };
        let d = decay_at_step(&cfg, 1_000_000);
        assert!(approx(d, 0.95, 1e-6), "got {}", d);
    }

    // ── Karras momentum policy — A3 (AsymFlow trainer) ───────────────────
    //
    // Mirror of LakonLab `ema_hook.py:135-138` with `gamma=7.0,
    // start_iter=100, max_momentum=1.0` (the asymflux2 config). The trainer
    // calls `decay_karras(cfg, step)` with `step = iter + 1` (1-based
    // step counter), so for `iter ∈ {99, 100, 101, 110, 120}` we pass
    // `step ∈ {100, 101, 102, 111, 121}` and `start_iter = 100`.
    //
    // Hand-computed values (matches `t = max(step - start_iter, 1)` then
    // `(1 - 1/t)^(gamma+1)` capped at max_momentum):
    //   iter=99  → step=100 → t_raw=0 → pre-warmup → 0
    //   iter=100 → step=101 → t = 1                → (1 - 1)^8 = 0
    //   iter=101 → step=102 → t = 2                → (0.5)^8 = 0.00390625
    //   iter=110 → step=111 → t = 11               → (10/11)^8 ≈ 0.4665074
    //   iter=120 → step=121 → t = 21               → (20/21)^8 ≈ 0.6797068
    //
    // No GPU — pure scalar arithmetic, EMA test acceptance per plan §A3.
    #[test]
    fn karras_matches_lakonlab_curve() {
        let cfg = KarrasConfig {
            gamma: 7.0,
            start_iter: 100,
            max_momentum: 1.0,
        };
        let cases = [
            (100u64, 0.0_f32),
            (101, 0.0),
            (102, 0.5_f32.powi(8)),
            (111, (10.0_f32 / 11.0).powi(8)),
            (121, (20.0_f32 / 21.0).powi(8)),
        ];
        for (step, expected) in cases {
            let got = decay_karras(&cfg, step);
            assert!(
                approx(got, expected, 1e-5),
                "karras decay step={step}: got {got} expected {expected}"
            );
        }
    }

    #[test]
    fn karras_caps_at_max_momentum() {
        let cfg = KarrasConfig {
            gamma: 7.0,
            start_iter: 0,
            max_momentum: 0.5,
        };
        // Large step → (1 - 1/big)^8 ≈ 1.0; cap forces 0.5.
        let d = decay_karras(&cfg, 1_000_000);
        assert!(
            approx(d, 0.5, 1e-6),
            "expected cap at 0.5, got {d}"
        );
    }

    #[test]
    fn karras_default_matches_asymflux2_config() {
        // asymflux2_klein_32gpus.py:210-213 declares
        //   start_iter=100, gamma=7.0, max_momentum=1.0
        // KarrasConfig::default() must return that exact shape.
        let cfg = KarrasConfig::default();
        assert_eq!(cfg.start_iter, 100);
        assert!(approx(cfg.gamma, 7.0, 1e-9));
        assert!(approx(cfg.max_momentum, 1.0, 1e-9));
    }

    #[test]
    fn update_after_step_offset() {
        // With update_after_step=10, step=11 should have effective_step=1
        // and return the same value as step=1 with update_after_step=0.
        let cfg_a = EmaConfig::default();
        let cfg_b = EmaConfig {
            update_after_step: 10,
            ..Default::default()
        };
        let a = decay_at_step(&cfg_a, 1);
        let b = decay_at_step(&cfg_b, 11);
        assert!(approx(a, b, 1e-7), "{} != {}", a, b);
    }
}
