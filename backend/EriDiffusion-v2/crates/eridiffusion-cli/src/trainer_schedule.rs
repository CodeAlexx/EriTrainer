//! Shared step-cadence helpers for save/sample/validation events.
//!
//! Trainers count user-visible steps from 1, while most loops index from 0.
//! Keeping the boundary here avoids small off-by-one drift between models.

#[inline]
pub fn cadence_enabled(every: usize) -> bool {
    every > 0
}

#[inline]
pub fn cadence_fires(every: usize, step_num: usize, total_steps: usize) -> bool {
    every > 0 && step_num > 0 && step_num % every == 0 && step_num < total_steps
}

#[inline]
pub fn cadence_fires_zero_based(every: usize, step: usize, total_steps: usize) -> bool {
    cadence_fires(every, step + 1, total_steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cadence_is_disabled_at_zero() {
        assert!(!cadence_enabled(0));
        assert!(!cadence_fires(0, 10, 100));
    }

    #[test]
    fn cadence_skips_final_step() {
        assert!(cadence_fires(5, 5, 20));
        assert!(!cadence_fires(5, 20, 20));
    }

    #[test]
    fn zero_based_wrapper_uses_next_user_step() {
        assert!(cadence_fires_zero_based(5, 4, 20));
        assert!(!cadence_fires_zero_based(5, 19, 20));
    }
}
