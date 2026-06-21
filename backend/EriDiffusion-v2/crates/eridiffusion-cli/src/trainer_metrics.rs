use std::time::{Duration, Instant};

pub fn timings_enabled() -> bool {
    std::env::var("ERITRAINER_TIMINGS").map_or(false, |value| {
        !matches!(value.as_str(), "" | "0" | "false" | "FALSE" | "off" | "OFF")
    })
}

pub struct PhaseTimer {
    phase: &'static str,
    start: Instant,
    enabled: bool,
    finished: bool,
}

impl PhaseTimer {
    pub fn new(phase: &'static str) -> Self {
        Self {
            phase,
            start: Instant::now(),
            enabled: timings_enabled(),
            finished: false,
        }
    }

    pub fn finish(mut self) -> Duration {
        self.finished = true;
        let elapsed = self.start.elapsed();
        if self.enabled {
            log::info!("[trainer-timing] {} {:.3}s", self.phase, elapsed.as_secs_f64());
        }
        elapsed
    }
}

impl Drop for PhaseTimer {
    fn drop(&mut self) {
        if self.enabled && !self.finished {
            log::info!(
                "[trainer-timing] {} {:.3}s",
                self.phase,
                self.start.elapsed().as_secs_f64()
            );
        }
    }
}

pub fn phase(phase: &'static str) -> PhaseTimer {
    PhaseTimer::new(phase)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[test]
    fn timings_are_off_by_default_for_quiet_trainers() {
        let _guard = env_lock();
        std::env::remove_var("ERITRAINER_TIMINGS");
        assert!(!timings_enabled());
    }

    #[test]
    fn timings_accept_truthy_env() {
        let _guard = env_lock();
        std::env::set_var("ERITRAINER_TIMINGS", "1");
        assert!(timings_enabled());
        std::env::remove_var("ERITRAINER_TIMINGS");
    }
}
