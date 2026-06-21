use flame_core::{CudaDevice, DType};
use std::sync::Arc;

pub fn init_logging() {
    let _ = env_logger::builder().try_init();
}

pub fn global_bf16_device() -> Arc<CudaDevice> {
    let _timing = crate::trainer_metrics::phase("runtime.global_bf16_device");
    flame_core::config::set_default_dtype(DType::BF16);
    flame_core::global_cuda_device()
}

pub fn global_device() -> Arc<CudaDevice> {
    let _timing = crate::trainer_metrics::phase("runtime.global_device");
    flame_core::global_cuda_device()
}

pub fn cuda_device_bf16(index: usize) -> anyhow::Result<Arc<CudaDevice>> {
    let _timing = crate::trainer_metrics::phase("runtime.cuda_device_bf16");
    flame_core::config::set_default_dtype(DType::BF16);
    CudaDevice::new(index).map_err(|err| anyhow::anyhow!("CudaDevice::new({index}): {err}"))
}

pub fn cuda_device(index: usize) -> anyhow::Result<Arc<CudaDevice>> {
    let _timing = crate::trainer_metrics::phase("runtime.cuda_device");
    CudaDevice::new(index).map_err(|err| anyhow::anyhow!("CudaDevice::new({index}): {err}"))
}

pub fn set_flame_seed(seed: u64) -> anyhow::Result<()> {
    let _timing = crate::trainer_metrics::phase("runtime.set_flame_seed");
    flame_core::rng::set_seed(seed).map_err(|err| anyhow::anyhow!("flame_core set_seed: {err}"))
}

pub struct EnvVarGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvVarGuard {
    pub fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        std::env::set_var(key, value);
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(value) = self.previous.as_ref() {
            std::env::set_var(self.key, value);
        } else {
            std::env::remove_var(self.key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_guard_restores_existing_value() {
        std::env::set_var("ERITRAINER_TEST_ENV_GUARD", "old");
        {
            let _guard = EnvVarGuard::set("ERITRAINER_TEST_ENV_GUARD", "new");
            assert_eq!(
                std::env::var("ERITRAINER_TEST_ENV_GUARD").unwrap(),
                "new"
            );
        }
        assert_eq!(
            std::env::var("ERITRAINER_TEST_ENV_GUARD").unwrap(),
            "old"
        );
        std::env::remove_var("ERITRAINER_TEST_ENV_GUARD");
    }
}
