use anyhow::Context;
use eridiffusion_core::config::{TrainConfig, TrainingMethod};
use eridiffusion_core::training::board::BoardWriter;
use eridiffusion_core::training::features::multi_backend::MultiBackend;
use eridiffusion_core::training::training_features::timestep_dist::{
    TimestepConfig, TimestepDistribution,
};
use flame_core::{CudaDevice, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr as _;
use std::sync::Arc;

pub fn init_logging() {
    crate::trainer_runtime::init_logging();
}

pub fn ensure_output_dir(path: &Path) -> anyhow::Result<()> {
    crate::trainer_preflight::ensure_output_dir(path)
}

pub fn board_resume_step(start_step: usize) -> Option<u64> {
    if start_step > 0 {
        Some(start_step as u64)
    } else {
        None
    }
}

pub fn open_board_writer(output_dir: &Path, resume_step: Option<u64>) -> Option<BoardWriter> {
    let board = BoardWriter::open(output_dir, BoardWriter::new_session_id(), resume_step)
        .map_err(|err| log::warn!("board.db open failed: {err}"))
        .ok();
    if let Some(board) = &board {
        log::info!(
            "SerenityBoard: writing scalars to {}",
            board.db_path.display()
        );
    }
    board
}

pub fn warn_unsupported_multi_backend_flags(cache_dirs: &[PathBuf], weights: &[f32]) {
    if !cache_dirs.is_empty() || !weights.is_empty() {
        log::warn!("--multi-backend-* flags are Klein-only in Phase 2; ignored here");
    }
}

pub fn warn_unsupported_validation_prompts_file(path: Option<&Path>) {
    if path.is_some() {
        log::warn!("--validation-prompts-file is Klein-only in Phase 2; ignored here");
    }
}

pub fn load_train_config(path: &Path) -> anyhow::Result<TrainConfig> {
    let _timing = crate::trainer_metrics::phase("config.load_train_config");
    let file = std::fs::File::open(path)
        .with_context(|| format!("open train config '{}'", path.display()))?;
    serde_json::from_reader(file)
        .with_context(|| format!("parse train config '{}'", path.display()))
}

pub fn load_train_config_or_default(path: Option<&Path>) -> anyhow::Result<TrainConfig> {
    match path {
        Some(path) => load_train_config(path),
        None => Ok(TrainConfig::default()),
    }
}

pub fn apply_lora_basics(config: &mut TrainConfig, rank: usize, lora_alpha: f64, lr: f32) {
    config.training_method = TrainingMethod::Lora;
    config.lora_rank = rank as u64;
    config.lora_alpha = lora_alpha;
    config.learning_rate = lr as f64;
}

pub fn init_bf16_cuda() -> Arc<CudaDevice> {
    crate::trainer_runtime::global_bf16_device()
}

pub fn init_cuda() -> Arc<CudaDevice> {
    crate::trainer_runtime::global_device()
}

pub fn init_bf16_cuda_index(index: usize) -> anyhow::Result<Arc<CudaDevice>> {
    crate::trainer_runtime::cuda_device_bf16(index)
}

pub fn init_cuda_index(index: usize) -> anyhow::Result<Arc<CudaDevice>> {
    crate::trainer_runtime::cuda_device(index)
}

pub fn set_flame_seed(seed: u64) -> anyhow::Result<()> {
    crate::trainer_runtime::set_flame_seed(seed)
}

pub fn cadence_enabled(every: usize) -> bool {
    crate::trainer_schedule::cadence_enabled(every)
}

pub fn cadence_fires(every: usize, step_num: usize, total_steps: usize) -> bool {
    crate::trainer_schedule::cadence_fires(every, step_num, total_steps)
}

pub fn cadence_fires_zero_based(every: usize, step: usize, total_steps: usize) -> bool {
    crate::trainer_schedule::cadence_fires_zero_based(every, step, total_steps)
}

pub fn sample_logit_normal_timestep(
    rng: &mut rand::rngs::StdRng,
    num_train_timesteps: usize,
    bias: f32,
    scale: f32,
    shift: f32,
    min_strength: f32,
    max_strength: f32,
) -> f32 {
    use rand_distr::{Distribution, Normal};

    let normal = Normal::new(bias, scale).unwrap();
    let z = normal.sample(rng);
    let logit_normal = 1.0 / (1.0 + (-z).exp());
    let min_t = num_train_timesteps as f32 * min_strength.max(0.0);
    let max_t = num_train_timesteps as f32 * max_strength.min(1.0);
    let t = logit_normal * (max_t - min_t) + min_t;
    apply_timestep_shift(t, num_train_timesteps, shift)
}

pub fn apply_timestep_shift(t: f32, num_train_timesteps: usize, shift: f32) -> f32 {
    if (shift - 1.0).abs() < 1e-6 {
        t
    } else {
        num_train_timesteps as f32 * shift * t / ((shift - 1.0) * t + num_train_timesteps as f32)
    }
}

pub fn build_timestep_config(
    distribution: &str,
    weight: f32,
    bias: f32,
    min_strength: f32,
    max_strength: f32,
) -> anyhow::Result<TimestepConfig> {
    let dist = TimestepDistribution::from_str(distribution)
        .map_err(|err| anyhow::anyhow!("--timestep-distribution: {err}"))?;
    Ok(TimestepConfig {
        distribution: dist,
        noising_weight: weight,
        noising_bias: bias,
        min_strength: min_strength.max(0.0),
        max_strength: max_strength.min(1.0),
    })
}

pub fn build_full_strength_timestep_config(
    distribution: &str,
    weight: f32,
    bias: f32,
) -> anyhow::Result<TimestepConfig> {
    build_timestep_config(distribution, weight, bias, 0.0, 1.0)
}

pub fn collect_safetensor_shards(path: &Path, label: &str) -> anyhow::Result<Vec<PathBuf>> {
    crate::trainer_cache::collect_safetensor_shards(path, label)
}

pub fn list_cache_safetensors(path: &Path) -> anyhow::Result<Vec<PathBuf>> {
    crate::trainer_cache::list_safetensors(path)
}

pub fn list_cache_safetensors_or_empty(path: &Path) -> anyhow::Result<Vec<PathBuf>> {
    crate::trainer_cache::list_safetensors_or_empty(path)
}

pub fn list_cache_files(path: &Path, extensions: &[&str]) -> anyhow::Result<Vec<PathBuf>> {
    crate::trainer_cache::list_files_with_extensions(path, extensions, "cache")
}

pub fn list_cache_files_or_empty(
    path: &Path,
    extensions: &[&str],
) -> anyhow::Result<Vec<PathBuf>> {
    crate::trainer_cache::list_files_with_extensions_or_empty(path, extensions, "cache")
}

pub fn build_multi_backend(
    cache_dirs: &[PathBuf],
    weights: &[f32],
    repeats: &[u32],
) -> anyhow::Result<Option<MultiBackend>> {
    if cache_dirs.is_empty() && weights.is_empty() {
        return Ok(None);
    }

    if cache_dirs.is_empty() || weights.is_empty() {
        anyhow::bail!(
            "multi-backend: must set BOTH --multi-backend-cache-dirs and --multi-backend-weights, or neither"
        );
    }

    if cache_dirs.len() != weights.len() {
        anyhow::bail!(
            "--multi-backend-cache-dirs ({}) and --multi-backend-weights ({}) must have equal length",
            cache_dirs.len(),
            weights.len()
        );
    }

    let backend = if repeats.is_empty() {
        MultiBackend::new(cache_dirs, weights)?
    } else {
        if repeats.len() != cache_dirs.len() {
            anyhow::bail!(
                "--multi-backend-repeats ({}) must match --multi-backend-cache-dirs ({})",
                repeats.len(),
                cache_dirs.len()
            );
        }
        log::info!("[multi-backend-repeats] {:?}", repeats);
        MultiBackend::new_with_repeats(cache_dirs, weights, repeats)?
    };

    log::info!(
        "[multi-backend] {} backends, {} total samples",
        backend.backends.len(),
        backend.total_files()
    );
    for (i, count) in backend.per_backend_counts().iter().enumerate() {
        log::info!(
            "  backend[{i}] dir={} files={} weight={:.4}",
            cache_dirs[i].display(),
            count,
            backend.weights[i]
        );
    }

    Ok(Some(backend))
}

pub fn load_safetensors_file_or_dir(
    path: &Path,
    device: &Arc<CudaDevice>,
) -> flame_core::Result<HashMap<String, Tensor>> {
    if path.is_file() {
        return flame_core::serialization::load_file(path, device);
    }

    let mut entries: Vec<PathBuf> = std::fs::read_dir(path)
        .map_err(|err| {
            flame_core::Error::Io(format!(
                "read safetensors directory '{}': {err}",
                path.display()
            ))
        })?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("safetensors"))
        .collect();
    entries.sort();

    if entries.is_empty() {
        return Err(flame_core::Error::Io(format!(
            "no safetensors at '{}'",
            path.display()
        )));
    }

    let mut all = HashMap::new();
    for path in entries {
        all.extend(flame_core::serialization::load_file(&path, device)?);
    }
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_safetensor_shards_sorts_directory_entries() {
        let root = std::env::temp_dir().join(format!(
            "eritrainer-shards-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("b.safetensors"), b"").unwrap();
        std::fs::write(root.join("a.safetensors"), b"").unwrap();
        std::fs::write(root.join("ignored.bin"), b"").unwrap();

        let shards = collect_safetensor_shards(&root, "").unwrap();
        let names: Vec<_> = shards
            .iter()
            .map(|path| path.file_name().unwrap().to_str().unwrap().to_owned())
            .collect();

        assert_eq!(names, ["a.safetensors", "b.safetensors"]);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn logit_normal_timestep_shift_identity_matches_legacy_formula() {
        let t = apply_timestep_shift(123.5, 1000, 1.0);
        assert!((t - 123.5).abs() < 1e-6);
    }

    #[test]
    fn build_timestep_config_clamps_strength_range() {
        let cfg = build_timestep_config("uniform", 0.0, 0.0, -0.25, 1.25).unwrap();
        assert_eq!(cfg.min_strength, 0.0);
        assert_eq!(cfg.max_strength, 1.0);
    }

    fn make_cache_dir(name: &str, count: usize) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "eritrainer-multi-backend-{name}-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        for i in 0..count {
            std::fs::write(root.join(format!("sample_{i}.safetensors")), b"placeholder").unwrap();
        }
        root
    }

    #[test]
    fn build_multi_backend_is_disabled_when_args_empty() {
        let backend = build_multi_backend(&[], &[], &[]).unwrap();
        assert!(backend.is_none());
    }

    #[test]
    fn build_multi_backend_rejects_partial_args() {
        let dir = make_cache_dir("partial", 1);
        let err = match build_multi_backend(&[dir.clone()], &[], &[]) {
            Ok(_) => panic!("partial multi-backend args should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("must set BOTH"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn build_multi_backend_applies_repeats() {
        let dir_a = make_cache_dir("a", 2);
        let dir_b = make_cache_dir("b", 3);
        let backend = build_multi_backend(
            &[dir_a.clone(), dir_b.clone()],
            &[1.0, 1.0],
            &[2, 1],
        )
        .unwrap()
        .unwrap();

        assert_eq!(backend.total_files(), 5);
        assert!((backend.weights[0] - (2.0 / 3.0)).abs() < 1e-6);
        assert!((backend.weights[1] - (1.0 / 3.0)).abs() < 1e-6);

        let _ = std::fs::remove_dir_all(dir_a);
        let _ = std::fs::remove_dir_all(dir_b);
    }
}
