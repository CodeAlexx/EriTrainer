//! Lightweight EriTrainer entrypoint.
//!
//! This binary intentionally avoids `eridiffusion-cli`, `eridiffusion-core`,
//! and `flame-core` so `train --list`, config resolution, and dry-run dispatch
//! do not trigger CUDA compilation. Model-specific trainers still live in
//! `eridiffusion-cli` and are launched only after a model id is resolved.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use clap::Parser;
use eritrainer_registry::{find_trainer, trainer_ids, TrainerSpec};
use serde_json::Value;

#[derive(Parser, Debug)]
#[command(name = "train")]
struct Args {
    /// Model id, such as `klein`, `sdxl`, `zimage`, or `hidream_o1`.
    #[arg(long)]
    model: Option<String>,
    /// TrainConfig JSON. Used to infer `model_type` when `--model` is absent.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Print the known model ids and exit.
    #[arg(long)]
    list: bool,
    /// Print the resolved command without spawning it.
    #[arg(long)]
    dry_run: bool,
    /// Arguments forwarded to the selected model-specific trainer. Put these
    /// after `--`, for example: `train --model klein -- --cache-dir ...`.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    trainer_args: Vec<String>,
}

fn main() -> ExitCode {
    let args = Args::parse();

    if args.list {
        for id in trainer_ids() {
            println!("{id}");
        }
        return ExitCode::SUCCESS;
    }

    let model_id = match args.model.as_deref() {
        Some(model) => model.to_string(),
        None => match args.config.as_ref().and_then(|path| model_type_from_config(path).ok()) {
            Some(model) => model,
            None => {
                eprintln!("provide --model or --config with a string model_type");
                return ExitCode::from(2);
            }
        },
    };

    let Some(spec) = find_trainer(&model_id) else {
        let known = trainer_ids().collect::<Vec<_>>().join(", ");
        eprintln!("unknown model '{model_id}'. Known models: {known}");
        return ExitCode::from(2);
    };

    let mut forwarded = args.trainer_args;
    if let Some(config) = args.config {
        if !contains_flag(&forwarded, "--config") {
            forwarded.splice(0..0, [String::from("--config"), config.to_string_lossy().into_owned()]);
        }
    }

    let profile = current_profile();
    let (program, mut command_args) = resolve_trainer_command(spec, profile);
    command_args.extend(forwarded);

    if args.dry_run {
        println!("{} {}", program.display(), shell_join(&command_args));
        return ExitCode::SUCCESS;
    }

    match Command::new(&program).args(&command_args).status() {
        Ok(status) if status.success() => ExitCode::SUCCESS,
        Ok(status) => ExitCode::from(status.code().unwrap_or(1) as u8),
        Err(err) => {
            eprintln!("failed to launch {}: {err}", program.display());
            ExitCode::from(1)
        }
    }
}

fn model_type_from_config(path: &Path) -> anyhow::Result<String> {
    let text = std::fs::read_to_string(path)?;
    let value: Value = serde_json::from_str(&text)?;
    for key in ["model_type", "modelType", "model"] {
        if let Some(s) = value.get(key).and_then(Value::as_str) {
            return Ok(s.to_string());
        }
    }
    anyhow::bail!("{} has no string model_type", path.display())
}

fn contains_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag || arg.starts_with(&format!("{flag}=")))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CargoProfile {
    Debug,
    Release,
}

fn current_profile() -> CargoProfile {
    if cfg!(debug_assertions) {
        CargoProfile::Debug
    } else {
        CargoProfile::Release
    }
}

fn resolve_trainer_command(spec: &TrainerSpec, profile: CargoProfile) -> (PathBuf, Vec<String>) {
    let current = std::env::current_exe().ok();
    if let Some(parent) = current.as_ref().and_then(|p| p.parent()) {
        let sibling = parent.join(spec.train_bin);
        if sibling.is_file() {
            return (sibling, Vec::new());
        }
    }

    let manifest = workspace_manifest();
    let mut args = vec![String::from("run")];
    if profile == CargoProfile::Release {
        args.push(String::from("--release"));
    }
    args.extend([
        String::from("--manifest-path"),
        manifest.to_string_lossy().into_owned(),
        String::from("--package"),
        String::from("eridiffusion-cli"),
        String::from("--bin"),
        spec.train_bin.to_string(),
        String::from("--"),
    ]);
    (PathBuf::from("cargo"), args)
}

fn workspace_manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Cargo.toml")
}

fn shell_join(args: &[String]) -> String {
    args.iter()
        .map(|arg| {
            if arg.chars().all(|c| c.is_ascii_alphanumeric() || "-_./:=+".contains(c)) {
                arg.clone()
            } else {
                format!("'{}'", arg.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn missing_spec() -> TrainerSpec {
        TrainerSpec {
            id: "demo",
            train_bin: "missing_train_demo",
            prepare_bin: None,
            sample_bin: None,
            aliases: &[],
            local_sampler: false,
            note: "",
        }
    }

    #[test]
    fn debug_fallback_does_not_force_release_build() {
        let (_program, args) = resolve_trainer_command(&missing_spec(), CargoProfile::Debug);

        assert!(!args.iter().any(|arg| arg == "--release"));
        assert!(args.iter().any(|arg| arg == "missing_train_demo"));
        assert!(args.windows(2).any(|pair| pair == ["--package", "eridiffusion-cli"]));
    }

    #[test]
    fn release_fallback_keeps_release_build() {
        let (_program, args) = resolve_trainer_command(&missing_spec(), CargoProfile::Release);

        assert!(args.iter().any(|arg| arg == "--release"));
    }

    #[test]
    fn config_model_type_accepts_legacy_keys() {
        let root = std::env::temp_dir().join(format!("eritrainer-dispatch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("config.json");
        std::fs::write(&path, r#"{"modelType":"klein"}"#).unwrap();

        assert_eq!(model_type_from_config(&path).unwrap(), "klein");
        let _ = std::fs::remove_dir_all(root);
    }
}
