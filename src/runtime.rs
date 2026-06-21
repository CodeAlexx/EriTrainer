//! Runtime bridge: launch an EDv2 `train_*` binary as a subprocess, tail its
//! stdout progress line, expose live stats to the UI. Mirrors the Mojo
//! `TrainerRuntimeBridge.mojo` behavior (config-driven launch + progress poll).
//!
//! The monitor contract is the single stdout line every EDv2 trainer emits via
//! `eridiffusion_core::training::progress::log_step_with_resume`:
//!
//! `[<tag>] step N/T | epoch e/E | loss X.XXXX | grad_norm X.XXXX | X.Xs/step | elapsed H:MM:SS | ETA H:MM:SS`
//!
//! Owned by the orchestrator (validation backbone). The subprocess wiring in
//! `start()`/`tick()` is intentionally minimal here; the stdout-tail thread +
//! command path resolution land alongside the M1 Klein launch.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;

use serde::Serialize;

use crate::config::TrainConfig;

#[derive(Clone, Default)]
pub struct LiveStats {
    pub step: u64,
    pub total_steps: u64,
    pub epoch: u64,
    pub total_epochs: u64,
    pub loss: f32,
    pub smooth_loss: f32,
    pub grad_norm: f32,
    pub learning_rate: f32,
    pub speed_s_step: f32,
    pub elapsed_secs: u64,
    pub eta_secs: u64,
    pub gpu_util: f32,
    pub vram_gb: f32,
    pub vram_total_gb: f32,
    pub temp_c: i32,
    pub cpu_util: f32,
    pub ram_gb: f32,
    pub ram_total_gb: f32,
}

#[derive(Default)]
pub struct Runtime {
    pub child: Option<Child>,
    pub has_running: bool,
    pub paused: bool,
    pub live: LiveStats,
    pub status_text: String,
    pub last_command: String,
    pub backend_label: String,
    pub gpu_name: String,
    pub gpu_driver: String,
    pub cpu_name: String,
    /// Where live stats are coming from: "waiting" | "stdout" | "callbacks".
    pub progress_source: String,
    pub logs: Vec<String>,
    pub samples: Vec<String>,
    pub checkpoints: Vec<String>,
    /// Receiver fed by the stdout/stderr tail threads of the live child.
    pub rx: Option<Receiver<String>>,
}

impl Runtime {
    /// Launch the selected model's EDv2 trainer as a subprocess, piping its
    /// stdout/stderr into a tail thread → channel. Fails LOUD (status + log, no
    /// fake "running") when the command can't be built or the process can't be
    /// spawned — a blocked launch must never look like a started run.
    pub fn start(&mut self, cfg: &TrainConfig) {
        if self.child.is_some() {
            self.stop();
        }
        self.live = LiveStats {
            total_steps: cfg.max_train_steps.max(1.0) as u64,
            ..LiveStats::default()
        };
        self.backend_label = cfg.model_type.clone();
        self.progress_source = String::from("waiting");

        // For models whose trainer REQUIRES --config, auto-generate the EDv2
        // TrainConfig JSON from the UI fields when the user hasn't supplied one,
        // so a run launches from the form (mirrors the Mojo
        // `_write_runner_train_config`). A user-set Run Config is left untouched.
        let mut eff = cfg.clone();
        if needs_generated_config(&eff.model_type) && eff.run_config_path.trim().is_empty() {
            match write_runner_config(&eff) {
                Ok(path) => {
                    self.push_log(format!("wrote runner config {path}"));
                    eff.run_config_path = path;
                }
                Err(e) => {
                    self.has_running = false;
                    self.status_text = format!("config gen failed: {e}");
                    self.push_log(format!("CONFIG GEN FAILED: {e}"));
                    return;
                }
            }
        }

        let (program, args) = match build_command(&eff) {
            Ok(pa) => pa,
            Err(e) => {
                self.has_running = false;
                self.status_text = format!("launch blocked: {e}");
                self.push_log(format!("LAUNCH BLOCKED: {e}"));
                return;
            }
        };
        self.last_command = format!("{} {}", program, args.join(" "));

        let mut cmd = Command::new(&program);
        cmd.args(&args).stdout(Stdio::piped()).stderr(Stdio::piped());
        for (k, v) in launch_env(cfg) {
            cmd.env(k, v);
        }
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                self.has_running = false;
                self.status_text = format!("spawn failed: {e}");
                self.push_log(format!("SPAWN FAILED ({program}): {e}"));
                return;
            }
        };

        let (tx, rx) = mpsc::channel::<String>();
        if let Some(out) = child.stdout.take() {
            let tx = tx.clone();
            thread::spawn(move || {
                for line in BufReader::new(out).lines().map_while(Result::ok) {
                    if tx.send(line).is_err() {
                        break;
                    }
                }
            });
        }
        if let Some(err) = child.stderr.take() {
            let tx = tx.clone();
            thread::spawn(move || {
                for line in BufReader::new(err).lines().map_while(Result::ok) {
                    if tx.send(format!("[stderr] {line}")).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);

        self.child = Some(child);
        self.rx = Some(rx);
        self.has_running = true;
        self.paused = false;
        self.status_text = String::from("running");
        self.push_log(format!("launch {}", self.last_command));
    }

    pub fn stop(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
        self.rx = None;
        self.has_running = false;
        self.paused = false;
        self.status_text = String::from("stopped");
    }

    /// Once per frame: refresh system metrics, drain captured stdout/stderr into
    /// the parser + log, and detect process exit.
    pub fn tick(&mut self) {
        crate::sysmetrics::refresh(self);

        let mut lines: Vec<String> = Vec::new();
        if let Some(rx) = self.rx.as_ref() {
            while let Ok(line) = rx.try_recv() {
                lines.push(line);
            }
        }
        for line in lines {
            if self.apply_progress_line(&line) {
                self.progress_source = String::from("stdout");
            }
            self.push_log(line);
        }

        if let Some(child) = self.child.as_mut() {
            if let Ok(Some(status)) = child.try_wait() {
                self.has_running = false;
                self.paused = false;
                self.status_text = if status.success() {
                    String::from("completed")
                } else {
                    format!("exited ({status})")
                };
                self.child = None;
                self.rx = None;
            }
        }
    }

    /// Append a log line, capping the ring at 200 entries.
    fn push_log(&mut self, line: String) {
        self.logs.push(line);
        const MAX: usize = 200;
        if self.logs.len() > MAX {
            let drop = self.logs.len() - MAX;
            self.logs.drain(0..drop);
        }
    }

    /// Feed one captured stdout line through the progress parser. Returns true
    /// if it was a recognized progress line (advanced the step counter).
    pub fn apply_progress_line(&mut self, line: &str) -> bool {
        parse_progress_line(line, &mut self.live)
    }

    pub fn progress_fraction(&self) -> f32 {
        if self.live.total_steps == 0 {
            0.0
        } else {
            (self.live.step as f32 / self.live.total_steps as f32).clamp(0.0, 1.0)
        }
    }
}

/// Parse one progress line emitted by EDv2's `progress::log_step_with_resume`.
///
/// Tolerates the optional `[tag] ` prefix and any segment ordering. Updates
/// only the fields present. Returns true iff a `step N/T` segment was found
/// (so non-progress stdout lines are ignored).
pub fn parse_progress_line(line: &str, out: &mut LiveStats) -> bool {
    // Strip an optional leading `[tag] ` prefix.
    let trimmed = line.trim_start();
    let body = if trimmed.starts_with('[') {
        match trimmed.find(']') {
            Some(i) => &trimmed[i + 1..],
            None => trimmed,
        }
    } else {
        trimmed
    };

    let mut found_step = false;
    for raw in body.split('|') {
        let seg = raw.trim();
        if let Some(rest) = seg.strip_prefix("step ") {
            if let Some((n, t)) = rest.split_once('/') {
                if let (Ok(n), Ok(t)) = (n.trim().parse::<u64>(), t.trim().parse::<u64>()) {
                    out.step = n;
                    out.total_steps = t;
                    found_step = true;
                }
            }
        } else if let Some(rest) = seg.strip_prefix("epoch ") {
            if let Some((e, te)) = rest.split_once('/') {
                if let (Ok(e), Ok(te)) = (e.trim().parse::<u64>(), te.trim().parse::<u64>()) {
                    out.epoch = e;
                    out.total_epochs = te;
                }
            }
        } else if let Some(rest) = seg.strip_prefix("loss ") {
            if let Ok(v) = rest.trim().parse::<f32>() {
                out.loss = v;
            }
        } else if let Some(rest) = seg.strip_prefix("grad_norm ") {
            if let Ok(v) = rest.trim().parse::<f32>() {
                out.grad_norm = v;
            }
        } else if let Some(rest) = seg.strip_suffix("s/step") {
            if let Ok(v) = rest.trim().parse::<f32>() {
                out.speed_s_step = v;
            }
        } else if let Some(rest) = seg.strip_prefix("elapsed ") {
            out.elapsed_secs = parse_hms(rest.trim());
        } else if let Some(rest) = seg.strip_prefix("ETA ") {
            out.eta_secs = parse_hms(rest.trim());
        }
    }
    found_step
}

/// Parse `H:MM:SS` (or `MM:SS`, or `SS`) into total seconds.
fn parse_hms(s: &str) -> u64 {
    let mut secs = 0u64;
    for part in s.split(':') {
        secs = secs * 60 + part.trim().parse::<u64>().unwrap_or(0);
    }
    secs
}

// --- Runner --config generation (write the EDv2 TrainConfig JSON from the UI) ---

/// EDv2 `TrainConfig` schema (subset) emitted as the trainer `--config` JSON.
/// Mirrors the working example configs (e.g. `configs/klein9b_alina.json`) — the
/// load-bearing fields EDv2's `from_json_path` reads; extras are tolerated.
#[derive(Serialize)]
struct RunnerConfig {
    base_model_name: String,
    training_method: String,
    peft_type: String,
    lora_rank: u64,
    lora_alpha: f64,
    learning_rate: f64,
    batch_size: u64,
    epochs: u64,
    timestep_distribution: String,
    timestep_shift: f32,
    min_noising_strength: f32,
    max_noising_strength: f32,
    mse_strength: f32,
    mae_strength: f32,
    loss_weight_fn: String,
    clip_grad_norm: f32,
}

impl RunnerConfig {
    fn from_ui(cfg: &TrainConfig) -> Self {
        let or = |s: &str, d: &str| {
            if s.trim().is_empty() {
                d.to_string()
            } else {
                s.to_string()
            }
        };
        RunnerConfig {
            base_model_name: cfg.base_model_path.clone(),
            training_method: String::from("LORA"),
            peft_type: or(&cfg.peft_type, "LORA"),
            lora_rank: cfg.lora_rank.max(1.0) as u64,
            lora_alpha: cfg.lora_alpha as f64,
            learning_rate: cfg.learning_rate as f64,
            batch_size: cfg.batch_size.max(1.0) as u64,
            epochs: cfg.epochs.max(1.0) as u64,
            timestep_distribution: or(&cfg.timestep_distribution, "LOGIT_NORMAL"),
            timestep_shift: cfg.timestep_shift,
            min_noising_strength: cfg.min_noising_strength,
            max_noising_strength: cfg.max_noising_strength,
            mse_strength: cfg.mse_strength,
            mae_strength: cfg.mae_strength,
            loss_weight_fn: or(&cfg.loss_weight_fn, "CONSTANT"),
            clip_grad_norm: cfg.clip_grad_norm,
        }
    }
}

/// Serialize the runner `--config` JSON from the UI fields.
pub fn runner_config_json(cfg: &TrainConfig) -> String {
    serde_json::to_string_pretty(&RunnerConfig::from_ui(cfg))
        .unwrap_or_else(|_| String::from("{}"))
}

/// Which models' trainers REQUIRE `--config` — EriTrainer auto-generates one
/// from the form when the user hasn't supplied a Run Config path.
fn needs_generated_config(model_type: &str) -> bool {
    matches!(model_type, "klein" | "ernie" | "anima" | "sd35")
}

/// Write the generated runner config to disk; return its path. Lands in the
/// output dir (or the temp dir if unset) as `eritrainer_run_config.json`.
fn write_runner_config(cfg: &TrainConfig) -> Result<String, String> {
    let dir = if cfg.output_dir.trim().is_empty() {
        std::env::temp_dir()
    } else {
        PathBuf::from(&cfg.output_dir)
    };
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    let path = dir.join("eritrainer_run_config.json");
    std::fs::write(&path, runner_config_json(cfg))
        .map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(path.to_string_lossy().into_owned())
}

// --- Launch command construction ---

/// EDv2 workspace dir; override with `ERITRAINER_EDV2_DIR`.
fn edv2_dir() -> PathBuf {
    match std::env::var("ERITRAINER_EDV2_DIR") {
        Ok(d) if !d.trim().is_empty() => PathBuf::from(d),
        _ => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("backend/EriDiffusion-v2"),
    }
}

/// How to launch an EDv2 bin: prefer a prebuilt `target/{release,debug}/<bin>`,
/// else fall back to `cargo run --bin <bin>` in this app's current profile.
fn resolve_launcher(bin: &str) -> (String, Vec<String>) {
    let dir = edv2_dir();
    let release = dir.join("target/release").join(bin);
    let debug = dir.join("target/debug").join(bin);
    if release.is_file() {
        (release.to_string_lossy().into_owned(), Vec::new())
    } else if debug.is_file() {
        (debug.to_string_lossy().into_owned(), Vec::new())
    } else {
        let manifest = dir.join("Cargo.toml");
        (String::from("cargo"), cargo_run_args(&manifest, bin))
    }
}

fn cargo_run_args(manifest: &Path, bin: &str) -> Vec<String> {
    let mut args = vec![String::from("run")];
    if !cfg!(debug_assertions) {
        args.push(String::from("--release"));
    }
    args.extend([
        String::from("--manifest-path"),
        manifest.to_string_lossy().into_owned(),
        String::from("--bin"),
        String::from(bin),
        String::from("--"),
    ]);
    args
}

/// Env for the spawned trainer: libtorch on LD_LIBRARY_PATH (required for the
/// EDv2 bins to load — they link libtorch), RUST_LOG so the progress lines
/// print, and the Klein-safe FLAME allocator flags. The rest of the parent env
/// is inherited (Command default). `ERITRAINER_LIBTORCH` overrides the libtorch
/// dir; an existing LD_LIBRARY_PATH is preserved (prepended, not clobbered).
fn launch_env(cfg: &TrainConfig) -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = Vec::new();

    let libtorch = std::env::var("ERITRAINER_LIBTORCH")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| String::from("/home/alex/libs/libtorch/lib"));
    let ld = match std::env::var("LD_LIBRARY_PATH") {
        Ok(existing) if !existing.is_empty() => format!("{libtorch}:{existing}"),
        _ => libtorch,
    };
    env.push((String::from("LD_LIBRARY_PATH"), ld));

    if std::env::var("RUST_LOG").is_err() {
        env.push((String::from("RUST_LOG"), String::from("info")));
    }

    // Klein-safe allocator flags: the approved klein recipe runs with the pool
    // OFF — pool ON triggers a step-2 CUDA crash on the 9B (run_k9_sanitizer.sh).
    if cfg.model_type == "klein" {
        env.push((String::from("FLAME_ALLOC_POOL"), String::from("0")));
        env.push((String::from("FLAME_USE_STATIC_SLAB"), String::from("0")));
    }
    env
}

/// Build (program, args) for the selected model's trainer through the local
/// `train --model <id>` dispatcher. All registered trainer IDs map to their
/// real per-model CLI; `config::model_verified` records which have current
/// end-to-end smoke evidence.

fn require(path: &str, what: &str) -> Result<(), String> {
    if path.trim().is_empty() {
        Err(format!("{what} is required"))
    } else {
        Ok(())
    }
}

/// Trailing flags common to most trainers: --steps/--rank/--lora-alpha/--lr,
/// then optional --batch-size / --warmup-steps / --offload / --output-dir.
fn common_train_flags(cfg: &TrainConfig, batch: bool, warmup: bool, offload: bool) -> Vec<String> {
    let mut a = vec![
        "--steps".to_string(),
        (cfg.max_train_steps.max(1.0) as u64).to_string(),
        "--rank".to_string(),
        (cfg.lora_rank.max(1.0) as u64).to_string(),
        "--lora-alpha".to_string(),
        cfg.lora_alpha.to_string(),
        "--lr".to_string(),
        cfg.learning_rate.to_string(),
    ];
    if batch {
        a.push("--batch-size".into());
        a.push((cfg.batch_size.max(1.0) as u64).to_string());
    }
    if warmup {
        a.push("--warmup-steps".into());
        a.push((cfg.learning_rate_warmup_steps.max(0.0) as u64).to_string());
    }
    if offload && cfg.activation_offloading {
        a.push("--offload".into());
    }
    if !cfg.output_dir.trim().is_empty() {
        a.push("--output-dir".into());
        a.push(cfg.output_dir.clone());
    }
    a
}

/// flux.1-dev: `--transformer` (file), `--config` optional, `--offload`, no batch.
fn flux_args(cfg: &TrainConfig) -> Result<Vec<String>, String> {
    require(&cfg.base_model_path, "base model path (--transformer)")?;
    require(&cfg.cache_dir, "cache dir (--cache-dir)")?;
    let mut a = Vec::new();
    if !cfg.run_config_path.trim().is_empty() {
        a.push("--config".into());
        a.push(cfg.run_config_path.clone());
    }
    a.push("--cache-dir".into());
    a.push(cfg.cache_dir.clone());
    a.push("--transformer".into());
    a.push(cfg.base_model_path.clone());
    a.extend(common_train_flags(cfg, false, true, true));
    Ok(a)
}

/// qwenimage: `--model` checkpoint, no config / batch / offload.
fn qwenimage_args(cfg: &TrainConfig) -> Result<Vec<String>, String> {
    require(&cfg.base_model_path, "base model path (--model)")?;
    require(&cfg.cache_dir, "cache dir (--cache-dir)")?;
    let mut a = vec![
        "--cache-dir".into(),
        cfg.cache_dir.clone(),
        "--model".into(),
        cfg.base_model_path.clone(),
    ];
    a.extend(common_train_flags(cfg, false, true, false));
    Ok(a)
}

/// ideogram4: `--model` = the transformer safetensors FILE. The preset's base
/// path is the diffusers DIR, so resolve `<dir>/transformer/diffusion_pytorch_model.safetensors`;
/// a path already ending in `.safetensors` is used as-is. `--cache-dir` =
/// prepare_ideogram output. No batch/offload; warmup supported.
fn ideogram_args(cfg: &TrainConfig) -> Result<Vec<String>, String> {
    require(&cfg.base_model_path, "base model path (Ideogram-4 dir or transformer .safetensors)")?;
    require(&cfg.cache_dir, "cache dir (--cache-dir)")?;
    let base = cfg.base_model_path.trim_end_matches('/');
    let model = if base.ends_with(".safetensors") {
        base.to_string()
    } else {
        format!("{base}/transformer/diffusion_pytorch_model.safetensors")
    };
    let mut a = vec![
        "--cache-dir".into(),
        cfg.cache_dir.clone(),
        "--model".into(),
        model,
    ];
    a.extend(common_train_flags(cfg, false, true, false));
    Ok(a)
}

/// acestep (audio): `--model` checkpoint, no config / batch / offload.
fn acestep_args(cfg: &TrainConfig) -> Result<Vec<String>, String> {
    require(&cfg.base_model_path, "base model path (--model)")?;
    require(&cfg.cache_dir, "cache dir (--cache-dir)")?;
    let mut a = vec![
        "--cache-dir".into(),
        cfg.cache_dir.clone(),
        "--model".into(),
        cfg.base_model_path.clone(),
    ];
    a.extend(common_train_flags(cfg, false, true, false));
    Ok(a)
}

/// ltx2 (video): `--config` required, NO checkpoint flag (model from config), batch.
fn ltx2_args(cfg: &TrainConfig) -> Result<Vec<String>, String> {
    require(&cfg.run_config_path, "run config path (--config)")?;
    require(&cfg.cache_dir, "cache dir (--cache-dir)")?;
    let mut a = vec![
        "--config".into(),
        cfg.run_config_path.clone(),
        "--cache-dir".into(),
        cfg.cache_dir.clone(),
    ];
    a.extend(common_train_flags(cfg, true, true, false));
    Ok(a)
}

/// slider_klein: `--config` + `--transformer`, batch + offload.
fn slider_klein_args(cfg: &TrainConfig) -> Result<Vec<String>, String> {
    require(&cfg.run_config_path, "run config path (--config)")?;
    require(&cfg.cache_dir, "cache dir (--cache-dir)")?;
    require(&cfg.base_model_path, "base model path (--transformer)")?;
    let mut a = vec![
        "--config".into(),
        cfg.run_config_path.clone(),
        "--cache-dir".into(),
        cfg.cache_dir.clone(),
        "--transformer".into(),
        cfg.base_model_path.clone(),
    ];
    a.extend(common_train_flags(cfg, true, true, true));
    Ok(a)
}

/// asymflow: `--config` + `--transformer` + `--asymflow-adapter` (aux), batch + offload.
fn asymflow_args(cfg: &TrainConfig) -> Result<Vec<String>, String> {
    require(&cfg.run_config_path, "run config path (--config)")?;
    require(&cfg.cache_dir, "cache dir (--cache-dir)")?;
    require(&cfg.base_model_path, "base model path (--transformer)")?;
    require(&cfg.aux_model_path, "aux model path (--asymflow-adapter)")?;
    let mut a = vec![
        "--config".into(),
        cfg.run_config_path.clone(),
        "--cache-dir".into(),
        cfg.cache_dir.clone(),
        "--transformer".into(),
        cfg.base_model_path.clone(),
        "--asymflow-adapter".into(),
        cfg.aux_model_path.clone(),
    ];
    a.extend(common_train_flags(cfg, true, true, true));
    Ok(a)
}

/// wan22 (video, 2-model): `--config` + `--low-noise` (base) [+ `--high-noise` aux]
/// [+ `--vae`], batch + offload.
fn wan22_args(cfg: &TrainConfig) -> Result<Vec<String>, String> {
    require(&cfg.run_config_path, "run config path (--config)")?;
    require(&cfg.cache_dir, "cache dir (--cache-dir)")?;
    require(&cfg.base_model_path, "base model path (--low-noise)")?;
    let mut a = vec![
        "--config".into(),
        cfg.run_config_path.clone(),
        "--cache-dir".into(),
        cfg.cache_dir.clone(),
        "--low-noise".into(),
        cfg.base_model_path.clone(),
    ];
    if !cfg.aux_model_path.trim().is_empty() {
        a.push("--high-noise".into());
        a.push(cfg.aux_model_path.clone());
    }
    if !cfg.vae_override.trim().is_empty() {
        a.push("--vae".into());
        a.push(cfg.vae_override.clone());
    }
    a.extend(common_train_flags(cfg, true, true, true));
    Ok(a)
}

/// u1 (SenseNova U1): different shape — `--model-path` + `--steps` + `--lr`, plus
/// optional `--data-dir` / `--lora-save-to`. No rank / lora-alpha / cache-dir.
fn u1_args(cfg: &TrainConfig) -> Result<Vec<String>, String> {
    require(&cfg.base_model_path, "base model path (--model-path)")?;
    let mut a = vec![
        "--model-path".into(),
        cfg.base_model_path.clone(),
        "--steps".into(),
        (cfg.max_train_steps.max(1.0) as u64).to_string(),
        "--lr".into(),
        cfg.learning_rate.to_string(),
    ];
    if !cfg.dataset_path.trim().is_empty() {
        a.push("--data-dir".into());
        a.push(cfg.dataset_path.clone());
    }
    if !cfg.output_dir.trim().is_empty() {
        a.push("--lora-save-to".into());
        a.push(cfg.output_dir.clone());
    }
    Ok(a)
}

pub fn build_command(cfg: &TrainConfig) -> Result<(String, Vec<String>), String> {
    let args = match cfg.model_type.as_str() {
        "ideogram4" => ideogram_args(cfg)?,
        "klein" => klein_args(cfg)?,
        "sdxl" => sdxl_args(cfg)?,
        "zimage" => zimage_args(cfg)?,
        "chroma" => chroma_args(cfg)?,
        "ernie" => ernie_args(cfg)?,
        "anima" => anima_args(cfg)?,
        // model_type "hidream" -> bin train_hidream_o1 (name reconciliation).
        "hidream" => hidream_args(cfg)?,
        "sd35" => sd35_args(cfg)?,
        "l2p" => l2p_args(cfg)?,
        // --- UNVERIFIED (wired from the CLI; not smoke-tested) ---
        "flux" => flux_args(cfg)?,
        "qwenimage" => qwenimage_args(cfg)?,
        "acestep" => acestep_args(cfg)?,
        "ltx2" => ltx2_args(cfg)?,
        "slider_klein" => slider_klein_args(cfg)?,
        "asymflow" => asymflow_args(cfg)?,
        "wan22" => wan22_args(cfg)?,
        "u1" => u1_args(cfg)?,
        other => {
            return Err(format!(
                "launch for model '{other}' is not wired (18 train_* targets are registered; unknown model_type)"
            ))
        }
    };
    let (program, mut full) = resolve_launcher("train");
    full.push(String::from("--model"));
    full.push(cfg.model_type.clone());
    full.push(String::from("--"));
    full.extend(args);
    full.extend(sample_flags(cfg));
    Ok((program, full))
}

/// In-trainer sampling flags from the Sampling-tab fields. Per-model: the
/// asset-flag names differ, hidream takes ONLY `--sample-every`, and sdxl/anima
/// have NO in-trainer sampling (separate `sample_<m>` bin) so emit nothing.
/// Always emits `--sample-every 0` to DISABLE when sampling is off or a required
/// sample asset is missing — several trainers default `--sample-every` to
/// non-zero, which would otherwise try to sample with no assets and fail.
fn sample_flags(cfg: &TrainConfig) -> Vec<String> {
    // No in-trainer sampling wired here — never emit sample flags. sdxl/anima
    // use a separate sample bin; the newly-wired (unverified) models are not
    // sampling-wired yet (and some lack `--sample-every`, which would clap-reject).
    if matches!(
        cfg.model_type.as_str(),
        "ideogram4" | "sdxl" | "anima" | "flux" | "qwenimage" | "acestep" | "ltx2" | "slider_klein"
            | "asymflow" | "wan22" | "u1"
    ) {
        return Vec::new();
    }
    let off = vec![String::from("--sample-every"), String::from("0")];
    let every = cfg.sample_after.max(0.0) as u64;
    if every == 0 {
        return off; // sampling disabled in the UI
    }
    // HiDream-O1 takes only --sample-every (prompt/assets come from --model-path).
    if cfg.model_type == "hidream" {
        return vec![String::from("--sample-every"), every.to_string()];
    }
    let prompt = cfg
        .samples
        .first()
        .map(|s| s.prompt.trim().to_string())
        .unwrap_or_default();
    if prompt.is_empty() {
        return off; // no sample prompt set
    }
    let vae = cfg.sample_vae_path.trim();
    let enc = cfg.sample_encoder_path.trim();
    let tok = cfg.sample_tokenizer_path.trim();
    let pair = |f: &str, v: &str| vec![f.to_string(), v.to_string()];
    let assets: Vec<String> = match cfg.model_type.as_str() {
        "klein" | "zimage" => {
            if vae.is_empty() || enc.is_empty() || tok.is_empty() {
                return off;
            }
            [pair("--sample-vae", vae), pair("--sample-qwen3", enc), pair("--sample-tokenizer", tok)].concat()
        }
        "chroma" => {
            if vae.is_empty() || enc.is_empty() || tok.is_empty() {
                return off;
            }
            [pair("--sample-vae", vae), pair("--sample-t5", enc), pair("--sample-t5-tokenizer", tok)].concat()
        }
        "ernie" => {
            if vae.is_empty() || enc.is_empty() || tok.is_empty() {
                return off;
            }
            [pair("--sample-vae", vae), pair("--sample-text-ckpt", enc), pair("--sample-tokenizer", tok)].concat()
        }
        "l2p" => {
            if enc.is_empty() || tok.is_empty() {
                return off; // pixel-space: no VAE
            }
            [pair("--sample-qwen3", enc), pair("--sample-tokenizer", tok)].concat()
        }
        "sd35" => {
            // SD3.5: CLIP-L (generic encoder/tokenizer) + CLIP-G + T5 required.
            // VAE is OPTIONAL — train_sd35 falls back to the main checkpoint's
            // VAE when --sample-vae is omitted (verified: cap=[1,154,4096] +
            // checkpoint-VAE decode renders a real 1024 sample).
            let cg = cfg.sample_clip_g_path.trim();
            let cgt = cfg.sample_clip_g_tokenizer_path.trim();
            let t5 = cfg.sample_t5_path.trim();
            let t5t = cfg.sample_t5_tokenizer_path.trim();
            if enc.is_empty()
                || tok.is_empty()
                || cg.is_empty()
                || cgt.is_empty()
                || t5.is_empty()
                || t5t.is_empty()
            {
                return off;
            }
            let mut a = Vec::new();
            if !vae.is_empty() {
                a.extend(pair("--sample-vae", vae));
            }
            a.extend(pair("--sample-clip-l", enc));
            a.extend(pair("--sample-clip-l-tokenizer", tok));
            a.extend(pair("--sample-clip-g", cg));
            a.extend(pair("--sample-clip-g-tokenizer", cgt));
            a.extend(pair("--sample-t5", t5));
            a.extend(pair("--sample-t5-tokenizer", t5t));
            a
        }
        _ => return off,
    };
    let mut a = vec![
        String::from("--sample-every"),
        every.to_string(),
        String::from("--sample-prompt"),
        prompt,
        String::from("--sample-size"),
        (cfg.sample_size.max(64.0) as u64).to_string(),
        String::from("--sample-steps"),
        (cfg.sample_steps.max(1.0) as u64).to_string(),
        String::from("--sample-cfg"),
        cfg.sample_cfg.to_string(),
    ];
    if let Some(s) = cfg.samples.first() {
        if !s.negative_prompt.trim().is_empty() {
            a.push(String::from("--sample-neg-prompt"));
            a.push(s.negative_prompt.clone());
        }
        a.push(String::from("--sample-seed"));
        a.push((s.seed.max(0) as u64).to_string());
    }
    a.extend(assets);
    a
}

/// Klein argv mirroring `train_klein.rs` clap flags. Fails loud on any missing
/// required path (`--config`, `--cache-dir`, `--transformer`).
fn klein_args(cfg: &TrainConfig) -> Result<Vec<String>, String> {
    if cfg.run_config_path.trim().is_empty() {
        return Err(String::from("run config path (--config) is required"));
    }
    if cfg.cache_dir.trim().is_empty() {
        return Err(String::from("cache dir (--cache-dir) is required"));
    }
    if cfg.base_model_path.trim().is_empty() {
        return Err(String::from("base model path (--transformer) is required"));
    }
    let mut a: Vec<String> = vec![
        "--config".into(),
        cfg.run_config_path.clone(),
        "--cache-dir".into(),
        cfg.cache_dir.clone(),
        "--transformer".into(),
        cfg.base_model_path.clone(),
        "--steps".into(),
        (cfg.max_train_steps.max(1.0) as u64).to_string(),
        "--rank".into(),
        (cfg.lora_rank.max(1.0) as u64).to_string(),
        "--lora-alpha".into(),
        cfg.lora_alpha.to_string(),
        "--lr".into(),
        cfg.learning_rate.to_string(),
        "--warmup-steps".into(),
        (cfg.learning_rate_warmup_steps.max(0.0) as u64).to_string(),
        "--batch-size".into(),
        (cfg.batch_size.max(1.0) as u64).to_string(),
    ];
    if !cfg.output_dir.trim().is_empty() {
        a.push("--output-dir".into());
        a.push(cfg.output_dir.clone());
    }
    if cfg.activation_offloading {
        a.push("--offload".into());
    }
    if cfg.min_snr_gamma_flow > 0.0 {
        a.push("--min-snr-gamma".into());
        a.push(cfg.min_snr_gamma_flow.to_string());
    }
    if cfg.caption_dropout > 0.0 {
        a.push("--caption-dropout-probability".into());
        a.push(cfg.caption_dropout.to_string());
    }
    Ok(a)
}

/// SDXL argv mirroring `train_sdxl.rs` clap flags. Differs from Klein: the
/// checkpoint flag is `--unet` (not `--transformer`), `--config` is OPTIONAL,
/// and there is NO `--batch-size` / `--offload` flag (passing them would make
/// clap reject the launch). Fails loud on missing `--cache-dir` / `--unet`.
fn sdxl_args(cfg: &TrainConfig) -> Result<Vec<String>, String> {
    if cfg.cache_dir.trim().is_empty() {
        return Err(String::from("cache dir (--cache-dir) is required"));
    }
    if cfg.base_model_path.trim().is_empty() {
        return Err(String::from("base model path (--unet) is required"));
    }
    let mut a: Vec<String> = Vec::new();
    // --config is optional for SDXL (defaults to TrainConfig::default()).
    if !cfg.run_config_path.trim().is_empty() {
        a.push("--config".into());
        a.push(cfg.run_config_path.clone());
    }
    a.push("--cache-dir".into());
    a.push(cfg.cache_dir.clone());
    a.push("--unet".into());
    a.push(cfg.base_model_path.clone());
    a.push("--steps".into());
    a.push((cfg.max_train_steps.max(1.0) as u64).to_string());
    a.push("--rank".into());
    a.push((cfg.lora_rank.max(1.0) as u64).to_string());
    a.push("--lora-alpha".into());
    a.push(cfg.lora_alpha.to_string());
    a.push("--lr".into());
    a.push(cfg.learning_rate.to_string());
    a.push("--warmup-steps".into());
    a.push((cfg.learning_rate_warmup_steps.max(0.0) as u64).to_string());
    if !cfg.output_dir.trim().is_empty() {
        a.push("--output-dir".into());
        a.push(cfg.output_dir.clone());
    }
    if cfg.min_snr_gamma_flow > 0.0 {
        a.push("--min-snr-gamma".into());
        a.push(cfg.min_snr_gamma_flow.to_string());
    }
    if cfg.caption_dropout > 0.0 {
        a.push("--caption-dropout-probability".into());
        a.push(cfg.caption_dropout.to_string());
    }
    Ok(a)
}

/// Z-Image argv mirroring `train_zimage.rs` clap flags. Checkpoint flag is
/// `--model` (a directory of DiT shards); `--config` is OPTIONAL; HAS
/// `--batch-size` and `--warmup-steps`. Fails loud on missing `--cache-dir` /
/// `--model`.
fn zimage_args(cfg: &TrainConfig) -> Result<Vec<String>, String> {
    if cfg.cache_dir.trim().is_empty() {
        return Err(String::from("cache dir (--cache-dir) is required"));
    }
    if cfg.base_model_path.trim().is_empty() {
        return Err(String::from("base model path (--model) is required"));
    }
    let mut a: Vec<String> = Vec::new();
    if !cfg.run_config_path.trim().is_empty() {
        a.push("--config".into());
        a.push(cfg.run_config_path.clone());
    }
    a.push("--cache-dir".into());
    a.push(cfg.cache_dir.clone());
    a.push("--model".into());
    a.push(cfg.base_model_path.clone());
    a.push("--steps".into());
    a.push((cfg.max_train_steps.max(1.0) as u64).to_string());
    a.push("--rank".into());
    a.push((cfg.lora_rank.max(1.0) as u64).to_string());
    a.push("--lora-alpha".into());
    a.push(cfg.lora_alpha.to_string());
    a.push("--lr".into());
    a.push(cfg.learning_rate.to_string());
    a.push("--batch-size".into());
    a.push((cfg.batch_size.max(1.0) as u64).to_string());
    a.push("--warmup-steps".into());
    a.push((cfg.learning_rate_warmup_steps.max(0.0) as u64).to_string());
    if !cfg.output_dir.trim().is_empty() {
        a.push("--output-dir".into());
        a.push(cfg.output_dir.clone());
    }
    if cfg.min_snr_gamma_flow > 0.0 {
        a.push("--min-snr-gamma".into());
        a.push(cfg.min_snr_gamma_flow.to_string());
    }
    if cfg.caption_dropout > 0.0 {
        a.push("--caption-dropout-probability".into());
        a.push(cfg.caption_dropout.to_string());
    }
    Ok(a)
}

/// Chroma argv mirroring `train_chroma.rs` clap flags. Checkpoint flag is
/// `--transformer` (like Klein); `--config` is OPTIONAL; supports `--offload`;
/// has NO `--batch-size` (unlike Klein/Z-Image). Fails loud on missing
/// `--cache-dir` / `--transformer`.
fn chroma_args(cfg: &TrainConfig) -> Result<Vec<String>, String> {
    if cfg.cache_dir.trim().is_empty() {
        return Err(String::from("cache dir (--cache-dir) is required"));
    }
    if cfg.base_model_path.trim().is_empty() {
        return Err(String::from("base model path (--transformer) is required"));
    }
    let mut a: Vec<String> = Vec::new();
    if !cfg.run_config_path.trim().is_empty() {
        a.push("--config".into());
        a.push(cfg.run_config_path.clone());
    }
    a.push("--cache-dir".into());
    a.push(cfg.cache_dir.clone());
    a.push("--transformer".into());
    a.push(cfg.base_model_path.clone());
    a.push("--steps".into());
    a.push((cfg.max_train_steps.max(1.0) as u64).to_string());
    a.push("--rank".into());
    a.push((cfg.lora_rank.max(1.0) as u64).to_string());
    a.push("--lora-alpha".into());
    a.push(cfg.lora_alpha.to_string());
    a.push("--lr".into());
    a.push(cfg.learning_rate.to_string());
    a.push("--warmup-steps".into());
    a.push((cfg.learning_rate_warmup_steps.max(0.0) as u64).to_string());
    if !cfg.output_dir.trim().is_empty() {
        a.push("--output-dir".into());
        a.push(cfg.output_dir.clone());
    }
    if cfg.activation_offloading {
        a.push("--offload".into());
    }
    if cfg.min_snr_gamma_flow > 0.0 {
        a.push("--min-snr-gamma".into());
        a.push(cfg.min_snr_gamma_flow.to_string());
    }
    if cfg.caption_dropout > 0.0 {
        a.push("--caption-dropout-probability".into());
        a.push(cfg.caption_dropout.to_string());
    }
    Ok(a)
}

/// Ernie argv mirroring `train_ernie.rs` clap flags. DIFFERENT shape: there is
/// NO checkpoint flag — `train_ernie` reads the model path from `--config`'s
/// `base_model_name`, so `--config` is REQUIRED and `base_model_path` is NOT a
/// CLI arg here (set it inside the run-config JSON instead). No `--batch-size`,
/// no `--offload`.
fn ernie_args(cfg: &TrainConfig) -> Result<Vec<String>, String> {
    if cfg.run_config_path.trim().is_empty() {
        return Err(String::from(
            "run config path (--config) is required for Ernie (it reads the model path from the config's base_model_name)",
        ));
    }
    if cfg.cache_dir.trim().is_empty() {
        return Err(String::from("cache dir (--cache-dir) is required"));
    }
    let mut a: Vec<String> = vec![
        "--config".into(),
        cfg.run_config_path.clone(),
        "--cache-dir".into(),
        cfg.cache_dir.clone(),
        "--steps".into(),
        (cfg.max_train_steps.max(1.0) as u64).to_string(),
        "--rank".into(),
        (cfg.lora_rank.max(1.0) as u64).to_string(),
        "--lora-alpha".into(),
        cfg.lora_alpha.to_string(),
        "--lr".into(),
        cfg.learning_rate.to_string(),
        "--warmup-steps".into(),
        (cfg.learning_rate_warmup_steps.max(0.0) as u64).to_string(),
    ];
    if !cfg.output_dir.trim().is_empty() {
        a.push("--output-dir".into());
        a.push(cfg.output_dir.clone());
    }
    if cfg.min_snr_gamma_flow > 0.0 {
        a.push("--min-snr-gamma".into());
        a.push(cfg.min_snr_gamma_flow.to_string());
    }
    if cfg.caption_dropout > 0.0 {
        a.push("--caption-dropout-probability".into());
        a.push(cfg.caption_dropout.to_string());
    }
    Ok(a)
}

/// Anima argv mirroring `train_anima.rs` clap flags. Checkpoint flag is
/// `--dit-path`; `--config` is ALSO required (carries dataset/recipe). No
/// `--batch-size`, no `--offload`. Fails loud on missing `--config` /
/// `--cache-dir` / `--dit-path`.
fn anima_args(cfg: &TrainConfig) -> Result<Vec<String>, String> {
    if cfg.run_config_path.trim().is_empty() {
        return Err(String::from("run config path (--config) is required"));
    }
    if cfg.cache_dir.trim().is_empty() {
        return Err(String::from("cache dir (--cache-dir) is required"));
    }
    if cfg.base_model_path.trim().is_empty() {
        return Err(String::from("base model path (--dit-path) is required"));
    }
    let mut a: Vec<String> = vec![
        "--config".into(),
        cfg.run_config_path.clone(),
        "--cache-dir".into(),
        cfg.cache_dir.clone(),
        "--dit-path".into(),
        cfg.base_model_path.clone(),
        "--steps".into(),
        (cfg.max_train_steps.max(1.0) as u64).to_string(),
        "--rank".into(),
        (cfg.lora_rank.max(1.0) as u64).to_string(),
        "--lora-alpha".into(),
        cfg.lora_alpha.to_string(),
        "--lr".into(),
        cfg.learning_rate.to_string(),
        "--warmup-steps".into(),
        (cfg.learning_rate_warmup_steps.max(0.0) as u64).to_string(),
    ];
    if !cfg.output_dir.trim().is_empty() {
        a.push("--output-dir".into());
        a.push(cfg.output_dir.clone());
    }
    if cfg.min_snr_gamma_flow > 0.0 {
        a.push("--min-snr-gamma".into());
        a.push(cfg.min_snr_gamma_flow.to_string());
    }
    if cfg.caption_dropout > 0.0 {
        a.push("--caption-dropout-probability".into());
        a.push(cfg.caption_dropout.to_string());
    }
    Ok(a)
}

/// HiDream-O1 argv mirroring `train_hidream_o1.rs` clap flags. Checkpoint flag
/// is `--model-path` (the full model dir); `--config` OPTIONAL. Has NO
/// `--warmup-steps`, NO `--min-snr-gamma`, NO `--batch-size`, NO `--offload`
/// (passing any would make clap reject). Model_type "hidream" maps to the bin
/// `train_hidream_o1`.
fn hidream_args(cfg: &TrainConfig) -> Result<Vec<String>, String> {
    if cfg.cache_dir.trim().is_empty() {
        return Err(String::from("cache dir (--cache-dir) is required"));
    }
    if cfg.base_model_path.trim().is_empty() {
        return Err(String::from("base model path (--model-path) is required"));
    }
    let mut a: Vec<String> = Vec::new();
    if !cfg.run_config_path.trim().is_empty() {
        a.push("--config".into());
        a.push(cfg.run_config_path.clone());
    }
    a.push("--cache-dir".into());
    a.push(cfg.cache_dir.clone());
    a.push("--model-path".into());
    a.push(cfg.base_model_path.clone());
    a.push("--steps".into());
    a.push((cfg.max_train_steps.max(1.0) as u64).to_string());
    a.push("--rank".into());
    a.push((cfg.lora_rank.max(1.0) as u64).to_string());
    a.push("--lora-alpha".into());
    a.push(cfg.lora_alpha.to_string());
    a.push("--lr".into());
    a.push(cfg.learning_rate.to_string());
    if !cfg.output_dir.trim().is_empty() {
        a.push("--output-dir".into());
        a.push(cfg.output_dir.clone());
    }
    if cfg.caption_dropout > 0.0 {
        a.push("--caption-dropout-probability".into());
        a.push(cfg.caption_dropout.to_string());
    }
    Ok(a)
}

/// SD3.5 argv mirroring `train_sd35.rs` clap flags. Checkpoint flag is
/// `--transformer`; `--config` is REQUIRED (base TrainConfig); HAS `--batch-size`
/// + `--warmup-steps` + `--min-snr-gamma`; NO `--offload`. Fails loud on missing
/// `--config` / `--cache-dir` / `--transformer`.
fn sd35_args(cfg: &TrainConfig) -> Result<Vec<String>, String> {
    if cfg.run_config_path.trim().is_empty() {
        return Err(String::from("run config path (--config) is required"));
    }
    if cfg.cache_dir.trim().is_empty() {
        return Err(String::from("cache dir (--cache-dir) is required"));
    }
    if cfg.base_model_path.trim().is_empty() {
        return Err(String::from("base model path (--transformer) is required"));
    }
    let mut a: Vec<String> = vec![
        "--config".into(),
        cfg.run_config_path.clone(),
        "--cache-dir".into(),
        cfg.cache_dir.clone(),
        "--transformer".into(),
        cfg.base_model_path.clone(),
        "--steps".into(),
        (cfg.max_train_steps.max(1.0) as u64).to_string(),
        "--rank".into(),
        (cfg.lora_rank.max(1.0) as u64).to_string(),
        "--lora-alpha".into(),
        cfg.lora_alpha.to_string(),
        "--lr".into(),
        cfg.learning_rate.to_string(),
        "--batch-size".into(),
        (cfg.batch_size.max(1.0) as u64).to_string(),
        "--warmup-steps".into(),
        (cfg.learning_rate_warmup_steps.max(0.0) as u64).to_string(),
    ];
    if !cfg.output_dir.trim().is_empty() {
        a.push("--output-dir".into());
        a.push(cfg.output_dir.clone());
    }
    if cfg.min_snr_gamma_flow > 0.0 {
        a.push("--min-snr-gamma".into());
        a.push(cfg.min_snr_gamma_flow.to_string());
    }
    if cfg.caption_dropout > 0.0 {
        a.push("--caption-dropout-probability".into());
        a.push(cfg.caption_dropout.to_string());
    }
    Ok(a)
}

/// Z-Image L2P (pixel-space) argv mirroring `train_l2p.rs` clap flags. UNUSUAL
/// flag names: `--model` (checkpoint), `--cache` (NOT --cache-dir), `--output`
/// (NOT --output-dir, REQUIRED), `--lora-rank` (NOT --rank), `--train-shift`
/// (the trainer is compiled for a fixed shift and fails loud on a mismatch).
/// `--config` optional; NO --batch-size/--warmup-steps/--offload/--min-snr-gamma.
fn l2p_args(cfg: &TrainConfig) -> Result<Vec<String>, String> {
    if cfg.base_model_path.trim().is_empty() {
        return Err(String::from("base model path (--model) is required"));
    }
    if cfg.cache_dir.trim().is_empty() {
        return Err(String::from("cache dir (--cache) is required"));
    }
    if cfg.output_dir.trim().is_empty() {
        return Err(String::from("output dir (--output) is required (l2p has no default)"));
    }
    let mut a: Vec<String> = vec![
        "--model".into(),
        cfg.base_model_path.clone(),
        "--cache".into(),
        cfg.cache_dir.clone(),
        "--output".into(),
        cfg.output_dir.clone(),
        "--steps".into(),
        (cfg.max_train_steps.max(1.0) as u64).to_string(),
        "--lr".into(),
        cfg.learning_rate.to_string(),
        "--lora-rank".into(),
        (cfg.lora_rank.max(1.0) as u64).to_string(),
        "--lora-alpha".into(),
        cfg.lora_alpha.to_string(),
        "--train-shift".into(),
        cfg.timestep_shift.to_string(),
    ];
    // The L2P checkpoint is ~19.5GB; grad checkpointing is needed to fit a 24GB
    // card. The preset enables it by default; the UI toggle maps here.
    if cfg.gradient_checkpointing {
        a.push("--grad-checkpoint".into());
    }
    if !cfg.run_config_path.trim().is_empty() {
        a.push("--config".into());
        a.push(cfg.run_config_path.clone());
    }
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Sample;

    #[test]
    fn cargo_fallback_uses_current_profile() {
        let args = cargo_run_args(Path::new("/tmp/edv2/Cargo.toml"), "train");

        if cfg!(debug_assertions) {
            assert!(!args.iter().any(|arg| arg == "--release"));
        } else {
            assert!(args.iter().any(|arg| arg == "--release"));
        }
        assert!(args.iter().any(|arg| arg == "train"));
    }

    #[test]
    fn parses_full_klein_progress_line() {
        let line = "[klein-lora] step 107/500 | epoch 2/10 | loss 0.0421 | grad_norm 0.1730 | 4.5s/step | elapsed 0:08:01 | ETA 0:29:27";
        let mut s = LiveStats::default();
        assert!(parse_progress_line(line, &mut s));
        assert_eq!(s.step, 107);
        assert_eq!(s.total_steps, 500);
        assert_eq!(s.epoch, 2);
        assert_eq!(s.total_epochs, 10);
        assert!((s.loss - 0.0421).abs() < 1e-6, "loss={}", s.loss);
        assert!((s.grad_norm - 0.1730).abs() < 1e-6, "grad={}", s.grad_norm);
        assert!((s.speed_s_step - 4.5).abs() < 1e-6, "speed={}", s.speed_s_step);
        assert_eq!(s.elapsed_secs, 8 * 60 + 1);
        assert_eq!(s.eta_secs, 29 * 60 + 27);
    }

    #[test]
    fn parses_line_without_tag_prefix() {
        let line = "step 1/10 | epoch 1/1 | loss 1.2345 | grad_norm 0.5000 | 0.3s/step | elapsed 0:00:00 | ETA 0:00:03";
        let mut s = LiveStats::default();
        assert!(parse_progress_line(line, &mut s));
        assert_eq!(s.step, 1);
        assert_eq!(s.total_steps, 10);
        assert!((s.loss - 1.2345).abs() < 1e-6);
    }

    #[test]
    fn ignores_non_progress_stdout() {
        let mut s = LiveStats::default();
        // An OT_DEBUG diagnostic line — must NOT be mistaken for progress.
        assert!(!parse_progress_line(
            "[OT_DEBUG step=5] grad_norm_pre_clip=1.2e-3",
            &mut s
        ));
        assert_eq!(s.step, 0);
        assert!(!parse_progress_line("Training complete: 1000 steps", &mut s));
    }

    #[test]
    fn progress_fraction_clamps() {
        let mut rt = Runtime::default();
        rt.live.step = 250;
        rt.live.total_steps = 1000;
        assert!((rt.progress_fraction() - 0.25).abs() < 1e-6);
        rt.live.total_steps = 0;
        assert_eq!(rt.progress_fraction(), 0.0);
    }

    #[test]
    fn parses_real_train_klein_run_line() {
        // Captured verbatim from a real `target/release/train_klein` run
        // (3-step smoke on the eri2_klein9b_512 cache, 2026-06-19).
        let line = "[Klein-lora] step 1/3 | epoch 1/1 | loss 1.0703 | grad_norm 0.0076 | 5.3s/step | elapsed 0:00:05 | ETA 0:00:10";
        let mut s = LiveStats::default();
        assert!(parse_progress_line(line, &mut s));
        assert_eq!(s.step, 1);
        assert_eq!(s.total_steps, 3);
        assert_eq!(s.epoch, 1);
        assert!((s.loss - 1.0703).abs() < 1e-6, "loss={}", s.loss);
        assert!((s.grad_norm - 0.0076).abs() < 1e-6, "grad={}", s.grad_norm);
        assert!((s.speed_s_step - 5.3).abs() < 1e-6, "speed={}", s.speed_s_step);
        assert_eq!(s.eta_secs, 10);
    }

    #[test]
    fn parses_real_train_l2p_run_line() {
        // Captured verbatim from a real `target/release/train_l2p` run
        // (3-step smoke on an 8-sample 512px pixel cache, 2026-06-19).
        let line = "[L2P-lora] step 1/3 | epoch 1/1 | loss 0.0145 | grad_norm 0.0009 | 6.0s/step | elapsed 0:00:06 | ETA 0:00:12";
        let mut s = LiveStats::default();
        assert!(parse_progress_line(line, &mut s));
        assert_eq!(s.step, 1);
        assert_eq!(s.total_steps, 3);
        assert!((s.loss - 0.0145).abs() < 1e-6, "loss={}", s.loss);
        assert!((s.speed_s_step - 6.0).abs() < 1e-6, "speed={}", s.speed_s_step);
        assert_eq!(s.eta_secs, 12);
    }

    #[test]
    fn parses_real_train_sd35_run_line() {
        // Captured verbatim from a real `target/release/train_sd35` run
        // (3-step smoke on an 8-sample 1024px cache, 2026-06-19). Tag has a dot.
        let line = "[SD3.5-lora] step 1/3 | epoch 1/1 | loss 0.7028 | grad_norm 0.0108 | 4.6s/step | elapsed 0:00:04 | ETA 0:00:09";
        let mut s = LiveStats::default();
        assert!(parse_progress_line(line, &mut s));
        assert_eq!(s.step, 1);
        assert_eq!(s.total_steps, 3);
        assert!((s.loss - 0.7028).abs() < 1e-6, "loss={}", s.loss);
        assert!((s.grad_norm - 0.0108).abs() < 1e-6, "grad={}", s.grad_norm);
        assert!((s.speed_s_step - 4.6).abs() < 1e-6, "speed={}", s.speed_s_step);
        assert_eq!(s.eta_secs, 9);
    }

    #[test]
    fn parses_real_train_hidream_o1_run_line() {
        // Captured verbatim from a real `target/release/train_hidream_o1` run
        // (3-step smoke on an 8-sample 512px cache, 2026-06-19).
        let line = "[HiDreamO1-lora] step 1/3 | epoch 1/1 | loss 0.2927 | grad_norm 0.2897 | 3.5s/step | elapsed 0:00:03 | ETA 0:00:06";
        let mut s = LiveStats::default();
        assert!(parse_progress_line(line, &mut s));
        assert_eq!(s.step, 1);
        assert_eq!(s.total_steps, 3);
        assert!((s.loss - 0.2927).abs() < 1e-6, "loss={}", s.loss);
        assert!((s.grad_norm - 0.2897).abs() < 1e-6, "grad={}", s.grad_norm);
        assert!((s.speed_s_step - 3.5).abs() < 1e-6, "speed={}", s.speed_s_step);
        assert_eq!(s.eta_secs, 6);
    }

    #[test]
    fn parses_real_train_anima_run_line() {
        // Captured verbatim from a real `target/release/train_anima` run
        // (3-step smoke on an 8-sample 512px cache, 2026-06-19).
        let line = "[anima-lora] step 1/3 | epoch 1/1 | loss 0.0821 | grad_norm 0.2295 | 4.3s/step | elapsed 0:00:04 | ETA 0:00:08";
        let mut s = LiveStats::default();
        assert!(parse_progress_line(line, &mut s));
        assert_eq!(s.step, 1);
        assert_eq!(s.total_steps, 3);
        assert!((s.loss - 0.0821).abs() < 1e-6, "loss={}", s.loss);
        assert!((s.grad_norm - 0.2295).abs() < 1e-6, "grad={}", s.grad_norm);
        assert!((s.speed_s_step - 4.3).abs() < 1e-6, "speed={}", s.speed_s_step);
        assert_eq!(s.eta_secs, 8);
    }

    #[test]
    fn parses_real_train_ernie_run_line() {
        // Captured verbatim from a real `target/release/train_ernie` run
        // (3-step smoke on an 8-sample 512px cache, 2026-06-19).
        let line = "[ERNIE-lora] step 1/3 | epoch 1/1 | loss 0.9406 | grad_norm 0.0021 | 3.4s/step | elapsed 0:00:03 | ETA 0:00:06";
        let mut s = LiveStats::default();
        assert!(parse_progress_line(line, &mut s));
        assert_eq!(s.step, 1);
        assert_eq!(s.total_steps, 3);
        assert!((s.loss - 0.9406).abs() < 1e-6, "loss={}", s.loss);
        assert!((s.grad_norm - 0.0021).abs() < 1e-6, "grad={}", s.grad_norm);
        assert!((s.speed_s_step - 3.4).abs() < 1e-6, "speed={}", s.speed_s_step);
        assert_eq!(s.eta_secs, 6);
    }

    #[test]
    fn parses_real_train_chroma_run_line() {
        // Captured verbatim from a real `target/release/train_chroma` run
        // (3-step smoke on an 8-sample 512px cache, 2026-06-19).
        let line = "[Chroma-lora] step 1/3 | epoch 1/1 | loss 0.4617 | grad_norm 0.0061 | 8.5s/step | elapsed 0:00:08 | ETA 0:00:16";
        let mut s = LiveStats::default();
        assert!(parse_progress_line(line, &mut s));
        assert_eq!(s.step, 1);
        assert_eq!(s.total_steps, 3);
        assert!((s.loss - 0.4617).abs() < 1e-6, "loss={}", s.loss);
        assert!((s.grad_norm - 0.0061).abs() < 1e-6, "grad={}", s.grad_norm);
        assert!((s.speed_s_step - 8.5).abs() < 1e-6, "speed={}", s.speed_s_step);
        assert_eq!(s.eta_secs, 16);
    }

    #[test]
    fn parses_real_train_zimage_run_line() {
        // Captured verbatim from a real `target/release/train_zimage` run
        // (3-step smoke on an 8-sample 512px cache, 2026-06-19).
        let line = "[Z-Image-lora] step 1/3 | epoch 1/1 | loss 0.4745 | grad_norm 0.0001 | 4.2s/step | elapsed 0:00:04 | ETA 0:00:08";
        let mut s = LiveStats::default();
        assert!(parse_progress_line(line, &mut s));
        assert_eq!(s.step, 1);
        assert_eq!(s.total_steps, 3);
        assert!((s.loss - 0.4745).abs() < 1e-6, "loss={}", s.loss);
        assert!((s.speed_s_step - 4.2).abs() < 1e-6, "speed={}", s.speed_s_step);
        assert_eq!(s.eta_secs, 8);
    }

    #[test]
    fn parses_real_train_sdxl_run_line() {
        // Captured verbatim from a real `target/release/train_sdxl` run
        // (3-step smoke on an 8-sample 512px cache, 2026-06-19).
        let line = "[SDXL-lora] step 1/3 | epoch 1/1 | loss 0.1063 | grad_norm 0.0371 | 7.7s/step | elapsed 0:00:07 | ETA 0:00:15";
        let mut s = LiveStats::default();
        assert!(parse_progress_line(line, &mut s));
        assert_eq!(s.step, 1);
        assert_eq!(s.total_steps, 3);
        assert!((s.loss - 0.1063).abs() < 1e-6, "loss={}", s.loss);
        assert!((s.grad_norm - 0.0371).abs() < 1e-6, "grad={}", s.grad_norm);
        assert!((s.speed_s_step - 7.7).abs() < 1e-6, "speed={}", s.speed_s_step);
        assert_eq!(s.eta_secs, 15);
    }

    #[test]
    fn klein_args_builds_expected_flags() {
        let mut cfg = TrainConfig::default();
        cfg.architecture_index = 1;
        cfg.apply_model_preset(false); // klein
        cfg.run_config_path = "/cfg/klein9b_alina.json".into();
        cfg.cache_dir = "/cache/klein".into();
        cfg.base_model_path = "/models/klein9b".into();
        cfg.max_train_steps = 1000.0;
        cfg.lora_rank = 16.0;
        cfg.output_dir = "/out".into();
        let joined = klein_args(&cfg).expect("klein args ok").join(" ");
        assert!(joined.contains("--config /cfg/klein9b_alina.json"), "{joined}");
        assert!(joined.contains("--cache-dir /cache/klein"), "{joined}");
        assert!(joined.contains("--transformer /models/klein9b"), "{joined}");
        assert!(joined.contains("--steps 1000"), "{joined}");
        assert!(joined.contains("--rank 16"), "{joined}");
        assert!(joined.contains("--output-dir /out"), "{joined}");
    }

    #[test]
    fn klein_args_fails_loud_on_missing_paths() {
        let cfg = TrainConfig::default(); // empty config/cache/transformer paths
        assert!(klein_args(&cfg).is_err());
    }

    #[test]
    fn build_command_rejects_unwired_model() {
        let mut cfg = TrainConfig::default();
        cfg.model_type = "wan22".into(); // genuinely not wired (video, deferred)
        assert!(build_command(&cfg).is_err());
    }

    #[test]
    fn sample_flags_disabled_emits_off() {
        let mut cfg = TrainConfig::default();
        cfg.model_type = "klein".into();
        cfg.sample_after = 0.0;
        assert_eq!(sample_flags(&cfg), vec!["--sample-every", "0"]);
    }

    #[test]
    fn sample_flags_klein_enabled_with_assets() {
        let mut cfg = TrainConfig::default();
        cfg.model_type = "klein".into();
        cfg.sample_after = 200.0;
        cfg.sample_vae_path = "/vae.safetensors".into();
        cfg.sample_encoder_path = "/qwen3".into();
        cfg.sample_tokenizer_path = "/tok.json".into();
        cfg.samples = vec![crate::config::Sample {
            prompt: "a cat".into(),
            negative_prompt: String::new(),
            seed: 42,
        }];
        let j = sample_flags(&cfg).join(" ");
        assert!(j.contains("--sample-every 200"), "{j}");
        assert!(j.contains("--sample-prompt a cat"), "{j}");
        assert!(j.contains("--sample-vae /vae.safetensors"), "{j}");
        assert!(j.contains("--sample-qwen3 /qwen3"), "{j}");
        assert!(j.contains("--sample-tokenizer /tok.json"), "{j}");
    }

    #[test]
    fn sample_flags_enabled_but_missing_assets_disables() {
        let mut cfg = TrainConfig::default();
        cfg.model_type = "klein".into();
        cfg.sample_after = 200.0; // on, but no asset paths set
        cfg.samples = vec![crate::config::Sample {
            prompt: "a cat".into(),
            negative_prompt: String::new(),
            seed: 1,
        }];
        assert_eq!(sample_flags(&cfg), vec!["--sample-every", "0"]);
    }

    #[test]
    fn sample_flags_hidream_only_every() {
        let mut cfg = TrainConfig::default();
        cfg.model_type = "hidream".into();
        cfg.sample_after = 100.0;
        assert_eq!(sample_flags(&cfg), vec!["--sample-every", "100"]);
    }

    #[test]
    fn sample_flags_sdxl_and_anima_emit_nothing() {
        let mut cfg = TrainConfig::default();
        cfg.sample_after = 100.0;
        for m in ["sdxl", "anima"] {
            cfg.model_type = m.into();
            assert!(sample_flags(&cfg).is_empty(), "{m} must emit no --sample-* flags");
        }
    }

    #[test]
    fn sd35_sample_flags_need_encoders_vae_optional() {
        let mut cfg = TrainConfig::default();
        cfg.model_type = "sd35".into();
        cfg.sample_after = 100.0;
        cfg.samples = vec![Sample {
            prompt: "a cat".into(),
            negative_prompt: String::new(),
            seed: 0,
        }];
        cfg.sample_encoder_path = "/clipl".into();
        cfg.sample_tokenizer_path = "/clipl_tok".into();
        cfg.sample_clip_g_path = "/clipg".into();
        cfg.sample_clip_g_tokenizer_path = "/clipg_tok".into();
        cfg.sample_t5_path = "/t5".into();
        cfg.sample_t5_tokenizer_path = "/t5_tok".into();
        // VAE omitted -> still samples (trainer falls back to the checkpoint VAE).
        let j = sample_flags(&cfg).join(" ");
        for f in [
            "--sample-clip-l /clipl",
            "--sample-clip-l-tokenizer /clipl_tok",
            "--sample-clip-g /clipg",
            "--sample-clip-g-tokenizer /clipg_tok",
            "--sample-t5 /t5",
            "--sample-t5-tokenizer /t5_tok",
        ] {
            assert!(j.contains(f), "missing `{f}` in `{j}`");
        }
        assert!(!j.contains("--sample-vae"), "vae must be omitted when unset: {j}");
        // VAE set -> emitted.
        cfg.sample_vae_path = "/vae".into();
        assert!(sample_flags(&cfg).join(" ").contains("--sample-vae /vae"));
        // Drop a required encoder -> sampling disabled (fail-closed).
        cfg.sample_t5_path = String::new();
        assert_eq!(
            sample_flags(&cfg),
            vec!["--sample-every".to_string(), "0".to_string()]
        );
    }

    #[test]
    fn sample_flags_new_models_emit_nothing() {
        let mut cfg = TrainConfig::default();
        cfg.sample_after = 100.0;
        for m in ["flux", "qwenimage", "acestep", "ltx2", "slider_klein", "asymflow", "wan22", "u1"] {
            cfg.model_type = m.into();
            assert!(sample_flags(&cfg).is_empty(), "{m} must not emit sample flags (unverified)");
        }
    }

    fn with_paths(model: &str) -> TrainConfig {
        let mut cfg = TrainConfig::default();
        cfg.model_type = model.into();
        cfg.base_model_path = "/m.safetensors".into();
        cfg.cache_dir = "/c".into();
        cfg.run_config_path = "/cfg.json".into();
        cfg.aux_model_path = "/aux.safetensors".into();
        cfg.output_dir = "/out".into();
        cfg
    }

    #[test]
    fn ideogram_args_resolves_transformer_file_and_flags() {
        // dir base path -> resolve to the transformer .safetensors file
        let mut cfg = TrainConfig::default();
        cfg.model_type = "ideogram4".into();
        cfg.base_model_path = "/home/alex/.serenity/models/ideogram-4-fp8".into();
        cfg.cache_dir = "/cache".into();
        cfg.output_dir = "/out".into();
        let j = ideogram_args(&cfg).unwrap().join(" ");
        assert!(
            j.contains("--model /home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"),
            "{j}"
        );
        assert!(j.contains("--cache-dir /cache") && j.contains("--steps") && j.contains("--rank"));
        // an explicit .safetensors path is used as-is
        cfg.base_model_path = "/m.safetensors".into();
        assert!(ideogram_args(&cfg).unwrap().join(" ").contains("--model /m.safetensors"));
        // build_command routes through the unified trainer entrypoint.
        cfg.base_model_path = "/home/alex/.serenity/models/ideogram-4-fp8".into();
        let (program, full) = build_command(&cfg).unwrap();
        let cmd = format!("{program} {}", full.join(" "));
        let direct_train = std::path::Path::new(&program)
            .file_name()
            .and_then(|name| name.to_str())
            == Some("train");
        let cargo_train = program == "cargo"
            && full
                .windows(2)
                .any(|pair| pair[0] == "--bin" && pair[1] == "train");
        assert!(
            direct_train || cargo_train,
            "build_command missing unified train launcher: {cmd}"
        );
        assert!(
            cmd.contains("--model ideogram4"),
            "build_command missing ideogram model dispatch: {cmd}"
        );
        // verified + no sample flags
        assert!(crate::config::model_verified("ideogram4"));
        cfg.sample_after = 100.0;
        assert!(sample_flags(&cfg).is_empty());
    }

    #[test]
    fn new_models_checkpoint_flags() {
        assert!(flux_args(&with_paths("flux")).unwrap().join(" ").contains("--transformer /m.safetensors"));
        assert!(qwenimage_args(&with_paths("qwenimage")).unwrap().join(" ").contains("--model /m.safetensors"));
        assert!(acestep_args(&with_paths("acestep")).unwrap().join(" ").contains("--model /m.safetensors"));
        // ltx2: config-only, no checkpoint flag
        let l = ltx2_args(&with_paths("ltx2")).unwrap().join(" ");
        assert!(l.contains("--config /cfg.json") && !l.contains("--transformer"), "{l}");
        assert!(slider_klein_args(&with_paths("slider_klein")).unwrap().join(" ").contains("--transformer /m.safetensors"));
        // asymflow: needs the adapter
        let a = asymflow_args(&with_paths("asymflow")).unwrap().join(" ");
        assert!(a.contains("--asymflow-adapter /aux.safetensors"), "{a}");
        // wan22: low-noise = base, high-noise = aux
        let w = wan22_args(&with_paths("wan22")).unwrap().join(" ");
        assert!(w.contains("--low-noise /m.safetensors") && w.contains("--high-noise /aux.safetensors"), "{w}");
        // u1: model-path, no --rank
        let u = u1_args(&with_paths("u1")).unwrap().join(" ");
        assert!(u.contains("--model-path /m.safetensors") && !u.contains("--rank"), "{u}");
    }

    #[test]
    fn asymflow_fails_loud_without_adapter() {
        let mut cfg = with_paths("asymflow");
        cfg.aux_model_path = String::new();
        assert!(asymflow_args(&cfg).is_err());
    }

    #[test]
    fn model_verified_flags_the_nine() {
        for m in ["klein", "sdxl", "zimage", "chroma", "ernie", "anima", "hidream", "sd35", "l2p"] {
            assert!(crate::config::model_verified(m), "{m} should be verified");
        }
        for m in ["flux", "qwenimage", "acestep", "ltx2", "slider_klein", "asymflow", "wan22", "u1"] {
            assert!(!crate::config::model_verified(m), "{m} should be UNVERIFIED");
        }
    }

    #[test]
    fn sdxl_args_uses_unet_and_omits_batch_size() {
        let mut cfg = TrainConfig::default();
        cfg.architecture_index = 3;
        cfg.apply_model_preset(false); // sdxl
        cfg.cache_dir = "/cache/sdxl".into();
        cfg.base_model_path = "/models/sdxl_unet.safetensors".into();
        cfg.max_train_steps = 500.0;
        let joined = sdxl_args(&cfg).expect("sdxl args ok").join(" ");
        assert!(joined.contains("--unet /models/sdxl_unet.safetensors"), "{joined}");
        assert!(joined.contains("--cache-dir /cache/sdxl"), "{joined}");
        assert!(joined.contains("--steps 500"), "{joined}");
        assert!(!joined.contains("--transformer"), "sdxl uses --unet not --transformer: {joined}");
        assert!(!joined.contains("--batch-size"), "train_sdxl has no --batch-size: {joined}");
        assert!(!joined.contains("--offload"), "train_sdxl has no --offload: {joined}");
    }

    #[test]
    fn sdxl_config_is_optional() {
        let mut cfg = TrainConfig::default();
        cfg.model_type = "sdxl".into();
        cfg.cache_dir = "/cache/sdxl".into();
        cfg.base_model_path = "/models/u.safetensors".into();
        let a = sdxl_args(&cfg).expect("ok without a run config");
        assert!(!a.join(" ").contains("--config"), "no --config when path empty");
    }

    #[test]
    fn sdxl_args_fails_loud_on_missing_unet() {
        let mut cfg = TrainConfig::default();
        cfg.model_type = "sdxl".into();
        cfg.cache_dir = "/cache/sdxl".into();
        assert!(sdxl_args(&cfg).is_err());
    }

    #[test]
    fn zimage_args_uses_model_flag_and_batch_size() {
        let mut cfg = TrainConfig::default();
        cfg.architecture_index = 7;
        cfg.apply_model_preset(false); // zimage
        cfg.cache_dir = "/cache/zimage".into();
        cfg.base_model_path = "/models/zimage_base/transformer".into();
        cfg.max_train_steps = 200.0;
        let joined = zimage_args(&cfg).expect("zimage args ok").join(" ");
        assert!(joined.contains("--model /models/zimage_base/transformer"), "{joined}");
        assert!(joined.contains("--cache-dir /cache/zimage"), "{joined}");
        assert!(joined.contains("--batch-size"), "zimage has --batch-size: {joined}");
        assert!(joined.contains("--steps 200"), "{joined}");
        assert!(!joined.contains("--transformer"), "zimage uses --model: {joined}");
        assert!(!joined.contains("--unet"), "{joined}");
        assert!(!joined.contains("--config"), "no --config when path empty: {joined}");
    }

    #[test]
    fn zimage_args_fails_loud_on_missing_model() {
        let mut cfg = TrainConfig::default();
        cfg.model_type = "zimage".into();
        cfg.cache_dir = "/cache/z".into();
        assert!(zimage_args(&cfg).is_err());
    }

    #[test]
    fn chroma_args_uses_transformer_and_omits_batch_size() {
        let mut cfg = TrainConfig::default();
        cfg.architecture_index = 4;
        cfg.apply_model_preset(false); // chroma
        cfg.cache_dir = "/cache/chroma".into();
        cfg.base_model_path = "/models/chroma1hd.safetensors".into();
        cfg.max_train_steps = 250.0;
        let joined = chroma_args(&cfg).expect("chroma args ok").join(" ");
        assert!(joined.contains("--transformer /models/chroma1hd.safetensors"), "{joined}");
        assert!(joined.contains("--cache-dir /cache/chroma"), "{joined}");
        assert!(joined.contains("--steps 250"), "{joined}");
        assert!(!joined.contains("--batch-size"), "train_chroma has no --batch-size: {joined}");
        assert!(!joined.contains("--unet"), "{joined}");
        assert!(!joined.contains("--model "), "chroma uses --transformer: {joined}");
    }

    #[test]
    fn chroma_args_fails_loud_on_missing_transformer() {
        let mut cfg = TrainConfig::default();
        cfg.model_type = "chroma".into();
        cfg.cache_dir = "/cache/chroma".into();
        assert!(chroma_args(&cfg).is_err());
    }

    #[test]
    fn runner_config_json_carries_ui_fields() {
        let mut cfg = TrainConfig::default();
        cfg.architecture_index = 1;
        cfg.apply_model_preset(false); // klein
        cfg.base_model_path = "/models/klein.safetensors".into();
        cfg.lora_rank = 16.0;
        cfg.learning_rate = 3e-5;
        let v: serde_json::Value =
            serde_json::from_str(&runner_config_json(&cfg)).expect("valid json");
        assert_eq!(v["base_model_name"], "/models/klein.safetensors");
        assert_eq!(v["training_method"], "LORA");
        assert_eq!(v["lora_rank"], 16);
        assert!((v["learning_rate"].as_f64().unwrap() - 3e-5).abs() < 1e-9);
    }

    #[test]
    fn needs_generated_config_matches_required_set() {
        for m in ["klein", "ernie", "anima", "sd35"] {
            assert!(needs_generated_config(m), "{m} should require generated config");
        }
        for m in ["sdxl", "zimage", "chroma", "hidream"] {
            assert!(!needs_generated_config(m), "{m} should not");
        }
    }

    #[test]
    fn write_runner_config_writes_parseable_file() {
        let mut cfg = TrainConfig::default();
        cfg.model_type = "klein".into();
        cfg.base_model_path = "/models/k.safetensors".into();
        cfg.output_dir = std::env::temp_dir()
            .join("eritrainer_test_runcfg")
            .to_string_lossy()
            .into_owned();
        let path = write_runner_config(&cfg).expect("write ok");
        let text = std::fs::read_to_string(&path).expect("read back");
        let v: serde_json::Value = serde_json::from_str(&text).expect("valid json");
        assert_eq!(v["base_model_name"], "/models/k.safetensors");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn l2p_args_uses_unusual_flag_names() {
        let mut cfg = TrainConfig::default();
        cfg.architecture_index = 8;
        cfg.apply_model_preset(false); // l2p
        cfg.cache_dir = "/cache/l2p".into();
        cfg.base_model_path = "/models/l2p.safetensors".into();
        cfg.output_dir = "/out/l2p".into();
        cfg.max_train_steps = 100.0;
        let joined = l2p_args(&cfg).expect("l2p args ok").join(" ");
        assert!(joined.contains("--model /models/l2p.safetensors"), "{joined}");
        assert!(joined.contains("--cache /cache/l2p"), "{joined}");
        assert!(joined.contains("--output /out/l2p"), "{joined}");
        assert!(joined.contains("--lora-rank"), "l2p uses --lora-rank: {joined}");
        assert!(joined.contains("--train-shift"), "{joined}");
        // The negative space: l2p must NOT emit these (clap would reject).
        assert!(!joined.contains("--cache-dir"), "l2p uses --cache not --cache-dir: {joined}");
        assert!(!joined.contains("--output-dir"), "l2p uses --output not --output-dir: {joined}");
        assert!(!joined.contains("--rank "), "l2p uses --lora-rank not --rank: {joined}");
        assert!(!joined.contains("--batch-size"), "{joined}");
        assert!(!joined.contains("--offload"), "{joined}");
    }

    #[test]
    fn l2p_args_fails_loud_on_missing_output() {
        let mut cfg = TrainConfig::default();
        cfg.model_type = "l2p".into();
        cfg.cache_dir = "/cache/l2p".into();
        cfg.base_model_path = "/models/l2p.safetensors".into();
        cfg.output_dir = String::new(); // clear the non-empty default
        // l2p --output is required (no clap default) -> fail loud when empty.
        assert!(l2p_args(&cfg).is_err());
    }

    #[test]
    fn sd35_args_uses_transformer_and_requires_config() {
        let mut cfg = TrainConfig::default();
        cfg.model_type = "sd35".into();
        cfg.cache_dir = "/cache/sd35".into();
        cfg.base_model_path = "/models/sd3.5_medium.safetensors".into();
        cfg.max_train_steps = 150.0;
        // No run_config_path -> fail loud (sd35 --config is required).
        assert!(sd35_args(&cfg).is_err());
        cfg.run_config_path = "/cfg/sd35_smoke.json".into();
        let joined = sd35_args(&cfg).expect("sd35 args ok").join(" ");
        assert!(joined.contains("--transformer /models/sd3.5_medium.safetensors"), "{joined}");
        assert!(joined.contains("--config /cfg/sd35_smoke.json"), "{joined}");
        assert!(joined.contains("--batch-size"), "sd35 has --batch-size: {joined}");
        assert!(joined.contains("--warmup-steps"), "sd35 has --warmup-steps: {joined}");
        assert!(!joined.contains("--unet"), "{joined}");
        assert!(!joined.contains("--offload"), "train_sd35 has no --offload: {joined}");
    }

    #[test]
    fn sd35_preset_sets_full_recipe_not_failloud() {
        let mut cfg = TrainConfig::default();
        cfg.model_type_index = 3; // STABLE_DIFFUSION_35
        cfg.apply_model_preset(true);
        assert_eq!(cfg.model_type, "sd35");
        assert!(!cfg.base_model_path.is_empty(), "sd35 preset must set the checkpoint path");
        assert!(!cfg.cache_dir.is_empty(), "sd35 preset must set a cache dir");
        assert!((cfg.timestep_shift - 3.0).abs() < 1e-6);
    }

    #[test]
    fn hidream_args_uses_model_path_and_omits_warmup() {
        let mut cfg = TrainConfig::default();
        cfg.model_type = "hidream".into();
        cfg.cache_dir = "/cache/hidream".into();
        cfg.base_model_path = "/models/HiDream-O1".into();
        cfg.max_train_steps = 100.0;
        let joined = hidream_args(&cfg).expect("hidream args ok").join(" ");
        assert!(joined.contains("--model-path /models/HiDream-O1"), "{joined}");
        assert!(joined.contains("--cache-dir /cache/hidream"), "{joined}");
        assert!(joined.contains("--steps 100"), "{joined}");
        assert!(!joined.contains("--warmup-steps"), "train_hidream_o1 has no --warmup-steps: {joined}");
        assert!(!joined.contains("--min-snr-gamma"), "no --min-snr-gamma: {joined}");
        assert!(!joined.contains("--batch-size"), "no --batch-size: {joined}");
        assert!(!joined.contains("--transformer"), "{joined}");
    }

    #[test]
    fn build_command_maps_hidream_to_train_hidream_o1() {
        let mut cfg = TrainConfig::default();
        cfg.model_type = "hidream".into();
        cfg.cache_dir = "/cache/h".into();
        cfg.base_model_path = "/models/h".into();
        let (program, args) = build_command(&cfg).expect("ok");
        // The UI launches the unified trainer; the local registry maps
        // `hidream` to the `train_hidream_o1` shim inside the backend.
        let in_program = program.ends_with("train");
        let has_unified_bin = args.windows(2).any(|w| w == ["--bin", "train"]);
        let has_model = args.windows(2).any(|w| w == ["--model", "hidream"]);
        assert!(
            in_program || has_unified_bin,
            "expected unified train bin: program={program} args={args:?}"
        );
        assert!(has_model, "expected hidream model dispatch: args={args:?}");
    }

    #[test]
    fn anima_args_uses_dit_path_and_requires_config() {
        let mut cfg = TrainConfig::default();
        cfg.model_type = "anima".into();
        cfg.cache_dir = "/cache/anima".into();
        cfg.base_model_path = "/models/anima_dit.safetensors".into();
        // No run_config_path -> fail loud.
        assert!(anima_args(&cfg).is_err());
        cfg.run_config_path = "/cfg/anima_smoke.json".into();
        let joined = anima_args(&cfg).expect("anima args ok").join(" ");
        assert!(joined.contains("--dit-path /models/anima_dit.safetensors"), "{joined}");
        assert!(joined.contains("--config /cfg/anima_smoke.json"), "{joined}");
        assert!(joined.contains("--cache-dir /cache/anima"), "{joined}");
        assert!(!joined.contains("--transformer"), "anima uses --dit-path: {joined}");
        assert!(!joined.contains("--unet"), "{joined}");
        assert!(!joined.contains("--batch-size"), "train_anima has no --batch-size: {joined}");
    }

    #[test]
    fn ernie_args_requires_config_and_has_no_checkpoint_flag() {
        let mut cfg = TrainConfig::default();
        cfg.model_type = "ernie".into();
        cfg.cache_dir = "/cache/ernie".into();
        // No run_config_path -> must fail loud (Ernie reads model from config).
        assert!(ernie_args(&cfg).is_err());
        cfg.run_config_path = "/cfg/boxjana_ernie_lora.json".into();
        let joined = ernie_args(&cfg).expect("ernie args ok").join(" ");
        assert!(joined.contains("--config /cfg/boxjana_ernie_lora.json"), "{joined}");
        assert!(joined.contains("--cache-dir /cache/ernie"), "{joined}");
        assert!(!joined.contains("--transformer"), "ernie has no checkpoint flag: {joined}");
        assert!(!joined.contains("--unet"), "{joined}");
        assert!(!joined.contains("--model "), "{joined}");
        assert!(!joined.contains("--batch-size"), "train_ernie has no --batch-size: {joined}");
    }
}
