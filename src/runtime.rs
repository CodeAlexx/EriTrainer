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
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;

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

        let (program, args) = match build_command(cfg) {
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

// --- Launch command construction (M1: Klein wired; others fail loud) ---

/// EDv2 workspace dir; override with `ERITRAINER_EDV2_DIR`.
fn edv2_dir() -> PathBuf {
    match std::env::var("ERITRAINER_EDV2_DIR") {
        Ok(d) if !d.trim().is_empty() => PathBuf::from(d),
        _ => PathBuf::from("/home/alex/EriDiffusion/EriDiffusion-v2"),
    }
}

/// How to launch an EDv2 bin: prefer a prebuilt `target/{release,debug}/<bin>`,
/// else fall back to `cargo run --release --bin <bin>` against the workspace.
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
        (
            String::from("cargo"),
            vec![
                String::from("run"),
                String::from("--release"),
                String::from("--manifest-path"),
                manifest.to_string_lossy().into_owned(),
                String::from("--bin"),
                String::from(bin),
                String::from("--"),
            ],
        )
    }
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

/// Build (program, args) for the selected model's trainer. Only Klein is wired
/// in M1; every other model returns Err so the launch fails loud instead of
/// spawning a trainer with the wrong argv.
pub fn build_command(cfg: &TrainConfig) -> Result<(String, Vec<String>), String> {
    let (bin, args) = match cfg.model_type.as_str() {
        "klein" => ("train_klein", klein_args(cfg)?),
        "sdxl" => ("train_sdxl", sdxl_args(cfg)?),
        "zimage" => ("train_zimage", zimage_args(cfg)?),
        "chroma" => ("train_chroma", chroma_args(cfg)?),
        other => {
            return Err(format!(
                "launch for model '{other}' is not wired yet (wired: klein, sdxl, zimage, chroma)"
            ))
        }
    };
    let (program, mut full) = resolve_launcher(bin);
    full.extend(args);
    Ok((program, full))
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

#[cfg(test)]
mod tests {
    use super::*;

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
        cfg.model_type = "chroma".into(); // not yet wired
        assert!(build_command(&cfg).is_err());
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
}
