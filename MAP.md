# EriTrainer MAP

Wayfinding for cold-start. See `CLAUDE.md` for hard rules, `TENETS.md` for the
non-negotiable principles, `ERITRAINER_PLAN.md` for the original build spec, and
`README.md` for the user-facing overview. This file is "where does X live" + "how
do I add a model".

## 1. What this is

A native **egui (eframe 0.31, glow)** UI that **launches the EriDiffusion-v2
`train_*` binaries** and tails their stdout. It is a launcher/monitor, NOT a
trainer. It has **no `eridiffusion-core` dependency** — it builds without a GPU
toolchain (eframe + serde only). Run: `cargo run`; test: `cargo test`.

## 2. Module layout (`src/`)

| Path | Role | Owner |
|---|---|---|
| `main.rs` | eframe `App`, 3-column shell (left `SidePanel` nav / center `TopBottomPanel` topbar + `CentralPanel` tabs / right `SidePanel` rail), per-frame `rt.tick()` + repaint cadence. | backbone |
| `runtime.rs` | `Runtime`, `LiveStats`, `parse_progress_line` (the monitor contract), `build_command` + per-model `*_args`, `launch_env`, `resolve_launcher`, `start`/`stop`/`tick` (spawn + stdout/stderr tail thread → mpsc → parse). | backbone |
| `config.rs` | `TrainConfig` (full OneTrainer field set), `Section` (12 nav entries), `apply_model_preset` + `arch_index_for_model_type`/`model_type_for_arch_index`, the lever machinery (`ignored_lever_summary`/`active_lever_keys`/`supported_lever_keys`), `validate`, `SERENITY_*` path consts. | surface |
| `widgets.rs` | Form-row helpers: `form_panel`, `field_row`, `edit_row`, `slider_row`, `drag_row`, `combo_row`, `combo_str_row`, `toggle_row`. | surface |
| `theme.rs`, `shell.rs`, `nav.rs`, `topbar.rs`, `rail.rs`, `sysmetrics.rs` | Chrome: serenity dark theme, 3-col metrics, left nav, top action bar (Start/Stop + validation/capability banner), live-status rail, nvidia-smi/`/proc` metrics. | chrome |
| `tabs/mod.rs` + `tabs/*.rs` | Per-section panels (general, model, training, dataset, sampling, lora, captioner, concepts, backup, cloud, runs, logs). | surface |

`main.rs` and `runtime.rs` are the backbone (own the seams + the monitor
contract). `config.rs`/`widgets.rs`/`tabs/*` are the surface. `theme…sysmetrics`
are the chrome. The split came from the original 6-agent build (2 builders ×
disjoint files; see `ERITRAINER_PLAN.md`).

## 3. Launch + monitor flow

```
TopBar "Start" → Runtime::start(cfg)
  → build_command(cfg)              # dispatch on cfg.model_type → (program, args)
      → <model>_args(cfg)           # per-model clap-flag mapping; fail-loud on missing paths
      → resolve_launcher(bin)       # prebuilt target/{release,debug}/<bin>, else `cargo run`
  → Command + launch_env(cfg)       # libtorch on LD_LIBRARY_PATH, RUST_LOG, klein FLAME flags
  → spawn, pipe stdout+stderr → tail threads → mpsc::Sender
Runtime::tick() (each frame)
  → sysmetrics::refresh(rt)         # GPU/CPU/RAM (throttled)
  → drain rx → parse_progress_line  # sets live stats + progress_source="stdout"
  → child.try_wait()                # detect exit → status "completed"/"exited"
rail.rs renders Runtime.live
```

**Monitor contract** = the one stdout line every EDv2 trainer prints via
`eridiffusion-core::training::progress::log_step_with_resume`:
`[<tag>] step N/T | epoch e/E | loss X | grad_norm X | X.Xs/step | elapsed H:MM:SS | ETA H:MM:SS`.
`parse_progress_line` tolerates the `[tag]` prefix (any tag, incl. dotted like
`[SD3.5-lora]`) and segment order. Each wired model has a `parses_real_train_<m>_run_line`
test using a line captured from a real run.

## 4. Wired models (8) — the per-model CLI zoo

Each `train_*` has its OWN clap flags. There is no template — read the real
binary's `Args` before wiring. Current mapping (`build_command`):

| model_type | bin | checkpoint flag | `--config` | `--batch-size` | `--offload` | notes |
|---|---|---|---|---|---|---|
| klein | train_klein | `--transformer` (dir of shards ok) | required | yes | yes | FLAME_ALLOC_POOL=0 |
| sdxl | train_sdxl | `--unet` (file) | optional | no | no | |
| zimage | train_zimage | `--model` (single file ONLY) | optional | yes | no | dir errors "Is a directory" |
| chroma | train_chroma | `--transformer` (file) | optional | no | yes | T5 encoder |
| ernie | train_ernie | none (model from config) | required | no | no | Mistral-3B encoder |
| anima | train_anima | `--dit-path` | required | no | no | Qwen3-0.6B + T5 |
| hidream | **train_hidream_o1** | `--model-path` (dir) | optional | no | no | name reconciliation |
| sd35 | train_sd35 | `--transformer` (file) | required | yes | no | 3 encoders; prep 1024-only |

Deferred (not wired): flux-1-dev, qwenimage/qwen-edit, big models; wan22, ltx2,
acestep, l2p, u1, asymflow, slider_klein.

## 5. Where to start — add a model

1. `awk '/struct Args/{f=1} f{print} /^}/{if(f)exit}' <EDv2>/crates/eridiffusion-cli/src/bin/train_<m>.rs | grep -E "^\s{4}[a-z_]+:"` — get the EXACT field list (the `-B1 grep` trick MISSES doc-commented args; use awk).
2. Add `<m>_args(cfg) -> Result<Vec<String>, String>` in `runtime.rs` mirroring those flags; fail loud on every missing required path.
3. Add the `build_command` match arm `"<model_type>" => ("train_<m>", <m>_args(cfg)?)`. Reconcile model_type → bin name if they differ (see hidream).
4. Set/verify the `apply_model_preset` arm in `config.rs` (checkpoint path, cache, recipe). If the checkpoint must be a single file, point the preset there.
5. Add unit tests: a `<m>_args_*` flag test (assert the right checkpoint flag + the absent flags) and, after a real run, a `parses_real_train_<m>_run_line` test.
6. Verify against the real binary (TENETS §3): `cargo build --release --bin train_<m> --bin prepare_<m>` (LIBTORCH+LD_LIBRARY_PATH); build an 8-sample 512 cache from `/home/alex/EriDiffusion/datasets/40_woman` via `prepare_<m>` with that model's VAE+encoder; `train_<m> --steps 3`; confirm `[<m>-lora]` lines + a LoRA.
7. `cargo test` green → commit + push.

## 6. Gotchas

- **Checkpoint flag differs every model** (table §4). Passing the wrong one (or an unsupported `--batch-size`/`--offload`/`--warmup-steps`) makes clap reject and the launch fails. Read the Args.
- **Single-file vs dir checkpoints**: zimage `--model`, and `prepare_ernie` vae/text-ckpt, want a single `.safetensors` FILE — a sharded/diffusers DIR errors `mmap … Is a directory`. klein `--transformer` and hidream `--model-path` accept dirs.
- **The EDv2 bins link libtorch.** They won't even load without it on `LD_LIBRARY_PATH` — `launch_env` adds `/home/alex/libs/libtorch/lib` (override `ERITRAINER_LIBTORCH`).
- **Klein needs `FLAME_ALLOC_POOL=0`** (pool ON = step-2 CUDA crash on 9B) — `launch_env` sets it for klein.
- **`prepare_sd35` is 1024-only** (`--resolution 1024`); others take 512.
- **`prepare_ernie` uses `--size`** (not `--resolution`) and has NO `--max-samples` (point it at a subset dir).
- **Don't pipe `cargo … | tail`** when you need the exit code — `tail` masks it. Use `; echo "RC=$?"` without a pipe, or `${PIPESTATUS[0]}`.
- **No xvfb in the sandbox** → the live GUI can't be smoke-run headless; the binary builds + `cargo test` cover everything except the visual.

## 7. Related docs

- `CLAUDE.md` — agent session entry, hard rules, doc-update obligations.
- `TENETS.md` — launcher-not-trainer, honesty discipline, verify-each-model.
- `ERITRAINER_PLAN.md` — original build spec + the 6-agent loop + milestones.
- `README.md` — user-facing overview, build/run.
