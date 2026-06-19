# EriTrainer (for Claude sessions)

A native **egui (eframe 0.31, glow)** UI that launches the EriDiffusion-v2
`train_*` binaries and tails their stdout. Launcher/monitor, **not a trainer**.
~30 Rust files, no `eridiffusion-core` dep (builds without a GPU toolchain).
Run `cargo run`; test `cargo test` (offline-buildable: eframe 0.31 is cached).

## Read these first

- [`TENETS.md`](./TENETS.md) — the 5 non-negotiables: launcher-not-trainer,
  no-lying-widget (honesty discipline), verify-each-model-against-its-binary,
  read-each-CLI, mirror-the-Mojo-UI. **Read before touching launch or widget code.**
- [`MAP.md`](./MAP.md) — module layout, the launch/monitor flow, the per-model
  CLI table, and the "add a model" recipe. The first place for "where is X".
- [`ERITRAINER_PLAN.md`](./ERITRAINER_PLAN.md) — original build spec + the 6-agent
  build/skeptic/bugfix loop that created this + milestones.

## Quick orientation

- **Backbone** (owns the seams + monitor contract): `main.rs` (eframe App, 3-col
  shell) and `runtime.rs` (`Runtime`, `parse_progress_line`, `build_command` +
  per-model `*_args`, `launch_env`, spawn + stdout-tail).
- **Surface**: `config.rs` (`TrainConfig`, `Section`, `apply_model_preset`, lever
  machinery), `widgets.rs` (form rows), `tabs/*` (per-section panels).
- **Chrome**: `theme/shell/nav/topbar/rail/sysmetrics`.
- **Monitor contract**: the one stdout line every EDv2 trainer prints via
  `eridiffusion-core::training::progress::log_step_with_resume` — see MAP §3.
- **Runner `--config` generation**: `runtime.rs::write_runner_config` emits an
  EDv2 `TrainConfig` JSON from the form for models in `needs_generated_config`
  (klein/ernie/anima/sd35) when Run Config is empty. Keep `RunnerConfig`'s field
  set aligned with EDv2's loader (mirror the working `configs/*.json`).

## Hard rules

- **NEVER add a training loop / model / kernel here.** It's a launcher (TENET 1).
- **NEVER modify the EDv2 `train_*` binaries to suit the UI.** Map to their real
  CLI instead.
- **NEVER add a `eridiffusion-core` (or CUDA/flame-core) dependency.**
- **NEVER ship a widget that doesn't do what it implies.** Mark `[not wired]` +
  list it in `ignored_lever_summary`, or make a blocked launch fail loud (TENET 2).
- **NEVER template a model's `*_args` from another model.** Read its `Args`
  struct via `awk` (the `grep -B1` trick misses doc-commented args) — TENET 4.
- **NEVER claim a model is wired without running its real binary** (TENET 3): a
  `parses_real_train_<m>_run_line` test from a captured line + an end-to-end smoke.

## Adding a model

See MAP §5 for the full recipe. Short version: read `train_<m>` Args → write
`<m>_args` (fail loud on missing paths) → add the `build_command` arm (reconcile
model_type→bin if needed) → set the `apply_model_preset` arm → unit tests →
`cargo build --release --bin train_<m> --bin prepare_<m>` (LIBTORCH +
LD_LIBRARY_PATH) → build an 8-sample cache from `datasets/40_woman` → `--steps 3`
smoke → real-line test → `cargo test` → commit + push.

## Known launch landmines (see MAP §6 for the full list)

- Checkpoint flag differs per model (`--transformer`/`--unet`/`--model`/
  `--dit-path`/`--model-path`/none). Wrong flag = clap reject = failed launch.
- Single-file-only checkpoints (zimage `--model`; `prepare_ernie` vae/text) error
  `mmap … Is a directory` on a sharded dir.
- EDv2 bins link libtorch → must be on `LD_LIBRARY_PATH` (`launch_env` handles it).
- Klein needs `FLAME_ALLOC_POOL=0`. `prepare_sd35` is 1024-only. `prepare_ernie`
  uses `--size` and has no `--max-samples`.
- `cargo … | tail` masks the exit code — use `; echo RC=$?` without a pipe.

## When you change things

- New wired model → update the MAP §4 table + add the two test kinds.
- New convention/gotcha → add to MAP §6.
- New module or seam → update MAP §2.
- New principle the work taught you → consider TENETS.

These docs are curated, not generated. A 5-minute update beats a 30-minute
rediscovery next session.
