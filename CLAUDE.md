# EriTrainer (for Claude sessions)

A native **egui (eframe 0.31, glow)** UI plus an embedded EriDiffusion-v2 trainer
backend under `backend/`. The UI crate is still a launcher/monitor, not a CUDA
trainer, and keeps no `eridiffusion-core` dependency so it builds without a GPU
toolchain. The backend copy owns the Rust trainers, local `flame-core`,
`eri-lycoris`, and pinned `cudarc`.

Run the UI with `cargo run`; test the UI with `cargo test`. Backend checks use
`cargo --manifest-path backend/EriDiffusion-v2/Cargo.toml ...`.

## Read these first

- [`TENETS.md`](./TENETS.md) — the 5 non-negotiables: launcher-not-trainer,
  no-lying-widget (honesty discipline), verify-each-model-against-its-binary,
  read-each-CLI, mirror-the-Mojo-UI. **Read before touching launch or widget code.**
- [`MAP.md`](./MAP.md) — module layout, the launch/monitor flow, the per-model
  CLI table, and the "add a model" recipe. The first place for "where is X".
- [`ERITRAINER_PLAN.md`](./ERITRAINER_PLAN.md) — original build spec + the 6-agent
  build/skeptic/bugfix loop that created this + milestones.

## Quick orientation

- **Backbone** (owns the UI seams + monitor contract): `main.rs` (eframe App, 3-col
  shell) and `runtime.rs` (`Runtime`, `parse_progress_line`, `build_command` +
  per-model `*_args`, `launch_env`, spawn + stdout-tail).
- **Surface**: `config.rs` (`TrainConfig`, `Section`, `apply_model_preset`, lever
  machinery), `widgets.rs` (form rows), `tabs/*` (per-section panels).
- **Chrome**: `theme/shell/nav/topbar/rail/sysmetrics`.
- **Embedded backend**: `backend/EriDiffusion-v2` with a lightweight
  `eritrainer-dispatch` binary named `train`, `eritrainer-registry`, and
  model-specific trainer bins in `eridiffusion-cli`.
- **Shared trainer boundary**: `eridiffusion-cli/src/trainer_pipeline.rs` and
  `trainer_common.rs` own common lifecycle/progress/optimizer/EMA/checkpoint
  services. Model-specific files keep load/cache/forward/loss/sampling/key layout.
- **Monitor contract**: the one stdout line every EDv2 trainer prints via
  `eridiffusion-core::training::progress::log_step_with_resume` — see MAP §3.
- **Runner `--config` generation**: `runtime.rs::write_runner_config` emits an
  EDv2 `TrainConfig` JSON from the form for models in `needs_generated_config`
  (klein/ernie/anima/sd35) when Run Config is empty. Keep `RunnerConfig`'s field
  set aligned with EDv2's loader (mirror the working `configs/*.json`).

## Hard rules

- **NEVER add a training loop / model / kernel to the UI crate.** The UI is a
  launcher (TENET 1).
- **NEVER modify backend `train_*` binaries just to fake UI support.** Map the UI
  to their real CLI. Backend trainer refactors are valid only when they improve
  the trainer itself and preserve standalone CLI behavior.
- **NEVER add a `eridiffusion-core` (or CUDA/flame-core) dependency.**
- **NEVER ship a widget that doesn't do what it implies.** Mark `[not wired]` +
  list it in `ignored_lever_summary`, or make a blocked launch fail loud (TENET 2).
- **NEVER template a model's `*_args` from another model.** Read its `Args`
  struct via `awk` (the `grep -B1` trick misses doc-commented args) — TENET 4.
- **NEVER claim a model is GPU-verified without running its real binary**
  (TENET 3). Registry/compile coverage and end-to-end finite-loss smoke coverage
  are separate statuses.
- **NEVER add `rustfrt` / `rust_frt` / `rust-frt`.** It is explicitly forbidden.
- **NEVER add an active `inference-flame` trainer dependency.** Copy known-good
  sampler/model helper code into `eridiffusion-core` when training needs it.

## Adding a model

See MAP §5 for the full recipe. Short version: register the model in
`eritrainer-registry`, read `train_<m>` Args, write/update `<m>_args` in the UI
runtime if the UI launches it directly, keep the unified backend dispatcher path
working, set the `apply_model_preset` arm, add unit tests, compile the real bin,
then run a tiny finite-loss smoke when weights/cache exist.

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

- New registered or smoke-verified model → update MAP §4 and README status.
- New convention/gotcha → add to MAP §6.
- New module or seam → update MAP §2.
- New principle the work taught you → consider TENETS.

These docs are curated, not generated. A 5-minute update beats a 30-minute
rediscovery next session.
