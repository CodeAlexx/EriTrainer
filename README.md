# EriTrainer

A native [egui](https://github.com/emilk/egui) training UI for the EriDiffusion-v2
Rust diffusion trainers. EriTrainer mirrors the look and behavior of the Mojo
`serenity-trainer` UI (whose tab layout in turn mirrors OneTrainer) and **launches
the existing `train_*` trainer binaries unchanged** — it is a launcher/monitor, not
a new trainer.

## Design

- **Launcher, not a trainer.** EriTrainer writes/forwards a run configuration, spawns
  the selected `train_<model>` binary as a subprocess, and tails its stdout. It has
  **no dependency on the training stack** (no CUDA/libtorch needed to build the UI).
- **Monitor contract.** Every EriDiffusion-v2 trainer prints one progress line per step:
  ```
  [<tag>] step N/T | epoch e/E | loss X.XXXX | grad_norm X.XXXX | X.Xs/step | elapsed H:MM:SS | ETA H:MM:SS
  ```
  EriTrainer parses that line (`src/runtime.rs::parse_progress_line`) and drives the
  live status rail.
- **Honesty discipline.** No widget silently lies: any control the launched runner
  does not consume is labelled `[not wired]` and named in a pre-launch capability
  warning. Unbuilt tabs say so rather than rendering blank.

## Layout

Three-column shell, like the Mojo trainer UI:

- **Left nav** — General · Model · LoRA/OFT · Dataset · Captioner · Validations ·
  Training · Sampling · Backup · Cloud · Runs · Logs.
- **Center** — top action bar (Start/Stop, validation + capability warning) and the
  selected tab's form panels.
- **Right rail** — live status (step/epoch/loss/grad/LR/speed/ETA), artifacts, and
  GPU/CPU/RAM hardware stats.

## Build & run

```sh
cargo run            # opens the UI (eframe + egui, glow backend)
cargo test           # unit tests (progress parser, model presets, launch command)
```

To launch training you also need the EriDiffusion-v2 trainers built and a prepared
latent cache. EriTrainer resolves the trainer binary from
`$ERITRAINER_EDV2_DIR/target/{release,debug}/train_<model>` (default
`/home/alex/EriDiffusion/EriDiffusion-v2`), falling back to `cargo run`. The EDv2
binaries link libtorch, so EriTrainer puts it on `LD_LIBRARY_PATH`
(`$ERITRAINER_LIBTORCH`, default `/home/alex/libs/libtorch/lib`).

In the UI: pick the model on the **Model** tab (Model Type / Architecture apply a
per-model preset), set the base model + run-config paths, the cache dir on **General**,
and the schedule on **Training**, then **Start training**.

## Status

- Shell, tabs, config model, live rail, and system metrics: built.
- **9 models** wired and verified end-to-end against their real binaries (each: a
  short smoke run produced finite/decreasing loss + a LoRA checkpoint, and the
  trainer's progress line is covered by a parser test): klein, sdxl, zimage,
  chroma, ernie, anima, hidream-o1, sd35, l2p. Each `train_*` has its own CLI —
  see the per-model table in `MAP.md`.
- **Runner `--config` generation**: models whose trainer requires `--config`
  (klein/ernie/anima/sd35) auto-write an EDv2 `TrainConfig` JSON from the form
  when you haven't supplied a Run Config path.
- **Config save/load**: the General tab can save the current run config to
  `~/.config/eritrainer/configs/<run_name>.json` and reload any saved config.
- **File/folder pickers**: path fields (Base Model, Run Config, Cache, Workspace,
  Destination, Dataset, Sample Dir) have a native "Browse…" button (rfd).
- Remaining: the deferred/video models (flux, qwen, wan22, ltx2, …); a polished
  live-GUI pass.

See `MAP.md` for the code map + add-a-model recipe and `ERITRAINER_PLAN.md` for the
full build spec.
