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
- **Klein** launch path wired and verified end-to-end against the real `train_klein`
  binary (a short smoke run produced decreasing loss and a LoRA checkpoint).
- Remaining: wire the other models' launch arguments (each `train_*` has its own CLI),
  and a polished live-GUI pass.

See `ERITRAINER_PLAN.md` for the full build spec.
