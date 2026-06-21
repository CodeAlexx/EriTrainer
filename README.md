# EriTrainer

A native [egui](https://github.com/emilk/egui) training UI plus local Rust
trainer backend for EriDiffusion. EriTrainer mirrors the look and behavior of
the Mojo `serenity-trainer` UI (whose tab layout in turn mirrors OneTrainer) and
now owns a copied EDv2/flame-core backend under `backend/`.

## Design

- **UI separated from backend.** The egui app still builds without linking the
  CUDA/flame-core stack, but it launches the local backend by default:
  `backend/EriDiffusion-v2` → `eridiffusion-core` → local `flame-core`.
- **Unified trainer front door.** The UI launches the local `train` dispatcher,
  which resolves the model through a lightweight registry and forwards to the
  existing per-model trainer shim while the OneTrainer-style setup/loop migration proceeds.
- **OneTrainer-style layer split.** The shared trainer pipeline owns lifecycle,
  config/preflight, cache discovery, runtime/device setup, metrics, progress,
  and save/sample orchestration. Model-specific code owns model load, cache
  tensor schema, forward/loss, sampler details, and checkpoint key layout.
- **No inference runtime dependency.** Trainer sampling/model helper code belongs
  in `eridiffusion-core` / `eridiffusion-core::sampler`; default trainer crates
  should not depend on `inference-flame`.
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

To launch training you also need a prepared latent cache and the model weights.
EriTrainer resolves trainer binaries from
`$ERITRAINER_EDV2_DIR/target/{release,debug}/train` (default
`./backend/EriDiffusion-v2`), falling back to `cargo run --bin train` in the
current app profile. The `train` dispatcher itself does not link the CUDA stack;
the selected model trainer does. The trainer backend links libtorch, so EriTrainer puts it on `LD_LIBRARY_PATH`
(`$ERITRAINER_LIBTORCH`, default `/home/alex/libs/libtorch/lib`).

In the UI: pick the model on the **Model** tab (Model Type / Architecture apply a
per-model preset), set the base model + run-config paths, the cache dir on **General**,
and the schedule on **Training**, then **Start training**.

## Status

- Shell, tabs, config model, live rail, and system metrics: built.
- Local backend copied into `backend/` with local `flame-core`,
  `eri-lycoris/lycoris-rs`, and pinned `cudarc`.
- Lightweight dispatcher package `eritrainer-dispatch` and shared
  `eritrainer-registry` added.
- `inference-flame` removed from the migrated default trainer dependency graph;
  HiDream-O1 helpers, AsymFlow math, and Oklab color math are local core modules.
- **18 trainer models** registered behind the local `train --model <id>` dispatcher:
  acestep, anima, asymflow, chroma, ernie, flux, hidream-o1, ideogram4, klein,
  l2p, ltx2, qwenimage, sd35, sdxl, slider-klein, u1, wan22, zimage. Each
  `train_*` still has its own model-specific CLI; the dispatcher and UI give them
  one front door.
- **Shared trainer pipeline migration:** trainer lifecycle code now lives in
  `trainer_pipeline` / `trainer_common` for logging, progress/timing, loss
  accounting, optimizer stepping, gradient policy/clipping helpers, EMA swaps, and
  checkpoint saves. Model-specific files keep model load, cache schema,
  forward/loss, sampling, and checkpoint key details.
- **GPU smoke coverage:** SDXL was re-smoked through the local dispatcher on
  2026-06-21: `train --model sdxl` ran one step with finite loss/grad norm and
  saved both EDv2 and ComfyUI/Kohya LoRA files. Other models retain their prior
  verification status or compile-only status as listed in `MAP.md`.
- **Runner `--config` generation**: models whose trainer requires `--config`
  (klein/ernie/anima/sd35) auto-write an EDv2 `TrainConfig` JSON from the form
  when you haven't supplied a Run Config path.
- **Config save/load**: the General tab can save the current run config to
  `~/.config/eritrainer/configs/<run_name>.json` and reload any saved config.
- **File/folder pickers**: path fields (Base Model, Run Config, Cache, Workspace,
  Destination, Dataset, Sample Dir) have a native "Browse…" button (rfd).
- Remaining: a polished live-GUI pass and broader real GPU smoke coverage for
  the models that have only been compile-verified in this repository.

See `MAP.md` for the code map + add-a-model recipe and `ERITRAINER_PLAN.md` for the
full build spec.
