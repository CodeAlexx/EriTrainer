# EriTrainer MAP

Wayfinding for cold-start. See `CLAUDE.md` for hard rules, `TENETS.md` for the
non-negotiable principles, `ERITRAINER_PLAN.md` for the original build spec, and
`README.md` for the user-facing overview. This file is "where does X live" + "how
do I add a model".

## 1. What this is

A native **egui (eframe 0.31, glow)** UI that launches the local EriTrainer
backend and tails trainer stdout. The UI crate still has no direct
`eridiffusion-core` dependency, but the repository now owns the backend under
`backend/` instead of relying on `/home/alex/EriDiffusion/EriDiffusion-v2`.
Run UI tests with `cargo test`; backend dispatcher checks with
`cargo check --manifest-path backend/EriDiffusion-v2/Cargo.toml -p eritrainer-dispatch --bin train`.

Backend boundary: `flame-core` is local and required for trainer acceleration.
Default trainer crates must not depend on `inference-flame`; sampler/helper code
needed by training should live under `eridiffusion-core`.

## 2. Module layout (`src/`)

| Path | Role | Owner |
|---|---|---|
| `main.rs` | eframe `App`, 3-column shell (left `SidePanel` nav / center `TopBottomPanel` topbar + `CentralPanel` tabs / right `SidePanel` rail), per-frame `rt.tick()` + repaint cadence. | backbone |
| `runtime.rs` | `Runtime`, `LiveStats`, `parse_progress_line` (the monitor contract), `build_command` + per-model `*_args`, `launch_env`, `resolve_launcher`, `start`/`stop`/`tick` (spawn + stdout/stderr tail thread → mpsc → parse). | backbone |
| `config.rs` | `TrainConfig` (full OneTrainer field set), `Section` (12 nav entries), `apply_model_preset` + `arch_index_for_model_type`/`model_type_for_arch_index`, the lever machinery (`ignored_lever_summary`/`active_lever_keys`/`supported_lever_keys`), `validate`, `SERENITY_*` path consts. | surface |
| `widgets.rs` | Form-row helpers: `form_panel`, `field_row`, `edit_row`, `browse_row` (native file/folder picker via rfd), `slider_row`, `drag_row`, `combo_row`, `combo_str_row`, `toggle_row`. | surface |
| `theme.rs`, `shell.rs`, `nav.rs`, `topbar.rs`, `rail.rs`, `sysmetrics.rs` | Chrome: serenity dark theme, 3-col metrics, left nav, top action bar (Start/Stop + validation/capability banner), live-status rail, nvidia-smi/`/proc` metrics. | chrome |
| `tabs/mod.rs` + `tabs/*.rs` | Per-section panels (general, model, training, dataset, sampling, lora, captioner, concepts, backup, cloud, runs, logs). | surface |

`main.rs` and `runtime.rs` are the backbone (own the seams + the monitor
contract). `config.rs`/`widgets.rs`/`tabs/*` are the surface. `theme…sysmetrics`
are the chrome. The split came from the original 6-agent build (2 builders ×
disjoint files; see `ERITRAINER_PLAN.md`).

## 3. Launch + monitor flow

```
TopBar "Start" → Runtime::start(cfg)
  → (if needs_generated_config(model_type) && Run Config empty)
      write_runner_config(cfg)      # emit EDv2 TrainConfig JSON from the form → set --config
  → build_command(eff)              # dispatch on cfg.model_type → local `train --model <id> -- ...`
                                    #   + sample_flags(cfg) appended (in-trainer --sample-*)
      → <model>_args(cfg)           # per-model clap-flag mapping; fail-loud on missing paths
      → resolve_launcher("train")   # prebuilt backend target/{release,debug}/train, else `cargo run`
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
`[SD3.5-lora]`) and segment order. Parser tests cover the common progress-line
shape, and verified verticals keep captured real-run examples where available.

## 4. Registered trainers (18) — the per-model CLI zoo

Each old `train_*` still has its OWN clap flags. `build_command` now creates
those args and forwards them through the unified backend dispatcher:
`train --model <model_type> -- <legacy trainer args>`.

The backend registry contains 18 canonical trainer IDs:
acestep, anima, asymflow, chroma, ernie, flux, hidream_o1, ideogram4, klein,
l2p, ltx2, qwenimage, sd35, sdxl, slider_klein, u1, wan22, zimage.

The shared trainer pipeline owns lifecycle services common to those trainers:
progress/timing/loss accounting, optimizer stepping, gradient policy and clipping
helpers, EMA swap/restore/final swap, board status, and checkpoint save wrappers.
Model-specific files keep model loading, cache tensor schema, forward/loss,
sampling, and checkpoint key layout.

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
| ideogram4 | train_ideogram | `--model` (file) | optional | yes | no | FP8 transformer; save/sample/resume wired |

Also wired: **l2p** (`train_l2p`, `--model`/`--cache`/`--output`/`--lora-rank`/
`--train-shift`, grad-checkpoint on by default for the ~19.5GB pixel checkpoint).

The remaining registered trainers are dispatched through the registry and checked
against their real binaries at compile time: flux `--transformer`,
qwenimage/acestep `--model`, ltx2 config-only, slider_klein
`--config`+`--transformer`, asymflow `--config`+`--transformer`+`--asymflow-adapter`,
wan22 `--config`+`--low-noise`[+`--high-noise`/`--vae`], u1 `--model-path`/`--steps`/`--lr`,
and the other registry-only bins. Models without a recent finite-loss GPU smoke
remain **UNVERIFIED** in the UI.
`config::model_verified()` returns false for them and the top bar shows an
UNVERIFIED badge. `aux_model_path` carries the 2nd checkpoint (asymflow adapter,
wan22 high-noise). Those remaining UI paths are NOT sampling-wired
(`sample_flags` emits nothing).

**Runner `--config` generation**: models requiring `--config` (klein/ernie/anima/
sd35, see `needs_generated_config`) auto-write an EDv2-schema `TrainConfig` JSON
from the form (`write_runner_config` → `RunnerConfig`, mirrors the example configs)
when Run Config is empty. A user-set Run Config path is left untouched.

**Config save/load**: `config.rs::{save_to, load_from, configs_dir,
list_saved_configs}` persist the UI `TrainConfig` as JSON under
`~/.config/eritrainer/configs/`; the General tab's CONFIG panel saves/loads them.

**In-trainer sampling**: `runtime.rs::sample_flags(cfg)` emits the per-model
`--sample-*` set from the Sampling tab (Sample After + prompts + the SAMPLE
ASSETS card: VAE/Encoder/Tokenizer). Asset-flag NAMES differ per model
(klein/zimage `--sample-qwen3`, chroma `--sample-t5`, ernie `--sample-text-ckpt`,
l2p `--sample-qwen3` no-VAE, hidream `--sample-every` only). Emits
`--sample-every 0` to DISABLE when off / assets missing (trainers default it on).
**sdxl/anima** have NO in-trainer sampling (separate `sample_<m>` bin → emit
nothing). **sd35** is a 3-encoder model: CLIP-L (generic Encoder/Tokenizer) +
CLIP-G + T5 (+ tokenizers) are required, VAE is OPTIONAL (train_sd35 falls back
to the main checkpoint's VAE). Verified live: klein + zimage + **sd35** produce
real sample images — sd35 GPU smoke (sd3.5_medium + clip_l/clip_g/t5xxl_enconly,
1024) rendered a coherent portrait, `cap=[1,154,4096] pooled=[1,2048]`.
**ideogram4** now has trainer-side latent previews and a standalone
`sample_ideogram` smoke path with prompt cache under `cache/text/`, but the UI
Sampling tab still needs a dedicated Ideogram asset/flag pass before it should be
treated as UI-complete.

**Current dispatcher smoke evidence (2026-06-21):** `train --model sdxl` through
`eritrainer-dispatch` ran one 512-cache step with
`loss 0.1063`, `grad_norm 0.0176`, and saved
`/tmp/eritrainer_sdxl_smoke/sdxl_lora_1steps.safetensors` plus the ComfyUI/Kohya
companion. This proves the unified dispatcher path, shared progress line,
backward/optimizer step, and final save for one representative trainer.
Ideogram standalone sampling was also smoke-tested at 256px/2 steps:
`sample_ideogram` wrote `cache/text`, `cache/latents`, `manifests`, and three
PNGs under `output/ideogram_sampler_test`; the second run hit all prompt caches
and skipped Qwen text-encoder loading.

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
- **Ideogram-4 captions must be MINIFIED JSON.** ai-toolkit conditions ideogram on structured JSON (`high_level_description`/`style_description`/`compositional_deconstruction`+bboxes) and `digest_caption_string`→`to_model_string` **minifies** it before the encoder. `prepare_ideogram` currently encodes the **raw** caption file (`raw_cap.trim()`) — feed it toolkit's pretty (`indent=2`) JSON and you get ~29% extra whitespace/newline tokens vs toolkit (measured: 331 vs 257 on one caption) → off-distribution conditioning. Pre-minify the JSON (or add a minify step) before caching. Prose `.txt` is NOT what ideogram trains on. See `trainer/parity/IDEOGRAM_PARITY_LEDGER.md`.
- **LR schedule defaults to Constant, but past ideogram runs baked cosine→~0.** `run_simple_step_trainer` uses `TrainConfig::default()` (Constant, `lr_min_factor 0`); a fresh `train_ideogram --lr 1e-4` holds 1e-4 (matches toolkit). The eri2 runs that "didn't learn" baked `lr=3.4e-11` (cosine-to-zero) — verify the effective schedule, don't assume the CLI flag is the whole story.
- **Backward verification: use the adjoint test, not cross-impl grad cosine or param finite-difference** (both confound on bf16; see flame-core `docs/FLAME_CONVENTIONS.md` + the ledger). Ideogram block backward is adjoint-verified clean.

## 7. Related docs

- `CLAUDE.md` — agent session entry, hard rules, doc-update obligations.
- `TENETS.md` — launcher-not-trainer, honesty discipline, verify-each-model.
- `ERITRAINER_PLAN.md` — original build spec + the 6-agent loop + milestones.
- `README.md` — user-facing overview, build/run.
