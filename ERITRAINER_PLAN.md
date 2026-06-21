# EriTrainer — build spec & agent brief

**What:** A native **egui (eframe 0.31, glow)** training UI for the EriDiffusion-v2
(EDv2) Rust trainers. It **looks and behaves like the Mojo `serenity-trainer` UI**
and **launches the existing EDv2 `train_*` binaries unchanged** (it does NOT
replace them). OneTrainer is the field/organization reference (already mirrored
by the Mojo tabs).

**Repo:** `/home/alex/EriTrainer` (standalone, like serenityUI/serenity-trainer).
**Binary:** `eritrainer`. No `eridiffusion-core` dep — the UI is a launcher, so it
stays free of the CUDA/flame-core stack and builds without a GPU toolchain.

---

## Source of truth (READ THESE — do not invent layout)

The Mojo trainer UI, file-by-file:
- Shell/layout/dispatch: `/home/alex/serenity-trainer/src/serenity_trainer/ui/TrainUI.mojo`
- Top action bar: `.../ui/TopBar.mojo`
- Tabs: `.../ui/{GeneralTab,ModelTab,LoraTab,DatasetTab,CaptionerTab,ConceptTab,TrainingTab,SamplingTab,CloudTab}.mojo`
- Config model (all fields): `.../ui/TrainerConfigModel.mojo`
- Runtime bridge (launch + poll behavior): `.../ui/TrainerRuntimeBridge.mojo`
- Media gallery/lightbox: `.../ui/TrainerMediaGallery.mojo`

The monitor contract (what every EDv2 trainer prints) is fixed by:
- `backend/EriDiffusion-v2/crates/eridiffusion-core/src/training/progress.rs`
  → one stdout line per step:
  `[<tag>] step N/T | epoch e/E | loss X.XXXX | grad_norm X.XXXX | X.Xs/step | elapsed H:MM:SS | ETA H:MM:SS`
  `parse_progress_line` in `src/runtime.rs` already parses it (unit-tested).

EDv2 train bins to launch: `crates/eridiffusion-cli/src/bin/train_*.rs` (18 trainer targets
registered through the local dispatcher).
Klein = `train_klein` is the M1 reference vertical.

---

## Look + behavior to reproduce (the map)

- **3-column shell**, ~1480×920, dark serenity theme.
  - **Left nav:** header (Serenity / Native trainer UI) → CONFIGURE: General · Model ·
    LoRA/OFT · Dataset · Captioner · Validations · Training · Sampling · Backup ·
    Cloud · Runs · Logs → PROJECT: run name.
  - **Center:** top action bar + scrollable tab body.
  - **Right rail:** status pill (IDLE/RUNNING/PAUSED) · progress bar · stats (Step,
    Epoch, Loss, Smooth, Grad, LR, Speed s/step, ETA, Status, Command, Source) ·
    artifacts (Samples, Checkpoints, Backend) · hardware (GPU model/driver/util/temp,
    VRAM, CPU model/util, RAM). Media lightbox overlay.
- **Top bar** row 1: `SECTION — <name>` · run-name edit · `Live target — <backend>` ·
  Start/Stop. Row 2: `VALIDATION — …` + capability warning · Pause/Sample/Save.
- **Tabs:** titled form panels of slider / drag / combo / toggle / read-only rows.
- **Behavior:** Start → write runner config JSON → spawn `train_<model>` → tail its
  stdout progress line → drive the rail live; Stop → kill the child. Pause/Sample/Save
  are command-file events no trainer consumes yet.

### Honesty discipline (NON-NEGOTIABLE — from the Mojo UI)
No widget may silently lie. Any control the launched runner does NOT consume must be
labeled `[not wired]` and named in the pre-launch capability warning
(`TrainConfig::ignored_lever_summary`). A blank tab must say "not yet built", never
render as if functional.

---

## Module layout & file ownership (disjoint — no two agents edit the same file)

Orchestrator-owned (the validation backbone, already written):
- `src/main.rs` — eframe app, 3-column layout, dispatch.
- `src/runtime.rs` — Runtime, LiveStats, `parse_progress_line` + tests.

**Builder-A (chrome):** `src/theme.rs`, `src/shell.rs`, `src/nav.rs`,
`src/topbar.rs`, `src/rail.rs`, `src/sysmetrics.rs`.

**Builder-B (surface):** `src/config.rs`, `src/widgets.rs`, `src/tabs/*.rs`.

Cross-cutting signature changes (e.g. a new `Runtime` field, a new `TrainConfig`
field consumed by another module) must be requested from the orchestrator, not
made unilaterally — the backbone owns those seams.

---

## Milestones & gates (orchestrator validates each)

- **M0 — skeleton (DONE):** compiles offline; window opens; 3-column shell; nav
  switches; `cargo test` (parser) green.
- **STATUS 2026-06-19:** M0 + chrome + surface + **M1 Klein launch wiring all DONE**
  and gated — `cargo check`/`build` clean (0 warnings), `cargo test` 11/11. Built via
  the 6-agent loop (2 builders → 2 skeptics → 2 bugfix), orchestrator-gated. The real
  Klein training run is user-triggered (needs GPU + built `train_klein` + checkpoints/
  cache + an EDv2 `--config` JSON). This status is superseded by the 2026-06-21
  backend copy/dispatcher status below.
- **STATUS 2026-06-21:** local EDv2 backend copied under `backend/`; lightweight
  `train --model <id>` dispatcher and registry added; 18 trainer models registered
  and compile-checked as real trainer binaries. The trainer lifecycle migration now
  centralizes common progress/timing/loss accounting, optimizer stepping, gradient
  policy/clipping helpers, EMA, board status, and checkpoint-save wrappers in
  `trainer_pipeline` / `trainer_common`, while model files keep model-specific load,
  cache, forward/loss, sampling, and checkpoint key behavior. Full GPU smoke remains
  a separate gate for models that are compile-verified only. Representative smoke:
  SDXL via `train --model sdxl` emitted `step 1/1 | loss 0.1063 | grad_norm 0.0176`
  and saved final LoRA artifacts.
- **M1 — Klein vertical:**
  - Builder-A: full rail (all groups), real `sysmetrics` (nvidia-smi + /proc),
    serenity-style theme/nav/topbar, capability-warning banner live.
  - Builder-B: full `TrainConfig` (OneTrainer field set per `TrainerConfigModel.mojo`),
    `widgets.rs` rows (slider/drag/combo/toggle/field), Training+Model+General+
    Dataset+Sampling tabs populated.
  - Orchestrator: wire `Runtime::start` to actually spawn `train_klein` with a
    runner config + argv, stdout-tail thread → `apply_progress_line`, `Runtime::tick`
    drains it; `Stop` kills the child.
  - **Gate:** `cargo check` + `cargo test` green; window runs; launching Klein shows
    live step/loss/grad/s-step/ETA moving in the rail; Stop ends it; no unwired
    widget lies.
- **M2+ — tab fan-out + remaining model polish** (LoRA/Captioner/Cloud/Backup/Runs/Logs,
  then broader smoke/polish coverage for registered `train_*` targets), each gated the same way.

## 6-agent loop (build → skeptic → bugfix; orchestrator gates)
- **2 Builders** (A=chrome, B=surface), disjoint files, parallel.
- **2 Skeptics:** verify against this spec + the Mojo source files + the honesty rule;
  MEASURE (compile, run, read) — do not assert. Report concrete defects with file:line.
- **2 Bugfix:** fix skeptic findings + any compile errors; keep files disjoint per fix.
- **Orchestrator:** owns backbone, defines chunks, runs `cargo check`/`cargo test`,
  launches the app, checks fidelity, decides pass/fail, iterates.
