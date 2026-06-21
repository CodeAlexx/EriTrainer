# EriTrainer tenets

The non-negotiable principles for working on EriTrainer. Every widget, every
launch path, every model wiring answers to these. If a change violates a tenet,
the change is wrong, even if it compiles and the window opens.

---

## 1. The UI is a launcher; the backend is the trainer

The root egui app spawns the local EriDiffusion-v2 backend and tails trainer
stdout. The UI crate does **not** own a training loop, a model, or a kernel.
The repository now also vendors the trainer backend under `backend/`, and backend
trainer work belongs there.

The corollary: **never reimplement training logic in the UI, and never modify
backend trainers merely to make a UI widget look wired.** The standalone CLI
trainers stay usable on their own. If a launch needs something the trainer does
not expose, read the trainer's real CLI and map to it. If the trainer genuinely
needs a new capability, make that change in the backend as trainer work, not as
a UI illusion.

This is also why the UI crate has **no `eridiffusion-core` dependency**: a launcher
that drags in the CUDA/flame-core stack stops being buildable without a GPU and
starts being tempted to "just call the trainer directly." Keep the boundary.

Backend boundary: common lifecycle belongs in `trainer_pipeline.rs` /
`trainer_common.rs`; model-specific code owns model load, cache tensor schema,
forward/loss, sampling, and checkpoint keys.

---

## 2. No widget may silently lie (the honesty discipline)

A control that looks editable must actually affect the run, or be visibly marked
otherwise. This is inherited from the Mojo serenity-trainer and is the single
most important UI rule.

- A control the launched runner does NOT consume → suffix `[not wired]` AND name
  it in the pre-launch capability warning (`TrainConfig::ignored_lever_summary`,
  surfaced in the top bar). Pause/Sample/Save are `[not wired]` today.
- An unbuilt tab says so ("not yet built") — never a blank panel that reads as
  functional.
- A launch that can't be built (unwired model, missing required path) **fails
  loud**: status + log say why, and `has_running` stays false. A blocked launch
  must never animate the rail as if a run started.
- A model selector that doesn't change the launch identity is the canonical lie
  (it was a real block-bug: the Model dropdown was decorative until
  `apply_model_preset` was ported). Selecting a model MUST drive `model_type`/
  `backend_target` so the launcher, the live-target label, and the capability
  warning all follow it.

The decorative-widget bug is "wrong training, silently." Treat it as severe.

---

## 3. Separate registered, compile-checked, and GPU-verified

A model can be registered in the dispatcher, compile-checked as a real binary,
or GPU-verified with a finite-loss run. Do not collapse those into one status.

- Dispatcher registration means `train --model <id>` resolves to the intended
  `train_<m>` binary.
- Compile coverage means the real binary builds under the backend manifest.
- GPU verification means the real binary runs a step, emits the progress line,
  has finite loss/grad norm, and writes the expected checkpoint or artifact.
- "It should accept these flags" is a hypothesis. The clap parser's verdict is
  evidence. Read the `Args` struct; don't assume it matches another model.

---

## 4. Read each trainer's CLI — never template across models

The per-model CLI genuinely differs. Checkpoint flags alone span
`--transformer` / `--unet` / `--model` / `--dit-path` / `--model-path` / none.
`--config`, `--batch-size`, `--warmup-steps`, `--offload` are present for some
models and absent (clap-rejected) for others. Single-file vs directory
checkpoints differ.

So: **extract the real field list** (`awk` the `struct Args`, because the quick
`grep -B1` misses doc-commented args) before writing `<m>_args`. Copying another
model's mapping and tweaking the name is how you ship a launch that clap rejects.

---

## 5. Mirror the Mojo serenity-trainer; OneTrainer is the shape

The look and behavior target is the Mojo `serenity-trainer` UI (3-column shell,
the 12 nav sections, the form-panel tabs, the live rail), which itself mirrors
OneTrainer's tab layout. When unsure how something should look or where a field
belongs, the answer is in `/home/alex/serenity-trainer/src/serenity_trainer/ui/`
— read it, don't invent.

For backend trainer structure, OneTrainer is also the layer-boundary reference:
one shared trainer pipeline, per-model setup/model code, and no repeated
lifecycle boilerplate in every trainer.

---

## 6. Forbidden dependencies stay forbidden

`rustfrt`, `rust_frt`, and `rust-frt` are not allowed. Default trainer crates
must not gain an active `inference-flame` dependency. Known-good inference math
may be copied into `eridiffusion-core` when training needs it, but the trainer
runtime must remain self-contained.

---

## How the tenets compose

Tenet 1 sets the UI/backend boundary. Tenet 2 keeps the UI honest within it.
Tenets 3–4 keep each model wiring correct against reality. Tenet 5 keeps the
shape faithful. Tenet 6 protects the dependency boundary. A change that "works"
but violates 1, 2, or 6 is a regression even when the demo looks fine.
