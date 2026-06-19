# EriTrainer tenets

The non-negotiable principles for working on EriTrainer. Every widget, every
launch path, every model wiring answers to these. If a change violates a tenet,
the change is wrong, even if it compiles and the window opens.

---

## 1. EriTrainer is a launcher, not a trainer

EriTrainer spawns the existing EriDiffusion-v2 `train_*` binaries and tails their
stdout. It does **not** own a training loop, a model, or a kernel.

The corollary: **never reimplement training logic here, and never modify the EDv2
trainers to suit the UI.** The standalone CLI trainers stay usable on their own
("we may need that" — the directive that created this repo). If a launch needs
something the trainer doesn't expose, the fix is to read the trainer's real CLI
and map to it — or, if genuinely missing, to change the trainer in its own repo
with its own review, not to fake it in the UI.

This is also why EriTrainer has **no `eridiffusion-core` dependency**: a launcher
that drags in the CUDA/flame-core stack stops being buildable without a GPU and
starts being tempted to "just call the trainer directly." Keep the boundary.

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

## 3. Verify every model against its real binary (measurement beats assertion)

A model is not "wired" because `<m>_args` compiles. It is wired when the real
`train_<m>` binary **accepts the generated argv and runs a step**.

- Each wired model has a `parses_real_train_<m>_run_line` test built from a line
  captured from an actual run — not a hand-written guess.
- Each model was proven end-to-end: compile `train_<m>` + `prepare_<m>`, build a
  small real cache, run `--steps 3`, observe decreasing/finite loss + a written
  LoRA. Argv-acceptance alone (reaching the cache stage without "unexpected
  argument") is the minimum bar; a real step is the goal.
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

---

## How the tenets compose

Tenet 1 sets the boundary (launcher). Tenet 2 keeps the UI honest within it.
Tenets 3–4 keep each model wiring correct against reality. Tenet 5 keeps the
shape faithful. A change that "works" but violates 1 or 2 is a regression even
when the demo looks fine.
