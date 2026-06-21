# EriTrainer Backend

This directory is the local trainer runtime owned by EriTrainer.

Layout:

- `EriDiffusion-v2/` - Rust trainer workspace copied from EDv2.
- `flame-core/` - local GPU tensor/autograd backend. This is the critical
  training acceleration layer.
- `eri-lycoris/lycoris-rs/` - local LyCORIS/LoRA support.
- `cudarc-pinctx/` - pinned `cudarc` patch used by `flame-core` and trainers.

The unified `train` front door is intentionally lightweight:
`EriDiffusion-v2/crates/eritrainer-dispatch` depends on
`eritrainer-registry`, not on `flame-core`. Listing models and resolving
dry-run commands should not compile CUDA; only the selected model-specific
`eridiffusion-cli` trainer does.

Current registered trainer IDs: `acestep`, `anima`, `asymflow`, `chroma`,
`ernie`, `flux`, `hidream_o1`, `ideogram4`, `klein`, `l2p`, `ltx2`,
`qwenimage`, `sd35`, `sdxl`, `slider_klein`, `u1`, `wan22`, `zimage`.

Boundary rule: default trainer crates must not depend on `inference-flame`.
Known-good inference code can be copied into `eridiffusion-core` when training
needs it, but the trainer runtime should compile and launch without an
`inference-flame` crate dependency. Samplers live under
`eridiffusion-core::sampler`.

Modular trainer rule: follow the OneTrainer boundary.

- Shared trainer pipeline code belongs in `eridiffusion-cli/src/trainer_*`.
  It owns launch lifecycle, config/preflight, cache discovery, device/runtime
  setup, metrics, progress, save cadence, and sampler orchestration.
- Model-specific code stays in the model trainer/model modules: model loading,
  cache tensor schema, forward/loss construction, model-specific validation or
  sampling, and checkpoint key layout.
- `trainer_common.rs` is the compatibility facade for existing `train_*`
  binaries while they migrate toward `trainer_pipeline.rs`.

Current shared pipeline coverage:

- all trainer bins compile through the local backend manifest;
- all `train_*.rs` bins route per-step progress/timing/loss accounting through
  `ManualTrainLoopRun` or `run_manual_train_loop`;
- raw board-open, progress-line, completion-status, backward, optimizer-step,
  EMA, grad-policy, gradient-clip helper, and checkpoint-save calls are
  centralized in `trainer_common.rs` / `trainer_pipeline.rs`;
- representative dispatcher GPU smoke passed for SDXL on 2026-06-21
  (`step 1/1`, finite loss/grad norm, final LoRA + ComfyUI companion saved).
