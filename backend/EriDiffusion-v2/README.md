# EriDiffusion v2

Pure-Rust diffusion-model training framework. Built on [`flame-core`](https://github.com/CodeAlexx/Flame) for the tensor + autograd layer. No Python at runtime.

## Why

A pure-Rust trainer for diffusion models, against custom CUDA kernels, with no Python on the runtime path. The full pipeline — data prep, autograd, optimizer, sampler, save/resume — lives in this repo and `flame-core`. The design target is large-model LoRA training on a single 24 GB consumer GPU at wall-clock parity with the fastest established trainers.

As of 2026-05-15, on a 24 GB consumer GPU, Klein 9B (FLUX.2-klein-base-9B) LoRA training runs at **2.30 s/step steady-state** (steps 11–100, 100-step run), **271 s total wall** including model load + offloader init + Adam prewarm. Settings: dataset 118 samples at 512 with aspect bucketing, LoRA r16/α16, AdamW (β1=0.9, β2=0.999, ε=1e-8, wd=0.01), lr 3e-5, batch 1, bf16, `--offload`. See [`flame-core/docs/FLAME_MODULES.md`](https://github.com/CodeAlexx/Flame/blob/main/docs/FLAME_MODULES.md) → `offload/mod.rs` for telemetry.

## Companion repos

- [`flame-core`](https://github.com/CodeAlexx/Flame) — tensor library, autograd v3, CUDA kernels, optimizer states, block offloader, static-slab allocator. EDv2 is its biggest consumer.
- [`inference-flame`](https://github.com/CodeAlexx/inference-flame) — inference-only counterpart. Shares model definitions with EDv2 trainers.
- [`cudarc-pinctx`](https://github.com/CodeAlexx/cudarc-pinctx) — vendored cudarc 0.11.9 + flame-core-specific patches (external-pointer Drop hook, sync-alloc escape hatch, pinned-context plumbing).
This also applies to flame-inference, some may work some may not. a massive speed gains across the entire engine.
## Trainers

> **EriTrainer embedded-backend status (2026-06-21):** this workspace is now
> copied under `/home/alex/EriTrainer/backend/EriDiffusion-v2` and registered
> behind a lightweight `train --model <id>` dispatcher. The dispatcher and
> registry do not link `flame-core`; the selected model-specific trainer binary
> does. All 18 `train_*` targets compile through the local manifest. Shared
> trainer lifecycle code is centralized in `eridiffusion-cli/src/trainer_pipeline.rs`
> and `trainer_common.rs`; model-specific files keep model load, cache schema,
> forward/loss, sampling, and checkpoint key layout.
>
> Default trainer crates must not gain an active `inference-flame` dependency.
> Sampler/model helper code needed by training belongs in `eridiffusion-core`.
> `rustfrt` / `rust_frt` / `rust-frt` is forbidden.

| Model | Binaries (under `crates/eridiffusion-cli/src/bin/`) | Last known status | Verified against current flame-core? |
| --- | --- | --- | --- |
| ACE-Step | `train_acestep` | registered + compile-checked; trainer consumes upstream ACE-Step tensors | compile-only |
| Anima (Cosmos-Predict2 + LLM-Adapter) | `train_anima`, `prepare_anima`, `sample_anima` | registered + compile-checked; prior rank-32 smoke clean | prior smoke |
| AsymFlow | `train_asymflow`, `prepare_asymflow` | registered + compile-checked; AsymFlow helpers local in core | compile-only |
| Chroma | `train_chroma`, `prepare_chroma`, `sample_chroma` | registered + compile-checked; local sampler | prior smoke |
| ERNIE-Image | `train_ernie`, `prepare_ernie`, `sample_ernie` | registered + compile-checked | prior smoke |
| FLUX.1 | `train_flux`, `prepare_flux`, `sample_flux` | registered + compile-checked; local sampler | compile-only |
| HiDream-O1 | `train_hidream_o1`, `prepare_hidream_o1` | registered + compile-checked; helpers local in core; sampler still pending | smoke only |
| Ideogram-4 | `train_ideogram`, `prepare_ideogram` | registered + compile-checked | compile-only |
| **Klein (FLUX.2)** | `train_klein`, `prepare_klein`, `sample_klein` | registered + compile-checked; 100-step Klein 9B + offload previously verified | verified |
| L2P | `train_l2p`, `prepare_l2p` | registered + compile-checked; trainer includes local L2P sampling helpers | prior smoke |
| LTX-2 | `train_ltx2`, `prepare_ltx2`, `sample_ltx2` | registered + compile-checked; local sampler | compile-only |
| Qwen-Image-2512 | `train_qwenimage`, `prepare_qwenimage`, `sample_qwenimage` | registered + compile-checked; local sampler | compile-only |
| SD3.5 Medium | `train_sd35`, `prepare_sd35`, `sample_sd35` | registered + compile-checked; local sampler | prior smoke |
| **SDXL** | `train_sdxl`, `prepare_sdxl`, `sample_sdxl` | registered + compile-checked; 2026-06-21 dispatcher smoke saved LoRA artifacts | verified smoke |
| Slider Klein | `train_slider_klein`, `prepare_klein`, `sample_klein` | registered + compile-checked; shared Klein cache/sampler | compile-only |
| SenseNova U1 | `train_u1`, `sample_u1` | registered + compile-checked; folder-mode trainer | compile-only |
| Wan 2.2 | `train_wan22`, `prepare_wan22`, `sample_wan22` | registered + compile-checked; local model/sampler modules | compile-only |
| Z-Image | `train_zimage`, `prepare_zimage`, `sample_zimage` | registered + compile-checked; local sampler | prior smoke |

LoRA is the production target across all trainers that have shipped. **LyCORIS variants** (LoCon, LoHa, LoKr) are wired in via the in-repo `crates/eridiffusion-core/src/lycoris.rs` port, but the layer is still under active work and not yet end-to-end functional on every trainer — treat LyCORIS support as in-progress. Full fine-tune is supported on most trainers.

## What's actually in this repo

```
crates/
  eridiffusion-core/        # the engine
    src/models/             # per-model DiT/UNet definitions
    src/encoders/           # VAE encoders/decoders + text encoders
                            #   (Qwen3, Qwen2.5-VL, Mistral-3B, Gemma3,
                            #    T5-XXL, CLIP-L/G, BPE/byte fallback)
    src/sampler/            # flow-matching schedules, Euler + CFG denoise,
                            # CFG-Zero* gates, validation/sample harness
    src/training/           # BlockOffloader integration, activation offload,
                            #   checkpoint save/resume, EMA, LR schedule,
                            #   logging (SerenityBoard / scalar SQLite DB),
                            #   per-step prewarm, gradient clip
    src/data/               # bucketed latent dataset reader (eats prepare_*
                            #   output, hands batches to the trainer)
    src/lora/               # LoRA wrapper, save format = PEFT/edv2-reference
    src/lycoris.rs          # LoCon/LoHa/LoKr port (in-tree)
  eridiffusion-cli/         # the binaries
    src/bin/                # train_*, prepare_*, sample_*, convert_* tools
```

The pipeline is always:

```
prepare_<model> <raw images + captions>  →  cache/<run>/*.safetensors
                                          (latents + encoded text per sample)
                       ↓
train_<model> --cache-dir cache/<run>     →  output/<run>/*.safetensors
              --transformer <base model>     (LoRA / checkpoint snapshots)
                       ↓
sample_<model> --lora <produced LoRA>     →  PNG samples
```

## Build

CUDA-enabled host. Requires libtorch for some auxiliary toolchain bits at runtime (one binary path during prep uses it).

```bash
cargo build --release
```

Per-binary:

```bash
cargo build --release --bin train_klein
cargo build --release --bin prepare_klein
```

At runtime:

```bash
export LD_LIBRARY_PATH=/path/to/libtorch/lib
export FLAME_ALLOC_POOL=1
```

## Run — quickstart: Klein 9B LoRA

```bash
# 1) Prepare cache (~3 min for 118 samples at 512 with aspect bucketing).
target/release/prepare_klein \
  --input-dir /path/to/images \
  --output-dir cache/myrun \
  --vae-ckpt /path/to/flux2-vae.safetensors \
  --qwen3 /path/to/Qwen3-8B/snapshot \
  --tokenizer-path /path/to/Qwen3-8B/snapshot/tokenizer.json \
  --resolution 512 --bucketing

# 2) Train (100 steps, ~4 min wall on a 24 GB consumer GPU).
target/release/train_klein \
  --config configs/klein9b_alina.json \
  --transformer /path/to/flux-2-klein-base-9b.safetensors \
  --cache-dir cache/myrun \
  --output-dir output/myrun \
  --steps 100 \
  --rank 16 --lora-alpha 16.0 \
  --batch-size 1 --offload \
  --sample-every 0 --warmup-steps 100
```

`--offload` activates flame-core's `BlockOffloader` (resident-set conductor) — required for Klein 9B on 24 GB. The config file carries lr, optimizer betas, weight decay, timestep distribution, MSE/MAE weights, clip-grad-norm.

`prepare_*` and `sample_*` arguments mirror the trainer for each model.

## Performance & memory innovations

What drives EDv2's current per-step wall on this hardware:

### 1. Static-slab allocator with strict-reset guard (R1a–R2c, May 2026)

Transient per-step BF16/F32 allocations land in a single large `cudaMalloc`'d slab (`flame_core::static_slab_v2::StaticSlabAllocator`) instead of cycling through cudart. A `StepSlabGuard` RAII type wraps the trainer's per-step body — at scope exit, the slab cursor is reset; a strict invariant panics if any tensor leaked the scope. Resolves the BF16 use-after-free bug class structurally; no `cudaFree` on the hot path.

Adam `m`/`v` is **pre-warmed** before the slab env-flag activates so optimizer state lands in the legacy pool. The pre-warm hook (`Adam::ensure_state_initialized` / `AdamW::ensure_state_initialized`) is called from `train_klein.rs` right before `FLAME_USE_STATIC_SLAB=1` is set.

### 2. Resident-set conductor for block offloading (R2c-perf)

`flame_core::offload::BlockOffloader` replaces the original two-slot ping-pong with a resident-set conductor policy:

- Knows forward AND backward traversal — pre-stages block N-1 during backward replay of block N.
- Fractional VRAM budget instead of fixed slot count (`FLAME_BLOCK_OFFLOAD_SLOTS` is now a ceiling, not the policy itself).
- Multiple async H2D prefetches in flight, queued by per-slot CUDA events.
- Lazy eviction via `cudaStreamWaitEvent` on previous `compute_done` — no host-side `cudaEventSynchronize`.

Telemetry on a 6-step run: 258 AwaitHit / 0 AwaitMiss.

HiDream-O1 note (2026-05-21): O1 uses the structured prefix-causal/full
attention route corresponding to ai-toolkit's `use_flash_attn=True` training
path, but production parity is still red. The fixed-input gate now fails
honestly when forward/objective/per-layer metrics fail; current first failure
is `forward::layer00.attn_out` after `layer00.sdpa_out` remains within the
current threshold. The pinned Full-model loss is close (`rel ~= 8.0e-5`) but
above the strict `1e-5` gate, so do not treat the trainer as validated or run
the 1000-step `/eri2` proof yet. Production defaults remain velocity loss,
the ai-toolkit/public O1 LoRA surface (252 language-layer adapters plus the
five O1 head adapters), and `--export-scale=1.0`; `--no-resident-lora` is only
a transformer-only ablation.

The remaining speed gap is not a materialized-mask or block-loader tuning issue:
`checkpoint_offload_boundary` stores boundary inputs and recomputes the 36 Qwen
decoder blocks during backward. Speed work after LoRA validity should reduce
checkpoint coverage when VRAM allows or add a true no-recompute
activation/sub-tape offload path.

### 3. Frozen-weight gradient skip (autograd, May 15)

`Op::MatMul`, `Op::Mul`, and `RmsNorm` backward all gate per-operand gradient computation on `requires_grad()`. Klein's base-model linears and frozen RMSNorm scales now do zero gradient work — only LoRA A/B parameter gradients and the activation-gradient path remain. Verified by the `rms_norm_vs_primitive_zimage` parity test; no impact on LoRA learnability.

### 4. Range-aware BF16 trap (R2c)

The diagnostic trap that helped land R1a–R2c is range-aware: views/mid-allocation pointers validated by `Tensor::cat` resolve against the live BF16 allocation, not stale exact-pointer history under cudart address reuse. Available for any future regression: `FLAME_POOL_TRAP_BF16=1 FLAME_POOL_TRAP_BACKTRACE=1`.

### 5. Adam multi-tensor fused kernel + checkpoint-recompute prefetch hook

Multi-tensor AdamW step uses one fused NVRTC kernel for all params; backward-time prefetch reads `AutogradContext::is_checkpoint_recompute()` to trigger H2D in the backward direction without disturbing forward.

See [`flame-core/docs/FLAME_MODULES.md`](https://github.com/CodeAlexx/Flame/blob/main/docs/FLAME_MODULES.md) and [`flame-core/docs/FLAME_CONVENTIONS.md`](https://github.com/CodeAlexx/Flame/blob/main/docs/FLAME_CONVENTIONS.md) for the full design and gotchas.

## Status

Active. The redesign documented in `flame-core/HANDOFF_2026-05-15_OT_STATIC_SLAB_REDESIGN.md` shipped end-to-end on May 15. Klein 9B is the verified workload; Z-Image inference is also clean.

## License

MIT.
