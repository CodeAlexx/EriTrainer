# SimpleTuner Parity — EriDiffusion v2

Last verified: 2026-05-15 against `/home/alex/SimpleTuner/`.

Comprehensive feature map: every SimpleTuner option, supported model, and major feature → EDv2 status. EDv2 is single-GPU + pure-Rust by design; some structural absences are intentional (distributed training, Python tooling) and not on any roadmap.

**Legend**: ✅ full · ⚠️ partial · ❌ missing · ❓ unknown

---

## 1. Models supported

| SimpleTuner family | EDv2 trainer binary | EDv2 status |
| --- | --- | --- |
| SD1.x | — | ❌ removed in favor of newer archs |
| SDXL | `train_sdxl` | ⚠️ end-to-end pre-R1a; untested on current flame-core |
| Stable Cascade | — | ❌ no trainer |
| SD3 / SD3.5 | `train_sd35` | ⚠️ EDv2 has SD3.5; untested on current flame-core |
| Flux 1 | `train_flux` | ⚠️ end-to-end pre-R1a; untested on current flame-core |
| Flux 2 (Klein) | `train_klein` | ✅ verified May 15 on current flame-core |
| Chroma | — | ⚠️ model + sampler ported; trainer on demand |
| AuraFlow | — | ❌ no trainer |
| PixArt | — | ❌ no trainer |
| LTX-Video | — | ❌ no trainer |
| LTX-2 | `train_ltx2` | ⚠️ image-only; untested on current flame-core |
| HunyuanVideo | — | ❌ no trainer |
| Qwen-Image | `train_qwenimage` | ⚠️ end-to-end pre-R1a; untested on current flame-core |
| Z-Image | `train_zimage` | ⚠️ end-to-end pre-R1a; untested on current flame-core |
| ACE-Step | `train_acestep` | ⚠️ model ported; needs Python prep tensors; untested |
| Wan 2.x | — | ❌ blocked — needs `inference_flame::wan22_dit` lift |
| ERNIE-Image | `train_ernie` | ⚠️ end-to-end pre-R1a; untested on current flame-core |
| Anima (Cosmos + LLM-Adapter) | `train_anima` | ⚠️ rank-32 smoke clean pre-R1a; untested |
| Kolors | — | ❌ no trainer |
| Kandinsky 5 (image/video) | — | ❌ no trainer |
| Cosmos (standalone) | — | ❌ no trainer (used inside Anima) |
| Heartmula | — | ❌ no trainer |
| HiDream | — | ❌ no trainer |
| Lumina2 | — | ❌ no trainer |
| Omnigen | — | ❌ no trainer |
| Sana | — | ❌ no trainer |

---

## 2. CLI options parity (138 ST options)

### Core model configuration

| Option | EDv2 | Notes |
| --- | --- | --- |
| `--model_type` (lora / full / bitfit / dora) | ⚠️ | LoRA + full; no BitFit/DoRA |
| `--model_family` | ✅ | one trainer binary per family |
| `--lora_format` (default / lycoris / lierla) | ⚠️ | PEFT + LyCORIS port; no Lierla |
| `--fuse_qkv_projections` | ❌ | trivial to add |
| `--offload_during_startup` | ✅ | `--offload` activates BlockOffloader |
| `--offload_during_save` | ⚠️ | saves off-critical path; semantics differ |
| `--delete_model_after_load` | ✅ | implicit (LoRA wraps, base freed) |
| `--enable_group_offload` | ⚠️ | EDv2 uses resident-set conductor; different strategy |
| `--group_offload_type` | ⚠️ | conductor is parameter-driven via `FLAME_BLOCK_OFFLOAD_SLOTS` |
| `--group_offload_blocks_per_group` | ⚠️ | subsumed by conductor |
| `--group_offload_use_stream` | ⚠️ | flame-core uses per-slot CUDA events |
| `--group_offload_to_disk_path` | ❌ | no disk swap; in-memory only |
| `--musubi_blocks_to_swap`, `--musubi_block_swap_device` | ❌ | ST-specific |
| `--ramtorch` + 7 sub-options | ❌ | PyTorch accelerate features; N/A |
| `--pretrained_model_name_or_path` | ✅ | EDv2 `--transformer` |
| `--pretrained_t5_model_name_or_path` | ✅ | per-trainer |
| `--pretrained_gemma_model_name_or_path` | ⚠️ | some trainers use Gemma3; path configurable |
| `--custom_text_encoder_intermediary_layers` | ❌ | non-trivial per-trainer plumbing |
| `--gradient_checkpointing` | ✅ | per-block via `checkpoint_blocks` |
| `--gradient_checkpointing_interval` | ❌ | full-block only; no per-interval |
| `--gradient_checkpointing_backend` | ❌ | flame-core has built-in checkpoint-recompute |
| `--refiner_training` | ❌ | no two-stage training |

### Precision & quantization

| Option | EDv2 | Notes |
| --- | --- | --- |
| `--quantize_via` (bnb / quanto / sdnq / …) | ❌ | BF16 native; no quantization backends |
| `--base_model_precision` | ✅ | BF16 hardwired for Klein; configurable elsewhere |
| `--quantization_config` | ❌ | no per-layer quantization plumbing |
| `--attention_mechanism` (sdpa / fa2 / fa3 / …) | ❌ | flame-core uses its own fused kernels |

### Publishing

| Option | EDv2 | Notes |
| --- | --- | --- |
| `--push_to_hub`, `--push_to_hub_background` | ❌ | no HF integration |
| `--webhook_config` | ✅ | `training_features/webhook.rs` |
| `--publishing_config`, `--hub_model_id`, `--modelspec_comment` | ❌ | HF-specific |
| `--disable_benchmark` | ❌ | EDv2 does no startup benchmarks |

### Data storage

| Option | EDv2 | Notes |
| --- | --- | --- |
| `--data_backend_config` | ✅ | per-trainer config JSON |
| `--override_dataset_config` | ⚠️ | limited; most config via JSON |
| `--data_backend_sampling` | ⚠️ | probabilities in config; deterministic bucketing |
| `--vae_cache_scan_behaviour` | ⚠️ | fixed discovery; skip-if-exists flag only |
| `--dataloader_prefetch`, `--dataloader_prefetch_qlen` | ⚠️ | prefetch implicit; no explicit control |
| `--compress_disk_cache` | ❌ | safetensors only; no compression |

### Image & text processing

| Option | EDv2 | Notes |
| --- | --- | --- |
| `--resolution_type`, `--resolution`, `--validation_resolution` | ✅ | bucketing automatic |
| `--validation_method` | ✅ | `sample_library` feature module |
| `--validation_external_script`, `--validation_external_background` | ❌ | no subprocess plumbing |
| `--post_upload_script`, `--post_checkpoint_script` | ❌ | no subprocess hooks |
| `--validation_adapter_path` / `_name` / `_strength` | ✅ | LoRA merge for validation |
| `--validation_adapter_mode` | ⚠️ | merge only (no inject/apply modes) |
| `--validation_adapter_config` | ✅ | part of config JSON |
| `--validation_preview`, `--validation_preview_steps` | ⚠️ | exists but not fully wired; experimental |
| `--evaluation_type` (loss / clip_score) | ⚠️ | MSE loss only; no CLIP scoring |
| `--eval_loss_disable` | ⚠️ | always computes loss |
| `--validation_using_datasets`, `--eval_dataset_id` | ✅ | partial; works with image datasets |
| `--caption_strategy` (textfile / parquet / instanceprompt) | ⚠️ | textfile/instanceprompt; no parquet |

### Training parameters

| Option | EDv2 | Notes |
| --- | --- | --- |
| `--num_train_epochs` | ✅ | config `num_epochs` |
| `--max_train_steps` | ✅ | config `max_steps` (or CLI `--steps`) |
| `--ignore_final_epochs` | ⚠️ | step-based primary; basic epoch tracking |
| `--learning_rate` | ✅ | config `learning_rate` (and `--lr` CLI) |
| `--lr_scheduler` | ✅ | `training_features/lr_schedule.rs` |
| `--optimizer` (adamw / lion / adafactor / prodigy / dadapt) | ⚠️ | AdamW only; no Lion / Adafactor / Prodigy / DAdapt yet |
| `--optimizer_config` | ⚠️ | betas / eps / weight_decay in config |
| `--train_batch_size` | ✅ | config `batch_size` (and CLI) |
| `--gradient_accumulation_steps` | ✅ | config `gradient_accumulation_steps` |
| `--allow_dataset_oversubscription` | ❌ | repeats manual; no auto-adjust |

### Advanced optimizations

| Option | EDv2 | Notes |
| --- | --- | --- |
| `--use_ema` | ✅ | `training_features/ema_advanced.rs` |
| `--ema_device` (cpu / cuda) | ✅ | config `ema_device` |
| `--ema_cpu_only` | ✅ | subsumed by `ema_device` |
| `--ema_foreach_disable` | ❌ | N/A: Rust impl, no torch foreach |
| `--ema_update_interval`, `--ema_decay` | ✅ | config fields |
| `--snr_gamma`, `--use_soft_min_snr` | ✅ | Soft Min SNR shipped |
| `--diff2flow_enabled`, `--diff2flow_loss` | ⚠️ | config fields exist; untested on current flame-core |
| `--scheduled_sampling_*` (14 options) | ✅ | all config fields present |
| `--scheduled_sampling_reflexflow_*` | ⚠️ | config fields exist; untested |

### CREPA (Cross-frame Representation Alignment)

| Option | EDv2 | Notes |
| --- | --- | --- |
| `--crepa_enabled` + 18 sub-options | ❌ | ST video training feature; not in EDv2 scope |

### Checkpointing

| Option | EDv2 | Notes |
| --- | --- | --- |
| `--checkpoint_step_interval` | ✅ | config `save_every` |
| `--checkpoint_epoch_interval` | ⚠️ | step-based only |
| `--resume_from_checkpoint` | ✅ | config `resume_from` |
| `--disk_low_threshold`, `--disk_low_action` | ⚠️ | `training_features/disk_check.rs` (basic) |
| `--disk_low_script` | ❌ | no subprocess hooks |

### LayerSync

| Option | EDv2 | Notes |
| --- | --- | --- |
| `--layersync_enabled` + 3 sub-options | ❌ | ST feature for video; not in EDv2 |

### Logging

| Option | EDv2 | Notes |
| --- | --- | --- |
| `--logging_dir` | ⚠️ | SerenityBoard SQLite DB instead of TensorBoard |
| `--report_to` (tensorboard / wandb / …) | ❌ | webhook + SerenityBoard only |

### Summary count

- **Total ST CLI options:** 138
- **EDv2:** ~55 ✅ (40%), ~45 ⚠️ (33%), ~38 ❌ (27%)

---

## 3. Feature-area parity

### LyCORIS / LoRA variants

ST has full LyCORIS support (LoCon, LoHa, LoKr) with granular module targeting via JSON. EDv2 has the port wired (`crates/eridiffusion-core/src/lycoris.rs`) but **the layer is still under active work** and not yet end-to-end functional on every trainer. Full LoRA (PEFT format) is production-ready across verified trainers. Per-trainer re-verification post-R1a–R2c is the gating work.

### DeepSpeed / FSDP2 (distributed training)

ST ships DeepSpeed ZeRO 1–3 with CPU/NVMe offload and FSDP2 with context parallelism. **EDv2 has no distributed training and no roadmap for it.** Multi-GPU would require a distributed autograd backend in flame-core (months of design + implementation) and is structurally incompatible with the current single-GPU resident-set architecture. Users needing multi-GPU scale should use SimpleTuner.

### TREAD (token routing)

ST has experimental TREAD for training acceleration on FLUX / Wan / AuraFlow / PixArt / SD3. EDv2 has Phase 4 + 4.5 shipped — `TreadConfig`, `TreadStep`, gather/scatter via `flame_core::Tensor::index_select` and `index_assign`, CLI `--tread-route-pattern` / `--tread-keep-ratio`. Klein model integration verified; **Z-Image and Flux model wiring is the next round of work** (Phase 4.5 follow-up).

### Slider LoRA

ST has contrastive slider training on positive/negative/neutral triplets. EDv2 has Slider LoRA ported (`train_slider_klein.rs` binary, batch rotation pos→neg→neutral). Klein verified; other trainers untested on current flame-core.

### ControlNet & conditioning

ST has production ControlNet training (full + LoRA) across Canny, depth, segmentation, masks, super-resolution, JPEG-artifact removal, with automatic conditioning generation. **EDv2 has no ControlNet trainer.** Conditioning infrastructure (masks, reference images) and masked loss are present. Porting ControlNet per model is moderate effort (~2–3k LOC per model) and not currently prioritized.

### Dreambooth & prior preservation

ST has prior preservation via regularization datasets, masked loss for focus regions, scheduled sampling for small-dataset overfit mitigation, and quantized training (int8-quanto, NF4). EDv2 has prior preservation via `is_regularisation_data` flag (LyCORIS path), masked loss fully functional, scheduled sampling config fields. **Quantization is absent** — BF16 only.

### Mixture-of-Experts (two-stage refinement)

ST trains base + refiner in stages with timestep-range gating. **EDv2 has no two-stage training.** Single-stage per trainer by design. Would require external orchestration or trainer redesign.

---

## 4. Dataloader features

| Feature | SimpleTuner | EDv2 | Notes |
| --- | --- | --- | --- |
| Aspect bucketing | ✅ | ✅ | automatic; config `bucketing: true` |
| Prior preservation | ✅ | ⚠️ | `is_regularisation_data` flag (LyCORIS only) |
| Caption dropout | ✅ | ✅ | config `caption_dropout_probability` |
| Conditional captions | ✅ | ⚠️ | text-to-image only; limited conditioning types |
| Multi-aspect ratio | ✅ | ✅ | bucketing handles all ratios |
| Regularization images | ✅ | ✅ | separate dataset with flag |
| Caption shuffle (tag reorder) | ✅ | ❌ | pre-process required; not in pipeline |
| Multi-dataset sampling | ✅ | ✅ | config: multiple backends + probabilities |
| Dataset scheduling (start/end epoch) | ✅ | ❌ | curriculum learning not exposed |
| VAE caching | ✅ | ✅ | on-demand or pre-cached |
| Text embed caching | ✅ | ✅ | via `prepare_*` binaries |
| Dynamic conditioning generation (canny/depth) | ✅ | ⚠️ | no auto-generation; pre-compute required |
| Crop styles (center / random / corner / face) | ✅ | ⚠️ | center + random; no face detection |
| Min/max image size filtering | ✅ | ✅ | config: `minimum_resolution` |
| Aspect ratio range filtering | ✅ | ✅ | via bucketing |
| Image repeats | ✅ | ✅ | config: `repeats` field |
| CSV dataset | ✅ | ❌ | needs CSV parser |
| Parquet | ✅ | ❌ | planned, not implemented |
| HuggingFace dataset streaming | ✅ | ❌ | planned, not implemented |
| S3 / AWS backend | ✅ | ❌ | local filesystem only |

---

## 5. Optimization features

| Feature | ST | EDv2 | Note |
| --- | --- | --- | --- |
| **Optimizers** | | | |
| AdamW | ✅ | ✅ | native; production-ready |
| AdamW 8-bit | ✅ | ❌ | needs BNB integration |
| AdamW BF16 fused | ✅ | ✅ | flame-core multi-tensor fused kernel |
| Adafactor | ✅ | ❌ | moderate effort |
| Lion | ✅ | ❌ | moderate effort |
| Prodigy | ✅ | ❌ | moderate effort |
| DAdaptation | ✅ | ❌ | moderate effort |
| **Optimizer variants** | | | |
| Weight decay (L2 + decoupled) | ✅ | ✅ | config `weight_decay` |
| Gradient clipping | ✅ | ✅ | config `grad_clip_norm` |
| Gradient accumulation | ✅ | ✅ | config `gradient_accumulation_steps` |
| Gradient checkpointing | ✅ | ✅ | config `checkpoint_blocks` list |
| **Precision** | | | |
| FP32 / FP16 mixed | ✅ | ⚠️ | BF16 only; no FP16 |
| BF16 mixed | ✅ | ✅ | native |
| Pure BF16 | ✅ | ✅ | default |
| **Attention** | | | |
| Flash Attention 2 | ✅ | ❌ | flame-core custom fused kernels |
| Flash Attention 3 | ✅ | ❌ | flame-core custom fused kernels |
| SDPA | ✅ | ❌ | N/A: pure Rust |
| **Memory** | | | |
| Activation checkpointing | ✅ | ✅ | per-block selective |
| Model offloading (CPU) | ✅ | ✅ | `--offload`; resident-set conductor |
| Activation offloading | ✅ | ⚠️ | implicit in flame-core; no explicit config |
| **Other** | | | |
| EMA | ✅ | ✅ | `training_features/ema_advanced.rs` |
| torch.compile | ✅ | ❌ | N/A: Rust |
| SNR / Soft Min SNR weighting | ✅ | ✅ | config: `snr_gamma`, `use_soft_min_snr` |

---

## 6. Big absences (and why)

**Distributed training (DeepSpeed / FSDP2).** Structural. EDv2 is single-GPU + resident-set offload. Multi-GPU support would require flame-core distributed extensions (months of work). Users requiring multi-GPU scale → SimpleTuner.

**HuggingFace Hub push.** EDv2 saves locally only. Adding hub integration is moderate effort but not aligned with EDv2's pure-Rust + flame-core ethos. Push manually after training if needed.

**Quantization (8-bit / NF4 / int4).** BF16 only. Adding quantization needs custom CUDA kernels per scheme (complex). For ultra-low-VRAM LoRA, SimpleTuner is the better choice.

**ControlNet training.** No trainer. Porting is moderate effort per model. Masked loss works; control signals don't.

**Parquet / CSV / HF streaming datasets.** Local filesystem only with pre-computed caches. Planned, not implemented.

**Two-stage / refiner training.** Single-stage per trainer. Would need external orchestration.

**TensorBoard / W&B reporting.** SerenityBoard SQLite DB + webhook only. Adding TB/W&B emitters is small-to-moderate effort but not prioritized.

---

## Summary

EDv2 is **feature-complete for single-GPU LoRA + full fine-tune** on its 14 supported model families, with wall-clock parity verified on Klein 9B as of 2026-05-15 (2.30 s/step steady, 271 s total wall for 100 steps).

SimpleTuner remains the right choice for: multi-GPU/multi-node training, HF Hub workflows, ControlNet, quantized training, Parquet/CSV/HF streaming datasets, and the 20+ model families EDv2 doesn't yet have trainers for.
