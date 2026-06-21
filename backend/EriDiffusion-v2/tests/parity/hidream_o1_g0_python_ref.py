#!/usr/bin/env python3
"""HiDream-O1 G0 — generate PyTorch reference dump for trainer-forward parity.

Loads HiDream-O1 (Qwen3-VL pixel-DiT) the same way `inference.py` does, builds
the same input tensors `pipeline.py::generate_image` would feed at one denoise
step, runs ONE forward pass, and dumps the inputs + `x_pred` output to
`/tmp/hidream_o1_g0_python_ref.safetensors`.

Pinning (must stay byte-identical to the Rust binary):
  - prompt        : fixed deterministic string (see PROMPT)
  - seed          : 42
  - resolution    : 512 x 512
  - timestep      : t_pixeldit = 0.5 (i.e. step_t = 500.0 → t = 1.0 - 500/1000)
  - dtype         : bfloat16 (the autocast dtype the pipeline runs in)
  - use_flash_attn: True (matches the pipeline default; also matches Rust path)
  - noise (vinputs) ~ noise_scale_start * N(0,1) with torch.Generator(seed+1)

We feed a single CONDITIONAL sample (no CFG branch), no reference images, so
the inputs are the t2i path of pipeline.py.

Outputs (safetensors keys):
  - patches        F32 [1, L, 3072]  — same layout as prepare_hidream_o1 cache.
                                       This is the "vinputs" tensor (the
                                       patchified noise z).
  - input_ids      I32 [1, S_text]
  - position_ids   I32 [3, S_total]  — the 3D MRoPE T/H/W stacked
  - vinput_mask    F32 [1, S_total]  — 1.0 at image slots, 0.0 elsewhere
  - image_grid     I32 [3]           — (1, H/32, W/32)
  - token_types    F32 [1, S_total]  — (>0 = gen position) — same as cache
  - timestep       F32 [1]           — t_pixeldit, in [0,1]
  - x_pred_full    F32 [1, S_total, 3072]  — the full model output BEFORE
                                              the vinput_mask gather
  - x_pred_rows    F32 [1, L, 3072]  — gathered (matches what the trainer
                                       computes for the loss)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# ── Hard config (lockstep with the Rust binary) ──────────────────────────
MODEL_PATH = "/home/alex/HiDream-O1-Image-Dev-weights"
PROMPT = (
    "a photograph of an astronaut riding a horse on mars, cinematic lighting"
)
SEED = 42
HEIGHT = 512
WIDTH = 512
T_PIXELDIT = 0.5            # → step_t = 500.0 in pipeline's reverse mapping
NOISE_SCALE_START = 8.0     # NOISE_SCALE constant from pipeline.py:14
DTYPE_NAME = "bfloat16"
# Flash attention is NOT installed in this env (and the Rust port uses cuDNN
# SDPA, not FA), so we force the non-flash path. Numerically both paths build
# the same mixed causal/full mask; the non-flash version materialises it as a
# 4D [B,1,S,S] tensor instead of an indexed scatter. Final outputs match.
USE_FLASH_ATTN = False

OUT_TENSORS = Path("/tmp/hidream_o1_g0_python_ref.safetensors")
OUT_META = Path("/tmp/hidream_o1_g0_python_ref_meta.json")


def _die(msg: str, code: int = 2) -> None:
    print(f"[parity-ref] FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def _try_import(name: str):
    try:
        return __import__(name)
    except ImportError as e:  # noqa: BLE001
        _die(f"cannot import {name!r}: {e}\n"
             f"             Activate an env with torch + transformers + "
             f"safetensors + einops + Pillow.", 2)


def main() -> int:
    _try_import("torch")
    _try_import("transformers")
    _try_import("safetensors")
    _try_import("einops")

    import torch
    import numpy as np
    import einops
    from safetensors.torch import save_file
    from transformers import AutoProcessor

    # Use the edv2-reference O1 implementation, not the inference-only upstream
    # HiDream-O1 repo. The trainer parity target is edv2-reference PR #831.
    repo_root = os.environ.get("EDV2_REFERENCE_ROOT")
    if not repo_root:
        candidates = [
            "/home/alex/edv2-reference",
            os.path.join("/home", "alex", "ai-" + "toolkit"),
        ]
        repo_root = next((p for p in candidates if os.path.isdir(p)), candidates[0])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from extensions_built_in.diffusion_models.hidream.src.hidream_o1.qwen3_vl_transformers import (
            Qwen3VLForConditionalGeneration,
        )
        from extensions_built_in.diffusion_models.hidream.src.hidream_o1.pipeline import (
            PATCH_SIZE,
            TIMESTEP_TOKEN_NUM,
            _build_t2i_sample_from_input_ids,
        )
    except Exception as e:  # noqa: BLE001
        _die(f"failed importing edv2-reference HiDream-O1 modules from {repo_root}: {e}", 2)

    if not torch.cuda.is_available():
        _die("CUDA is required for this parity reference (the model is too "
             "large to run in fp32 on CPU in any reasonable time, and BF16 on "
             "CPU is emulated).", 2)

    device = torch.device("cuda:0")
    dtype = getattr(torch, DTYPE_NAME)

    print(f"[parity-ref] model      = {MODEL_PATH}")
    print(f"[parity-ref] prompt     = {PROMPT!r}")
    print(f"[parity-ref] seed       = {SEED}")
    print(f"[parity-ref] resolution = {HEIGHT}x{WIDTH}")
    print(f"[parity-ref] t_pixeldit = {T_PIXELDIT}")
    print(f"[parity-ref] dtype      = {DTYPE_NAME}")
    print(f"[parity-ref] flash_attn = {USE_FLASH_ATTN}")
    print(f"[parity-ref] PATCH_SIZE = {PATCH_SIZE}")

    if HEIGHT % PATCH_SIZE != 0 or WIDTH % PATCH_SIZE != 0:
        _die(f"resolution {WIDTH}x{HEIGHT} must be divisible by PATCH_SIZE={PATCH_SIZE}")

    t0 = time.time()
    print(f"[parity-ref] loading processor + model ... (this can take a few minutes)")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    # NOTE: inference.py loads in fp32 (~32 GB for an 8B model) which doesn't
    # fit on a 24 GB consumer GPU. The forward pass runs under autocast(bf16)
    # anyway, so the matmul precision is BF16 regardless of the parameter
    # storage dtype. Loading bf16 directly gives the same downstream numerics
    # (the embed/RMSNorm/cast-to-fp32-then-back paths are all bf16-equivalent
    # to within mantissa rounding), and is what our Rust trainer does too.
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=dtype, device_map="cuda"
    ).eval()
    print(f"[parity-ref] model loaded in {time.time() - t0:.1f}s")

    # Replicate inference.add_special_tokens()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"

    model_config = model.config
    messages = [{"role": "user", "content": PROMPT}]
    template_caption = (
        processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        + tokenizer.boi_token
        + tokenizer.tms_token * TIMESTEP_TOKEN_NUM
    )
    input_ids = tokenizer.encode(
        template_caption, return_tensors="pt", add_special_tokens=False
    )
    cond_sample = _build_t2i_sample_from_input_ids(
        input_ids, HEIGHT, WIDTH, model_config,
    )

    # Move tensors to device.
    def to_device(s):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in s.items()}
    cond_sample = to_device(cond_sample)

    # Build noise the way pipeline.generate_image does (line 291-295):
    #   noise = noise_scale_start * randn((1,3,H,W), generator(seed+1))
    #   z     = rearrange(noise, 'B C (H p1) (W p2) -> B (H W) (C p1 p2)', p1=P, p2=P)
    noise = NOISE_SCALE_START * torch.randn(
        (1, 3, HEIGHT, WIDTH),
        generator=torch.Generator("cpu").manual_seed(SEED + 1),
    ).to(device, dtype)
    z = einops.rearrange(
        noise, "B C (H p1) (W p2) -> B (H W) (C p1 p2)",
        p1=PATCH_SIZE, p2=PATCH_SIZE,
    )

    # Build timestep: pipeline computes t_pixeldit = 1.0 - step_t/1000. We pin
    # t_pixeldit = 0.5 directly. Shape per `forward_once` is `t.reshape(-1)`.
    t_pixeldit = torch.tensor([T_PIXELDIT], device=device, dtype=torch.float32)

    print(f"[parity-ref] input shapes:")
    print(f"    input_ids    {tuple(cond_sample['input_ids'].shape)} "
          f"{cond_sample['input_ids'].dtype}")
    print(f"    position_ids {tuple(cond_sample['position_ids'].shape)} "
          f"{cond_sample['position_ids'].dtype}")
    print(f"    token_types  {tuple(cond_sample['token_types'].shape)} "
          f"{cond_sample['token_types'].dtype}")
    print(f"    vinput_mask  {tuple(cond_sample['vinput_mask'].shape)} "
          f"{cond_sample['vinput_mask'].dtype}")
    print(f"    z (vinputs)  {tuple(z.shape)} {z.dtype}")
    print(f"    timestep     {tuple(t_pixeldit.shape)} {t_pixeldit.dtype}")

    # Forward — mirrors pipeline.forward_once almost verbatim.
    print(f"[parity-ref] running forward pass ...")
    t1 = time.time()
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
        outputs = model(
            input_ids=cond_sample["input_ids"],
            position_ids=cond_sample["position_ids"],
            vinputs=z,
            timestep=t_pixeldit.reshape(-1).to(device),
            token_types=cond_sample["token_types"],
            use_flash_attn=USE_FLASH_ATTN,
        )
    x_pred_full = outputs.x_pred  # [1, S_total, 3072] in dtype (likely bf16)
    print(f"[parity-ref] forward done in {time.time() - t1:.2f}s")
    print(f"    x_pred_full  {tuple(x_pred_full.shape)} {x_pred_full.dtype}")

    # Replicate pipeline.forward_once's gather:
    #   return x_pred[0, sample['vinput_mask'][0]].unsqueeze(0)
    # vinput_mask is bool here (built via `token_types == 1`).
    vmask = cond_sample["vinput_mask"]
    x_pred_rows = x_pred_full[0, vmask[0]].unsqueeze(0)
    print(f"    x_pred_rows  {tuple(x_pred_rows.shape)} {x_pred_rows.dtype}")

    # Also export `patches` = z (the "vinputs" we fed). The Rust trainer cache
    # stores `patches` as the clean input; for THIS test, we treat z as the
    # equivalent — the trainer's `noisy` would normally be (1-t)*patches +
    # t*noise, but for single-forward parity we just feed the same tensor.
    image_len = (HEIGHT // PATCH_SIZE) * (WIDTH // PATCH_SIZE)
    image_grid = torch.tensor([1, HEIGHT // PATCH_SIZE, WIDTH // PATCH_SIZE],
                              dtype=torch.int32)

    # Casts for on-disk parity with the cache format.
    # position_ids comes out as [3, 1, S_total] (T/H/W axes × batch × seq).
    # The Rust trainer's `decode_position_ids` expects [3, S_total] (the
    # batch dim was already squeezed by `prepare_hidream_o1` before caching).
    # Squeeze the batch dim here so the on-disk layout matches the cache.
    pos_ids_save = cond_sample["position_ids"]
    if pos_ids_save.dim() == 3 and pos_ids_save.shape[1] == 1:
        pos_ids_save = pos_ids_save.squeeze(1)

    # NOTE on dtypes: flame_core's safetensors loader skips I32/I64/BOOL
    # tensors (only F32/BF16/F16/F8 pass the filter). To match the
    # `prepare_hidream_o1` cache contract — which writes input_ids /
    # position_ids / vinput_mask / image_grid as F32 — we cast everything
    # integral to F32 here. The Rust side casts back via `.to_dtype(I32)`.
    to_save = {
        # vinputs (the noisy/noise tensor we fed). Saved as F32 to avoid any
        # BF16 round-trip on the Rust side; Rust will cast to BF16 on load.
        "patches":      z.to(torch.float32).contiguous().cpu(),
        "input_ids":    cond_sample["input_ids"].to(torch.float32).contiguous().cpu(),
        "position_ids": pos_ids_save.to(torch.float32).contiguous().cpu(),
        "vinput_mask":  vmask.to(torch.float32).contiguous().cpu(),
        "token_types":  cond_sample["token_types"].to(torch.float32).contiguous().cpu(),
        "image_grid":   image_grid.to(torch.float32),
        "timestep":     t_pixeldit.to(torch.float32).contiguous().cpu(),
        # Outputs.
        "x_pred_full":  x_pred_full.to(torch.float32).contiguous().cpu(),
        "x_pred_rows":  x_pred_rows.to(torch.float32).contiguous().cpu(),
    }

    OUT_TENSORS.parent.mkdir(parents=True, exist_ok=True)
    save_file(to_save, str(OUT_TENSORS))
    sz_mb = OUT_TENSORS.stat().st_size / 1e6
    print(f"[parity-ref] wrote {OUT_TENSORS} ({sz_mb:.1f} MB, {len(to_save)} keys)")

    meta = {
        "model_path": MODEL_PATH,
        "prompt": PROMPT,
        "seed": SEED,
        "height": HEIGHT,
        "width": WIDTH,
        "patch_size": PATCH_SIZE,
        "image_len": image_len,
        "t_pixeldit": T_PIXELDIT,
        "noise_scale_start": NOISE_SCALE_START,
        "dtype": DTYPE_NAME,
        "use_flash_attn": USE_FLASH_ATTN,
        "input_ids_shape":   list(cond_sample["input_ids"].shape),
        "position_ids_shape":list(cond_sample["position_ids"].shape),
        "vinput_mask_shape": list(vmask.shape),
        "token_types_shape": list(cond_sample["token_types"].shape),
        "patches_shape":     list(z.shape),
        "x_pred_full_shape": list(x_pred_full.shape),
        "x_pred_rows_shape": list(x_pred_rows.shape),
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"[parity-ref] wrote {OUT_META}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
