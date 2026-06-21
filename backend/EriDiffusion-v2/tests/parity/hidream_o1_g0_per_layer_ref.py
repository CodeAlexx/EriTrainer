#!/usr/bin/env python3
"""HiDream-O1 G0 — DEEP-INVESTIGATION per-layer reference dump.

Companion to `hidream_o1_g0_python_ref.py`. Reuses the same pinning
(prompt/seed/resolution/timestep/dtype) so the dumped inputs match the
existing ref dump byte-for-byte; ADDITIONALLY dumps every decoder layer's
output via `return_mid_results_layers=list(range(36))`.

Output: /tmp/hidream_o1_g0_per_layer_ref.safetensors
  - All inputs (same keys as the original ref)
  - `hidden_layer_00` .. `hidden_layer_35` — F32 [1, S_total, 4096] per-layer
    OUTPUT (= post-residual+MLP, the captured `hidden_states` at the end of
    `Qwen3VLTextDecoderLayer.forward`)
  - `hidden_final_norm`  — F32 [1, S_total, 4096] AFTER the final RMSNorm
  - `x_pred_full`        — F32 [1, S_total, 3072]  (after FinalLayer)
  - `x_pred_rows`        — F32 [1, L, 3072]        (gathered image rows)

Also dumps a special "layer-0 in fp64" tensor for the isolated-ground-truth
test (Step 2): we re-run JUST layer 0 in F64 on CPU starting from the same
`hidden_input_layer_00` we captured, and save the result as
`hidden_layer_00_fp64`. This gives us a per-element ground-truth reference.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Same pinning constants as the original ref.
MODEL_PATH = "/home/alex/HiDream-O1-Image-Dev-weights"
PROMPT = (
    "a photograph of an astronaut riding a horse on mars, cinematic lighting"
)
SEED = 42
HEIGHT = 512
WIDTH = 512
T_PIXELDIT = 0.5
NOISE_SCALE_START = 8.0
DTYPE_NAME = "bfloat16"
USE_FLASH_ATTN = False

OUT_TENSORS = Path("/tmp/hidream_o1_g0_per_layer_ref.safetensors")
OUT_META = Path("/tmp/hidream_o1_g0_per_layer_ref_meta.json")


def _die(msg: str, code: int = 2) -> None:
    print(f"[per-layer-ref] FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def main() -> int:
    import torch
    import numpy as np
    import einops
    from safetensors.torch import save_file
    from transformers import AutoProcessor

    repo_root = "/home/alex/HiDream-O1-Image"
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
        from models.pipeline import build_t2i_text_sample, PATCH_SIZE
    except Exception as e:  # noqa: BLE001
        _die(f"import HiDream-O1 modules from {repo_root}: {e}", 2)

    if not torch.cuda.is_available():
        _die("CUDA required", 2)

    device = torch.device("cuda:0")
    dtype = getattr(torch, DTYPE_NAME)

    print(f"[per-layer-ref] loading processor + model ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=dtype, device_map="cuda"
    ).eval()
    print(f"[per-layer-ref] model loaded in {time.time() - t0:.1f}s")

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"

    cond_sample = build_t2i_text_sample(
        PROMPT, HEIGHT, WIDTH, tokenizer, processor, model.config,
    )
    cond_sample = {k: (v.to(device) if torch.is_tensor(v) else v)
                   for k, v in cond_sample.items()}

    noise = NOISE_SCALE_START * torch.randn(
        (1, 3, HEIGHT, WIDTH),
        generator=torch.Generator("cpu").manual_seed(SEED + 1),
    ).to(device, dtype)
    z = einops.rearrange(
        noise, "B C (H p1) (W p2) -> B (H W) (C p1 p2)",
        p1=PATCH_SIZE, p2=PATCH_SIZE,
    )
    t_pixeldit = torch.tensor([T_PIXELDIT], device=device, dtype=torch.float32)

    num_layers = model.config.text_config.num_hidden_layers
    print(f"[per-layer-ref] num_layers = {num_layers}; dumping all of them")

    # Forward with per-layer capture.
    print(f"[per-layer-ref] running forward (capture all {num_layers} layers) ...")
    t1 = time.time()
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
        outputs = model(
            input_ids=cond_sample["input_ids"],
            position_ids=cond_sample["position_ids"],
            vinputs=z,
            timestep=t_pixeldit.reshape(-1).to(device),
            token_types=cond_sample["token_types"],
            use_flash_attn=USE_FLASH_ATTN,
            return_mid_results_layers=list(range(num_layers)),
        )
    x_pred_full = outputs.x_pred
    mid_results = outputs.mid_results
    assert mid_results is not None, "mid_results missing — return_mid_results_layers not honored?"
    assert len(mid_results) == num_layers, f"expected {num_layers} mid_results got {len(mid_results)}"
    print(f"[per-layer-ref] forward done in {time.time() - t1:.2f}s")

    # Gather image-row tail.
    vmask = cond_sample["vinput_mask"]
    x_pred_rows = x_pred_full[0, vmask[0]].unsqueeze(0)

    image_len = (HEIGHT // PATCH_SIZE) * (WIDTH // PATCH_SIZE)
    image_grid = torch.tensor([1, HEIGHT // PATCH_SIZE, WIDTH // PATCH_SIZE],
                              dtype=torch.int32)
    pos_ids_save = cond_sample["position_ids"]
    if pos_ids_save.dim() == 3 and pos_ids_save.shape[1] == 1:
        pos_ids_save = pos_ids_save.squeeze(1)

    # ── Step 2: Layer-0 in fp64 on CPU as ground-truth reference. ────────
    # We need the INPUT to layer 0. That input is `inputs_embeds`, which is
    # the concat of (text_emb + scattered timestep) and (patch_embed). We
    # can capture it via a forward hook on the first layer.
    print(f"[per-layer-ref] running layer-0 fp64 ground-truth pass ...")
    captured = {}

    def cap_input_hook(module, args, kwargs):
        # args[0] is hidden_states
        captured["layer0_input_bf16"] = args[0].detach().clone()
        # capture position_embeddings too if present in kwargs
        if "position_embeddings" in kwargs:
            pe = kwargs["position_embeddings"]
            captured["pe_cos"] = pe[0].detach().clone()
            captured["pe_sin"] = pe[1].detach().clone()
        if "attention_mask" in kwargs:
            captured["attn_mask"] = kwargs["attention_mask"]
        if "position_ids" in kwargs:
            captured["text_position_ids"] = kwargs["position_ids"].detach().clone()
        return None  # don't modify

    # Hook the first text-decoder layer.
    text_layers = model.model.language_model.layers
    h = text_layers[0].register_forward_pre_hook(cap_input_hook, with_kwargs=True)
    try:
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
            _ = model(
                input_ids=cond_sample["input_ids"],
                position_ids=cond_sample["position_ids"],
                vinputs=z,
                timestep=t_pixeldit.reshape(-1).to(device),
                token_types=cond_sample["token_types"],
                use_flash_attn=USE_FLASH_ATTN,
            )
    finally:
        h.remove()
    print(f"[per-layer-ref] captured layer0 input shape={tuple(captured['layer0_input_bf16'].shape)}")

    # Re-run layer 0 in fp64 (on GPU is fine — 4096-hidden × ~800 tokens × fp64 ≈ tiny).
    print(f"[per-layer-ref] re-running layer 0 in fp64 ...")
    layer0 = text_layers[0]
    # Materialize a fp64 copy of just layer 0.
    layer0_fp64 = layer0.to(torch.float64) if False else layer0  # AVOID destroying main model
    # Instead: rebuild layer 0 in fp64 using its state_dict.
    import copy
    layer0_fp64 = copy.deepcopy(layer0).to(torch.float64)
    inp_fp64 = captured["layer0_input_bf16"].to(torch.float64)
    cos_fp64 = captured["pe_cos"].to(torch.float64)
    sin_fp64 = captured["pe_sin"].to(torch.float64)
    am = captured.get("attn_mask")
    am_fp64 = am.to(torch.float64) if am is not None else None
    pos_ids = captured.get("text_position_ids")
    with torch.no_grad():
        # autocast OFF for true fp64.
        out_fp64 = layer0_fp64(
            hidden_states=inp_fp64,
            position_embeddings=(cos_fp64, sin_fp64),
            attention_mask=am_fp64,
            position_ids=pos_ids,
        )
    print(f"[per-layer-ref] layer-0 fp64 output shape={tuple(out_fp64.shape)} dtype={out_fp64.dtype}")
    del layer0_fp64
    torch.cuda.empty_cache()

    # Final norm output (the input to FinalLayer) for completeness.
    # The Python pipeline runs `hidden = norm(hidden); x_pred = final(hidden)`.
    # We don't get hidden_final_norm via hooks here; the final layer's input
    # IS the norm-output. To match, capture the input to model.final_layer.
    final_norm_captured = {}
    def fl_pre_hook(module, args):
        final_norm_captured["x"] = args[0].detach().clone()
        return None
    fl = model.model.final_layer2  # Python attribute is `final_layer2`
    h2 = fl.register_forward_pre_hook(fl_pre_hook)
    try:
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
            _ = model(
                input_ids=cond_sample["input_ids"],
                position_ids=cond_sample["position_ids"],
                vinputs=z,
                timestep=t_pixeldit.reshape(-1).to(device),
                token_types=cond_sample["token_types"],
                use_flash_attn=USE_FLASH_ATTN,
            )
    finally:
        h2.remove()
    final_norm_in = final_norm_captured["x"]

    # ── Build the save dict. ─────────────────────────────────────────────
    to_save = {
        "patches":      z.to(torch.float32).contiguous().cpu(),
        "input_ids":    cond_sample["input_ids"].to(torch.float32).contiguous().cpu(),
        "position_ids": pos_ids_save.to(torch.float32).contiguous().cpu(),
        "vinput_mask":  vmask.to(torch.float32).contiguous().cpu(),
        "token_types":  cond_sample["token_types"].to(torch.float32).contiguous().cpu(),
        "image_grid":   image_grid.to(torch.float32),
        "timestep":     t_pixeldit.to(torch.float32).contiguous().cpu(),
        "x_pred_full":  x_pred_full.to(torch.float32).contiguous().cpu(),
        "x_pred_rows":  x_pred_rows.to(torch.float32).contiguous().cpu(),
        "hidden_final_norm": final_norm_in.to(torch.float32).contiguous().cpu(),
        # Layer-0 input + fp64 output for Step 2.
        "hidden_input_layer_00": captured["layer0_input_bf16"].to(torch.float32).contiguous().cpu(),
        "hidden_layer_00_fp64": out_fp64.to(torch.float32).contiguous().cpu(),  # save as f32 (safetensors fp64 ok? safer to keep f32 for harness compat)
        "pe_cos": captured["pe_cos"].to(torch.float32).contiguous().cpu(),
        "pe_sin": captured["pe_sin"].to(torch.float32).contiguous().cpu(),
    }
    if am is not None and torch.is_tensor(am):
        to_save["attn_mask"] = am.to(torch.float32).contiguous().cpu()

    for i, h_state in enumerate(mid_results):
        to_save[f"hidden_layer_{i:02d}"] = (
            h_state.to(torch.float32).contiguous().cpu()
        )

    OUT_TENSORS.parent.mkdir(parents=True, exist_ok=True)
    save_file(to_save, str(OUT_TENSORS))
    sz_mb = OUT_TENSORS.stat().st_size / 1e6
    print(f"[per-layer-ref] wrote {OUT_TENSORS} ({sz_mb:.1f} MB, {len(to_save)} keys)")

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
        "num_layers": num_layers,
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"[per-layer-ref] wrote {OUT_META}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
