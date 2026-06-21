#!/usr/bin/env python3
"""HiDream M1.5 — generate Llama-3.1-8B-Instruct PyTorch reference dump.

Mirrors the encoder use exposed by HiDream-I1's pipeline (see
`pipeline_hidream_image.py::_get_llama3_prompt_embeds`):

    outputs = self.text_encoder_4(
        text_input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    prompt_embeds = outputs.hidden_states[1:]            # drop embed output
    prompt_embeds = torch.stack(prompt_embeds, dim=0)    # [num_layers, B, S, D]

We load `transformers.LlamaModel` directly (NOT `LlamaForCausalLM`), so the
returned `hidden_states` tuple is `[embed, l0, l1, ..., l_{n-2},
norm(l_{n-1})]` — the final entry passes through `model.norm`, matching the
HF behaviour the Rust port mimics in
`encoders/llama3.rs::encode_all_hidden_states`.

Outputs:
  - /tmp/hidream_m1_llama3_ref.safetensors  (input_ids, attn_mask,
    layer_00..layer_31, stacked_hidden)
  - /tmp/hidream_m1_llama3_ref_meta.json    (prompt, tokenizer/model name,
    dtype, max_length, pad_token_id, num_layers, hidden_size)

If `transformers` cannot locate the model on disk OR can't download it,
this script prints a clear error and exits 2. It will NEVER try to
install pip packages.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# --- Hard config (must stay in lockstep with the Rust binary) ------------
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
PROMPT = (
    "The image depicts a woman with long, dark hair, sitting by a window "
    "in soft natural light, contemplative mood."
)
MAX_LENGTH = 128                  # HiDream pipeline default
PAD_TOKEN_ID_OVERRIDE = 128_004   # unsloth <|finetune_right_pad_id|>; HF
                                  # default leaves this unset which would
                                  # break right-pad tokenisation.
DTYPE_NAME = "bfloat16"
OUT_TENSORS = Path("/tmp/hidream_m1_llama3_ref.safetensors")
OUT_META = Path("/tmp/hidream_m1_llama3_ref_meta.json")


def _require(import_str: str):
    try:
        return __import__(import_str)
    except ImportError as e:  # noqa: BLE001
        print(
            f"[parity-ref] FATAL: cannot import {import_str!r}: {e}\n"
            "             This script REQUIRES a Python env with torch + "
            "transformers + safetensors.\n"
            "             We do NOT auto-install. Activate the right venv "
            "and re-run.",
            file=sys.stderr,
        )
        sys.exit(2)


def main() -> int:
    _require("torch")
    _require("transformers")
    _require("safetensors")

    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, LlamaModel

    print(f"[parity-ref] model = {MODEL_NAME}")
    print(f"[parity-ref] prompt = {PROMPT!r}")
    print(f"[parity-ref] max_length = {MAX_LENGTH}, pad_id override = {PAD_TOKEN_ID_OVERRIDE}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:  # noqa: BLE001
        print(
            f"[parity-ref] FATAL: failed to load tokenizer for {MODEL_NAME!r}: {e}\n"
            "             Either the model isn't cached at "
            "$HF_HOME/$XDG_CACHE_HOME/huggingface/hub and the host has no "
            "internet, or the HF id is wrong.",
            file=sys.stderr,
        )
        return 2

    # unsloth's tokenizer already sets pad_token=<|finetune_right_pad_id|>=128004,
    # but force-override to be deterministic regardless of revision.
    tokenizer.pad_token_id = PAD_TOKEN_ID_OVERRIDE
    # `padding_side="right"` to match the Rust `pad_and_mask` helper, which
    # produces right-padded sequences. (HF Llama tokenizer defaults to LEFT.)
    tokenizer.padding_side = "right"

    enc = tokenizer(
        PROMPT,
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]            # [1, 128] int64
    attention_mask = enc["attention_mask"]  # [1, 128] int64 (1=real, 0=pad)
    real_tokens = int(attention_mask.sum().item())
    print(f"[parity-ref] tokenised: {real_tokens} real tokens, "
          f"{MAX_LENGTH - real_tokens} pads")

    try:
        model = LlamaModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=getattr(torch, DTYPE_NAME),
            attn_implementation="eager",  # match the Rust SDPA path semantics
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"[parity-ref] FATAL: failed to load LlamaModel weights for {MODEL_NAME!r}: {e}\n"
            "             Weights must be present at "
            "$HF_HOME/huggingface/hub. Download with:\n"
            f"             huggingface-cli download {MODEL_NAME}\n",
            file=sys.stderr,
        )
        return 2

    if not torch.cuda.is_available():
        print("[parity-ref] WARNING: CUDA unavailable, running on CPU (slow, "
              "and BF16 on CPU is emulated — Rust BF16 may show >0.1 "
              "max_abs vs this reference).", file=sys.stderr)
        device = "cpu"
    else:
        device = "cuda:0"
    model = model.to(device).eval()
    input_ids_dev = input_ids.to(device)
    attention_mask_dev = attention_mask.to(device)

    print(f"[parity-ref] running forward on {device} ...")
    with torch.no_grad():
        out = model(
            input_ids_dev,
            attention_mask=attention_mask_dev,
            output_hidden_states=True,
            return_dict=True,
        )

    # hidden_states: tuple of (num_layers + 1) tensors.
    # [0]   = embedding output
    # [1:]  = per-layer outputs, with model.norm applied to the LAST entry.
    hidden_states = out.hidden_states
    num_layers_plus_one = len(hidden_states)
    num_layers = num_layers_plus_one - 1
    assert num_layers == 32, (
        f"expected 32 Llama-3.1-8B layers, got {num_layers}"
    )

    per_layer = hidden_states[1:]  # tuple of 32 tensors, each [1, 128, 4096]
    # stack to [num_layers, 1, S, hidden]
    stacked = torch.stack(per_layer, dim=0).contiguous().cpu()
    print(f"[parity-ref] stacked hidden states: {tuple(stacked.shape)} "
          f"dtype={stacked.dtype}")

    # Save flat: layer_NN keys (Rust comparison side) PLUS the stacked tensor
    # PLUS the inputs (so the Rust side can sanity-check tokenisation).
    to_save = {
        "input_ids": input_ids.to(torch.int32).contiguous().cpu(),
        "attention_mask": attention_mask.to(torch.int32).contiguous().cpu(),
        "stacked_hidden": stacked,
    }
    for i, t in enumerate(per_layer):
        to_save[f"layer_{i:02d}"] = t.detach().contiguous().cpu()

    OUT_TENSORS.parent.mkdir(parents=True, exist_ok=True)
    save_file(to_save, str(OUT_TENSORS))
    print(f"[parity-ref] wrote {OUT_TENSORS} "
          f"({OUT_TENSORS.stat().st_size / 1e6:.1f} MB, {len(to_save)} keys)")

    meta = {
        "prompt": PROMPT,
        "model_name": MODEL_NAME,
        "tokenizer_name": MODEL_NAME,
        "dtype": DTYPE_NAME,
        "max_length": MAX_LENGTH,
        "pad_token_id": PAD_TOKEN_ID_OVERRIDE,
        "num_layers": num_layers,
        "hidden_size": int(stacked.shape[-1]),
        "seq_len": int(stacked.shape[-2]),
        "real_tokens": real_tokens,
        "input_ids": input_ids[0].tolist(),
        "attention_mask": attention_mask[0].tolist(),
        "device_used": device,
        "padding_side": "right",
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"[parity-ref] wrote {OUT_META}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
