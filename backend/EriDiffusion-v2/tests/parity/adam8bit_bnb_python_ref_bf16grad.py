#!/usr/bin/env python3
"""Adam-8bit blockwise — bnb 0.49.2 reference dump, BF16-grad self-consistency.

bnb's host dispatch for BF16 grads upcasts via `g.float()` BEFORE calling
the kernel — so there's no native BF16 bnb kernel to test against. This
script approximates that by generating a BF16 grad, round-tripping it back
to F32 (a no-op precision-wise: F32 -> BF16 -> F32 == BF16-precision F32),
then running the bnb F32 kernel.

The Rust side runs our *native* BF16-grad kernel on the BF16 tensor (no
host-side upcast). The two paths should match to within BF16-precision
noise. Code mismatches up to ~1-2 elements are legitimate (BF16 rounding
of the grad shifts a normalized moment across a LUT-tiebreak boundary).
Wider divergence (>5 elements or |param Δ| > 1e-3) is a real kernel bug.

Output: tests/parity/adam8bit_data/bf16grad/{before,after}.safetensors + hp
"""
from __future__ import annotations

import json
import os
import sys

import torch

try:
    import bitsandbytes as bnb
    from bitsandbytes import functional as F
except Exception as e:  # pragma: no cover
    print(f"BLOCKED: bitsandbytes import failed: {e}", file=sys.stderr)
    sys.exit(2)

try:
    from safetensors.torch import save_file
except Exception as e:  # pragma: no cover
    print(f"BLOCKED: safetensors import failed: {e}", file=sys.stderr)
    sys.exit(2)

if not torch.cuda.is_available():
    print("BLOCKED: CUDA not available", file=sys.stderr)
    sys.exit(2)

NUMEL = 2048
LR, BETA1, BETA2 = 1e-3, 0.9, 0.999
EPS = 1e-8
WD = 0.0
STEP = 1
BLOCKSIZE = 256

print(f"bitsandbytes version: {bnb.__version__}")

torch.manual_seed(42)
device = "cuda:0"

param = torch.randn(NUMEL, dtype=torch.float32, device=device)
grad_f32_raw = torch.randn(NUMEL, dtype=torch.float32, device=device) * 0.1
# Cast to BF16 then back to F32 — this is what the bnb host dispatch does
# (`g.float()` on a BF16 tensor) AND it's the same precision the Rust
# native BF16-grad kernel sees inside __bfloat162float(grad[i]).
grad_bf16 = grad_f32_raw.to(torch.bfloat16)
grad_for_bnb = grad_bf16.float()

# Save the BF16 tensor so the Rust harness reads the SAME bytes.
n_blocks = (NUMEL + BLOCKSIZE - 1) // BLOCKSIZE
state1 = torch.zeros(NUMEL, dtype=torch.uint8, device=device)
state2 = torch.zeros(NUMEL, dtype=torch.uint8, device=device)
absmax1 = torch.zeros(n_blocks, dtype=torch.float32, device=device)
absmax2 = torch.zeros(n_blocks, dtype=torch.float32, device=device)

qmap1 = F.create_dynamic_map(signed=True).to(device)
qmap2 = F.create_dynamic_map(signed=False).to(device)

before = {
    "param":     param.clone(),
    "grad_bf16": grad_bf16.clone(),       # BF16 raw bytes for Rust kernel
    "grad_f32":  grad_for_bnb.clone(),    # what bnb actually consumed
    "state1":    state1.clone(),
    "state2":    state2.clone(),
    "absmax1":   absmax1.clone(),
    "absmax2":   absmax2.clone(),
    "qmap1":     qmap1.clone(),
    "qmap2":     qmap2.clone(),
}

F.optimizer_update_8bit_blockwise(
    "adam",
    grad_for_bnb, param, state1, state2,
    BETA1, BETA2, 0.0, 0.0,
    EPS, STEP, LR,
    qmap1, qmap2, absmax1, absmax2,
    WD, 1.0, False,
)
torch.cuda.synchronize()

after = {
    "param":   param.clone(),
    "state1":  state1.clone(),
    "state2":  state2.clone(),
    "absmax1": absmax1.clone(),
    "absmax2": absmax2.clone(),
}

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "adam8bit_data", "bf16grad")
os.makedirs(out_dir, exist_ok=True)

save_file(
    {f"before.{k}": v.cpu().contiguous() for k, v in before.items()},
    os.path.join(out_dir, "before.safetensors"),
)
save_file(
    {f"after.{k}": v.cpu().contiguous() for k, v in after.items()},
    os.path.join(out_dir, "after.safetensors"),
)
with open(os.path.join(out_dir, "hyperparams.json"), "w") as f:
    json.dump(
        {
            "lr": LR, "beta1": BETA1, "beta2": BETA2, "eps": EPS,
            "wd": WD, "step": STEP, "blocksize": BLOCKSIZE, "numel": NUMEL,
            "grad_dtype": "bf16",
            "bnb_version": bnb.__version__,
            "torch_version": torch.__version__,
        },
        f,
        indent=2,
    )

print(f"dumped to {out_dir}")
print(f"  param after mean = {after['param'].mean().item():+.6e}")
