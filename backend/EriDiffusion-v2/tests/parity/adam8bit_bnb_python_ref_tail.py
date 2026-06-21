#!/usr/bin/env python3
"""Adam-8bit blockwise — bnb 0.49.2 reference dump, TAIL-BLOCK (numel=2050).

NUMEL=2050 -> 9 blocks of 256, last block has 2 active elements (idx
2048,2049) + 254 inactive lanes. This exercises the kernel's guard against
inactive lanes polluting the block-wide absmax reduction.

Output: tests/parity/adam8bit_data/tail/{before,after}.safetensors + hyperparams.json
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

NUMEL = 2050           # <-- 8 full blocks + 1 tail block with 2 active lanes
LR, BETA1, BETA2 = 1e-3, 0.9, 0.999
EPS = 1e-8
WD = 0.0
STEP = 1
BLOCKSIZE = 256

print(f"bitsandbytes version: {bnb.__version__}")

torch.manual_seed(42)
device = "cuda:0"

param = torch.randn(NUMEL, dtype=torch.float32, device=device)
grad = torch.randn(NUMEL, dtype=torch.float32, device=device) * 0.1

n_blocks = (NUMEL + BLOCKSIZE - 1) // BLOCKSIZE
assert n_blocks == 9, f"expected 9 blocks for numel={NUMEL}, got {n_blocks}"
state1 = torch.zeros(NUMEL, dtype=torch.uint8, device=device)
state2 = torch.zeros(NUMEL, dtype=torch.uint8, device=device)
absmax1 = torch.zeros(n_blocks, dtype=torch.float32, device=device)
absmax2 = torch.zeros(n_blocks, dtype=torch.float32, device=device)

qmap1 = F.create_dynamic_map(signed=True).to(device)
qmap2 = F.create_dynamic_map(signed=False).to(device)

before = {
    "param":   param.clone(),
    "grad":    grad.clone(),
    "state1":  state1.clone(),
    "state2":  state2.clone(),
    "absmax1": absmax1.clone(),
    "absmax2": absmax2.clone(),
    "qmap1":   qmap1.clone(),
    "qmap2":   qmap2.clone(),
}

F.optimizer_update_8bit_blockwise(
    "adam",
    grad, param, state1, state2,
    BETA1, BETA2, 0.0, 0.0,
    EPS, STEP, LR,
    qmap1, qmap2, absmax1, absmax2,
    WD, 1.0, False,
)
torch.cuda.synchronize()

after = {
    "param":   param.clone(),
    "grad":    grad.clone(),
    "state1":  state1.clone(),
    "state2":  state2.clone(),
    "absmax1": absmax1.clone(),
    "absmax2": absmax2.clone(),
}

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "adam8bit_data", "tail")
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
            "n_blocks": n_blocks,
            "bnb_version": bnb.__version__,
            "torch_version": torch.__version__,
        },
        f,
        indent=2,
    )

print(f"dumped to {out_dir}")
print(f"  tail block absmax1={absmax1[-1].item():.6e}  absmax2={absmax2[-1].item():.6e}")
print(f"  param[2048..]={after['param'][2048:].tolist()}")
