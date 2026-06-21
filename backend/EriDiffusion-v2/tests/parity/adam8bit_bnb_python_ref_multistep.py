#!/usr/bin/env python3
"""Adam-8bit blockwise — bnb 0.49.2 reference dump, MULTI-STEP (10 steps).

Same setup as `adam8bit_bnb_python_ref.py` (NUMEL=2048, wd=0, zero initial
state) but runs the bnb kernel 10 times in a row, dumping a BEFORE/AFTER
snapshot pair for each step. The Rust harness loads step-i's BEFORE (which
is step-(i-1)'s AFTER), runs ONE step through our kernel, and diffs against
step-i's AFTER.

Dumps under tests/parity/adam8bit_data/multistep/:
  - hyperparams.json                  — lr, betas, eps, wd, blocksize, numel, n_steps
  - qmaps.safetensors                 — qmap1, qmap2 (constants, one copy)
  - step_{1..10}_before.safetensors   — param, grad, state1, state2, absmax1, absmax2
  - step_{1..10}_after.safetensors    — param, state1, state2, absmax1, absmax2

Each step uses a fresh grad re-sampled from the same generator
(torch.manual_seed(42) at the top; the generator advances naturally so
each step gets a different but reproducible grad).
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
BLOCKSIZE = 256
N_STEPS = 10

print(f"bitsandbytes version: {bnb.__version__}")
print(f"torch:                {torch.__version__}")
print(f"device:               {torch.cuda.get_device_name(0)}")

torch.manual_seed(42)
device = "cuda:0"

param = torch.randn(NUMEL, dtype=torch.float32, device=device)

n_blocks = (NUMEL + BLOCKSIZE - 1) // BLOCKSIZE
state1 = torch.zeros(NUMEL, dtype=torch.uint8, device=device)
state2 = torch.zeros(NUMEL, dtype=torch.uint8, device=device)
absmax1 = torch.zeros(n_blocks, dtype=torch.float32, device=device)
absmax2 = torch.zeros(n_blocks, dtype=torch.float32, device=device)

qmap1 = F.create_dynamic_map(signed=True).to(device)
qmap2 = F.create_dynamic_map(signed=False).to(device)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "adam8bit_data", "multistep")
os.makedirs(out_dir, exist_ok=True)

save_file(
    {"qmap1": qmap1.cpu(), "qmap2": qmap2.cpu()},
    os.path.join(out_dir, "qmaps.safetensors"),
)

for step in range(1, N_STEPS + 1):
    # Fresh grad per step (generator advances).
    grad = torch.randn(NUMEL, dtype=torch.float32, device=device) * 0.1

    before = {
        "param":   param.clone(),
        "grad":    grad.clone(),
        "state1":  state1.clone(),
        "state2":  state2.clone(),
        "absmax1": absmax1.clone(),
        "absmax2": absmax2.clone(),
    }

    F.optimizer_update_8bit_blockwise(
        "adam",
        grad, param, state1, state2,
        BETA1, BETA2, 0.0, 0.0,
        EPS, step, LR,
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

    save_file(
        {f"before.{k}": v.cpu().contiguous() for k, v in before.items()},
        os.path.join(out_dir, f"step_{step}_before.safetensors"),
    )
    save_file(
        {f"after.{k}": v.cpu().contiguous() for k, v in after.items()},
        os.path.join(out_dir, f"step_{step}_after.safetensors"),
    )

    print(
        f"step {step:2d}: param mean={param.mean().item():+.6e} "
        f"absmax1[0]={absmax1[0].item():.6e} absmax2[0]={absmax2[0].item():.6e}"
    )

with open(os.path.join(out_dir, "hyperparams.json"), "w") as f:
    json.dump(
        {
            "lr": LR, "beta1": BETA1, "beta2": BETA2, "eps": EPS,
            "wd": WD, "blocksize": BLOCKSIZE, "numel": NUMEL,
            "n_steps": N_STEPS,
            "bnb_version": bnb.__version__,
            "torch_version": torch.__version__,
        },
        f,
        indent=2,
    )

print(f"\ndumped {N_STEPS} steps to {out_dir}")
