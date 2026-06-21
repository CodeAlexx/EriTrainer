#!/usr/bin/env python3
"""Adam-8bit blockwise — bnb 0.49.2 reference dump.

Synthesizes a small CUDA tensor and runs ONE step of
`F.optimizer_update_8bit_blockwise(optimizer_name="adam", ...)` to produce
byte-for-byte input/output snapshots the Rust parity binary
`parity_adam8bit_bnb` consumes.

Block size is hardcoded to 256 in bnb. NUMEL=2048 gives 8 full blocks (no
tail) so we exercise the steady-state path; a follow-up can add a tail-block
case.

Pinning rules (must stay byte-identical to the Rust harness):
  - torch.manual_seed(42)  — both `param` and `grad` use the SAME generator
    in the order [param, grad], matching the Rust loader's read order from
    the safetensors dump (not a sequence-position issue but worth recording).
  - betas = (0.9, 0.999), eps = 1e-8, lr = 1e-3, wd = 0.0, step = 1
  - state1/state2/absmax1/absmax2 zero-initialized exactly as bnb's
    Optimizer8bit.init_state does (`optim/optimizer.py:497-519`).
  - qmap1 = F.create_dynamic_map(signed=True ).to(device)  — m's LUT
  - qmap2 = F.create_dynamic_map(signed=False).to(device)  — v's LUT

bnb 0.49.2 has additional `beta3` and `alpha` positional args in
`F.optimizer_update_8bit_blockwise` (added for AdEMAMix/Lion); the
"adam" optimizer_name ignores them but they MUST be passed (no
defaults). We pass beta3=0.0, alpha=0.0.

Outputs (under tests/parity/adam8bit_data/):
  - before.safetensors  — keys: before/{param,grad,state1,state2,
                                          absmax1,absmax2,qmap1,qmap2}
  - after.safetensors   — keys: after/{param,grad,state1,state2,
                                          absmax1,absmax2}
  - hyperparams.json    — lr, beta1, beta2, eps, wd, step, blocksize, numel
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

# -------- Pinned hyperparameters --------
NUMEL = 2048                              # 8 full blocks of 256
LR, BETA1, BETA2 = 1e-3, 0.9, 0.999
EPS = 1e-8
WD = 0.0
STEP = 1
BLOCKSIZE = 256

print(f"bitsandbytes version: {bnb.__version__}")
print(f"torch:                {torch.__version__}")
print(f"device:               {torch.cuda.get_device_name(0)}")

torch.manual_seed(42)
device = "cuda:0"

param = torch.randn(NUMEL, dtype=torch.float32, device=device)
grad = torch.randn(NUMEL, dtype=torch.float32, device=device) * 0.1

# -------- Allocate state EXACTLY as bnb Optimizer8bit.init_state does --------
n_blocks = (NUMEL + BLOCKSIZE - 1) // BLOCKSIZE
state1 = torch.zeros(NUMEL, dtype=torch.uint8, device=device)
state2 = torch.zeros(NUMEL, dtype=torch.uint8, device=device)
absmax1 = torch.zeros(n_blocks, dtype=torch.float32, device=device)
absmax2 = torch.zeros(n_blocks, dtype=torch.float32, device=device)

# bnb's `fill_qmap` populates name2qmap["dynamic"]   = create_dynamic_map(True)
#                              name2qmap["udynamic"] = create_dynamic_map(False)
# (see optim/optimizer2state.py:fill_qmap and optim/optimizer.py:504-505).
qmap1 = F.create_dynamic_map(signed=True).to(device)
qmap2 = F.create_dynamic_map(signed=False).to(device)
assert qmap1.shape == (256,) and qmap1.dtype == torch.float32
assert qmap2.shape == (256,) and qmap2.dtype == torch.float32

# -------- Snapshot BEFORE --------
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

# -------- Run ONE blockwise 8-bit step --------
# Signature (functional.py:1394-1413, bnb 0.49.2):
#   optimizer_update_8bit_blockwise(
#       optimizer_name, g, p, state1, state2,
#       beta1, beta2, beta3, alpha, eps, step, lr,
#       qmap1, qmap2, absmax1, absmax2,
#       weight_decay=0.0, gnorm_scale=1.0, skip_zeros=False,
#   )
F.optimizer_update_8bit_blockwise(
    "adam",       # optimizer_name (NOT "adamw" — decoupled WD is folded by C++ kernel)
    grad,         # g
    param,        # p
    state1,
    state2,
    BETA1, BETA2,
    0.0,          # beta3 (unused for "adam")
    0.0,          # alpha (unused for "adam")
    EPS,
    STEP,
    LR,
    qmap1,
    qmap2,
    absmax1,
    absmax2,
    WD,           # weight_decay
    1.0,          # gnorm_scale
    False,        # skip_zeros
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

# -------- Persist --------
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adam8bit_data")
os.makedirs(out_dir, exist_ok=True)

# Note: bnb's reference is the GROUND TRUTH. We save with a "before/" or
# "after/" key prefix so the Rust loader can pull them all out of one map.
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
            "lr": LR,
            "beta1": BETA1,
            "beta2": BETA2,
            "eps": EPS,
            "wd": WD,
            "step": STEP,
            "blocksize": BLOCKSIZE,
            "numel": NUMEL,
            "bnb_version": bnb.__version__,
            "torch_version": torch.__version__,
        },
        f,
        indent=2,
    )

# Quick visibility prints (Rust diff is the gate).
def stats(name, t):
    f = t.detach().to(dtype=torch.float64).cpu()
    print(f"  {name:18s} dtype={str(t.dtype):14s} numel={t.numel():>6d} "
          f"min={float(f.min()):+.6g} max={float(f.max()):+.6g} "
          f"mean={float(f.mean()):+.6g}")

print("\nBEFORE:")
for k in ["param", "grad", "state1", "state2", "absmax1", "absmax2"]:
    stats(k, before[k])
print("\nAFTER (post bnb step):")
for k in ["param", "state1", "state2", "absmax1", "absmax2"]:
    stats(k, after[k])

print(f"\ndumped to {out_dir}")
print(f"  before.safetensors   ({os.path.getsize(os.path.join(out_dir, 'before.safetensors'))} bytes)")
print(f"  after.safetensors    ({os.path.getsize(os.path.join(out_dir, 'after.safetensors'))} bytes)")
print(f"  hyperparams.json     ({os.path.getsize(os.path.join(out_dir, 'hyperparams.json'))} bytes)")
