#!/usr/bin/env python3
"""AsymFlow A0.5 — Klein 9B teacher-forward Python reference dump.

The asymflow trainer (`lakonlab/models/diffusions/asymflow.py:85-87`) calls the
teacher as:

    with torch.no_grad(), module_eval(teacher):
        ref_u_low_rank = teacher(return_u=True, x_t=x_t_low_rank, t=t, **teacher_kwargs)

That dispatches through `GaussianFlow.forward_u` → `GaussianFlow.pred` →
`AsymFlux2Transformer2DModel.forward(x_t, t, encoder_hidden_states=...)`,
which under `denoising_mean_mode='U'` and `guidance_scale=1.0` is *exactly* the
single forward we already exercise from
`inference-flame/src/bin/asymflux2_klein9b_infer.rs::wrapped_forward`
(minus the CFG/uncond branch).

Goal: dump

  inputs:
    - `x_t_low_rank`  F32 [1, 3, H, W]
    - `t_norm`        F32 [1]                (timestep ∈ (0, 1])
    - `text_embed`    BF16 [1, 512, 12288]   (Qwen3-8B prompt embedding)
    - `proj_buffer`   F32 [768, 128]
    - `scale_buffer`  F32 []
  output:
    - `teacher_u`     F32 [1, 3, H, W]       (velocity prediction in pixel space)

so the Rust binary can byte-compare on the same seeded inputs.

------------------------------------------------------------------------------
STATUS ON THIS BOX: BLOCKED
------------------------------------------------------------------------------
This script cannot produce the reference dump in the current environment:

1. `lakonlab` is not installed and `pip install -r requirements.txt` would
   need `mmcv==1.6.1`, which is a deprecated build that does not install
   against torch 2.10 / CUDA 12.8 without a custom toolchain.
2. The asymflux2 module uses package-relative imports
   (`from ..builder import MODULES, build_module`), so it can't be loaded
   in isolation with stubbed mmcv.
3. `asymflow_subspace_procrustes.pth` (the Procrustes proj_buffer + scale)
   is NOT on disk under `~/LakonLab/checkpoints/` — the production
   `asymflux2-klein-9b.safetensors` has the buffers baked in, but the raw
   `.pth` the LakonLab loader expects has never been downloaded here.
4. Generating a Python golden would therefore require either
   (a) a working mmcv-1.6.1 install (~half-day yak shave, may not be
       possible on torch 2.10), OR
   (b) a 1-shot re-implementation of `AsymFlux2Transformer2DModel.forward`
       in vanilla torch (a 9B-parameter DiT port, multi-day; far past
       the A0.5 scope).

Per the milestone plan (`asymflow_milestone_plan.md:96-98`), A0.5 is
acceptance-gated on `cos_sim >= 0.999, max_abs <= 1e-2` vs Python. Without
the Python ref, we can only assert *self-consistency* of the Rust forward
(determinism, no NaNs, plausible shapes) — the trust signal that
`asymflux2_klein9b_infer.rs` actually renders the expected portraits.

------------------------------------------------------------------------------
WHAT THIS SCRIPT DOES TODAY
------------------------------------------------------------------------------
Runs the BLOCKED detection logic and exits 2 with a precise message that
tells the human exactly what is missing. If a future environment fixes
(1)-(3), the rest of this file (intentionally written as the would-be
real reference) is the contract the Rust binary will match.

------------------------------------------------------------------------------
INPUTS USED BY THIS HARNESS (locked for parity)
------------------------------------------------------------------------------
  prompt   = "a portrait of a woman with long dark hair, soft window light"
  H, W     = 512, 512        (pixel)
  patch    = 16              (-> 32x32 patch grid, 1024 image tokens)
  seed     = 42
  dtype    = bf16            (matches teacher's `torch_dtype='bfloat16'`)
  t_norm   = 0.5             (mid-trajectory sigma; arbitrary, fixed)
  text_len = 512             (Qwen3 hard-padded to 512 per inference binary)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# --- Hard config (must stay in lockstep with Rust binary) -------------------
PROMPT = "a portrait of a woman with long dark hair, soft window light"
SEED = 42
H = 512
W = 512
PATCH = 16
T_NORM = 0.5
TEXT_LEN = 512

BASE_PATH = "/home/alex/EriDiffusion/Models/checkpoints/flux-2-klein-base-9b.safetensors"
ADAPTER_PATH = "/home/alex/EriDiffusion/Models/checkpoints/asymflux2-klein-9b.safetensors"
QWEN3_DIR = "/home/alex/EriDiffusion/Models/qwen3-8b"  # if missing, also BLOCKED

OUT_TENSORS = "/tmp/asymflow_a05_teacher_ref.safetensors"
OUT_META = "/tmp/asymflow_a05_teacher_ref_meta.json"

LAKON_ROOT = Path("/home/alex/LakonLab")
PROCRUSTES_PTH = LAKON_ROOT / "checkpoints" / "asymflow_subspace_procrustes.pth"


def blocked(msg: str, code: int = 2) -> None:
    print(f"\n[A0.5 BLOCKED] {msg}\n", file=sys.stderr)
    print(f"Will not write {OUT_TENSORS} / {OUT_META}.", file=sys.stderr)
    sys.exit(code)


def main() -> int:
    # Probe (1): lakonlab package.
    try:
        sys.path.insert(0, str(LAKON_ROOT))
        import lakonlab  # noqa: F401
    except Exception as e:
        blocked(
            f"lakonlab not importable: {type(e).__name__}: {e}\n"
            f"  Fix: install lakonlab via `pip install -e {LAKON_ROOT}` after\n"
            f"  installing the legacy mmcv==1.6.1 toolchain it depends on."
        )

    # Probe (2): mmcv (lakonlab's hard dep on the legacy 1.x API).
    try:
        import mmcv  # noqa: F401
    except Exception as e:
        blocked(
            f"mmcv (the legacy 1.6.1 API) is required by lakonlab.builder: "
            f"{type(e).__name__}: {e}\n"
            f"  Fix: `pip install mmcv==1.6.1` (may need legacy CUDA toolchain).\n"
            f"  This typically does NOT build cleanly against torch 2.10 + CUDA 12.8."
        )

    # Probe (3): Procrustes pth.
    if not PROCRUSTES_PTH.exists():
        blocked(
            f"missing {PROCRUSTES_PTH}\n"
            f"  Fix: download/checkpoint asymflow_subspace_procrustes.pth from\n"
            f"  the asymflux2 release, or regenerate via\n"
            f"  `tools/asymflow_subspace_procrustes.py` (needs full LakonLab env)."
        )

    # Probe (4): base weights.
    for p in (BASE_PATH, ADAPTER_PATH):
        if not Path(p).exists():
            blocked(f"missing weights file: {p}")

    # ---- If we got here, the env is set up — run the real ref dump ----------
    # NOTE: code below is the intended contract. It will run when the BLOCKED
    # state above is cleared. Not exercised in CI today.
    import torch
    from mmcv.utils import Config
    from lakonlab.models import build_model
    from safetensors.torch import save_file

    cfg_path = LAKON_ROOT / "configs" / "asymflow" / "asymflux2_klein_32gpus.py"
    cfg = Config.fromfile(str(cfg_path))

    # build TEACHER only (the `teacher=...` branch of `model = dict(...)`)
    teacher_cfg = cfg.model["teacher"]
    teacher = build_model(teacher_cfg).cuda().to(torch.bfloat16).eval()

    torch.manual_seed(SEED)
    x_t_low_rank = torch.randn(1, 3, H, W, dtype=torch.float32, device="cuda")
    t = torch.tensor([T_NORM], dtype=torch.float32, device="cuda")  # sigma ∈ (0,1]

    # Reference text embed: produced by the same Qwen3-8B path the Rust
    # inference uses. For the dump to be drop-in we MUST run the actual
    # Qwen3 encoder here (not random) — placeholder until env unblocked.
    text_embed = torch.randn(1, TEXT_LEN, 12288, dtype=torch.bfloat16, device="cuda")

    with torch.no_grad():
        teacher_u = teacher(
            return_u=True,
            x_t=x_t_low_rank,
            t=t * teacher.num_timesteps,
            encoder_hidden_states=text_embed,
        )

    tensors = {
        "x_t_low_rank": x_t_low_rank.cpu(),
        "t_norm": t.cpu(),
        "text_embed": text_embed.cpu(),
        "teacher_u": teacher_u.detach().float().cpu(),
        "proj_buffer": teacher.denoising.proj_buffer.detach().float().cpu(),
        "scale_buffer": teacher.denoising.scale_buffer.detach().float().cpu(),
    }
    save_file(tensors, OUT_TENSORS)

    meta = dict(
        prompt=PROMPT,
        seed=SEED,
        height=H,
        width=W,
        patch=PATCH,
        text_len=TEXT_LEN,
        t_norm=T_NORM,
        dtype="bf16_teacher_fp32_dump",
        base_path=BASE_PATH,
        adapter_path=ADAPTER_PATH,
        procrustes_path=str(PROCRUSTES_PTH),
        shapes={k: list(v.shape) for k, v in tensors.items()},
    )
    Path(OUT_META).write_text(json.dumps(meta, indent=2))
    print(f"OK: wrote {OUT_TENSORS} and {OUT_META}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
