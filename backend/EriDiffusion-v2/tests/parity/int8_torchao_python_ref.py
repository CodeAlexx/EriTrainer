#!/usr/bin/env python3
"""torchao 0.14.1 `int8_weight_only_quantized_training` reference dump.

Drives the EXACT torchao API path used by SimpleTuner under
`model_type=lora` + `--base-quant int8-torchao`. We bypass the module-
swap helper (`quantize_(...)`) and call the underlying primitives so we
can capture intermediate tensors:

  - `quantize_int8_rowwise(W)` -> (int_data, scale)
      torchao/prototype/quantized_training/int8.py:23-52
  - `Int8QuantizedTrainingLinearWeight(int_data, scale)` wrapper
      torchao/prototype/quantized_training/int8.py:55-115
  - `F.linear(x, qw, bias)` -> dispatches to
      `_Int8WeightOnlyLinear.apply(x, qw, bias)` (int8.py:179-181, 146-173)
  - `y.sum().backward()` -> drives the autograd `backward` (int8.py:162-173)

Outputs (under tests/parity/int8_torchao_qt_data/):
  - ref.safetensors  with keys:
      w_f32     [OUT, IN]  F32   input weight (pre-quant)
      x_bf16    [B, IN]    BF16  input activation (requires_grad=True)
      b_bf16    [OUT]      BF16  bias
      int_data  [OUT, IN]  I8    torchao quantization output
      scale     [OUT]      F32   torchao per-row scale
      y_bf16    [B, OUT]   BF16  forward output
      grad_x    [B, IN]    BF16  d(sum(y))/dx via torchao backward

Seed is fixed to 42. Shapes are small enough that the whole run finishes
in < 5 s on a single GPU.

Exit codes: 0 OK, 2 BLOCKED (missing deps / no CUDA).
"""

from __future__ import annotations

import os
import sys

import torch

try:
    from torchao.prototype.quantized_training.int8 import (
        quantize_int8_rowwise,
        Int8QuantizedTrainingLinearWeight,
    )
except Exception as e:  # pragma: no cover
    print(f"BLOCKED: torchao import failed: {e}", file=sys.stderr)
    sys.exit(2)

if not torch.cuda.is_available():
    print("BLOCKED: CUDA not available", file=sys.stderr)
    sys.exit(2)

try:
    from safetensors.torch import save_file
except Exception as e:
    print(f"BLOCKED: safetensors import failed: {e}", file=sys.stderr)
    sys.exit(2)

OUT = 256
IN = 384
B = 8
SEED = 42
OUT_DIR = "/home/alex/EriDiffusion/EriDiffusion-v2/tests/parity/int8_torchao_qt_data"


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0")

    torch.manual_seed(SEED)
    # BF16 base weight (matches SimpleTuner reality: diffusion base
    # weights are BF16, then torchao's `quantize_(model, ...)` walks
    # `nn.Linear` modules and replaces `module.weight` with
    # `Int8QuantizedTrainingLinearWeight.from_float(module.weight)`,
    # which produces a scale matching the input dtype — BF16 here.
    # See torchao int8.py:40 (`scale = tensor.abs().amax(1) / 127`,
    # "same dtype as tensor"). The wrapper appears as `scale.dtype`
    # (int8.py:71-73), so F.linear keeps the BF16 compute path).
    w_bf16 = torch.randn(OUT, IN, dtype=torch.bfloat16, device=device)
    # We also keep an F32 copy of EXACTLY the same numerical values to
    # feed to our Rust quantizer for an apples-to-apples comparison
    # (torchao internally upcasts to F32 for the rounding step at
    # int8.py:42, so we must mirror that to match the codes exactly).
    w_f32 = w_bf16.detach().clone().to(torch.float32)

    b = torch.randn(OUT, dtype=torch.bfloat16, device=device)
    x = torch.randn(B, IN, dtype=torch.bfloat16, device=device, requires_grad=True)

    # 1. Quantize EXACTLY via torchao. Pass BF16 weight directly so the
    #    returned scale is BF16 — matches SimpleTuner.
    int_data, scale = quantize_int8_rowwise(w_bf16, stochastic_rounding=False)
    assert int_data.dtype == torch.int8
    assert int_data.shape == (OUT, IN)
    assert scale.dtype == torch.bfloat16, f"scale dtype is {scale.dtype}"
    assert scale.shape == (OUT,)
    # Upcast scale to F32 for the safetensors dump so the Rust binary
    # can compare exactly against host-quantized F32 scales.
    scale_f32 = scale.detach().to(torch.float32)

    # 2. Wrap and forward through F.linear (hits _Int8WeightOnlyLinear.apply).
    qw = Int8QuantizedTrainingLinearWeight(int_data, scale)
    # qw appears as scale.dtype = float32; F.linear will produce y in
    # input.dtype (= bf16). Verify by sniffing.
    y = torch.nn.functional.linear(x, qw, b)
    assert y.dtype == torch.bfloat16, f"y dtype = {y.dtype}"
    assert y.shape == (B, OUT)

    # 3. Backward: produce grad_x with grad_output = ones (so it's
    #    equivalent to y.sum().backward()).
    y.sum().backward()
    grad_x = x.grad.detach().clone()
    assert grad_x.dtype == torch.bfloat16

    # 4. Sanity print: first 10 codes + first 5 scales (lets the parity
    #    binary verify the dump came from the right seed).
    flat_codes = int_data.detach().contiguous().cpu().flatten()
    flat_scales = scale.detach().contiguous().cpu()
    print(f"OK seed={SEED} OUT={OUT} IN={IN} B={B}")
    print("first10 int_data:", flat_codes[:10].tolist())
    print("first5  scale   :", flat_scales[:5].tolist())

    # 5. Dump.
    out_path = os.path.join(OUT_DIR, "ref.safetensors")
    save_file({
        # F32 copy of the BF16 source weight (so our Rust quantizer sees
        # the exact same numerical values that torchao saw internally
        # after its upcast-to-F32 at int8.py:42).
        "w_f32":      w_f32.detach().contiguous().cpu(),
        # Also dump the original BF16 weight in case future tests want
        # to mirror SimpleTuner's `quantize_(model, ...)` invocation
        # with a real BF16 nn.Linear (currently unused by the Rust gate).
        "w_bf16":     w_bf16.detach().contiguous().cpu(),
        "x_bf16":     x.detach().contiguous().cpu(),
        "b_bf16":     b.detach().contiguous().cpu(),
        "int_data":   int_data.detach().contiguous().cpu(),
        "scale_bf16": scale.detach().contiguous().cpu(),
        "scale":      scale_f32.detach().contiguous().cpu(),
        "y_bf16":     y.detach().contiguous().cpu(),
        "grad_x":     grad_x.detach().contiguous().cpu(),
    }, out_path)
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
