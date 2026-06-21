#!/usr/bin/env python3
"""SDPA backward direction parity vs PyTorch.

klein's attention calls `flame_core::attention::sdpa(q,k,v,None)` on
[B,H,S,D] BF16 tensors (H=32, D=128). Its backward routes through
`try_cudnn_sdpa_backward`. The autograd source warns this path can produce
"~5%-direction-wrong dq/dk/dv that compounds multiplicatively across all 25
Klein blocks". This test measures that directly.

Method: inject a FIXED upstream grad G via loss = sum(out * G), so dOut = G
EXACTLY in both frameworks. dQ/dK/dV are then the pure SDPA backward of G.
Compare direction (cosine) to PyTorch's SDPA backward.
"""
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

torch.manual_seed(7)
dev = "cuda"
B, H, S, D = 1, 32, 256, 128          # klein: 32 heads, head_dim 128
scale = 1.0 / (D ** 0.5)

Q = torch.randn(B, H, S, D, device=dev)
K = torch.randn(B, H, S, D, device=dev)
V = torch.randn(B, H, S, D, device=dev)
G = torch.randn(B, H, S, D, device=dev)   # the fixed upstream grad (dOut)

save_file({"Q": Q.cpu(), "K": K.cpu(), "V": V.cpu(), "G": G.cpu()},
          "fixture.safetensors")

def run(regime):
    if regime == "f32":
        q = Q.clone().requires_grad_(True)
        k = K.clone().requires_grad_(True)
        v = V.clone().requires_grad_(True)
        g = G
    else:  # bf16mirror: matches our path (bf16 in, fp32 softmax inside, bf16 out)
        q = Q.to(torch.bfloat16).clone().requires_grad_(True)
        k = K.to(torch.bfloat16).clone().requires_grad_(True)
        v = V.to(torch.bfloat16).clone().requires_grad_(True)
        g = G.to(torch.bfloat16)
    # math-kernel SDPA = fp32 softmax internally, matches flame's fp32-softmax sdpa
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = F.scaled_dot_product_attention(q, k, v, scale=scale)
    loss = (out.float() * g.float()).sum()
    loss.backward()
    return out.detach().float(), q.grad.detach().float(), k.grad.detach().float(), v.grad.detach().float()

ref = {}
for regime in ("f32", "bf16mirror"):
    out, dq, dk, dv = run(regime)
    ref[f"out.{regime}"] = out.cpu()
    ref[f"dQ.{regime}"] = dq.cpu()
    ref[f"dK.{regime}"] = dk.cpu()
    ref[f"dV.{regime}"] = dv.cpu()
    print(f"[{regime:10}] |out|={out.norm():.4e} |dQ|={dq.norm():.4e} "
          f"|dK|={dk.norm():.4e} |dV|={dv.norm():.4e}")

save_file(ref, "ref.safetensors")
print(f"config B={B} H={H} S={S} D={D} scale={scale}")
print("wrote fixture.safetensors + ref.safetensors")
