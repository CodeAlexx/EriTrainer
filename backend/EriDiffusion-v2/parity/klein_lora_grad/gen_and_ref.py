#!/usr/bin/env python3
"""
Single-step LoRA gradient parity fixture + PyTorch reference.

Mirrors eridiffusion-core/src/lora.rs::LoRALinear::forward_delta:
    a   = A.to(bf16); b = B.to(bf16)
    inter = X @ a^T                  # [.,rank]
    out   = inter @ b^T              # [.,out]
    delta = (out * (alpha/rank))     # scale in compute dtype
    loss  = mean( (delta.float() - target)^2 )

Two references are produced from the SAME fixture:
  * f32        — everything in fp32 (mathematical ground truth).
  * bf16mirror — X bf16, A/B cast to bf16 in the forward (grad flows back to
                 the f32 leaf via the differentiable cast), matmuls in bf16,
                 loss in f32. This is EXACTLY what our Rust path does, so it
                 is the correct ground truth for our implementation.

If our Rust grad matches bf16mirror -> our backward is correct (and any
learning weakness is recipe/precision, not a backward bug).
If it matches neither -> our backward is wrong, and the A/B grad diff
localizes it.
"""
import torch
from safetensors.torch import save_file

torch.manual_seed(1234)
dev = "cuda"

# One real klein module shape: double_blocks.0.img_attn.proj
IN, OUT, RANK, ALPHA = 4096, 4096, 16, 16.0
SEQ = 256
scale = ALPHA / RANK

# Fixture (stored f32). Scales chosen to resemble a trained adapter:
X = torch.randn(1, SEQ, IN, device=dev) * 1.0
T = torch.randn(1, SEQ, OUT, device=dev) * 1.0          # MSE target (= the "flow" target stand-in)
A = (torch.randn(RANK, IN, device=dev) * (1.0 / IN ** 0.5))  # kaiming-ish, like our init
B = torch.randn(OUT, RANK, device=dev) * 2.0e-3          # small, like a partially-trained lora_B

save_file(
    {"X": X.cpu(), "T": T.cpu(), "A": A.cpu(), "B": B.cpu()},
    "fixture.safetensors",
)

def run(regime):
    Af = A.clone().float().requires_grad_(True)
    Bf = B.clone().float().requires_grad_(True)
    if regime == "f32":
        x = X.float()
        a, b = Af, Bf
        inter = x @ a.t()
        out = inter @ b.t()
        delta = out * scale
    elif regime == "bf16mirror":
        x = X.to(torch.bfloat16)
        a = Af.to(torch.bfloat16)
        b = Bf.to(torch.bfloat16)
        inter = x @ a.t()
        out = inter @ b.t()
        delta = (out * scale).float()
    else:
        raise ValueError(regime)
    loss = ((delta.float() - T.float()) ** 2).mean()
    loss.backward()
    return loss.item(), delta.detach().float(), Af.grad.detach().clone(), Bf.grad.detach().clone()

out = {}
for regime in ("f32", "bf16mirror"):
    loss, delta, gA, gB = run(regime)
    out[f"grad_A.{regime}"] = gA.cpu()
    out[f"grad_B.{regime}"] = gB.cpu()
    out[f"delta.{regime}"] = delta.cpu()
    print(f"[{regime:10}] loss={loss:.6e}  "
          f"|gA|={gA.norm().item():.6e}  |gB|={gB.norm().item():.6e}  "
          f"|delta|={delta.norm().item():.6e}")

save_file(out, "ref.safetensors")
print("wrote fixture.safetensors + ref.safetensors")
print(f"config: IN={IN} OUT={OUT} RANK={RANK} ALPHA={ALPHA} SEQ={SEQ} scale={scale}")
