#!/usr/bin/env python3
"""TASK A — single-block LoRA-PATH backward parity at LARGE ‖B‖ (de-conflated).

Extends parity/klein_sb to inject the TRAINED block-N LoRA into the diffusers
Flux2 single block on CONTROLLED seeded inputs, and dump:
  - forward output (LoRA-injected)  -> validity gate vs our side (cos > 0.999)
  - dL/d(lora_B) for linear1 (to_qkv_mlp_proj) and linear2 (to_out)
  - dL/d(lora_A) too (for completeness)
on a fixed random upstream grad G (so the loss is L = <out, G>).

The LoRA is injected as a PARALLEL low-rank branch with leaf A,B parameters
(NOT baked into W) so torch returns the true dL/dB. The branch matches our
`LoRALinear::forward_delta` byte-for-byte:
    delta = scale * (x @ A^T @ B^T),  scale = alpha/rank,
    A,B cast to BF16 before the matmuls (the base linear is BF16),
    A:[rank,in], B:[out,rank]  (lora_A=down, lora_B=up).

Block index is arg1 (default 0). Run once per block we want to test.

Dumps to sb_lora_fixture_b{BLK}.safetensors:
  hidden_states [1,N,4096], mod_shift/scale/gate [1,1,4096],
  rope_cos/rope_sin [N,128], img_ids [N,4], G [1,N,4096],
  output [1,N,4096],
  lin1_A,lin1_B,lin2_A,lin2_B (the injected weights, F32),
  grad_lin1_A, grad_lin1_B, grad_lin2_A, grad_lin2_B (dL/d that param, F32).
"""
import sys
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

TF_DIR = ("/home/alex/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-9B/"
          "snapshots/32773329fbe7e81a90ef971740e8ba4b0364ecf3/transformer")
LORA = "/home/alex/EriDiffusion/EriDiffusion-v2/output/klein_ckpts/klein_lora_700steps.safetensors"

BLK = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OUT = f"/home/alex/EriDiffusion/EriDiffusion-v2/parity/klein_sb/sb_lora_fixture_b{BLK}.safetensors"

dev = "cuda"
dt = torch.bfloat16
DIM = 4096
HEADS = 32
HEAD_DIM = 128
MLP_HIDDEN = 12288
THETA = 2000
AXES = [32, 32, 32, 32]
RANK = 16
ALPHA = 16.0
SCALE = ALPHA / RANK  # 1.0

GH, GW = 16, 16
N = GH * GW

torch.manual_seed(20260527)

hidden_f32 = torch.randn(1, N, DIM, device=dev, dtype=torch.float32)
mod_shift = torch.randn(1, 1, DIM, device=dev, dtype=torch.float32) * 0.1
mod_scale = torch.randn(1, 1, DIM, device=dev, dtype=torch.float32) * 0.1
mod_gate = torch.randn(1, 1, DIM, device=dev, dtype=torch.float32) * 0.1
G = torch.randn(1, N, DIM, device=dev, dtype=torch.float32)

temb_mod = torch.cat([mod_shift, mod_scale, mod_gate], dim=-1).to(dt)

t_ax = torch.arange(1); h_ax = torch.arange(GH); w_ax = torch.arange(GW); l_ax = torch.arange(1)
img_ids = torch.cartesian_prod(t_ax, h_ax, w_ax, l_ax).to(dev).float()

from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed
pos_embed = Flux2PosEmbed(theta=THETA, axes_dim=AXES)
rope_cos, rope_sin = pos_embed(img_ids)
rope_cos = rope_cos.to(dev); rope_sin = rope_sin.to(dev)
image_rotary_emb = (rope_cos, rope_sin)

from diffusers import Flux2Transformer2DModel
tf = Flux2Transformer2DModel.from_pretrained(TF_DIR, torch_dtype=dt).to(dev)
tf.eval()
block = tf.single_transformer_blocks[BLK]
for p in block.parameters():
    p.requires_grad_(False)

# ---- load the trained LoRA for this block ----
from safetensors import safe_open
with safe_open(LORA, "pt", device=dev) as f:
    lin1_A = f.get_tensor(f"single_blocks.{BLK}.linear1.lora_A.weight").float()  # [rank, 4096]
    lin1_B = f.get_tensor(f"single_blocks.{BLK}.linear1.lora_B.weight").float()  # [36864, rank]
    lin2_A = f.get_tensor(f"single_blocks.{BLK}.linear2.lora_A.weight").float()  # [rank, 16384]
    lin2_B = f.get_tensor(f"single_blocks.{BLK}.linear2.lora_B.weight").float()  # [4096, rank]

print(f"block {BLK}: lin1_A{tuple(lin1_A.shape)} lin1_B{tuple(lin1_B.shape)} "
      f"|lin1_B|={lin1_B.norm():.3f}  lin2_A{tuple(lin2_A.shape)} lin2_B{tuple(lin2_B.shape)} "
      f"|lin2_B|={lin2_B.norm():.3f}", flush=True)

# leaf params (F32) — torch returns dL/d these
lin1_A = lin1_A.clone().requires_grad_(True)
lin1_B = lin1_B.clone().requires_grad_(True)
lin2_A = lin2_A.clone().requires_grad_(True)
lin2_B = lin2_B.clone().requires_grad_(True)


def lora_delta(x, A, B, scale):
    # mirror LoRALinear::forward_delta: cast A,B to bf16, delta = scale * x @ A^T @ B^T
    Ab = A.to(dt)
    Bb = B.to(dt)
    inter = F.linear(x, Ab)        # x @ A^T  -> [.., rank]
    out = F.linear(inter, Bb)      # @ B^T    -> [.., out]
    return out * scale


# ---- monkeypatch to_qkv_mlp_proj (linear1) and to_out (linear2) ----
attn = block.attn
base_lin1 = attn.to_qkv_mlp_proj
base_lin2 = attn.to_out


class LoraWrap(torch.nn.Module):
    def __init__(self, base, A, B, scale):
        super().__init__()
        self.base = base
        self.A = A
        self.B = B
        self.scale = scale

    def forward(self, x):
        return self.base(x) + lora_delta(x, self.A, self.B, self.scale)


attn.to_qkv_mlp_proj = LoraWrap(base_lin1, lin1_A, lin1_B, SCALE)
attn.to_out = LoraWrap(base_lin2, lin2_A, lin2_B, SCALE)

# ---- forward + backward ----
hidden_in = hidden_f32.to(dt)  # NOT a leaf here; we only need param grads
out = block(
    hidden_states=hidden_in,
    encoder_hidden_states=None,
    temb_mod=temb_mod,
    image_rotary_emb=image_rotary_emb,
)
if isinstance(out, (tuple, list)):
    out = out[-1]

loss = (out.float() * G).sum()
loss.backward()

print(f"out {tuple(out.shape)} |out|={out.float().norm():.4e}  loss={loss.item():.6e}", flush=True)
print(f"  |dL/d lin1_B|={lin1_B.grad.norm():.4e}  |dL/d lin2_B|={lin2_B.grad.norm():.4e}", flush=True)
print(f"  |dL/d lin1_A|={lin1_A.grad.norm():.4e}  |dL/d lin2_A|={lin2_A.grad.norm():.4e}", flush=True)

save_file({
    "hidden_states": hidden_f32.contiguous().cpu(),
    "mod_shift": mod_shift.contiguous().cpu(),
    "mod_scale": mod_scale.contiguous().cpu(),
    "mod_gate": mod_gate.contiguous().cpu(),
    "rope_cos": rope_cos.float().contiguous().cpu(),
    "rope_sin": rope_sin.float().contiguous().cpu(),
    "img_ids": img_ids.contiguous().cpu(),
    "G": G.contiguous().cpu(),
    "output": out.detach().float().contiguous().cpu(),
    "lin1_A": lin1_A.detach().float().contiguous().cpu(),
    "lin1_B": lin1_B.detach().float().contiguous().cpu(),
    "lin2_A": lin2_A.detach().float().contiguous().cpu(),
    "lin2_B": lin2_B.detach().float().contiguous().cpu(),
    "grad_lin1_A": lin1_A.grad.detach().float().contiguous().cpu(),
    "grad_lin1_B": lin1_B.grad.detach().float().contiguous().cpu(),
    "grad_lin2_A": lin2_A.grad.detach().float().contiguous().cpu(),
    "grad_lin2_B": lin2_B.grad.detach().float().contiguous().cpu(),
}, OUT)
print(f"wrote {OUT}", flush=True)
