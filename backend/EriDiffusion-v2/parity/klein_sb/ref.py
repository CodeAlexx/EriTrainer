#!/usr/bin/env python3
"""STANDALONE single-block forward+backward parity reference.

Isolates diffusers Flux2 single-stream block[0] on CONTROLLED, seeded random
inputs/weights so our Rust `single_block_forward_standalone` can be compared
on the IDENTICAL data. Removes the full-model forward-act conflation that
invalidated prior backward measurements.

Dumps to sb_fixture.safetensors:
  hidden_states  [1,N,4096]   block input (leaf, requires_grad)
  mod_shift/scale/gate [1,1,4096]
  rope_cos/rope_sin [N,128]    diffusers Flux2PosEmbed interleaved tables
  img_ids        [N,4]         positional ids (so Rust rebuilds rope its way)
  G              [1,N,4096]    upstream grad
  output         [1,N,4096]    block forward output
  grad_hidden_states [1,N,4096] dL/d(hidden_states)

  cargo build/run parity_klein_single_block reads this.
"""
import torch
from safetensors.torch import save_file
from safetensors import safe_open

CKPT = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"
TF_DIR = ("/home/alex/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-9B/"
          "snapshots/32773329fbe7e81a90ef971740e8ba4b0364ecf3/transformer")
OUT = "/home/alex/EriDiffusion/EriDiffusion-v2/parity/klein_sb/sb_fixture.safetensors"

dev = "cuda"
dt = torch.bfloat16
DIM = 4096
HEADS = 32
HEAD_DIM = 128
MLP_HIDDEN = 12288
THETA = 2000
AXES = [32, 32, 32, 32]

# Grid: a 16x16 image grid -> N=256 tokens (encoder_hidden_states=None so the
# block treats `hidden` as the full stream).
GH, GW = 16, 16
N = GH * GW

torch.manual_seed(20260527)

# ---- controlled seeded inputs (build in F32 then cast; keep F32 master) ----
hidden_f32 = torch.randn(1, N, DIM, device=dev, dtype=torch.float32)
# modulation params: small, like a real adaLN output
mod_shift = torch.randn(1, 1, DIM, device=dev, dtype=torch.float32) * 0.1
mod_scale = torch.randn(1, 1, DIM, device=dev, dtype=torch.float32) * 0.1
mod_gate = torch.randn(1, 1, DIM, device=dev, dtype=torch.float32) * 0.1
G = torch.randn(1, N, DIM, device=dev, dtype=torch.float32)

# temb_mod for Flux2Modulation.split(temb_mod, 1)[0] -> (shift, scale, gate)
temb_mod = torch.cat([mod_shift, mod_scale, mod_gate], dim=-1).to(dt)  # [1,1,3*DIM]

# ---- positional ids: img grid (t,h,w,l) cartesian, exactly like the full model ----
t_ax = torch.arange(1); h_ax = torch.arange(GH); w_ax = torch.arange(GW); l_ax = torch.arange(1)
img_ids = torch.cartesian_prod(t_ax, h_ax, w_ax, l_ax).to(dev).float()  # [N,4]

# ---- diffusers Flux2PosEmbed -> interleaved (cos,sin) [N,128] ----
from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed
pos_embed = Flux2PosEmbed(theta=THETA, axes_dim=AXES)
rope_cos, rope_sin = pos_embed(img_ids)  # each [N,128], interleaved (repeat_interleave_real=True)
rope_cos = rope_cos.to(dev); rope_sin = rope_sin.to(dev)
image_rotary_emb = (rope_cos, rope_sin)
print(f"rope_cos {tuple(rope_cos.shape)} rope_sin {tuple(rope_sin.shape)} N={N}", flush=True)

# ---- load diffusers single block[0] with real weights ----
from diffusers import Flux2Transformer2DModel
tf = Flux2Transformer2DModel.from_pretrained(TF_DIR, torch_dtype=dt).to(dev)
tf.eval()
block = tf.single_transformer_blocks[0]
for p in block.parameters():
    p.requires_grad_(False)

# sanity: print the weight shapes the block actually uses, to confirm fusion layout
sd = block.state_dict()
for k in sd:
    print("  block param:", k, tuple(sd[k].shape), flush=True)

# ---- forward + backward ----
hidden_leaf = hidden_f32.clone().requires_grad_(True)
hidden_in = hidden_leaf.to(dt)

out = block(
    hidden_states=hidden_in,
    encoder_hidden_states=None,
    temb_mod=temb_mod,
    image_rotary_emb=image_rotary_emb,
)
# block returns a single tensor when split_hidden_states is False
if isinstance(out, (tuple, list)):
    out = out[-1]

loss = (out.float() * G).sum()
loss.backward()
grad_hidden = hidden_leaf.grad.detach().float()

print(f"out {tuple(out.shape)} |out|={out.float().norm():.4e}  loss={loss.item():.6e}  "
      f"|grad_hidden|={grad_hidden.norm():.4e}", flush=True)

# ---- BF16 precision-floor probe: deepcopy block to F32, run on F32 inputs ----
import copy
import numpy as np
block_f32 = copy.deepcopy(block).to(torch.float32)
temb_mod_f32 = torch.cat([mod_shift, mod_scale, mod_gate], dim=-1).to(torch.float32)
if isinstance(image_rotary_emb, tuple):
    image_rotary_emb_f32 = tuple(t.float() for t in image_rotary_emb)
else:
    image_rotary_emb_f32 = image_rotary_emb.float()
with torch.no_grad():
    out_f32 = block_f32(
        hidden_states=hidden_f32.detach().clone(),
        encoder_hidden_states=None,
        temb_mod=temb_mod_f32,
        image_rotary_emb=image_rotary_emb_f32,
    )
    if isinstance(out_f32, (tuple, list)):
        out_f32 = out_f32[-1]
a = out.detach().float().contiguous().cpu().reshape(-1).numpy()
b = out_f32.detach().float().contiguous().cpu().reshape(-1).numpy()
def _cos(x, y):
    return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))
def _relL2(x, y):
    return float(np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-30))
print(f"\n=== BF16 PRECISION FLOOR (diffusers single-block: its own BF16 vs F32) ===", flush=True)
print(f"  cos(BF16, F32)   = {_cos(a, b):.6f}", flush=True)
print(f"  relL2(BF16 vs F32) = {_relL2(a, b):.4e}", flush=True)
print(f"  |out_BF16|={np.linalg.norm(a):.4e}  |out_F32|={np.linalg.norm(b):.4e}", flush=True)
print(f"  Compare to ours-vs-diffusers-BF16 (existing SB_PARITY): relL2 ~1.7e-3", flush=True)
del block_f32

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
    "grad_hidden_states": grad_hidden.contiguous().cpu(),
}, OUT)
print(f"wrote {OUT}", flush=True)
