#!/usr/bin/env python3
"""TASK B — full-model LoRA param-grad parity reference (conflated by the 4.2%
forward diff, but broad).

Injects the TRAINED LoRA (klein_lora_700steps.safetensors) into the diffusers
Flux2 full klein model as PARALLEL low-rank branches with leaf A/B params, on
the SAME fwd_fixture input (x_t, txt, timestep) our side uses. Runs full
fwd+bwd, dumps dL/d(lora_A) and dL/d(lora_B) per module under OUR key names so
the Rust side can compare param-grad cos per module.

Module map (ours -> diffusers):
  single_blocks.N.linear1   -> single_transformer_blocks[N].attn.to_qkv_mlp_proj
  single_blocks.N.linear2   -> single_transformer_blocks[N].attn.to_out
  double_blocks.N.img_attn.to_{q,k,v} -> transformer_blocks[N].attn.to_{q,k,v}
  double_blocks.N.img_attn.proj       -> transformer_blocks[N].attn.to_out.0
  double_blocks.N.txt_attn.to_{q,k,v} -> transformer_blocks[N].attn.add_{q,k,v}_proj
  double_blocks.N.txt_attn.proj       -> transformer_blocks[N].attn.to_add_out
  double_blocks.N.img_mlp.0/2 -> transformer_blocks[N].ff.linear_in/linear_out
  double_blocks.N.txt_mlp.0/2 -> transformer_blocks[N].ff_context.linear_in/linear_out

Output: lora_paramgrad_ref.safetensors with grad_<ourkey>.lora_{A,B} entries.
"""
import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file as st_load
from safetensors import safe_open

TF_DIR = ("/home/alex/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-9B/"
          "snapshots/32773329fbe7e81a90ef971740e8ba4b0364ecf3/transformer")
LORA = "/home/alex/EriDiffusion/EriDiffusion-v2/output/klein_ckpts/klein_lora_700steps.safetensors"
FIX = "/home/alex/EriDiffusion/EriDiffusion-v2/parity/klein_fwd/fwd_fixture.safetensors"
OUT = "/home/alex/EriDiffusion/EriDiffusion-v2/parity/klein_fwd/lora_paramgrad_ref.safetensors"

dev = "cuda"
dt = torch.bfloat16
RANK = 16
ALPHA = 16.0
SCALE = ALPHA / RANK  # 1.0

# --- reuse the EXACT fwd_fixture our side uses ---
fix = st_load(FIX, device=dev)
x_t = fix["x_t"].to(dt)                      # [1,128,H,W]
txt = fix["text_embedding"].to(dt)           # [1,512,12288]
target = fix["target"].to(torch.float32)     # [1,128,H,W]
sigma = float(fix["timestep"][0].item())
B, C, H, W = x_t.shape
print(f"x_t={tuple(x_t.shape)} txt={tuple(txt.shape)} sigma={sigma}", flush=True)

from diffusers import Flux2Transformer2DModel
tf = Flux2Transformer2DModel.from_pretrained(TF_DIR, torch_dtype=dt).to(dev)
tf.enable_gradient_checkpointing()
tf.train()
for p in tf.parameters():
    p.requires_grad_(False)

# --- module map ---
def lora_targets():
    m = {}
    nd = len(tf.transformer_blocks)
    ns = len(tf.single_transformer_blocks)
    for n in range(ns):
        m[f"single_blocks.{n}.linear1"] = tf.single_transformer_blocks[n].attn.to_qkv_mlp_proj
        m[f"single_blocks.{n}.linear2"] = tf.single_transformer_blocks[n].attn.to_out
    for n in range(nd):
        blk = tf.transformer_blocks[n]
        m[f"double_blocks.{n}.img_attn.to_q"] = blk.attn.to_q
        m[f"double_blocks.{n}.img_attn.to_k"] = blk.attn.to_k
        m[f"double_blocks.{n}.img_attn.to_v"] = blk.attn.to_v
        m[f"double_blocks.{n}.img_attn.proj"] = blk.attn.to_out[0]
        m[f"double_blocks.{n}.txt_attn.to_q"] = blk.attn.add_q_proj
        m[f"double_blocks.{n}.txt_attn.to_k"] = blk.attn.add_k_proj
        m[f"double_blocks.{n}.txt_attn.to_v"] = blk.attn.add_v_proj
        m[f"double_blocks.{n}.txt_attn.proj"] = blk.attn.to_add_out
        m[f"double_blocks.{n}.img_mlp.0"] = blk.ff.linear_in
        m[f"double_blocks.{n}.img_mlp.2"] = blk.ff.linear_out
        m[f"double_blocks.{n}.txt_mlp.0"] = blk.ff_context.linear_in
        m[f"double_blocks.{n}.txt_mlp.2"] = blk.ff_context.linear_out
    return m

targets = lora_targets()


def lora_delta(x, A, B, scale):
    Ab = A.to(dt); Bb = B.to(dt)
    return F.linear(F.linear(x, Ab), Bb) * scale


class LoraWrap(torch.nn.Module):
    def __init__(self, base, A, B, scale):
        super().__init__()
        self.base = base
        self.A = A
        self.B = B
        self.scale = scale

    def forward(self, x):
        return self.base(x) + lora_delta(x, self.A, self.B, self.scale)


def set_submodule(root, dotted, new):
    # dotted is a module path relative to a parent; replace it in place.
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


# inject all adapters; keep leaf params keyed by our name
params = {}  # ourkey -> (A_param, B_param)
with safe_open(LORA, "pt", device=dev) as f:
    for ourkey, mod in targets.items():
        A = f.get_tensor(f"{ourkey}.lora_A.weight").float().clone().requires_grad_(True)
        Bw = f.get_tensor(f"{ourkey}.lora_B.weight").float().clone().requires_grad_(True)
        wrap = LoraWrap(mod, A, Bw, SCALE)
        # find dotted path of `mod` within tf and replace
        # easier: re-resolve the parent via the same logic as lora_targets
        params[ourkey] = (A, Bw)
        # replace
        # build the dotted path:
        if ourkey.startswith("single_blocks"):
            n = ourkey.split(".")[1]
            which = ourkey.split(".")[2]
            sub = "to_qkv_mlp_proj" if which == "linear1" else "to_out"
            set_submodule(tf.single_transformer_blocks[int(n)].attn, sub, wrap)
        else:
            parts = ourkey.split(".")
            n = int(parts[1]); grp = parts[2]; name = parts[3] if len(parts) > 3 else None
            blk = tf.transformer_blocks[n]
            if grp == "img_attn":
                sub = {"to_q": ("attn", "to_q"), "to_k": ("attn", "to_k"),
                       "to_v": ("attn", "to_v"), "proj": ("attn", "to_out.0")}[name]
                set_submodule(getattr(blk, sub[0]), sub[1], wrap)
            elif grp == "txt_attn":
                sub = {"to_q": "add_q_proj", "to_k": "add_k_proj",
                       "to_v": "add_v_proj", "proj": "to_add_out"}[name]
                set_submodule(blk.attn, sub, wrap)
            elif grp == "img_mlp":
                set_submodule(blk.ff, "linear_in" if name == "0" else "linear_out", wrap)
            elif grp == "txt_mlp":
                set_submodule(blk.ff_context, "linear_in" if name == "0" else "linear_out", wrap)

print(f"injected {len(params)} LoRA adapters", flush=True)

# --- forward + backward (matches ref.py packing) ---
def pack(l):
    b, c, h, w = l.shape
    return l.reshape(b, c, h * w).permute(0, 2, 1)

hidden = pack(x_t)
t_ax = torch.arange(1); h_ax = torch.arange(H); w_ax = torch.arange(W); l_ax = torch.arange(1)
img_ids = torch.cartesian_prod(t_ax, h_ax, w_ax, l_ax).to(dev).unsqueeze(0).expand(B, -1, -1).float()
n_txt = txt.shape[1]
txt_ids = torch.zeros(B, n_txt, 4, device=dev).float()
timestep = torch.tensor([sigma], device=dev, dtype=dt).expand(B)
guidance = torch.full([B], 4.0, device=dev, dtype=torch.float32)

out = tf(hidden_states=hidden, encoder_hidden_states=txt, timestep=timestep,
         img_ids=img_ids, txt_ids=txt_ids, guidance=guidance, return_dict=False)[0]
vel = out[:, :H * W, :].permute(0, 2, 1).reshape(B, C, H, W)
loss = ((vel.float() - target.float()) ** 2).mean()
loss.backward()
print(f"|vel|={vel.float().norm():.4e} loss={loss.item():.6e}", flush=True)

dump = {}
for ourkey, (A, Bw) in params.items():
    dump[f"grad_{ourkey}.lora_A"] = A.grad.detach().float().contiguous().cpu()
    dump[f"grad_{ourkey}.lora_B"] = Bw.grad.detach().float().contiguous().cpu()
# also dump the LoRA-injected velocity so the Rust side can verify forward cos
dump["velocity_lora_ref"] = vel.detach().float().contiguous().cpu()
save_file(dump, OUT)
print(f"wrote {OUT} with {len(dump)} grads", flush=True)
