"""Gate B ORACLE: true LoRA gradient via PyTorch autograd through ai-toolkit's
(tested) ideogram4 transformer, with a LoRA matching our forward_delta exactly
(bf16-cast, scale=alpha/rank) at the same target linears (qkv/o/w1/w2/w3/adaln).

Dumps per-module grad_A/grad_B + byte-identical inputs (noisy latent, target,
t, text features, A, B) + the velocity, for the Rust consumer to compare ours.
"""
import sys, glob
import torch
from safetensors.torch import save_file, safe_open

sys.path.insert(0, "/home/alex/ai-toolkit")
from extensions_built_in.diffusion_models.ideogram4.src.transformer import (
    Ideogram4Config, Ideogram4Transformer2DModel,
)
from extensions_built_in.diffusion_models.ideogram4.src.pipeline import (
    predict_velocity, pad_text_features,
)
from extensions_built_in.diffusion_models.ideogram4.ideogram4 import (
    _load_component_state_dict, _dequantize_fp8_state_dict,
)

BASE = "/home/alex/.serenity/models/ideogram-4-fp8"
OUT = "/home/alex/EriTrainer/trainer/parity/ideogram_lora_grad/oracle_dump.safetensors"
DEV, DT = "cuda", torch.bfloat16
RANK, ALPHA = 16, 16.0
SCALE = ALPHA / RANK
TARGETS = ["attention.qkv", "attention.o", "feed_forward.w1", "feed_forward.w2",
           "feed_forward.w3", "adaln_modulation"]

torch.manual_seed(1234)


class LoRALin(torch.nn.Module):
    """Matches eridiffusion-core forward_delta: delta = scale*(x @ A^T @ B^T),
    A/B cast to bf16 in the forward; A=[rank,in], B=[out,rank], F32 params."""
    def __init__(self, base, key):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        inf, outf = base.in_features, base.out_features
        g = torch.Generator().manual_seed(hash(key) & 0xFFFFFFFF)
        bound = 1.0 / (inf ** 0.5)
        a = (torch.rand(RANK, inf, generator=g) * 2 - 1) * bound
        b = (torch.rand(outf, RANK, generator=g) * 2 - 1) * bound  # NON-zero (non-degenerate)
        self.A = torch.nn.Parameter(a.float())
        self.B = torch.nn.Parameter(b.float())
        self.key = key

    def forward(self, x):
        base = self.base(x)
        x2 = x.reshape(-1, x.shape[-1])
        a = self.A.to(torch.bfloat16)
        b = self.B.to(torch.bfloat16)
        delta = (x2 @ a.t() @ b.t()) * SCALE
        delta = delta.reshape(*x.shape[:-1], -1)
        return base + delta.to(base.dtype)


def getmod(root, path):
    m = root
    for p in path.split("."):
        m = getattr(m, p)
    return m


def setmod(root, path, val):
    parts = path.split(".")
    m = root
    for p in parts[:-1]:
        m = getattr(m, p)
    setattr(m, parts[-1], val)


print("loading transformer ...", flush=True)
cfg = Ideogram4Config()
with torch.device("meta"):
    tf = Ideogram4Transformer2DModel(cfg)
sd = _dequantize_fp8_state_dict(
    _load_component_state_dict(BASE, "transformer", "diffusion_pytorch_model"), DT, DEV, False)
tf.load_state_dict(sd, assign=True)
del sd
hd = cfg.emb_dim // cfg.num_heads
tf.rotary_emb.register_buffer(
    "inv_freq",
    1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)),
    persistent=False)
tf.to(DEV)
tf.enable_gradient_checkpointing()

print(f"wrapping {len(tf.layers)} layers x {len(TARGETS)} LoRA modules ...", flush=True)
loras = {}
for i, layer in enumerate(tf.layers):
    for tgt in TARGETS:
        key = f"{i}.{tgt}"
        base = getmod(layer, tgt)
        w = LoRALin(base, key).to(DEV)
        setmod(layer, tgt, w)
        loras[key] = w

# ---- fixed input from a real cached eri2 sample ----
cf = sorted(glob.glob("/home/alex/EriTrainer/output/eri2/train_cache/*.safetensors"))[0]
with safe_open(cf, framework="pt") as h:
    clean = h.get_tensor("latent").to(DEV, DT)          # [1,128,gh,gw]
    text = h.get_tensor("text_embedding").to(DEV, DT)   # [1, seq, 53248]
print(f"sample: clean={tuple(clean.shape)} text={tuple(text.shape)}", flush=True)

gen = torch.Generator(device=DEV).manual_seed(777)
noise = torch.randn(clean.shape, generator=gen, device=DEV, dtype=DT)
t_scalar = 0.5
t = torch.tensor([t_scalar], device=DEV)
noisy = ((1 - t_scalar) * clean.float() + t_scalar * noise.float()).to(DT)  # flow: t=1 noise
target = (noise.float() - clean.float()).to(DT)                              # noise - clean

llm_features, text_mask = pad_text_features([text[0]], DEV, DT)
pred_v = predict_velocity(tf, noisy, t, llm_features, text_mask)             # noise-clean conv
loss = (pred_v.float() - target.float()).pow(2).mean()
print(f"oracle loss = {loss.item():.6e}", flush=True)
loss.backward()

# ---- dump grads + inputs ----
out = {
    "noisy": noisy.float().cpu().contiguous(),
    "target": target.float().cpu().contiguous(),
    "t": t.float().cpu().contiguous(),
    "text": text.float().cpu().contiguous(),
    "pred_v": pred_v.detach().float().cpu().contiguous(),
}
ng = 0
for key, w in loras.items():
    out[f"A.{key}"] = w.A.detach().float().cpu().contiguous()
    out[f"B.{key}"] = w.B.detach().float().cpu().contiguous()
    out[f"gA.{key}"] = w.A.grad.detach().float().cpu().contiguous()
    out[f"gB.{key}"] = w.B.grad.detach().float().cpu().contiguous()
    ng += 1
save_file(out, OUT)
print(f"OK dumped {ng} modules ({ng*2} grad tensors) + inputs -> {OUT}", flush=True)
# headline magnitudes for a sanity glance
import math
g2 = sum(float(out[f"gA.{k}"].pow(2).sum() + out[f"gB.{k}"].pow(2).sum()) for k in loras)
print(f"oracle ||g||2 = {math.sqrt(g2):.6e}", flush=True)
