"""Gate B oracle SMOKE: prove ai-toolkit's ideogram4 transformer loads (bf16)
and predict_velocity runs on a fixed input, before building the grad-dump gen."""
import sys, torch
sys.path.insert(0, "/home/alex/ai-toolkit")

from extensions_built_in.diffusion_models.ideogram4.src.transformer import (
    Ideogram4Config,
    Ideogram4Transformer2DModel,
)
from extensions_built_in.diffusion_models.ideogram4.src.pipeline import (
    predict_velocity,
    pad_text_features,
)
from extensions_built_in.diffusion_models.ideogram4.ideogram4 import (
    _load_component_state_dict,
    _dequantize_fp8_state_dict,
)

BASE = "/home/alex/.serenity/models/ideogram-4-fp8"
DEV = "cuda"
DT = torch.bfloat16

print("building transformer (meta) ...", flush=True)
cfg = Ideogram4Config()
with torch.device("meta"):
    tf = Ideogram4Transformer2DModel(cfg)

print("loading + dequantizing fp8 transformer weights ...", flush=True)
sd = _load_component_state_dict(BASE, "transformer", "diffusion_pytorch_model")
sd = _dequantize_fp8_state_dict(sd, DT, DEV, False)
tf.load_state_dict(sd, assign=True)
del sd

head_dim = cfg.emb_dim // cfg.num_heads
inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
tf.rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)
tf.to(DEV)
print(f"transformer loaded: emb_dim={cfg.emb_dim} heads={cfg.num_heads} head_dim={head_dim}", flush=True)

# fixed dummy input: 512px latent (128ch, 32x32), one caption of len 177, dim 53248
latent = torch.randn(1, 128, 32, 32, device=DEV, dtype=DT)
t = torch.tensor([0.5], device=DEV)
text_feat = torch.randn(177, 53248, device=DEV, dtype=DT)
llm_features, text_mask = pad_text_features([text_feat], DEV, DT)
print(f"inputs: latent={tuple(latent.shape)} t={t.item()} llm={tuple(llm_features.shape)} mask={tuple(text_mask.shape)}", flush=True)

with torch.no_grad():
    pred = predict_velocity(tf, latent, t, llm_features, text_mask)
print(f"OK predict_velocity -> {tuple(pred.shape)} {pred.dtype}  mean={pred.float().mean().item():.4e}", flush=True)
