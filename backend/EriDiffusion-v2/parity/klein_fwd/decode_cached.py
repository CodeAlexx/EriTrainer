#!/usr/bin/env python3
"""Decode a cached training latent through the klein VAE the diffusers way.
If it reconstructs Alina -> prepare_klein normalization is correct.
If garbage -> our cached latents are off-distribution (the training-data bug).
"""
import glob, torch
from safetensors.torch import load_file
from diffusers import AutoencoderKLFlux2
from diffusers.image_processor import VaeImageProcessor

VAE_DIR = ("/home/alex/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-9B/"
           "snapshots/32773329fbe7e81a90ef971740e8ba4b0364ecf3/vae")
CACHE = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/alina_klein9b"
dev = "cuda"; dt = torch.float32

f = sorted(glob.glob(f"{CACHE}/*.safetensors"))[0]
lat = load_file(f, device="cpu")["latent"].to(dev, dt)   # [1,128,40,28]
print("cached latent", tuple(lat.shape), "mean", float(lat.mean()), "std", float(lat.std()))

vae = AutoencoderKLFlux2.from_pretrained(VAE_DIR, torch_dtype=dt).to(dev).eval()

def unpatchify(latents):
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c // 4, h * 2, w * 2)

with torch.no_grad():
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(dev, dt)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(dev, dt)
    print("bn_mean[:4]", bn_mean.flatten()[:4].tolist(), "bn_std[:4]", bn_std.flatten()[:4].tolist())
    den = lat * bn_std + bn_mean          # inverse bn
    den = unpatchify(den)                  # [1,32,80,56]
    print("unpatchified", tuple(den.shape))
    img = vae.decode(den, return_dict=False)[0]
    proc = VaeImageProcessor(vae_scale_factor=8)
    pil = proc.postprocess(img, output_type="pil")[0]
    out = "/home/alex/EriDiffusion/inference-flame/output/klein_cached_latent_decoded.png"
    pil.save(out)
    print("saved", out)
