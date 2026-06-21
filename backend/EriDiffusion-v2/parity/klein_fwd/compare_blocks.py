#!/usr/bin/env python3
"""Per-block forward-activation cos AND backward-grad cos, ours vs diffusers.

Disambiguates:
  forward cos HIGH (~1.0) + backward cos LOW  => backward-composition bug
  forward cos and backward cos DRIFT together => forward-precision drift
"""
import torch
from safetensors.torch import load_file

ours_a_raw = load_file("our_block_acts.safetensors")
ref_a  = load_file("block_acts_ref.safetensors")
ours_g = load_file("our_block_grads.safetensors")
ref_g  = load_file("block_grads_ref.safetensors")

# our forward acts now carry a "#N" occurrence suffix (#0 = initial forward).
# Collapse to base label, preferring #0.
ours_a = {}
for k, v in ours_a_raw.items():
    base = k.split('#')[0]
    occ = int(k.split('#')[1]) if '#' in k else 0
    if base not in ours_a or occ == 0:
        ours_a[base] = v

def cos(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    n = min(a.numel(), b.numel())
    if a.numel() != b.numel():
        print(f"    [warn] size {a.numel()} vs {b.numel()}, truncating to {n}")
    a, b = a[:n], b[:n]
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item(), a.norm().item(), b.norm().item()

labels = ["sb_normed", "sb_out", "img_only"] + [f"dbl{i}_img" for i in range(8)]
print(f"{'block':<14}{'fwd cos':>10}{'|f|ours':>11}{'|f|ref':>11}   "
      f"{'bwd cos':>10}{'|g|ours':>11}{'|g|ref':>11}")
for lb in labels:
    fc = ga = gb = bc = ba = bb = float('nan')
    if lb in ours_a and lb in ref_a:
        fc, fa, fb = cos(ours_a[lb], ref_a[lb])
    else:
        fa = fb = float('nan')
    if lb in ours_g and lb in ref_g:
        bc, ba, bb = cos(ours_g[lb], ref_g[lb])
    else:
        ba = bb = float('nan')
    print(f"{lb:<14}{fc:>10.4f}{fa:>11.3e}{fb:>11.3e}   {bc:>10.4f}{ba:>11.3e}{bb:>11.3e}")
print("\nforward cos ~1.0 but backward cos low  => BACKWARD bug (proof)")
print("forward cos drifts with backward cos    => forward-precision drift")
