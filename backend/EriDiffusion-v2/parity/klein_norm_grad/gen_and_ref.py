#!/usr/bin/env python3
"""RMSNorm + LayerNorm backward parity vs PyTorch.

klein applies head-RMSNorm to q & k in every block (norm::rms_norm over D=128,
with scale), and LayerNorm (no affine) in the modulation. Tests both backwards
by injecting a fixed upstream grad G via loss=sum(out*G) so dOut=G exactly.
"""
import torch
from safetensors.torch import save_file
torch.manual_seed(3)
dev = "cuda"
EPS = 1e-6

# ---- RMSNorm (QK norm): [N, D=128], weighted ----
N, D = 1024, 128
x = torch.randn(N, D, device=dev)
w = (torch.randn(D, device=dev) * 0.1 + 1.0)   # scale ~1
Gr = torch.randn(N, D, device=dev)
xf = x.clone().requires_grad_(True); wf = w.clone().requires_grad_(True)
rms = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * wf
(rms.float() * Gr.float()).sum().backward()

# ---- LayerNorm (modulation), no affine: [M, DIM=3072] ----
M, DIM = 256, 3072
y = torch.randn(M, DIM, device=dev)
Gl = torch.randn(M, DIM, device=dev)
yf = y.clone().requires_grad_(True)
ln = torch.nn.functional.layer_norm(yf, (DIM,), None, None, EPS)
(ln.float() * Gl.float()).sum().backward()

# ---- SwiGLU split-lastdim: out = silu(X[:half]) * X[half:]  ([P, 2*H]) ----
P, H = 512, 2048
sx = torch.randn(P, 2 * H, device=dev)
Gs = torch.randn(P, H, device=dev)
sxf = sx.clone().requires_grad_(True)
g_, u_ = sxf[..., :H], sxf[..., H:]
sout = torch.nn.functional.silu(g_) * u_
(sout.float() * Gs.float()).sum().backward()
print(f"[swiglu] |out|={sout.norm():.4e} |dx|={sxf.grad.norm():.4e}")

save_file({
    "rms_x": x.cpu(), "rms_w": w.cpu(), "rms_G": Gr.cpu(),
    "rms_out": rms.detach().float().cpu(), "rms_dx": xf.grad.float().cpu(), "rms_dw": wf.grad.float().cpu(),
    "ln_x": y.cpu(), "ln_G": Gl.cpu(),
    "ln_out": ln.detach().float().cpu(), "ln_dx": yf.grad.float().cpu(),
    "sw_x": sx.cpu(), "sw_G": Gs.cpu(),
    "sw_out": sout.detach().float().cpu(), "sw_dx": sxf.grad.float().cpu(),
}, "fixture.safetensors")
print(f"[rms] |out|={rms.norm():.4e} |dx|={xf.grad.norm():.4e} |dw|={wf.grad.norm():.4e}")
print(f"[ln ] |out|={ln.norm():.4e} |dx|={yf.grad.norm():.4e}")
print("wrote fixture.safetensors")
