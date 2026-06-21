"""
Generate ΔW reference values from upstream LyCORIS for parity testing.

Builds synthetic LoCon/LoHa/LoKr/Full adapters with deterministic small
weights, computes ΔW via lycoris.functional.{locon,loha,lokr}.diff_weight,
and dumps each (adapter inputs + reference ΔW) to one safetensors file
under /tmp/eri_lycoris_parity_<algo>.safetensors.

Run:
    python3 tests/parity_dump.py

The Rust parity test (tests/parity.rs) reads these files, runs the Rust
implementation against the same inputs, and asserts element-wise diff.
"""
import sys
sys.path.insert(0, "/home/alex/lycoris-upstream-lib")

import torch
from safetensors.torch import save_file

from lycoris.functional.locon import diff_weight as locon_diff
from lycoris.functional.loha import diff_weight as loha_diff
from lycoris.functional.lokr import diff_weight as lokr_diff


OUT = "/tmp"


def dump(name: str, tensors: dict):
    path = f"{OUT}/eri_lycoris_parity_{name}.safetensors"
    save_file(tensors, path)
    print(f"  wrote {path}")


def make_locon_linear():
    """LoCon Linear: in=4, out=8, rank=2, gamma=1.0 (alpha/rank)."""
    out_dim, in_dim, rank = 8, 4, 2
    down = torch.arange(rank * in_dim, dtype=torch.float32).reshape(rank, in_dim) * 0.1
    up = torch.arange(out_dim * rank, dtype=torch.float32).reshape(out_dim, rank) * 0.1
    alpha = torch.tensor(float(rank))
    gamma = alpha.item() / rank  # = 1.0
    delta = locon_diff(down, up, None, gamma=gamma)
    dump("locon_linear", {
        "down": down.contiguous(), "up": up.contiguous(),
        "alpha": alpha, "delta": delta.contiguous(),
    })


def make_locon_conv():
    """LoCon Conv2d non-Tucker: in=4, out=8, kH=3, kW=3, rank=2."""
    in_ch, out_ch, kh, kw, rank = 4, 8, 3, 3, 2
    down = torch.arange(rank * in_ch * kh * kw, dtype=torch.float32).reshape(
        rank, in_ch * kh * kw,
    ) * 0.01
    up = torch.arange(out_ch * rank, dtype=torch.float32).reshape(out_ch, rank) * 0.01
    alpha = torch.tensor(float(rank))
    gamma = alpha.item() / rank  # = 1.0
    delta_2d = locon_diff(down, up, None, gamma=gamma)
    delta_4d = delta_2d.reshape(out_ch, in_ch, kh, kw).contiguous()
    dump("locon_conv", {
        "down": down.contiguous(), "up": up.contiguous(),
        "alpha": alpha, "delta": delta_4d,
    })


def make_loha_linear():
    """LoHa Linear: 4x4, rank=2, gamma=1.0."""
    out_dim, in_dim, rank = 4, 4, 2
    # Upstream order: (w1d=down, w1u=up, w2d=down, w2u=up, t1, t2)
    # w1d/w2d are [rank, in], w1u/w2u are [out, rank].
    w1d = torch.arange(rank * in_dim, dtype=torch.float32).reshape(rank, in_dim) * 0.1 + 1.0
    w1u = torch.arange(out_dim * rank, dtype=torch.float32).reshape(out_dim, rank) * 0.1
    w2d = torch.arange(rank * in_dim, dtype=torch.float32).reshape(rank, in_dim) * 0.1 + 1.5
    w2u = torch.arange(out_dim * rank, dtype=torch.float32).reshape(out_dim, rank) * 0.1 + 0.5
    alpha = torch.tensor(float(rank))
    gamma = alpha.item() / rank  # = 1.0
    delta = loha_diff(w1d, w1u, w2d, w2u, None, None, gamma=gamma)
    # Save with the LyCORIS on-disk naming convention (hada_w1_a is w1d,
    # hada_w1_b is w1u, etc. — "_a" is down, "_b" is up).
    dump("loha_linear", {
        "hada_w1_a": w1d.contiguous(), "hada_w1_b": w1u.contiguous(),
        "hada_w2_a": w2d.contiguous(), "hada_w2_b": w2u.contiguous(),
        "alpha": alpha, "delta": delta.contiguous(),
    })


def make_lokr_linear_full():
    """LoKr Linear with full W1 and full W2 (no rank decomp). 2x2 ⊗ 2x2 -> 4x4."""
    w1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    w2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    alpha = torch.tensor(1.0)
    # gamma=1.0; internally rank = gamma = 1.0 since w1a/w2a both None,
    # so scale = gamma/rank = 1.0
    delta = lokr_diff(w1, None, None, w2, None, None, None, gamma=1.0)
    dump("lokr_linear_full", {
        "lokr_w1": w1.contiguous(), "lokr_w2": w2.contiguous(),
        "alpha": alpha, "delta": delta.contiguous(),
    })


def make_lokr_linear_lr():
    """LoKr Linear with low-rank w1=w1a@w1b, full w2."""
    w1a = torch.tensor([[1.0, 0.5], [0.0, 1.0]])
    w1b = torch.tensor([[2.0, 0.0], [1.0, 1.0]])
    w2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    alpha = torch.tensor(2.0)
    rank = w1a.shape[1]
    # Upstream computes scale = gamma / rank, so pass gamma = alpha
    delta = lokr_diff(None, w1a, w1b, w2, None, None, None, gamma=alpha.item())
    dump("lokr_linear_lr", {
        "lokr_w1_a": w1a.contiguous(), "lokr_w1_b": w1b.contiguous(),
        "lokr_w2": w2.contiguous(),
        "alpha": alpha, "delta": delta.contiguous(),
    })


def make_full_linear():
    """Full adapter: just a stored .diff."""
    diff = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                         [0.5, 0.6, 0.7, 0.8]])
    dump("full_linear", {"diff": diff.contiguous()})


def main():
    print("Dumping LyCORIS reference deltas to /tmp/eri_lycoris_parity_*.safetensors:")
    make_locon_linear()
    make_locon_conv()
    make_loha_linear()
    make_lokr_linear_full()
    make_lokr_linear_lr()
    make_full_linear()
    print("done.")


if __name__ == "__main__":
    main()
