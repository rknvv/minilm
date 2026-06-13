"""CUDA parity test for models/fused_loss.py (vendored large-chunk liger FLCE)."""

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from liger_kernel.transformers.functional import (  # noqa: E402
    liger_fused_linear_cross_entropy as liger_flce,
)

from models.fused_loss import flce_large_chunk  # noqa: E402

BT, H, V = 8192, 1152, 153856
CHUNK = 4096
IGNORE = -100


def run_case(name, targets):
    torch.manual_seed(0)
    h0 = torch.randn(BT, H, device="cuda", dtype=torch.bfloat16) * 0.5
    w0 = torch.randn(V, H, device="cuda", dtype=torch.float32) * 0.02

    def grads(fn):
        h = h0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = fn(h, w)
        loss.backward()
        return loss.float(), h.grad.float(), w.grad.float()

    def ref_fn(h, w):
        logits = (h @ w.t()).float()
        return torch.nn.functional.cross_entropy(logits, targets, ignore_index=IGNORE)

    def stock_fn(h, w):
        return liger_flce(h, w, targets, ignore_index=IGNORE)

    def ours_fn(h, w):
        return flce_large_chunk(h, w, targets, IGNORE, CHUNK)

    ref = grads(ref_fn)
    stock = grads(stock_fn)
    ours = grads(ours_fn)

    def rel(a, b):
        return ((a - b).abs().max() / b.abs().max().clamp(min=1e-12)).item()

    for label, idx in (("loss", 0), ("dh", 1), ("dW", 2)):
        d_stock = rel(ours[idx], stock[idx])
        d_ref = rel(ours[idx], ref[idx])
        print(
            f"[{name}] {label}: ours vs stock rel={d_stock:.3e}, "
            f"ours vs full-CE rel={d_ref:.3e}"
        )
        assert d_stock < 2e-2, f"{name}/{label}: vendored vs stock liger diverged"
        assert d_ref < 5e-2, f"{name}/{label}: vendored vs full-CE diverged"
    print(
        f"[{name}] loss values: ref={ref[0].item():.6f} "
        f"stock={stock[0].item():.6f} ours={ours[0].item():.6f}"
    )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        eval_loss = flce_large_chunk(h0, w0, targets, IGNORE, CHUNK).float()
    assert rel(eval_loss, ours[0]) < 1e-3, f"{name}: eval-path loss mismatch"
    print(f"[{name}] eval (no-grad) path OK\n")


def main():
    assert torch.cuda.is_available(), "CUDA required"
    torch.manual_seed(0)

    t_plain = torch.randint(0, V, (BT,), device="cuda")
    run_case("pretrain (no ignore)", t_plain)

    t_ignored = t_plain.clone()
    t_ignored[torch.rand(BT, device="cuda") < 0.3] = IGNORE
    run_case("sft (30% ignore_index)", t_ignored)

    print("LOSS PARITY: ALL CASES PASSED")


if __name__ == "__main__":
    main()
