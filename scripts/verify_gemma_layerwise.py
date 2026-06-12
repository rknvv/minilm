"""Strict layer-by-layer comparison of our MiniLM (Gemma-3) vs HF Gemma3ForCausalLM.

Goal: determine whether the small fp32 logit difference is pure floating-point
accumulation (gradual, machine-epsilon per identical op) or a real bug (a jump
at one layer / module).
"""

import json

import torch

from models.lm_head import MiniLMForCausalLM

D = "external/gemma3-1b-base/gemma-3-1b-pt-ruen"
DT = torch.float32


def capture(module_list, store):
    handles = []
    for i, m in enumerate(module_list):

        def hook(mod, inp, out, idx=i):
            store[idx] = (out[0] if isinstance(out, tuple) else out).detach().float()

        handles.append(m.register_forward_hook(hook))
    return handles


def run(seq_len: int, attn_impl: str):
    from transformers import Gemma3ForCausalLM

    torch.manual_seed(0)
    ids = torch.randint(0, 183927, (1, seq_len))

    ours = MiniLMForCausalLM.from_pretrained(D, max_seq_len=2048).to(DT).eval()
    hf = Gemma3ForCausalLM.from_pretrained(
        D, torch_dtype=DT, attn_implementation=attn_impl
    ).eval()

    o_hidden: dict = {}
    h_hidden: dict = {}
    ho = capture(ours.model.layers, o_hidden)
    hh = capture(hf.model.layers, h_hidden)

    with torch.no_grad():
        o_h = ours.model(ids)
        o_logits = torch.nn.functional.linear(o_h, ours.output.weight).float()
        ref = hf(ids).logits.float()

    for x in ho + hh:
        x.remove()

    print(f"\n=== seq_len={seq_len}, HF attn_implementation={attn_impl} ===")
    worst = 0.0
    for i in range(len(o_hidden)):
        d = (o_hidden[i] - h_hidden[i]).abs().max().item()
        worst = max(worst, d)
        tag = " (local)" if (i + 1) % 6 != 0 else " (global)"
        if i < 3 or i >= len(o_hidden) - 2 or d > 1e-3:
            print(f"  layer {i:2d}{tag}: max|dh|={d:.3e}")
    print(f"  >>> worst per-layer hidden diff: {worst:.3e}")

    ld = (o_logits - ref).abs()
    agree = (o_logits.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(
        f"  logits: max|diff|={ld.max():.3e}  mean|diff|={ld.mean():.3e}  argmax={agree*100:.2f}%"
    )
    return worst


def main():
    cfg = json.load(open(f"{D}/config.json"))
    print(
        "eps fp32 ~1.2e-7; logits are ~tens in magnitude, "
        f"hidden_size={cfg['hidden_size']}, layers={cfg['num_hidden_layers']}"
    )
    run(16, "sdpa")
    run(16, "eager")
    run(600, "sdpa")
    run(600, "eager")


if __name__ == "__main__":
    main()
