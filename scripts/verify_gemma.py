"""Verify the Gemma-3 refactor: build, load HF weights, compare logits vs HF."""

import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from config import ModelArgs
from models.minilm import MiniLM
from models.lm_head import MiniLMForCausalLM
from training.checkpoint import load_gemma_hf_weights

HF_DIR = "external/gemma3-1b-base/gemma-3-1b-pt-ruen-v2"
ST_PATH = f"{HF_DIR}/model.safetensors"


def build_args() -> ModelArgs:
    with open(f"{HF_DIR}/config.json") as f:
        c = json.load(f)
    return ModelArgs(
        dim=c["hidden_size"],
        n_layers=c["num_hidden_layers"],
        n_heads=c["num_attention_heads"],
        n_kv_heads=c["num_key_value_heads"],
        head_dim=c["head_dim"],
        vocab_size=c["vocab_size"],
        intermediate_size=c["intermediate_size"],
        norm_eps=c["rms_norm_eps"],
        max_seq_len=2048,
        dropout=0.0,
        query_pre_attn_scalar=c["query_pre_attn_scalar"],
        rope_theta=c["rope_theta"],
        rope_local_base_freq=c["rope_local_base_freq"],
        sliding_window=c["sliding_window"],
        sliding_window_pattern=c["sliding_window_pattern"],
        use_liger=False,
        ce_chunk_size=0,
    )


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} (CUDA exercises the FlexAttention local-layer path)")
    args = build_args()
    print(
        f"ModelArgs OK: dim={args.dim} layers={args.n_layers} vocab={args.vocab_size}"
    )

    model = MiniLMForCausalLM(MiniLM(args), args).to(torch.float32).eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Built MiniLMForCausalLM: params={n_params:,}")

    load_gemma_hf_weights(model, ST_PATH)
    assert (
        model.output.weight.data_ptr() == model.model.tok_embeddings.weight.data_ptr()
    )
    print("Weights loaded; output tied to embeddings.")

    seq_len = 600
    ids = torch.randint(0, args.vocab_size, (1, seq_len), device=device)

    with torch.no_grad():
        h = model.model(ids)  # [1, T, dim]
        ours = torch.nn.functional.linear(h, model.output.weight).float()

    print(f"Our logits: {tuple(ours.shape)}")

    with torch.no_grad():
        model.model.use_cache = True
        prefill_last = model(ids[:, :-1], start_pos=0)[0]
        decode_last = model(ids[:, -1:], start_pos=seq_len - 1)[0]
        model.model.use_cache = False
        model.model.reset_kv_caches()
    cache_diff = max(
        (ours[:, -2] - prefill_last).abs().max().item(),
        (ours[:, -1] - decode_last).abs().max().item(),
    )
    print(f"KV-cache vs full forward: max|diff|={cache_diff:.6f}")
    assert cache_diff < 1e-2, "cache path diverged from full forward"

    try:
        from transformers import Gemma3ForCausalLM
    except Exception as e:  # noqa: BLE001
        print(f"[skip HF compare] transformers Gemma3 unavailable: {e}")
        return

    hf = (
        Gemma3ForCausalLM.from_pretrained(
            HF_DIR, torch_dtype=torch.float32, attn_implementation="eager"
        )
        .eval()
        .to(device)
    )
    with torch.no_grad():
        ref = hf(ids).logits.float()

    diff = (ours - ref).abs()
    cos = torch.nn.functional.cosine_similarity(
        ours.reshape(-1), ref.reshape(-1), dim=0
    ).item()
    print(f"HF logits: {tuple(ref.shape)}")
    print(f"max|diff|={diff.max().item():.6f}  mean|diff|={diff.mean().item():.6f}")
    print(f"cosine_sim={cos:.8f}")

    agree = (ours.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(f"argmax agreement={agree*100:.2f}%")

    ok = diff.max().item() < 1e-2 and agree > 0.999
    print("RESULT:", "PASS" if ok else "CHECK")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
