"""Export a MiniLM (Gemma-3 architecture) checkpoint to HF Gemma3ForCausalLM."""
import os
from typing import Optional

import torch
import yaml

from config import ModelArgs
from training.checkpoint import extract_model_state_dict, normalize_state_dict_keys


def _load_model_args(ckpt_path: str, config_path: Optional[str]) -> ModelArgs:
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(ckpt_path)), "all_config.yaml"
        )
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Model args not found: pass --config_path or keep all_config.yaml next "
            f"to the checkpoint (looked at {config_path})."
        )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return ModelArgs(**cfg["model"])


def build_hf_state_dict(state_dict: dict, args: ModelArgs) -> dict:
    """Inverse of training.checkpoint._convert_gemma_hf_state_dict.

    Input keys are MiniLMForCausalLM-level: "model.<inner>" and "output.weight"
    (tied, skipped — HF re-ties via tie_word_embeddings).
    """
    out: dict = {}
    out["model.embed_tokens.weight"] = state_dict["model.tok_embeddings.weight"]
    out["model.norm.weight"] = state_dict["model.norm.weight"]

    for i in range(args.n_layers):
        src = f"model.layers.{i}"
        dst = f"model.layers.{i}"

        out[f"{dst}.self_attn.q_proj.weight"] = state_dict[f"{src}.attention.wq.weight"]
        out[f"{dst}.self_attn.k_proj.weight"] = state_dict[f"{src}.attention.wk.weight"]
        out[f"{dst}.self_attn.v_proj.weight"] = state_dict[f"{src}.attention.wv.weight"]
        out[f"{dst}.self_attn.o_proj.weight"] = state_dict[f"{src}.attention.wo.weight"]
        out[f"{dst}.self_attn.q_norm.weight"] = state_dict[
            f"{src}.attention.q_norm.weight"
        ]
        out[f"{dst}.self_attn.k_norm.weight"] = state_dict[
            f"{src}.attention.k_norm.weight"
        ]
        out[f"{dst}.mlp.gate_proj.weight"] = state_dict[
            f"{src}.feed_forward.gate_proj.weight"
        ]
        out[f"{dst}.mlp.up_proj.weight"] = state_dict[
            f"{src}.feed_forward.up_proj.weight"
        ]
        out[f"{dst}.mlp.down_proj.weight"] = state_dict[
            f"{src}.feed_forward.down_proj.weight"
        ]
        out[f"{dst}.input_layernorm.weight"] = state_dict[f"{src}.input_layernorm.weight"]
        out[f"{dst}.post_attention_layernorm.weight"] = state_dict[
            f"{src}.post_attention_layernorm.weight"
        ]
        out[f"{dst}.pre_feedforward_layernorm.weight"] = state_dict[
            f"{src}.pre_feedforward_layernorm.weight"
        ]
        out[f"{dst}.post_feedforward_layernorm.weight"] = state_dict[
            f"{src}.post_feedforward_layernorm.weight"
        ]

    return out


def build_gemma3_config(args: ModelArgs):
    from transformers import Gemma3TextConfig

    layer_types = [
        "full_attention"
        if (i + 1) % args.sliding_window_pattern == 0
        else "sliding_attention"
        for i in range(args.n_layers)
    ]
    return Gemma3TextConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.dim,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        num_key_value_heads=(
            args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        ),
        head_dim=args.head_dim,
        max_position_embeddings=args.max_seq_len,
        rms_norm_eps=args.norm_eps,
        rope_theta=args.rope_theta,
        rope_local_base_freq=args.rope_local_base_freq,
        query_pre_attn_scalar=args.query_pre_attn_scalar,
        sliding_window=args.sliding_window,
        layer_types=layer_types,
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        tie_word_embeddings=True,
    )


def convert(ckpt_path: str, out_dir: str, config_path: Optional[str] = None) -> None:
    from transformers import Gemma3ForCausalLM

    args = _load_model_args(ckpt_path, config_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = normalize_state_dict_keys(extract_model_state_dict(checkpoint))

    hf_config = build_gemma3_config(args)
    hf_state = build_hf_state_dict(state_dict, args)

    model = Gemma3ForCausalLM(hf_config)
    missing, unexpected = model.load_state_dict(hf_state, strict=False)

    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading HF model: {unexpected}")
    # lm_head.weight is tied to embeddings; anything else missing is a bug.
    real_missing = [k for k in missing if k != "lm_head.weight"]
    if real_missing:
        raise RuntimeError(f"Missing keys when loading HF model: {real_missing}")
    model.tie_weights()

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    print(f"Saved HF Gemma3ForCausalLM to {out_dir}")


if __name__ == "__main__":
    try:
        import fire

        fire.Fire(convert)
    except ImportError:
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("--ckpt_path", required=True)
        p.add_argument("--out_dir", required=True)
        p.add_argument("--config_path", default=None)
        a = p.parse_args()
        convert(a.ckpt_path, a.out_dir, a.config_path)
