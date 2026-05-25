import os
from typing import Optional

import torch
import yaml

from config import ModelArgs
from training.checkpoint import extract_model_state_dict, normalize_state_dict_keys

def _ffn_hidden_dim(args: ModelArgs) -> int:
    hidden_dim = 4 * args.dim
    hidden_dim = int(2 * hidden_dim / 3)
    if args.ffn_dim_multiplier is not None:
        hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
    return args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

def _load_model_args(ckpt_path: str, config_path: Optional[str]) -> ModelArgs:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "all_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Model args not found: pass --config_path or keep all_config.yaml next "
            f"to the checkpoint (looked at {config_path})."
        )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return ModelArgs(**cfg["model"])

def build_hf_state_dict(state_dict: dict, args: ModelArgs) -> dict:

    out: dict = {}
    out["model.embed_tokens.weight"] = state_dict["model.tok_embeddings.weight"]
    out["model.norm.weight"] = state_dict["model.norm.weights"]

    out["lm_head.weight"] = state_dict.get(
        "output.weight", state_dict["model.tok_embeddings.weight"]
    )

    for i in range(args.n_layers):
        src = f"model.layers.{i}"
        dst = f"model.layers.{i}"

        out[f"{dst}.self_attn.q_proj.weight"] = state_dict[f"{src}.attention.wq.weight"]
        out[f"{dst}.self_attn.k_proj.weight"] = state_dict[f"{src}.attention.wk.weight"]
        out[f"{dst}.self_attn.v_proj.weight"] = state_dict[f"{src}.attention.wv.weight"]
        out[f"{dst}.self_attn.o_proj.weight"] = state_dict[f"{src}.attention.wo.weight"]
        out[f"{dst}.mlp.gate_proj.weight"] = state_dict[f"{src}.feed_forward.w1.weight"]
        out[f"{dst}.mlp.up_proj.weight"] = state_dict[f"{src}.feed_forward.w3.weight"]
        out[f"{dst}.mlp.down_proj.weight"] = state_dict[f"{src}.feed_forward.w2.weight"]
        out[f"{dst}.input_layernorm.weight"] = state_dict[f"{src}.attn_norm.weights"]
        out[f"{dst}.post_attention_layernorm.weight"] = state_dict[f"{src}.ffn_norm.weights"]

    return out

def build_llama_config(args: ModelArgs):
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.dim,
        intermediate_size=_ffn_hidden_dim(args),
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        num_key_value_heads=args.n_kv_heads if args.n_kv_heads is not None else args.n_heads,
        max_position_embeddings=args.max_seq_len,
        rms_norm_eps=args.norm_eps,
        rope_theta=10000.0,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
    )

def convert(ckpt_path: str, out_dir: str, config_path: Optional[str] = None) -> None:
    from transformers import LlamaForCausalLM

    args = _load_model_args(ckpt_path, config_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = normalize_state_dict_keys(extract_model_state_dict(checkpoint))

    hf_config = build_llama_config(args)
    hf_state = build_hf_state_dict(state_dict, args)

    model = LlamaForCausalLM(hf_config)
    missing, unexpected = model.load_state_dict(hf_state, strict=False)

    unexpected = [k for k in unexpected]
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading HF model: {unexpected}")

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    print(f"Saved HF LlamaForCausalLM to {out_dir}")

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
