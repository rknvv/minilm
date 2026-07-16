import logging
import os
from collections.abc import Mapping
from safetensors.torch import load_file

import torch

logger = logging.getLogger(__name__)

_STATE_DICT_WRAPPER_PREFIXES = ("module.", "_orig_mod.")


def strip_state_dict_wrappers(key: str) -> str:
    normalized = key
    while True:
        updated = normalized
        for prefix in _STATE_DICT_WRAPPER_PREFIXES:
            if updated.startswith(prefix):
                updated = updated[len(prefix) :]
                break
        if updated == normalized:
            return normalized
        normalized = updated


def extract_model_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dict or a raw state_dict.")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        state_dict = checkpoint.get("state_dict")

    if not isinstance(state_dict, dict):
        if checkpoint and all(
            isinstance(k, str) and isinstance(v, torch.Tensor)
            for k, v in checkpoint.items()
        ):
            state_dict = checkpoint
        else:
            raise KeyError(
                "Checkpoint must contain 'model_state_dict', 'state_dict', or be a raw state_dict."
            )

    return validate_state_dict(state_dict)


def validate_state_dict(
    state_dict: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    validated: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError(
                "Model state_dict must map str keys to torch.Tensor values."
            )
        validated[key] = value
    return validated


def normalize_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
    *,
    add_module_prefix: bool = False,
) -> dict[str, torch.Tensor]:
    normalized_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized_key = strip_state_dict_wrappers(key)
        if add_module_prefix:
            normalized_key = f"module.{normalized_key}"
        normalized_state[normalized_key] = value
    return normalized_state


def load_pretrained_weights(model: torch.nn.Module, checkpoint_path: str) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at: {checkpoint_path}"
        )

    if checkpoint_path.endswith(".safetensors"):
        load_gemma_hf_weights(model, checkpoint_path)
        return

    logger.info(f"Loading pretrained weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = normalize_state_dict_keys(extract_model_state_dict(checkpoint))
    result = model.load_state_dict(state_dict, strict=False)

    if result.missing_keys:
        logger.warning(
            f"Missing keys while loading pretrained weights: {result.missing_keys}"
        )
    if result.unexpected_keys:
        logger.warning(
            f"Unexpected keys while loading pretrained weights: {result.unexpected_keys}"
        )


def _convert_gemma_hf_state_dict(
    hf: Mapping[str, torch.Tensor], n_layers: int
) -> dict[str, torch.Tensor]:
    """Map HF Gemma3ForCausalLM param names onto the inner MiniLM module names."""
    out: dict[str, torch.Tensor] = {}
    out["tok_embeddings.weight"] = hf["model.embed_tokens.weight"]
    out["norm.weight"] = hf["model.norm.weight"]

    for i in range(n_layers):
        src = f"model.layers.{i}"
        dst = f"layers.{i}"
        out[f"{dst}.attention.wq.weight"] = hf[f"{src}.self_attn.q_proj.weight"]
        out[f"{dst}.attention.wk.weight"] = hf[f"{src}.self_attn.k_proj.weight"]
        out[f"{dst}.attention.wv.weight"] = hf[f"{src}.self_attn.v_proj.weight"]
        out[f"{dst}.attention.wo.weight"] = hf[f"{src}.self_attn.o_proj.weight"]
        out[f"{dst}.attention.q_norm.weight"] = hf[f"{src}.self_attn.q_norm.weight"]
        out[f"{dst}.attention.k_norm.weight"] = hf[f"{src}.self_attn.k_norm.weight"]
        out[f"{dst}.feed_forward.gate_proj.weight"] = hf[f"{src}.mlp.gate_proj.weight"]
        out[f"{dst}.feed_forward.up_proj.weight"] = hf[f"{src}.mlp.up_proj.weight"]
        out[f"{dst}.feed_forward.down_proj.weight"] = hf[f"{src}.mlp.down_proj.weight"]
        out[f"{dst}.input_layernorm.weight"] = hf[f"{src}.input_layernorm.weight"]
        out[f"{dst}.post_attention_layernorm.weight"] = hf[
            f"{src}.post_attention_layernorm.weight"
        ]
        out[f"{dst}.pre_feedforward_layernorm.weight"] = hf[
            f"{src}.pre_feedforward_layernorm.weight"
        ]
        out[f"{dst}.post_feedforward_layernorm.weight"] = hf[
            f"{src}.post_feedforward_layernorm.weight"
        ]
    return out


def load_gemma_hf_weights(model: torch.nn.Module, safetensors_path: str) -> None:
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Safetensors file not found: {safetensors_path}")

    logger.info(f"Loading Gemma HF weights from {safetensors_path}...")
    hf = load_file(safetensors_path)

    inner = model.model  # type: ignore[attr-defined]
    state_dict = _convert_gemma_hf_state_dict(hf, model.args.n_layers)  # type: ignore[attr-defined]

    result = inner.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys when loading Gemma weights: {result.unexpected_keys}"
        )
    if result.missing_keys:
        logger.warning(
            f"Missing keys while loading Gemma weights: {result.missing_keys}"
        )

    model.output.weight = inner.tok_embeddings.weight  # type: ignore[attr-defined]
    logger.info("Gemma HF weights loaded and output head re-tied to embeddings.")
