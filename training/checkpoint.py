import logging
import os
from collections.abc import Mapping

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
            raise TypeError("Model state_dict must map str keys to torch.Tensor values.")
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
        raise FileNotFoundError(f"Pretrained checkpoint not found at: {checkpoint_path}")

    logger.info(f"Loading pretrained weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = normalize_state_dict_keys(extract_model_state_dict(checkpoint))
    result = model.load_state_dict(state_dict, strict=False)

    if result.missing_keys:
        logger.warning(f"Missing keys while loading pretrained weights: {result.missing_keys}")
    if result.unexpected_keys:
        logger.warning(f"Unexpected keys while loading pretrained weights: {result.unexpected_keys}")
