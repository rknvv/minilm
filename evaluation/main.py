import logging
import math
import os
from typing import Optional, Tuple

import torch
import yaml
from torch.utils.data import DataLoader

from config import ModelArgs, TrainConfig
from models.minilm import MiniLM
from models.lm_head import MiniLMForCausalLM
from dataio.dataset import MemmapDataset
from dataio.loaders import resolve_token_dtype
from training.checkpoint import extract_model_state_dict, normalize_state_dict_keys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluation")

def _pick_device(device: Optional[str]) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _sibling_config(ckpt_path: str, config_path: Optional[str]) -> str:
    if config_path is not None:
        return config_path
    candidate = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "all_config.yaml")
    if not os.path.exists(candidate):
        raise FileNotFoundError(
            f"Pass --config_path or keep all_config.yaml next to the checkpoint "
            f"(looked at {candidate})."
        )
    return candidate

def _train_config_from_snapshot(section: dict) -> TrainConfig:
    known = {k: v for k, v in section.items() if k in TrainConfig.model_fields}
    dropped = sorted(set(section) - set(known))
    if dropped:
        logger.warning("Ignoring retired config keys from snapshot: %s", dropped)
    return TrainConfig(**known)

def load_model(
    ckpt_path: str, config_path: Optional[str] = None, device: Optional[str] = None
) -> Tuple[MiniLMForCausalLM, ModelArgs, TrainConfig]:

    cfg = yaml.safe_load(open(_sibling_config(ckpt_path, config_path)))
    model_args = ModelArgs(**cfg["model"])
    train_cfg = _train_config_from_snapshot(cfg["train"])

    model = MiniLMForCausalLM(MiniLM(model_args), model_args)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = normalize_state_dict_keys(extract_model_state_dict(checkpoint))
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        logger.warning("Missing keys on load: %s", result.missing_keys)
    if result.unexpected_keys:
        logger.warning("Unexpected keys on load: %s", result.unexpected_keys)

    dev = _pick_device(device)
    model.to(dev).eval()
    return model, model_args, train_cfg

@torch.no_grad()
def perplexity(
    ckpt_path: str,
    val_bin: Optional[str] = None,
    config_path: Optional[str] = None,
    batch_size: int = 8,
    max_batches: Optional[int] = 100,
    device: Optional[str] = None,
) -> float:

    model, model_args, train_cfg = load_model(ckpt_path, config_path, device)
    dev = _pick_device(device)

    if val_bin is None:
        val_bin = os.path.join(train_cfg.dataset_dir, "val.bin")
    token_dtype = resolve_token_dtype(train_cfg, model_args)
    dataset = MemmapDataset(
        val_bin,
        model_args.max_seq_len,
        memmap_dtype=token_dtype,
        vocab_size=model_args.vocab_size,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss, n_batches = 0.0, 0
    for batch in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        inputs = batch["input_ids"].to(dev)
        targets = batch["labels"].to(dev)
        _, loss = model(inputs, targets=targets, ignore_index=train_cfg.ignore_index)
        total_loss += float(loss)
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    ppl = math.exp(avg_loss) if avg_loss < 30 else float("inf")
    logger.info(
        "Eval on %s | batches=%d | loss=%.4f | perplexity=%.3f",
        val_bin, n_batches, avg_loss, ppl,
    )
    return ppl

if __name__ == "__main__":
    try:
        import fire

        fire.Fire({"perplexity": perplexity})
    except ImportError:
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("--ckpt_path", required=True)
        p.add_argument("--val_bin", default=None)
        p.add_argument("--config_path", default=None)
        p.add_argument("--batch_size", type=int, default=8)
        p.add_argument("--max_batches", type=int, default=100)
        p.add_argument("--device", default=None)
        a = p.parse_args()
        perplexity(a.ckpt_path, a.val_bin, a.config_path, a.batch_size, a.max_batches, a.device)
