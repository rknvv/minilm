import json
import logging
import os
from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from config import ModelArgs, TrainConfig
from dataio.dataset import MemmapDataset

logger = logging.getLogger(__name__)

TOKEN_DTYPES = {"uint16": np.uint16, "uint32": np.uint32}


class SkippableDistributedSampler(DistributedSampler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.skip_samples = 0

    def __iter__(self):
        indices = list(super().__iter__())
        if self.skip_samples > 0:
            indices = indices[self.skip_samples :]
        return iter(indices)


def resolve_token_dtype(train_cfg: TrainConfig, model_args: ModelArgs) -> np.dtype:
    meta_path = os.path.join(train_cfg.dataset_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        name = meta.get("token_dtype")
        if name in TOKEN_DTYPES:
            if train_cfg.token_dtype not in ("auto", name):
                logger.warning(
                    f"train.token_dtype={train_cfg.token_dtype} contradicts "
                    f"{meta_path} ({name}); using meta.json."
                )
            meta_vocab = meta.get("vocab_size")
            if meta_vocab is not None and meta_vocab != model_args.vocab_size:
                logger.warning(
                    f"meta.json vocab_size={meta_vocab} != model vocab_size="
                    f"{model_args.vocab_size}; check tokenizer/model pairing."
                )
            return np.dtype(TOKEN_DTYPES[name])

    if train_cfg.token_dtype != "auto":
        return np.dtype(TOKEN_DTYPES[train_cfg.token_dtype])

    if model_args.vocab_size - 1 <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def build_datasets(
    train_cfg: TrainConfig, model_args: ModelArgs
) -> Tuple[Dataset, Dataset]:
    token_dtype = resolve_token_dtype(train_cfg, model_args)
    logger.info(f"Memmap token dtype: {token_dtype.name}")
    train_dataset = MemmapDataset(
        os.path.join(train_cfg.dataset_dir, "train.bin"),
        sequence_length=model_args.max_seq_len,
        memmap_dtype=token_dtype,
        vocab_size=model_args.vocab_size,
    )
    eval_dataset = MemmapDataset(
        os.path.join(train_cfg.dataset_dir, "val.bin"),
        sequence_length=model_args.max_seq_len,
        memmap_dtype=token_dtype,
        vocab_size=model_args.vocab_size,
    )
    return train_dataset, eval_dataset


def build_dataloaders(
    train_cfg: TrainConfig,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    world_size: int,
    rank: int,
    device: str,
) -> Tuple[DataLoader, DataLoader]:
    pin_memory = device.startswith("cuda")
    persistent = train_cfg.num_workers > 0

    train_sampler = SkippableDistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=train_cfg.seed,
    )
    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.train_batch_size,
        sampler=train_sampler,
        num_workers=train_cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        drop_last=True,
    )
    eval_drop_last = len(eval_dataset) >= train_cfg.eval_batch_size  # type: ignore[arg-type]
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=train_cfg.eval_batch_size,
        sampler=eval_sampler,
        num_workers=train_cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        drop_last=eval_drop_last,
    )
    return train_loader, eval_loader
