import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from config import ModelArgs, TrainConfig
from dataio.dataset import MemmapDataset, SFTDataset


def build_datasets(
    train_cfg: TrainConfig, model_args: ModelArgs
) -> Tuple[Dataset, Dataset]:
    task = str(train_cfg.task).lower()

    if task == "pretrain":
        train_dataset = MemmapDataset(
            os.path.join(train_cfg.dataset_dir, "train.bin"),
            sequence_length=model_args.max_seq_len,
            memmap_dtype=np.uint16,
        )
        eval_dataset = MemmapDataset(
            os.path.join(train_cfg.dataset_dir, "val.bin"),
            sequence_length=model_args.max_seq_len,
            memmap_dtype=np.uint16,
        )
        return train_dataset, eval_dataset

    if task == "sft":
        if not train_cfg.train_data_path:
            raise ValueError("For task='sft', train_data_path is required.")
        if not train_cfg.tokenizer_path:
            raise ValueError("For task='sft', tokenizer_path is required.")

        eval_path = train_cfg.eval_data_path or train_cfg.train_data_path
        train_dataset = SFTDataset(
            data_path=train_cfg.train_data_path,
            tokenizer_path=train_cfg.tokenizer_path,
            max_seq_len=model_args.max_seq_len,
            ignore_idx=train_cfg.sft_ignore_idx,
        )
        eval_dataset = SFTDataset(
            data_path=eval_path,
            tokenizer_path=train_cfg.tokenizer_path,
            max_seq_len=model_args.max_seq_len,
            ignore_idx=train_cfg.sft_ignore_idx,
        )
        return train_dataset, eval_dataset

    raise ValueError("train.task must be one of: 'pretrain', 'sft'.")


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

    train_sampler: Optional[DistributedSampler] = None
    eval_sampler: Optional[DistributedSampler] = None
    train_shuffle = True
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        eval_sampler = DistributedSampler(
            eval_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.train_batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=train_cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=train_cfg.eval_batch_size,
        sampler=eval_sampler,
        num_workers=train_cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    return train_loader, eval_loader
