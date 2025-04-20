import random
from pathlib import Path
from typing import Any, Dict, Union
import math
import os
import numpy as np
import torch


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, file_path, sequence_length=512, block_size=100000, shuffle_blocks=True
    ):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.block_size = block_size
        self.shuffle_blocks = shuffle_blocks

        self.data = np.memmap(file_path, mode="r", dtype=np.uint16)
        self.total_tokens = len(self.data)
        self.total_blocks = self.total_tokens // self.block_size
        print(f"Total tokens: {self.total_tokens}, Total blocks: {self.total_blocks}")

        self.block_indices = list(range(self.total_blocks))
        if self.shuffle_blocks:
            random.shuffle(self.block_indices)

    def __iter__(self):
        for block_idx in self.block_indices:
            block_start = block_idx * self.block_size
            block_end = min(block_start + self.block_size, self.total_tokens)

            block_data = self.data[block_start:block_end]

            for seq_start in range(
                0, len(block_data) - self.sequence_length - 1, self.sequence_length
            ):
                if seq_start + self.sequence_length + 1 > len(block_data):
                    break

                x = block_data[seq_start : seq_start + self.sequence_length]
                y = block_data[seq_start + 1 : seq_start + self.sequence_length + 1]

                yield torch.tensor(x, dtype=torch.long), torch.tensor(
                    y, dtype=torch.long
                )


class MemMapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        memmap_file: Union[str, Path],
        chunk_size: int = 1024,
        memmap_dtype=np.uint16,
    ):
        self.path = Path(memmap_file)
        if not self.path.exists():
            raise FileNotFoundError(f"Memmap file not found: {self.path}")

        self.chunk_size = chunk_size
        self.dtype = memmap_dtype
        self.item_size = np.dtype(self.dtype).itemsize

        file_size = self.path.stat().st_size
        self.total_chunks = file_size // (self.item_size * self.chunk_size)
        self.effective_len = max(0, self.total_chunks - 1)

        self.data = np.memmap(self.path, dtype=self.dtype, mode="r")

    def __len__(self) -> int:
        return self.effective_len

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < 0:
            index = self.effective_len + index

        start_idx = index * self.chunk_size
        end_idx = start_idx + self.chunk_size

        x_data = self.data[start_idx:end_idx]
        y_data = self.data[start_idx + 1 : end_idx + 1]

        x = torch.from_numpy(x_data.astype(np.int64))
        y = torch.from_numpy(y_data.astype(np.int64))

        return {"input_ids": x, "labels": y}


def generate_affine_parameters(num_blocks, seed=42):
    random.seed(seed)
    multiplier = random.randrange(1, num_blocks)
    while math.gcd(multiplier, num_blocks) != 1:
        multiplier = random.randrange(1, num_blocks)
    offset = random.randrange(0, num_blocks)
    return multiplier, offset


def affine_permutation(i, multiplier, offset, num_blocks):
    return (i * multiplier + offset) % num_blocks


class AffinePermutedMapDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, split, block_size, seed=42):
        self.block_size = block_size
        self.seed = seed

        filename = os.path.join(data_path, f"{split}.bin")
        self.data = np.memmap(filename, mode="r", dtype=np.uint16)
        self.num_blocks = (len(self.data) - 1) // self.block_size

        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = epoch
        self.multiplier, self.offset = generate_affine_parameters(
            self.num_blocks, seed=self.seed + self._epoch
        )
        print(
            f"Dataset Rank (Implicit): Set epoch {epoch}, seed {self.seed + self._epoch}"
        )

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        try:
            multiplier = self.multiplier
            offset = self.offset
        except AttributeError:
            print("Warning: Affine parameters not set via set_epoch, using epoch 0.")
            self.set_epoch(0)
            multiplier = self.multiplier
            offset = self.offset

        block_idx = affine_permutation(idx, multiplier, offset, self.num_blocks)

        start = int(block_idx * self.block_size)

        numpy_x = self.data[start : start + self.block_size].astype(np.int64)
        numpy_y = self.data[start + 1 : start + 1 + self.block_size].astype(np.int64)

        x_sample = torch.from_numpy(numpy_x)
        y_sample = torch.from_numpy(numpy_y)

        return {"input_ids": x_sample, "labels": y_sample}
