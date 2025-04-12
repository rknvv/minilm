import random
from pathlib import Path
from typing import Any, Dict, Union

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
