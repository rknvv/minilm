import json
from typing import Any

import torch
import numpy as np

from dataio.tokenizer import Tokenizer


class MemmapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path: str,
        sequence_length: int,
        memmap_dtype=np.uint16,
    ) -> None:
        self.sequence_length = int(sequence_length)
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be > 0.")

        self.data = np.memmap(file_path, dtype=memmap_dtype, mode="r")
        self.num_sequences = max(0, (len(self.data) - 1) // self.sequence_length)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int):
        start_idx = idx * self.sequence_length
        end_idx = start_idx + self.sequence_length
        x = self.data[start_idx:end_idx]
        y = self.data[start_idx + 1 : end_idx + 1]
        if len(x) != self.sequence_length or len(y) != self.sequence_length:
            raise IndexError(
                f"Invalid sample at idx={idx}: got x={len(x)}, y={len(y)}, "
                f"expected={self.sequence_length}."
            )
        return {
            "input_ids": torch.from_numpy(x.astype(np.int64)),
            "labels": torch.from_numpy(y.astype(np.int64)),
        }


class SFTDataset(torch.utils.data.Dataset):
    """Llama-like format."""

    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        max_seq_len: int = 1024,
        ignore_idx: int = -100,
    ) -> None:
        self.tokenizer = Tokenizer(tokenizer_path)
        self.max_seq_len = max_seq_len
        self.ignore_idx = ignore_idx
        self.data: list[list[dict[str, Any]]] = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    self.data.append(obj["messages"])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        messages = self.data[idx]
        input_ids, labels = self.tokenizer.build_chat_example(
            messages=messages,
            max_seq_len=self.max_seq_len,
            ignore_idx=self.ignore_idx,
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
