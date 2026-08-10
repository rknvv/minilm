from typing import Optional

import torch
import numpy as np

_VOCAB_CHECK_TOKENS = 1_000_000


class MemmapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path: str,
        sequence_length: int,
        memmap_dtype=np.uint16,
        vocab_size: Optional[int] = None,
    ) -> None:
        self.sequence_length = int(sequence_length)
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be > 0.")

        if vocab_size is not None and vocab_size - 1 > np.iinfo(memmap_dtype).max:
            raise ValueError(
                f"{file_path}: dtype {np.dtype(memmap_dtype).name} cannot represent "
                f"vocab_size {vocab_size}; the data must be stored as a wider dtype "
                f"(see meta.json / token_dtype)."
            )

        self.data = np.memmap(file_path, dtype=memmap_dtype, mode="r")
        self.num_sequences = max(0, (len(self.data) - 1) // self.sequence_length)

        if vocab_size is not None and len(self.data) > 0:
            sample = np.asarray(self.data[: min(len(self.data), _VOCAB_CHECK_TOKENS)])
            max_id = int(sample.max())
            if max_id >= vocab_size:
                raise ValueError(
                    f"{file_path}: token id {max_id} >= vocab_size {vocab_size}. "
                    f"The file was likely written with a different token dtype "
                    f"than {np.dtype(memmap_dtype).name} (see meta.json / token_dtype)."
                )

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
