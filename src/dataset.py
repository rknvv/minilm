import torch
import numpy as np


class MemmapDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, sequence_length):
        self.data = np.memmap(file_path, dtype=np.uint16, mode="r")
        self.sequence_length = sequence_length
        self.num_sequences = len(self.data) // sequence_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.sequence_length
        end_idx = start_idx + self.sequence_length
        x = self.data[start_idx:end_idx]
        y = self.data[start_idx + 1 : end_idx + 1]
        return {
            "input_ids": torch.from_numpy(x.astype(np.int64)),
            "labels": torch.from_numpy(y.astype(np.int64)),
        }
