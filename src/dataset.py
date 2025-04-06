from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch.utils.data import Dataset


def get_document_lengths(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    doc_boundaries = torch.cat(
        [
            torch.tensor([-1], dtype=torch.int32),
            (input_ids == eos_token_id).nonzero(as_tuple=True)[0].to(dtype=torch.int32),
            torch.tensor(
                [] if input_ids[-1] == eos_token_id else [input_ids.shape[0] - 1],
                dtype=torch.int32,
            ),
        ]
    )
    return doc_boundaries[1:] - doc_boundaries[:-1]


def get_bytes_range(source, bytes_start: int, num_bytes: int) -> bytes:
    with open(source, "rb") as f:
        f.seek(bytes_start)
        return f.read(num_bytes)


def file_size(path) -> int:
    return os.stat(path).st_size


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


class MemMapDataset(Dataset[Dict[str, Any]]):
    def __init__(
        self,
        *paths,
        chunk_size: int = 1024,
        memmap_dtype: Union[
            Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]
        ] = np.uint16,
    ):
        if not paths:
            raise ValueError("At least one path is required")

        self._memmap_paths = paths
        self._chunk_size = chunk_size
        self._mmap_offsets: Optional[List[Tuple[int, int]]] = None
        self._num_instances: Optional[int] = None
        self.dtype = memmap_dtype

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def max_seq_len(self) -> int:
        return self.chunk_size

    @property
    def offsets(self) -> List[Tuple[int, int]]:
        if self._mmap_offsets is None:
            import concurrent.futures

            self._mmap_offsets = []

            path_to_length = {}
            path_to_mask_path = {}
            mask_path_to_length = {}

            with concurrent.futures.ThreadPoolExecutor() as executor:
                path_futures = []
                mask_path_futures = []
                for i, path in enumerate(self._memmap_paths):
                    path_futures.append(executor.submit(self._get_file_length, path))

                for future in concurrent.futures.as_completed(path_futures):
                    path, length = future.result()
                    path_to_length[path] = length

                for future in concurrent.futures.as_completed(mask_path_futures):
                    path, length = future.result()
                    mask_path_to_length[path] = length

            start_offset = 0
            for path in self._memmap_paths:
                length = path_to_length[path]
                if mask_path_to_length:
                    mask_path = path_to_mask_path[path]
                    if length != mask_path_to_length[mask_path]:
                        raise ValueError(
                            f"masking file '{mask_path}' should be the same size as '{path}'"
                        )
                end_offset = start_offset + length
                self._mmap_offsets.append((start_offset, end_offset))
                start_offset += length
        return self._mmap_offsets

    def _read_chunk_from_memmap(self, path, index: int, dtype=None) -> torch.Tensor:
        dtype = dtype or self.dtype
        item_size = dtype(0).itemsize
        bytes_start = index * item_size * self._chunk_size
        num_bytes = item_size * self._chunk_size
        buffer = get_bytes_range(path, bytes_start, num_bytes)
        array = np.frombuffer(buffer, dtype=dtype)
        if dtype == np.bool_:
            return torch.tensor(array)
        else:
            return torch.tensor(array.astype(np.int_), dtype=torch.long)

    def _get_file_length(self, path, dtype=None):
        dtype = dtype or self.dtype
        item_size = dtype(0).itemsize
        return path, file_size(path) // (item_size * self._chunk_size)

    def __len__(self) -> int:
        if self._num_instances is None:
            self._num_instances = self.offsets[-1][1]
        return self._num_instances

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)
        pos_index = index if index >= 0 else len(self) + index

        memmap_index: Optional[int] = None

        memmap_local_index: Optional[int] = None
        for i, (offset_start, offset_end) in enumerate(self.offsets):
            if offset_start <= pos_index < offset_end:
                memmap_index = i
                memmap_local_index = pos_index - offset_start

        input_ids = self._read_chunk_from_memmap(
            self._memmap_paths[memmap_index], memmap_local_index
        )
        out: Dict[str, Any] = {"input_ids": input_ids}

        return out


class MemMapDatasetForLM(torch.utils.data.Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        *paths: Union[str, Path],
        chunk_size: int = 1024,
        memmap_dtype: Union[
            Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]
        ] = np.uint16,
    ):
        if not paths:
            raise ValueError("At least one path is required")

        self._memmap_paths = [Path(p) for p in paths]
        if not all(p.exists() for p in self._memmap_paths):
            missing = [p for p in self._memmap_paths if not p.exists()]
            raise FileNotFoundError(f"Memmap files not found: {missing}")

        self._chunk_size = chunk_size
        self.dtype = memmap_dtype
        self._item_size = self.dtype(0).itemsize

        self._mmap_offsets: List[Tuple[int, int]] = []
        self._total_chunks: int = 0
        self._calculate_offsets_and_total_chunks()

        self._effective_len = max(0, self._total_chunks - 1)

    def _calculate_offsets_and_total_chunks(self):
        import concurrent.futures

        path_to_length = {}
        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for path in self._memmap_paths:
                futures.append(executor.submit(self._get_num_chunks_in_file, path))

            for future in concurrent.futures.as_completed(futures):
                path, num_chunks = future.result()
                path_to_length[path] = num_chunks

        start_offset = 0
        calculated_offsets = []
        for path in self._memmap_paths:
            num_chunks = path_to_length.get(path)
            if num_chunks == 0:
                continue

            end_offset = start_offset + num_chunks
            calculated_offsets.append((start_offset, end_offset))
            start_offset += num_chunks

        self._mmap_offsets = calculated_offsets
        self._total_chunks = start_offset

    @property
    def offsets(self) -> List[Tuple[int, int]]:
        return self._mmap_offsets

    def _get_num_chunks_in_file(self, path: Path) -> Tuple[Path, int]:
        fsize = file_size(path)
        num_chunks = fsize // (self._item_size * self._chunk_size)
        return path, num_chunks

    def _get_path_and_local_index(self, global_chunk_index: int) -> Tuple[Path, int]:
        memmap_file_index: Optional[int] = None
        memmap_local_chunk_index: Optional[int] = None

        for i, (offset_start, offset_end) in enumerate(self.offsets):
            if offset_start <= global_chunk_index < offset_end:
                memmap_file_index = i
                memmap_local_chunk_index = global_chunk_index - offset_start
                break

        return self._memmap_paths[memmap_file_index], memmap_local_chunk_index

    def _read_chunk_from_memmap(
        self, path: Path, local_chunk_index: int
    ) -> torch.Tensor:
        bytes_start = local_chunk_index * self._item_size * self._chunk_size

        mmap_obj = np.memmap(
            path,
            dtype=self.dtype,
            mode="r",
            offset=bytes_start,
            shape=(self._chunk_size,),
        )

        array = np.array(mmap_obj, dtype=self.dtype)
        del mmap_obj

        return torch.tensor(array.astype(np.int64), dtype=torch.long)

    def __len__(self) -> int:
        return self._effective_len

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        effective_index = index if index >= 0 else self._effective_len + index

        global_idx_x = effective_index
        global_idx_y_chunk = effective_index + 1

        path_x, local_idx_x = self._get_path_and_local_index(global_idx_x)
        path_y_chunk, local_idx_y_chunk = self._get_path_and_local_index(
            global_idx_y_chunk
        )

        x_chunk = self._read_chunk_from_memmap(path_x, local_idx_x)
        y_following_chunk = self._read_chunk_from_memmap(
            path_y_chunk, local_idx_y_chunk
        )

        combined_for_y = torch.cat((x_chunk[1:], y_following_chunk[0:1]))

        x = x_chunk
        y = combined_for_y

        return {"input_ids": x, "labels": y}
