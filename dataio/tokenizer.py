# SentencePiece tokenizer model
# Reference: https://github.com/meta-llama/llama/blob/main/llama/tokenizer.py

import logging
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.typing import DTypeLike
from sentencepiece import SentencePieceProcessor

logger = logging.getLogger()


class Tokenizer:
    """SentencePiece model wrapper with flexible I/O types."""

    def __init__(self, model_file: str) -> None:
        self._sp: Any = SentencePieceProcessor()
        self._sp.Load(model_file)
        self.bos_id = self._sp.bos_id()
        self.eos_id = self._sp.eos_id()
        self.pad_id = self._sp.pad_id()
        self.vocab_size = self._sp.vocab_size()
        self.start_header_id = self.piece_to_id("<r0>")
        self.end_header_id = self.piece_to_id("<r1>")
        self.eot_id = self.eos_id
        self._nl_ids = self._sp.encode("\n\n", out_type=int)

        self._special_ids = {
            token_id
            for token_id in (
                self.bos_id, self.eos_id, self.pad_id,
                self.start_header_id, self.end_header_id,
            )
            if token_id >= 0
        }

        logger.info(
            f"Vocab size: {self.vocab_size}\nBOS id: {self.bos_id}\nEOS id: {self.eos_id}"
        )
        assert self._sp.vocab_size() == self._sp.get_piece_size()

    def _format_header(self, role: str) -> list[int]:
        role_ids = self._sp.encode(role, out_type=int)
        return [self.start_header_id] + role_ids + [self.end_header_id] + self._nl_ids

    def _build_chat_message(
        self,
        role: str,
        content: str,
        ignore_idx: int | None = None,
    ) -> tuple[list[int], list[int] | None]:
        header = self._format_header(role)
        content_ids = self._sp.encode(content, out_type=int)
        message_ids = header + content_ids + [self.eot_id]
        if ignore_idx is None:
            return message_ids, None

        if role == "assistant":
            message_labels = [ignore_idx] * len(header) + content_ids + [self.eot_id]
        else:
            message_labels = [ignore_idx] * len(message_ids)
        return message_ids, message_labels

    def _build_chat_sequence(
        self,
        messages: list[dict[str, str]],
        ignore_idx: int | None = None,
        add_generation_prompt: bool = False,
    ) -> tuple[list[int], list[int] | None]:
        token_ids: list[int] = [self.bos_id]
        labels = [ignore_idx] if ignore_idx is not None else None

        for msg in messages:
            message_ids, message_labels = self._build_chat_message(
                msg["role"],
                msg["content"],
                ignore_idx=ignore_idx,
            )
            token_ids.extend(message_ids)
            if labels is not None and message_labels is not None:
                labels.extend(message_labels)

        if add_generation_prompt:
            assistant_header = self._format_header("assistant")
            token_ids.extend(assistant_header)
            if labels is not None:
                labels.extend([ignore_idx] * len(assistant_header))

        return token_ids, labels

    def build_chat_example(
        self,
        messages: list[dict[str, str]],
        max_seq_len: int,
        ignore_idx: int = -100,
    ) -> tuple[list[int], list[int]]:
        input_ids, labels = self._build_chat_sequence(messages, ignore_idx=ignore_idx)
        assert labels is not None

        target_with_shift = max_seq_len + 1
        if len(input_ids) > target_with_shift:
            input_ids = input_ids[:target_with_shift]
            labels = labels[:target_with_shift]

        pad_len = target_with_shift - len(input_ids)
        if pad_len > 0:
            pad_token_id = self.pad_id if self.pad_id >= 0 else 0
            input_ids.extend([pad_token_id] * pad_len)
            labels.extend([ignore_idx] * pad_len)

        return input_ids[:-1], labels[1:]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        return_tensors: Literal["pt", "np"] | None = None,
    ) -> list[int] | str | torch.Tensor | np.ndarray:
        """Format chat messages into token IDs (like HF apply_chat_template).

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            add_generation_prompt: Append assistant header for generation.
            tokenize: If True return token IDs, else return formatted string.
            return_tensors: "pt" or "np" to return 2D tensor [1, seq_len].
        """
        token_ids, _ = self._build_chat_sequence(
            messages,
            add_generation_prompt=add_generation_prompt,
        )

        if not tokenize:
            return self._sp.decode(token_ids)

        if return_tensors == "pt":
            return torch.tensor([token_ids], dtype=torch.long)
        if return_tensors == "np":
            return np.array([token_ids], dtype=np.int64)
        return token_ids

    def __call__(
        self,
        text: str | list[str],
        bos: bool = False,
        eos: bool = True,
        return_tensors: Literal["np", "pt", "numpy", "tensor", "torch", "list"]
        | bool = "pt",
        dtype: DTypeLike | torch.dtype | None = None,
        device: torch.device | str | None = None,
        padding: bool = True,
        pad_value: int | None = None,
    ) -> np.ndarray | torch.Tensor | list[list[int]]:
        """Batch-style interface (like HF). Always returns 2D: [batch_size, seq_len]."""
        mode = self._normalize_return_tensors(return_tensors)

        if isinstance(text, str):
            text = [text]

        if not all(isinstance(s, str) for s in text):
            raise TypeError("All elements of text must be strings.")

        batch: list[list[int]] = [
            self._encode_one(s, bos=bos, eos=eos) for s in text
        ]

        if padding and len(batch) > 1:
            resolved_pad = (
                pad_value
                if pad_value is not None
                else (self.pad_id if self.pad_id >= 0 else 0)
            )
            batch = self._pad_batch(batch, pad_token_id=resolved_pad)

        if mode == "list":
            return batch

        return self._format_output(
            token_ids=batch,
            return_tensors=mode,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _normalize_return_tensors(
        return_tensors: Literal["np", "pt", "numpy", "tensor", "torch", "list"]
        | bool,
    ) -> Literal["np", "pt", "list"]:
        if isinstance(return_tensors, bool):
            return "pt" if return_tensors else "np"
        mapping: dict[str, Literal["np", "pt", "list"]] = {
            "np": "np",
            "numpy": "np",
            "pt": "pt",
            "tensor": "pt",
            "torch": "pt",
            "list": "list",
        }
        mode = mapping.get(return_tensors.lower())
        if mode is None:
            raise ValueError(
                "return_tensors must be one of: 'np', 'pt', 'list' (aliases: 'numpy', 'tensor', 'torch')."
            )
        return mode

    @staticmethod
    def _to_int_list(
        token_ids: list[int] | np.ndarray | torch.Tensor,
    ) -> list[int]:
        if isinstance(token_ids, torch.Tensor):
            if token_ids.ndim > 1:
                raise ValueError("Expected 0D/1D torch tensor with token ids.")
            token_ids = token_ids.detach().cpu().reshape(-1).tolist()
        elif isinstance(token_ids, np.ndarray):
            if token_ids.ndim > 1:
                raise ValueError("Expected 0D/1D numpy array with token ids.")
            token_ids = token_ids.reshape(-1).tolist()

        if isinstance(token_ids, list):
            if token_ids and isinstance(token_ids[0], (list, np.ndarray, torch.Tensor)):
                raise ValueError("Expected 1D list[int], got nested sequence.")
            return [int(token_id) for token_id in token_ids]

        raise TypeError("token_ids must be list[int], numpy.ndarray or torch.Tensor.")

    @staticmethod
    def _is_batch_input(
        token_ids: list[int] | list[list[int]] | np.ndarray | torch.Tensor,
    ) -> bool:
        if isinstance(token_ids, (torch.Tensor, np.ndarray)):
            return token_ids.ndim == 2
        return bool(token_ids) and isinstance(
            token_ids[0], (list, np.ndarray, torch.Tensor)
        )

    def _to_batch_int_list(
        self, token_ids: list[int] | list[list[int]] | np.ndarray | torch.Tensor
    ) -> list[list[int]]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu()
            if token_ids.ndim > 2:
                raise ValueError("Expected 0D/1D/2D torch tensor with token ids.")
            if token_ids.ndim == 2:
                return [self._to_int_list(row) for row in token_ids]
            return [self._to_int_list(token_ids)]

        if isinstance(token_ids, np.ndarray):
            if token_ids.ndim > 2:
                raise ValueError("Expected 0D/1D/2D numpy array with token ids.")
            if token_ids.ndim == 2:
                return [self._to_int_list(row) for row in token_ids]
            return [self._to_int_list(token_ids)]

        if isinstance(token_ids, list):
            if not token_ids:
                return []
            if self._is_batch_input(token_ids):
                batch: list[list[int]] = []
                for row in token_ids:
                    if not isinstance(row, (list, np.ndarray, torch.Tensor)):
                        raise ValueError(
                            "Batch token ids must be list/np.ndarray/torch.Tensor rows."
                        )
                    batch.append(self._to_int_list(row))
                return batch
            return [self._to_int_list(token_ids)]

        raise TypeError("token_ids must be list/np.ndarray/torch.Tensor.")

    @staticmethod
    def _resolve_dtype(
        return_tensors: Literal["np", "pt"],
        dtype: DTypeLike | torch.dtype | None,
    ) -> DTypeLike | torch.dtype:
        if return_tensors == "pt":
            if dtype is None:
                return torch.long
            if isinstance(dtype, torch.dtype):
                return dtype
            raise TypeError("For return_tensors='pt', dtype must be a torch dtype.")

        if isinstance(dtype, torch.dtype):
            raise TypeError("For return_tensors='np', dtype must be a numpy dtype.")
        return np.int64 if dtype is None else dtype

    @staticmethod
    def _pad_batch(
        batch_token_ids: list[list[int]],
        pad_token_id: int,
    ) -> list[list[int]]:
        max_len = max((len(seq) for seq in batch_token_ids), default=0)
        if max_len == 0:
            return batch_token_ids
        return [seq + [pad_token_id] * (max_len - len(seq)) for seq in batch_token_ids]

    @staticmethod
    def _format_output(
        token_ids: list[int] | list[list[int]],
        return_tensors: Literal["np", "pt"],
        device: torch.device | str | None,
        dtype: DTypeLike | torch.dtype | None,
    ) -> np.ndarray | torch.Tensor:
        resolved_dtype = Tokenizer._resolve_dtype(
            return_tensors=return_tensors,
            dtype=dtype,
        )
        if return_tensors == "pt":
            return torch.tensor(
                token_ids,
                dtype=cast(torch.dtype, resolved_dtype),
                device=device,
            )
        return np.asarray(token_ids, dtype=cast(DTypeLike, resolved_dtype))

    def _encode_one(self, text: str, bos: bool, eos: bool) -> list[int]:
        token_ids = self._sp.encode(text, out_type=int)
        if bos:
            token_ids = [self.bos_id] + token_ids
        if eos:
            token_ids = token_ids + [self.eos_id]
        return token_ids

    def decode(
        self,
        token_ids: list[int] | list[list[int]] | np.ndarray | torch.Tensor,
        skip_special_tokens: bool = False,
    ) -> str | list[str]:
        if self._is_batch_input(token_ids):
            return [
                self._decode_one(ids, skip_special_tokens=skip_special_tokens)
                for ids in self._to_batch_int_list(token_ids)
            ]
        return self._decode_one(token_ids, skip_special_tokens=skip_special_tokens)

    def _decode_one(self, token_ids: Any, skip_special_tokens: bool = False) -> str:
        ids = self._to_int_list(token_ids)
        if skip_special_tokens:
            ids = [tid for tid in ids if tid not in self._special_ids]
            return self._sp.decode(ids)

        parts: list[str] = []
        segment: list[int] = []
        for tid in ids:
            if tid in self._special_ids:
                if segment:
                    parts.append(self._sp.decode(segment))
                    segment = []
                parts.append(self._sp.id_to_piece(tid))
            else:
                segment.append(tid)
        if segment:
            parts.append(self._sp.decode(segment))
        return "".join(parts)

    def piece_to_id(self, piece: str) -> int:
        return int(self._sp.piece_to_id(piece))

    def id_to_piece(self, token_id: int) -> str:
        return str(self._sp.id_to_piece(token_id))
