import argparse
import json
import os
import sys
from typing import Any

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer as TokenizersTokenizer
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class HFTokenizer:
    def __init__(self, path: str) -> None:
        self._tk = TokenizersTokenizer.from_file(path)
        self.vocab_size = self._tk.get_vocab_size()
        bos = self._tk.token_to_id("<bos>")
        eos = self._tk.token_to_id("<eos>")
        self.bos_id = -1 if bos is None else bos
        self.eos_id = -1 if eos is None else eos

    def __call__(
        self,
        text: str,
        bos: bool,
        eos: bool,
        dtype=None,
    ) -> np.ndarray:
        ids = self._tk.encode(text, add_special_tokens=False).ids
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        return np.asarray(ids, dtype=dtype if dtype is not None else np.int64)


def load_tokenizer(path: str) -> HFTokenizer:
    if not path.endswith(".json"):
        raise ValueError(
            f"Expected a HF tokenizer.json, got {path!r}. The CPT vocabulary ships "
            "with the pruned Gemma-3 checkpoint (external/.../tokenizer.json)."
        )
    return HFTokenizer(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize text corpus into train/val .bin files."
    )
    parser.add_argument(
        "--tokenizer_path",
        required=True,
        type=str,
        help="Path to the HF tokenizer.json shipped with the pruned Gemma-3 checkpoint.",
    )
    parser.add_argument(
        "--dataset_files",
        nargs="+",
        required=True,
        help="One or more input text files for datasets.load_dataset(..., data_files=...).",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="Directory where train.bin and val.bin will be written.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.01,
        help="Validation split ratio in [0, 1).",
    )
    parser.add_argument(
        "--split_buckets",
        type=int,
        default=10_000,
        help="Modulo buckets for deterministic train/val split.",
    )
    parser.add_argument(
        "--token_dtype",
        choices=["auto", "uint16", "uint32"],
        default="auto",
        help="Storage dtype for token ids; auto picks by tokenizer vocab size.",
    )
    parser.add_argument(
        "--add_bos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepend BOS to each document (Gemma pretraining framing).",
    )
    parser.add_argument(
        "--add_eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append EOS to each document.",
    )
    return parser.parse_args()


def _split_bucket(i: int, buckets: int) -> int:
    z = (i + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) % buckets


def resolve_token_dtype(token_dtype: str, vocab_size: int) -> np.dtype:
    max_id = vocab_size - 1
    if token_dtype == "auto":
        return np.dtype(np.uint16 if max_id <= np.iinfo(np.uint16).max else np.uint32)
    dtype = np.dtype(token_dtype)
    if max_id > np.iinfo(dtype).max:
        raise ValueError(
            f"Tokenizer vocab_size={vocab_size} does not fit {dtype.name}; "
            "use --token_dtype auto or uint32."
        )
    return dtype


def tokenize_corpus(
    tokenizer: HFTokenizer,
    dataset_files: list[str],
    output_path: str,
    val_ratio: float,
    split_buckets: int,
    token_dtype: str = "auto",
    add_bos: bool = True,
    add_eos: bool = True,
) -> None:
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("--val_ratio must be in [0, 1).")
    if split_buckets <= 0:
        raise ValueError("--split_buckets must be > 0.")
    dtype = resolve_token_dtype(token_dtype, tokenizer.vocab_size)
    if add_bos and tokenizer.bos_id < 0:
        raise ValueError("--add_bos requested but the tokenizer has no BOS id.")
    print(
        f"Token dtype: {dtype.name} (vocab_size={tokenizer.vocab_size}), "
        f"add_bos={add_bos}, add_eos={add_eos}"
    )

    data_files: str | list[str]
    if len(dataset_files) == 1:
        data_files = dataset_files[0]
    else:
        data_files = dataset_files

    dataset = load_dataset("text", data_files=data_files, streaming=True)

    def tokenize(item: dict[str, Any]) -> dict[str, np.ndarray]:
        return {
            "ids": tokenizer(item["text"], bos=add_bos, eos=add_eos, dtype=dtype)
        }

    tokenized_data = dataset.map(tokenize, remove_columns="text")

    os.makedirs(output_path, exist_ok=True)
    train_filename = os.path.join(output_path, "train.bin")
    val_filename = os.path.join(output_path, "val.bin")

    val_threshold = int(val_ratio * split_buckets)
    train_docs, val_docs = 0, 0
    train_tokens, val_tokens = 0, 0

    with open(train_filename, "wb") as train_f, open(val_filename, "wb") as val_f:
        for i, item in enumerate(tqdm(tokenized_data["train"], desc="Processing data")):
            token_ids = item["ids"]
            if not isinstance(token_ids, np.ndarray):
                token_ids = np.asarray(token_ids, dtype=dtype)
            else:
                token_ids = token_ids.astype(dtype, copy=False)

            is_val = _split_bucket(i, split_buckets) < val_threshold
            if is_val:
                token_ids.tofile(val_f)
                val_docs += 1
                val_tokens += int(token_ids.size)
            else:
                token_ids.tofile(train_f)
                train_docs += 1
                train_tokens += int(token_ids.size)

    meta = {
        "token_dtype": dtype.name,
        "vocab_size": int(tokenizer.vocab_size),
        "add_bos": add_bos,
        "add_eos": add_eos,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
    }
    meta_filename = os.path.join(output_path, "meta.json")
    with open(meta_filename, "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"Done. Train: {train_docs} docs / {train_tokens} tokens -> {train_filename}; "
        f"Val: {val_docs} docs / {val_tokens} tokens -> {val_filename}; "
        f"meta -> {meta_filename}"
    )


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer_path)
    tokenize_corpus(
        tokenizer=tokenizer,
        dataset_files=args.dataset_files,
        output_path=args.output_path,
        val_ratio=args.val_ratio,
        split_buckets=args.split_buckets,
        token_dtype=args.token_dtype,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
    )


if __name__ == "__main__":
    main()
