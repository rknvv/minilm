import argparse
import os
import sys
from typing import Any

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataio.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize text corpus into train/val .bin files.")
    parser.add_argument(
        "--tokenizer_path",
        required=True,
        type=str,
        help="Path to SentencePiece model.",
    )
    parser.add_argument(
        "--dataset_files",
        nargs="+",
        help="One or more input text files for datasets.load_dataset(..., data_files=...).",
    )
    parser.add_argument(
        "--dataset_dir",
        nargs="+",
        default=None,
        help="Backward-compatible alias of --dataset_files.",
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
    args = parser.parse_args()
    if not args.dataset_files and not args.dataset_dir:
        parser.error("one of --dataset_files or --dataset_dir is required")
    return args


def tokenize_corpus(
    tokenizer: Tokenizer,
    dataset_files: list[str],
    output_path: str,
    val_ratio: float,
    split_buckets: int,
) -> None:
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("--val_ratio must be in [0, 1).")
    if split_buckets <= 0:
        raise ValueError("--split_buckets must be > 0.")
    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError(
            f"Tokenizer vocab_size={tokenizer.vocab_size} does not fit uint16. "
            "Increase storage dtype and align trainer memmap dtype."
        )

    data_files: str | list[str]
    if len(dataset_files) == 1:
        data_files = dataset_files[0]
    else:
        data_files = dataset_files

    dataset = load_dataset("text", data_files=data_files, streaming=True)

    def tokenize(item: dict[str, Any]) -> dict[str, np.ndarray]:
        token_ids = tokenizer(
            item["text"],
            bos=False,
            eos=True,
            return_tensors="np",
            dtype=np.uint16,
        )
        if isinstance(token_ids, np.ndarray):
            return {"ids": token_ids.reshape(-1)}
        return {"ids": np.asarray(token_ids, dtype=np.uint16)}

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
                token_ids = np.asarray(token_ids, dtype=np.uint16)
            else:
                token_ids = token_ids.astype(np.uint16, copy=False)

            is_val = (i % split_buckets) < val_threshold
            if is_val:
                token_ids.tofile(val_f)
                val_docs += 1
                val_tokens += int(token_ids.size)
            else:
                token_ids.tofile(train_f)
                train_docs += 1
                train_tokens += int(token_ids.size)

    print(
        f"Done. Train: {train_docs} docs / {train_tokens} tokens -> {train_filename}; "
        f"Val: {val_docs} docs / {val_tokens} tokens -> {val_filename}"
    )


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer(model_file=args.tokenizer_path)
    dataset_files = args.dataset_files or args.dataset_dir
    tokenize_corpus(
        tokenizer=tokenizer,
        dataset_files=dataset_files,
        output_path=args.output_path,
        val_ratio=args.val_ratio,
        split_buckets=args.split_buckets,
    )


if __name__ == "__main__":
    main()
