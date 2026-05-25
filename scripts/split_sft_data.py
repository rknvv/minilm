"""Split SFT JSONL file into train/val sets."""

import random
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to input JSONL file")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = input_path.parent

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    random.seed(args.seed)
    random.shuffle(lines)

    val_size = int(len(lines) * args.val_ratio)
    val_lines = lines[:val_size]
    train_lines = lines[val_size:]

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    with open(val_path, "w", encoding="utf-8") as f:
        f.writelines(val_lines)

    print(f"Train: {len(train_lines)} -> {train_path}")
    print(f"Val:   {len(val_lines)} -> {val_path}")


if __name__ == "__main__":
    main()
