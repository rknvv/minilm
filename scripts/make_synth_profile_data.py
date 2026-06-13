"""Generate synthetic random-token memmap data for SPEED profiling only."""

import argparse
import json
import os

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out_dir", default=os.path.join(PROJECT_ROOT, "data", "profile_synth")
    )
    p.add_argument("--vocab_size", type=int, default=153856)
    p.add_argument("--train_tokens", type=int, default=4_194_304)
    p.add_argument("--val_tokens", type=int, default=524_288)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for name, n in (("train.bin", args.train_tokens), ("val.bin", args.val_tokens)):
        path = os.path.join(args.out_dir, name)
        with open(path, "wb") as f:
            remaining = n
            chunk = 1 << 22
            while remaining > 0:
                m = min(chunk, remaining)
                rng.integers(0, args.vocab_size, size=m, dtype=np.uint32).tofile(f)
                remaining -= m
        print(f"wrote {path}: {n:,} uint32 tokens ({n * 4 / 2**20:.1f} MiB)")

    meta = {"token_dtype": "uint32", "vocab_size": args.vocab_size}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"wrote {os.path.join(args.out_dir, 'meta.json')}: {meta}")


if __name__ == "__main__":
    main()
