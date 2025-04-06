import os

import numpy as np
import argparse

from src.tokenizer import Tokenizer
from datasets import load_dataset
from tqdm import tqdm


if __name__ == "__main__":
    # os.chdir("../")
    parser = argparse.ArgumentParser(description="Loading the tokenizer")
    parser.add_argument(
        "--tokenizer_path", required=True, type=str, help="Path to tokenizer"
    )
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    args = parser.parse_args()

    tokenizer = Tokenizer(model_file=args.tokenizer_path)

    dataset = load_dataset(
        "text",
        data_files=args.dataset_dir,
        streaming=True,
    )

    def tokenize(item):
        ids = tokenizer.encode(item["text"], bos=False, eos=True)
        return {"ids": ids}

    tokenized_data = dataset.map(tokenize, remove_columns="text")

    filename = os.path.join(args.output_path, "data.bin")
    with open(filename, "wb") as f:
        for item in tqdm(tokenized_data["train"], desc="Processing data"):
            ids = item["ids"]
            np.array(ids, dtype=np.uint16).tofile(f)
