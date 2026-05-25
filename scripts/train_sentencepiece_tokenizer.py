import os

import sentencepiece as spm

def train_tokenizer(
    input: str,
    model_prefix: str,
    vocab_size: int = 32768,
    num_special_tokens: int = 13,
    character_coverage: float = 0.9994999766349792,
    input_sentence_size: int = 1_000_000,
    max_sentence_length: int = 30000,
) -> None:

    special_tokens = [f"<r{i}>" for i in range(num_special_tokens)]

    options = dict(
        input=input,
        input_format="text",
        model_type="bpe",
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        input_sentence_size=input_sentence_size,
        max_sentence_length=max_sentence_length,
        seed_sentencepiece_size=1000000,
        shuffle_input_sentence=True,
        character_coverage=character_coverage,
        byte_fallback=True,
        split_digits=True,
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_by_number=True,
        shrinking_factor=0.75,
        max_sentencepiece_length=16,
        add_dummy_prefix=True,
        num_sub_iterations=2,
        escape_whitespaces=True,
        enable_differential_privacy=False,
        allow_whitespace_only_pieces=True,
        treat_whitespace_as_suffix=False,
        pad_id=-1,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        control_symbols=special_tokens,
        num_threads=os.cpu_count(),
        train_extremely_large_corpus=False,
    )

    spm.SentencePieceTrainer.train(**options)

if __name__ == "__main__":
    try:
        import fire

        fire.Fire(train_tokenizer)
    except ImportError:
        import argparse

        parser = argparse.ArgumentParser(description="Train a SentencePiece BPE tokenizer.")
        parser.add_argument("--input", required=True)
        parser.add_argument("--model_prefix", required=True)
        parser.add_argument("--vocab_size", type=int, default=32768)
        parser.add_argument("--num_special_tokens", type=int, default=13)
        args = parser.parse_args()
        train_tokenizer(
            input=args.input,
            model_prefix=args.model_prefix,
            vocab_size=args.vocab_size,
            num_special_tokens=args.num_special_tokens,
        )
