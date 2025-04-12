import sentencepiece as spm
import os

options = dict(
    input="/Users/sandzhar/code/projects/minilm/data/pretrain/corpus.txt",
    input_format="text",
    model_type="bpe",
    model_prefix="bpe_ru_16384_vocab",
    vocab_size=16384,
    normalization_rule_name="identity",
    remove_extra_whitespaces=False,
    input_sentence_size=1000000,
    max_sentence_length=4192,
    seed_sentencepiece_size=1000000,
    shuffle_input_sentence=True,
    character_coverage=0.99995,
    byte_fallback=True,
    split_digits=True,
    split_by_unicode_script=True,
    split_by_whitespace=True,
    split_by_number=True,
    max_sentencepiece_length=16,
    add_dummy_prefix=True,
    allow_whitespace_only_pieces=True,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=3,
    num_threads=os.cpu_count(),
    train_extremely_large_corpus=True,
)

spm.SentencePieceTrainer.train(**options)
