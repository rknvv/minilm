import sentencepiece as spm
import os

special_tokens = [f"<r{i}>" for i in range(13)]

options = dict(
    input="/Volumes/KINGSTON/LLM_300m_prepare/data/corpus_ver_2/corpus.txt",
    input_format="text",
    model_type="bpe",
    model_prefix="bpe_65k",
    vocab_size=65536,
    normalization_rule_name="identity",
    remove_extra_whitespaces=False,
    input_sentence_size=1000000,
    max_sentence_length=30000,
    seed_sentencepiece_size=1000000,
    shuffle_input_sentence=True,
    character_coverage=0.9994999766349792,
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
