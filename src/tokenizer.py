# SentencePiece tokenizer model
# Reference: https://github.com/meta-llama/llama/blob/main/llama/tokenizer.py

import logging
from typing import List

from sentencepiece import SentencePieceProcessor


logger = logging.getLogger()


class Tokenizer:
    """SentencePiece model wrapper"""

    def __init__(self, model_file: str):
        self.bpe_model = SentencePieceProcessor(model_file=model_file)
        self.bos_id = self.bpe_model.bos_id()
        self.eos_id = self.bpe_model.eos_id()
        self.vocab_size = self.bpe_model.vocab_size()
        logger.info(
            f"Vocab size: {self.vocab_size}\nBOS id: {self.bos_id}\nEOS id: {self.eos_id}"
        )
        assert self.bpe_model.vocab_size() == self.bpe_model.get_piece_size()

    def encode(self, text: str, bos: bool, eos: bool) -> List[str]:
        output = self.bpe_model.encode(text)
        if bos:
            output = [self.bos_id] + output
        if eos:
            output = output + [self.eos_id]
        return output

    def decode(self, text: List[str]) -> str:
        return self.bpe_model.decode(text)
