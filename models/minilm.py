import logging
import math
from typing import cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from config import ModelArgs
from models.layers import RMSNorm, precompute_rope_cache
from models.transformer import TransformerBlock

logger = logging.getLogger()

class MiniLM(nn.Module):
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    layers: nn.ModuleList

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        assert args.vocab_size > 0

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleList(
            [TransformerBlock(args) for _ in range(args.n_layers)]
        )

        self.norm = RMSNorm(args.dim, norm_eps=args.norm_eps)

        cos, sin = precompute_rope_cache(
            self.args.dim // self.args.n_heads, self.args.max_seq_len * 2
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("w2.weight") or pn.endswith("wo.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * args.n_layers)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:

        _, seq_len = tokens.shape
        cos_buffer = cast(torch.Tensor, self.rope_cos)
        sin_buffer = cast(torch.Tensor, self.rope_sin)

        assert (
            seq_len <= self.args.max_seq_len
        ), f"Cannot forward sequence of length {seq_len}, max is {self.args.max_seq_len}"
        assert (
            start_pos + seq_len <= cos_buffer.shape[0]
        ), "RoPE cache exceeded"

        h = self.tok_embeddings(tokens)
        cos = cos_buffer[start_pos : start_pos + seq_len]
        sin = sin_buffer[start_pos : start_pos + seq_len]

        use_checkpoint = self.args.gradient_checkpointing and self.training
        for layer in self.layers:
            block = cast(TransformerBlock, layer)
            if use_checkpoint:
                h = checkpoint(block, h, start_pos, cos, sin, use_reentrant=False)
            else:
                h = block(h, start_pos, cos, sin)

        h = self.norm(h)
        return h
