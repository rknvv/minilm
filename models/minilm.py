import logging
from typing import Optional, Tuple, cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from config import ModelArgs
from models.layers import (
    HAS_FLEX,
    RMSNorm,
    build_flex_block_mask,
    build_local_bool_mask,
    precompute_rope_cache,
)
from models.transformer import TransformerBlock

logger = logging.getLogger()


class MiniLM(nn.Module):
    rope_cos_global: torch.Tensor
    rope_sin_global: torch.Tensor
    rope_cos_local: torch.Tensor
    rope_sin_local: torch.Tensor
    layers: nn.ModuleList

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        assert args.vocab_size > 0

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleList(
            [TransformerBlock(args, i) for i in range(args.n_layers)]
        )

        self.norm = RMSNorm(args.dim, norm_eps=args.norm_eps)

        rope_len = args.max_seq_len
        g_cos, g_sin = precompute_rope_cache(
            args.head_dim, rope_len, theta=args.rope_theta
        )
        l_cos, l_sin = precompute_rope_cache(
            args.head_dim, rope_len, theta=args.rope_local_base_freq
        )
        self.register_buffer("rope_cos_global", g_cos, persistent=False)
        self.register_buffer("rope_sin_global", g_sin, persistent=False)
        self.register_buffer("rope_cos_local", l_cos, persistent=False)
        self.register_buffer("rope_sin_local", l_sin, persistent=False)

        self.use_cache = False
        self._local_mask_cache: dict = {}
        self._flex_block_mask = None
        self._flex_block_mask_key: Optional[tuple] = None

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reset_kv_caches(self) -> None:
        for layer in self.layers:
            block = cast(TransformerBlock, layer)
            block.attention.k_cache = None
            block.attention.v_cache = None

    def _get_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, device.type, device.index)
        mask = self._local_mask_cache.get(key)
        if mask is None:
            mask = build_local_bool_mask(seq_len, self.args.sliding_window, device)
            self._local_mask_cache[key] = mask
        return mask

    def _get_flex_block_mask(self, seq_len: int, device: torch.device):
        key = (seq_len, device.type, device.index)
        if self._flex_block_mask_key != key:
            self._flex_block_mask = build_flex_block_mask(
                seq_len, self.args.sliding_window, device
            )
            self._flex_block_mask_key = key
        return self._flex_block_mask

    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        _, seq_len = tokens.shape

        assert (
            seq_len <= self.args.max_seq_len
        ), f"Cannot forward sequence of length {seq_len}, max is {self.args.max_seq_len}"
        assert (
            start_pos + seq_len <= self.args.max_seq_len
        ), "RoPE cache / KV cache exceeded"

        h = self.tok_embeddings(tokens)
        normalizer = torch.tensor(self.args.dim**0.5, dtype=h.dtype)
        h = h * normalizer

        sl = slice(start_pos, start_pos + seq_len)
        rope = (
            cast(torch.Tensor, self.rope_cos_global)[sl],
            cast(torch.Tensor, self.rope_sin_global)[sl],
            cast(torch.Tensor, self.rope_cos_local)[sl],
            cast(torch.Tensor, self.rope_sin_local)[sl],
        )

        use_cache = self.use_cache and not self.training
        local_mask: Optional[torch.Tensor] = None
        local_block_mask = None
        if use_cache:
            if seq_len > 1:
                assert start_pos == 0, (
                    "KV-cache prefill must start at position 0; incremental "
                    "decode is single-token."
                )
                local_mask = self._get_local_mask(seq_len, h.device)
        else:
            dropout_active = self.training and self.args.dropout > 0.0
            if (
                HAS_FLEX
                and h.is_cuda
                and not dropout_active
                and self.args.use_flex_attention
            ):
                local_block_mask = self._get_flex_block_mask(seq_len, h.device)
            else:
                local_mask = self._get_local_mask(seq_len, h.device)

        attn_ctx = (local_mask, local_block_mask, use_cache)

        use_checkpoint = self.args.gradient_checkpointing and self.training
        for layer in self.layers:
            block = cast(TransformerBlock, layer)
            if use_checkpoint:
                h = checkpoint(block, h, start_pos, rope, attn_ctx, use_reentrant=False)
            else:
                h = block(h, start_pos, rope, attn_ctx)

        return self.norm(h)
