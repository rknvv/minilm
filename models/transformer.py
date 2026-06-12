from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from config import ModelArgs
from models.layers import Attention, FeedForward, RMSNorm


class TransformerBlock(nn.Module):
    """Gemma-3 decoder layer with sandwich normalization."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.attention = Attention(args, layer_idx)
        self.feed_forward = FeedForward(args.dim, args.intermediate_size)

        self.input_layernorm = RMSNorm(args.dim, norm_eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, norm_eps=args.norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(args.dim, norm_eps=args.norm_eps)
        self.post_feedforward_layernorm = RMSNorm(args.dim, norm_eps=args.norm_eps)

        self.is_global = self.attention.is_global

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        rope: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        attn_ctx: Tuple[Optional[torch.Tensor], Any, bool],
    ) -> torch.Tensor:
        cos_g, sin_g, cos_l, sin_l = rope
        local_mask, local_block_mask, use_cache = attn_ctx
        if self.is_global:
            cos, sin = cos_g, sin_g
        else:
            cos, sin = cos_l, sin_l

        residual = x
        x = self.input_layernorm(x)
        x = self.attention(
            x, start_pos, cos, sin, local_mask, local_block_mask, use_cache
        )
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.feed_forward(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x
        return x
