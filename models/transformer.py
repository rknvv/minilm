import torch
import torch.nn as nn

from config import ModelArgs
from models.layers import Attention, FeedForward, RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            args=args,
        )
        self.attn_norm = RMSNorm(args.dim, norm_eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, norm_eps=args.norm_eps)

        self.dropout = nn.Dropout(args.dropout)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        attn_out = self.attention(self.attn_norm(x), start_pos, cos, sin)
        h = x + self.dropout(attn_out)
        ffn_out = self.feed_forward(self.ffn_norm(h))
        out = h + self.dropout(ffn_out)
        return out
