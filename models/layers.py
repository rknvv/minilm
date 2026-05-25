import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelArgs

logger = logging.getLogger()

try:
    from liger_kernel.transformers.functional import (
        liger_fused_linear_cross_entropy as liger_flce,
    )
    from liger_kernel.transformers.swiglu import LigerSiLUMulFunction

    HAS_LIGER = True
except ImportError:
    liger_flce = None
    LigerSiLUMulFunction = None
    HAS_LIGER = False

def liger_enabled(args: ModelArgs, x: torch.Tensor) -> bool:

    return bool(getattr(args, "use_liger", False)) and HAS_LIGER and x.is_cuda

class RMSNorm(nn.Module):

    def __init__(self, dim, norm_eps=1e-6):
        super().__init__()
        self.norm_eps = norm_eps
        self.weights = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.norm_eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weights

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        args: Optional[ModelArgs] = None,
    ):
        super().__init__()
        self.args = args
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        if self.args is not None and liger_enabled(self.args, x):

            return self.w2(LigerSiLUMulFunction.apply(self.w1(x), self.w3(x)))
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

def precompute_rope_cache(
    dim: int, end: int, theta: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:

    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    xq_f = xq.float()
    xk_f = xk.float()
    xq_out = xq_f * cos + rotate_half(xq_f) * sin
    xk_out = xk_f * cos + rotate_half(xk_f) * sin
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x, n_rep):
    B, T, kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            x.unsqueeze(3)
            .expand(size=(B, T, kv_heads, n_rep, head_dim))
            .reshape(B, T, kv_heads * n_rep, head_dim)
        )

def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

class Attention(nn.Module):
    k_cache: Optional[torch.Tensor]
    v_cache: Optional[torch.Tensor]

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert (
            self.n_heads % self.n_kv_heads == 0
        ), "n_heads must be divisible by n_kv_heads"
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.k_cache = None
        self.v_cache = None

    def _init_kv_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        cache_shape = (
            batch_size,
            self.args.max_seq_len,
            self.n_kv_heads,
            self.head_dim,
        )

        if (
            self.k_cache is None
            or self.k_cache.shape[0] < batch_size
            or self.k_cache.device != device
            or self.k_cache.dtype != dtype
        ):
            current_batch_capacity = (
                self.k_cache.shape[0] if self.k_cache is not None else 0
            )
            new_shape = max(batch_size, current_batch_capacity)
            cache_shape = (
                new_shape,
                self.args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            )

            self.k_cache = torch.empty(
                cache_shape,
                device=device,
                dtype=dtype,
            )
            self.v_cache = torch.empty_like(self.k_cache)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        if not self.training:
            if self.k_cache is None:
                self._init_kv_cache(batch_size, x.device, x.dtype)
            assert self.k_cache is not None and self.v_cache is not None

            self.k_cache[:batch_size, start_pos : start_pos + seq_len] = xk
            self.v_cache[:batch_size, start_pos : start_pos + seq_len] = xv

            keys = self.k_cache[:batch_size, : start_pos + seq_len]
            values = self.v_cache[:batch_size, : start_pos + seq_len]
        else:
            keys, values = xk, xv

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        dropout_p = self.args.dropout if self.training else 0.0

        is_causal = (self.training or start_pos == 0) and seq_len > 1
        attn_output = F.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        output = self.wo(attn_output)

        return output
