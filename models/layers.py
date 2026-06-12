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

    HAS_LIGER = True
except ImportError:
    liger_flce = None
    HAS_LIGER = False

try:
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
    )

    HAS_FLEX = True
except ImportError:
    create_block_mask = None
    flex_attention = None
    HAS_FLEX = False

_flex_attention_compiled = None


def _flex_sdpa(q, k, v, block_mask, scale: float, enable_gqa: bool):
    """flex_attention is only fast when compiled; compile once, lazily."""
    global _flex_attention_compiled
    if _flex_attention_compiled is None:
        _flex_attention_compiled = torch.compile(flex_attention, dynamic=False)
    return _flex_attention_compiled(
        q, k, v, block_mask=block_mask, scale=scale, enable_gqa=enable_gqa
    )


def sliding_window_causal_mask_mod(window: int):
    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx < window)

    return mask_mod


def build_flex_block_mask(seq_len: int, window: int, device: torch.device):
    """BlockMask for FlexAttention local layers: causal + sliding window."""
    assert create_block_mask is not None
    return create_block_mask(
        sliding_window_causal_mask_mod(window),
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,  # type: ignore[arg-type]
    )


def liger_enabled(args: ModelArgs, x: torch.Tensor) -> bool:
    return bool(getattr(args, "use_liger", False)) and HAS_LIGER and x.is_cuda


class RMSNorm(nn.Module):
    """Gemma-style RMSNorm: output = normed(x) * (1 + weight), computed in float32.

    The weight is zero-initialized so an untrained norm is the identity.
    """

    def __init__(self, dim: int, norm_eps: float = 1e-6):
        super().__init__()
        self.norm_eps = norm_eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class FeedForward(nn.Module):
    """Gemma-3 GeGLU MLP with gelu(tanh) activation."""

    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


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


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    B, T, kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x.unsqueeze(3)
        .expand(size=(B, T, kv_heads, n_rep, head_dim))
        .reshape(B, T, kv_heads * n_rep, head_dim)
    )


def build_local_bool_mask(
    seq_len: int, window: int, device: torch.device
) -> torch.Tensor:
    """Boolean [seq, seq] keep-mask for local layers: causal + sliding window.

    Position i attends to [max(0, i - window + 1), i]. Boolean masks are
    dtype-agnostic: SDPA converts them to an additive bias in the query dtype,
    so autocast/bf16/fp32 paths all stay consistent. Global layers never need
    a mask (is_causal=True covers them).
    """
    idx = torch.arange(seq_len, device=device)
    causal = idx[None, :] <= idx[:, None]
    return causal & (idx[None, :] > (idx[:, None] - window))


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

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert (
            self.n_heads % self.n_kv_heads == 0
        ), "n_heads must be divisible by n_kv_heads"
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim

        self.is_global = (layer_idx + 1) % args.sliding_window_pattern == 0
        self.sliding_window = args.sliding_window
        self.scale = args.query_pre_attn_scalar ** -0.5

        self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim, norm_eps=args.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, norm_eps=args.norm_eps)

        self.k_cache = None
        self.v_cache = None

    def _init_kv_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        current_batch_capacity = (
            self.k_cache.shape[0] if self.k_cache is not None else 0
        )
        if (
            self.k_cache is None
            or self.k_cache.shape[0] < batch_size
            or self.k_cache.device != device
            or self.k_cache.dtype != dtype
        ):
            new_batch = max(batch_size, current_batch_capacity)
            cache_shape = (
                new_batch,
                self.args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            )
            self.k_cache = torch.empty(cache_shape, device=device, dtype=dtype)
            self.v_cache = torch.empty_like(self.k_cache)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        local_mask: Optional[torch.Tensor],
        local_block_mask,
        use_cache: bool,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        xq = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # QK-norm (per head, float32) before RoPE.
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        if use_cache:
            if self.k_cache is None:
                self._init_kv_cache(batch_size, x.device, x.dtype)
            assert self.k_cache is not None and self.v_cache is not None

            self.k_cache[:batch_size, start_pos : start_pos + seq_len] = xk
            self.v_cache[:batch_size, start_pos : start_pos + seq_len] = xv

            end = start_pos + seq_len
            # Incremental decode (seq_len == 1): bound local layers to the
            # last `sliding_window` keys. Prefill attends to the full range;
            # the local_mask / is_causal enforce window and causality there.
            if seq_len == 1 and (not self.is_global) and end > self.sliding_window:
                lo = end - self.sliding_window
            else:
                lo = 0
            keys = self.k_cache[:batch_size, lo:end]
            values = self.v_cache[:batch_size, lo:end]
        else:
            keys, values = xk, xv

        # GQA: SDPA/flex broadcast KV heads natively on cuda/cpu; only exotic
        # backends need materialized repeat_kv.
        use_gqa = self.n_rep > 1 and x.device.type in ("cuda", "cpu")
        if self.n_rep > 1 and not use_gqa:
            keys = repeat_kv(keys, self.n_rep)
            values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        dropout_p = self.args.dropout if self.training else 0.0

        if (not use_cache) and (not self.is_global) and local_block_mask is not None:
            # Local layer fast path: FlexAttention with a block-sparse
            # sliding-window mask (real FLOP savings vs a dense [T, T] mask).
            attn_output = _flex_sdpa(
                xq, keys, values, local_block_mask, self.scale, self.n_rep > 1
            )
        else:
            if self.is_global:
                # Causal global attention. Prefill writes the cache from
                # position 0, so q_len == kv_len and is_causal is exact;
                # decode steps (seq_len == 1) attend to the whole cache.
                attn_mask = None
                is_causal = seq_len > 1
            else:
                attn_mask = local_mask if seq_len > 1 else None
                is_causal = False
            attn_output = F.scaled_dot_product_attention(
                xq,
                keys,
                values,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=self.scale,
                enable_gqa=use_gqa,
            )

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        return self.wo(attn_output)
