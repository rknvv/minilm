import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelArgs

logger = logging.getLogger()


class RMSNorm(nn.Module):
    """See: https://arxiv.org/pdf/1910.07467"""

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
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    n_dim = x.ndim
    assert 0 <= 1 < n_dim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [1] * n_dim
    shape[1] = x.shape[1]
    shape[-1] = x.shape[-1]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """See: https://arxiv.org/pdf/2104.09864"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
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


class Attention(nn.Module):
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

        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

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
            new_shape = max(batch_size, self.k_cache.shape[0] if self.k_cache else 0)
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
        self, x: torch.Tensor, start_pos: int, freq_cis: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freq_cis)

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
            # is_causal=(start_pos == 0),
            is_causal=is_causal,
        )

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        output = self.wo(attn_output)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.args = args
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attn_norm = RMSNorm(args.dim, norm_eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, norm_eps=args.norm_eps)

        self.dropout = nn.Dropout(args.dropout)

    def forward(
        self, x: torch.Tensor, start_pos: int, freq_cis: torch.Tensor
    ) -> torch.Tensor:
        attn_out = self.attention(self.attn_norm(x), start_pos, freq_cis)
        h = x + self.dropout(attn_out)

        ffn_out = self.feed_forward(self.ffn_norm(h))
        out = h + self.dropout(ffn_out)
        return out


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class MiniLM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        assert args.vocab_size > 0

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleList(
            [TransformerBlock(layer_id, args) for layer_id in range(args.n_layers)]
        )

        self.norm = RMSNorm(args.dim, norm_eps=args.norm_eps)

        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.output.weight = nn.Parameter(self.tok_embeddings.weight.data)

        self.tok_embeddings.weight = self.output.weight

        freqs_cis = precompute_freqs_cis(
            self.args.dim // self.args.n_heads, self.args.max_seq_len * 2
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

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

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, seq_len = tokens.shape
        assert (
            seq_len <= self.args.max_seq_len
        ), f"Cannot forward sequence of length {seq_len}, max is {self.args.max_seq_len}"
        assert (
            # start_pos + seq_len <= self.args.max_seq_len * 2
            start_pos + seq_len
            <= self.freqs_cis.shape[0]
        ), "Frequency buffer exceeded"

        h = self.tok_embeddings(tokens)

        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis)

        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.output(h[:, [-1], :])
            logits = logits.squeeze(1)
            loss = None

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int,
        eos_id: int,
        pad_id: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        self.eval()
        device = prompt_tokens.device
        batch_size, prompt_len = prompt_tokens.shape
        max_seq_len = self.args.max_seq_len

        # for block in self.layers:
        #     if block.attention.k_cache is None:
        #         block.attention._init_kv_cache(
        #             batch_size, device, self.tok_embeddings.weight.dtype
        #         )

        logger.info("Processing prompt...")
        _ = self(prompt_tokens, start_pos=0)
        logger.info("Prompt processing finished.")

        total_len = min(max_seq_len, prompt_len + max_new_tokens)
        tokens = torch.full(
            (batch_size, total_len), -1, dtype=torch.long, device=device
        )
        tokens[:, :prompt_len] = prompt_tokens

        eos_reached = torch.tensor([False] * batch_size, device=device)
        input_token = prompt_tokens[:, -1:]
        current_pos = prompt_len

        for _ in range(max_new_tokens):
            if current_pos >= max_seq_len:
                logger.warning(f"Max sequence length {max_seq_len} reached.")
                break

            logits, _ = self(input_token, start_pos=current_pos - 1)

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                if top_p > 0.0 and top_p < 1.0:  # Top p
                    next_token_val = sample_top_p(probs, top_p)
                elif top_k is not None and top_k > 0:  # Top k
                    v, _ = torch.topk(probs, k=min(top_k, probs.size(-1)))
                    probs[probs < v[:, [-1]]] = 0.0
                    probs.div_(probs.sum(dim=-1, keepdim=True))
                    next_token_val = torch.multinomial(probs, num_samples=1)
                else:  # Просто multinomial
                    next_token_val = torch.multinomial(probs, num_samples=1)
            else:  # Жадно
                next_token_val = torch.argmax(logits, dim=-1, keepdim=True)

            next_token = tokens[:, current_pos].clone()
            next_token = torch.where(
                eos_reached, next_token, next_token_val.squeeze(-1)
            )
            tokens[:, current_pos] = next_token

            eos_reached |= (~eos_reached) & (next_token == eos_id)

            input_token = next_token.unsqueeze(-1)
            current_pos += 1

            if eos_reached.all():
                logger.info("EOS token generated by all sequences in batch.")
                break

        for block in self.layers:
            block.attention.k_cache = None
            block.attention.v_cache = None

        final_tokens = []
        for i in range(batch_size):
            seq = tokens[i, :current_pos]
            eos_idx = torch.where(seq == eos_id)[0]
            if len(eos_idx) > 0:
                seq = seq[: eos_idx[0] + 1]
            final_tokens.append(seq)

        max_len_final = max(len(t) for t in final_tokens)
        final_padded_tokens = torch.full(
            (batch_size, max_len_final), pad_id, dtype=torch.long, device=device
        )
        for i, seq in enumerate(final_tokens):
            final_padded_tokens[i, : len(seq)] = seq

        self.train()
        return final_padded_tokens
