import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import ModelArgs
from models.minilm import MiniLM
from models.layers import liger_enabled, liger_flce, sample_top_p

logger = logging.getLogger()

class MiniLMForCausalLM(nn.Module):
    def __init__(self, model: MiniLM, args: ModelArgs):
        super().__init__()
        self.model = model
        self.args = args

        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.output.weight = self.model.tok_embeddings.weight

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        targets: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        h = self.model(tokens, start_pos)

        if targets is not None:
            if liger_enabled(self.args, h):

                loss = liger_flce(
                    h.reshape(-1, h.size(-1)),
                    self.output.weight,
                    targets.reshape(-1),
                    ignore_index=ignore_index,
                )
                return None, loss
            if self.args.ce_chunk_size and self.args.ce_chunk_size > 0:

                loss = self._chunked_cross_entropy(h, targets, ignore_index)
                return None, loss
            logits = self.output(h)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=ignore_index,
            )
            return logits, loss

        logits = self.output(h[:, [-1], :])
        logits = logits.squeeze(1)
        return logits, None

    def _chunked_cross_entropy(
        self,
        h: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int,
    ) -> torch.Tensor:

        chunk_size = self.args.ce_chunk_size
        weight = self.output.weight

        h_flat = h.reshape(-1, h.size(-1))
        t_flat = targets.reshape(-1)

        def chunk_loss_sum(h_chunk: torch.Tensor, t_chunk: torch.Tensor) -> torch.Tensor:
            logits_chunk = F.linear(h_chunk, weight).float()
            return F.cross_entropy(
                logits_chunk,
                t_chunk,
                ignore_index=ignore_index,
                reduction="sum",
            )

        total_loss = h_flat.new_zeros((), dtype=torch.float32)
        for h_chunk, t_chunk in zip(
            h_flat.split(chunk_size), t_flat.split(chunk_size)
        ):
            if self.training:
                loss_sum = checkpoint(
                    chunk_loss_sum, h_chunk, t_chunk, use_reentrant=False
                )
            else:
                loss_sum = chunk_loss_sum(h_chunk, t_chunk)
            total_loss = total_loss + loss_sum

        n_valid = (t_flat != ignore_index).sum().clamp(min=1)
        return total_loss / n_valid

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
        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)
        device = prompt_tokens.device
        batch_size, prompt_len = prompt_tokens.shape
        max_seq_len = self.args.max_seq_len

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
                if top_p > 0.0 and top_p < 1.0:
                    next_token_val = sample_top_p(probs, top_p)
                elif top_k is not None and top_k > 0:
                    v, _ = torch.topk(probs, k=min(top_k, probs.size(-1)))
                    probs[probs < v[:, [-1]]] = 0.0
                    probs.div_(probs.sum(dim=-1, keepdim=True))
                    next_token_val = torch.multinomial(probs, num_samples=1)
                else:
                    next_token_val = torch.multinomial(probs, num_samples=1)
            else:
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

        for layer in self.model.layers:
            layer.attention.k_cache = None
            layer.attention.v_cache = None

        final_tokens = []
        for i in range(batch_size):
            seq = tokens[i, :current_pos]
            generated = seq[prompt_len:]
            eos_idx = torch.where(generated == eos_id)[0]
            if len(eos_idx) > 0:
                seq = seq[: prompt_len + eos_idx[0] + 1]
            final_tokens.append(seq)

        max_len_final = max(len(t) for t in final_tokens)
        final_padded_tokens = torch.full(
            (batch_size, max_len_final), pad_id, dtype=torch.long, device=device
        )
        for i, seq in enumerate(final_tokens):
            final_padded_tokens[i, : len(seq)] = seq

        self.train()
        return final_padded_tokens
