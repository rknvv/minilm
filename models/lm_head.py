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

        # Generation defaults (overridden by from_pretrained from config.json).
        self.bos_token_id: Optional[int] = None
        self.eos_token_id: Optional[int] = None
        self.pad_token_id: Optional[int] = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device | str] = None,
        **model_overrides,
    ) -> "MiniLMForCausalLM":
        """Build the model from a HF Gemma3 checkpoint and load its weights.

        Args:
            model_path: directory with config.json + model.safetensors, or a
                path to the .safetensors file (config.json must sit next to it).
            dtype: cast model to this dtype before loading (RoPE buffers are kept
                in float32 regardless, matching HF).
            device: move the model to this device after loading.
            **model_overrides: forwarded to ModelArgs.from_hf (e.g.
                max_seq_len=2048, gradient_checkpointing=True, use_liger=True).
        """
        import json
        import os

        from training.checkpoint import load_gemma_hf_weights

        if os.path.isdir(model_path):
            config_path = os.path.join(model_path, "config.json")
            weights_path = os.path.join(model_path, "model.safetensors")
        else:
            weights_path = model_path
            config_path = os.path.join(os.path.dirname(model_path), "config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found at: {config_path}")

        with open(config_path) as f:
            hf_config = json.load(f)

        args = ModelArgs.from_hf(hf_config, **model_overrides)
        model = cls(MiniLM(args), args)

        model.bos_token_id = hf_config.get("bos_token_id")
        eos = hf_config.get("eos_token_id")
        model.eos_token_id = eos[0] if isinstance(eos, (list, tuple)) else eos
        model.pad_token_id = hf_config.get("pad_token_id")

        if dtype is not None:
            model = model.to(dtype)
            # Keep RoPE caches in float32 for rotation precision (HF does too).
            inner = model.model
            for name in (
                "rope_cos_global",
                "rope_sin_global",
                "rope_cos_local",
                "rope_sin_local",
            ):
                setattr(inner, name, getattr(inner, name).float())

        load_gemma_hf_weights(model, weights_path)

        if device is not None:
            model = model.to(device)

        return model

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
        prompt_tokens,
        max_new_tokens: int = 64,
        eos_id: Optional[int] = None,
        pad_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        self.model.use_cache = True

        if eos_id is None:
            eos_id = self.eos_token_id
        if pad_id is None:
            pad_id = self.pad_token_id if self.pad_token_id is not None else 0
        if eos_id is None:
            raise ValueError("eos_id is required (no eos_token_id set on the model).")

        device = next(self.parameters()).device
        if not isinstance(prompt_tokens, torch.Tensor):
            prompt_tokens = torch.as_tensor(prompt_tokens, dtype=torch.long)
        prompt_tokens = prompt_tokens.to(device=device, dtype=torch.long)
        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)
        batch_size, prompt_len = prompt_tokens.shape
        max_seq_len = self.args.max_seq_len

        try:
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
        finally:
            self.model.use_cache = False
            self.model.reset_kv_caches()
            if was_training:
                self.train()

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

        return final_padded_tokens
