import logging
from typing import Optional

import torch
from transformers import get_cosine_schedule_with_warmup

from config import TrainConfig

logger = logging.getLogger(__name__)

MUON_MOMENTUM = 0.95


def build_optimizer(
    model: torch.nn.Module,
    train_cfg: TrainConfig,
    fsdp_mesh=None,
) -> torch.optim.Optimizer:
    from dion import Muon

    hidden_matrix_params = []
    embed_and_head_params = []
    scalar_params = []

    for n, p in model.named_parameters():
        if p.dim() < 2:
            scalar_params.append(p)
        elif "tok_embeddings" in n or "output" in n:
            embed_and_head_params.append(p)
        else:
            hidden_matrix_params.append(p)

    seen_ids: dict[int, str] = {}
    for group_name, group_params in (
        ("hidden_matrix", hidden_matrix_params),
        ("embed_and_head", embed_and_head_params),
        ("scalar", scalar_params),
    ):
        for p in group_params:
            assert id(p) not in seen_ids, (
                f"Parameter shared between optimizer groups "
                f"'{seen_ids.get(id(p))}' and '{group_name}'"
            )
            seen_ids[id(p)] = group_name

    param_groups = [
        dict(params=hidden_matrix_params, algorithm="muon"),
        dict(params=embed_and_head_params, algorithm="adamw"),
        dict(params=scalar_params, algorithm="adamw"),
    ]
    return Muon(
        param_groups,
        lr=train_cfg.learning_rate,
        mu=MUON_MOMENTUM,
        betas=(train_cfg.beta1, train_cfg.beta2),
        weight_decay=train_cfg.weight_decay,
        distributed_mesh=fsdp_mesh,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, train_cfg: TrainConfig
) -> Optional[object]:
    if not train_cfg.decay_lr:
        return None
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_cfg.warmup_iters,
        num_training_steps=train_cfg.max_iters,
    )
