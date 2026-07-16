import logging
import math
from typing import Optional

import torch
from dion import Muon

from config import TrainConfig

logger = logging.getLogger(__name__)

MUON_MOMENTUM = 0.95


def build_optimizer(
    model: torch.nn.Module,
    train_cfg: TrainConfig,
    process_group=None,
) -> torch.optim.Optimizer:
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

    adamw_lr = train_cfg.learning_rate
    if train_cfg.muon_lr is not None:
        muon_lr = train_cfg.muon_lr
    else:
        muon_lr = adamw_lr
        logger.warning(
            "muon_lr is not set; Muon group falls back to learning_rate=%.2e. "
            "Muon typically wants ~10-20x the AdamW lr.",
            adamw_lr,
        )

    param_groups = [
        dict(params=hidden_matrix_params, algorithm="muon", lr=muon_lr),
        dict(params=embed_and_head_params, algorithm="adamw", lr=adamw_lr),
        dict(params=scalar_params, algorithm="adamw", lr=adamw_lr),
    ]
    return Muon(
        param_groups,
        distributed_mesh=process_group,
        lr=adamw_lr,
        mu=MUON_MOMENTUM,
        betas=(train_cfg.beta1, train_cfg.beta2),
        weight_decay=train_cfg.weight_decay,
        use_triton=train_cfg.muon_use_triton,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, train_cfg: TrainConfig
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    if not train_cfg.decay_lr:
        return None

    warmup = train_cfg.warmup_iters
    total = train_cfg.max_iters
    floor = train_cfg.min_lr_ratio

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
