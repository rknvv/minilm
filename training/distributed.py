import logging
import os
import random
from typing import NamedTuple

import numpy as np
import torch
import torch.distributed as dist

from config import TrainConfig

logger = logging.getLogger(__name__)


class DistInfo(NamedTuple):
    rank: int
    local_rank: int
    world_size: int
    device: str
    master: bool
    seed: int


def setup_distributed(cfg: TrainConfig) -> DistInfo:
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available()
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=cfg.backend, device_id=torch.device(device)
        )
        master = rank == 0
        seed_offset = rank
        logger.info(f"DDP enabled. Rank {rank}/{world_size} on device {device}")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        master = True
        seed_offset = 0
        if cfg.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        elif cfg.device == "mps" and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"DDP not enabled. Running on device {device}")

    cfg.device = device

    seed = cfg.seed + seed_offset
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return DistInfo(rank, local_rank, world_size, device, master, seed)


def cleanup_distributed(info: DistInfo) -> None:
    if info.world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
