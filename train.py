import os
import random
import argparse

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.dataset import IterableDataset
from src.model import MiniLM
from src.config import ModelArgs, TrainConfig
from src.trainer import Trainer


def setup_ddp(cfg: TrainConfig):
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        dist.init_process_group(backend=cfg.backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank

        assert (
            cfg.gradient_accumulation_steps % ddp_world_size == 0
        ), "gradient_accumulation_steps must be divisible by DDP world size"
        gradient_accumulation_steps_this_rank = (
            cfg.gradient_accumulation_steps // ddp_world_size
        )
        print(
            f"Rank {ddp_rank}: Grad accum steps per rank: {gradient_accumulation_steps_this_rank}"
        )
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = cfg.device if torch.cuda.is_available() else "cpu"
        master_process = True
        seed_offset = 0
        gradient_accumulation_steps_this_rank = cfg.gradient_accumulation_steps

    cfg.world_size = ddp_world_size
    cfg.master_process = master_process
    cfg.gradient_accumulation_steps_this_rank = gradient_accumulation_steps_this_rank
    cfg.effective_batch_size = cfg.batch_size * cfg.gradient_accumulation_steps
    cfg.tokens_per_iter = cfg.effective_batch_size * cfg.context_length * cfg.world_size

    random.seed(42 + seed_offset)
    np.random.seed(42 + seed_offset)
    torch.manual_seed(42 + seed_offset)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + seed_offset)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return ddp_rank, ddp_local_rank, ddp_world_size, device, master_process


def main():
    ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = setup_ddp()
