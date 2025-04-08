import os
import random
import yaml
import fire
import logging
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from src.model import MiniLM
from src.config import ModelArgs, TrainConfig
from src.dataset import MemMapDatasetForLM
from src.trainer import Trainer


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def setup_ddp(cfg: TrainConfig):
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available()
        dist.init_process_group(backend=cfg.backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        log.info(f"DDP enabled. Rank {ddp_rank}/{ddp_world_size} on device {device}")
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        if cfg.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        elif cfg.device == "mps" and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        log.info(f"DDP not enabled. Running on device {device}")

    cfg.device = device
    cfg.ddp_rank = ddp_rank
    cfg.ddp_world_size = ddp_world_size
    cfg.master_process = master_process

    seed = 42 + seed_offset
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return ddp_rank, ddp_world_size, device, master_process


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    print(cfg["train"])

    train_cfg = TrainConfig(**cfg["train"])
    model_args = ModelArgs(**cfg["model"])
    return train_cfg, model_args


def train_model(yaml_path=None):
    train_cfg, model_args = load_config(yaml_path)

    ddp_rank, ddp_world_size, device, master_process = setup_ddp(train_cfg)

    if master_process:
        log.info("--- Training Configuration ---")
        log.info(f"Train Config: {asdict(train_cfg)}")
        log.info("-----------------------------")
        log.info("---- Model Configuration ----")
        log.info(f"Model Args: {asdict(model_args)}")
        log.info("-----------------------------")

    if master_process:
        log.info("Setting up datasets and dataloaders...")

    train_dataset = MemMapDatasetForLM(
        os.path.join(train_cfg.dataset_dir, "train.bin"),
        chunk_size=model_args.max_seq_len,
        memmap_dtype=np.uint16,
    )
    eval_dataset = MemMapDatasetForLM(
        os.path.join(train_cfg.dataset_dir, "val.bin"),
        chunk_size=model_args.max_seq_len,
        memmap_dtype=np.uint16,
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
    )
    eval_sampler = DistributedSampler(
        eval_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.train_batch_size,
        sampler=train_sampler,
        num_workers=train_cfg.num_workers,
        pin_memory=True if device.startswith("cuda") else False,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=train_cfg.eval_batch_size,
        sampler=eval_sampler,
        num_workers=train_cfg.num_workers,
        pin_memory=True if device.startswith("cuda") else False,
    )

    if master_process:
        log.info("Datasets and dataloaders are ready.")

    model = MiniLM(model_args)
    if master_process:
        log.info(
            f"Initializing model: MiniLM with vocab_size={model_args.vocab_size} and {sum(p.numel() for p in model.parameters())} parameters."
        )

    if train_cfg.compile:
        if master_process:
            log.info("Compiling the model...")
        model = torch.compile(model)
        dummy_input = torch.randn(1, 1)
        model(dummy_input)
        if master_process:
            log.info("Compiling complete.")

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=(train_cfg.beta1, train_cfg.beta2),
        eps=1e-8,
    )

    scheduler = None
    if train_cfg.decay_lr:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=train_cfg.warmup_iters,
            num_training_steps=train_cfg.max_iters,
        )

    if master_process:
        log.info("Initializing trainer...")

    trainer = Trainer(
        train_cfg=train_cfg,
        model_cfg=model_args,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        scheduler=scheduler,
        eval_loader=eval_loader,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        master_process=master_process,
        resume_from_checkpoint=train_cfg.resume_from_checkpoint,
    )

    try:
        trainer.train()
    finally:
        if ddp_world_size > 1:
            dist.destroy_process_group()


def main():
    fire.Fire(train_model)


if __name__ == "__main__":
    main()
