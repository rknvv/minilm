import os
import random
import datetime
import argparse
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


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MiniLM Model using argparse")
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--resume_from_checkpoint", type=str2bool)
    parser.add_argument("--eval_interval", type=int)
    parser.add_argument("--log_interval", type=int)
    parser.add_argument("--eval_iters", type=int)

    parser.add_argument("--always_save_checkpoint", type=str2bool)

    parser.add_argument("--wandb_log", type=str2bool)

    parser.add_argument("--wandb_project", type=str)
    default_wandb_run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=default_wandb_run_name,
    )

    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--context_length", type=int)
    parser.add_argument("--num_workers", type=int)

    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--grad_clip", type=float)
    parser.add_argument("--decay_lr", type=str2bool)

    parser.add_argument("--warmup_iters", type=int)
    parser.add_argument("--lr_decay_iters", type=int)
    parser.add_argument("--max_iters", type=int)

    parser.add_argument("--device", type=str)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16", "float16"],
    )

    parser.add_argument("--compile", type=str2bool)

    parser.add_argument("--backend", type=str)

    parser.add_argument("--eval_only", type=str2bool)
    parser.set_defaults(eval_only=False)

    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--model_n_layers", type=int, default=16)
    parser.add_argument("--model_n_heads", type=int, default=8)
    parser.add_argument("--model_n_kv_heads", type=int, default=8)
    parser.add_argument(
        "--model_vocab_size",
        type=int,
        default=8192,
        dest="vocab_size",
    )
    parser.add_argument(
        "--model_multiple_of",
        type=int,
        default=256,
        dest="multiple_of",
    )

    parser.add_argument(
        "--model_ffn_dim_multiplier",
        type=float,
        default=None,
        dest="ffn_dim_multiplier",
    )
    parser.add_argument(
        "--model_norm_eps",
        type=float,
        default=1e-5,
        dest="norm_eps",
    )

    parser.add_argument(
        "--model_max_batch_size",
        type=int,
        default=32,
        dest="max_batch_size",
    )
    parser.add_argument(
        "--model_max_seq_len",
        type=int,
        default=512,
        dest="max_seq_len",
    )

    parser.add_argument("--flash_attn", type=str2bool)

    parser.add_argument(
        "--model_dropout",
        type=float,
        default=0.1,
        dest="dropout",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    train_config_arg_names = [
        "out_dir",
        "dataset_dir",
        "resume_from_checkpoint",
        "eval_interval",
        "log_interval",
        "eval_iters",
        "always_save_checkpoint",
        "wandb_log",
        "wandb_project",
        "wandb_run_name",
        "gradient_accumulation_steps",
        "train_batch_size",
        "eval_batch_size",
        "context_length",
        "num_workers",
        "learning_rate",
        "max_iters",
        "weight_decay",
        "beta1",
        "beta2",
        "grad_clip",
        "decay_lr",
        "warmup_iters",
        "lr_decay_iters",
        "device",
        "dtype",
        "compile",
        "backend",
        "eval_only",
    ]
    model_config_arg_names = [
        "dim",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "vocab_size",
        "multiple_of",
        "ffn_dim_multiplier",
        "norm_eps",
        "max_batch_size",
        "max_seq_len",
        "flash_attn",
        "dropout",
    ]

    train_kwargs = {
        name: getattr(args, name)
        for name in train_config_arg_names
        if hasattr(args, name)
    }
    model_kwargs = {
        name: getattr(args, name)
        for name in model_config_arg_names
        if hasattr(args, name)
    }

    train_cfg = TrainConfig(**train_kwargs)
    model_args = ModelArgs(**model_kwargs)
    # for field in fields(TrainConfig):
    #     if hasattr(args, field.name):
    #         setattr(train_cfg, field.name, getattr(args, field.name))
    # for field in fields(ModelArgs):
    #     if hasattr(args, field.name):
    #         setattr(model_args, field.name, getattr(args, field.name))

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

    if master_process:
        log.info(f"Initializing model: MiniLM with vocab_size={model_args.vocab_size}")
    model = MiniLM(model_args)

    if train_cfg.compile:
        if master_process:
            log.info("Compiling the model...")
        model = torch.compile(model)

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=(train_cfg.beta1, train_cfg.beta2),
        eps=1e-8,
    )

    lr_scheduler = None
    if train_cfg.decay_lr:
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=train_cfg.warmup_iters,
            num_training_steps=train_cfg.max_iters,
        )

    if master_process:
        log.info("Initializing trainer...")

    print(train_cfg.device)
    trainer = Trainer(
        cfg=train_cfg,
        model_args=model_args,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        lr_scheduler=lr_scheduler,
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


if __name__ == "__main__":
    main()
