import time
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 16
    n_heads: int = 8
    n_kv_heads: Optional[int] = 8
    vocab_size: int = 8192
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 512
    flash_attn: bool = False
    dropout: float = 0.1


@dataclass
class TrainConfig:
    out_dir: str = "out"
    dataset_dir: str = "./data/pretrain"
    init_from: str = "scratch"
    resume_from_checkpoint: bool = True

    eval_interval: int = 10
    log_interval: int = 1
    eval_iters: int = 1
    always_save_checkpoint: bool = True
    wandb_log: bool = False
    wandb_project: str = "minilm-pretrain"
    wandb_run_name: str = "run"

    gradient_accumulation_steps: int = 1
    train_batch_size: int = 1
    eval_batch_size: int = 1
    context_length: int = 4

    num_workers: int = 4

    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000

    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = False
    backend: str = "nccl"

    eval_only: bool = False

    def __post_init__(self):
        if self.dtype == "bfloat16" and not (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        ):
            self.dtype = "float16"
        if self.dtype == "float16" and not torch.cuda.is_available():
            self.dtype = "float32"

        if self.wandb_run_name == "run":
            self.wandb_run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
