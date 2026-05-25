from typing import Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

class ModelArgs(BaseModel):

    model_config = ConfigDict(extra="forbid")

    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = 6
    vocab_size: int = 16384
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_seq_len: int = 1024
    dropout: float = 0.1

    gradient_checkpointing: bool = False

    ce_chunk_size: int = Field(default=0, ge=0)

    use_liger: bool = False

    @model_validator(mode="after")
    def _check_shapes(self) -> "ModelArgs":
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})")
        if self.n_kv_heads is not None and self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )

        if self.vocab_size > 65535:
            raise ValueError(f"vocab_size ({self.vocab_size}) must fit uint16 (<= 65535)")
        return self

class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Literal["pretrain", "sft"] = "pretrain"

    out_dir: str = "out"
    dataset_dir: str = "./data/pretrain"
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    pretrained_checkpoint: Optional[str] = None
    sft_ignore_idx: int = -100

    resume_from_checkpoint: bool = True
    resume_checkpoint_kind: Literal["latest", "best", "auto"] = "latest"

    eval_interval: int = Field(default=10, ge=1)
    log_interval: int = Field(default=1, ge=1)
    eval_iters: int = Field(default=1, ge=1)
    always_save_checkpoint: bool = True
    wandb_log: bool = False
    wandb_project: str = "minilm-pretrain"
    wandb_run_name: str = "run"

    gradient_accumulation_steps: int = Field(default=1, ge=1)
    train_batch_size: int = Field(default=1, ge=1)
    eval_batch_size: int = Field(default=1, ge=1)

    num_workers: int = Field(default=4, ge=0)

    learning_rate: float = 6e-4
    max_iters: int = Field(default=600000, ge=1)
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = Field(default=2000, ge=0)

    device: str = "cuda"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    compile: bool = False
    backend: str = "nccl"

    parallel: Literal["ddp", "fsdp2"] = "ddp"
    reshard_after_forward: bool = False
    ema: Optional[float] = None

    eval_only: bool = False

    @model_validator(mode="after")
    def _check_schedule(self) -> "TrainConfig":
        if self.decay_lr and self.warmup_iters > self.max_iters:
            raise ValueError(
                f"warmup_iters ({self.warmup_iters}) cannot exceed max_iters ({self.max_iters})"
            )
        if self.task == "sft" and not self.train_data_path:
            raise ValueError("task='sft' requires train_data_path")
        if self.ema is not None and not (0.0 < self.ema < 1.0):
            raise ValueError(f"ema decay must be in (0, 1), got {self.ema}")
        return self


def load_config(path: str) -> Tuple[TrainConfig, ModelArgs]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return TrainConfig(**cfg["train"]), ModelArgs(**cfg["model"])
