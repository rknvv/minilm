from typing import Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelArgs(BaseModel):

    model_config = ConfigDict(extra="forbid")

    # Gemma-3-1B defaults
    dim: int = 1152
    n_layers: int = 26
    n_heads: int = 4
    n_kv_heads: Optional[int] = 1
    head_dim: int = 256
    vocab_size: int = 153856  # pruned Gemma-3 vocab, 601 x 256
    intermediate_size: int = 6912
    norm_eps: float = 1e-6
    max_seq_len: int = 2048
    dropout: float = 0.0

    # Gemma-3 attention specifics
    query_pre_attn_scalar: float = 256.0
    rope_theta: float = 1_000_000.0
    rope_local_base_freq: float = 10_000.0
    sliding_window: int = 512
    sliding_window_pattern: int = 6

    gradient_checkpointing: bool = False

    ce_chunk_size: int = Field(default=8192, ge=0)

    use_liger: bool = True
    use_flex_attention: bool = True

    @model_validator(mode="after")
    def _check_shapes(self) -> "ModelArgs":
        if self.n_kv_heads is not None and self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        return self

    @classmethod
    def from_hf(cls, hf_config: dict, **overrides) -> "ModelArgs":
        """Build ModelArgs from a HF Gemma3 config.json dict."""
        args = dict(
            dim=hf_config["hidden_size"],
            n_layers=hf_config["num_hidden_layers"],
            n_heads=hf_config["num_attention_heads"],
            n_kv_heads=hf_config["num_key_value_heads"],
            head_dim=hf_config["head_dim"],
            vocab_size=hf_config["vocab_size"],
            intermediate_size=hf_config["intermediate_size"],
            norm_eps=hf_config["rms_norm_eps"],
            query_pre_attn_scalar=hf_config["query_pre_attn_scalar"],
            rope_theta=hf_config["rope_theta"],
            rope_local_base_freq=hf_config["rope_local_base_freq"],
            sliding_window=hf_config["sliding_window"],
            sliding_window_pattern=hf_config["sliding_window_pattern"],
        )
        args.update(overrides)
        return cls(**args)


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    out_dir: str = "out"
    dataset_dir: str = "./data/pretrain"
    pretrained_checkpoint: Optional[str] = None
    ignore_index: int = -100

    resume_from_checkpoint: bool = True
    resume_checkpoint_kind: Literal["latest", "best", "auto"] = "latest"

    seed: int = 42
    token_dtype: Literal["auto", "uint16", "uint32"] = "auto"

    eval_interval: int = Field(default=1000, ge=1)
    log_interval: int = Field(default=20, ge=1)
    eval_iters: int = Field(default=100, ge=1)
    always_save_checkpoint: bool = True
    wandb_log: bool = False
    wandb_project: str = "gemma3_1b_cpt_ruen"
    wandb_run_name: str = "run"

    gradient_accumulation_steps: int = Field(default=1, ge=1)
    train_batch_size: int = Field(default=1, ge=1)
    eval_batch_size: int = Field(default=1, ge=1)

    num_workers: int = Field(default=4, ge=0)

    learning_rate: float = 5e-5
    muon_lr: Optional[float] = 5e-4
    muon_use_triton: bool = False
    muon_distributed: bool = True
    max_iters: int = Field(default=100000, ge=1)
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = Field(default=2000, ge=0)
    min_lr_ratio: float = Field(default=0.1, ge=0.0, le=1.0)

    device: str = "cuda"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"

    ddp_bucket_cap_mb: Optional[int] = None
    compile: bool = False

    compile_mode: Optional[str] = None
    compile_backend: str = "inductor"
    backend: str = "nccl"

    profile: bool = False

    ema: Optional[float] = None

    eval_only: bool = False

    @model_validator(mode="after")
    def _check_schedule(self) -> "TrainConfig":
        if self.decay_lr and self.warmup_iters > self.max_iters:
            raise ValueError(
                f"warmup_iters ({self.warmup_iters}) cannot exceed max_iters ({self.max_iters})"
            )
        if self.ema is not None and not (0.0 < self.ema < 1.0):
            raise ValueError(f"ema decay must be in (0, 1), got {self.ema}")
        if self.muon_lr is not None and self.muon_lr <= 0:
            raise ValueError(f"muon_lr must be > 0, got {self.muon_lr}")
        return self


def load_config(path: str) -> Tuple[TrainConfig, ModelArgs]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return TrainConfig(**cfg["train"]), ModelArgs(**cfg["model"])
