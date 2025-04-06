# Reference: https://github.com/allenai/OLMo/blob/main/olmo/train.py

import time
import logging
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from .config import TrainConfig, ModelArgs

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        model_args: ModelArgs,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        lr_scheduler: Optional[_LRScheduler] = None,
        eval_loader: Optional[DataLoader] = None,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        master_process: bool = True,
        resume_from_checkpoint: bool = True,
    ):
        self.cfg = cfg
        self.model_args = model_args
        self.model = model.to(cfg.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.lr_scheduler = lr_scheduler
        self.device = cfg.device
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.master_process = master_process
        self.resume_from_checkpoint = resume_from_checkpoint

        if self.ddp_world_size > 1:
            self.model = self.model.to(f"cuda:{self.ddp_rank}")
            self.model = DDP(self.model, device_ids=[self.ddp_rank])
            log.info(f"Rank {self.ddp_rank}: Wrapped model with DDP.")

        self.ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }.get(cfg.dtype, torch.float32)
        if self.ptdtype != torch.float32 and self.device == "cpu":
            log.warning(
                f"AMP ({cfg.dtype}) is not well supported on CPU. Using float32 context."
            )
            self.ctx = nullcontext()
        else:
            self.ctx = torch.amp.autocast(device_type=self.device, dtype=self.ptdtype)

        self.scaler = torch.amp.GradScaler(enabled=(cfg.dtype == "float16"))

        self.cfg.out_dir = Path(cfg.out_dir)
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.cfg.out_dir / "ckpt.pt"
        self.best_checkpoint_path = self.cfg.out_dir / "ckpt_best.pt"

        self.global_step = 0
        self.tokens_seen = 0
        self.best_eval_loss = float("inf")
        self.running_loss = 0.0
        # self.iter_start_time = time.time()

        if self.resume_from_checkpoint:
            self.load_checkpoint()

        self.wandb_run = None
        if cfg.wandb_log and self.master_process:
            self._init_wandb()

        self.gradient_accumulation_steps_this_rank = (
            self.cfg.gradient_accumulation_steps
        )

        if self.master_process:
            log.info(f"Trainer initialized on rank {self.ddp_rank}.")
            log.info(f"Using device: {self.device}")
            log.info(f"Data type for AMP: {self.cfg.dtype} ({self.ptdtype})")
            log.info(
                f"Gradient Accumulation Steps: {self.gradient_accumulation_steps_this_rank}"
            )
            log.info(f"Max iterations: {self.cfg.max_iters}")
            log.info(f"Saving checkpoints to: {self.cfg.out_dir}")

    def _init_wandb(self):
        """Initializes Weights & Biases."""
        if self.wandb_run is not None:
            return
        try:
            import wandb

            config_dict = vars(self.cfg)
            config_dict.update(
                {f"model_{k}": v for k, v in vars(self.model_args).items()}
            )

            self.wandb_run = wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_run_name,
                config=config_dict,
                resume="allow",
                id=(wandb.util.generate_id() if self.global_step == 0 else None),
            )
            log.info(
                f"WandB initialized for run: {self.wandb_run.name} (ID: {self.wandb_run.id})"
            )
        except ImportError:
            log.warning("WandB requested but not installed. Skipping.")
            self.cfg.wandb_log = False
        except Exception as e:
            log.error(f"Failed to initialize WandB: {e}")
            self.cfg.wandb_log = False

    @property
    def max_steps(self) -> int:
        return self.cfg.max_iters

    def _get_model_module(self) -> torch.nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _train_step(self, batch):
        inputs, targets = batch.values()
        inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(
            self.device, non_blocking=True
        )

        with self.ctx:
            model_module = self._get_model_module()
            _, loss = model_module(inputs, targets=targets)

            if self.gradient_accumulation_steps_this_rank > 1:
                loss = loss / self.gradient_accumulation_steps_this_rank

        if not torch.isfinite(loss).all():
            log.error(
                f"Rank {self.ddp_rank}: Non-finite loss detected: {loss.item()}! Skipping backward/step."
            )
            return torch.tensor(float("nan"), device=self.device)

        self.scaler.scale(loss).backward()

        return loss.detach() * self.gradient_accumulation_steps_this_rank

    def train(self):
        if self.master_process:
            log.info(
                f"Starting training from global step {self.global_step} for {self.cfg.max_iters} iterations."
            )
            eff_batch_size = (
                self.cfg.train_batch_size
                * self.cfg.gradient_accumulation_steps
                * self.ddp_world_size
            )
            log.info(f"Effective batch size: {eff_batch_size}")

        train_iter = iter(self.train_loader)
        self.model.train()
        self.running_loss = 0.0
        self.iter_start_time = time.time()

        while self.global_step < self.cfg.max_iters:
            is_last_accumulation_step = (
                self.global_step + 1
            ) % self.gradient_accumulation_steps_this_rank == 0
            sync_context = (
                self.model.no_sync()
                if (self.ddp_world_size > 1 and not is_last_accumulation_step)
                else nullcontext()
            )

            try:
                batch = next(train_iter)
            except StopIteration:
                log.info("Training dataloader iterator exhausted. Resetting.")
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            with sync_context:
                micro_step_loss = self._train_step(batch)

            if torch.isnan(micro_step_loss):
                log.warning(
                    f"Skipping optimizer step at global_step {self.global_step} due to non-finite loss."
                )
                self.running_loss = 0.0
            else:
                self.running_loss += micro_step_loss.item()

            if is_last_accumulation_step:
                if self.cfg.grad_clip > 0.0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self._get_model_module().parameters(), self.cfg.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                if (
                    self.global_step > 0
                    and self.global_step % self.cfg.log_interval == 0
                    and self.master_process
                ):
                    iter_time_elapsed = time.time() - self.iter_start_time
                    iters_since_last_log = (
                        self.cfg.log_interval
                        * self.gradient_accumulation_steps_this_rank
                    )
                    avg_iter_time_ms = (iter_time_elapsed / iters_since_last_log) * 1000

                    avg_loss = self.running_loss / iters_since_last_log
                    current_lr = self.optimizer.param_groups[0]["lr"]

                    log_msg = (
                        f"Iter {self.global_step}/{self.cfg.max_iters} | "
                        f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                        f"Avg Iter Time: {avg_iter_time_ms:.2f}ms"
                    )
                    log.info(log_msg)

                    if self.wandb_run:
                        metrics = {
                            "train/loss": avg_loss,
                            "train/learning_rate": current_lr,
                            "system/avg_iter_time_ms": avg_iter_time_ms,
                        }
                        if self.tokens_seen > 0:
                            metrics["train/tokens_seen"] = self.tokens_seen
                        self.wandb_run.log(metrics, step=self.global_step)

                    self.running_loss = 0.0
                    self.iter_start_time = time.time()

                if (
                    self.global_step > 0
                    and self.global_step % self.cfg.eval_interval == 0
                ):
                    eval_loss = self.evaluate()
                    if self.master_process:
                        log.info(
                            f"Validation Loss at step {self.global_step}: {eval_loss:.4f}"
                        )
                        if self.wandb_run:
                            self.wandb_run.log(
                                {"eval/loss": eval_loss}, step=self.global_step
                            )

                        is_best = eval_loss < self.best_eval_loss
                        if is_best:
                            self.best_eval_loss = eval_loss
                            log.info(
                                f"New best validation loss: {self.best_eval_loss:.4f}"
                            )

                        if self.cfg.always_save_checkpoint or is_best:
                            self.save_checkpoint(is_best=is_best)

                    if self.ddp_world_size > 1:
                        dist.barrier()

                    if self.cfg.eval_only:
                        log.info("Evaluation finished (eval_only=True). Exiting.")
                        self.cleanup()
                        return

                self.global_step += 1

            if self.global_step >= self.cfg.max_iters:
                log.info(
                    f"Reached maximum iterations ({self.cfg.max_iters}). Finishing training."
                )
                break

        log.info("Training loop finished.")
        self.cleanup()

    @torch.no_grad()
    def evaluate(self) -> float:
        if self.eval_loader is None:
            log.warning(
                "Evaluation requested but eval_loader not provided. Returning NaN."
            )
            return float("nan")

        model_module = self._get_model_module()
        was_training = model_module.training
        model_module.eval()

        if self.master_process:
            log.info(f"Starting evaluation for {self.cfg.eval_iters} iterations...")

        local_total_loss = 0.0
        local_total_tokens = 0
        evaluated_iters = 0

        eval_iter = iter(self.eval_loader)
        for i in range(self.cfg.eval_iters):
            try:
                batch = next(eval_iter)
            except StopIteration:
                if self.master_process:
                    log.warning(f"Evaluation dataset exhausted after {i} iterations.")
                break

            inputs, targets = batch.values()
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(
                self.device, non_blocking=True
            )

            with self.ctx:
                _, loss = model_module(inputs, targets=targets)

            if not torch.isfinite(loss).all():
                log.warning(
                    f"Rank {self.ddp_rank}: Non-finite loss detected during evaluation: {loss.item()}. Skipping batch."
                )
                continue

            batch_tokens = inputs.numel()
            local_total_loss += loss.item() * batch_tokens
            local_total_tokens += batch_tokens
            evaluated_iters += 1

        if self.ddp_world_size > 1:
            loss_sum_tensor = torch.tensor(
                [local_total_loss], dtype=torch.float64, device=self.device
            )
            tokens_sum_tensor = torch.tensor(
                [local_total_tokens], dtype=torch.long, device=self.device
            )

            dist.all_reduce(loss_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(tokens_sum_tensor, op=dist.ReduceOp.SUM)

            global_total_loss = loss_sum_tensor.item()
            global_total_tokens = tokens_sum_tensor.item()
        else:
            global_total_loss = local_total_loss
            global_total_tokens = local_total_tokens

        if was_training:
            model_module.train()

        if global_total_tokens == 0:
            log.error("No tokens were processed during evaluation. Returning NaN.")
            return float("nan")

        avg_loss = global_total_loss / global_total_tokens
        try:
            perplexity = math.exp(avg_loss)
            if self.master_process:
                log.info(
                    f"Evaluation complete: Avg Loss = {avg_loss:.4f}, Perplexity = {perplexity:.4f} (over {global_total_tokens} tokens, {evaluated_iters} iters)"
                )
        except OverflowError:
            perplexity = float("inf")
            if self.master_process:
                log.warning(
                    f"Evaluation complete: Avg Loss = {avg_loss:.4f} (Perplexity overflowed)"
                )
        return avg_loss

    def save_checkpoint(self, is_best: bool = False):
        """Saves model, optimizer, scheduler, and trainer state."""
        if not self.master_process:
            return

        log.info(f"Saving checkpoint at step {self.global_step}...")
        save_path = self.best_checkpoint_path if is_best else self.checkpoint_path

        model_to_save = self._get_model_module()

        checkpoint = {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "best_eval_loss": self.best_eval_loss,
            "config": {
                "batch_size": self.cfg.train_batch_size,
                "learning_rate": self.cfg.learning_rate,
                "device": self.cfg.device,
                "dtype": self.cfg.dtype,
                "max_iters": self.cfg.max_iters,
            },
        }

        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        temp_save_path = save_path.with_suffix(".tmp")

        try:
            torch.save(checkpoint, temp_save_path)
            temp_save_path.rename(save_path)
            log.info(f"Checkpoint saved successfully to {save_path}")

            if is_best:
                log.info(f"This is a new best model (loss: {self.best_eval_loss:.4f})!")

            if self.wandb_run:
                self.wandb_run.summary["best_eval_loss"] = self.best_eval_loss
                if is_best:
                    pass

        except Exception as e:
            log.error(f"Failed to save checkpoint: {e}")
            if temp_save_path.exists():
                try:
                    temp_save_path.unlink()
                except:
                    pass

    def load_checkpoint(self, load_path: Optional[Path] = None):
        path_to_load = None

        if load_path is not None:
            if load_path.exists():
                path_to_load = load_path
                log.info(
                    f"Attempting to load checkpoint from specified path: {path_to_load}"
                )
            else:
                log.warning(
                    f"Specified checkpoint file not found at {load_path}. Cannot load."
                )
                return
        else:
            if self.best_checkpoint_path.exists():
                path_to_load = self.best_checkpoint_path
                log.info(
                    f"Found best checkpoint. Attempting to load from: {path_to_load}"
                )
            elif self.checkpoint_path.exists():
                path_to_load = self.checkpoint_path
                log.info(
                    f"Best checkpoint ({self.best_checkpoint_path.name}) not found. "
                    f"Attempting to load from regular checkpoint: {path_to_load}"
                )
            else:
                log.warning(
                    f"No checkpoint file found at '{self.best_checkpoint_path}' or "
                    f"'{self.checkpoint_path}'. Starting training from scratch."
                )
                return

        if path_to_load is None:
            log.error(
                "Internal error: path_to_load is None despite checks. Cannot load checkpoint."
            )
            return

        log.info(f"Loading checkpoint from {path_to_load}...")
        map_location = "cpu"

        try:
            checkpoint = torch.load(path_to_load, map_location=map_location)

            model_state = checkpoint["model_state_dict"]
            current_is_ddp = isinstance(self.model, DDP)
            checkpoint_was_ddp = any(
                k.startswith("module.") for k in model_state.keys()
            )

            model_to_load = self._get_model_module()

            if current_is_ddp and not checkpoint_was_ddp:
                log.info(
                    "Current model is DDP, checkpoint is not. Adding 'module.' prefix to keys."
                )
                model_state = {f"module.{k}": v for k, v in model_state.items()}
                incompatible_keys = self.model.load_state_dict(
                    model_state, strict=False
                )
            elif not current_is_ddp and checkpoint_was_ddp:
                log.info(
                    "Current model is not DDP, checkpoint is. Removing 'module.' prefix from keys."
                )
                model_state = {
                    k.replace("module.", "", 1): v
                    for k, v in model_state.items()
                    if k.startswith("module.")
                }
                incompatible_keys = model_to_load.load_state_dict(
                    model_state, strict=False
                )
            else:
                target_model_for_load = self.model if current_is_ddp else model_to_load
                incompatible_keys = target_model_for_load.load_state_dict(
                    model_state, strict=False
                )

            if incompatible_keys.missing_keys:
                log.warning(
                    f"Missing keys when loading model state_dict: {incompatible_keys.missing_keys}"
                )
            if incompatible_keys.unexpected_keys:
                log.warning(
                    f"Unexpected keys when loading model state_dict: {incompatible_keys.unexpected_keys}"
                )
            if (
                not incompatible_keys.missing_keys
                and not incompatible_keys.unexpected_keys
            ):
                log.info(f"Successfully loaded model state_dict from {path_to_load}")
            else:
                log.warning(
                    f"Model state_dict loaded from {path_to_load} with some mismatches."
                )

            self.model.to(self.device)
            log.info(f"Moved model to target device: {self.device}")

            if "optimizer_state_dict" in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
                    log.info("Successfully loaded optimizer state_dict.")
                except Exception as e:
                    log.error(
                        f"Failed to load optimizer state_dict: {e}. Optimizer state may be reset."
                    )
            else:
                log.warning(
                    "Optimizer state_dict not found in checkpoint. Optimizer not restored."
                )

            if (
                self.lr_scheduler
                and "lr_scheduler_state_dict" in checkpoint
                and checkpoint["lr_scheduler_state_dict"]
            ):
                try:
                    self.lr_scheduler.load_state_dict(
                        checkpoint["lr_scheduler_state_dict"]
                    )
                    log.info("Successfully loaded LR scheduler state_dict.")
                except Exception as e:
                    log.error(
                        f"Failed to load LR scheduler state_dict: {e}. Scheduler state may be reset."
                    )
            elif self.lr_scheduler:
                log.warning(
                    "LR scheduler state_dict not found in checkpoint or is empty. Scheduler not restored."
                )

            if "scaler_state_dict" in checkpoint and self.scaler:
                try:
                    self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                    log.info("Successfully loaded GradScaler state_dict.")
                except Exception as e:
                    log.error(
                        f"Failed to load GradScaler state_dict: {e}. Scaler state may be reset."
                    )
            elif self.scaler.is_enabled():
                log.warning(
                    "GradScaler state_dict not found in checkpoint. Scaler state not restored."
                )

            self.global_step = checkpoint.get("global_step", 0)
            self.tokens_seen = checkpoint.get("tokens_seen", 0)
            self.best_eval_loss = checkpoint.get("best_eval_loss", float("inf"))

            log.info(
                f"Checkpoint loaded successfully from {path_to_load}. Resuming training from global_step={self.global_step}"
            )
            log.info(f"Tokens seen according to checkpoint: {self.tokens_seen}")
            log.info(
                f"Best evaluation loss recorded in checkpoint: {self.best_eval_loss:.4f}"
            )

        except FileNotFoundError:
            log.error(
                f"Checkpoint file {path_to_load} not found during torch.load attempt. Starting from scratch."
            )
            self._reset_trainer_state()
        except Exception as e:
            log.error(
                f"Failed to load checkpoint from {path_to_load}: {e}. Traceback:",
                exc_info=True,
            )
            log.error("Resorting to starting training from scratch.")
            self._reset_trainer_state()

    def _reset_trainer_state(self):
        log.warning("Resetting trainer state (step, tokens, best_loss).")
        self.global_step = 0
        self.tokens_seen = 0
        self.best_eval_loss = float("inf")

    def cleanup(self):
        if self.wandb_run:
            log.info("Finishing WandB run...")
            self.wandb_run.finish()
            self.wandb_run = None
            log.info("WandB run finished.")
