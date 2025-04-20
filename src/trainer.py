# Reference: https://github.com/allenai/OLMo/blob/main/olmo/train.py

import math
import logging
from pathlib import Path
from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import TrainConfig, ModelArgs

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        train_cfg: TrainConfig,
        model_cfg: ModelArgs,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        eval_loader: Optional[torch.utils.data.DataLoader] = None,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        master_process: bool = True,
        resume_from_checkpoint: bool = False,
    ):
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg

        self.model = model
        self.device = train_cfg.device

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.master_process = master_process
        self.resume_from_checkpoint = resume_from_checkpoint

        if self.ddp_world_size > 1:
            self.model = self.model.to(f"cuda:{self.ddp_rank}")
            if self.train_cfg.compile:
                if self.master_process:
                    logger.info("Compiling the model...")
                self.model = torch.compile(self.model)
                if self.master_process:
                    logger.info("Successfully compiled.")
            self.model = DDP(self.model, device_ids=[self.ddp_rank])
            logger.info(f"Rank [{self.ddp_rank}]: Wrapped model with DDP")
        else:
            self.model = self.model.to(self.device)
            if self.train_cfg.compile:
                if self.master_process:
                    logger.info("Compiling the model...")
                self.model = torch.compile(self.model)
                if self.master_process:
                    logger.info("Successfully compiled.")

        self.ptdtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(train_cfg.dtype, torch.float16)
        if self.ptdtype != torch.float32 and self.device == "cpu":
            logger.warning(
                f"AMP [{self.train_cfg.dtype}] is not supported on CPU. Using float32 context"
            )
            self.ctx = nullcontext()
        else:
            self.ctx = torch.amp.autocast(device_type=self.device, dtype=self.ptdtype)

        self.scaler = torch.amp.GradScaler(enabled=(self.train_cfg.dtype == "float16"))

        self.train_cfg.out_dir = Path(self.train_cfg.out_dir)
        self.train_cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.train_cfg.out_dir / "ckpt.pt"
        self.best_checkpoint_path = self.train_cfg.out_dir / "ckpt_best.pt"

        self.global_step = 0
        self.tokens_seen = 0
        self.best_eval_loss = float("inf")
        self.running_loss = 0.0

        if self.resume_from_checkpoint:
            self.load_checkpoint()

        self.wandb_run = None
        if self.train_cfg.wandb_log and self.master_process:
            self._init_wandb()

    def _init_wandb(self):
        try:
            import wandb

            config_dict = vars(self.train_cfg)
            config_dict.update(
                {f"model_{k}": v for k, v in vars(self.model_cfg).items()}
            )

            self.wandb_run = wandb.init(
                project=self.train_cfg.wandb_project,
                name=self.train_cfg.wandb_run_name,
                config=config_dict,
                resume="allow",
                id=(wandb.util.generate_id() if self.global_step == 0 else None),
            )
        except ImportError:
            logger.warning("WandB is not installed. Skipping...")
            self.train_cfg.wandb_log = False
            return

    def _train_step(self, batch):
        inputs = batch["input_ids"].to(self.device, non_blocking=True)
        targets = batch["labels"].to(self.device, non_blocking=True)

        with self.ctx:
            _, loss = self.model(inputs, targets=targets)
            loss = loss / self.train_cfg.gradient_accumulation_steps

        self.scaler.scale(loss).backward()

        return loss

    def train(self):
        if self.master_process:
            logger.info(
                f"Starting training process from global step/max iterations: [{self.global_step}/{self.train_cfg.max_iters}]"
            )
            effective_batch_size = (
                self.train_cfg.train_batch_size
                * self.train_cfg.gradient_accumulation_steps
                * self.ddp_world_size
            )
            logger.info(f"Effective batch size: {effective_batch_size}")

        if self.ddp_world_size > 1:
            dist.barrier()

        current_epoch = 0
        if self.ddp_world_size > 1 and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(current_epoch)
        train_iterator = iter(self.train_loader)

        self.model.train()
        self.running_loss = 0.0
        micro_step_count = 0

        while self.global_step < self.train_cfg.max_iters:
            try:
                batch = next(train_iterator)
            except StopIteration:
                logger.info(
                    f"Rank [{self.ddp_rank}], Iter [{self.global_step}]. StopIteration exception. Resetting DataLoader..."
                )
                current_epoch += 1
                if self.ddp_world_size > 1 and hasattr(
                    self.train_loader.sampler, "set_epoch"
                ):
                    logger.info(f"Setting DistributedSampler epoch to {current_epoch}")
                    self.train_loader.sampler.set_epoch(current_epoch)
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)
            except Exception as e:
                logger.error(f"Error: {e}.")
                if self.ddp_world_size > 1:
                    dist.barrier()
                break

            is_sync_step = (
                micro_step_count + 1
            ) == self.train_cfg.gradient_accumulation_steps

            ddp_sync_context = (
                self.model.no_sync()
                if (self.ddp_world_size > 1 and not is_sync_step)
                else nullcontext()
            )
            with ddp_sync_context:
                micro_step_loss = self._train_step(batch)

            self.running_loss += micro_step_loss.item()
            micro_step_count += 1
            if micro_step_count == self.train_cfg.gradient_accumulation_steps:
                if self.train_cfg.grad_clip > 0.0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.train_cfg.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

                if self.ddp_world_size > 1:
                    loss_tensor = torch.tensor([self.running_loss], device=self.device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    avg_loss = loss_tensor.item() / self.ddp_world_size
                else:
                    avg_loss = self.running_loss

                if self.master_process:
                    logger.info(
                        f"Rank: {self.ddp_rank}, Iter: {self.global_step}, LR: {self.optimizer.param_groups[0]['lr']:.2e}, Train loss: {avg_loss}"
                    )
                self.running_loss = 0.0

                if (
                    self.global_step % self.train_cfg.log_interval == 0
                    and self.master_process
                ):
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    if self.wandb_run:
                        wandb_metrics = {
                            "train/loss": avg_loss,
                            "train/learning_rate": current_lr,
                        }
                        self.wandb_run.log(wandb_metrics, step=self.global_step)

                if self.global_step % self.train_cfg.eval_interval == 0:
                    if self.ddp_world_size > 1:
                        dist.barrier()
                    eval_loss = self.evaluate()
                    if self.master_process:
                        logger.info(f"Iter: {self.global_step}, Eval loss: {eval_loss}")
                        if self.wandb_run:
                            self.wandb_run.log(
                                {"eval/loss": eval_loss}, step=self.global_step
                            )
                            is_best = eval_loss < self.best_eval_loss
                            if is_best:
                                self.best_eval_loss = eval_loss
                            if self.train_cfg.always_save_checkpoint or is_best:
                                self.save_checkpoint(is_best=is_best)
                        if self.train_cfg.eval_only and self.master_process:
                            return
                micro_step_count = 0

        if self.ddp_world_size > 1:
            dist.barrier()

        if self.master_process:
            logger.info(f"Finished train at step: {self.global_step}")

    @torch.no_grad()
    def evaluate(self):
        if self.eval_loader is None:
            logger.warning(
                "Evaluation requested but eval_loader is None. Returning nan..."
            )
            return float("nan")
        self.model.eval()

        if self.master_process:
            logger.info(
                f"Starting evaluation for {self.train_cfg.eval_iters} iterations..."
            )
        total_loss = 0.0

        eval_iterator = iter(self.eval_loader)
        for _ in tqdm(range(self.train_cfg.eval_iters)):
            batch = next(eval_iterator)
            inputs = batch["input_ids"].to(self.device, non_blocking=True)
            targets = batch["labels"].to(self.device, non_blocking=True)

            with self.ctx:
                _, loss = self.model(inputs, targets=targets)

            total_loss += loss.item()

        self.model.train()

        if self.ddp_world_size > 1:
            loss_tensor = torch.tensor(
                [total_loss], dtype=torch.float64, device=self.device
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            global_total_loss = loss_tensor.item()
            total_eval_iters = self.train_cfg.eval_iters * self.ddp_world_size
        else:
            global_total_loss = total_loss
            total_eval_iters = self.train_cfg.eval_iters

        avg_loss = global_total_loss / total_eval_iters
        perplexity = math.exp(avg_loss)
        if self.master_process:
            logger.info(
                f"Evaluation complete: Avg loss: {avg_loss}, Perplexity: {perplexity}"
            )

        return avg_loss

    def save_checkpoint(self, is_best: bool):
        if not self.master_process:
            return

        logger.info(f"Saving checkpoint at step {self.global_step}...")
        save_path = self.best_checkpoint_path if is_best else self.checkpoint_path

        model_to_save = self._get_model_module()

        checkpoint = {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": {
                "batch_size": self.train_cfg.train_batch_size,
                "learning_rate": self.train_cfg.learning_rate,
                "device": self.train_cfg.device,
                "dtype": self.train_cfg.dtype,
                "max_iters": self.train_cfg.max_iters,
            },
        }

        tmp_save_path = save_path.with_suffix(".tmp")

        torch.save(checkpoint, tmp_save_path)
        tmp_save_path.rename(save_path)
        logger.info(f"Successfully saved checkpoint to: {save_path}")

        if is_best:
            logger.info(f"This is a new best model. Eval loss: {self.best_eval_loss}")

    def _get_model_module(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def load_checkpoint(self):
        if self.best_checkpoint_path.exists():
            load_path = self.best_checkpoint_path
        elif self.checkpoint_path.exists():
            load_path = self.checkpoint_path
        else:
            logger.warning(
                f"No checkpoint file found at [{self.best_checkpoint_path}] or [{self.checkpoint_path}]. Starting training from scratch"
            )
            return
        logger.info(f"Loading checkpoint from {load_path}...")

        try:
            checkpoint = torch.load(load_path, map_location="cpu")
            model_state = checkpoint["model_state_dict"]

            current_is_ddp = isinstance(self.model, DDP)
            checkpoint_was_ddp = any(
                k.startswith("module.") for k in model_state.keys()
            )

            model_to_load = (
                self._get_model_module() if not current_is_ddp else self.model
            )

            if current_is_ddp and not checkpoint_was_ddp:
                logger.info(
                    "Current model is DDP, checkpoint is not. Adding 'module.' prefix."
                )
                model_state = {f"module.{k}": v for k, v in model_state.items()}
            elif not current_is_ddp and checkpoint_was_ddp:
                logger.info(
                    "Current model is not DDP, checkpoint is. Removing 'module.' prefix."
                )
                model_state = {
                    k.replace("module.", "", 1): v
                    for k, v in model_state.items()
                    if k.startswith("module.")
                }

            model = model_to_load.load_state_dict(model_state, strict=False)

            if model.missing_keys:
                logger.warning(f"Missing keys: {model.missing_keys}")
            if model.unexpected_keys:
                logger.warning(f"Unexpected keys: {model.unexpected_keys}")

            success = not (model.missing_keys or model.unexpected_keys)
            logger.info(
                f"Model loaded {'successfully' if success else 'with some mismatches'}."
            )

            self.model.to(self.device)
            logger.info(f"Moved model to target device: {self.device}")

            if "optimizer_state_dict" in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
                    logger.info("Successfully loaded optimizer state dict.")
                except Exception as e:
                    logger.error(f"Failed to load optimizer state dict. Error: {e}")
            else:
                logger.error("Cannot find optimizer state dict in checkpoint.")

            if self.scheduler and checkpoint["lr_scheduler_state_dict"]:
                try:
                    self.scheduler.load_state_dict(
                        checkpoint["lr_scheduler_state_dict"]
                    )
                    logger.info("Successfully loaded scheduler state dict.")
                except Exception as e:
                    logger.error(f"Failed to load scheduler state dict. Error: {e}")
            elif self.scheduler:
                logger.error(
                    "Cannot find scheduler state dict in checkpoint. Init from scratch."
                )

            if self.scaler and checkpoint["scaler_state_dict"]:
                try:
                    self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                    logger.info("Successfully loaded GradScaler state dict")
                except Exception as e:
                    logger.error(f"Failed tp load GradScaler state dict. Error: {e}")
            elif self.scaler:
                logger.error(
                    "Cannot find GradScaler state dict in checkpoint. Init from scratch."
                )

            self.global_step = checkpoint.get("global_step", 0)
            self.tokens_seen = checkpoint.get("tokens_seen", 0)
            self.best_eval_loss = checkpoint.get("best_eval_loss", float("inf"))

            logger.info(
                f"Checkpoint loaded succesfully from {load_path}. Resume training from global step: {self.global_step}"
            )
            logger.info(
                f"Tokens already seen according to checkpoint: {self.tokens_seen}"
            )
            logger.info(f"Current best evaluation loss: {self.best_eval_loss}")

        except Exception as e:
            logger.error(
                f"Failed to load checkpoint from {load_path}: {str(e)}\nInit from scratch."
            )

    def cleanup(self):
        if self.wandb_run:
            logger.info("Finish WandB session...")
            self.wandb_run.finish()
            self.wandb_run = None
