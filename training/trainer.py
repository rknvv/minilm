# Reference: https://github.com/allenai/OLMo/blob/main/olmo/train.py

import math
import logging
import os
from contextlib import nullcontext
from typing import Any, Iterator, Optional, cast

import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from training.checkpoint import normalize_state_dict_keys, validate_state_dict
from training.ema import EMA
from config import TrainConfig, ModelArgs

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        train_cfg: TrainConfig,
        model_cfg: ModelArgs,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader[dict[str, torch.Tensor]],
        scheduler: Optional[Any] = None,
        eval_loader: Optional[
            torch.utils.data.DataLoader[dict[str, torch.Tensor]]
        ] = None,
        ddp_rank: int = 0,
        ddp_local_rank: int = 0,
        ddp_world_size: int = 1,
        master_process: bool = True,
    ) -> None:
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg

        self.model = model
        self.device = train_cfg.device

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.ddp_rank = ddp_rank
        self.ddp_local_rank = ddp_local_rank
        self.ddp_world_size = ddp_world_size
        self.master_process = master_process

        self.ptdtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(train_cfg.dtype, torch.float16)

        self.use_fsdp2 = self.train_cfg.parallel == "fsdp2" and self.ddp_world_size > 1

        if self.ddp_world_size > 1:
            local_device = f"cuda:{self.ddp_local_rank}"
            self.device = local_device
            torch.cuda.set_device(self.ddp_local_rank)
            if self.use_fsdp2:
                if self.master_process:
                    logger.info("Using FSDP2-wrapped model.")
                if self.train_cfg.compile:
                    self.model = cast(torch.nn.Module, torch.compile(self.model))
                    if self.master_process:
                        logger.info("Compiled FSDP2 model.")
            else:
                self.model = self.model.to(local_device)
                if self.train_cfg.compile:
                    if self.master_process:
                        logger.info("Compiling the model...")
                    self.model = cast(torch.nn.Module, torch.compile(self.model))
                    if self.master_process:
                        logger.info("Successfully compiled.")
                self.model = DDP(
                    self.model,
                    device_ids=[self.ddp_local_rank],
                    output_device=self.ddp_local_rank,
                )
                logger.info(
                    f"Rank [{self.ddp_rank}] local rank [{self.ddp_local_rank}]: Wrapped model with DDP"
                )
        else:
            self.model = self.model.to(self.device)
            if self.train_cfg.compile:
                if self.master_process:
                    logger.info("Compiling the model...")
                compiled_model = torch.compile(self.model)
                if isinstance(compiled_model, torch.nn.Module):
                    self.model = compiled_model
                elif self.master_process:
                    logger.warning(
                        "Compiled model is not nn.Module, using eager module."
                    )
                if self.master_process:
                    logger.info("Successfully compiled.")

        autocast_device_type = "cuda" if self.device.startswith("cuda") else self.device
        if self.use_fsdp2 or (
            self.ptdtype != torch.float32 and autocast_device_type == "cpu"
        ):
            self.ctx = nullcontext()
        else:
            self.ctx = torch.autocast(
                device_type=autocast_device_type, dtype=self.ptdtype
            )

        self.scaler = torch.amp.GradScaler(  # type: ignore
            "cuda",
            enabled=(
                self.train_cfg.dtype == "float16"
                and autocast_device_type == "cuda"
                and not self.use_fsdp2
            ),
        )

        self.out_dir = self.train_cfg.out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.out_dir, "ckpt.pt")
        self.best_checkpoint_path = os.path.join(self.out_dir, "ckpt_best.pt")
        self.dist_checkpoint_path = os.path.join(self.out_dir, "ckpt_dist")
        self.dist_best_path = os.path.join(self.out_dir, "ckpt_best_dist")

        self.global_step = 0
        self.tokens_seen = 0
        self.best_eval_loss = float("inf")
        self.running_loss = 0.0

        self.train_epoch = 0
        self.eval_epoch = 0
        self.train_iter: Optional[Iterator[dict[str, torch.Tensor]]] = None
        self.eval_iter: Optional[Iterator[dict[str, torch.Tensor]]] = None

        self.ema: Optional[EMA] = None
        if self.train_cfg.ema is not None:
            self.ema = EMA(self._get_model_module(), self.train_cfg.ema)

        if self.train_cfg.resume_from_checkpoint:
            self.load_checkpoint()

        self.wandb_run = None
        if self.train_cfg.wandb_log and self.master_process:
            self._init_wandb()

    def _init_wandb(self) -> None:
        try:
            import wandb

            config_dict = dict(self.train_cfg.model_dump())
            config_dict.update(
                {f"model_{k}": v for k, v in self.model_cfg.model_dump().items()}
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

    def _train_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = batch["input_ids"].to(self.device, non_blocking=True)
        targets = batch["labels"].to(self.device, non_blocking=True)

        with self.ctx:
            outputs = self.model(
                inputs,
                targets=targets,
                ignore_index=self.train_cfg.sft_ignore_idx,
            )
            if not isinstance(outputs, tuple) or len(outputs) != 2:
                raise RuntimeError("Model forward must return (logits, loss).")
            _, loss = outputs
            if not isinstance(loss, torch.Tensor):
                raise RuntimeError(
                    "Model forward must return Tensor loss when targets are passed."
                )
            loss = loss / self.train_cfg.gradient_accumulation_steps

        self.scaler.scale(loss).backward()
        return loss

    def _grad_sync_context(self, is_sync_step: bool):
        if self.ddp_world_size <= 1:
            return nullcontext()
        if self.use_fsdp2:
            getattr(self.model, "set_requires_gradient_sync")(is_sync_step)
            return nullcontext()
        if isinstance(self.model, DDP) and not is_sync_step:
            return self.model.no_sync()
        return nullcontext()

    def _reset_train_iterator(self) -> None:
        sampler = self.train_loader.sampler
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(self.train_epoch)
        self.train_iter = iter(self.train_loader)

    def _next_train_batch(self) -> dict[str, torch.Tensor]:
        if self.train_iter is None:
            self._reset_train_iterator()
        assert self.train_iter is not None
        try:
            return next(self.train_iter)
        except StopIteration:
            self.train_epoch += 1
            self._reset_train_iterator()
            assert self.train_iter is not None
            return next(self.train_iter)

    def _reset_eval_iterator(self) -> None:
        if self.eval_loader is None:
            self.eval_iter = None
            return
        sampler = self.eval_loader.sampler
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(self.eval_epoch)
        self.eval_iter = iter(self.eval_loader)

    def _next_eval_batch(self) -> dict[str, torch.Tensor]:
        if self.eval_iter is None:
            self._reset_eval_iterator()
        assert self.eval_iter is not None
        try:
            return next(self.eval_iter)
        except StopIteration:
            self.eval_epoch += 1
            self._reset_eval_iterator()
            assert self.eval_iter is not None
            return next(self.eval_iter)

    def train(self) -> None:
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

        self.model.train()
        self.running_loss = 0.0
        micro_step_count = 0

        while self.global_step < self.train_cfg.max_iters:
            try:
                train_batch = self._next_train_batch()
            except Exception as e:
                logger.error(f"Error: {e}.")
                if self.ddp_world_size > 1:
                    dist.barrier()
                break

            if self.train_cfg.task == "sft":
                non_pad = (train_batch["labels"] != self.train_cfg.sft_ignore_idx).sum().item()
                self.tokens_seen += non_pad * self.ddp_world_size
            else:
                self.tokens_seen += (
                    int(train_batch["input_ids"].numel()) * self.ddp_world_size
                )

            is_sync_step = (
                micro_step_count + 1
            ) == self.train_cfg.gradient_accumulation_steps

            with self._grad_sync_context(is_sync_step):
                micro_step_loss = self._train_step(train_batch)

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

                if self.ema is not None:
                    self.ema.update(self._get_model_module())

                self.optimizer.zero_grad(set_to_none=True)

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
                        f"Iter: {self.global_step}, LR: {self.optimizer.param_groups[0]['lr']:.2e}, Train loss: {avg_loss}"
                    )
                self.running_loss = 0.0

                if (
                    self.global_step % self.train_cfg.log_interval == 0
                    and self.master_process
                ):
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    if self.wandb_run:
                        self.wandb_run.log(
                            {
                                "train/loss": avg_loss,
                                "train/learning_rate": current_lr,
                                "train/tokens_seen": self.tokens_seen,
                            },
                            step=self.global_step,
                        )

                if self.global_step % self.train_cfg.eval_interval == 0:
                    if self.ddp_world_size > 1:
                        dist.barrier()
                    eval_loss = self.evaluate()
                    if self.master_process:
                        logger.info(f"Iter: {self.global_step}, Eval loss: {eval_loss}")
                        if self.wandb_run:
                            self.wandb_run.log(
                                {
                                    "eval/loss": eval_loss,
                                    "train/tokens_seen": self.tokens_seen,
                                },
                                step=self.global_step,
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
    def evaluate(self) -> float:
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

        ema_ctx = (
            self.ema.average_parameters(self._get_model_module())
            if self.ema is not None
            else nullcontext()
        )

        with ema_ctx:
            for _ in tqdm(
                range(self.train_cfg.eval_iters), disable=not self.master_process
            ):
                batch = self._next_eval_batch()
                inputs = batch["input_ids"].to(self.device, non_blocking=True)
                targets = batch["labels"].to(self.device, non_blocking=True)

                with self.ctx:
                    outputs = self.model(
                        inputs,
                        targets=targets,
                        ignore_index=self.train_cfg.sft_ignore_idx,
                    )
                    if not isinstance(outputs, tuple) or len(outputs) != 2:
                        raise RuntimeError("Model forward must return (logits, loss).")
                    _, loss = outputs
                    if not isinstance(loss, torch.Tensor):
                        raise RuntimeError(
                            "Model forward must return Tensor loss for eval."
                        )

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
        try:
            perplexity = math.exp(avg_loss)
        except OverflowError:
            perplexity = float("inf")

        if self.master_process:
            logger.info(
                f"Evaluation complete: Avg loss: {avg_loss}, Perplexity: {perplexity}"
            )

        return avg_loss

    def save_checkpoint(self, is_best: bool) -> None:
        if self.use_fsdp2:
            self._save_distributed_checkpoint(is_best)
            return
        if not self.master_process:
            return

        logger.info(f"Saving checkpoint at step {self.global_step}...")
        save_path = self.best_checkpoint_path if is_best else self.checkpoint_path

        model_to_save = self._get_model_module()

        checkpoint = {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "best_eval_loss": self.best_eval_loss,
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "scaler_state_dict": self.scaler.state_dict(),
            "ema_state_dict": self.ema.state_dict() if self.ema is not None else None,
            "config": {
                "batch_size": self.train_cfg.train_batch_size,
                "learning_rate": self.train_cfg.learning_rate,
                "device": self.train_cfg.device,
                "dtype": self.train_cfg.dtype,
                "max_iters": self.train_cfg.max_iters,
            },
        }

        tmp_save_path = f"{save_path}.tmp"

        torch.save(checkpoint, tmp_save_path)
        os.replace(tmp_save_path, save_path)
        logger.info(f"Successfully saved checkpoint to: {save_path}")

        if is_best:
            logger.info(f"This is a new best model. Eval loss: {self.best_eval_loss}")

    def _save_distributed_checkpoint(self, is_best: bool) -> None:
        from training.dcp_checkpoint import save_distributed

        directory = self.dist_best_path if is_best else self.dist_checkpoint_path
        save_distributed(self.model, directory)

        if self.master_process:
            meta = {
                "global_step": self.global_step,
                "tokens_seen": self.tokens_seen,
                "best_eval_loss": self.best_eval_loss,
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
            }
            torch.save(meta, os.path.join(directory, "meta.pt"))
            logger.info(
                f"Saved distributed (model-only) checkpoint to: {directory}. "
                "Optimizer/EMA state is not persisted under FSDP2."
            )
        if self.ddp_world_size > 1:
            dist.barrier()

    def _load_distributed_checkpoint(self) -> None:
        from training.dcp_checkpoint import load_distributed

        preference = self.train_cfg.resume_checkpoint_kind
        if preference == "best":
            candidates = [self.dist_best_path]
        elif preference == "latest":
            candidates = [self.dist_checkpoint_path]
        else:
            candidates = [self.dist_checkpoint_path, self.dist_best_path]
        directory = next((c for c in candidates if os.path.isdir(c)), None)
        if directory is None:
            logger.warning("No distributed checkpoint found. Init from scratch.")
            return

        load_distributed(self.model, directory)

        meta_path = os.path.join(directory, "meta.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            self.global_step = int(meta.get("global_step", 0))
            self.tokens_seen = int(meta.get("tokens_seen", 0))
            self.best_eval_loss = float(meta.get("best_eval_loss", float("inf")))
            sched_state = meta.get("scheduler_state_dict")
            if self.scheduler is not None and isinstance(sched_state, dict):
                self.scheduler.load_state_dict(sched_state)
        logger.info(
            f"Loaded distributed checkpoint from {directory}. Resume at step {self.global_step}."
        )

    def _get_model_module(self) -> torch.nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _resolve_resume_checkpoint_path(self) -> Optional[str]:
        preference = self.train_cfg.resume_checkpoint_kind
        if preference == "latest":
            candidates = [self.checkpoint_path]
        elif preference == "best":
            candidates = [self.best_checkpoint_path]
        else:
            candidates = [self.checkpoint_path, self.best_checkpoint_path]

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        logger.warning(
            "No checkpoint file found for resume_checkpoint_kind=%s. Checked: %s",
            preference,
            candidates,
        )
        return None

    def load_checkpoint(self) -> None:
        if self.use_fsdp2:
            self._load_distributed_checkpoint()
            return
        load_path = self._resolve_resume_checkpoint_path()
        if load_path is None:
            return
        logger.info(f"Loading checkpoint from {load_path}...")

        try:
            checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, dict):
                logger.error("Checkpoint has invalid format. Init from scratch.")
                return

            model_state_raw = checkpoint.get("model_state_dict")
            if not isinstance(model_state_raw, dict):
                logger.error(
                    "Cannot find model_state_dict in checkpoint. Init from scratch."
                )
                return
            model_state = validate_state_dict(model_state_raw)

            current_is_ddp = isinstance(self.model, DDP)
            model_to_load = self.model if current_is_ddp else self._get_model_module()
            model_state = normalize_state_dict_keys(
                model_state,
                add_module_prefix=current_is_ddp,
            )

            load_result = model_to_load.load_state_dict(model_state, strict=False)

            if load_result.missing_keys:
                logger.warning(f"Missing keys: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                logger.warning(f"Unexpected keys: {load_result.unexpected_keys}")

            success = not (load_result.missing_keys or load_result.unexpected_keys)
            logger.info(
                f"Model loaded {'successfully' if success else 'with some mismatches'}."
            )

            self.model.to(self.device)
            logger.info(f"Moved model to target device: {self.device}")

            optimizer_state = checkpoint.get("optimizer_state_dict")
            if isinstance(optimizer_state, dict):
                try:
                    self.optimizer.load_state_dict(optimizer_state)
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
                    logger.info("Successfully loaded optimizer state dict.")
                except Exception as e:
                    logger.error(f"Failed to load optimizer state dict. Error: {e}")
            else:
                logger.error("Cannot find optimizer state dict in checkpoint.")

            scheduler_state = checkpoint.get("scheduler_state_dict")
            if self.scheduler and isinstance(scheduler_state, dict):
                self.scheduler.load_state_dict(scheduler_state)
                logger.info("Successfully loaded scheduler state dict.")
            elif self.scheduler:
                logger.error(
                    "Cannot find scheduler state dict in checkpoint. Init from scratch."
                )

            scaler_state = checkpoint.get("scaler_state_dict")
            if scaler_state:
                try:
                    self.scaler.load_state_dict(scaler_state)
                    logger.info("Successfully loaded GradScaler state dict")
                except Exception as e:
                    logger.error(f"Failed to load GradScaler state dict. Error: {e}")
            else:
                logger.error(
                    "Cannot find GradScaler state dict in checkpoint. Init from scratch."
                )

            ema_state = checkpoint.get("ema_state_dict")
            if self.ema is not None and isinstance(ema_state, dict):
                self.ema.load_state_dict(ema_state)
                logger.info("Successfully loaded EMA state dict.")

            self.global_step = int(checkpoint.get("global_step", 0))
            self.tokens_seen = int(checkpoint.get("tokens_seen", 0))
            self.best_eval_loss = float(checkpoint.get("best_eval_loss", float("inf")))

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

    def cleanup(self) -> None:
        if self.wandb_run is not None:
            logger.info("Finish WandB session...")
            self.wandb_run.finish()
            self.wandb_run = None
