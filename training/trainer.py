# Reference: https://github.com/allenai/OLMo/blob/main/olmo/train.py

import math
import logging
import os
import threading
import time
from contextlib import nullcontext
from typing import Any, Iterator, Optional, cast

import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DistributedSampler

from training.checkpoint import normalize_state_dict_keys, validate_state_dict
from training.ema import EMA
from config import TrainConfig, ModelArgs

logger = logging.getLogger(__name__)

_GPU_PEAK_FLOPS = {
    "H200": 989e12,
    "H100": 989e12,
    "A100": 312e12,
    "L40S": 362e12,
    "RTX 5090": 209.5e12,
    "RTX 4090": 165e12,
    "RTX 3090": 71e12,
}


def _to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_to_cpu(v) for v in obj]
        return tuple(converted) if isinstance(obj, tuple) else converted
    return obj


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

        if self.ddp_world_size > 1:
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.ddp_local_rank)
            self.model = self.model.to(self.device)
            ddp_kwargs: dict[str, Any] = dict(
                device_ids=[self.ddp_local_rank],
                output_device=self.ddp_local_rank,
                gradient_as_bucket_view=True,
            )
            if self.train_cfg.ddp_bucket_cap_mb is not None:
                ddp_kwargs["bucket_cap_mb"] = self.train_cfg.ddp_bucket_cap_mb
            self.model = DDP(self.model, **ddp_kwargs)
            logger.info(
                f"Rank [{self.ddp_rank}] local rank [{self.ddp_local_rank}]: Wrapped model with DDP"
            )
            if self.train_cfg.compile:
                self.model = self._compile_model(self.model)
                if self.master_process:
                    logger.info("Compiled DDP-wrapped model.")
        else:
            self.model = self.model.to(self.device)
            if self.train_cfg.compile:
                self.model = self._compile_model(self.model)
                if self.master_process:
                    logger.info("Compiled model.")

        autocast_device_type = "cuda" if self.device.startswith("cuda") else self.device
        if self.ptdtype == torch.float32 or autocast_device_type == "cpu":
            self.ctx = nullcontext()
        else:
            self.ctx = torch.autocast(
                device_type=autocast_device_type, dtype=self.ptdtype
            )

        self.scaler = torch.amp.GradScaler(  # type: ignore
            "cuda",
            enabled=(
                self.train_cfg.dtype == "float16" and autocast_device_type == "cuda"
            ),
        )

        self.out_dir = self.train_cfg.out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.out_dir, "ckpt.pt")
        self.best_checkpoint_path = os.path.join(self.out_dir, "ckpt_best.pt")

        self.global_step = 0
        self.tokens_seen = 0
        self.best_eval_loss = float("inf")

        self.train_epoch = 0
        self.samples_in_epoch = 0
        self.eval_epoch = 0
        self.train_iter: Optional[Iterator[dict[str, torch.Tensor]]] = None
        self.eval_iter: Optional[Iterator[dict[str, torch.Tensor]]] = None

        self.flops_per_token = self._estimate_flops_per_token()
        self.peak_flops = self._detect_peak_flops()

        self.ema: Optional[EMA] = None
        if self.train_cfg.ema is not None:
            self.ema = EMA(self._get_model_module(), self.train_cfg.ema)

        self._save_thread: Optional[threading.Thread] = None

        self._wandb_resume_id: Optional[str] = None
        if self.train_cfg.resume_from_checkpoint:
            self.load_checkpoint()

        self.wandb_run = None
        if self.train_cfg.wandb_log and self.master_process:
            self._init_wandb()

        if self.master_process:
            self._log_loss_path()

    def _compile_model(self, m: torch.nn.Module) -> torch.nn.Module:
        kwargs: dict[str, Any] = {}
        backend = self.train_cfg.compile_backend
        mode = self.train_cfg.compile_mode
        if backend and backend != "inductor":
            kwargs["backend"] = backend
        elif mode:
            kwargs["mode"] = mode

        base = m.module if isinstance(m, DDP) else m
        trunk = getattr(base, "model", None)
        if isinstance(trunk, torch.nn.Module):
            trunk.compile(**kwargs)
            if self.master_process:
                logger.info(
                    "Compiled transformer trunk in place (loss head stays eager)"
                    + (f", kwargs={kwargs}" if kwargs else "")
                )
            return m
        if self.master_process and kwargs:
            logger.info(f"torch.compile kwargs: {kwargs}")
        return cast(torch.nn.Module, torch.compile(m, **kwargs))

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
                id=(self._wandb_resume_id or wandb.util.generate_id()),
            )
        except ImportError:
            logger.warning("WandB is not installed. Skipping...")
            self.train_cfg.wandb_log = False

    def _log_loss_path(self) -> None:
        from models.layers import HAS_LIGER
        from models.lm_head import HAS_FUSED_LOSS

        args = self.model_cfg
        on_cuda = self.device.startswith("cuda")
        if args.use_liger and HAS_LIGER and on_cuda:
            if HAS_FUSED_LOSS and args.ce_chunk_size and args.ce_chunk_size > 0:
                path = (
                    "liger fused linear cross-entropy "
                    f"(vendored large-chunk, chunk={args.ce_chunk_size})"
                )
            else:
                path = "liger fused linear cross-entropy"
        elif args.ce_chunk_size and args.ce_chunk_size > 0:
            path = f"chunked cross-entropy (chunk={args.ce_chunk_size})"
            if args.use_liger and not HAS_LIGER:
                logger.warning(
                    "use_liger=True but liger-kernel is not installed; "
                    "falling back to chunked CE."
                )
        else:
            path = "full-logits cross-entropy"
        logger.info(f"Loss path: {path}")

    def _estimate_flops_per_token(self) -> float:
        m = self.model_cfg
        d, L, H = m.dim, m.n_layers, m.n_heads
        hd = m.head_dim
        kv_heads = m.n_kv_heads if m.n_kv_heads is not None else H
        T = m.max_seq_len
        w = m.sliding_window

        attn_params = d * H * hd + 2 * d * kv_heads * hd + H * hd * d
        mlp_params = 3 * d * m.intermediate_size
        head_params = d * m.vocab_size

        def avg_kv_len(window: Optional[int]) -> float:
            if window is None or window >= T:
                return (T + 1) / 2
            return (window * (window + 1) / 2 + (T - window) * window) / T

        n_global = sum(1 for i in range(L) if (i + 1) % m.sliding_window_pattern == 0)
        n_local = L - n_global
        attn_fwd = 4 * H * hd * (n_global * avg_kv_len(None) + n_local * avg_kv_len(w))

        fwd = 2 * (L * (attn_params + mlp_params) + head_params) + attn_fwd
        return 3.0 * fwd

    def _detect_peak_flops(self) -> Optional[float]:
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name(torch.cuda.current_device())
        for key, val in _GPU_PEAK_FLOPS.items():
            if key in name:
                return val
        logger.warning(f"Unknown GPU '{name}' for MFU estimation.")
        return None

    def _build_profiler(self):
        if not self.train_cfg.profile:
            return None

        cuda = self.device.startswith("cuda")
        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
        sort_key = "self_cuda_time_total" if cuda else "self_cpu_time_total"

        def on_trace_ready(prof) -> None:
            path = os.path.join(
                self.out_dir,
                f"trace_step{self.global_step}_rank{self.ddp_rank}.json",
            )
            prof.export_chrome_trace(path)
            logger.info(f"Profiler trace saved to {path}")
            if self.master_process:
                logger.info(
                    "\n" + prof.key_averages().table(sort_by=sort_key, row_limit=25)
                )

        prof = profile(
            activities=activities,
            schedule=schedule(wait=8, warmup=2, active=3, repeat=1),
            on_trace_ready=on_trace_ready,
        )
        prof.start()
        if self.master_process:
            logger.info(
                "Profiler armed: trace covers optimizer steps 11-13 "
                "(needs max_iters >= 13)."
            )
        return prof

    def _train_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = batch["input_ids"].to(self.device, non_blocking=True)
        targets = batch["labels"].to(self.device, non_blocking=True)

        with self.ctx:
            outputs = self.model(
                inputs,
                targets=targets,
                ignore_index=self.train_cfg.ignore_index,
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

    def _ddp_module(self) -> Optional[DDP]:
        m = self.model
        while True:
            if isinstance(m, DDP):
                return m
            if hasattr(m, "_orig_mod"):
                m = m._orig_mod
            else:
                return None

    def _get_model_module(self) -> torch.nn.Module:
        m = self.model
        while True:
            if hasattr(m, "_orig_mod"):
                m = m._orig_mod
            elif isinstance(m, DDP):
                m = m.module
            else:
                return m

    def _grad_sync_context(self, is_sync_step: bool):
        ddp = self._ddp_module()
        if ddp is not None and not is_sync_step:
            return ddp.no_sync()
        return nullcontext()

    def _reset_train_iterator(self) -> None:
        sampler = self.train_loader.sampler
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(self.train_epoch)
        if hasattr(sampler, "skip_samples"):
            sampler.skip_samples = self.samples_in_epoch
        self.train_iter = iter(self.train_loader)

    def _next_train_batch(self) -> dict[str, torch.Tensor]:
        if self.train_iter is None:
            self._reset_train_iterator()
        assert self.train_iter is not None
        try:
            batch = next(self.train_iter)
        except StopIteration:
            self.train_epoch += 1
            self.samples_in_epoch = 0
            self._reset_train_iterator()
            assert self.train_iter is not None
            batch = next(self.train_iter)
        self.samples_in_epoch += self.train_cfg.train_batch_size
        return batch

    def _reset_eval_iterator(self) -> None:
        if self.eval_loader is None:
            self.eval_iter = None
            return
        sampler = self.eval_loader.sampler
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(self.eval_epoch)
        self.eval_iter = iter(self.eval_loader)

    def _next_eval_batch(self) -> dict[str, torch.Tensor]:
        assert self.eval_iter is not None
        try:
            return next(self.eval_iter)
        except StopIteration:
            self._reset_eval_iterator()
            assert self.eval_iter is not None
            return next(self.eval_iter)

    def train(self) -> None:
        if self.train_cfg.eval_only:
            if self.ddp_world_size > 1:
                dist.barrier()
            eval_loss = self.evaluate()
            if self.master_process:
                logger.info(f"eval_only: Eval loss: {eval_loss}")
            return

        if self.master_process:
            logger.info(
                f"Starting training process from global step/max iterations: [{self.global_step}/{self.train_cfg.max_iters}]"
            )
            effective_batch_size = (
                self.train_cfg.train_batch_size
                * self.train_cfg.gradient_accumulation_steps
                * self.ddp_world_size
            )
            logger.info(
                f"Effective batch size: {effective_batch_size} "
                f"({effective_batch_size * self.model_cfg.max_seq_len:,} tokens/step)"
            )

        if self.ddp_world_size > 1:
            dist.barrier()

        self.model.train()
        micro_step_count = 0
        profiler = self._build_profiler()

        loss_since_log = torch.zeros((), device=self.device)
        steps_since_log = 0
        tokens_since_log = 0
        cuda = self.device.startswith("cuda")
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t_log = time.perf_counter()

        while self.global_step < self.train_cfg.max_iters:
            train_batch = self._next_train_batch()

            batch_tokens = int(train_batch["input_ids"].numel())
            self.tokens_seen += batch_tokens * self.ddp_world_size
            tokens_since_log += batch_tokens * self.ddp_world_size

            is_sync_step = (
                micro_step_count + 1
            ) == self.train_cfg.gradient_accumulation_steps

            with self._grad_sync_context(is_sync_step):
                micro_step_loss = self._train_step(train_batch)

            loss_since_log += micro_step_loss.detach()
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
                steps_since_log += 1
                if profiler is not None:
                    profiler.step()

                if self.global_step == 1:
                    loss_since_log = loss_since_log * 0.0
                    steps_since_log = 0
                    tokens_since_log = 0
                    if cuda:
                        torch.cuda.reset_peak_memory_stats()
                    t_log = time.perf_counter()

                if (
                    self.global_step % self.train_cfg.log_interval == 0
                    and steps_since_log > 0
                ):
                    loss_t = loss_since_log / steps_since_log
                    if self.ddp_world_size > 1:
                        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
                        loss_t = loss_t / self.ddp_world_size
                    avg_loss = loss_t.item()

                    now = time.perf_counter()
                    dt = max(now - t_log, 1e-9)
                    tokens_per_s = tokens_since_log / dt
                    ms_per_step = dt / steps_since_log * 1000.0
                    mfu = None
                    if self.peak_flops is not None:
                        per_gpu_tps = tokens_per_s / self.ddp_world_size
                        mfu = self.flops_per_token * per_gpu_tps / self.peak_flops
                    peak_mem_gb = (
                        torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
                    )
                    current_lr = self.optimizer.param_groups[0]["lr"]

                    if self.master_process:
                        logger.info(
                            f"Iter: {self.global_step}, LR: {current_lr:.2e}, "
                            f"Train loss: {avg_loss:.4f}, {tokens_per_s:,.0f} tok/s, "
                            f"{ms_per_step:.0f} ms/step"
                            + (f", MFU: {mfu * 100:.1f}%" if mfu is not None else "")
                            + (f", peak mem: {peak_mem_gb:.1f} GiB" if cuda else "")
                        )
                        if self.wandb_run:
                            metrics = {
                                "train/loss": avg_loss,
                                "train/learning_rate": current_lr,
                                "train/tokens_seen": self.tokens_seen,
                                "perf/tokens_per_s": tokens_per_s,
                                "perf/ms_per_step": ms_per_step,
                                "perf/peak_mem_gib": peak_mem_gb,
                            }
                            if mfu is not None:
                                metrics["perf/mfu"] = mfu
                            self.wandb_run.log(metrics, step=self.global_step)

                    loss_since_log = loss_since_log * 0.0
                    steps_since_log = 0
                    tokens_since_log = 0
                    if cuda:
                        torch.cuda.reset_peak_memory_stats()
                    t_log = time.perf_counter()

                if self.global_step % self.train_cfg.eval_interval == 0:
                    t_block = time.perf_counter()
                    if self.ddp_world_size > 1:
                        dist.barrier()
                    eval_loss = self.evaluate()
                    is_best = eval_loss < self.best_eval_loss
                    if is_best:
                        self.best_eval_loss = eval_loss
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
                    if self.train_cfg.always_save_checkpoint or is_best:
                        self.save_checkpoint(is_best=is_best)
                    if self.master_process:
                        logger.info(
                            "eval+checkpoint block: "
                            f"{time.perf_counter() - t_block:.1f}s "
                            "(disk write continues in background)"
                        )
                    t_log = time.perf_counter()

                micro_step_count = 0

        if profiler is not None:
            profiler.stop()

        self._join_pending_save()

        if self.ddp_world_size > 1:
            dist.barrier()

        if self.master_process:
            logger.info(f"Finished train at step: {self.global_step}")

    @torch.no_grad()
    def evaluate(self) -> float:
        if self.eval_loader is None or len(self.eval_loader) == 0:
            logger.warning(
                "Evaluation requested but eval_loader is empty. Returning nan..."
            )
            return float("nan")
        self.model.eval()

        self._reset_eval_iterator()

        if self.master_process:
            logger.info(
                f"Starting evaluation for {self.train_cfg.eval_iters} iterations..."
            )
        total_loss = torch.zeros((), dtype=torch.float64, device=self.device)

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
                        ignore_index=self.train_cfg.ignore_index,
                    )
                    if not isinstance(outputs, tuple) or len(outputs) != 2:
                        raise RuntimeError("Model forward must return (logits, loss).")
                    _, loss = outputs
                    if not isinstance(loss, torch.Tensor):
                        raise RuntimeError(
                            "Model forward must return Tensor loss for eval."
                        )

                total_loss += loss.detach().double()

        self.model.train()

        if self.ddp_world_size > 1:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_eval_iters = self.train_cfg.eval_iters * self.ddp_world_size
        else:
            total_eval_iters = self.train_cfg.eval_iters

        avg_loss = total_loss.item() / total_eval_iters
        try:
            perplexity = math.exp(avg_loss)
        except OverflowError:
            perplexity = float("inf")

        if self.master_process:
            logger.info(
                f"Evaluation complete: Avg loss: {avg_loss}, Perplexity: {perplexity}"
            )

        return avg_loss

    def _join_pending_save(self) -> None:
        if self._save_thread is not None:
            self._save_thread.join()
            self._save_thread = None

    def save_checkpoint(self, is_best: bool) -> None:
        """Snapshot state to CPU, then serialize on a background thread."""
        if not self.master_process:
            return

        self._join_pending_save()
        logger.info(f"Saving checkpoint at step {self.global_step} (async)...")

        model_to_save = self._get_model_module()

        checkpoint = {
            "model_state_dict": _to_cpu(model_to_save.state_dict()),
            "optimizer_state_dict": _to_cpu(self.optimizer.state_dict()),
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "best_eval_loss": self.best_eval_loss,
            "train_epoch": self.train_epoch,
            "samples_in_epoch": self.samples_in_epoch,
            "wandb_run_id": (
                self.wandb_run.id
                if self.wandb_run is not None
                else self._wandb_resume_id
            ),
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

        step = self.global_step
        best_loss = self.best_eval_loss

        def _write() -> None:
            try:
                tmp_save_path = f"{self.checkpoint_path}.tmp"
                torch.save(checkpoint, tmp_save_path)
                os.replace(tmp_save_path, self.checkpoint_path)
                logger.info(
                    f"Checkpoint for step {step} saved to: {self.checkpoint_path}"
                )

                if is_best:
                    tmp_best_path = f"{self.best_checkpoint_path}.tmp"
                    if os.path.exists(tmp_best_path):
                        os.remove(tmp_best_path)
                    os.link(self.checkpoint_path, tmp_best_path)
                    os.replace(tmp_best_path, self.best_checkpoint_path)
                    logger.info(
                        f"New best model (eval loss: {best_loss}). "
                        f"Hardlinked to: {self.best_checkpoint_path}"
                    )
            except Exception:
                logger.exception(
                    f"Async checkpoint write FAILED at step {step} "
                    f"(ckpt.pt keeps its previous state)."
                )

        self._save_thread = threading.Thread(
            target=_write, name="ckpt-save", daemon=True
        )
        self._save_thread.start()

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
            model_state = normalize_state_dict_keys(
                validate_state_dict(model_state_raw)
            )

            model_to_load = self._get_model_module()
            load_result = model_to_load.load_state_dict(model_state, strict=False)

            if load_result.missing_keys:
                logger.warning(f"Missing keys: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                logger.warning(f"Unexpected keys: {load_result.unexpected_keys}")

            success = not (load_result.missing_keys or load_result.unexpected_keys)
            logger.info(
                f"Model loaded {'successfully' if success else 'with some mismatches'}."
            )

            optimizer_state = checkpoint.get("optimizer_state_dict")
            if isinstance(optimizer_state, dict):
                try:
                    self.optimizer.load_state_dict(optimizer_state)
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

            ema_state = checkpoint.get("ema_state_dict")
            if self.ema is not None and isinstance(ema_state, dict):
                self.ema.load_state_dict(ema_state)
                logger.info("Successfully loaded EMA state dict.")

            self.global_step = int(checkpoint.get("global_step", 0))
            self.tokens_seen = int(checkpoint.get("tokens_seen", 0))
            self.best_eval_loss = float(checkpoint.get("best_eval_loss", float("inf")))
            self.train_epoch = int(checkpoint.get("train_epoch", 0))
            self.samples_in_epoch = int(checkpoint.get("samples_in_epoch", 0))
            wandb_run_id = checkpoint.get("wandb_run_id")
            if isinstance(wandb_run_id, str):
                self._wandb_resume_id = wandb_run_id

            logger.info(
                f"Checkpoint loaded succesfully from {load_path}. Resume training from global step: {self.global_step}"
            )
            logger.info(
                f"Tokens already seen according to checkpoint: {self.tokens_seen}. "
                f"Data position: epoch {self.train_epoch}, {self.samples_in_epoch} samples in."
            )
            logger.info(f"Current best evaluation loss: {self.best_eval_loss}")

        except Exception as e:
            logger.error(
                f"Failed to load checkpoint from {load_path}: {str(e)}\nInit from scratch."
            )

    def cleanup(self) -> None:
        self._join_pending_save()
        if self.wandb_run is not None:
            logger.info("Finish WandB session...")
            self.wandb_run.finish()
            self.wandb_run = None
