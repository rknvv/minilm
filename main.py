import argparse
import logging

from config import load_config, TrainConfig, ModelArgs
from models.minilm import MiniLM
from models.lm_head import MiniLMForCausalLM
from dataio.loaders import build_datasets, build_dataloaders
from training.distributed import setup_distributed, cleanup_distributed
from training.optim import build_optimizer, build_scheduler
from training.checkpoint import load_pretrained_weights
from training.trainer import Trainer
from utils.functions import snapshot_run

try:
    import fire
except ImportError:
    fire = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

SNAPSHOT_CODE_MODULES = [
    "models.layers",
    "models.transformer",
    "models.minilm",
    "models.lm_head",
    "config",
]


def train_model(yaml_path: str | None = None, **overrides) -> None:
    if yaml_path is None:
        raise ValueError("yaml_path is required.")

    train_cfg, model_args = load_config(yaml_path)

    if overrides:
        train_over = {
            k: v for k, v in overrides.items() if k in TrainConfig.model_fields
        }
        model_over = {k: v for k, v in overrides.items() if k in ModelArgs.model_fields}
        unknown = set(overrides) - set(train_over) - set(model_over)
        if unknown:
            raise ValueError(
                f"Unknown override keys: {sorted(unknown)}. "
                f"Valid: train={sorted(TrainConfig.model_fields)}, "
                f"model={sorted(ModelArgs.model_fields)}"
            )
        if train_over:
            train_cfg = TrainConfig(**{**train_cfg.model_dump(), **train_over})
        if model_over:
            model_args = ModelArgs(**{**model_args.model_dump(), **model_over})
        log.info(f"Applied CLI overrides: {overrides}")

    dist_info = setup_distributed(train_cfg)

    if dist_info.master:
        log.info(f"Train config: {train_cfg.model_dump()}")
        log.info(f"Model args: {model_args.model_dump()}")
        snapshot_run(
            out_dir=train_cfg.out_dir,
            config_sections={
                "train": train_cfg.model_dump(),
                "model": model_args.model_dump(),
            },
            code_modules=SNAPSHOT_CODE_MODULES,
        )

    train_dataset, eval_dataset = build_datasets(train_cfg, model_args)
    train_loader, eval_loader = build_dataloaders(
        train_cfg,
        train_dataset,
        eval_dataset,
        world_size=dist_info.world_size,
        rank=dist_info.rank,
        device=dist_info.device,
    )

    model = MiniLMForCausalLM(MiniLM(model_args), model_args)
    if dist_info.master:
        n_params = sum(p.numel() for p in model.parameters())
        log.info(
            f"MiniLMForCausalLM: vocab_size={model_args.vocab_size}, params={n_params:,}"
        )

    if train_cfg.pretrained_checkpoint:
        if train_cfg.resume_from_checkpoint:
            log.warning(
                "Both pretrained_checkpoint and resume_from_checkpoint are set. "
                "Ignoring pretrained_checkpoint and resuming from out_dir checkpoint."
            )
        else:
            load_pretrained_weights(model, train_cfg.pretrained_checkpoint)

    muon_pg = None
    if dist_info.world_size > 1 and train_cfg.muon_distributed:
        import torch.distributed as dist

        muon_pg = dist.group.WORLD
        if dist_info.master:
            log.info(
                "Muon: distributed Newton-Schulz over %d ranks (muon_distributed=true)",
                dist_info.world_size,
            )
    optimizer = build_optimizer(model, train_cfg, process_group=muon_pg)
    scheduler = build_scheduler(optimizer, train_cfg)

    trainer = Trainer(
        train_cfg=train_cfg,
        model_cfg=model_args,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        scheduler=scheduler,
        eval_loader=eval_loader,
        ddp_rank=dist_info.rank,
        ddp_local_rank=dist_info.local_rank,
        ddp_world_size=dist_info.world_size,
        master_process=dist_info.master,
    )

    try:
        trainer.train()
    finally:
        trainer.cleanup()
        cleanup_distributed(dist_info)


def main() -> None:
    if fire is not None:
        fire.Fire(train_model)
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", required=True, type=str)
    args = parser.parse_args()
    train_model(yaml_path=args.yaml_path)


if __name__ == "__main__":
    main()
