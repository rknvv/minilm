import argparse
import logging

from config import load_config
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


def train_model(yaml_path: str | None = None) -> None:
    if yaml_path is None:
        raise ValueError("yaml_path is required.")

    train_cfg, model_args = load_config(yaml_path)
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
        log.info(f"MiniLMForCausalLM: vocab_size={model_args.vocab_size}, params={n_params:,}")

    if train_cfg.pretrained_checkpoint:
        if train_cfg.resume_from_checkpoint:
            log.warning(
                "Both pretrained_checkpoint and resume_from_checkpoint are set. "
                "Ignoring pretrained_checkpoint and resuming from out_dir checkpoint."
            )
        else:
            load_pretrained_weights(model, train_cfg.pretrained_checkpoint)

    fsdp_mesh = None
    if train_cfg.parallel == "fsdp2" and dist_info.world_size > 1:
        from training.parallel import apply_fsdp2, build_fsdp_mesh

        model = model.to(dist_info.device)
        fsdp_mesh = build_fsdp_mesh(dist_info.world_size)
        model = apply_fsdp2(
            model, train_cfg.dtype, fsdp_mesh, train_cfg.reshard_after_forward
        )

    optimizer = build_optimizer(model, train_cfg, fsdp_mesh=fsdp_mesh)
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
