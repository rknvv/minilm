import json
import os
import sys
import tempfile

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import ModelArgs, TrainConfig
from models.minilm import MiniLM
from models.lm_head import MiniLMForCausalLM

TINY = dict(
    dim=64,
    n_layers=4,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    vocab_size=70000,
    intermediate_size=128,
    max_seq_len=64,
    sliding_window=8,
    sliding_window_pattern=2,
    query_pre_attn_scalar=16.0,
)


def test_model_paths() -> None:
    torch.manual_seed(0)
    args = ModelArgs(**TINY)
    model = MiniLMForCausalLM(MiniLM(args), args)
    B, T = 2, 32
    ids = torch.randint(0, args.vocab_size, (B, T))
    targets = torch.randint(0, args.vocab_size, (B, T))

    model.train()
    logits, loss = model(ids, targets=targets)
    loss.backward()
    assert logits.shape == (B, T, args.vocab_size)
    model.zero_grad(set_to_none=True)
    print(f"train fwd/bwd OK, loss={loss.item():.4f}")

    args_ck = ModelArgs(**{**TINY, "ce_chunk_size": 16})
    model_ck = MiniLMForCausalLM(MiniLM(args_ck), args_ck)
    model_ck.load_state_dict(model.state_dict())
    model_ck.train()
    _, loss_ck = model_ck(ids, targets=targets)
    assert abs(loss_ck.item() - loss.item()) < 1e-4, (loss_ck.item(), loss.item())
    print(f"chunked CE parity OK ({loss_ck.item():.6f} vs {loss.item():.6f})")

    args_gc = ModelArgs(**{**TINY, "gradient_checkpointing": True})
    model_gc = MiniLMForCausalLM(MiniLM(args_gc), args_gc)
    model_gc.load_state_dict(model.state_dict())
    model_gc.train()
    _, loss_gc = model_gc(ids, targets=targets)
    loss_gc.backward()
    assert abs(loss_gc.item() - loss.item()) < 1e-4
    print("gradient checkpointing OK")

    model.eval()
    with torch.no_grad():
        logits_eval, _ = model(ids, targets=targets)
    for layer in model.model.layers:
        assert layer.attention.k_cache is None, "eval allocated KV cache!"
    assert torch.allclose(logits, logits_eval, atol=1e-5), "eval != train logits"
    print("eval path OK (no KV cache, logits match train path)")

    model.eval()
    with torch.no_grad():
        full_logits, _ = model(ids, targets=targets)  # [B, T, V], no cache
        model.model.use_cache = True
        prefill_last = model(ids[:, :-1], start_pos=0)[0]  # logits at pos T-2
        step_last = model(ids[:, -1:], start_pos=T - 1)[0]  # logits at pos T-1
        model.model.use_cache = False
        model.model.reset_kv_caches()
    assert torch.allclose(full_logits[:, -2], prefill_last, atol=1e-4), (
        (full_logits[:, -2] - prefill_last).abs().max()
    )
    assert torch.allclose(full_logits[:, -1], step_last, atol=1e-4), (
        (full_logits[:, -1] - step_last).abs().max()
    )
    print("KV-cache parity OK (prefill + decode == full forward)")

    model.train()
    out = model.generate(ids[:, :8], max_new_tokens=4, eos_id=-1, temperature=0.0)
    assert model.training, "generate did not restore train mode"
    assert model.model.use_cache is False
    for layer in model.model.layers:
        assert layer.attention.k_cache is None
    assert out.shape[0] == B and out.shape[1] >= 8
    print("generate OK (mode restored, caches freed)")


def test_data_pipeline() -> None:
    from dataio.dataset import MemmapDataset
    from dataio.loaders import SkippableDistributedSampler, resolve_token_dtype

    with tempfile.TemporaryDirectory() as d:
        vocab = 70000
        tokens = np.random.randint(0, vocab, size=5000, dtype=np.uint32)
        bin_path = os.path.join(d, "train.bin")
        tokens.tofile(bin_path)

        ds = MemmapDataset(
            bin_path, sequence_length=64, memmap_dtype=np.uint32, vocab_size=vocab
        )
        item = ds[0]
        assert item["input_ids"].dtype == torch.int64
        assert (item["labels"][:-1] == item["input_ids"][1:]).all()
        print(f"MemmapDataset uint32 OK ({len(ds)} sequences)")

        try:
            MemmapDataset(
                bin_path, sequence_length=64, memmap_dtype=np.uint16, vocab_size=vocab
            )
        except ValueError as e:
            print(f"wrong-dtype detection OK: {str(e)[:80]}...")
        else:
            raise AssertionError("uint16 read of uint32 file passed vocab check!?")

        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"token_dtype": "uint32", "vocab_size": vocab}, f)
        tc = TrainConfig(dataset_dir=d, token_dtype="auto")
        ma = ModelArgs(**TINY)
        assert resolve_token_dtype(tc, ma) == np.dtype(np.uint32)
        print("resolve_token_dtype via meta.json OK")

        tc2 = TrainConfig(dataset_dir=os.path.join(d, "nope"), token_dtype="auto")
        assert resolve_token_dtype(tc2, ma) == np.dtype(np.uint32)
        small = ModelArgs(**{**TINY, "vocab_size": 32000})
        assert resolve_token_dtype(tc2, small) == np.dtype(np.uint16)
        print("resolve_token_dtype auto OK")

        sampler = SkippableDistributedSampler(
            ds, num_replicas=1, rank=0, shuffle=True, seed=42
        )
        sampler.set_epoch(0)
        full = list(iter(sampler))
        sampler.skip_samples = 3
        skipped = list(iter(sampler))
        assert skipped == full[3:]
        print("SkippableDistributedSampler OK")


def test_trainer_cycle() -> None:
    """Few steps -> save -> resume: state and data position restored. CPU, world=1."""
    from dataio.loaders import build_dataloaders
    from dataio.dataset import MemmapDataset
    from training.optim import build_scheduler
    from training.trainer import Trainer

    with tempfile.TemporaryDirectory() as d:
        vocab = 70000
        np.random.seed(0)
        data_dir = os.path.join(d, "data")
        os.makedirs(data_dir)
        np.random.randint(0, vocab, size=40000, dtype=np.uint32).tofile(
            os.path.join(data_dir, "train.bin")
        )
        np.random.randint(0, vocab, size=8000, dtype=np.uint32).tofile(
            os.path.join(data_dir, "val.bin")
        )
        with open(os.path.join(data_dir, "meta.json"), "w") as f:
            json.dump({"token_dtype": "uint32", "vocab_size": vocab}, f)

        args = ModelArgs(**TINY)
        cfg = TrainConfig(
            out_dir=os.path.join(d, "out"),
            dataset_dir=data_dir,
            device="cpu",
            dtype="float32",
            compile=False,
            max_iters=4,
            eval_interval=2,
            eval_iters=2,
            log_interval=2,
            train_batch_size=4,
            eval_batch_size=4,
            gradient_accumulation_steps=2,
            num_workers=0,
            wandb_log=False,
            resume_from_checkpoint=False,
            warmup_iters=1,
            learning_rate=1e-3,
        )

        def make_trainer(resume: bool) -> Trainer:
            cfg2 = cfg.model_copy(update={"resume_from_checkpoint": resume})
            from dataio.loaders import build_datasets

            train_ds, eval_ds = build_datasets(cfg2, args)
            tl, el = build_dataloaders(cfg2, train_ds, eval_ds, 1, 0, "cpu")
            model = MiniLMForCausalLM(MiniLM(args), args)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg2.learning_rate)
            sched = build_scheduler(opt, cfg2)
            return Trainer(
                train_cfg=cfg2,
                model_cfg=args,
                model=model,
                optimizer=opt,
                train_loader=tl,
                scheduler=sched,
                eval_loader=el,
            )

        t1 = make_trainer(resume=False)
        t1.train()
        assert t1.global_step == 4
        assert t1.samples_in_epoch == 4 * 2 * 4
        assert os.path.exists(t1.checkpoint_path), "latest ckpt must always exist"
        assert os.path.exists(t1.best_checkpoint_path), "best ckpt expected"
        print(
            f"trainer ran OK: step={t1.global_step}, samples_in_epoch={t1.samples_in_epoch}"
        )

        t2 = make_trainer(resume=True)
        assert t2.global_step == 4, t2.global_step
        assert t2.samples_in_epoch == t1.samples_in_epoch
        assert t2.best_eval_loss == t1.best_eval_loss
        t2._reset_train_iterator()
        assert t2.train_loader.sampler.skip_samples == t1.samples_in_epoch
        print("trainer resume OK: step/data position/best loss restored")

        import glob

        cfg_prof = cfg.model_copy(
            update={
                "out_dir": os.path.join(d, "out_prof"),
                "max_iters": 14,
                "eval_interval": 100,
                "profile": True,
            }
        )
        from dataio.loaders import build_datasets

        train_ds, eval_ds = build_datasets(cfg_prof, args)
        tl, el = build_dataloaders(cfg_prof, train_ds, eval_ds, 1, 0, "cpu")
        model = MiniLMForCausalLM(MiniLM(args), args)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        t3 = Trainer(
            train_cfg=cfg_prof,
            model_cfg=args,
            model=model,
            optimizer=opt,
            train_loader=tl,
            eval_loader=el,
        )
        t3.train()
        traces = glob.glob(os.path.join(cfg_prof.out_dir, "trace_*.json"))
        assert traces, "profiler did not export a trace"
        print(f"profiler OK: exported {os.path.basename(traces[0])}")


if __name__ == "__main__":
    test_model_paths()
    test_data_pipeline()
    test_trainer_cycle()
    print("\nALL SMOKE TESTS PASSED")
