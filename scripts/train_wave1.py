"""Wave 1 entry point — long-doc TF NTP pretraining (plan §4.5).

Reads pre-tokenized parquet from preprocess_longdoc.py (and optionally
synthesize_needle.py), packs into D*T_window chunks, runs cross-window
TBPTT with Phase1Trainer.

Features:
  - LR warmup + cosine decay
  - Gradient clipping
  - Checkpoint save / resume (model + optimizer + scheduler + RNG state)
  - Per-step metrics (loss, grad_norm, LR)

Usage:
    python scripts/train_wave1.py \\
        --data-paths \\
            data/wave1/fineweb_edu.parquet \\
            data/wave1/wikipedia_en.parquet \\
            data/wave1/slimpajama_6b.parquet \\
            data/wave1/needle.parquet \\
        --batch-size 2 --num-steps 1000 \\
        --checkpoint-out outputs/wave1/ckpt.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.data.tokenizer import get_tokenizer
from src.trajectory_memory.integrated_lm import IntegratedLM
from src.trajectory_memory.training import (
    Phase1Trainer,
    WarmupCosineScheduler,
    build_optimizer,
    capture_rng_state,
    load_checkpoint,
    save_checkpoint,
)
from src.trajectory_memory.training.loaders import LongDocDataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-paths", nargs="+", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-steps", type=int, default=1000)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--lr-memory", type=float, default=3e-4)
    ap.add_argument("--lr-adapter", type=float, default=1e-4)
    ap.add_argument("--lr-min-ratio", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--config-tier", choices=["small", "medium", "large"],
                    default="medium")
    ap.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint-in", type=Path, default=None)
    ap.add_argument("--checkpoint-out", type=Path, default=None)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--val-data-paths", nargs="+", type=Path, default=None,
                    help="held-out val parquets (e.g., needle.val.parquet for "
                         "memory-bridging probe). If set, eval at each save.")
    ap.add_argument("--val-batches", type=int, default=20,
                    help="number of val batches to average per eval pass")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile model.forward_window. ~28% speedup at "
                         "BS=2 with ~2 min cold-start. Recommended for "
                         "production runs (see docs/bench_results.md).")
    args = ap.parse_args()

    cfg = getattr(TrajMemConfig, args.config_tier)()
    print(f"Config tier: {args.config_tier}")
    print(f"  N={cfg.N}, J={cfg.J}, K_read={cfg.K_read}, D={cfg.D}, T_window={cfg.T_window}")

    tokenizer = get_tokenizer()
    pad_id = tokenizer.pad_token_id

    model = IntegratedLM(cfg, model_name=args.model_name, attach_lm=True).to(args.device)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if args.compile:
        model.forward_window = torch.compile(
            model.forward_window, mode="default", dynamic=False,
        )
        print("Compiled model.forward_window (cold-start on first step ~1-3 min).")

    optimizer = build_optimizer(model, lr_memory=args.lr_memory, lr_adapter=args.lr_adapter)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.num_steps,
        lr_min_ratio=args.lr_min_ratio,
    )
    trainer = Phase1Trainer(
        model, optimizer, scheduler=scheduler, grad_clip=args.grad_clip,
        pad_token_id=tokenizer.pad_token_id,
    )

    if args.checkpoint_in is not None:
        ckpt = load_checkpoint(
            args.checkpoint_in, model=model,
            optimizer=optimizer, scheduler=scheduler,
            map_location=args.device,
        )
        trainer.load_state_dict({"step_count": ckpt.get("step", 0)})
        print(f"Resumed from {args.checkpoint_in} at step {ckpt.get('step')}")

    dataset = LongDocDataset(
        args.data_paths,
        chunk_tokens=cfg.D * cfg.T_window,
        pad_id=pad_id, drop_short=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    val_loader = None
    if args.val_data_paths:
        val_dataset = LongDocDataset(
            args.val_data_paths,
            chunk_tokens=cfg.D * cfg.T_window,
            pad_id=pad_id, drop_short=False,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
        print(f"Validation: {len(args.val_data_paths)} parquet(s), "
              f"{args.val_batches} batches per eval")

    def run_val() -> float:
        if val_loader is None:
            return float("nan")
        losses_v: list[float] = []
        for i, chunk_v in enumerate(val_loader):
            if i >= args.val_batches:
                break
            losses_v.append(trainer.eval_wave1(chunk_v.to(args.device)))
        return sum(losses_v) / max(len(losses_v), 1)

    print(f"Starting Wave 1 training: {args.num_steps} steps "
          f"(starting from step {trainer.step_count})")
    losses: list = []
    t_start = time.time()
    while trainer.step_count < args.num_steps:
        for chunk in loader:
            if trainer.step_count >= args.num_steps:
                break
            chunk = chunk.to(args.device)
            metrics = trainer.step_wave1(chunk)
            losses.append(metrics.loss)

            step = trainer.step_count
            if step % args.log_every == 0:
                avg = sum(losses[-args.log_every:]) / max(len(losses[-args.log_every:]), 1)
                elapsed = time.time() - t_start
                lrs = " ".join(f"{lr:.2e}" for lr in metrics.lr)
                print(f"  step {step:>5}  loss={metrics.loss:.4f}  avg10={avg:.4f}  "
                      f"grad_norm={metrics.grad_norm:.2f}  lr=[{lrs}]  "
                      f"({elapsed/max(step, 1):.2f}s/step)")

            if step > 0 and step % args.save_every == 0:
                if val_loader is not None:
                    val_loss = run_val()
                    print(f"  step {step:>5}  val_loss={val_loss:.4f}")
                if args.checkpoint_out is not None:
                    save_checkpoint(
                        args.checkpoint_out,
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        step=step,
                        rng_state=capture_rng_state(),
                        extra={"config": cfg.__dict__, "losses": losses},
                    )
                    print(f"  saved checkpoint to {args.checkpoint_out} at step {step}")

    if args.checkpoint_out is not None:
        save_checkpoint(
            args.checkpoint_out,
            model=model, optimizer=optimizer, scheduler=scheduler,
            step=trainer.step_count,
            rng_state=capture_rng_state(),
            extra={"config": cfg.__dict__, "losses": losses},
        )
        print(f"Final checkpoint saved to {args.checkpoint_out}")


if __name__ == "__main__":
    main()
