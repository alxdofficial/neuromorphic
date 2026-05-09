"""Wave 2 entry point — long-chat TF NTP (plan §4.5).

Reads TurnPair parquet from preprocess_chat.py, length-buckets, runs
Phase1Trainer.step_wave2.

Usage:
    python scripts/train_wave2.py \\
        --data-paths \\
            data/wave2/ultrachat.parquet \\
            data/wave2/wildchat_long.parquet \\
        --batch-size 2 --num-steps 500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

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
from src.trajectory_memory.training.loaders import TurnPairDataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-paths", nargs="+", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-steps", type=int, default=500)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--lr-memory", type=float, default=3e-4)
    ap.add_argument("--lr-adapter", type=float, default=1e-4)
    ap.add_argument("--lr-min-ratio", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--config-tier", choices=["small", "medium", "large"],
                    default="medium")
    ap.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint-in", type=Path, default=None,
                    help="resume from a Wave 1 or Wave 2 checkpoint")
    ap.add_argument("--checkpoint-out", type=Path, default=None)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    cfg = getattr(TrajMemConfig, args.config_tier)()
    tokenizer = get_tokenizer()

    model = IntegratedLM(cfg, model_name=args.model_name, attach_lm=True).to(args.device)
    optimizer = build_optimizer(model, lr_memory=args.lr_memory, lr_adapter=args.lr_adapter)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.num_steps,
        lr_min_ratio=args.lr_min_ratio,
    )
    trainer = Phase1Trainer(model, optimizer, scheduler=scheduler, grad_clip=args.grad_clip)

    if args.checkpoint_in:
        ckpt = load_checkpoint(
            args.checkpoint_in, model=model,
            optimizer=optimizer, scheduler=scheduler,
            map_location=args.device,
        )
        trainer.load_state_dict({"step_count": ckpt.get("step", 0)})
        print(f"Resumed from {args.checkpoint_in} step {ckpt.get('step')}")

    dataset = TurnPairDataset(
        args.data_paths, batch_size=args.batch_size, pad_id=tokenizer.pad_token_id,
    )
    print(f"Wave 2 dataset: {len(dataset._rows)} TurnPairs, {len(dataset)} batches/epoch")

    losses: list = []
    t_start = time.time()
    while trainer.step_count < args.num_steps:
        for batch in dataset:
            if trainer.step_count >= args.num_steps:
                break
            batch.prior_ids = batch.prior_ids.to(args.device)
            batch.response_ids = batch.response_ids.to(args.device)
            batch.prior_mask = batch.prior_mask.to(args.device)
            batch.response_mask = batch.response_mask.to(args.device)

            metrics = trainer.step_wave2(batch)
            losses.append(metrics.loss)

            step = trainer.step_count
            if step % args.log_every == 0:
                avg = sum(losses[-args.log_every:]) / max(len(losses[-args.log_every:]), 1)
                elapsed = time.time() - t_start
                lrs = " ".join(f"{lr:.2e}" for lr in metrics.lr)
                print(f"  step {step:>5}  loss={metrics.loss:.4f}  avg10={avg:.4f}  "
                      f"grad_norm={metrics.grad_norm:.2f}  lr=[{lrs}]  "
                      f"({elapsed/max(step, 1):.2f}s/step)")

            if args.checkpoint_out is not None and step > 0 and step % args.save_every == 0:
                save_checkpoint(
                    args.checkpoint_out,
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    step=step,
                    rng_state=capture_rng_state(),
                    extra={"config": cfg.__dict__, "losses": losses},
                )
                print(f"  saved checkpoint to {args.checkpoint_out} at step {step}")

    if args.checkpoint_out:
        save_checkpoint(
            args.checkpoint_out,
            model=model, optimizer=optimizer, scheduler=scheduler,
            step=trainer.step_count,
            rng_state=capture_rng_state(),
            extra={"config": cfg.__dict__, "losses": losses},
        )
        print(f"Final checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    main()
