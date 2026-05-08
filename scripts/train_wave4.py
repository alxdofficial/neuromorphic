"""Wave 4 entry point — long-session GRPO (plan §4.5).

Reads TurnPair parquet (same format as Wave 2), samples J responses per
TurnPair via the model, scores against ground-truth response with
exact_match + BERT cosine, runs group-relative policy gradient.

Usage:
    python scripts/train_wave4.py \\
        --data-paths data/wave4/wildchat_long.parquet \\
        --num-samples 4 --num-steps 200
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
    Phase2Trainer,
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
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--num-steps", type=int, default=200)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--lr-memory", type=float, default=1e-4)
    ap.add_argument("--lr-adapter", type=float, default=5e-5)
    ap.add_argument("--lr-min-ratio", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--config-tier", choices=["small", "medium", "large"],
                    default="medium")
    ap.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint-in", type=Path, default=None)
    ap.add_argument("--checkpoint-out", type=Path, default=None)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=5)
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
    trainer = Phase2Trainer(model, optimizer, scheduler=scheduler, grad_clip=args.grad_clip)

    if args.checkpoint_in:
        ckpt = load_checkpoint(
            args.checkpoint_in, model=model,
            optimizer=optimizer, scheduler=scheduler,
            map_location=args.device,
        )
        trainer.load_state_dict({"step_count": ckpt.get("step", 0)})

    dataset = TurnPairDataset(args.data_paths, batch_size=1, pad_id=tokenizer.pad_token_id)
    print(f"Wave 4 dataset: {len(dataset._rows)} TurnPairs")

    rewards_history: list = []
    t_start = time.time()
    while trainer.step_count < args.num_steps:
        for batch in dataset:
            if trainer.step_count >= args.num_steps:
                break
            prior_mask = batch.prior_mask[0]
            prompt_ids = batch.prior_ids[0][prior_mask].to(args.device)
            response_mask = batch.response_mask[0]
            gold_resp = batch.response_ids[0][response_mask]
            gold_text = tokenizer.decode(gold_resp.tolist(), skip_special_tokens=True)

            metrics = trainer.step(
                prompt_ids,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                reward_kind="exact_match_or_bert_cosine",
                gold=gold_text,
                meta={"all_answers": [gold_text]},
                tokenizer=tokenizer,
                temperature=args.temperature,
            )
            mean_r = sum(metrics.rewards) / max(len(metrics.rewards), 1)
            rewards_history.append(mean_r)

            step = trainer.step_count
            if step % args.log_every == 0:
                avg = sum(rewards_history[-args.log_every:]) / max(
                    len(rewards_history[-args.log_every:]), 1
                )
                elapsed = time.time() - t_start
                print(f"  step {step:>4}  loss={metrics.policy_loss:.4f}  "
                      f"r={[f'{r:.2f}' for r in metrics.rewards]}  "
                      f"avg_r10={avg:.3f}  grad_norm={metrics.grad_norm:.2f}  "
                      f"({elapsed/max(step, 1):.2f}s/step)")

            if args.checkpoint_out is not None and step > 0 and step % args.save_every == 0:
                save_checkpoint(
                    args.checkpoint_out,
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    step=step,
                    rng_state=capture_rng_state(),
                    extra={"config": cfg.__dict__, "rewards_history": rewards_history},
                )

    if args.checkpoint_out:
        save_checkpoint(
            args.checkpoint_out,
            model=model, optimizer=optimizer, scheduler=scheduler,
            step=trainer.step_count,
            rng_state=capture_rng_state(),
            extra={"config": cfg.__dict__, "rewards_history": rewards_history},
        )


if __name__ == "__main__":
    main()
