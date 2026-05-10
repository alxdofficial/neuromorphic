"""Wave 4 entry point — long-session GRPO (plan §4.5).

Reads TurnPair parquet (same format as Wave 2), samples J responses per
TurnPair via the model, scores against ground-truth response with
exact_match + BERT cosine, runs group-relative policy gradient.

Usage:
    # W4 reuses the W2 WildChat parquet (plan §4.5 — no separate W4 dataset).
    python scripts/train_wave4.py \\
        --data-paths data/wave2/wildchat_long.parquet \\
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
    ap.add_argument("--num-samples", type=int, default=8,
                    help="GRPO group size K. Default 8 matches TRL.")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--num-steps", type=int, default=200)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--lr-memory", type=float, default=1e-4)
    ap.add_argument("--lr-adapter", type=float, default=5e-5)
    ap.add_argument("--lr-min-ratio", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="Sampling temperature for AR rollouts. 0.7 matches "
                         "DeepSeek-R1 / verl defaults; lower variance at small K.")
    ap.add_argument("--config-tier", choices=["small", "medium", "large"],
                    default="medium")
    ap.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint-in", type=Path, default=None,
                    help="checkpoint to resume from OR warm-start from. "
                         "When loading W3 to start W4, always pass --warm-start.")
    ap.add_argument("--warm-start", action="store_true",
                    help="Load only model weights (no optimizer/scheduler/step).")
    ap.add_argument("--checkpoint-out", type=Path, default=None)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--no-compile", dest="compile", action="store_false",
                    help="Disable torch.compile (default ON).")
    ap.set_defaults(compile=True)
    ap.add_argument("--clip-eps", type=float, default=0.2,
                    help="PPO IS-ratio clip; default 0.2 (TRL/verl).")
    ap.add_argument("--clip-eps-higher", type=float, default=None,
                    help="DeepSeek-R1's `clip_higher` upper bound.")
    ap.add_argument("--kl-coef", type=float, default=0.001,
                    help="KL(π_θ || π_ref) coefficient. 0 disables.")
    args = ap.parse_args()

    cfg = getattr(TrajMemConfig, args.config_tier)()
    tokenizer = get_tokenizer()

    model = IntegratedLM(cfg, model_name=args.model_name, attach_lm=True).to(args.device)
    if args.compile:
        model.forward_window = torch.compile(
            model.forward_window, mode="default", dynamic=True,
        )
        print("Compiled model.forward_window (cold-start ~1-3 min on first step).")
    optimizer = build_optimizer(model, lr_memory=args.lr_memory, lr_adapter=args.lr_adapter)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.num_steps,
        lr_min_ratio=args.lr_min_ratio,
    )
    trainer = Phase2Trainer(
        model, optimizer, scheduler=scheduler, grad_clip=args.grad_clip,
        clip_eps=args.clip_eps,
        clip_eps_higher=args.clip_eps_higher,
        kl_coef=args.kl_coef,
    )

    resumed_ref = False
    if args.checkpoint_in:
        if args.warm_start:
            ckpt = load_checkpoint(
                args.checkpoint_in, model=model,
                optimizer=None, scheduler=None,
                map_location=args.device,
            )
            print(f"Warm-started from {args.checkpoint_in}")
        else:
            ckpt = load_checkpoint(
                args.checkpoint_in, model=model,
                optimizer=optimizer, scheduler=scheduler,
                map_location=args.device,
            )
            ts = ckpt.get("trainer_state", {"step_count": ckpt.get("step", 0)})
            trainer.load_state_dict(ts)
            if trainer.ref_state is not None:
                resumed_ref = True
            from src.trajectory_memory.training.checkpoint import restore_rng_state
            if "rng_state" in ckpt:
                restore_rng_state(ckpt["rng_state"])
            print(f"Resumed from {args.checkpoint_in} step {ckpt.get('step')}")

    if args.kl_coef > 0:
        if not args.checkpoint_in:
            raise SystemExit(
                "ERROR: --kl-coef > 0 requires --checkpoint-in. Without it, "
                "set_reference_state() snapshots random init as π_ref."
            )
        if args.warm_start or not resumed_ref:
            trainer.set_reference_state()
            print(f"Reference policy snapshot taken (kl_coef={args.kl_coef}).")
        else:
            print(f"Reference policy restored from checkpoint "
                  f"(kl_coef={args.kl_coef}).")

    dataset = TurnPairDataset(args.data_paths, batch_size=1, pad_id=tokenizer.pad_token_id)
    if trainer.step_count > 0:
        dataset._epoch = trainer.step_count  # B6 — fresh shuffle on resume
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
                      f"clip_frac={metrics.clip_fraction:.3f}  "
                      f"mean_ratio={metrics.mean_ratio:.3f}  "
                      f"kl={metrics.kl_to_ref:.4f}  "
                      f"({elapsed/max(step, 1):.2f}s/step)")

            if args.checkpoint_out is not None and step > 0 and step % args.save_every == 0:
                save_checkpoint(
                    args.checkpoint_out,
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    step=step,
                    rng_state=capture_rng_state(),
                    extra={"config": cfg.__dict__, "rewards_history": rewards_history},
                    trainer_state=trainer.state_dict(),
                )

    if args.checkpoint_out:
        save_checkpoint(
            args.checkpoint_out,
            model=model, optimizer=optimizer, scheduler=scheduler,
            step=trainer.step_count,
            rng_state=capture_rng_state(),
            extra={"config": cfg.__dict__, "rewards_history": rewards_history},
            trainer_state=trainer.state_dict(),
        )


if __name__ == "__main__":
    main()
