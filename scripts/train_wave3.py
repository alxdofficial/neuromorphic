"""Wave 3 entry point — verifiable-reward GRPO (plan §4.5).

Usage:
    python scripts/train_wave3.py \\
        --data-paths \\
            data/wave3/gsm8k.parquet \\
            data/wave3/numinamath.parquet \\
            data/wave3/humaneval.parquet \\
            data/wave3/narrativeqa.parquet \\
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
from src.trajectory_memory.training.loaders import PromptResponseDataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-paths", nargs="+", required=True, type=Path)
    ap.add_argument("--num-samples", type=int, default=8,
                    help="GRPO group size K. Default 8 matches TRL's "
                         "current num_generations default; DeepSeek-R1 "
                         "used 16. K<8 makes group-relative advantage "
                         "estimates noisy.")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--num-steps", type=int, default=200)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--lr-memory", type=float, default=1e-4)
    ap.add_argument("--lr-adapter", type=float, default=5e-5)
    ap.add_argument("--lr-min-ratio", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="Sampling temperature for AR rollouts. 0.7 matches "
                         "DeepSeek-R1 / verl defaults. Lower temperature "
                         "reduces both reward variance and group-relative "
                         "advantage noise at small K.")
    ap.add_argument("--config-tier", choices=["small", "medium", "large"],
                    default="medium")
    ap.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint-in", type=Path, default=None,
                    help="checkpoint to resume from OR warm-start from. "
                         "When loading W1/W2 to start W3, always pass --warm-start.")
    ap.add_argument("--warm-start", action="store_true",
                    help="Load only model weights (no optimizer/scheduler/step). "
                         "Use when transitioning waves.")
    ap.add_argument("--checkpoint-out", type=Path, default=None)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=5)
    # ── GRPO regularization (Phase B) ────────────────────────────────
    ap.add_argument("--no-compile", dest="compile", action="store_false",
                    help="Disable torch.compile (default ON). Phase 2's TF "
                         "replay benefits same as Phase 1; ~2 min cold-start.")
    ap.set_defaults(compile=True)
    ap.add_argument("--clip-eps", type=float, default=0.2,
                    help="PPO importance-sampling ratio clip width. "
                         "Default 0.2 matches TRL/verl. Set high (e.g. 10) "
                         "to effectively disable clipping.")
    ap.add_argument("--clip-eps-higher", type=float, default=None,
                    help="Asymmetric upper clip (DeepSeek-R1's `clip_higher`). "
                         "When set, ratio clamped to [1-clip_eps, 1+clip_eps_higher]. "
                         "R1 used 10. None → symmetric.")
    ap.add_argument("--kl-coef", type=float, default=0.001,
                    help="Weight on KL(π_θ || π_ref) regularization. "
                         "Default 0.001 matches verl. 0 disables KL term. "
                         "Reference is the loaded --checkpoint-in weights.")
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

    resumed_ref = False  # B7/N5 — track whether ckpt restored ref_state
    if args.checkpoint_in:
        if args.warm_start:
            ckpt = load_checkpoint(
                args.checkpoint_in, model=model,
                optimizer=None, scheduler=None,
                map_location=args.device,
            )
            print(f"Warm-started model from {args.checkpoint_in}")
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
            print(f"Resumed from {args.checkpoint_in}")

    # B7 + N5 fix — snapshot ref policy ONLY on warm-start (or fresh run
    # with --kl-coef>0 + a checkpoint to load). On full resume, ref_state
    # comes from the loaded checkpoint via load_state_dict — re-snapshotting
    # would drift the anchor to the mid-training resumed policy. Hard error
    # if user enables KL without giving us weights to anchor against.
    if args.kl_coef > 0:
        if not args.checkpoint_in:
            raise SystemExit(
                "ERROR: --kl-coef > 0 requires --checkpoint-in to define the "
                "reference policy. Snapshotting random init as π_ref would "
                "regularize toward noise. Set --kl-coef 0 to disable KL or "
                "pass --checkpoint-in <Phase 1 checkpoint>."
            )
        if args.warm_start or not resumed_ref:
            trainer.set_reference_state()
            print(f"Reference policy snapshot taken (kl_coef={args.kl_coef}).")
        else:
            print(f"Reference policy restored from checkpoint "
                  f"(kl_coef={args.kl_coef}).")

    dataset = PromptResponseDataset(args.data_paths)
    print(f"Wave 3 dataset: {len(dataset)} prompts")

    rewards_history: list = []
    t_start = time.time()
    while trainer.step_count < args.num_steps:
        for example in dataset:
            if trainer.step_count >= args.num_steps:
                break
            prompt_ids = torch.tensor(example["prompt_ids"], dtype=torch.int64).to(args.device)
            gold_text = tokenizer.decode(example["gold_ids"], skip_special_tokens=True)

            metrics = trainer.step(
                prompt_ids,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                reward_kind=example["reward_kind"],
                gold=gold_text,
                meta=example["meta"],
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
                print(f"  saved checkpoint at step {step}")

    if args.checkpoint_out:
        save_checkpoint(
            args.checkpoint_out,
            model=model, optimizer=optimizer, scheduler=scheduler,
            step=trainer.step_count,
            rng_state=capture_rng_state(),
            extra={"config": cfg.__dict__, "rewards_history": rewards_history},
            trainer_state=trainer.state_dict(),
        )
        print(f"Saved {args.checkpoint_out}")


if __name__ == "__main__":
    main()
