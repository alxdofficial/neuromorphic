"""Wave 3 entry point — verifiable-reward GRPO (plan §4.5).

**Strategy A (current default, 2026-05-11): memory-only training mix.**
We train Phase 2 ONLY on long-context QA where the memory module is the
load-bearing component. Reasoning datasets (gsm8k, numinamath, humaneval)
are *excluded* from the train mix — Llama-3.2-1B-base doesn't reason
zero-shot, and we don't want to confound the memory research story by
training a reasoner under GRPO. Those datasets become held-out evals
instead. See plan §4.5 for the full rationale.

Usage (Strategy A — memory mix, 91% prompts >2K context):
    python scripts/train_wave3.py \\
        --data-paths \\
            data/wave3/narrativeqa.train.parquet \\
            data/wave3/musique.parquet \\
            data/wave3/hotpotqa.parquet \\
            data/wave3/2wikimultihop.parquet \\
            data/wave3/quality.parquet \\
        --num-samples 8 --num-steps 1000

Usage (with source upweighting to bias toward NarrativeQA's hardest split):
    python scripts/train_wave3.py \\
        --data-paths data/wave3/narrativeqa.train.parquet \\
                     data/wave3/musique.parquet \\
                     data/wave3/hotpotqa.parquet \\
                     data/wave3/2wikimultihop.parquet \\
                     data/wave3/quality.parquet \\
        --source-weights "narrativeqa=2,musique=1,hotpotqa=1,2wikimultihop=1,quality=2" \\
        --num-samples 8 --num-steps 1000

If you want to include reasoning datasets despite Strategy A's argument
against it, add them explicitly with a small weight (e.g. gsm8k=1).
"""

from __future__ import annotations

import argparse
import math
import sys
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
    ap.add_argument("--kl-coef", type=float, default=0.0,
                    help="Weight on KL(π_θ || π_ref) regularization. "
                         "Default 0 (no KL anchor) — our frozen-Llama setup "
                         "already constrains the policy via the immutable LM "
                         "vocab distribution + PPO clip_eps, so a reference "
                         "snapshot adds ~2.5 GB VRAM with no measurable "
                         "convergence benefit. Set >0 (e.g. 0.001) only if "
                         "you have evidence of reward-hacking / drift.")
    ap.add_argument("--source-weights", type=str, default=None,
                    help="Per-source upsampling weights in 'name=w,name=w' "
                         "format. e.g. 'narrativeqa=3,musique=3,hotpotqa=3,"
                         "2wikimultihop=3,quality=3,gsm8k=1,numinamath=1,"
                         "humaneval=1'. Used to bias the training mix toward "
                         "long-context (memory-engaging) sources. Default: "
                         "uniform (1.0 for all).")
    ap.add_argument("--bs-outer", type=int, default=8,
                    help="BS_outer — number of prompts per optimizer step. "
                         "1 = single-prompt step() (legacy). >1 = batched "
                         "step_batched(): M prompts × K samples = M*K parallel "
                         "rollouts, per-group advantage, per-sample backward, "
                         "single optimizer.step. Requires length-bucketed "
                         "sampler (auto-enabled when --bs-outer > 1). "
                         "Bench 2026-05-13 at D_concept=1024, kl_coef=0, K=8 "
                         "(see docs/bench_results.md): M=8 fits at 18.5 GB "
                         "peak with +34% rollout throughput vs M=1 "
                         "(per-sample 0.72s → 0.54s). Diminishing returns "
                         "past M=8 — GPU util is already capped by per-token "
                         "AR kernel-launch overhead, not memory.")
    ap.add_argument("--bs-outer-min-prompt-len", type=int, default=None,
                    help="When --bs-outer > 1, only include prompts with "
                         "at least this many tokens. Ensures all prompts "
                         "in a batch hit the effective_lm_context cap after "
                         "prefill, so KV caches stack cleanly. Default: "
                         "auto-derived from cfg.effective_lm_context "
                         "(2048 for medium, 4096 for large).")
    args = ap.parse_args()

    # Allow TF32 for fp32 matmul (memory params, bridge, lm_head).
    torch.set_float32_matmul_precision("high")

    cfg = getattr(TrajMemConfig, args.config_tier)()
    tokenizer = get_tokenizer()

    # Auto-derive bs-outer min-prompt-len from cfg if not specified.
    # Stacked-cache path requires uniform post-prefill seq_length, which
    # only holds if every prompt is long enough to hit the truncation cap.
    if args.bs_outer_min_prompt_len is None:
        args.bs_outer_min_prompt_len = cfg.effective_lm_context

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

    source_weights: dict[str, float] | None = None
    if args.source_weights:
        source_weights = {}
        for item in args.source_weights.split(","):
            if "=" not in item:
                continue
            name, w = item.split("=", 1)
            source_weights[name.strip()] = float(w.strip())
    dataset = PromptResponseDataset(args.data_paths, source_weights=source_weights)
    if trainer.step_count > 0:
        dataset._epoch = trainer.step_count  # B6 — fresh shuffle on resume
    if source_weights:
        print(f"Wave 3 dataset: {len(dataset)} prompts (post-weighting). "
              f"Source breakdown: {dataset.source_breakdown()}")
    else:
        print(f"Wave 3 dataset: {len(dataset)} prompts")

    rewards_history: list = []
    t_start = time.time()

    if args.bs_outer > 1:
        # ── BS_outer > 1 path: length-bucketed batches, step_batched().
        print(f"BS_outer mode: M={args.bs_outer}, K={args.num_samples}, "
              f"min_prompt_len={args.bs_outer_min_prompt_len}.")
        while trainer.step_count < args.num_steps:
            saw_batch = False
            for batch in dataset.iter_batched(
                args.bs_outer, min_prompt_len=args.bs_outer_min_prompt_len,
            ):
                saw_batch = True
                if trainer.step_count >= args.num_steps:
                    break
                prompts = [
                    torch.tensor(ex["prompt_ids"], dtype=torch.int64).to(args.device).unsqueeze(0)
                    for ex in batch
                ]
                per_prompt_meta = [
                    {
                        "reward_kind": ex["reward_kind"],
                        "gold": tokenizer.decode(ex["gold_ids"], skip_special_tokens=True),
                        "meta": ex["meta"],
                    } for ex in batch
                ]
                metrics = trainer.step_batched(
                    prompts, per_prompt_meta,
                    num_samples=args.num_samples,
                    max_new_tokens=args.max_new_tokens,
                    tokenizer=tokenizer,
                    temperature=args.temperature,
                )
                if not math.isfinite(metrics.policy_loss):
                    print(f"FATAL: non-finite policy_loss ({metrics.policy_loss}) "
                          f"at step {trainer.step_count}. Aborting.", file=sys.stderr)
                    sys.exit(1)
                if not math.isfinite(metrics.grad_norm):
                    print(f"FATAL: non-finite grad_norm ({metrics.grad_norm}) at "
                          f"step {trainer.step_count}. Aborting.", file=sys.stderr)
                    sys.exit(1)
                mean_r = sum(metrics.rewards) / max(len(metrics.rewards), 1)
                rewards_history.append(mean_r)

                step = trainer.step_count
                if step % args.log_every == 0:
                    avg = sum(rewards_history[-args.log_every:]) / max(
                        len(rewards_history[-args.log_every:]), 1
                    )
                    elapsed = time.time() - t_start
                    print(f"  step {step:>4} M={args.bs_outer}  "
                          f"loss={metrics.policy_loss:.4f}  "
                          f"avg_r={mean_r:.3f}  avg_r10={avg:.3f}  "
                          f"grad_norm={metrics.grad_norm:.2f}  "
                          f"clip_frac={metrics.clip_fraction:.3f}  "
                          f"mean_ratio={metrics.mean_ratio:.3f}  "
                          f"kl={metrics.kl_to_ref:.4f}  "
                          f"({elapsed/max(step, 1):.2f}s/step)")

                if args.checkpoint_out is not None and step > 0 and step % args.save_every == 0:
                    save_checkpoint(
                        args.checkpoint_out,
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        step=step, trainer_state=trainer.state_dict(),
                        rng_state=capture_rng_state(),
                    )
                    print(f"  saved checkpoint at step {step}")
            if not saw_batch:
                raise RuntimeError(
                    f"iter_batched(batch_size={args.bs_outer}, "
                    f"min_prompt_len={args.bs_outer_min_prompt_len}) yielded no "
                    f"batches — dataset has {len(dataset)} prompts but none "
                    f"meet the min_prompt_len filter. Lower --bs-outer-min-prompt-len "
                    f"or supply longer prompts."
                )
        # Final checkpoint save (covers the case where num_steps isn't a
        # multiple of save_every).
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
        # Exit early — single-prompt loop below is for legacy --bs-outer=1.
        return

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
            if not math.isfinite(metrics.policy_loss):
                print(f"FATAL: non-finite policy_loss ({metrics.policy_loss}) "
                      f"at step {trainer.step_count}. Aborting.", file=sys.stderr)
                sys.exit(1)
            if not math.isfinite(metrics.grad_norm):
                print(f"FATAL: non-finite grad_norm ({metrics.grad_norm}) at "
                      f"step {trainer.step_count}. Aborting.", file=sys.stderr)
                sys.exit(1)
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
