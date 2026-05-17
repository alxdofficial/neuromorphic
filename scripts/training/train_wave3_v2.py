#!/usr/bin/env python3
"""Wave 3 GRPO training for the v2 architecture.

Loads a prompt-response parquet (data/wave3/*.parquet), rolls out J
responses per prompt under stochastic AR sampling, scores them via the
configured reward function, and applies GRPO policy gradient.

Usage:
    python scripts/training/train_wave3_v2.py \\
        --data-paths data/wave3/narrativeqa.train.parquet \\
        --warm-start outputs/wave2_v2/ckpt.pt \\
        --log-jsonl outputs/wave3_v2/train.jsonl \\
        --checkpoint-out outputs/wave3_v2/ckpt.pt \\
        --num-steps 5000 --n-rollouts 4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.trajectory_memory_v2._data import PromptResponseDataset
from src.trajectory_memory_v2.config import TrajMemV2Config
from src.trajectory_memory_v2.integrated_lm import IntegratedLMV2
from src.trajectory_memory_v2.wave3_trainer import Wave3TrainerV2
from scripts.training._dashboard import DashboardRenderer


def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-paths", nargs="+", required=True, type=Path)
    ap.add_argument("--config-tier", default="medium", choices=["small", "medium", "large"])

    ap.add_argument("--num-steps", type=int, default=5000)
    ap.add_argument("--n-rollouts", type=int, default=4)
    ap.add_argument("--max-response-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--kl-coef", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-adapter", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--plot-every-secs", type=float, default=180.0)
    ap.add_argument("--plot-out", default=None)
    ap.add_argument("--checkpoint-out", default=None)
    ap.add_argument("--warm-start", default=None,
                    help="Wave 2 v2 checkpoint to warm-start from")
    ap.add_argument("--checkpoint-in", default=None,
                    help="Resume from a previous Wave 3 ckpt")
    ap.add_argument("--log-jsonl", default=None)
    return ap.parse_args()


def _log_jsonl(path, record):
    if path is None: return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = getattr(TrajMemV2Config, args.config_tier)()
    cfg.validate()
    print(f"Architecture: vocabulary-trajectory v2 (Wave 3 GRPO)")
    print(f"Config: N={cfg.N} D={cfg.D_concept} K_max={cfg.K_max} J={cfg.J} "
          f"T_window={cfg.T_window}", flush=True)

    tok = get_tokenizer()
    print("Loading model...", flush=True)
    model = IntegratedLMV2(cfg, model_name="meta-llama/Llama-3.2-1B").to(args.device)

    if args.warm_start:
        ck = torch.load(args.warm_start, map_location=args.device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
        print(f"Warm-started from {args.warm_start} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

    # Optimizer (separate group for Llama backbone if it's unfrozen)
    adapter_params = [p for n, p in model.named_parameters() if not n.startswith("llama.")]
    llama_params = [p for n, p in model.named_parameters()
                    if n.startswith("llama.") and p.requires_grad]
    param_groups = [{"params": adapter_params, "lr": args.lr_adapter}]
    if llama_params:
        param_groups.append({"params": llama_params, "lr": args.lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    def lr_lambda(step):
        return min(1.0, step / max(1, args.warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    trainer = Wave3TrainerV2(
        model=model,
        optimizer=optimizer,
        pad_token_id=tok.pad_token_id,
        tokenizer=tok,
        scheduler=scheduler,
        grad_clip=args.grad_clip,
        n_rollouts=args.n_rollouts,
        max_response_tokens=args.max_response_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        kl_coef=args.kl_coef,
    )
    trainer.set_reference_state()
    print("Reference policy snapshot saved.", flush=True)

    dataset = PromptResponseDataset(parquet_paths=args.data_paths, seed=args.seed)
    print(f"Dataset: {len(dataset)} examples; sources={dataset.source_breakdown()}",
          flush=True)

    if args.checkpoint_in:
        ck = torch.load(args.checkpoint_in, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        trainer.load_state_dict(ck["trainer_state"])
        print(f"Resumed from {args.checkpoint_in} at step {trainer.step_count}", flush=True)

    # Dashboard (background plot renderer)
    plot_out = args.plot_out or (
        str(Path(args.log_jsonl).with_suffix(".plot.png")) if args.log_jsonl else None
    )
    dashboard = (
        DashboardRenderer(args.log_jsonl, plot_out, args.plot_every_secs)
        if args.log_jsonl and plot_out else None
    )
    if dashboard and args.plot_every_secs > 0:
        print(f"Dashboard: re-rendering {plot_out} every {args.plot_every_secs}s",
              flush=True)

    t_start = time.time()
    reward_window = []
    data_iter = iter(dataset)

    while trainer.step_count < args.num_steps:
        try:
            example = next(data_iter)
        except StopIteration:
            data_iter = iter(dataset)
            example = next(data_iter)

        t_step = time.time()
        try:
            metrics = trainer.step(example)
        except torch.cuda.OutOfMemoryError as e:
            print(f"FATAL: OOM at step {trainer.step_count}: {e}", file=sys.stderr)
            sys.exit(1)
        step_s = time.time() - t_step
        if not math.isfinite(metrics.loss):
            print(f"FATAL: non-finite loss at step {trainer.step_count}",
                  file=sys.stderr)
            sys.exit(1)

        step = trainer.step_count
        reward_window.append(metrics.mean_reward)
        if len(reward_window) > 100:
            reward_window.pop(0)

        if step % args.log_every == 0 or step == 1:
            avg_reward = sum(reward_window) / len(reward_window)
            print(
                f"step {step:>5}  loss={metrics.loss:.4f}  pg={metrics.pg_loss:.4f}  "
                f"kl={metrics.kl_loss:.4f}  R={metrics.mean_reward:.3f}  "
                f"avg{len(reward_window)}R={avg_reward:.3f}  "
                f"std_R={metrics.reward_std:.3f}  "
                f"resp_len={metrics.mean_response_len:.0f}  "
                f"edges={metrics.n_active_edges}  "
                f"({step_s:.2f}s/step)",
                flush=True,
            )

        _log_jsonl(args.log_jsonl, {
            "step": step, "phase": "train",
            "loss": metrics.loss, "pg_loss": metrics.pg_loss,
            "kl_loss": metrics.kl_loss,
            "mean_reward": metrics.mean_reward,
            "max_reward": metrics.max_reward,
            "min_reward": metrics.min_reward,
            "reward_std": metrics.reward_std,
            "mean_response_len": metrics.mean_response_len,
            "grad_norm": metrics.grad_norm,
            "n_active_edges": metrics.n_active_edges,
            "mean_effectiveness": metrics.mean_effectiveness,
            "mean_visit_count": metrics.mean_visit_count,
            "per_source": metrics.per_source,
            "step_s": step_s,
            "wall_s": time.time() - t_start,
        })

        if dashboard:
            dashboard.maybe_render()

        if args.checkpoint_out and step % args.save_every == 0:
            Path(args.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "trainer_state": trainer.state_dict(),
                "config": cfg,
            }, args.checkpoint_out)
            print(f"  saved ckpt @ step {step} → {args.checkpoint_out}", flush=True)

    if args.checkpoint_out:
        torch.save({
            "step": trainer.step_count,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "trainer_state": trainer.state_dict(),
            "config": cfg,
        }, args.checkpoint_out)
        print(f"Final ckpt → {args.checkpoint_out}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
