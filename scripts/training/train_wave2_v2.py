#!/usr/bin/env python3
"""Wave 2 chat SFT for the v2 architecture (streaming mode).

Reads condition on prev_window_hiddens (streaming/Hopfield mode).
Contrastive losses are off — alignment is by construction.

Usage:
    python scripts/training/train_wave2_v2.py \\
        --data-paths data/wave2/ultrachat.train.parquet \\
        --val-data-paths data/wave2/ultrachat.val.parquet \\
        --warm-start outputs/wave1_v2/ckpt.pt \\
        --log-jsonl outputs/wave2_v2/train.jsonl \\
        --checkpoint-out outputs/wave2_v2/ckpt.pt \\
        --num-steps 10000 --batch-size 4
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

from src.trajectory_memory_v2._data import TurnPairDataset
from src.trajectory_memory_v2.config import TrajMemV2Config
from src.trajectory_memory_v2.integrated_lm import IntegratedLMV2
from src.trajectory_memory_v2.wave2_trainer import Wave2TrainerV2


def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-paths", nargs="+", required=True, type=Path)
    ap.add_argument("--val-data-paths", nargs="+", type=Path, default=None)

    ap.add_argument("--config-tier", type=str, default="medium",
                    choices=["small", "medium", "large"])
    ap.add_argument("--n-override", type=int, default=None)

    ap.add_argument("--num-steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-adapter", type=float, default=3e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-prior-windows", type=int, default=4)

    ap.add_argument("--load-balance-coef", type=float, default=1e-4)
    ap.add_argument("--z-loss-coef", type=float, default=1e-4)

    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--checkpoint-out", type=str, default=None)
    ap.add_argument("--warm-start", type=str, default=None,
                    help="Path to a Wave 1 v2 ckpt to warm-start from")
    ap.add_argument("--checkpoint-in", type=str, default=None)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-jsonl", type=str, default=None)

    return ap.parse_args()


def _log_jsonl(path, record):
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = getattr(TrajMemV2Config, args.config_tier)()
    if args.n_override is not None:
        cfg.N = args.n_override
    cfg.validate()
    print(f"Architecture: vocabulary-trajectory v2 (sparse edges, streaming mode)")
    print(f"Config: N={cfg.N}, D_concept={cfg.D_concept}, K_max={cfg.K_max}, "
          f"J={cfg.J}, T_window={cfg.T_window}", flush=True)

    tokenizer = get_tokenizer()
    print("Loading model...", flush=True)
    model = IntegratedLMV2(cfg, model_name="meta-llama/Llama-3.2-1B").to(args.device)

    # Warm-start from Wave 1 v2 checkpoint (if provided)
    if args.warm_start:
        ck = torch.load(args.warm_start, map_location=args.device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
        print(f"Warm-started from {args.warm_start}", flush=True)
        if missing:
            print(f"  missing keys: {len(missing)}", flush=True)
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)}", flush=True)

    # Optimizer
    adapter_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("llama.")
    ]
    llama_params = [
        p for n, p in model.named_parameters()
        if n.startswith("llama.") and p.requires_grad
    ]
    param_groups = [{"params": adapter_params, "lr": args.lr_adapter}]
    if llama_params:
        param_groups.append({"params": llama_params, "lr": args.lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    trainer = Wave2TrainerV2(
        model=model,
        optimizer=optimizer,
        pad_token_id=tokenizer.pad_token_id,
        scheduler=scheduler,
        grad_clip=args.grad_clip,
        load_balance_coef=args.load_balance_coef,
        z_loss_coef=args.z_loss_coef,
        max_prior_windows=args.max_prior_windows,
    )

    # Data
    train_dataset = TurnPairDataset(
        parquet_paths=args.data_paths,
        batch_size=args.batch_size,
        pad_id=tokenizer.pad_token_id,
        bucket_window=32,
        seed=args.seed,
    )
    val_dataset = None
    if args.val_data_paths:
        val_dataset = TurnPairDataset(
            parquet_paths=args.val_data_paths,
            batch_size=args.batch_size,
            pad_id=tokenizer.pad_token_id,
            bucket_window=32,
            seed=args.seed + 1,
        )
    print(f"Train dataset: {len(train_dataset)} batches/epoch")
    if val_dataset:
        print(f"Val dataset:   {len(val_dataset)} batches/epoch")

    # Resume
    if args.checkpoint_in:
        ck = torch.load(args.checkpoint_in, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        trainer.load_state_dict(ck["trainer_state"])
        print(f"Resumed from {args.checkpoint_in} at step {trainer.step_count}")

    # Training loop
    t_start = time.time()
    loss_window = []
    train_iter = iter(train_dataset)

    while trainer.step_count < args.num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataset)
            batch = next(train_iter)

        t_step_start = time.time()
        try:
            metrics = trainer.step(batch)
        except torch.cuda.OutOfMemoryError as e:
            print(f"FATAL: OOM at step {trainer.step_count}: {e}", file=sys.stderr)
            sys.exit(1)
        step_s = time.time() - t_step_start
        if not math.isfinite(metrics.loss):
            print(f"FATAL: non-finite loss at step {trainer.step_count}",
                  file=sys.stderr)
            sys.exit(1)

        step = trainer.step_count
        loss_window.append(metrics.loss)
        if len(loss_window) > 50:
            loss_window.pop(0)
        wall_s_cumulative = time.time() - t_start

        if step % args.log_every == 0 or step == 1:
            avg_loss = sum(loss_window) / len(loss_window)
            print(
                f"step {step:>5}  loss={metrics.loss:.4f}  ans={metrics.answer_loss:.4f}  "
                f"avg{len(loss_window)}={avg_loss:.4f}  "
                f"edges={metrics.n_active_edges}({metrics.edge_active_fraction*100:.1f}%)  "
                f"fanout={metrics.mean_fan_out:.1f}  "
                f"vc={metrics.mean_visit_count:.1f}  "
                f"grad={metrics.grad_norm:.2f}  "
                f"({step_s:.2f}s/step)",
                flush=True,
            )

        _log_jsonl(args.log_jsonl, {
            "step": step, "phase": "train",
            "loss": metrics.loss, "answer_loss": metrics.answer_loss,
            "answer_tokens": metrics.answer_token_count,
            "aux_lb": metrics.aux_load_balance, "aux_z": metrics.aux_z_loss,
            "grad_norm": metrics.grad_norm,
            "n_active_edges": metrics.n_active_edges,
            "edge_active_fraction": metrics.edge_active_fraction,
            "mean_fan_out": metrics.mean_fan_out,
            "mean_edge_state_norm": metrics.mean_edge_state_norm,
            "mean_edge_specificity": metrics.mean_edge_specificity,
            "mean_visit_count": metrics.mean_visit_count,
            "mean_edge_age": metrics.mean_edge_age,
            "step_s": step_s,
            "wall_s_cumulative": wall_s_cumulative,
        })

        # ── Validation ──
        if val_dataset is not None and step % args.val_every == 0:
            val_losses, val_ans_losses = [], []
            val_iter_local = iter(val_dataset)
            for _ in range(args.val_batches):
                try:
                    vb = next(val_iter_local)
                except StopIteration:
                    break
                vm = trainer.eval_step(vb)
                val_losses.append(vm.loss)
                val_ans_losses.append(vm.answer_loss)
            if val_losses:
                avg_val = sum(val_losses) / len(val_losses)
                avg_val_ans = sum(val_ans_losses) / len(val_ans_losses)
                print(
                    f"  [val @ step {step}] loss={avg_val:.4f}  ans={avg_val_ans:.4f}  "
                    f"({len(val_losses)} batches)",
                    flush=True,
                )
                _log_jsonl(args.log_jsonl, {
                    "step": step, "phase": "val",
                    "loss": avg_val, "answer_loss": avg_val_ans,
                    "n_batches": len(val_losses),
                })

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
        print(f"Final ckpt → {args.checkpoint_out}")

    print("Done.")


if __name__ == "__main__":
    main()
