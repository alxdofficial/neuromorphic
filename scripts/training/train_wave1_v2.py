#!/usr/bin/env python3
"""Wave 1 retrieval pretraining for the v2 architecture.

Vocabulary-trajectory memory: writes go to sparse edge state via EMA,
reads autocomplete through the graph. See:
  - src/trajectory_memory_v2/
  - docs/design_vocabulary_trajectory.md

Usage:
    python scripts/training/train_wave1_v2.py \\
        --composite-dir data/wave1/composite_v1/train \\
        --composite-val-dir data/wave1/composite_v1/val \\
        --log-jsonl outputs/wave1_v2/train.jsonl \\
        --checkpoint-out outputs/wave1_v2/ckpt.pt \\
        --num-steps 25000 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.trajectory_memory_v2.config import TrajMemV2Config
from src.trajectory_memory_v2.integrated_lm import IntegratedLMV2
from src.trajectory_memory_v2.trainer import Phase1RetrievalTrainerV2
from scripts.data.wave1.common.sampler import CompositeRetrievalAdapter
from src.trajectory_memory_v2._data import RetrievalSampler
from scripts.training._dashboard import DashboardRenderer


def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    # ── Data ──
    ap.add_argument("--composite-dir", type=str, default=None)
    ap.add_argument("--composite-val-dir", type=str, default=None)
    ap.add_argument("--train-jsonl", type=str, default=None)
    ap.add_argument("--val-jsonl", type=str, default=None)

    # ── Model ──
    ap.add_argument("--config-tier", type=str, default="medium",
                    choices=["small", "medium", "large"])
    ap.add_argument("--n-override", type=int, default=None,
                    help="Override cfg.N for ablations")

    # ── Training ──
    ap.add_argument("--num-steps", type=int, default=25000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-adapter", type=float, default=3e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")

    # ── Loss coefficients ──
    ap.add_argument("--load-balance-coef", type=float, default=1e-4)
    ap.add_argument("--z-loss-coef", type=float, default=1e-4)
    ap.add_argument("--contrast-coef", type=float, default=0.1)
    ap.add_argument("--contrast-temperature", type=float, default=0.07)
    ap.add_argument("--per-step-contrast-coef", type=float, default=0.05)
    ap.add_argument("--per-step-contrast-temperature", type=float, default=0.07)

    # ── Output / Logging ──
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--checkpoint-out", type=str, default=None)
    ap.add_argument("--checkpoint-in", type=str, default=None)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--plot-every-secs", type=float, default=180.0,
                    help="Re-render dashboard PNG every N wall-clock seconds "
                    "(0 disables). Requires --log-jsonl.")
    ap.add_argument("--plot-out", default=None,
                    help="Dashboard PNG path. Defaults to <log_jsonl>.plot.png.")
    ap.add_argument("--log-jsonl", type=str, default=None)
    ap.add_argument("--plot-path", type=str, default=None)
    ap.add_argument("--baselines-json", type=str, default=None)

    return ap.parse_args()


def _log_jsonl(path: str | None, record: dict):
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # ── Model ──
    cfg = getattr(TrajMemV2Config, args.config_tier)()
    if args.n_override is not None:
        cfg.N = args.n_override
    cfg.validate()
    print(f"Architecture: vocabulary-trajectory v2 (sparse edges)")
    print(f"Config: N={cfg.N}, D_concept={cfg.D_concept}, K_max={cfg.K_max}, "
          f"J={cfg.J}, K_read={cfg.K_read}, K_write={cfg.K_write}, "
          f"T_window={cfg.T_window}", flush=True)

    tokenizer = get_tokenizer()
    print("Loading model...", flush=True)
    model = IntegratedLMV2(cfg, model_name="meta-llama/Llama-3.2-1B").to(args.device)

    # ── Optimizer / scheduler ──
    # Two param groups: adapter (memory + entry_proj + walker), Llama if unfrozen
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

    trainer = Phase1RetrievalTrainerV2(
        model=model,
        optimizer=optimizer,
        pad_token_id=tokenizer.pad_token_id,
        scheduler=scheduler,
        grad_clip=args.grad_clip,
        load_balance_coef=args.load_balance_coef,
        z_loss_coef=args.z_loss_coef,
        contrast_coef=args.contrast_coef,
        contrast_temperature=args.contrast_temperature,
        per_step_contrast_coef=args.per_step_contrast_coef,
        per_step_contrast_temperature=args.per_step_contrast_temperature,
    )

    # ── Samplers ──
    if args.composite_dir is not None:
        cd = Path(args.composite_dir)
        train_sampler = CompositeRetrievalAdapter(
            cd / "passages.jsonl", cd / "questions.jsonl",
            chunk_size=8, seed=args.seed,
        )
        val_sampler = None
        if args.composite_val_dir is not None:
            vd = Path(args.composite_val_dir)
            val_sampler = CompositeRetrievalAdapter(
                vd / "passages.jsonl", vd / "questions.jsonl",
                chunk_size=8, seed=args.seed + 1,
            )
        print(f"Train composite: {len(train_sampler.facts)} questions")
        if val_sampler:
            print(f"Val composite:   {len(val_sampler.facts)} questions")
    else:
        train_sampler = RetrievalSampler(args.train_jsonl, seed=args.seed)
        val_sampler = (
            RetrievalSampler(args.val_jsonl, seed=args.seed + 1)
            if args.val_jsonl else None
        )

    # ── Resume / warm-start ──
    if args.checkpoint_in:
        ck = torch.load(args.checkpoint_in, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        trainer.load_state_dict(ck["trainer_state"])
        print(f"Resumed from {args.checkpoint_in} at step {trainer.step_count}")

    # ── Dashboard (background plot renderer) ──
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

    # ── Training loop ──
    t_start = time.time()
    loss_window: list[float] = []
    while trainer.step_count < args.num_steps:
        batch = train_sampler.sample_batch(args.batch_size)
        t_step_start = time.time()
        try:
            metrics = trainer.step(batch)
        except torch.cuda.OutOfMemoryError as e:
            print(f"FATAL: OOM at step {trainer.step_count}: {e}", file=sys.stderr)
            sys.exit(1)
        step_s = time.time() - t_step_start
        if not math.isfinite(metrics.loss):
            print(f"FATAL: non-finite loss at step {trainer.step_count}: {metrics.loss}",
                  file=sys.stderr)
            sys.exit(1)
        step = trainer.step_count
        loss_window.append(metrics.loss)
        if len(loss_window) > 50:
            loss_window.pop(0)
        wall_s_cumulative = time.time() - t_start

        # ── Log ──
        if step % args.log_every == 0 or step == 1:
            avg_loss = sum(loss_window) / len(loss_window)
            print(
                f"step {step:>5}  loss={metrics.loss:.4f}  ans={metrics.answer_loss:.4f}  "
                f"acc={metrics.answer_acc:.3f}  avg{len(loss_window)}={avg_loss:.4f}  "
                f"L_c={metrics.l_contrast_entry:.2f}  L_ps={metrics.l_contrast_per_step:.2f}  "
                f"RW={metrics.rw_overlap:.3f}  "
                f"edges={metrics.n_active_edges}({metrics.edge_active_fraction*100:.1f}%)  "
                f"fanout={metrics.mean_fan_out:.1f}  "
                f"spec={metrics.mean_edge_specificity:.2f}  "
                f"vc={metrics.mean_visit_count:.1f}  "
                f"grad={metrics.grad_norm:.2f}  "
                f"({step_s:.2f}s/step)",
                flush=True,
            )
        _log_jsonl(args.log_jsonl, {
            "step": step, "phase": "train",
            "loss": metrics.loss, "answer_loss": metrics.answer_loss,
            "answer_acc": metrics.answer_acc,
            "answer_tokens": metrics.answer_token_count,
            "aux_lb": metrics.aux_load_balance, "aux_z": metrics.aux_z_loss,
            "grad_norm": metrics.grad_norm,
            "l_contrast_entry": metrics.l_contrast_entry,
            "l_contrast_per_step": metrics.l_contrast_per_step,
            "rw_overlap": metrics.rw_overlap,
            "rw_overlap_target": metrics.rw_overlap_target,
            "rw_overlap_all": metrics.rw_overlap_all,
            "rw_overlap_entry": metrics.rw_overlap_entry,
            "rw_overlap_hop": metrics.rw_overlap_hop,
            "r_unique_per_traj": metrics.r_unique_per_traj,
            "w_unique_per_traj": metrics.w_unique_per_traj,
            "r_unique_per_window": metrics.r_unique_per_window,
            "w_unique_per_window": metrics.w_unique_per_window,
            "read_entry_entropy": metrics.read_entry_entropy,
            "write_entry_entropy": metrics.write_entry_entropy,
            "grad_norm_read": metrics.grad_norm_read,
            "grad_norm_write": metrics.grad_norm_write,
            "grad_norm_entry_proj": metrics.grad_norm_entry_proj,
            "grad_norm_lambda_edge": metrics.grad_norm_lambda_edge,
            "grad_norm_concept_ids": metrics.grad_norm_concept_ids,
            "grad_norm_mem_inject": metrics.grad_norm_mem_inject,
            "grad_norm_read_attn": metrics.grad_norm_read_attn,
            "concept_ids_norm_mean": metrics.concept_ids_norm_mean,
            "concept_ids_norm_cv": metrics.concept_ids_norm_cv,
            "concept_ids_pairwise_cos": metrics.concept_ids_pairwise_cos,
            "entry_logits_max": metrics.entry_logits_max,
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

        if dashboard:
            dashboard.maybe_render()

        # ── Validation ──
        if val_sampler is not None and step % args.val_every == 0:
            val_losses, val_accs, val_overlaps = [], [], []
            agg_loss: dict[str, list[float]] = {}
            agg_acc: dict[str, list[float]] = {}
            for _ in range(args.val_batches):
                vb = val_sampler.sample_batch(args.batch_size)
                vm = trainer.eval_step(vb)
                val_losses.append(vm.loss)
                val_accs.append(vm.answer_acc)
                val_overlaps.append(vm.rw_overlap)
                for key, n in vm.per_key_n.items():
                    agg_loss.setdefault(key, []).extend([vm.per_key_loss[key]] * n)
                    agg_acc.setdefault(key, []).extend([vm.per_key_acc[key]] * n)
            avg_val = sum(val_losses) / len(val_losses)
            avg_acc = sum(val_accs) / len(val_accs)
            avg_overlap = sum(val_overlaps) / len(val_overlaps)
            per_key_loss_agg = {k: sum(v) / len(v) for k, v in agg_loss.items()}
            per_key_acc_agg = {k: sum(v) / len(v) for k, v in agg_acc.items()}
            per_key_n_agg = {k: len(v) for k, v in agg_loss.items()}
            print(
                f"  [val @ step {step}] loss={avg_val:.4f}  acc={avg_acc:.3f}  "
                f"RW={avg_overlap:.3f}  ({len(val_losses)} batches)",
                flush=True,
            )
            _log_jsonl(args.log_jsonl, {
                "step": step, "phase": "val",
                "loss": avg_val,
                "answer_acc": avg_acc,
                "rw_overlap": avg_overlap,
                "n_batches": len(val_losses),
                "per_key_loss": per_key_loss_agg,
                "per_key_acc": per_key_acc_agg,
                "per_key_n": per_key_n_agg,
            })

        # ── Checkpoint ──
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

    # Final save
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
