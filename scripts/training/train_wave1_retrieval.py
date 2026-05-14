#!/usr/bin/env python3
"""Wave 1 v4 — retrieval pretraining entry point.

Trains trajectory-memory via write-then-retrieve QA on the synthetic
worldspec dataset. See `docs/wave1_retrieval_pretraining.md` for protocol.

Run (smoke):
    python3 scripts/training/train_wave1_retrieval.py \
        --train-jsonl data/wave1_retrieval/facts_train.jsonl \
        --val-jsonl data/wave1_retrieval/facts_val.jsonl \
        --num-steps 10 --batch-size 4 --log-every 1 --val-every 5

Run (real):
    python3 scripts/training/train_wave1_retrieval.py \
        --train-jsonl data/wave1_retrieval/facts_train.jsonl \
        --val-jsonl data/wave1_retrieval/facts_val.jsonl \
        --num-steps 80000 --batch-size 8 \
        --checkpoint-out outputs/wave1_retrieval/ckpt.pt \
        --log-every 50 --val-every 500
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.trajectory_memory.config import TrajMemConfig  # noqa: E402
from src.trajectory_memory.integrated_lm import IntegratedLM  # noqa: E402
from src.trajectory_memory.training.phase1_retrieval import (  # noqa: E402
    Phase1RetrievalTrainer, RetrievalSampler,
)


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-jsonl", type=str, required=True,
                    help="Path to facts_train.jsonl")
    ap.add_argument("--val-jsonl", type=str, default=None,
                    help="Path to facts_val.jsonl (optional)")
    ap.add_argument("--num-steps", type=int, default=80000)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="M chunks per gradient update")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Adam LR for memory-side modules")
    ap.add_argument("--lr-adapter", type=float, default=1e-4,
                    help="Adam LR for mem_inject bridge (typically same as lr)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--config-tier", type=str, default="medium",
                    choices=["small", "medium", "large"])
    ap.add_argument("--tbptt-depth", type=int, default=9,
                    help="TBPTT depth in windows. Set to 9 (or higher) for "
                         "this protocol so write-gradients flow through all "
                         "8 writes + 1 read.")
    # Bumped from 1e-2 → 3e-2. Standard MoE uses ~1e-2 for 64-128 experts;
    # we have N=4096 cells, so spreading routing across all of them needs
    # proportionally stronger pressure. Combats the observed 8.7%-cell-
    # utilization mode collapse in the trajectory model.
    ap.add_argument("--load-balance-coef", type=float, default=3e-2)
    ap.add_argument("--z-loss-coef", type=float, default=1e-3)
    ap.add_argument("--flat-bank", action="store_true",
                    help="Architectural ablation: use FlatReadModule / "
                         "FlatWriteModule (top-K cell attention, no graph "
                         "walks). Same manifold size, simpler mechanism.")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=20,
                    help="Number of val batches per validation cycle.")
    ap.add_argument("--checkpoint-out", type=str, default=None)
    ap.add_argument("--checkpoint-in", type=str, default=None,
                    help="Resume from this checkpoint (full state).")
    ap.add_argument("--warm-start", type=str, default=None,
                    help="Load only model weights (not optim/sched/step).")
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--log-jsonl", type=str, default=None,
                    help="If set, write per-step JSON metrics to this path "
                         "(for downstream plotting).")
    return ap.parse_args()


def build_optimizer(model: IntegratedLM, lr: float, lr_adapter: float):
    """Two parameter groups: memory-side modules + mem_inject bridge.
    LR can differ for the bridge if desired (typically same)."""
    memory_params, adapter_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # mem_inject layer parameters live under "llama." path; everything
        # else with requires_grad is memory-side (read/write/manifold).
        if "mem_inject" in n or "memory_fn" in n or n.startswith("llama."):
            adapter_params.append(p)
        else:
            memory_params.append(p)
    groups = []
    if memory_params:
        groups.append({"params": memory_params, "lr": lr})
    if adapter_params:
        groups.append({"params": adapter_params, "lr": lr_adapter})
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)


class WarmupCosineLR:
    """Simple warmup-then-cosine LR schedule."""
    def __init__(self, optimizer, warmup_steps, total_steps, base_lrs=None):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.step_count = 0
        self.base_lrs = base_lrs or [g["lr"] for g in optimizer.param_groups]

    def step(self):
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            scale = self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps,
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * scale

    def state_dict(self):
        return {"step_count": self.step_count, "base_lrs": self.base_lrs}

    def load_state_dict(self, state):
        self.step_count = state["step_count"]
        self.base_lrs = state["base_lrs"]


def _log_jsonl(path: str | None, record: dict):
    if path is None:
        return
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # ── Model ──
    cfg = getattr(TrajMemConfig, args.config_tier)()
    cfg.D = max(cfg.D, args.tbptt_depth)
    cfg.flat_bank = args.flat_bank
    cfg.validate()
    print(f"Architecture: {'flat-bank (top-K)' if cfg.flat_bank else 'trajectory (graph walks)'}")
    print(f"Config: N={cfg.N}, D_concept={cfg.D_concept}, J={cfg.J}, "
          f"T_window={cfg.T_window}, D={cfg.D}, "
          f"effective_lm_context={cfg.effective_lm_context}", flush=True)

    tokenizer = get_tokenizer()
    print("Loading model...", flush=True)
    model = IntegratedLM(cfg, model_name="meta-llama/Llama-3.2-1B").to(args.device)
    model.train(True)
    # Activation checkpointing on Llama — required for BS=8 with 9 stacked
    # forwards.
    if hasattr(model.llama, "gradient_checkpointing_enable"):
        model.llama.gradient_checkpointing_enable()
        print("  Llama gradient_checkpointing enabled")

    optimizer = build_optimizer(model, lr=args.lr, lr_adapter=args.lr_adapter)
    scheduler = WarmupCosineLR(
        optimizer, warmup_steps=args.warmup_steps, total_steps=args.num_steps,
    )
    trainer = Phase1RetrievalTrainer(
        model, optimizer,
        pad_token_id=tokenizer.pad_token_id,
        scheduler=scheduler,
        grad_clip=args.grad_clip,
        load_balance_coef=args.load_balance_coef,
        z_loss_coef=args.z_loss_coef,
    )

    # ── Samplers ──
    train_sampler = RetrievalSampler(args.train_jsonl, seed=args.seed)
    val_sampler = None
    if args.val_jsonl:
        val_sampler = RetrievalSampler(args.val_jsonl, seed=args.seed + 1)
    print(f"Train pool: {len(train_sampler.facts)} facts, "
          f"{len(train_sampler.keys)} distinct (class,attr) keys")
    if val_sampler:
        print(f"Val pool:   {len(val_sampler.facts)} facts, "
              f"{len(val_sampler.keys)} distinct (class,attr) keys")

    # ── Resume / warm-start ──
    if args.checkpoint_in:
        ck = torch.load(args.checkpoint_in, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        trainer.load_state_dict(ck["trainer_state"])
        print(f"Resumed from {args.checkpoint_in} at step {trainer.step_count}")
    elif args.warm_start:
        ck = torch.load(args.warm_start, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        print(f"Warm-started from {args.warm_start}")

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
        step_s = time.time() - t_step_start   # pure trainer.step() time, no val/log overhead
        if not math.isfinite(metrics.loss):
            print(f"FATAL: non-finite loss ({metrics.loss}) at step "
                  f"{trainer.step_count}. Aborting.", file=sys.stderr)
            sys.exit(1)
        loss_window.append(metrics.loss)
        if len(loss_window) > 50:
            loss_window.pop(0)
        step = trainer.step_count
        wall_s_cumulative = time.time() - t_start

        # ── Log ──
        if step % args.log_every == 0 or step == 1:
            avg_loss = sum(loss_window) / len(loss_window)
            print(
                f"step {step:>5}  loss={metrics.loss:.4f}  acc={metrics.answer_acc:.3f}  "
                f"avg{len(loss_window)}={avg_loss:.4f}  "
                f"r_uf={metrics.r_uf:.3f}  w_uf={metrics.w_uf:.3f}  "
                f"w_gn={metrics.w_gn:.3f}  r_gn={metrics.r_gn:.3f}  mi_gn={metrics.mi_gn:.3f}  "
                f"R↔T={metrics.read_target_overlap:.3f}  "
                f"mi_scale={metrics.mem_inject_scale:.3f}  "
                f"|s|={metrics.state_norm_mean:.2f}±{metrics.state_norm_std:.2f}  "
                f"grad={metrics.grad_norm:.2f}  "
                f"({step_s:.2f}s/step)",
                flush=True,
            )
        _log_jsonl(args.log_jsonl, {
            "step": step, "phase": "train",
            "loss": metrics.loss, "answer_tokens": metrics.answer_token_count,
            "aux_lb": metrics.aux_load_balance, "aux_z": metrics.aux_z_loss,
            "grad_norm": metrics.grad_norm,
            "r_uf": metrics.r_uf, "w_uf": metrics.w_uf,
            "r_ent": metrics.r_ent,
            "w_gn": metrics.w_gn, "r_gn": metrics.r_gn, "mi_gn": metrics.mi_gn,
            "mem_inject_scale": metrics.mem_inject_scale,
            "read_logit_scale": metrics.read_logit_scale,
            "write_logit_scale": metrics.write_logit_scale,
            "answer_acc": metrics.answer_acc,
            "read_target_overlap": metrics.read_target_overlap,
            "state_norm_mean": metrics.state_norm_mean,
            "state_norm_std": metrics.state_norm_std,
            "step_s": step_s,
            "wall_s_cumulative": wall_s_cumulative,
        })

        # ── Validation ──
        if val_sampler and step % args.val_every == 0:
            val_losses, val_accs, val_overlaps = [], [], []
            # Aggregate per-class loss/acc across val batches.
            agg_loss: dict[str, list[float]] = {}
            agg_acc: dict[str, list[float]] = {}
            for _ in range(args.val_batches):
                vb = val_sampler.sample_batch(args.batch_size)
                vm = trainer.eval_step(vb)
                val_losses.append(vm.loss)
                val_accs.append(vm.answer_acc)
                val_overlaps.append(vm.read_target_overlap)
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
                f"read↔target_overlap={avg_overlap:.3f}  (over {len(val_losses)} batches)",
                flush=True,
            )
            _log_jsonl(args.log_jsonl, {
                "step": step, "phase": "val",
                "loss": avg_val,
                "answer_acc": avg_acc,
                "read_target_overlap": avg_overlap,
                "n_batches": len(val_losses),
                "per_key_loss": per_key_loss_agg,
                "per_key_acc": per_key_acc_agg,
                "per_key_n": per_key_n_agg,
            })

        # ── Save ──
        if args.checkpoint_out and step > 0 and step % args.save_every == 0:
            save_path = Path(args.checkpoint_out)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "trainer_state": trainer.state_dict(),
                "step": step,
                "config": cfg.__dict__,
            }, save_path)
            print(f"  saved ckpt @ step {step} → {save_path}", flush=True)

    # Final save.
    if args.checkpoint_out:
        save_path = Path(args.checkpoint_out)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "trainer_state": trainer.state_dict(),
            "step": trainer.step_count,
            "config": cfg.__dict__,
        }, save_path)
        print(f"Final ckpt → {save_path}", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
