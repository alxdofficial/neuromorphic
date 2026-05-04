#!/usr/bin/env python3
"""Training entry-point for the graph-walker integration.

Drives `phase1_pretrained_step` (parallel teacher-forced) over a chosen
data source. Supports two waves:

  - ``--data fineweb-edu``  (Wave 1) — natural-text bootstrap on
    ``data/phase_B/fineweb_edu.parquet`` (or any local FineWeb-edu parquet).
    On-the-fly Llama-3.2 tokenization.

  - ``--data ultrachat``    (Wave 2) — instruction/chat SFT on
    ``HuggingFaceH4/ultrachat_200k`` via the chat template.

Telemetry is captured via ``StatsCollector`` (writes a per-step jsonl
to ``--work-dir``). Checkpointing is bare-bones for v1: wrapper +
optimizer state every ``--ckpt-every`` steps; data iterator state is
not persisted (re-init from corpus start on resume — fine for
randomized FineWeb sampling).

LR schedule: linear warmup + cosine decay to 10% of peak (HF style).

Smoke usage:
    PYTHONPATH=. .venv/bin/python scripts/train_pretrained_gw.py \\
        --data fineweb-edu --max-steps 200 --bs 4 --T 256 \\
        --work-dir /tmp/gw_smoke --no-compile-block

Production usage (Wave 1 — 100M tokens):
    PYTHONPATH=. .venv/bin/python scripts/train_pretrained_gw.py \\
        --data fineweb-edu --max-steps 20000 --bs 20 --T 256 \\
        --work-dir outputs/wave1_fineweb --lr 1e-4 --warmup 200 \\
        --ckpt-every 1000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.data.phase1_loaders import (
    chat_sft_phase1_iter,
    fineweb_edu_phase1_iter,
)
from src.graph_walker.pretrained.config import PretrainedGWConfig
from src.graph_walker.pretrained.llm_wrapper import GraphWalkerPretrainedLM
from src.graph_walker.pretrained.train_phase1 import phase1_pretrained_step
from src.graph_walker.telemetry import StatsCollector


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # Model / config
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--inject-layer", type=int, default=8)
    ap.add_argument("--d-mem", type=int, default=256)
    ap.add_argument("--bs", type=int, default=20)
    ap.add_argument("--T", type=int, default=256)

    # Data
    ap.add_argument(
        "--data", choices=("fineweb-edu", "ultrachat"), required=True,
        help="Wave 1 = fineweb-edu; Wave 2 = ultrachat.",
    )
    ap.add_argument(
        "--fineweb-parquet",
        default="data/phase_B/fineweb_edu.parquet",
        help="Path to local FineWeb-edu parquet (Wave 1 only).",
    )
    ap.add_argument(
        "--ultrachat-name",
        default="HuggingFaceH4/ultrachat_200k",
        help="HF dataset name for chat SFT (Wave 2 only).",
    )
    ap.add_argument(
        "--chat-tokenizer",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help=(
            "Tokenizer to use for chat-template encoding (Wave 2 only). "
            "Llama-3.2-1B-Instruct has the chat_template; the base model's "
            "tokenizer does not. Token IDs are identical to the base model "
            "since they share a tokenizer."
        ),
    )

    # Training
    ap.add_argument("--max-steps", type=int, default=20_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lr-min-ratio", type=float, default=0.1,
                    help="Cosine decay floor as a fraction of peak LR.")

    # Compile / perf
    ap.add_argument("--compile-block", action="store_true", default=True,
                    help="Compile the walker block forward (production default).")
    ap.add_argument("--no-compile-block", dest="compile_block",
                    action="store_false")

    # Checkpointing / telemetry
    ap.add_argument("--work-dir", required=True,
                    help="Directory for telemetry jsonl + checkpoints.")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--resume", default=None,
                    help="Path to a checkpoint .pt to resume from.")

    return ap.parse_args()


def _build_lr_lambda(warmup: int, total_steps: int, min_ratio: float):
    """Linear warmup + cosine decay to `min_ratio * peak_lr`."""
    def fn(step: int) -> float:
        if step < warmup:
            return float(step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine
    return fn


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] work_dir={work_dir}")

    # --- Wrapper construction (production config) ---
    print(f"[setup] building wrapper: {args.model} inject_layer={args.inject_layer} "
          f"d_mem={args.d_mem} BS={args.bs} T={args.T}")
    cfg = PretrainedGWConfig.llama_1b(
        model_name=args.model,
        inject_layer=args.inject_layer,
        d_mem=args.d_mem,
        T=args.T,
        bs=args.bs,
    )
    wrapper = GraphWalkerPretrainedLM(cfg).to(device)
    wrapper.train(True)
    n_trainable = sum(p.numel() for _, p in wrapper.trainable_parameters())
    print(f"[setup] trainable params: {n_trainable/1e6:.1f}M")

    if args.compile_block and device.type == "cuda":
        print("[setup] compiling walker block (this takes a few minutes)...")
        wrapper.compile_walker_block()

    # --- Optimizer + LR schedule ---
    opt = torch.optim.AdamW(
        [p for _, p in wrapper.trainable_parameters()],
        lr=args.lr,
        fused=(device.type == "cuda"),
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=_build_lr_lambda(args.warmup, args.max_steps, args.lr_min_ratio),
    )

    # --- Resume (optional) ---
    start_step = 0
    if args.resume is not None:
        print(f"[setup] resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        wrapper.load_state_dict(ckpt["wrapper"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_step = int(ckpt["step"])
        print(f"[setup] resumed at step {start_step}")

    # --- Data iterator ---
    print(f"[setup] data source: {args.data}")
    if args.data == "fineweb-edu":
        # Fineweb-edu uses the BASE Llama-3.2 tokenizer (no chat template needed).
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        data_iter = fineweb_edu_phase1_iter(
            args.fineweb_parquet, tokenizer,
            bs=args.bs, T=args.T, device=device,
        )
    elif args.data == "ultrachat":
        # The base Llama-3.2-1B tokenizer has NO chat_template; the
        # `-Instruct` variant does. Token IDs are identical (shared
        # tokenizer), so using the Instruct tokenizer for encoding is
        # safe even though we wrap the base model.
        tokenizer = AutoTokenizer.from_pretrained(args.chat_tokenizer)
        data_iter = chat_sft_phase1_iter(
            args.ultrachat_name, tokenizer,
            bs=args.bs, T=args.T, device=device,
        )
    else:
        raise ValueError(args.data)

    # --- Train loop with telemetry ---
    print(f"[train] {args.max_steps - start_step} steps · "
          f"{args.bs * args.T} tokens/step · "
          f"~{(args.max_steps - start_step) * args.bs * args.T / 1e6:.1f}M tokens total")
    t_start = time.perf_counter()
    last_log_t = t_start

    with StatsCollector(work_dir=work_dir) as collector:
        step = start_step
        for batch in data_iter:
            stats = phase1_pretrained_step(
                wrapper, opt, batch,
                amp_dtype=torch.bfloat16 if device.type == "cuda" else None,
                grad_clip=args.grad_clip,
            )
            sched.step()

            row = collector.snapshot(
                wrapper, step=step, phase="phase1",
                stats=stats,
            )

            if step % args.log_every == 0:
                now = time.perf_counter()
                tps = (args.bs * args.T * args.log_every) / max(now - last_log_t, 1e-6)
                last_log_t = now
                print(
                    f"[step {step:>6}] loss={stats.loss:.4f} "
                    f"ce={stats.ce_loss:.4f} "
                    f"lr={sched.get_last_lr()[0]:.2e} "
                    f"grad={stats.grad_norm:.2f} "
                    f"inj={stats.inject_residual_norm:.2e} "
                    f"tps={tps/1000:.1f}k",
                    flush=True,
                )

            if step > 0 and step % args.ckpt_every == 0:
                ckpt_path = work_dir / f"ckpt_step{step}.pt"
                torch.save({
                    "step": step,
                    "wrapper": wrapper.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "config": vars(args),
                }, ckpt_path)
                print(f"[ckpt] saved {ckpt_path}", flush=True)

            step += 1
            if step >= args.max_steps:
                break

    # Final ckpt
    final_path = work_dir / "ckpt_final.pt"
    torch.save({
        "step": step,
        "wrapper": wrapper.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "config": vars(args),
    }, final_path)
    elapsed = time.perf_counter() - t_start
    print(f"[done] {step - start_step} steps in {elapsed/60:.1f} min · "
          f"final ckpt: {final_path}")


if __name__ == "__main__":
    main()
