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
to ``--work-dir``). Checkpointing is bare-bones for v1: model +
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

from pathlib import Path as _Path
from src.data.phase1_loaders import (
    chat_sft_phase1_iter,
    fineweb_edu_phase1_iter,
    pretokenized_phase1_iter,
)
from src.graph_walker.pretrained.config import PretrainedGWConfig
from src.graph_walker.pretrained.integrated_lm import IntegratedLM
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
        "--data", choices=("fineweb-edu", "ultrachat"),
        required=True,
        help="Wave 1 = fineweb-edu; Wave 2 = ultrachat. (Passphrase SFT "
             "was retired — see Wave 3 chat-injected GRPO via "
             "scripts/train_grpo.py.)",
    )
    ap.add_argument(
        "--fineweb-parquet",
        default="data/phase_B/fineweb_edu.parquet",
        help="Path to local FineWeb-edu parquet (Wave 1).",
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
    ap.add_argument("--seed", type=int, default=42, help="RNG seed.")

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
    ap.add_argument("--no-compile-block", dest="compile_walk_block",
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
    print(f"[setup] building model: {args.model} inject_layer={args.inject_layer} "
          f"d_mem={args.d_mem} BS={args.bs} T={args.T}")
    cfg = PretrainedGWConfig.llama_1b(
        model_name=args.model,
        inject_layer=args.inject_layer,
        d_mem=args.d_mem,
        T=args.T,
        bs=args.bs,
    )
    model = IntegratedLM(cfg).to(device)
    model.train(True)
    n_trainable = sum(p.numel() for _, p in model.trainable_parameters())
    print(f"[setup] trainable params: {n_trainable/1e6:.1f}M")

    if args.compile_walk_block and device.type == "cuda":
        print("[setup] compiling walker block (this takes a few minutes)...")
        model.compile_walker_block()

    # --- Optimizer + LR schedule ---
    opt = torch.optim.AdamW(
        [p for _, p in model.trainable_parameters()],
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
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_step = int(ckpt["step"])
        print(f"[setup] resumed at step {start_step}")

    # --- Data iterator ---
    print(f"[setup] data source: {args.data}")
    if args.data == "fineweb-edu":
        # Auto-detect pretokenized cache (~2x faster steady-state vs the
        # streaming-parquet+tokenize path). The preprocess script
        # `scripts/preprocess_fineweb_edu_llama32.py` writes the .bin.
        pre_bin = _Path("data/phase_B/fineweb_edu_llama32.bin")
        if pre_bin.exists():
            print(f"[setup] using pretokenized {pre_bin}")
            data_iter = pretokenized_phase1_iter(
                pre_bin, bs=args.bs, T=args.T, device=device, seed=args.seed,
            )
        else:
            print(f"[setup] pretokenized {pre_bin} missing — "
                  f"streaming parquet (slower)")
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            data_iter = fineweb_edu_phase1_iter(
                args.fineweb_parquet, tokenizer,
                bs=args.bs, T=args.T, device=device,
            )
    elif args.data == "ultrachat":
        # Auto-detect pretokenized cache. The .bin already has chat
        # templates baked in; per-batch tokenize+template is skipped.
        pre_bin = _Path("data/phase_B/ultrachat_llama32.bin")
        if pre_bin.exists():
            print(f"[setup] using pretokenized {pre_bin}")
            data_iter = pretokenized_phase1_iter(
                pre_bin, bs=args.bs, T=args.T, device=device, seed=args.seed,
            )
        else:
            print(f"[setup] pretokenized {pre_bin} missing — "
                  f"streaming HF dataset (slower; first run will download)")
            # The base Llama-3.2-1B tokenizer has NO chat_template; the
            # `-Instruct` variant does. Token IDs are identical (shared
            # tokenizer).
            tokenizer = AutoTokenizer.from_pretrained(args.chat_tokenizer)
            data_iter = chat_sft_phase1_iter(
                args.ultrachat_name, tokenizer,
                bs=args.bs, T=args.T, device=device,
            )
    else:
        raise ValueError(args.data)

    # --- Train loop with telemetry ---
    tokens_per_step = args.bs * args.T
    print(f"[train] {args.max_steps - start_step} steps · "
          f"{tokens_per_step} tokens/step · "
          f"~{(args.max_steps - start_step) * tokens_per_step / 1e6:.1f}M tokens total"
          f" · step_fn=phase1_pretrained_step")
    t_start = time.perf_counter()
    last_log_t = t_start

    with StatsCollector(work_dir=work_dir) as collector:
        step = start_step
        for batch in data_iter:
            stats = phase1_pretrained_step(
                model, opt, batch,
                amp_dtype=torch.bfloat16 if device.type == "cuda" else None,
                grad_clip=args.grad_clip,
            )
            sched.step()

            row = collector.snapshot(
                model, step=step, phase="phase1",
                stats=stats,
            )

            if step % args.log_every == 0:
                now = time.perf_counter()
                tps = (tokens_per_step * args.log_every) / max(now - last_log_t, 1e-6)
                last_log_t = now
                # Surface diagnostic flags from row[] dict so a quick
                # console scan catches dead walker / NaN / VRAM bumps
                # without leaving the terminal.
                vram = row.get("vram.peak_mb", 0.0)
                nan_g = row.get("nan.any_nan_grad", False)
                nan_p = row.get("nan.any_nan_param", False)
                has_dnm = row.get("walker.has_active_neuromod_delta", True)
                # Compact warning string; empty when everything healthy.
                warns = []
                if nan_g:  warns.append("NaN-grad")
                if nan_p:  warns.append("NaN-param")
                if not has_dnm:  warns.append("no-delta_nm")
                warn_str = (" ⚠ " + ",".join(warns)) if warns else ""
                inj_ratio = getattr(stats, "inject_residual_ratio", 0.0)
                print(
                    f"[step {step:>6}] loss={stats.loss:.4f} "
                    f"ce={stats.ce_loss:.4f} "
                    f"lr={sched.get_last_lr()[0]:.2e} "
                    f"grad={stats.grad_norm:.2f} "
                    f"inj_ratio={inj_ratio:.2e} "
                    f"vram={vram/1024:.1f}GB "
                        f"tps={tps/1000:.1f}k"
                        f"{warn_str}",
                        flush=True,
                    )

            if step > 0 and step % args.ckpt_every == 0:
                ckpt_path = work_dir / f"ckpt_step{step}.pt"
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
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
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "config": vars(args),
    }, final_path)
    elapsed = time.perf_counter() - t_start
    print(f"[done] {step - start_step} steps in {elapsed/60:.1f} min · "
          f"final ckpt: {final_path}")


if __name__ == "__main__":
    main()
