#!/usr/bin/env python3
"""Smoke test: overfit a tiny batch on the graph-walker."""

from __future__ import annotations

import argparse
import math
import time

import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.standalone import StandaloneLM
from src.graph_walker.train_phase1 import phase1_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--t", type=int, default=64)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scale", choices=["small", "dev"], default="small")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    torch.manual_seed(0)
    if args.scale == "small":
        cfg = GraphWalkerConfig(
            plane_rows=8, plane_cols=8, L=3, K=16,
            D_s=128, D_id=16, n_heads=2, n_hops=3,
            D_q_in=32, D_q_per_head=32, n_score_heads=2,
            vocab_size=2000,
        )
    else:
        cfg = GraphWalkerConfig()    # defaults

    lm = StandaloneLM(cfg).cuda()
    opt = torch.optim.AdamW(lm.parameters(), lr=args.lr, fused=True)

    total = sum(p.numel() for p in lm.parameters())
    emb = lm.token_emb.weight.numel()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: N={cfg.N} ({cfg.L} planes x {cfg.plane_rows}x{cfg.plane_cols})  "
          f"D_s={cfg.D_s} D_id={cfg.D_id} K={cfg.K}  "
          f"H={cfg.n_heads} L_walk={cfg.n_hops}  V={cfg.vocab_size}")
    print(f"Params: total={total/1e6:.2f}M  emb={emb/1e6:.2f}M  rest={(total-emb)/1e6:.2f}M")
    print()

    tokens = torch.randint(0, cfg.vocab_size, (args.bs, args.t), device="cuda")
    baseline = math.log(cfg.vocab_size)
    print(f"Overfit: BS={args.bs}, T={args.t}, lr={args.lr}, log(V)={baseline:.3f}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    losses = []
    for step in range(args.steps):
        stats = phase1_step(
            lm, opt, tokens,
            tbptt_block=cfg.tbptt_block, amp_dtype=torch.bfloat16,
            training_step=step,
        )
        losses.append(stats.loss)
        if step % 5 == 0 or step == args.steps - 1:
            elapsed = time.time() - t0
            dt = elapsed / (step + 1)
            toks_s = args.bs * args.t / dt
            print(f"  step {step:3d}  loss={stats.loss:.3f} "
                  f"(ce={stats.ce_loss:.3f}, bal={stats.load_balance_loss:.3f})  "
                  f"visit_H/log_N={stats.visit_entropy:.2f}  "
                  f"|grad|={stats.grad_norm:.2f}  "
                  f"{dt*1000:.0f} ms/step  {toks_s:.0f} tok/s")

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print()
    print(f"Summary: {losses[0]:.3f} → {losses[-1]:.3f}  (Δ={losses[0]-losses[-1]:+.3f})")
    print(f"Baseline log(V)={baseline:.3f}")
    print(f"Peak VRAM: {peak:.2f} GB")


if __name__ == "__main__":
    main()
