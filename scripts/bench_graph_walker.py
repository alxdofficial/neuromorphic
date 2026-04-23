#!/usr/bin/env python3
"""Benchmark GraphWalker throughput at one known-reasonable config.

Defaults: dev scale, BS=16, T=128, tbptt=16.
"""

from __future__ import annotations

import argparse
import time

import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.standalone import StandaloneLM
from src.graph_walker.train_phase1 import phase1_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--t", type=int, default=128)
    parser.add_argument("--tbptt", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    cfg = GraphWalkerConfig()
    lm = StandaloneLM(cfg).cuda()
    opt = torch.optim.AdamW(lm.parameters(), lr=1e-4, fused=True)

    tokens = torch.randint(0, cfg.vocab_size, (args.bs, args.t), device="cuda")

    for _ in range(5):
        phase1_step(lm, opt, tokens, tbptt_block=args.tbptt,
                    amp_dtype=torch.bfloat16, training_step=0)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for s in range(args.steps):
        phase1_step(lm, opt, tokens, tbptt_block=args.tbptt,
                    amp_dtype=torch.bfloat16, training_step=s)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / args.steps
    peak = torch.cuda.max_memory_allocated() / 1024**3

    total = sum(p.numel() for p in lm.parameters())
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: N={cfg.N} D_s={cfg.D_s} K={cfg.K} "
          f"H={cfg.n_heads} L_walk={cfg.n_hops}")
    print(f"   params={total/1e6:.1f}M  BS={args.bs} T={args.t} tbptt={args.tbptt}")
    print(f"   {dt*1000:.0f} ms/step  {args.bs*args.t/dt:.0f} tok/s  {peak:.2f} GB")


if __name__ == "__main__":
    main()
