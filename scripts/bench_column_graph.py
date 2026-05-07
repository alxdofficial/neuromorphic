#!/usr/bin/env python3
"""Benchmark ColumnGraph throughput at a known-reasonable config.

Defaults: dev scale, BS=16, T=128, tbptt=16, compiled. One config only.
Use --sweep to get the old multi-config matrix.
"""

from __future__ import annotations

import argparse
import time

import torch

from src.column_graph.config import ColumnGraphConfig
from src.column_graph.standalone import StandaloneLM
from src.column_graph.train_phase1 import phase1_step


def bench_one(cfg: ColumnGraphConfig, bs: int, t: int, tbptt: int, steps: int, compile_walk_block: bool):
    lm = StandaloneLM(cfg).cuda()
    opt = torch.optim.AdamW(lm.parameters(), lr=1e-4, fused=True)

    if compile_walk_block:
        lm.memory.compile_walk_block(mode="default")

    tokens = torch.randint(0, cfg.vocab_size, (bs, t), device="cuda")

    for _ in range(5):
        phase1_step(lm, opt, tokens, tbptt_block=tbptt, amp_dtype=torch.bfloat16)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        phase1_step(lm, opt, tokens, tbptt_block=tbptt, amp_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 1024**3

    total = sum(p.numel() for p in lm.parameters())
    print(f"compile={compile_walk_block} N={cfg.N} D_s={cfg.D_s} K={cfg.K} "
          f"BS={bs} T={t} tbptt={tbptt}")
    print(f"   params={total/1e6:.1f}M  {dt*1000:.0f} ms/step  "
          f"{bs*t/dt:.0f} tok/s  {peak:.2f} GB")

    del lm, opt, tokens
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--t", type=int, default=128)
    parser.add_argument("--tbptt", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="Run the old multi-config sweep (slow; many compile warmups).")
    args = parser.parse_args()

    cfg = ColumnGraphConfig()
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.sweep:
        configs = [(1, 32, 16), (4, 32, 16), (8, 128, 16), (16, 128, 16)]
        print("=== Uncompiled ===")
        for bs, t, tbptt in configs:
            try:
                bench_one(cfg, bs, t, tbptt, args.steps, compile_walk_block=False)
            except torch.cuda.OutOfMemoryError:
                print(f"   BS={bs} T={t}: OOM"); torch.cuda.empty_cache()
            print()
        print("=== Compiled (block) ===")
        for bs, t, tbptt in configs:
            try:
                bench_one(cfg, bs, t, tbptt, args.steps, compile_walk_block=True)
            except torch.cuda.OutOfMemoryError:
                print(f"   BS={bs} T={t}: OOM"); torch.cuda.empty_cache()
            print()
    else:
        bench_one(cfg, args.bs, args.t, args.tbptt, args.steps,
                  compile_walk_block=not args.no_compile)


if __name__ == "__main__":
    main()
