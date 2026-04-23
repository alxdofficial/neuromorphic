#!/usr/bin/env python3
"""Benchmark the ColumnGraph training throughput across configurations."""

from __future__ import annotations

import argparse
import time

import torch

from src.column_graph.config import ColumnGraphConfig
from src.column_graph.standalone import StandaloneLM
from src.column_graph.train_phase1 import phase1_step


def bench_one(cfg: ColumnGraphConfig, bs: int, t: int, tbptt: int, steps: int, compile_hot: bool):
    lm = StandaloneLM(cfg).cuda()
    opt = torch.optim.AdamW(lm.parameters(), lr=1e-4, fused=True)

    if compile_hot:
        lm.memory._hot_forward = torch.compile(
            lm.memory._hot_forward, mode="default", fullgraph=False, dynamic=False
        )

    tokens = torch.randint(0, cfg.vocab_size, (bs, t), device="cuda")

    # Warmup
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
    label = f"compile={compile_hot} N={cfg.N} D_s={cfg.D_s} K={cfg.K} BS={bs} T={t} tbptt={tbptt}"
    print(f"{label}")
    print(f"   params={total/1e6:.1f}M  {dt*1000:.0f} ms/step  {bs*t/dt:.0f} tok/s  {peak:.2f} GB")

    del lm, opt, tokens
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=["dev", "small"], default="dev")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    if args.scale == "small":
        cfg = ColumnGraphConfig(
            plane_rows=16, plane_cols=16, L=3,
            K=16, D_s=128, D_id=16,
            num_tiles_per_plane_dim=2, vocab_size=8000,
        )
        configs = [(2, 32, 16), (4, 32, 16), (8, 32, 16)]
    else:
        cfg = ColumnGraphConfig()
        configs = [(1, 32, 16), (2, 32, 16), (2, 64, 16)]

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=== Uncompiled ===")
    for bs, t, tbptt in configs:
        try:
            bench_one(cfg, bs, t, tbptt, args.steps, compile_hot=False)
        except torch.cuda.OutOfMemoryError:
            print(f"   BS={bs} T={t} tbptt={tbptt}: OOM")
            torch.cuda.empty_cache()
        print()

    print("=== Compiled (hot_forward) ===")
    for bs, t, tbptt in configs:
        try:
            bench_one(cfg, bs, t, tbptt, args.steps, compile_hot=True)
        except torch.cuda.OutOfMemoryError:
            print(f"   BS={bs} T={t} tbptt={tbptt}: OOM")
            torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
