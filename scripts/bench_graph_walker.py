#!/usr/bin/env python3
"""Benchmark GraphWalker throughput at one known-reasonable config.

Defaults: measured-sweet-spot on RTX 4090 (24 GB): BS=48, T=128, tbptt=32.
Around 11k tok/s at 104M params. Override with CLI flags.
"""

from __future__ import annotations

import argparse
import os
import time

# Enable expandable segments before importing torch — the CUDA caching
# allocator reads this env var at first CUDA-context creation, which PyTorch
# does lazily on first GPU op. Set it early so any later torch.cuda calls
# pick up the flag. Gives ~13% throughput headroom by reducing fragmentation
# pressure when batch-size grows.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.standalone import StandaloneLM
from src.graph_walker.train_phase1 import phase1_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=48)
    parser.add_argument("--t", type=int, default=128)
    parser.add_argument("--tbptt", type=int, default=32)
    parser.add_argument("--mod-period", type=int, default=None)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    cfg_overrides = {}
    if args.mod_period is not None:
        cfg_overrides["mod_period"] = args.mod_period
    # segment_T must be a multiple of mod_period (config post-init enforces
    # this to prevent silently dropped partial plasticity windows). The
    # bench's T_seq plays the role of segment_T, so mirror it here; also
    # match tbptt_block to what the bench actually uses.
    cfg_overrides["segment_T"] = args.t
    cfg_overrides["tbptt_block"] = args.tbptt
    cfg = GraphWalkerConfig(**cfg_overrides)
    lm = StandaloneLM(cfg).cuda()
    lm.memory.compile_step()
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
    print(f"Config: N={cfg.N} D_model={cfg.D_model} D_state={cfg.D_s} K={cfg.K} "
          f"H={cfg.n_heads} persistent_hop/token=1")
    print(f"   params={total/1e6:.1f}M  BS={args.bs} T={args.t} tbptt={args.tbptt}")
    print(f"   {dt*1000:.0f} ms/step  {args.bs*args.t/dt:.0f} tok/s  {peak:.2f} GB")


if __name__ == "__main__":
    main()
