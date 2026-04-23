#!/usr/bin/env python3
"""Smoke test — overfit ColumnGraph on a tiny random batch; report throughput."""

from __future__ import annotations

import argparse
import math
import time

import torch

from src.column_graph.config import ColumnGraphConfig
from src.column_graph.standalone import StandaloneLM
from src.column_graph.train_phase1 import phase1_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--t", type=int, default=64)
    parser.add_argument("--tbptt", type=int, default=32)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--amp", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--scale", choices=["dev", "small", "med"], default="dev")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    torch.manual_seed(0)
    if args.scale == "small":
        cfg = ColumnGraphConfig(
            plane_rows=16, plane_cols=16, L=3,
            K=16, D_s=128, D_id=16,
            num_tiles_per_plane_dim=2, vocab_size=8000,
        )
    elif args.scale == "dev":
        cfg = ColumnGraphConfig()    # defaults: 4 planes × 32×32, D_s=256, K=32
    else:  # med
        cfg = ColumnGraphConfig(
            plane_rows=48, plane_cols=48, L=6, K=48, D_s=384,
            num_tiles_per_plane_dim=6,
        )

    lm = StandaloneLM(cfg).cuda()
    total = sum(p.numel() for p in lm.parameters())
    emb = lm.token_emb.weight.numel()
    neuromod_params = sum(p.numel() for p in lm.memory.neuromod.parameters())
    column_mlps = sum(p.numel() for p in lm.memory.mlps.parameters())
    attn_io = (
        sum(p.numel() for p in lm.memory.in_q_proj.parameters())
        + sum(p.numel() for p in lm.memory.in_k_proj.parameters())
        + sum(p.numel() for p in lm.memory.in_v_proj.parameters())
        + sum(p.numel() for p in lm.memory.out_k_proj.parameters())
        + sum(p.numel() for p in lm.memory.out_v_proj.parameters())
    )
    plastic = lm.memory.E_bias_flat.numel() if hasattr(lm.memory, "E_bias_flat") else 0

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: planes={cfg.L}×{cfg.plane_rows}×{cfg.plane_cols}={cfg.N} cols  "
          f"D_s={cfg.D_s} D_id={cfg.D_id} K={cfg.K} V={cfg.vocab_size} T=1")
    print(f"Params: total={total/1e6:.2f}M  emb={emb/1e6:.2f}M  "
          f"col_mlps={column_mlps/1e6:.2f}M  "
          f"neuromod={neuromod_params/1e6:.2f}M  attn_io={attn_io/1e6:.2f}M  "
          f"plastic={plastic/1e3:.0f}K scalars")
    print()

    opt = torch.optim.AdamW(lm.parameters(), lr=args.lr, fused=True)
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else None

    tokens = torch.randint(0, cfg.vocab_size, (args.bs, args.t), device="cuda")
    baseline = math.log(cfg.vocab_size)

    print(f"Overfit: BS={args.bs}, T={args.t}, tbptt={args.tbptt}, "
          f"lr={args.lr}, amp={args.amp}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    losses = []
    for step in range(args.steps):
        stats = phase1_step(
            lm, opt, tokens,
            tbptt_block=args.tbptt, amp_dtype=amp_dtype,
        )
        losses.append(stats.loss)
        if step % 5 == 0 or step == args.steps - 1:
            elapsed = time.time() - t0
            dt = elapsed / (step + 1)
            toks_s = args.bs * args.t / dt
            k1 = stats.per_horizon_loss[0] if len(stats.per_horizon_loss) > 0 else float('nan')
            print(f"  step {step:3d}  loss={stats.loss:.4f}  "
                  f"k1={k1:.3f}  "
                  f"|grad|={stats.grad_norm:.2f}  "
                  f"{dt*1000:.0f} ms/step  "
                  f"{toks_s:.0f} tok/s")

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print()
    print(f"Summary: {losses[0]:.3f} → {losses[-1]:.3f}  "
          f"(Δ = {losses[0] - losses[-1]:+.3f})")
    print(f"Baseline log(V): {baseline:.3f}")
    print(f"Peak VRAM: {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
