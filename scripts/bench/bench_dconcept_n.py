#!/usr/bin/env python3
"""Measure forward+backward step time as a function of D_concept × N.

Tests the hypothesis that D and N should be roughly free-scale operations
because most hot-path ops only touch visited nodes (J·K visits per
trajectory), not the whole manifold.

The only O(N) hot-path op is entry routing (cosine vs all N concept_ids).
The only O(BS·N·D) hot-path memory is the concept_states runtime buffer.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _bench_common import cleanup_cuda  # noqa: E402

from src.trajectory_memory.config import TrajMemConfig  # noqa: E402
from src.trajectory_memory.integrated_lm import IntegratedLM  # noqa: E402
from src.trajectory_memory.training.phase1 import Phase1Trainer  # noqa: E402


def run_one(D_concept: int, N: int, BS: int = 4, n_warm: int = 2, n_iter: int = 5):
    cleanup_cuda()
    torch.manual_seed(0)
    cfg = TrajMemConfig.medium()
    cfg.D_concept = D_concept
    cfg.N = N
    # K_max_neighbors is capped at N//4 by validation; ensure sensible
    cfg.K_max_neighbors = min(cfg.K_max_neighbors, max(1, N // 4))
    cfg.radius = min(cfg.radius, max(1, N // 8))
    cfg.validate()
    T_chunk = cfg.D * cfg.T_window

    model = IntegratedLM(cfg, model_name="meta-llama/Llama-3.2-1B")
    model = model.to("cuda")
    model.train(True)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4,
    )
    trainer = Phase1Trainer(
        model, optim, grad_clip=1.0,
        pad_token_id=128001,
        use_kv_cache=False,
        load_balance_coef=1e-3, z_loss_coef=0.0,
    )

    vocab = 128256
    chunk = torch.randint(0, vocab, (BS, T_chunk), device="cuda")

    def step():
        return trainer.step_wave1(chunk).loss

    for _ in range(n_warm):
        try:
            _ = step()
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)[:80], "D": D_concept, "N": N}

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(n_iter):
        _ = step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n_iter
    peak = torch.cuda.max_memory_allocated() / 1e9
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    del model, optim, trainer
    cleanup_cuda()
    gc.collect()

    return {
        "D": D_concept, "N": N,
        "ms_per_step": dt * 1000,
        "peak_vram_gb": peak,
        "trainable_M": trainable,
        "tok_per_sec": BS * T_chunk / dt,
    }


def main():
    print(f"=== D_concept × N speed bench (BS=4, medium config) ===\n", flush=True)
    print(f"{'D':>6} {'N':>8} {'ms/step':>10} {'tok/s':>10} {'peak GB':>10} {'train M':>10}", flush=True)
    print("-" * 60, flush=True)
    configs = [
        (256, 4096),    # current baseline
        (256, 16384),   # N×4
        (256, 65536),   # N×16
        (1024, 4096),   # D×4
        (1024, 16384),  # D×4, N×4
        (1024, 65536),  # D×4, N×16
    ]
    base = None
    for D, N in configs:
        print(f"running D={D}, N={N} ...", flush=True)
        r = run_one(D, N)
        if "error" in r:
            print(f"{D:>6} {N:>8}   ERROR {r['error']}", flush=True)
            continue
        if base is None:
            base = r["ms_per_step"]
        slowdown = r["ms_per_step"] / base
        print(
            f"{r['D']:>6} {r['N']:>8} "
            f"{r['ms_per_step']:>10.1f} "
            f"{r['tok_per_sec']:>10.0f} "
            f"{r['peak_vram_gb']:>10.2f} "
            f"{r['trainable_M']:>10.2f}  "
            f"({slowdown:.2f}× vs D=256,N=4096)",
            flush=True,
        )


if __name__ == "__main__":
    main()
