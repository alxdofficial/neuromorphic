#!/usr/bin/env python3
"""Measure forward+backward step time as a function of D_concept.

Compares D_concept ∈ {256, 512, 1024} with the rest of the config held
at medium defaults. Used to decide whether bumping D_concept for more
per-concept capacity is affordable.

Run as:
    PYTHONPATH=. python scripts/bench/bench_dconcept.py
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


def run_one_d(D_concept: int, BS: int = 4, n_warm: int = 2, n_iter: int = 5):
    cleanup_cuda()
    torch.manual_seed(0)
    cfg = TrajMemConfig.medium()
    cfg.D_concept = D_concept
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
            return {"error": str(e), "D_concept": D_concept}

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
        "D_concept": D_concept,
        "ms_per_step": dt * 1000,
        "peak_vram_gb": peak,
        "trainable_M": trainable,
        "tok_per_sec": BS * T_chunk / dt,
    }


def main():
    print(f"=== D_concept speed bench (BS=4, medium config) ===\n", flush=True)
    results = []
    for D in [256, 512, 1024]:
        print(f"Running D_concept={D} ...", flush=True)
        r = run_one_d(D)
        results.append(r)
        print(f"  {r}\n", flush=True)

    print(f"\n{'D_concept':<10} {'ms/step':>10} {'tok/s':>10} {'peak GB':>10} {'trainable M':>12}",
          flush=True)
    base_ms = next((r.get("ms_per_step") for r in results if "ms_per_step" in r), 1)
    for r in results:
        if "error" in r:
            print(f"{r['D_concept']:<10}  ERROR: {r['error']}", flush=True)
            continue
        slowdown = r["ms_per_step"] / base_ms
        print(
            f"{r['D_concept']:<10} "
            f"{r['ms_per_step']:>10.1f} "
            f"{r['tok_per_sec']:>10.0f} "
            f"{r['peak_vram_gb']:>10.2f} "
            f"{r['trainable_M']:>12.2f}  "
            f"({slowdown:.2f}× vs D=256)",
            flush=True,
        )


if __name__ == "__main__":
    main()
