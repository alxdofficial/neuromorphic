"""Bench walker at the NEW production defaults (D_s=1024, K=64, N=2304).

For each B in the list, time eager phase1_step. Stops the eager probe at
OOM. Then tries one cudagraph capture at the largest fitting B for the
"absolute peak" number.

Usage:
    PYTHONPATH=. python scripts/bench_walker_prod.py [Bs...]
"""

from __future__ import annotations

import gc
import sys
import time

import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.standalone import StandaloneLM
from src.graph_walker.train_phase1 import phase1_step, phase1_step_cudagraph


def bench_one(B: int, mode: str) -> tuple[float, float, float] | None:
    """Returns (warmup_s, throughput_tps, peak_GB) or None on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        cfg = GraphWalkerConfig(use_neuromod=(mode == "eager"))
        # Cudagraph path requires use_neuromod=False; eager path supports both
        # but we keep neuromod ON for eager so we measure the real thing.
        lm = StandaloneLM(cfg).cuda()
        opt = torch.optim.Adam(lm.parameters(), lr=1e-4)
        tokens = torch.randint(0, cfg.vocab_size, (B, cfg.segment_T), device="cuda")

        step_fn = phase1_step_cudagraph if mode == "cudagraph" else phase1_step

        t0 = time.perf_counter()
        for i in range(3):
            step_fn(lm, opt, tokens, training_step=i)
        torch.cuda.synchronize()
        warm = time.perf_counter() - t0

        n_iters = 5 if mode == "eager" else 10
        start = time.perf_counter()
        for i in range(n_iters):
            step_fn(lm, opt, tokens, training_step=10 + i)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        tps = n_iters * B * cfg.segment_T / elapsed
        peak = torch.cuda.max_memory_allocated() / 1024**3
        return warm, tps, peak
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower():
            return None
        raise
    finally:
        try:
            del lm, opt, tokens
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    if not torch.cuda.is_available():
        print("no cuda")
        return

    cfg = GraphWalkerConfig()
    n_params = sum(p.numel() for p in StandaloneLM(cfg).parameters()) / 1e6
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(
        f"=== walker production-scale bench ===\n"
        f"  config: D_s={cfg.D_s}, D_id={cfg.D_id}, K={cfg.K}, N={cfg.N}, "
        f"depth={cfg.content_mlp_depth}, post_depth={cfg.post_model_depth}\n"
        f"  segment_T={cfg.segment_T}, mod_period={cfg.mod_period}, "
        f"K_horizons={cfg.K_horizons}\n"
        f"  ~{n_params:.0f}M params, {total_vram:.1f} GB GPU",
        flush=True,
    )

    Bs = [int(x) for x in sys.argv[1:]] or [1, 2, 4, 8, 16, 32]
    print(f"\n[eager — supports neuromod, no compile]", flush=True)
    print(f"{'B':>4} {'tok/s':>10} {'tok/s/B':>9} {'peak GB':>9} {'warmup s':>9}",
          flush=True)
    print("-" * 50, flush=True)
    last_ok_eager = None
    for B in Bs:
        r = bench_one(B, "eager")
        if r is None:
            print(f"{B:>4} OOM", flush=True)
            break
        warm, tps, peak = r
        print(f"{B:>4} {tps:>10.0f} {tps/B:>9.1f} {peak:>9.2f} {warm:>9.1f}",
              flush=True)
        last_ok_eager = B

    # Try cudagraph at the largest B that fit eager (cudagraph uses LESS VRAM).
    if last_ok_eager:
        print(f"\n[cudagraph — use_neuromod=False, ~250s+ compile per new shape]",
              flush=True)
        print(f"  trying B={last_ok_eager} ...", flush=True)
        r = bench_one(last_ok_eager, "cudagraph")
        if r is None:
            print(f"  OOM at cudagraph B={last_ok_eager}", flush=True)
        else:
            warm, tps, peak = r
            print(f"  B={last_ok_eager}  tok/s={tps:>10.0f}  "
                  f"tok/s/B={tps/last_ok_eager:>7.1f}  peak={peak:>5.2f} GB  "
                  f"warmup={warm:>5.1f}s", flush=True)


if __name__ == "__main__":
    main()
