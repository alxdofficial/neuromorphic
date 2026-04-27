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


def bench_one(B: int, mode: str, segment_T_override: int | None = None) -> tuple[float, float, float] | None:
    """Returns (warmup_s, throughput_tps, peak_GB) or None on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        # Cudagraph path requires use_neuromod=False; eager keeps neuromod on.
        # eager mode disables torch.compile so we measure RUNTIME memory and
        # speed (not compile-time autotune scratch, which spikes >>10 GB).
        kw = {
            "use_neuromod": (mode != "cudagraph"),
            "compile_on_train": (mode == "cudagraph"),
        }
        if segment_T_override is not None:
            kw["segment_T"] = segment_T_override
        cfg = GraphWalkerConfig(**kw)

        print(f"  [B={B} mode={mode}] building LM (topology + params)...",
              flush=True)
        t_init = time.perf_counter()
        lm = StandaloneLM(cfg, verbose=True).cuda()
        print(f"  [B={B} mode={mode}] LM built in {time.perf_counter() - t_init:.1f}s",
              flush=True)

        opt = torch.optim.Adam(lm.parameters(), lr=1e-4)
        tokens = torch.randint(0, cfg.vocab_size, (B, cfg.segment_T), device="cuda")
        step_fn = phase1_step_cudagraph if mode == "cudagraph" else phase1_step

        print(f"  [B={B} mode={mode}] warmup {3 if mode == 'eager' else 1} iter(s)...",
              flush=True)
        n_warm = 3 if mode == "eager" else 1
        t0 = time.perf_counter()
        if mode == "cudagraph":
            # Pass verbose=True deep into the cudagraph trainer so heartbeat
            # prints during the long compile + capture warmup.
            from src.graph_walker.triton.cudagraph_trainer import CapturedBlockTrainer
            # First call builds + warms up + captures via phase1_step_cudagraph;
            # we override verbose by patching the trainer call site lightly.
            orig_warmup = CapturedBlockTrainer.warmup_and_capture
            def _verbose_warmup(self, n_warmup=3):
                return orig_warmup(self, n_warmup=n_warmup, verbose=True)
            CapturedBlockTrainer.warmup_and_capture = _verbose_warmup
            try:
                for i in range(n_warm):
                    t_iter = time.perf_counter()
                    step_fn(lm, opt, tokens, training_step=i)
                    torch.cuda.synchronize()
                    print(f"    warmup {i+1}/{n_warm} done in "
                          f"{time.perf_counter() - t_iter:.1f}s", flush=True)
            finally:
                CapturedBlockTrainer.warmup_and_capture = orig_warmup
        else:
            for i in range(n_warm):
                t_iter = time.perf_counter()
                step_fn(lm, opt, tokens, training_step=i)
                torch.cuda.synchronize()
                print(f"    warmup {i+1}/{n_warm} done in "
                      f"{time.perf_counter() - t_iter:.1f}s", flush=True)
        warm = time.perf_counter() - t0

        n_iters = 5 if mode == "eager" else 10
        print(f"  [B={B} mode={mode}] timing {n_iters} iters...", flush=True)
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
    # Override segment_T to keep bench wall-time sane. Throughput is
    # tok/s, so the value chosen doesn't bias the throughput number much.
    bench_segment_T = 256
    print(f"\n[eager — supports neuromod, no compile, segment_T={bench_segment_T}]",
          flush=True)
    print(f"{'B':>4} {'tok/s':>10} {'tok/s/B':>9} {'peak GB':>9} {'warmup s':>9}",
          flush=True)
    print("-" * 50, flush=True)
    last_ok_eager = None
    eager_results = {}
    for B in Bs:
        r = bench_one(B, "eager", segment_T_override=bench_segment_T)
        if r is None:
            print(f"{B:>4} OOM", flush=True)
            break
        warm, tps, peak = r
        print(f"{B:>4} {tps:>10.0f} {tps/B:>9.1f} {peak:>9.2f} {warm:>9.1f}",
              flush=True)
        eager_results[B] = (tps, peak)
        last_ok_eager = B

    # Try cudagraph at the largest B that fit eager (cudagraph uses LESS VRAM).
    if last_ok_eager:
        print(f"\n[cudagraph — use_neuromod=False, ~250s+ compile per new shape]",
              flush=True)
        r = bench_one(last_ok_eager, "cudagraph", segment_T_override=bench_segment_T)
        if r is None:
            print(f"  OOM at cudagraph B={last_ok_eager}", flush=True)
        else:
            warm, tps, peak = r
            e_tps, e_peak = eager_results[last_ok_eager]
            print(f"  B={last_ok_eager}  cudagraph={tps:.0f} tok/s   "
                  f"eager={e_tps:.0f} tok/s   speedup={tps/e_tps:.1f}×   "
                  f"peak {peak:.2f} GB (eager {e_peak:.2f} GB)",
                  flush=True)


if __name__ == "__main__":
    main()
