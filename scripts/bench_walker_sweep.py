"""Find max-throughput batch via binary search; report VRAM overhead.

Strategy (each cudagraph probe is ~250s of Triton autotune — keep it cheap):
  1. Start with a candidate B (default 128).
  2. If it fits, double until OOM, then bisect downward.
  3. If it OOMs, halve until it fits, then bisect upward.
  4. At the winning B, also measure eager throughput + VRAM for overhead.

Probe budget: typically 3 cudagraph captures + 1 eager run.

Usage:
    PYTHONPATH=. python scripts/bench_walker_sweep.py [start_B] [max_probes]
"""

from __future__ import annotations

import gc
import sys
import time

import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.standalone import StandaloneLM
from src.graph_walker.train_phase1 import phase1_step, phase1_step_cudagraph


def make_lm() -> StandaloneLM:
    cfg = GraphWalkerConfig(
        plane_rows=8, plane_cols=8, L=4, K=16,
        D_model=512, D_s=256, D_id=32,
        n_heads=4, n_score_heads=4, D_q_per_head=32, D_q_in=32,
        K_horizons=4, K_buf=4,
        mod_period=64, tbptt_block=64, segment_T=128,
        ffn_mult_content=4,
        content_mlp_depth=2,
        post_model_depth=1,
        vocab_size=1024,
        use_neuromod=False,
        compile_on_train=False,
        state_dtype="bf16",
    )
    return StandaloneLM(cfg).cuda()


def time_one(B: int, mode: str) -> tuple[float, float, float] | None:
    """Returns (warmup_s, throughput_tps, peak_GB) or None on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        lm = make_lm()
        cfg = lm.cfg
        opt = torch.optim.Adam(lm.parameters(), lr=1e-4)
        tokens = torch.randint(0, cfg.vocab_size, (B, cfg.segment_T), device="cuda")

        step_fn = phase1_step_cudagraph if mode == "cudagraph" else phase1_step

        t0 = time.perf_counter()
        for i in range(3):
            step_fn(lm, opt, tokens, training_step=i)
        torch.cuda.synchronize()
        warm = time.perf_counter() - t0

        n_iters = 10
        start = time.perf_counter()
        for i in range(n_iters):
            step_fn(lm, opt, tokens, training_step=10 + i)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        tps = n_iters * B * cfg.segment_T / elapsed
        peak = torch.cuda.max_memory_allocated() / 1024**3
        return warm, tps, peak
    except torch.OutOfMemoryError:
        return None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
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

    start_B = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    max_probes = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(
        f"=== walker max-throughput probe (GPU {total_vram:.1f} GB) ===",
        flush=True,
    )
    print(
        f"{'B':>5} {'mode':>10} {'tok/s':>10} {'tok/s/B':>9} "
        f"{'peak GB':>9} {'warmup s':>9}",
        flush=True,
    )
    print("-" * 60, flush=True)

    # ---------- Phase 1: bracket the OOM threshold (cudagraph) -----------
    results: dict[int, tuple[float, float, float]] = {}
    probes = 0
    lower_ok = 0           # largest B known to fit
    upper_oom = None       # smallest B known to OOM

    B = start_B
    while probes < max_probes and (upper_oom is None or upper_oom - lower_ok > 1):
        r = time_one(B, "cudagraph")
        probes += 1
        if r is None:
            print(f"{B:>5} {'cudagraph':>10} {'OOM':>10}", flush=True)
            upper_oom = B
            # Halve: try midpoint between known-ok and OOM
            if lower_ok > 0:
                B = (lower_ok + upper_oom) // 2
            else:
                B = max(B // 2, 1)
        else:
            warm, tps, peak = r
            print(
                f"{B:>5} {'cudagraph':>10} {tps:>10.0f} {tps/B:>9.1f} "
                f"{peak:>9.2f} {warm:>9.1f}",
                flush=True,
            )
            results[B] = r
            lower_ok = max(lower_ok, B)
            if upper_oom is None:
                B = B * 2          # double until OOM
            else:
                B = (lower_ok + upper_oom) // 2

    # Pick the best (highest tok/s) cudagraph result.
    if not results:
        print("no cudagraph result fit", flush=True)
        return
    best_B, (best_warm, best_tps, best_peak) = max(
        results.items(), key=lambda kv: kv[1][1],
    )

    # ---------- Phase 2: eager comparison at the winning B ----------
    print(flush=True)
    print(f"--- eager comparison at B={best_B} for VRAM overhead ---",
          flush=True)
    eager = time_one(best_B, "eager")
    if eager is None:
        print(f"{best_B:>5} {'eager':>10} {'OOM':>10}", flush=True)
        eager_tps = eager_peak = None
    else:
        e_warm, eager_tps, eager_peak = eager
        print(
            f"{best_B:>5} {'eager':>10} {eager_tps:>10.0f} "
            f"{eager_tps/best_B:>9.1f} {eager_peak:>9.2f} {e_warm:>9.1f}",
            flush=True,
        )

    # ---------- Summary ----------
    print(flush=True)
    print("=== summary ===", flush=True)
    print(f"  probes used (cudagraph): {probes}/{max_probes}", flush=True)
    print(
        f"  peak throughput config: B={best_B}, "
        f"{best_tps:.0f} tok/s, peak {best_peak:.2f} GB",
        flush=True,
    )
    if eager_tps is not None:
        speedup = best_tps / eager_tps
        vram_overhead = best_peak - eager_peak
        vram_overhead_pct = 100.0 * vram_overhead / eager_peak
        print(
            f"  speedup vs eager:       {speedup:.2f}x "
            f"({eager_tps:.0f} → {best_tps:.0f} tok/s)",
            flush=True,
        )
        print(
            f"  VRAM overhead vs eager: {vram_overhead:+.2f} GB "
            f"({vram_overhead_pct:+.1f}%)",
            flush=True,
        )
    if upper_oom is not None:
        print(f"  OOM threshold:          B>={upper_oom}", flush=True)


if __name__ == "__main__":
    main()
