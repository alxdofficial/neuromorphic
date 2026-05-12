"""Shared bench utilities (timing primitive, OOM cleanup).

`bench()` does the warmup + sync + peak-mem accounting; the bench scripts
(`bench_trajmem.py`, `bench_compare.py`) call it for every measurement.
"""

from __future__ import annotations

import gc
import time

import torch


def cleanup_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def bench(name: str, fn, n_warmup: int, n_iter: int, BS: int, T: int):
    """Time a callable. Returns (tok_per_sec, peak_gb, ms_per_iter) or
    (None, None, None) on OOM.

    OOM cleanup is critical so the next bench in the same process starts
    with a clean slate."""
    try:
        torch.cuda.synchronize()
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fn()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        print(f"  {name:48s}    OOM        peak n/a       BS={BS} T={T}")
        cleanup_cuda()
        return None, None, None
    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    tps = (BS * T * n_iter) / elapsed
    ms = elapsed / n_iter * 1000
    print(f"  {name:48s} {tps/1000:6.1f}k tok/s   peak {peak_gb:5.2f} GB   "
          f"{ms:7.1f} ms/iter   BS={BS} T={T}")
    return tps, peak_gb, ms
