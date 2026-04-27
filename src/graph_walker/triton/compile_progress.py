"""Heartbeat helper to show progress during long torch.compile + cudagraph
warmups.

Usage:
    with compile_progress("phase1 cudagraph capture", interval_s=15):
        trainer.warmup_and_capture(n_warmup=3)

Prints something like:
    [phase1 cudagraph capture]   15s  GPU 4.2 GB  CPU workers active
    [phase1 cudagraph capture]   30s  GPU 4.2 GB  CPU workers active
    [phase1 cudagraph capture]   60s  GPU 4.2 GB  CPU workers active   (still compiling)
    [phase1 cudagraph capture]  120s  GPU 4.2 GB  CPU workers active
    [phase1 cudagraph capture]  done in 387.4s
"""

from __future__ import annotations

import contextlib
import os
import threading
import time

import torch


def _count_inductor_workers() -> int:
    """Best-effort count of inductor's parallel compile worker processes."""
    try:
        # Inductor spawns subprocess workers; count them via ps.
        import subprocess
        out = subprocess.check_output(
            ["pgrep", "-fc", "torch._inductor.compile_worker"],
            stderr=subprocess.DEVNULL,
        )
        return int(out.strip())
    except Exception:
        return 0


def _format_gb(bytes_: int) -> str:
    return f"{bytes_ / 1024**3:.2f} GB"


@contextlib.contextmanager
def compile_progress(label: str, interval_s: float = 15.0):
    """Print a heartbeat every ``interval_s`` seconds while the wrapped block
    runs. Helpful for long torch.compile + cudagraph capture warmups where
    PyTorch otherwise gives no progress signal.
    """
    start = time.perf_counter()
    stop_evt = threading.Event()

    def _heartbeat() -> None:
        last_print = start
        while not stop_evt.is_set():
            stop_evt.wait(timeout=1.0)
            now = time.perf_counter()
            if now - last_print >= interval_s:
                elapsed = now - start
                gpu_used = (
                    torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                )
                workers = _count_inductor_workers()
                worker_msg = (
                    f"  workers: {workers} active" if workers > 0
                    else "  (cache hit?)"
                )
                # Pad elapsed time so columns line up across many lines.
                hint = "  (still compiling)" if elapsed > 60 else ""
                print(
                    f"  [{label}] {elapsed:>6.0f}s  GPU {_format_gb(gpu_used)}"
                    f"{worker_msg}{hint}",
                    flush=True,
                )
                last_print = now

    t = threading.Thread(target=_heartbeat, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop_evt.set()
        t.join(timeout=2.0)
        elapsed = time.perf_counter() - start
        print(f"  [{label}] done in {elapsed:.1f}s", flush=True)


def configure_compile_logging(enable: bool = True) -> None:
    """Turn on PyTorch's built-in compilation metrics.

    When enabled, PyTorch logs each Dynamo + Inductor compile event with
    timing — useful to see ``what`` is being compiled, not just ``how long``.
    Set ``enable=False`` to silence.
    """
    if not enable:
        os.environ.pop("TORCH_COMPILE_DEBUG", None)
        os.environ.pop("TORCH_LOGS", None)
        return
    # `recompiles` shows when shape specialisation triggers a fresh compile;
    # `compile_metrics` prints per-frame compile time. Together they're
    # informative without the firehose of `+inductor`.
    os.environ["TORCH_LOGS"] = "recompiles,graph_breaks,perf_hints"
