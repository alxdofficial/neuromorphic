"""Trajectory-memory Phase 1 (W1 long-doc TF NTP) throughput bench.

Skips vanilla Llama paths (A/B/C) — hardware-bound and already known
from prior measurements (see `abandoned/graph-walker` branch's
`docs/bench_results.md` for those numbers). This bench only exercises
the trajmem-specific path (D).

Per project memory:
- "Bench with fixed params, never sweep" — one config tier per run.
- "Bench at each path's own optimal BS" — find max-fitting BS via
  doubling, report peak tok/s at that BS.

Default: medium config tier (v1 default per plan §4.4), BS-doubling sweep
from anchor=1 until OOM. Reports tok/s + peak GB at each BS.

Usage:
    # Full sweep at v1-default config (medium)
    PYTHONPATH=. python scripts/bench_trajmem.py

    # Single BS, no sweep (e.g., to validate a chosen production BS)
    PYTHONPATH=. python scripts/bench_trajmem.py --bs 4 --no-sweep

    # With torch.compile on the chunk forward
    PYTHONPATH=. python scripts/bench_trajmem.py --compile

    # Different tier (small / medium / large)
    PYTHONPATH=. python scripts/bench_trajmem.py --config-tier large
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _bench_common import bench, cleanup_cuda  # noqa: E402

from src.trajectory_memory.config import TrajMemConfig  # noqa: E402
from src.trajectory_memory.integrated_lm import IntegratedLM  # noqa: E402
from src.trajectory_memory.training import (  # noqa: E402
    Phase1Trainer, build_optimizer,
)


def _make_step_fn(trainer: Phase1Trainer, BS: int, T_chunk: int, vocab: int, device):
    """Build a closure that runs one Phase1Trainer.step_wave1 on random ids."""
    chunk = torch.randint(0, vocab, (BS, T_chunk), device=device)

    def step_fn():
        trainer.step_wave1(chunk)

    return step_fn


def run_one_bs(
    args, BS: int, *, llama_dtype: torch.dtype, device,
) -> tuple[float | None, float | None]:
    """Build a fresh model + optimizer + trainer at this BS, run the bench,
    return (tok/s, peak_gb) or (None, None) on OOM during construction or run.

    Each BS gets its own model instance so prior-BS state doesn't leak."""
    cfg = getattr(TrajMemConfig, args.config_tier)()
    T_chunk = cfg.D * cfg.T_window
    try:
        model = IntegratedLM(cfg, model_name=args.model, attach_lm=True).to(device)
        if args.compile:
            # Compile the per-window step. forward_window is the tightest
            # hot loop; run_chunk iterates D times over it.
            model.forward_window = torch.compile(
                model.forward_window, mode=args.compile_mode, dynamic=False,
            )
        optimizer = build_optimizer(
            model, lr_memory=3e-4, lr_adapter=1e-4,
        )
        trainer = Phase1Trainer(model, optimizer, scheduler=None, grad_clip=1.0)
    except torch.cuda.OutOfMemoryError:
        print(f"  [BS={BS}]                                          OOM during model build")
        cleanup_cuda()
        return None, None

    label = (
        f"trajmem step (tier={args.config_tier}"
        + (", compile" if args.compile else "")
        + f", BS={BS})"
    )
    step_fn = _make_step_fn(trainer, BS, T_chunk, model.llama.config.vocab_size, device)
    tps, peak_gb, _ = bench(
        label, step_fn, args.warmup, args.iter, BS, T_chunk,
    )

    del trainer, optimizer, model
    cleanup_cuda()
    return tps, peak_gb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--config-tier", choices=["small", "medium", "large"],
                    default="medium")
    ap.add_argument("--bs", type=int, default=1,
                    help="Anchor BS (sweep starts here) or fixed BS (with --no-sweep)")
    ap.add_argument("--no-sweep", dest="sweep", action="store_false",
                    help="Run only at --bs; skip the doubling sweep")
    ap.add_argument("--max-bs", type=int, default=64,
                    help="Cap for the doubling sweep")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the per-window forward")
    ap.add_argument("--compile-mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iter", type=int, default=10)
    ap.set_defaults(sweep=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    cfg = getattr(TrajMemConfig, args.config_tier)()
    T_chunk = cfg.D * cfg.T_window
    print(f"\n=== trajectory-memory Phase 1 throughput bench ===")
    print(f"  device:      {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    print(f"  model:       {args.model}")
    print(f"  config tier: {args.config_tier}  "
          f"(N={cfg.N}, J={cfg.J}, K_read={cfg.K_read}, "
          f"D={cfg.D}, T_window={cfg.T_window}, chunk={T_chunk} tokens)")
    print(f"  compile:     {'on (' + args.compile_mode + ')' if args.compile else 'off'}")
    print(f"  warmup={args.warmup}, iter={args.iter}")
    print()

    if not args.sweep:
        print(f"Running fixed BS={args.bs}.")
        run_one_bs(args, args.bs, llama_dtype=torch.bfloat16, device=device)
        return

    # ── Sweep mode: doubling from anchor until OOM ────────────────────
    print("Sweeping BS (doubling from anchor until OOM)...")
    rows: list[tuple[int, float | None, float | None]] = []
    bs = args.bs
    while bs <= args.max_bs:
        tps, mem = run_one_bs(args, bs, llama_dtype=torch.bfloat16, device=device)
        rows.append((bs, tps, mem))
        if tps is None:
            print(f"  -> OOM at BS={bs}; stopping sweep.")
            break
        bs *= 2

    # ── Summary ───────────────────────────────────────────────────────
    print()
    print("=== Summary ===")
    print(f"  {'BS':>6}  {'tok/s':>10}  {'peak GB':>10}")
    peak_tps = 0.0
    peak_bs = None
    for bs, tps, mem in rows:
        if tps is None:
            print(f"  {bs:>6}  {'OOM':>10}  {'-':>10}")
        else:
            print(f"  {bs:>6}  {tps/1000:>9.1f}k  {mem:>9.2f}")
            if tps > peak_tps:
                peak_tps = tps
                peak_bs = bs

    if peak_bs is not None:
        print()
        print(f"  Peak throughput: BS={peak_bs}  ->  {peak_tps/1000:.1f}k tok/s")
        print(f"  Recommended training BS at config tier '{args.config_tier}'"
              f"{' + compile' if args.compile else ''}: {peak_bs}")


if __name__ == "__main__":
    main()
