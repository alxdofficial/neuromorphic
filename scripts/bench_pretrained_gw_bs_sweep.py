"""Find peak throughput per path for vanilla Llama vs Llama+graph_walker.

Two phases per path (vanilla and Llama+GW are tested by the same subprocess
since they share Llama load + compile cost):

  Phase A (exponential expansion).
    Start at `--anchor` BS. Run subprocess, parse the four headline
    metrics (vanilla fwd/step, GW fwd/step) from stdout. Each metric is
    independently OOM-tolerant — the bench script catches per-test OOM
    and prints "OOM" lines. As long as ANY metric still produces a
    number for some path, double the BS and try again. Stop doubling
    once both paths' step-mode metrics have OOM'd (i.e. the more
    expensive of the two paths has hit its cliff at the current BS for
    both fwd and step).

  Phase B (downward halving, only if anchor itself OOM'd).
    If even the anchor BS doesn't fit, halve until something fits. This
    is the "user picked too high an anchor" rescue path. Bounded at
    BS=1.

After exhaustion, every visited BS is a measurement point. Pick the
peak tok/s per (path, mode) over the visited points and report the
peak/peak ratio. That's the production-meaningful "how slow am I at
my best operating point" answer.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/bench_pretrained_gw_bs_sweep.py \\
        --anchor 8 --max-bs 256
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class BSRow:
    bs: int
    vanilla_fwd_tps: float | None
    vanilla_fwd_gb: float | None
    vanilla_step_tps: float | None
    vanilla_step_gb: float | None
    gw_fwd_tps: float | None
    gw_fwd_gb: float | None
    gw_step_tps: float | None
    gw_step_gb: float | None
    error: str | None


_LINE_RE = re.compile(
    r"^\s*(Llama-1B[^\s]*\s+(?:vanilla|\+ GW)\s+(?:fwd|step))\s+"
    r"([\d\.]+)k tok/s\s+peak\s+([\d\.]+) GB"
)


def _parse(stdout: str) -> dict[str, tuple[float, float]]:
    """Map descriptive label → (tok/s, peak_gb). Lines with 'OOM' are
    simply absent from the dict — caller infers OOM from missing keys."""
    out = {}
    for line in stdout.splitlines():
        m = _LINE_RE.match(line)
        if m:
            label, tps_k, gb = m.groups()
            out[label.strip()] = (float(tps_k) * 1000.0, float(gb))
    return out


def _run_one(bs: int, env: dict[str, str], timeout: int) -> BSRow:
    cmd = [
        ".venv/bin/python", "scripts/bench_pretrained_gw.py",
        "--compile-block", "--bs", str(bs),
    ]
    print(f"\n=== BS={bs} ===  ({' '.join(cmd)})", flush=True)
    error = None
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=timeout,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired:
        return BSRow(bs, *([None] * 8), error=f"timeout after {timeout}s")

    parsed = _parse(stdout)
    if proc.returncode != 0:
        # Subprocess crashed entirely — extract whatever we got from stdout
        # before the crash, but flag the error so the operator sees it.
        tail = (stderr or stdout)[-1500:]
        error = f"rc={proc.returncode}: {tail.strip()[-400:]}"
        print(f"  subprocess crashed (rc={proc.returncode}); "
              f"parsed {len(parsed)} metric(s) from partial stdout",
              flush=True)

    row = BSRow(
        bs=bs,
        vanilla_fwd_tps=parsed.get("Llama-1B vanilla fwd", (None, None))[0],
        vanilla_fwd_gb=parsed.get("Llama-1B vanilla fwd", (None, None))[1],
        vanilla_step_tps=parsed.get("Llama-1B vanilla step", (None, None))[0],
        vanilla_step_gb=parsed.get("Llama-1B vanilla step", (None, None))[1],
        gw_fwd_tps=parsed.get("Llama-1B + GW fwd", (None, None))[0],
        gw_fwd_gb=parsed.get("Llama-1B + GW fwd", (None, None))[1],
        gw_step_tps=parsed.get("Llama-1B + GW step", (None, None))[0],
        gw_step_gb=parsed.get("Llama-1B + GW step", (None, None))[1],
        error=error,
    )

    def fmt(x): return f"{x/1000:5.1f}k" if x is not None else " OOM "
    print(f"  vanilla fwd {fmt(row.vanilla_fwd_tps)} step {fmt(row.vanilla_step_tps)}  |  "
          f"GW fwd {fmt(row.gw_fwd_tps)} step {fmt(row.gw_step_tps)}",
          flush=True)
    return row


def _all_step_oom(row: BSRow) -> bool:
    """Both step-mode benches OOM'd → no headroom left for further doubling.
    We sweep based on step-mode (the binding constraint for training); if
    forward-only still fits but step doesn't, doubling is a waste.
    """
    return row.vanilla_step_tps is None and row.gw_step_tps is None


def _any_metric_present(row: BSRow) -> bool:
    return any(
        x is not None for x in [
            row.vanilla_fwd_tps, row.vanilla_step_tps,
            row.gw_fwd_tps, row.gw_step_tps,
        ]
    )


def adaptive_search(
    anchor: int, max_bs: int, env: dict[str, str], timeout: int,
) -> list[BSRow]:
    """Phase A: exponentially expand from anchor while step-mode fits.
    Phase B: if anchor itself doesn't fit, halve down to find a fitting BS.
    """
    rows: list[BSRow] = []
    visited: set[int] = set()

    def visit(bs: int) -> BSRow:
        visited.add(bs)
        r = _run_one(bs, env, timeout)
        rows.append(r)
        return r

    # Phase A — expand.
    bs = anchor
    last_row: BSRow | None = None
    while bs <= max_bs:
        r = visit(bs)
        last_row = r
        if _all_step_oom(r):
            print(f"  Both step-mode paths OOM'd at BS={bs}; stopping expansion.",
                  flush=True)
            break
        if not _any_metric_present(r):
            # Subprocess failed entirely (compile crash, env error, etc.) —
            # don't keep doubling blindly.
            print(f"  No metrics produced at BS={bs} (subprocess error); "
                  "stopping expansion.", flush=True)
            break
        bs *= 2

    # Phase B — anchor or expansion missed some metrics. Halve from the
    # anchor to fill in any metric that never produced a number during
    # expansion (typical case: GW step OOMs at anchor=8 but fits at BS=4).
    # Bounded by visited set so we don't bench the same BS twice.
    metrics_present = {
        m: any(getattr(r, m, None) is not None for r in rows)
        for m in (
            "vanilla_fwd_tps", "vanilla_step_tps",
            "gw_fwd_tps", "gw_step_tps",
        )
    }
    missing = [m for m, ok in metrics_present.items() if not ok]
    if missing:
        print(f"\n  Missing metrics from expansion: {missing}; halving from "
              f"anchor BS={anchor} to fill in.", flush=True)
        bs = anchor // 2
        while bs >= 1:
            if bs in visited:
                bs //= 2
                continue
            r = visit(bs)
            metrics_present = {
                m: any(getattr(rr, m, None) is not None for rr in rows)
                for m in metrics_present
            }
            still_missing = [m for m, ok in metrics_present.items() if not ok]
            if not still_missing:
                print(f"  All metrics now have at least one data point.",
                      flush=True)
                break
            bs //= 2

    return rows


def report(rows: list[BSRow]) -> None:
    rows_sorted = sorted(rows, key=lambda r: r.bs)
    print()
    print("=" * 88)
    print(f"  {'BS':>4}  {'V-fwd':>7} {'V-step':>7}  {'GW-fwd':>7} {'GW-step':>7}  "
          f"{'V-step GB':>9} {'GW-step GB':>10}")
    print("=" * 88)
    for r in rows_sorted:
        def fmt(x): return f"{x/1000:>6.1f}k" if x is not None else "    OOM"
        def fgb(x): return f"{x:>8.2f}" if x is not None else "    n/a"
        print(f"  {r.bs:>4}  {fmt(r.vanilla_fwd_tps)} {fmt(r.vanilla_step_tps)}  "
              f"{fmt(r.gw_fwd_tps)} {fmt(r.gw_step_tps)}  "
              f"{fgb(r.vanilla_step_gb)} {fgb(r.gw_step_gb)}"
              + (f"  ERR" if r.error else ""))
    print("=" * 88)

    def peak(metric: str) -> tuple[int, float] | None:
        best: tuple[int, float] | None = None
        for r in rows_sorted:
            v = getattr(r, metric, None)
            if v is None:
                continue
            if best is None or v > best[1]:
                best = (r.bs, v)
        return best

    print()
    print("Peak throughput per path (BS that produced the peak in parens):")
    for label, metric in [
        ("Vanilla Llama  fwd ", "vanilla_fwd_tps"),
        ("Vanilla Llama  step", "vanilla_step_tps"),
        ("Llama+GW       fwd ", "gw_fwd_tps"),
        ("Llama+GW       step", "gw_step_tps"),
    ]:
        p = peak(metric)
        print(f"  {label}:  "
              + (f"{p[1]/1000:6.1f}k tok/s  (BS={p[0]})" if p else "n/a"))

    pv = peak("vanilla_step_tps")
    pg = peak("gw_step_tps")
    if pv is not None and pg is not None:
        print()
        print(f"  Step-mode peak slowdown:    {pv[1] / pg[1]:.2f}x  "
              f"(vanilla peak BS={pv[0]} → +mem peak BS={pg[0]})")

    pvf = peak("vanilla_fwd_tps")
    pgf = peak("gw_fwd_tps")
    if pvf is not None and pgf is not None:
        print(f"  Forward-mode peak slowdown: {pvf[1] / pgf[1]:.2f}x  "
              f"(vanilla peak BS={pvf[0]} → +mem peak BS={pgf[0]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", type=int, default=8,
                    help="Starting BS for the sweep (default 8). Tries this "
                         "first; doubles up while step-mode fits, halves down "
                         "if even anchor OOMs.")
    ap.add_argument("--max-bs", type=int, default=512,
                    help="Hard cap on BS during the doubling expansion.")
    ap.add_argument("--timeout", type=int, default=1500,
                    help="Per-BS subprocess timeout in seconds.")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": repo}
    os.chdir(repo)

    rows = adaptive_search(
        anchor=args.anchor, max_bs=args.max_bs, env=env, timeout=args.timeout,
    )
    report(rows)


if __name__ == "__main__":
    main()
