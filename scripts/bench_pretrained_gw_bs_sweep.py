"""Sweep BS for vanilla Llama vs Llama+graph_walker.

For each BS in `--bs-list`, runs `bench_pretrained_gw.py --compile-block --bs N`
in a subprocess (separate process per BS so each one cleanly handles its own
OOM, compile cache, and HF model load). Parses the four headline numbers
(vanilla fwd/step, GW fwd/step) from stdout, tracks peak tok/s for each
path, and reports the peak/peak ratio at the end.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/bench_pretrained_gw_bs_sweep.py \\
        --bs-list 1 4 16 32 64
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
    oom: bool
    error: str | None


_LINE_RE = re.compile(
    r"^\s*(Llama-1B[^\s]*\s+(?:vanilla|\+ GW)\s+(?:fwd|step))\s+"
    r"([\d\.]+)k tok/s\s+peak\s+([\d\.]+) GB"
)


def _parse(stdout: str) -> dict[str, tuple[float, float]]:
    """Map descriptive label → (tok/s_in_thousands, peak_gb)."""
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
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return BSRow(bs, None, None, None, None, None, None, None, None,
                     False, f"timeout after {timeout}s")
    blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if "out of memory" in blob.lower() or "outofmemoryerror" in blob.lower():
        print(f"  OOM at BS={bs}", flush=True)
        return BSRow(bs, None, None, None, None, None, None, None, None,
                     True, None)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-2000:]
        print(f"  ERROR rc={proc.returncode}\n{tail}", flush=True)
        return BSRow(bs, None, None, None, None, None, None, None, None,
                     False, f"rc={proc.returncode}")
    parsed = _parse(proc.stdout)
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
        oom=False, error=None,
    )
    print(f"  vanilla fwd {row.vanilla_fwd_tps and row.vanilla_fwd_tps/1000:>5.1f}k "
          f"step {row.vanilla_step_tps and row.vanilla_step_tps/1000:>5.1f}k  |  "
          f"GW fwd {row.gw_fwd_tps and row.gw_fwd_tps/1000:>5.1f}k "
          f"step {row.gw_step_tps and row.gw_step_tps/1000:>5.1f}k"
          f"   (peak GW step {row.gw_step_gb} GB)",
          flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs-list", type=int, nargs="+",
                    default=[1, 4, 16, 32, 64])
    ap.add_argument("--timeout", type=int, default=1500,
                    help="Per-BS subprocess timeout in seconds.")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": repo}
    os.chdir(repo)

    rows: list[BSRow] = []
    for bs in args.bs_list:
        row = _run_one(bs, env, args.timeout)
        rows.append(row)
        if row.oom:
            # Don't bother with higher BS once OOM hits — bigger BS will too.
            print(f"  Skipping BS > {bs} (vanilla and/or GW already OOM'd).",
                  flush=True)
            break

    print()
    print("=" * 86)
    print(f"  {'BS':>4}  {'V-fwd':>8} {'V-step':>8}  {'GW-fwd':>8} {'GW-step':>8}  "
          f"{'V-step GB':>9} {'GW-step GB':>10}")
    print("=" * 86)
    for r in rows:
        if r.oom:
            print(f"  {r.bs:>4}  OOM")
            continue
        if r.error:
            print(f"  {r.bs:>4}  ERROR: {r.error}")
            continue
        def fmt(x): return f"{x/1000:>7.1f}k" if x is not None else "    n/a"
        def fgb(x): return f"{x:>8.2f}" if x is not None else "    n/a"
        print(f"  {r.bs:>4}  {fmt(r.vanilla_fwd_tps)} {fmt(r.vanilla_step_tps)}  "
              f"{fmt(r.gw_fwd_tps)} {fmt(r.gw_step_tps)}  "
              f"{fgb(r.vanilla_step_gb)} {fgb(r.gw_step_gb)}")
    print("=" * 86)

    # Peak/peak ratio (the production-meaningful number).
    def peak(metric: str) -> tuple[int, float] | None:
        best: tuple[int, float] | None = None
        for r in rows:
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
        if p is None:
            print(f"  {label}:  n/a")
        else:
            print(f"  {label}:  {p[1]/1000:6.1f}k tok/s  (BS={p[0]})")

    pv = peak("vanilla_step_tps")
    pg = peak("gw_step_tps")
    if pv is not None and pg is not None:
        print()
        print(f"  Step-mode peak slowdown:  {pv[1] / pg[1]:.2f}x  "
              f"(vanilla peak BS={pv[0]} -> +mem peak BS={pg[0]})")

    pvf = peak("vanilla_fwd_tps")
    pgf = peak("gw_fwd_tps")
    if pvf is not None and pgf is not None:
        print(f"  Forward-mode peak slowdown:  {pvf[1] / pgf[1]:.2f}x  "
              f"(vanilla peak BS={pvf[0]} -> +mem peak BS={pgf[0]})")


if __name__ == "__main__":
    main()
