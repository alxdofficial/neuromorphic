#!/usr/bin/env python3
"""Fair-reporting tool: fit val loss curve per variant and extrapolate
the asymptotic L_infinity to compare architectures that converge at
different rates.

Reads the per-variant jsonl logs from a tranche/run output directory,
fits L(s) = a * s^(-b) + L_inf to each variant's val_loss_recon series,
and reports four numbers per variant:
  - observed_final_val     : actual val_recon at last training step
  - observed_best_val      : min val_recon over all eval points
                              past max(1000, n_steps/10)
  - extrap_L_inf           : asymptote of fitted power-law (compute-fair
                              estimate of "what would it converge to")
  - extrap_steps_to_within : extra steps the fit suggests are needed to
                              get within 5% of L_inf (cap at infinity)

This is the methodology from Domhan 2015 + Hoffmann 2022 (Chinchilla):
report both observed and extrapolated so reviewers can judge how much
slack each architecture has.

Usage:
  python scripts/repr_learning/extrap_val_curve.py outputs/repr_qa/v1h_t4k
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


def power_law(s, a, b, l_inf):
    """L(s) = a * s^(-b) + L_inf. Constrained: a > 0, b > 0, L_inf > 0."""
    return a * np.power(s, -b) + l_inf


def fit_one(steps: list[int], losses: list[float]) -> tuple:
    """Return (a, b, L_inf, l_inf_lower, l_inf_upper) or (None,) * 5 on
    failure. Bounds keep the fit physical."""
    s = np.array(steps, dtype=float)
    L = np.array(losses, dtype=float)
    if len(s) < 4:
        return (None, None, None, None, None)

    # Reasonable bounds. L_inf lower bound = 0.3: no QA val_recon plausibly
    # converges below this (we're predicting one-of-vocab tokens; even a
    # very confident model has finite CE). Without a meaningful lower
    # bound the fit collapses to (a, b, L_inf=0) when the curve is mostly
    # flat — that's spurious "infinite headroom" reporting.
    p0 = [max(L) - min(L), 0.5, max(0.5, min(L) * 0.8)]
    bounds = (
        [0.0, 0.01, 0.3],
        [1e3, 2.0, max(L) * 1.1 + 0.1],
    )
    try:
        popt, pcov = curve_fit(power_law, s, L, p0=p0, bounds=bounds,
                                maxfev=10_000)
    except Exception:
        return (None, None, None, None, None)

    a, b, l_inf = popt
    # 1-sigma error bar on L_inf
    try:
        l_inf_se = math.sqrt(pcov[2, 2])
    except Exception:
        l_inf_se = float("nan")
    return (a, b, l_inf, l_inf - 1.96 * l_inf_se, l_inf + 1.96 * l_inf_se)


def steps_to_within(a: float, b: float, l_inf: float,
                    fraction: float = 0.05) -> float:
    """Steps at which L(s) - L_inf <= fraction * L_inf.
    Returns inf if model is already past it at step 0 (degenerate)."""
    target = fraction * l_inf
    if a <= 0 or b <= 0:
        return float("inf")
    # a * s^(-b) <= target  ->  s >= (a / target)^(1/b)
    return (a / max(target, 1e-9)) ** (1.0 / b)


def load_variant_jsonl(jsonl_path: Path) -> tuple[list[int], list[float], dict]:
    """Returns (steps, val_losses, final_summary_dict)."""
    steps, losses = [], []
    final_summary = {}
    with jsonl_path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("phase") == "val" and "val_loss_recon" in row:
                # Skip the "eval-only" vanilla row at step 0
                if row.get("eval_only"):
                    final_summary = row
                    continue
                steps.append(int(row["step"]))
                losses.append(float(row["val_loss_recon"]))
            if row.get("final") or "final_val_loss_recon" in row:
                final_summary = row
    return steps, losses, final_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str,
                    help="Run directory (e.g. outputs/repr_qa/v1h_t4k). "
                         "Will scan run_dir's sibling per-variant dirs "
                         "(<out_tag>_<variant>/jsonl/<variant>.jsonl) under "
                         "outputs/repr_learning/.")
    ap.add_argument("--min-step-frac", type=float, default=0.1,
                    help="Use only val points past this fraction of max step "
                         "(filters warmup flukes). Default 0.1.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_tag = run_dir.name
    repr_root = Path("outputs/repr_learning")

    # Find per-variant jsonl files
    variant_dirs = sorted(repr_root.glob(f"{out_tag}_*"))
    if not variant_dirs:
        print(f"No per-variant dirs found at {repr_root}/{out_tag}_*", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*100}")
    print(f"  FAIR-REPORTING EXTRAPOLATION — {out_tag}")
    print(f"{'='*100}\n")
    print(f"  {'variant':<22} {'observed_final':>15} {'observed_best':>14} "
          f"{'extrap_L_inf':>13} {'steps_to_5%':>13} {'fit_a':>9} {'fit_b':>9}")
    print("  " + "-" * 96)

    for vdir in variant_dirs:
        variant = vdir.name.removeprefix(f"{out_tag}_")
        jsonl = vdir / "jsonl" / f"{variant}.jsonl"
        if not jsonl.exists():
            print(f"  {variant:<22} (no jsonl: {jsonl})")
            continue

        steps, losses, _ = load_variant_jsonl(jsonl)
        if not steps:
            print(f"  {variant:<22} (no val points)")
            continue

        max_step = max(steps)
        min_step_cutoff = max(1000, max_step * args.min_step_frac)
        filtered = [(s, L) for s, L in zip(steps, losses) if s >= min_step_cutoff]
        if len(filtered) < 4:
            # Fall back to all points if too few past warmup
            filtered = list(zip(steps, losses))

        s_f, L_f = zip(*filtered)
        # Match best.pt selection policy: lowest val past the warmup
        # window. Using min(losses) over all vals would let early-warmup
        # flukes show up as observed_best (and disagree with what we'd
        # actually reload as the "best ckpt").
        observed_best = min(L_f)
        observed_final = losses[-1]
        a, b, l_inf, _, _ = fit_one(list(s_f), list(L_f))

        if a is None:
            print(f"  {variant:<22} {observed_final:>15.4f} {observed_best:>14.4f} "
                  f"{'fit failed':>13}")
            continue

        s_to_5pct = steps_to_within(a, b, l_inf, fraction=0.05)
        s_str = f"{s_to_5pct/1000:>10.0f}k" if math.isfinite(s_to_5pct) and s_to_5pct < 1e9 else "         >∞"

        print(f"  {variant:<22} {observed_final:>15.4f} {observed_best:>14.4f} "
              f"{l_inf:>13.4f} {s_str:>13} {a:>9.3f} {b:>9.3f}")

    print()
    print("  observed_final = final val_recon at end of training")
    print("  observed_best  = lowest val_recon past warmup window (= best.pt selection)")
    print("  extrap_L_inf   = asymptote of power-law fit L(s) = a·s^(-b) + L∞")
    print("                   (estimate of what the model would converge to with infinite steps)")
    print("  steps_to_5%    = additional steps needed for fit to reach within 5% of L∞")
    print("                   (calibration of 'how much more would it improve')")
    print()


if __name__ == "__main__":
    main()
