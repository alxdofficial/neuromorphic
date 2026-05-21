#!/usr/bin/env python3
"""Health report for one or more trained variants.

Reads JSONL training logs and prints a concise convergence assessment:
  - Loss trajectory (initial, mid, final) on train + val
  - Memory dispersion progression
  - Codebook health (V2.1 + A only): unique-codes, entropy, pairwise cos
  - Gradient stability (max, median over training)
  - Throughput
  - Verdict: HEALTHY / WARN / FAIL with reasons

Usage:
    python scripts/repr_learning/health_report.py \\
        --jsonl outputs/repr_learning/v0_a/jsonl/flat_baseline.jsonl

    python scripts/repr_learning/health_report.py \\
        --jsonl-dir outputs/repr_learning/v0_a/jsonl
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median


def load_jsonl(path: Path) -> tuple[list[dict], list[dict]]:
    train, val = [], []
    if not path.exists() or path.stat().st_size == 0:
        return train, val
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("phase") == "val":
            val.append(r)
        else:
            train.append(r)
    return train, val


def _bin_trajectory(rows: list[dict], key: str, bins: int = 5) -> list[float]:
    """Bin a key into N quantiles, return mean per bin."""
    vals = [r[key] for r in rows if key in r and isinstance(r[key], (int, float))]
    if not vals:
        return []
    n = len(vals)
    bin_size = n // bins
    return [
        mean(vals[i * bin_size : (i + 1) * bin_size])
        for i in range(bins) if (i + 1) * bin_size <= n
    ]


def report_variant(jsonl_path: Path):
    print(f"\n{'=' * 78}")
    print(f"  {jsonl_path.name}")
    print('=' * 78)

    train, val = load_jsonl(jsonl_path)
    if not train:
        print("  (no train rows)")
        return

    variant = train[0].get("variant", "?")
    print(f"  variant         : {variant}")
    print(f"  train rows      : {len(train)}")
    print(f"  val rows        : {len(val)}")
    print(f"  last step       : {train[-1]['step']}")

    issues = []
    warnings = []

    # Loss trajectory
    loss_bins = _bin_trajectory(train, "loss_recon", bins=5)
    if loss_bins:
        print(f"\n  recon CE bins   : {' → '.join(f'{x:.3f}' for x in loss_bins)}")
        # Healthy: monotonic decrease across bins (allow small jitter)
        decreasing = all(loss_bins[i+1] < loss_bins[i] + 0.5 for i in range(len(loss_bins)-1))
        if not decreasing:
            issues.append("loss not monotonically decreasing across bins")
        if loss_bins[-1] > loss_bins[0] - 1.0:
            warnings.append(f"final loss only {loss_bins[0] - loss_bins[-1]:.2f} below initial — slow convergence")

    if val:
        v_first = val[0]["val_loss_recon"]
        v_last = val[-1]["val_loss_recon"]
        print(f"  val recon CE    : {v_first:.3f} → {v_last:.3f} (Δ {v_first - v_last:+.3f})")
        if v_last > v_first - 0.5:
            warnings.append(f"val loss barely moved: {v_first - v_last:+.3f} over training")

    # Memory dispersion
    disp_bins = _bin_trajectory(train, "mem_dispersion", bins=5)
    if disp_bins:
        print(f"  mem dispersion  : {' → '.join(f'{x:.3f}' for x in disp_bins)}")
        if disp_bins[-1] > 0.95:
            issues.append(f"memory tokens COLLAPSED (final cos = {disp_bins[-1]:.3f})")
        elif disp_bins[-1] > 0.8:
            warnings.append(f"memory tokens drifting toward collapse (final cos = {disp_bins[-1]:.3f})")

    # Routing health (codebook variants)
    ent_bins = _bin_trajectory(train, "routing_entropy", bins=5)
    if ent_bins and ent_bins[0] > 0:
        print(f"  routing entropy : {' → '.join(f'{x:.2f}' for x in ent_bins)}")
        # log(4096) ≈ 8.3 is uniform. Below ~2 is concerning concentration.
        if ent_bins[-1] < 1.0:
            issues.append(f"routing entropy collapsed (final = {ent_bins[-1]:.2f}, log(4096)=8.3)")
        elif ent_bins[-1] < ent_bins[0] - 3.0:
            warnings.append(f"routing entropy dropped sharply: {ent_bins[0]:.2f} → {ent_bins[-1]:.2f}")

    uniq_bins = _bin_trajectory(train, "unique_codes_per_batch", bins=5)
    if uniq_bins and uniq_bins[0] > 0:
        print(f"  unique codes/b  : {' → '.join(f'{x:.0f}' for x in uniq_bins)}")
        if uniq_bins[-1] < 10:
            issues.append(f"codebook usage collapsed: only {uniq_bins[-1]:.0f} distinct codes / batch")

    # Codebook health
    cb_cos_bins = _bin_trajectory(train, "codebook_pairwise_cos", bins=5)
    if cb_cos_bins and any(abs(x) > 0.001 for x in cb_cos_bins):
        print(f"  codebook cos    : {' → '.join(f'{x:+.3f}' for x in cb_cos_bins)}")
        if abs(cb_cos_bins[-1]) > 0.5:
            warnings.append(f"codebook entries aligning (final cos = {cb_cos_bins[-1]:+.3f})")

    # Gradient health
    gnorm_vals = [r["grad_norm"] for r in train if "grad_norm" in r]
    if gnorm_vals:
        gn_med = median(gnorm_vals)
        gn_max = max(gnorm_vals)
        gn_late_med = median([r["grad_norm"] for r in train[-len(train)//4:]
                               if "grad_norm" in r])
        print(f"  grad norm       : median {gn_med:.2f}, max {gn_max:.2f}, "
              f"late-median {gn_late_med:.2f}")
        if gn_max > 1e4:
            warnings.append(f"grad norm spike to {gn_max:.0f}")
        if gn_late_med > gn_med * 3:
            warnings.append("grad norm increasing late in training — instability")

    # Throughput
    sps = [r.get("text_tok_per_sec") for r in train if r.get("text_tok_per_sec")]
    if sps:
        print(f"  throughput      : {mean(sps):.0f} text tok/s")

    # Verdict
    print()
    if issues:
        print("  VERDICT: \033[31mFAIL\033[0m")
        for i in issues:
            print(f"    • {i}")
    elif warnings:
        print("  VERDICT: \033[33mWARN\033[0m")
        for w in warnings:
            print(f"    • {w}")
    else:
        print("  VERDICT: \033[32mHEALTHY\033[0m")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--jsonl", type=Path)
    g.add_argument("--jsonl-dir", type=Path)
    args = ap.parse_args()

    if args.jsonl:
        report_variant(args.jsonl)
    else:
        paths = sorted(args.jsonl_dir.glob("*.jsonl"))
        if not paths:
            print(f"No JSONL files in {args.jsonl_dir}", file=sys.stderr)
            sys.exit(1)
        for p in paths:
            report_variant(p)


if __name__ == "__main__":
    main()
