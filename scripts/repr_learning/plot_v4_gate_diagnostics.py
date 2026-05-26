#!/usr/bin/env python3
"""Plot v4 graph_baseline gate + clustering diagnostics over training.

Reads the trainer's jsonl and renders 4 panels:
  1. Gate distribution over training (stacked area: frac_anchor/loadbearer/jumpedship + g_mean line)
  2. Per-window breakdown — does window 0 specialize differently from window 3?
     (4 stacked-area sub-panels, one per window position)
  3. Per-slot specialization — are different slots in different regimes?
     (g_slot_std + g_slot_range over training; bigger = more specialization)
  4. Endpoint clustering — are clusters forming?
     (endpoint_eff_rank: lower = more clustered, endpoint_cos_max: hottest pair)

Usage:
    python -m scripts.repr_learning.plot_v4_gate_diagnostics \
        --jsonl outputs/repr_learning/v1h_t4k_v4_graph_baseline/jsonl/graph_baseline.jsonl \
        --out docs/plots/v4_gate_diagnostics.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def get_col(rows, key, default=np.nan):
    return np.array([r.get(key, default) for r in rows], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--smooth", type=int, default=20,
                    help="moving-average window for noisy curves")
    args = ap.parse_args()

    rows = load_jsonl(args.jsonl)
    if not rows:
        print(f"[error] no rows in {args.jsonl}")
        sys.exit(1)
    print(f"[load] {len(rows)} rows from {args.jsonl}")

    step = get_col(rows, "step")
    # Filter out val-only rows (val rows have no train-step gate keys)
    has_gate = ~np.isnan(get_col(rows, "graph_gate_mean_avg"))
    step = step[has_gate]
    rows = [r for r, m in zip(rows, has_gate) if m]
    print(f"[filter] {len(rows)} train rows with gate telemetry")

    fa = get_col(rows, "graph_frac_anchor_avg")
    fl = get_col(rows, "graph_frac_loadbearer_avg")
    fj = get_col(rows, "graph_frac_jumpedship_avg")
    g  = get_col(rows, "graph_gate_mean_avg")
    fs = get_col(rows, "graph_frac_selfpick_avg")
    g_std = get_col(rows, "graph_g_slot_std")
    g_rng = get_col(rows, "graph_g_slot_range")
    eff_rank = get_col(rows, "graph_endpoint_eff_rank")
    cos_max = get_col(rows, "graph_endpoint_cos_max")
    u_mean = get_col(rows, "graph_u_mean")
    recon = get_col(rows, "loss_recon")

    def smooth(y, w=args.smooth):
        if w <= 1 or len(y) < w: return y
        kernel = np.ones(w) / w
        return np.convolve(y, kernel, mode="same")

    fig = plt.figure(figsize=(15, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.25)

    # ── Panel 1: gate distribution overall ──
    ax = fig.add_subplot(gs[0, :])
    ax.stackplot(step, smooth(fa), smooth(fl), smooth(fj),
                 labels=["anchor (g<0.1)", "load-bearer (0.1≤g<0.7)", "jumped-ship (g≥0.7)"],
                 colors=["#1f77b4", "#ff7f0e", "#d62728"], alpha=0.85)
    ax2 = ax.twinx()
    ax2.plot(step, smooth(g), color="black", lw=1.5, label="mean g (right axis)")
    ax2.set_ylabel("mean g"); ax2.set_ylim(0, 1)
    ax.set_xlabel("step"); ax.set_ylabel("fraction of slots"); ax.set_ylim(0, 1)
    ax.set_title("Gate distribution over training — does the model specialize edge roles?")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")

    # ── Panel 2: per-window breakdown ──
    for wi in range(4):
        ax = fig.add_subplot(gs[1, wi // 2] if wi < 2 else gs[2, wi - 2])
        fa_w = get_col(rows, f"graph_frac_anchor_w{wi}")
        fl_w = get_col(rows, f"graph_frac_loadbearer_w{wi}")
        fj_w = get_col(rows, f"graph_frac_jumpedship_w{wi}")
        g_w  = get_col(rows, f"graph_g_mean_w{wi}")
        if np.all(np.isnan(fa_w)):
            ax.text(0.5, 0.5, f"no data for window {wi}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Window {wi}")
            continue
        ax.stackplot(step, smooth(fa_w), smooth(fl_w), smooth(fj_w),
                     colors=["#1f77b4", "#ff7f0e", "#d62728"], alpha=0.85)
        ax.plot(step, smooth(g_w), color="black", lw=1.0)
        ax.set_xlabel("step"); ax.set_ylabel("fraction"); ax.set_ylim(0, 1)
        ax.set_title(f"Window {wi} — gate distribution")

    # ── Panel 3: per-slot specialization ──
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(step, smooth(g_std), label="g std across slots", color="#1f77b4")
    ax.plot(step, smooth(g_rng), label="g range (max−min)", color="#d62728")
    ax.set_xlabel("step"); ax.set_ylabel("spread of g across slots")
    ax.set_title("Per-slot specialization — do slots differentiate?")
    ax.legend()
    ax.axhline(y=0.1, color="gray", ls="--", alpha=0.5, lw=0.8)
    ax.text(step[-1] if len(step) else 0, 0.1, " moderate", color="gray",
            va="bottom", fontsize=8)

    # ── Panel 4: endpoint clustering (threshold-free) ──
    ax = fig.add_subplot(gs[3, 1])
    ax2 = ax.twinx()
    ax.plot(step, smooth(eff_rank), label="effective rank (out of 136)",
            color="#1f77b4", lw=1.5)
    ax2.plot(step, smooth(cos_max), label="hottest pair cosine (max)",
             color="#d62728", lw=1.5)
    ax.set_xlabel("step"); ax.set_ylabel("# distinct endpoint directions", color="#1f77b4")
    ax2.set_ylabel("max pairwise cosine", color="#d62728")
    ax.set_title("Endpoint clustering — are nodes forming?")
    ax.tick_params(axis='y', labelcolor="#1f77b4")
    ax2.tick_params(axis='y', labelcolor="#d62728")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")

    # Headline summary at top
    if len(step):
        last = rows[-1]
        title = (f"v4 graph_baseline diagnostics — step {int(step[-1])}, "
                  f"val_recon={last.get('loss_recon', float('nan')):.3f}, "
                  f"g={last.get('graph_gate_mean_avg', float('nan')):.3f}, "
                  f"eff_rank={last.get('graph_endpoint_eff_rank', float('nan')):.1f}")
        fig.suptitle(title, fontsize=12)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"[output] wrote {args.out}")

    # Console summary
    if len(step):
        last = rows[-1]
        print("\nLast-step snapshot:")
        for k in ["graph_gate_mean_avg", "graph_frac_anchor_avg",
                   "graph_frac_loadbearer_avg", "graph_frac_jumpedship_avg",
                   "graph_g_slot_std", "graph_g_slot_range",
                   "graph_endpoint_eff_rank", "graph_endpoint_cos_max",
                   "graph_u_mean", "loss_recon"]:
            v = last.get(k)
            if v is not None:
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
