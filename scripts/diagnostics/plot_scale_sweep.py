#!/usr/bin/env python3
"""Plot the scale_sweep result curve. Used as meeting figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=Path("outputs/scale_sweep.json"))
    ap.add_argument("--vanilla-floor", type=float, default=2.4367,
                    help="Vanilla Llama train-floor CE for reference line")
    ap.add_argument("--output", type=Path, default=Path("outputs/scale_sweep.png"))
    args = ap.parse_args()

    d = json.load(open(args.sweep))
    factors = [r["factor"] for r in d["sweep"]]
    eff_scales = [r["effective_scale_mean"] for r in d["sweep"]]
    ces = [r["weighted_ce"] for r in d["sweep"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(eff_scales, ces, marker="o", linewidth=2, color="C0",
            label="Wave 1 ckpt, scale_raw × factor")
    ax.axhline(y=args.vanilla_floor, color="gray", linestyle="--",
               alpha=0.7, label=f"Vanilla Llama floor ({args.vanilla_floor:.4f})")
    # Annotate the trained scale
    for f, e, ce in zip(factors, eff_scales, ces):
        if f == 1.0:
            ax.annotate(f"trained\nscale={e:.4f}\nCE={ce:.4f}",
                        xy=(e, ce), xytext=(e + 0.02, ce + 0.04),
                        fontsize=9, ha="left",
                        arrowprops=dict(arrowstyle="->", color="C0", alpha=0.5))
    ax.set_xlabel("effective_scale (per-feature, |tanh(scale_raw)|)")
    ax.set_ylabel("weighted val CE (200 chunks × 4 sources)")
    ax.set_title("Memory injection adds pure noise — no sweet spot\n"
                 "(scale → 0 is optimal under current uniform routing)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=140)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
