#!/usr/bin/env python3
"""Plot per-task memory contribution across architectures.

Reads the three eval_full JSONs and produces a grouped bar chart showing
each architecture's per-task memory contribution Δ = (with-mem NLL) − (no-mem NLL).
Negative bars = memory helps; positive bars = memory hurts.

Usage:
    python scripts/eval/plot_per_task_memory_contribution.py
    -> writes outputs/figures/per_task_memory_contribution.png
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V1_5 = json.loads((ROOT / "outputs" / "v1.5" / "eval_full.json").read_text())
FLAT = json.loads((ROOT / "outputs" / "v1.2_flatbank" / "eval_full.json").read_text())
V2 = json.loads((ROOT / "outputs" / "wave1_v2" / "eval_full_10000.json").read_text())

# Reorder tasks by category: retrieval, state-tracking, format/binary
TASK_ORDER = [
    "biographical", "passphrase", "preferences",   # retrieval
    "boxes", "theory_of_mind", "triage", "revisions",  # state-tracking
    "calendar", "knights",                          # format-determined
]
TASK_LABELS = [
    "biographical", "passphrase", "preferences",
    "boxes", "theory_of_mind", "triage", "revisions",
    "calendar", "knights",
]
CATEGORY_BREAKS = [3, 7]  # vertical lines between categories


def delta(d, mode_with, mode_no, key):
    out = []
    for t in TASK_ORDER:
        out.append(d["modes"][mode_with][t][key] - d["modes"][mode_no][t][key])
    return out


def make_plot(metric_key, ylabel, suffix):
    v15_d = delta(V1_5, "v1", "v1_no_mem", metric_key)
    flat_d = delta(FLAT, "v1", "v1_no_mem", metric_key)
    v2_d = delta(V2, "v2", "v2_no_mem", metric_key)

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(TASK_ORDER))
    width = 0.27

    bars1 = ax.bar(x - width, v15_d, width, label="V1.5 trajectory + per-hop contrastive",
                    color="#d97757")
    bars2 = ax.bar(x, flat_d, width, label="flat-bank (top-K attention)",
                    color="#2a9d8f")
    bars3 = ax.bar(x + width, v2_d, width, label="V2 vocabulary-trajectory",
                    color="#5c80bc")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(
        f"Memory contribution per task per architecture ({suffix})\n"
        f"Δ = NLL(memory active) − NLL(memory disabled). Negative = memory helps. Positive = memory hurts.",
        fontsize=11
    )
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS, rotation=20, ha="right", fontsize=10)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)

    # Category dividers + annotations
    for break_i in CATEGORY_BREAKS:
        ax.axvline(break_i - 0.5, color="gray", linestyle=":", linewidth=0.7)
    ax.text(1, ax.get_ylim()[1] * 0.93, "retrieval", ha="center",
            fontsize=9, color="dimgray", style="italic")
    ax.text(5, ax.get_ylim()[1] * 0.93, "state-tracking", ha="center",
            fontsize=9, color="dimgray", style="italic")
    ax.text(7.5, ax.get_ylim()[1] * 0.93, "format-determined", ha="center",
            fontsize=9, color="dimgray", style="italic")

    # Annotate each bar with its value (small text)
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            if abs(h) < 0.04:
                continue
            ax.annotate(
                f"{h:+.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3 if h >= 0 else -10),
                textcoords="offset points",
                ha="center", fontsize=7,
            )

    plt.tight_layout()
    out_path = OUT_DIR / f"per_task_memory_contribution_{suffix}.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")
    plt.close(fig)


def make_absolute_nll_plot():
    """Companion plot: absolute mean NLL with memory for each model + baselines."""
    v15 = [V1_5["modes"]["v1"][t]["mean_nll"] for t in TASK_ORDER]
    flat = [FLAT["modes"]["v1"][t]["mean_nll"] for t in TASK_ORDER]
    v2 = [V2["modes"]["v2"][t]["mean_nll"] for t in TASK_ORDER]
    van_nc = [V2["modes"]["vanilla_nc"][t]["mean_nll"] for t in TASK_ORDER]
    van_fc = [V2["modes"]["vanilla_fc"][t]["mean_nll"] for t in TASK_ORDER]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(TASK_ORDER))
    width = 0.16

    ax.bar(x - 2*width, van_nc, width, label="Llama no-context (baseline)", color="#aaaaaa")
    ax.bar(x - width, van_fc, width, label="Llama full-context (ceiling)", color="#666666")
    ax.bar(x, v15, width, label="V1.5 trajectory + per-hop", color="#d97757")
    ax.bar(x + width, flat, width, label="flat-bank", color="#2a9d8f")
    ax.bar(x + 2*width, v2, width, label="V2 vocab-trajectory", color="#5c80bc")

    ax.set_ylabel("Mean NLL per content token (nats)", fontsize=11)
    ax.set_title(
        "Absolute NLL by task — with memory active (lower = better)\n"
        "composite_v1 val, 800 paired chunks per architecture",
        fontsize=11
    )
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS, rotation=20, ha="right", fontsize=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)

    for break_i in CATEGORY_BREAKS:
        ax.axvline(break_i - 0.5, color="gray", linestyle=":", linewidth=0.7)

    plt.tight_layout()
    out_path = OUT_DIR / "per_task_absolute_nll.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    make_plot("mean_nll", "Δ NLL (nats)  ←memory helps   memory hurts→",
              "mean_NLL")
    make_plot("mean_first_nll", "Δ first-token NLL (nats)",
              "first_token_NLL")
    make_absolute_nll_plot()
