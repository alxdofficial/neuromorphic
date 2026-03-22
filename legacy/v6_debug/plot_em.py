"""
Plot EM (Episodic Memory) diagnostics: slot strengths, budget utilization,
write strength, novelty, nonzero slots, activation norms.

v6: Collector writes global keys (not per-bank/block).

Usage:
    python -m src.debug.plot_em [metrics.jsonl] [output.png]

Diagnoses: EM never writing? Novelty too low? Budget full? Signal dead?
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Configuration — edit these directly
# ============================================================================
METRICS_FILE = "checkpoints/metrics.jsonl"
OUTPUT_FILE = "checkpoints/plot_em.png"
# ============================================================================


def load_full_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("full"):
                    records.append(r)
    return records


def main():
    path = METRICS_FILE
    if len(sys.argv) > 1:
        path = sys.argv[1]

    records = load_full_records(path)
    if not records:
        print(f"No full-collection records found in {path}")
        return

    nan = float("nan")
    steps = [r.get("step", i) for i, r in enumerate(records)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Episodic Memory Diagnostics (v6)", fontsize=14)

    # (0,0) EM slot strengths (mean/max)
    ax = axes[0, 0]
    for key, label, color in [
        ("em_S_mean", "S mean", "tab:blue"),
        ("em_S_max", "S max", "tab:red"),
    ]:
        vals = [r.get(key, nan) for r in records]
        if any(v is not None and v == v for v in vals):
            ax.plot(steps, vals, alpha=0.8, label=label, color=color)
    ax.set_xlabel("step")
    ax.set_ylabel("em_S")
    ax.set_title("EM Slot Strengths")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) EM budget utilization (sum of strengths)
    ax = axes[0, 1]
    vals = [r.get("em_S_sum", nan) for r in records]
    if any(v is not None and v == v for v in vals):
        ax.plot(steps, vals, alpha=0.8, color="tab:green")
    ax.set_xlabel("step")
    ax.set_ylabel("sum(em_S)")
    ax.set_title("EM Budget Utilization")
    ax.grid(True, alpha=0.3)

    # (0,2) Write strength (g_em) and novelty
    ax = axes[0, 2]
    for key, label, color in [
        ("em_g_em_mean", "g_em (write gate)", "tab:orange"),
        ("em_novelty_mean", "novelty", "tab:purple"),
    ]:
        vals = [r.get(key, nan) for r in records]
        if any(v is not None and v == v for v in vals):
            ax.plot(steps, vals, alpha=0.8, label=label, color=color)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.set_title("EM Write Strength & Novelty")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) EM nonzero slots
    ax = axes[1, 0]
    vals = [r.get("em_nonzero", nan) for r in records]
    if any(v is not None and v == v for v in vals):
        ax.plot(steps, vals, alpha=0.8, color="tab:blue")
    ax.set_xlabel("step")
    ax.set_ylabel("fraction nonzero")
    ax.set_title("EM Nonzero Slots (S > 0.01)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # (1,1) EM activation norms at integration point
    ax = axes[1, 1]
    for key, label, color in [
        ("act_norm_em", "em_trail", "tab:orange"),
        ("act_norm_H", "H (reference)", "tab:gray"),
    ]:
        vals = [r.get(key, nan) for r in records]
        if any(v is not None and v == v for v in vals):
            ax.plot(steps, vals, alpha=0.7, label=label, color=color)
    ax.set_xlabel("step")
    ax.set_ylabel("activation norm")
    ax.set_title("EM Signals at Integration Point")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # (1,2) EM gradient norms
    ax = axes[1, 2]
    for key, label, color in [
        ("gnorm_em", "em", "tab:blue"),
        ("gnorm_em_neuromod", "em_neuromod", "tab:orange"),
    ]:
        vals = [r.get(key, nan) for r in records]
        if any(v is not None and v == v for v in vals):
            ax.plot(steps, vals, alpha=0.8, label=label, color=color)
    ax.set_xlabel("step")
    ax.set_ylabel("gradient norm")
    ax.set_title("EM Gradient Norms")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    out = OUTPUT_FILE
    if len(sys.argv) > 2:
        out = sys.argv[2]
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
