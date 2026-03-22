"""
Plot PM (Procedural Memory) diagnostics: Frobenius norms, max element,
commit rate, eligibility, grad norms.

v6: Hebbian fast-weight W_pm. Collector writes global keys (not per-bank).

Usage:
    python -m src.debug.plot_pm [metrics.jsonl] [output.png]

Diagnoses: PM never committing? Frobenius exploding? Gradients dead?
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Configuration — edit these directly
# ============================================================================
METRICS_FILE = "checkpoints/metrics.jsonl"
OUTPUT_FILE = "checkpoints/plot_pm.png"
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
    fig.suptitle("Procedural Memory Diagnostics (v6 Hebbian)", fontsize=14)

    # (0,0) W_pm Frobenius norms over time
    ax = axes[0, 0]
    for key, label, color in [
        ("pm_W_frob_mean", "frob mean", "tab:blue"),
        ("pm_W_frob_max", "frob max", "tab:red"),
    ]:
        vals = [r.get(key, nan) for r in records]
        if any(v is not None and v == v for v in vals):  # has real data
            ax.plot(steps, vals, alpha=0.8, label=label, color=color)
    ax.set_xlabel("step")
    ax.set_ylabel("Frobenius norm")
    ax.set_title("W_pm Frobenius Norms")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) W_pm max absolute element
    ax = axes[0, 1]
    vals = [r.get("pm_W_max", nan) for r in records]
    if any(v is not None and v == v for v in vals):
        ax.plot(steps, vals, alpha=0.8, color="tab:purple")
    ax.set_xlabel("step")
    ax.set_ylabel("max |W_pm|")
    ax.set_title("W_pm Max Absolute Element")
    ax.grid(True, alpha=0.3)

    # (0,2) PM commit rate
    ax = axes[0, 2]
    vals = [r.get("pm_commit_rate", nan) for r in records]
    if any(v is not None and v == v for v in vals):
        ax.plot(steps, vals, alpha=0.8, color="tab:green")
    ax.set_xlabel("step")
    ax.set_ylabel("commit rate")
    ax.set_title("PM Commit Rate")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # (1,0) PM gradient norm
    ax = axes[1, 0]
    vals = [r.get("gnorm_pm", nan) for r in records]
    if any(v is not None and v == v for v in vals):
        ax.plot(steps, vals, alpha=0.8, color="tab:orange")
    ax.set_xlabel("step")
    ax.set_ylabel("gradient norm")
    ax.set_title("PM Gradient Norm")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # (1,1) PM activation norm at integration point
    ax = axes[1, 1]
    for key, label, color in [
        ("act_norm_pm", "pm_read", "tab:blue"),
        ("act_norm_H", "H (reference)", "tab:gray"),
    ]:
        vals = [r.get(key, nan) for r in records]
        if any(v is not None and v == v for v in vals):
            ax.plot(steps, vals, alpha=0.7, label=label, color=color)
    ax.set_xlabel("step")
    ax.set_ylabel("activation norm")
    ax.set_title("PM Signal at Integration Point")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # (1,2) PM/H ratio over time
    ax = axes[1, 2]
    h_vals = np.array([r.get("act_norm_H", nan) for r in records])
    pm_vals = np.array([r.get("act_norm_pm", nan) for r in records])
    ratio = pm_vals / np.maximum(h_vals, 1e-12)
    if any(np.isfinite(ratio)):
        ax.plot(steps, ratio, alpha=0.8, color="tab:blue")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="1:1")
    ax.axhline(y=0.1, color="gray", linestyle=":", alpha=0.3, label="0.1")
    ax.set_xlabel("step")
    ax.set_ylabel("pm / H ratio")
    ax.set_title("PM Signal Ratio (vs H)")
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
