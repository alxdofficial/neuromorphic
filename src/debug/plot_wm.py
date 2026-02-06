"""
Plot WM (Working Memory) diagnostics: buffer utilization, attention entropy.

Usage:
    python -m src.debug.plot_wm

Diagnoses: WM attention collapsed? Buffer bug? Entropy too low?
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Configuration — edit these directly
# ============================================================================
METRICS_FILE = "checkpoints/metrics.jsonl"
OUTPUT_FILE = "checkpoints/plot_wm.png"
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Working Memory Diagnostics", fontsize=14)

    # Buffer utilization
    ax = axes[0]
    vals = [r.get("wm_buffer_util", nan) for r in records]
    ax.plot(steps, vals, color="teal")
    ax.set_xlabel("step")
    ax.set_ylabel("fraction valid")
    ax.set_title("WM Buffer Utilization")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Attention entropy over time
    ax = axes[1]
    ent_mean = [r.get("wm_entropy_mean", nan) for r in records]
    ent_min = [r.get("wm_entropy_min", nan) for r in records]
    ent_max = [r.get("wm_entropy_max", nan) for r in records]
    ax.plot(steps, ent_mean, color="blue", label="mean")
    ax.fill_between(steps, ent_min, ent_max, alpha=0.2, color="blue")
    ax.set_xlabel("step")
    ax.set_ylabel("entropy (nats)")
    ax.set_title("WM Attention Entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropy histogram (last record)
    ax = axes[2]
    if records:
        last = records[-1]
        ent_m = last.get("wm_entropy_mean", nan)
        ent_s = last.get("wm_entropy_std", nan)
        ent_lo = last.get("wm_entropy_min", nan)
        ent_hi = last.get("wm_entropy_max", nan)
        if ent_m is not None and ent_s is not None:
            # Show bar chart of stats from the last step
            labels = ["min", "mean", "max"]
            values = [ent_lo or 0, ent_m or 0, ent_hi or 0]
            colors = ["#2196F3", "#4CAF50", "#FF5722"]
            ax.bar(labels, values, color=colors, alpha=0.8)
            ax.set_ylabel("entropy (nats)")
            ax.set_title(f"WM Entropy Distribution (step {last.get('step', '?')})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_FILE
    if len(sys.argv) > 2:
        out = sys.argv[2]
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
