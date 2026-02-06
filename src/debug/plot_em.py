"""
Plot EM (Episodic Memory) diagnostics: slot strengths, budget utilization,
write rate, novelty, nonzero slots.

Usage:
    python -m src.debug.plot_em

Diagnoses: EM never writing? Novelty too low? Budget full?
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


def find_keys(records, pattern):
    keys = set()
    for r in records:
        for k in r:
            if k.startswith(pattern):
                keys.add(k)
    return sorted(keys)


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
    fig.suptitle("Episodic Memory Diagnostics", fontsize=14)

    em_keys = find_keys(records, "em_b")

    # EM strength means
    ax = axes[0, 0]
    s_mean_keys = [k for k in em_keys if k.endswith("_S_mean")]
    for key in s_mean_keys:
        label = key.replace("em_", "").replace("_S_mean", "")
        vals = [r.get(key, nan) for r in records]
        ax.plot(steps, vals, alpha=0.7, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("em_S mean")
    ax.set_title("EM Slot Strengths (mean)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # EM budget utilization
    ax = axes[0, 1]
    s_sum_keys = [k for k in em_keys if k.endswith("_S_sum")]
    for key in s_sum_keys:
        label = key.replace("em_", "").replace("_S_sum", "")
        vals = [r.get(key, nan) for r in records]
        ax.plot(steps, vals, alpha=0.7, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("sum(em_S)")
    ax.set_title("EM Budget Utilization")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Write rate
    ax = axes[0, 2]
    wr_keys = find_keys(records, "em_write_rate_")
    for key in wr_keys:
        label = key.replace("em_write_rate_", "block ")
        vals = [r.get(key, nan) for r in records]
        ax.plot(steps, vals, alpha=0.7, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("write rate")
    ax.set_title("EM Write Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Novelty mean
    ax = axes[1, 0]
    nov_keys = find_keys(records, "em_novelty_mean_")
    for key in nov_keys:
        label = key.replace("em_novelty_mean_", "block ")
        vals = [r.get(key, nan) for r in records]
        ax.plot(steps, vals, alpha=0.7, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("novelty")
    ax.set_title("EM Candidate Novelty (mean)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Nonzero slots
    ax = axes[1, 1]
    nz_keys = [k for k in em_keys if k.endswith("_nonzero")]
    for key in nz_keys:
        label = key.replace("em_", "").replace("_nonzero", "")
        vals = [r.get(key, nan) for r in records]
        ax.plot(steps, vals, alpha=0.7, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("fraction nonzero")
    ax.set_title("EM Nonzero Slots")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Strength histogram (last record)
    ax = axes[1, 2]
    s_max_keys = [k for k in em_keys if k.endswith("_S_max")]
    if s_max_keys and records:
        last = records[-1]
        maxes = [last.get(k, 0) for k in s_max_keys]
        means = [last.get(k.replace("_S_max", "_S_mean"), 0) for k in s_max_keys]
        x_pos = range(len(s_max_keys))
        labels = [k.replace("em_", "").replace("_S_max", "") for k in s_max_keys]
        ax.bar(x_pos, maxes, alpha=0.5, label="max", color="red")
        ax.bar(x_pos, means, alpha=0.7, label="mean", color="blue")
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_ylabel("em_S value")
        ax.set_title("EM Strengths (last step)")
        ax.legend()
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
