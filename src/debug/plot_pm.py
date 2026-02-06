"""
Plot PM (Procedural Memory) diagnostics: slot strengths, budget utilization,
commit rate, eligibility norms, nonzero slots.

Usage:
    python -m src.debug.plot_pm

Diagnoses: PM never committing? Budget full? Eligibility diverging?
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


def find_keys(records, pattern):
    """Find all keys matching a prefix pattern."""
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
    fig.suptitle("Procedural Memory Diagnostics", fontsize=14)

    # PM strength means
    ax = axes[0, 0]
    keys = find_keys(records, "pm_b")
    a_mean_keys = [k for k in keys if k.endswith("_a_mean")]
    for key in a_mean_keys:
        label = key.replace("pm_", "").replace("_a_mean", "")
        vals = [r.get(key, nan) for r in records]
        ax.plot(steps, vals, alpha=0.7, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("pm_a mean")
    ax.set_title("PM Slot Strengths (mean)")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    # PM budget utilization (sum)
    ax = axes[0, 1]
    a_sum_keys = [k for k in keys if k.endswith("_a_sum")]
    for key in a_sum_keys:
        label = key.replace("pm_", "").replace("_a_sum", "")
        vals = [r.get(key, nan) for r in records]
        ax.plot(steps, vals, alpha=0.7, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("sum(pm_a)")
    ax.set_title("PM Budget Utilization")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    # Commit rate
    ax = axes[0, 2]
    cr_keys = find_keys(records, "pm_commit_rate_")
    for key in cr_keys:
        label = key.replace("pm_commit_rate_", "")
        vals = [r.get(key, nan) for r in records]
        ax.plot(steps, vals, alpha=0.7, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("commit rate")
    ax.set_title("PM Commit Rate")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Eligibility norms
    ax = axes[1, 0]
    elig_keys = [k for k in keys if k.endswith("_elig_norm")]
    for key in elig_keys:
        label = key.replace("pm_", "").replace("_elig_norm", "")
        vals = [r.get(key, nan) for r in records]
        ax.plot(steps, vals, alpha=0.7, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("||elig_K|| mean")
    ax.set_title("Eligibility Trace Norms")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    # PM strength histogram (last record)
    ax = axes[1, 1]
    a_max_keys = [k for k in keys if k.endswith("_a_max")]
    if a_max_keys and records:
        last = records[-1]
        maxes = [last.get(k, 0) for k in a_max_keys]
        means = [last.get(k.replace("_a_max", "_a_mean"), 0) for k in a_max_keys]
        x_pos = range(len(a_max_keys))
        labels = [k.replace("pm_", "").replace("_a_max", "") for k in a_max_keys]
        ax.bar(x_pos, maxes, alpha=0.5, label="max", color="red")
        ax.bar(x_pos, means, alpha=0.7, label="mean", color="blue")
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(labels, rotation=45, fontsize=6)
        ax.set_ylabel("pm_a value")
        ax.set_title("PM Strengths (last step)")
        ax.legend()
    ax.grid(True, alpha=0.3)

    # Nonzero slots
    ax = axes[1, 2]
    nz_keys = [k for k in keys if k.endswith("_nonzero")]
    for key in nz_keys:
        label = key.replace("pm_", "").replace("_nonzero", "")
        vals = [r.get(key, nan) for r in records]
        ax.plot(steps, vals, alpha=0.7, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("fraction nonzero")
    ax.set_title("PM Nonzero Slots")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    out = OUTPUT_FILE
    if len(sys.argv) > 2:
        out = sys.argv[2]
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
