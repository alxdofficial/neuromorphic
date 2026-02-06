"""
Plot gate distributions: gate_a saturation, gate_b magnitude, hidden state norms.

Usage:
    python -m src.debug.plot_gates

Diagnoses: Gates saturating? Dead updates? Exploding hidden states?
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Configuration — edit these directly
# ============================================================================
METRICS_FILE = "checkpoints/metrics.jsonl"
OUTPUT_FILE = "checkpoints/plot_gates.png"
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


def extract_block_layer_keys(records, prefix_pattern, suffix):
    """Find all keys matching b{i}_l{j}_{suffix}."""
    keys = set()
    for r in records:
        for k in r:
            if k.endswith(f"_{suffix}"):
                # Check it matches pattern like b0_l0_gate_a_mean
                parts = k.split("_")
                if len(parts) >= 3 and parts[0].startswith("b") and parts[1].startswith("l"):
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

    # Discover block/layer keys
    ga_mean_keys = extract_block_layer_keys(records, "b*_l*", "gate_a_mean")
    gb_abs_keys = extract_block_layer_keys(records, "b*_l*", "gate_b_abs_mean")
    h_norm_keys = extract_block_layer_keys(records, "b*_l*", "h_norm")
    ga_near1_keys = extract_block_layer_keys(records, "b*_l*", "gate_a_near1")
    ga_near0_keys = extract_block_layer_keys(records, "b*_l*", "gate_a_near0")

    n_rows = 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 4 * n_rows))
    fig.suptitle("Gate Distributions", fontsize=14)

    # Row 1: gate_a saturation (near-0 and near-1 fractions)
    ax = axes[0]
    for key in ga_near1_keys:
        label = key.replace("_gate_a_near1", "")
        vals = [r.get(key, nan) for r in records]
        ax.plot(steps, vals, alpha=0.7, label=f"{label} >0.9")
    for key in ga_near0_keys:
        label = key.replace("_gate_a_near0", "")
        vals = [r.get(key, nan) for r in records]
        ax.plot(steps, vals, alpha=0.7, linestyle="--", label=f"{label} <0.1")
    ax.set_xlabel("step")
    ax.set_ylabel("fraction")
    ax.set_title("Gate a Saturation (fraction near 0 or 1)")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Row 2: gate_b absolute mean
    ax = axes[1]
    for key in gb_abs_keys:
        label = key.replace("_gate_b_abs_mean", "")
        vals = [r.get(key, nan) for r in records]
        ax.plot(steps, vals, alpha=0.7, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("|gate_b| mean")
    ax.set_title("Gate b Magnitude")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Row 3: hidden state norms
    ax = axes[2]
    for key in h_norm_keys:
        label = key.replace("_h_norm", "")
        vals = [r.get(key, nan) for r in records]
        ax.plot(steps, vals, alpha=0.7, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("||h|| mean")
    ax.set_title("Hidden State Norms per Layer")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
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
