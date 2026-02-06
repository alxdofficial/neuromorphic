"""
Plot per-module gradient norms: module-level, block-level, layer heatmap, depth ratio.

Usage:
    python -m src.debug.plot_gradients

Diagnoses: Vanishing gradients? Exploding gradients? Asymmetric blocks?
"""

import json
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Configuration — edit these directly
# ============================================================================
METRICS_FILE = "checkpoints/metrics.jsonl"
OUTPUT_FILE = "checkpoints/plot_gradients.png"
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


def find_gnorm_keys(records, pattern="gnorm_"):
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

    all_gnorm_keys = find_gnorm_keys(records)
    # Split into categories
    block_keys = sorted([k for k in all_gnorm_keys if re.match(r"gnorm_block_\d+$", k)])
    layer_gate_keys = sorted([k for k in all_gnorm_keys if "_gates" in k])
    module_keys = sorted([k for k in all_gnorm_keys
                          if k not in block_keys and k not in layer_gate_keys
                          and "_pm" not in k])
    pm_keys = sorted([k for k in all_gnorm_keys if "_pm" in k])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Gradient Norm Diagnostics", fontsize=14)

    # Top-level module norms
    ax = axes[0, 0]
    for key in module_keys:
        label = key.replace("gnorm_", "")
        vals = [r.get(key, nan) for r in records]
        ax.plot(steps, vals, alpha=0.7, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("gradient norm")
    ax.set_title("Module-Level Gradient Norms")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Block-level norms
    ax = axes[0, 1]
    for key in block_keys:
        label = key.replace("gnorm_", "")
        vals = [r.get(key, nan) for r in records]
        ax.plot(steps, vals, alpha=0.7, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("gradient norm")
    ax.set_title("Block-Level Gradient Norms")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Layer gate norms heatmap (last record)
    ax = axes[1, 0]
    if layer_gate_keys and records:
        last = records[-1]
        # Parse block/layer indices
        bl_norms = {}
        for key in layer_gate_keys:
            m = re.match(r"gnorm_b(\d+)_l(\d+)_gates", key)
            if m:
                b, l = int(m.group(1)), int(m.group(2))
                bl_norms[(b, l)] = last.get(key, 0) or 0

        if bl_norms:
            max_b = max(b for b, l in bl_norms) + 1
            max_l = max(l for b, l in bl_norms) + 1
            heatmap = np.zeros((max_b, max_l))
            for (b, l), v in bl_norms.items():
                heatmap[b, l] = v
            im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd")
            ax.set_xlabel("layer")
            ax.set_ylabel("block")
            ax.set_title(f"Gate Gradient Norms (step {last.get('step', '?')})")
            plt.colorbar(im, ax=ax)
    ax.grid(False)

    # Depth ratio: first block vs last block grad norms over time
    ax = axes[1, 1]
    if len(block_keys) >= 2:
        first_key = block_keys[0]
        last_key = block_keys[-1]
        first_vals = np.array([r.get(first_key, nan) for r in records])
        last_vals = np.array([r.get(last_key, nan) for r in records])
        ratio = first_vals / np.maximum(last_vals, 1e-10)
        ax.plot(steps, ratio, color="purple")
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("step")
        ax.set_ylabel("ratio (first/last)")
        ax.set_title(f"Depth Ratio: {first_key} / {last_key}")
        ax.set_yscale("log")
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
