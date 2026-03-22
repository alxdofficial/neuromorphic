"""
Plot gradient and activation diagnostics: module norms, per-layer heatmaps,
activation magnitudes, memory signal ratios.

Usage:
    python -m src.debug.plot_gradients [metrics.jsonl] [output.png]

Diagnoses: Vanishing gradients? Exploding gradients? Weak memory signals?
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


def find_keys(records, prefix: str) -> list[str]:
    keys = set()
    for r in records:
        for k in r:
            if k.startswith(prefix):
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

    all_gnorm_keys = find_keys(records, "gnorm_")

    # Classify gnorm keys
    layer_keys = sorted([k for k in all_gnorm_keys if re.match(r"gnorm_layer_L\d+$", k)],
                        key=lambda k: int(re.search(r"\d+$", k).group()))
    # Legacy keys for backward compat with old metrics files
    stage1_layer_keys = sorted([k for k in all_gnorm_keys if re.match(r"gnorm_stage1_L\d+$", k)],
                               key=lambda k: int(re.search(r"\d+$", k).group()))
    stage3_layer_keys = sorted([k for k in all_gnorm_keys if re.match(r"gnorm_stage3_L\d+$", k)],
                               key=lambda k: int(re.search(r"\d+$", k).group()))
    per_layer_keys = set(layer_keys + stage1_layer_keys + stage3_layer_keys)

    # Legacy block keys (pre-v6)
    block_keys = sorted([k for k in all_gnorm_keys if re.match(r"gnorm_block_\d+$", k)])

    module_keys = sorted([k for k in all_gnorm_keys
                          if k not in per_layer_keys and k not in block_keys])

    # Activation norm keys
    act_keys = find_keys(records, "act_norm_")

    # Determine layout: 3 rows x 2 cols
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle("Gradient & Activation Diagnostics", fontsize=14)

    # (0,0) Module-level gradient norms (non-layer keys)
    ax = axes[0, 0]
    for key in module_keys:
        label = key.replace("gnorm_", "")
        vals = [r.get(key, nan) for r in records]
        ax.plot(steps, vals, alpha=0.7, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("gradient norm")
    ax.set_title("Module-Level Gradient Norms")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # (0,1) Per-layer heatmap (all layers, last record)
    ax = axes[0, 1]
    all_layer_keys = layer_keys or (stage1_layer_keys + stage3_layer_keys)
    if all_layer_keys and records:
        last = records[-1]
        labels = [k.replace("gnorm_", "") for k in all_layer_keys]
        vals = [last.get(k, 0) or 0 for k in all_layer_keys]
        n_layers = len(vals)
        # Build 2D heatmap: rows = time (last N records), cols = layers
        n_time = min(len(records), 50)  # last 50 full records
        recent = records[-n_time:]
        heatmap = np.zeros((n_time, n_layers))
        for ti, r in enumerate(recent):
            for li, k in enumerate(all_layer_keys):
                v = r.get(k)
                heatmap[ti, li] = v if v is not None and v > 0 else 1e-12
        heatmap = np.log10(np.maximum(heatmap, 1e-12))
        im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", origin="lower")
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
        step_labels = [str(r.get("step", "?")) for r in recent]
        n_yticks = min(10, n_time)
        ytick_idx = np.linspace(0, n_time - 1, n_yticks, dtype=int)
        ax.set_yticks(ytick_idx)
        ax.set_yticklabels([step_labels[i] for i in ytick_idx], fontsize=6)
        ax.set_xlabel("layer")
        ax.set_ylabel("step")
        ax.set_title("Per-Layer Grad Norms (log10)")
        plt.colorbar(im, ax=ax)
    else:
        ax.set_title("Per-Layer Grad Norms (no data)")
    ax.grid(False)

    # (1,0) Activation norms over time
    ax = axes[1, 0]
    if act_keys:
        for key in act_keys:
            label = key.replace("act_norm_", "")
            vals = [r.get(key, nan) for r in records]
            ax.plot(steps, vals, alpha=0.8, label=label, linewidth=1.5)
        ax.set_xlabel("step")
        ax.set_ylabel("activation norm")
        ax.set_title("Activation Norms at Integration Point")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
    else:
        ax.set_title("Activation Norms (no data)")
        ax.grid(True, alpha=0.3)

    # (1,1) Memory signal ratios (pm/H, em/H) over time
    ax = axes[1, 1]
    has_ratio_data = ("act_norm_H" in act_keys and
                      any(k in act_keys for k in ["act_norm_pm", "act_norm_em"]))
    if has_ratio_data:
        h_vals = np.array([r.get("act_norm_H", nan) for r in records])
        for mem_key, label, color in [
            ("act_norm_pm", "pm/H", "tab:blue"),
            ("act_norm_em", "em/H", "tab:orange"),
        ]:
            if mem_key in act_keys:
                mem_vals = np.array([r.get(mem_key, nan) for r in records])
                ratio = mem_vals / np.maximum(h_vals, 1e-12)
                ax.plot(steps, ratio, alpha=0.8, label=label, color=color, linewidth=1.5)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="1:1")
        ax.axhline(y=0.1, color="gray", linestyle=":", alpha=0.3)
        ax.set_xlabel("step")
        ax.set_ylabel("ratio to H")
        ax.set_title("Memory Signal Ratios (vs H)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
    else:
        ax.set_title("Memory Signal Ratios (no data)")
        ax.grid(True, alpha=0.3)

    # (2,0) Per-layer over time (line plot) — pre-memory layers
    ax = axes[2, 0]
    pre_mem_keys = [k for k in all_layer_keys if k in layer_keys[:len(layer_keys)//2]] or stage1_layer_keys
    if pre_mem_keys:
        for key in pre_mem_keys:
            label = key.replace("gnorm_", "")
            vals = [r.get(key, nan) for r in records]
            ax.plot(steps, vals, alpha=0.7, label=label)
        ax.set_xlabel("step")
        ax.set_ylabel("gradient norm")
        ax.set_title("Pre-Memory Per-Layer Grad Norms")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
    else:
        ax.set_title("Pre-Memory Per-Layer (no data)")
        ax.grid(True, alpha=0.3)

    # (2,1) Per-layer over time (line plot) — post-memory layers
    ax = axes[2, 1]
    post_mem_keys = [k for k in all_layer_keys if k in layer_keys[len(layer_keys)//2:]] or stage3_layer_keys
    if post_mem_keys:
        for key in post_mem_keys:
            label = key.replace("gnorm_", "")
            vals = [r.get(key, nan) for r in records]
            ax.plot(steps, vals, alpha=0.7, label=label)
        ax.set_xlabel("step")
        ax.set_ylabel("gradient norm")
        ax.set_title("Post-Memory Per-Layer Grad Norms")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
    else:
        ax.set_title("Post-Memory Per-Layer (no data)")
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
