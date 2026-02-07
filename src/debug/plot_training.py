"""
Plot training curves: loss, perplexity, learning rate, throughput, gradient norms, regularization.

Usage:
    python -m src.debug.plot_training

Diagnoses: Training broken? LR wrong? Data stall? Grad explosion?
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Configuration — edit these directly
# ============================================================================
METRICS_FILE = "checkpoints/metrics.jsonl"
OUTPUT_FILE = "checkpoints/plot_training.png"
SMOOTH_WINDOW = 20  # rolling average window for loss/ppl
# ============================================================================


def load_metrics(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def smooth(values, window):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main():
    path = METRICS_FILE
    if len(sys.argv) > 1:
        path = sys.argv[1]

    all_records = load_metrics(path)
    if not all_records:
        print(f"No records found in {path}")
        return
    records = [r for r in all_records if r.get("mode", "train") == "train"]
    val_records = [r for r in all_records if r.get("mode") == "val"]
    if not records:
        print(f"No training records found in {path}")
        return

    nan = float("nan")
    steps = [r.get("step", i) for i, r in enumerate(records)]
    loss = [r.get("loss", nan) for r in records]
    ppl = [r.get("ppl", nan) for r in records]
    lr = [r.get("lr", nan) for r in records]
    tok_s = [r.get("tok_s", nan) for r in records]
    grad_norm = [r.get("grad_norm", nan) for r in records]
    reg = [r.get("reg", nan) for r in records]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training Curves", fontsize=14)

    # Loss
    ax = axes[0, 0]
    ax.plot(steps, loss, alpha=0.3, color="blue", label="raw")
    if len(loss) > SMOOTH_WINDOW:
        s = smooth(np.array(loss), SMOOTH_WINDOW)
        ax.plot(steps[SMOOTH_WINDOW - 1:], s, color="blue", label=f"smooth({SMOOTH_WINDOW})")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    if val_records:
        v_steps = [r.get("step", i) for i, r in enumerate(val_records)]
        v_loss = [r.get("val_loss", nan) for r in val_records]
        ax.scatter(v_steps, v_loss, color="black", s=12, alpha=0.8, label="val")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Perplexity (log scale)
    ax = axes[0, 1]
    ax.plot(steps, ppl, alpha=0.3, color="orange", label="raw")
    if len(ppl) > SMOOTH_WINDOW:
        s = smooth(np.array(ppl), SMOOTH_WINDOW)
        ax.plot(steps[SMOOTH_WINDOW - 1:], s, color="orange", label=f"smooth({SMOOTH_WINDOW})")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("perplexity")
    ax.set_title("Perplexity")
    if val_records:
        v_steps = [r.get("step", i) for i, r in enumerate(val_records)]
        v_ppl = [r.get("val_ppl", nan) for r in val_records]
        ax.scatter(v_steps, v_ppl, color="black", s=12, alpha=0.8, label="val")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[0, 2]
    ax.plot(steps, lr, color="green")
    ax.set_xlabel("step")
    ax.set_ylabel("lr")
    ax.set_title("Learning Rate")
    ax.grid(True, alpha=0.3)

    # Throughput
    ax = axes[1, 0]
    ax.plot(steps, tok_s, alpha=0.5, color="purple")
    ax.set_xlabel("step")
    ax.set_ylabel("tok/s")
    ax.set_title("Throughput")
    ax.grid(True, alpha=0.3)

    # Gradient norm
    ax = axes[1, 1]
    ax.plot(steps, grad_norm, alpha=0.5, color="red")
    ax.set_xlabel("step")
    ax.set_ylabel("grad_norm")
    ax.set_title("Gradient Norm (after clipping)")
    ax.grid(True, alpha=0.3)

    # Regularization
    ax = axes[1, 2]
    ax.plot(steps, reg, alpha=0.5, color="brown")
    ax.set_xlabel("step")
    ax.set_ylabel("reg")
    ax.set_title("Regularization Loss")
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
