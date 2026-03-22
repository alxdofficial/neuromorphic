"""Plot v8 training curves and memory graph diagnostics.

Usage:
    python -m scripts.plot_training outputs/v8/<run_id>/
    python -m scripts.plot_training outputs/v8/<run_id>/ --snapshot 5000

Produces:
    <run_dir>/plots/training_curves.png     — loss, ppl, LR, throughput
    <run_dir>/plots/rl_curves.png           — RL policy loss, advantages, segment losses
    <run_dir>/plots/memory_health.png       — memory graph state norms, decay, usage
    <run_dir>/plots/memory_connectivity.png — (from snapshot) neuron graph visualization
"""

import json
import sys
import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_metrics(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def smooth(values, window=20):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def get_field(records, key, default=None):
    return [r.get(key, default) for r in records if key in r]


def get_steps(records, key):
    return [r["step"] for r in records if key in r]


def plot_training_curves(records, output_path):
    """Loss, PPL, LR, throughput."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Curves", fontsize=14)

    # Loss
    ax = axes[0, 0]
    steps = get_steps(records, "loss")
    loss = get_field(records, "loss")
    if steps:
        ax.plot(steps, loss, alpha=0.3, color='blue', linewidth=0.5)
        if len(loss) > 50:
            ax.plot(steps[:len(smooth(loss))], smooth(loss), color='blue', linewidth=1.5)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Step")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)

    # PPL (log scale)
    ax = axes[0, 1]
    ppl = get_field(records, "ppl")
    if steps:
        ax.plot(steps, ppl, alpha=0.3, color='orange', linewidth=0.5)
        if len(ppl) > 50:
            ax.plot(steps[:len(smooth(ppl))], smooth(ppl), color='orange', linewidth=1.5)
        ax.set_yscale('log')
        ax.set_ylabel("Perplexity")
        ax.set_xlabel("Step")
        ax.set_title("Perplexity (log)")
        ax.grid(True, alpha=0.3)

    # LR
    ax = axes[1, 0]
    lr = get_field(records, "lr")
    lr_steps = get_steps(records, "lr")
    if lr_steps:
        ax.plot(lr_steps, lr, color='green')
        ax.set_ylabel("Learning Rate")
        ax.set_xlabel("Step")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)

    # Throughput
    ax = axes[1, 1]
    tok_s = get_field(records, "tok_s")
    tok_steps = get_steps(records, "tok_s")
    if tok_steps:
        tok_k = [t / 1000 for t in tok_s]
        ax.plot(tok_steps, tok_k, alpha=0.3, color='purple', linewidth=0.5)
        if len(tok_k) > 50:
            ax.plot(tok_steps[:len(smooth(tok_k))], smooth(tok_k), color='purple', linewidth=1.5)
        ax.set_ylabel("K tok/s")
        ax.set_xlabel("Step")
        ax.set_title("Throughput")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_rl_curves(records, output_path):
    """RL policy loss, advantages, per-segment losses."""
    rl_records = [r for r in records if "rl_policy_loss" in r]
    if not rl_records:
        print("  No RL data found, skipping rl_curves")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RL / Neuromodulator Curves", fontsize=14)

    steps = [r["step"] for r in rl_records]

    # Policy loss
    ax = axes[0, 0]
    vals = [r["rl_policy_loss"] for r in rl_records]
    ax.plot(steps, vals, alpha=0.3, linewidth=0.5)
    if len(vals) > 50:
        ax.plot(steps[:len(smooth(vals))], smooth(vals), linewidth=1.5)
    ax.set_title("RL Policy Loss")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    # Advantage std
    ax = axes[0, 1]
    vals = [r.get("rl_adv_std", 0) for r in rl_records]
    ax.plot(steps, vals, alpha=0.3, linewidth=0.5)
    if len(vals) > 50:
        ax.plot(steps[:len(smooth(vals))], smooth(vals), linewidth=1.5)
    ax.set_title("Advantage Std (signal strength)")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    # Per-segment loss (first vs last)
    ax = axes[1, 0]
    seg_first = [r.get("rl_seg_loss_first", 0) for r in rl_records]
    seg_last = [r.get("rl_seg_loss_last", 0) for r in rl_records]
    if len(seg_first) > 50:
        ax.plot(steps[:len(smooth(seg_first))], smooth(seg_first), label="seg 0 (first)", linewidth=1.5)
        ax.plot(steps[:len(smooth(seg_last))], smooth(seg_last), label="seg 7 (last)", linewidth=1.5)
    else:
        ax.plot(steps, seg_first, label="seg 0 (first)", alpha=0.5)
        ax.plot(steps, seg_last, label="seg 7 (last)", alpha=0.5)
    ax.legend()
    ax.set_title("Per-Segment Loss (first vs last)")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    # Neuromod grad norm
    ax = axes[1, 1]
    vals = [r.get("rl_nm_grad_norm", 0) for r in rl_records]
    ax.plot(steps, vals, alpha=0.3, linewidth=0.5)
    if len(vals) > 50:
        ax.plot(steps[:len(smooth(vals))], smooth(vals), linewidth=1.5)
    ax.set_title("Neuromod Gradient Norm")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_memory_health(records, output_path):
    """Memory graph state health over training."""
    mem_records = [r for r in records if "mem_h_norm" in r]
    if not mem_records:
        print("  No memory metrics found, skipping memory_health")
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle("Memory Graph Health", fontsize=14)

    steps = [r["step"] for r in mem_records]

    def plot_smooth(ax, key, title, **kwargs):
        vals = [r.get(key, 0) for r in mem_records]
        ax.plot(steps, vals, alpha=0.3, linewidth=0.5, **kwargs)
        if len(vals) > 50:
            ax.plot(steps[:len(smooth(vals))], smooth(vals), linewidth=1.5, **kwargs)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

    plot_smooth(axes[0, 0], "mem_h_norm", "Hidden State Norm (h)")
    plot_smooth(axes[0, 1], "mem_msg_norm", "Message Norm (prev_messages)")
    plot_smooth(axes[1, 0], "mem_prim_std", "Primitive Diversity (std across neurons)")
    plot_smooth(axes[1, 1], "mem_decay_mean", "Mean Decay (higher = longer memory)")

    # Connection weights
    ax = axes[2, 0]
    cw_mean = [r.get("mem_cw_mean", 0) for r in mem_records]
    cw_std = [r.get("mem_cw_std", 0) for r in mem_records]
    ax.plot(steps, cw_mean, alpha=0.5, label="mean")
    ax.plot(steps, cw_std, alpha=0.5, label="std")
    ax.legend()
    ax.set_title("Connection Weight Distribution")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    # Usage + near-zero connections
    ax = axes[2, 1]
    usage = [r.get("mem_usage_frac", 0) for r in mem_records]
    near_zero = [r.get("mem_cw_near_zero", 0) for r in mem_records]
    ax.plot(steps, usage, label="neuron usage frac", alpha=0.7)
    ax.plot(steps, near_zero, label="near-zero conn frac", alpha=0.7)
    ax.legend()
    ax.set_title("Neuron Usage & Dead Connections")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_connectivity_snapshot(snapshot_path, output_path):
    """Visualize neuron graph connectivity from a snapshot."""
    snap = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    # Convert bf16 tensors to float for numpy
    for k, v in snap.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
            snap[k] = v.float()
    cfg = snap["config"]
    N = cfg["N"]
    C = cfg["C"]
    K = cfg["K"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Memory Graph Snapshot (step {snap['step']})", fontsize=14)

    # Per-neuron state norms
    ax = axes[0, 0]
    h_norm = snap["h_norm_per_neuron"].numpy()
    msg_norm = snap["msg_norm_per_neuron"].numpy()
    x = np.arange(N)
    ax.bar(x, h_norm, alpha=0.6, label="h norm", width=1.0)
    ax.bar(x, msg_norm, alpha=0.6, label="msg norm", width=1.0)
    ax.axvline(x=C - 0.5, color='red', linestyle='--', alpha=0.5, label=f"port neurons (0-{C-1})")
    ax.legend()
    ax.set_title("Per-Neuron State Norms")
    ax.set_xlabel("Neuron ID")

    # Decay per neuron
    ax = axes[0, 1]
    decay = snap["decay_per_neuron"].numpy()
    ax.bar(x, decay, width=1.0, alpha=0.7)
    ax.axvline(x=C - 0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1)
    ax.set_title("Decay per Neuron (higher = longer memory)")
    ax.set_xlabel("Neuron ID")

    # Connection weight heatmap (first 64 neurons × K connections)
    ax = axes[1, 0]
    cw = snap["cw_per_neuron"][:64].numpy()  # [64, K]
    im = ax.imshow(cw, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    plt.colorbar(im, ax=ax)
    ax.set_title("Connection Weights (neurons 0-63)")
    ax.set_xlabel(f"Connection index (K={K})")
    ax.set_ylabel("Neuron ID")

    # Connectivity fan-in histogram
    ax = axes[1, 1]
    conn_idx = snap["conn_indices"].numpy()  # [N, K]
    fan_in = np.bincount(conn_idx.flatten(), minlength=N)
    ax.bar(x, fan_in, width=1.0, alpha=0.7)
    ax.axvline(x=C - 0.5, color='red', linestyle='--', alpha=0.5, label="port neurons")
    ax.legend()
    ax.set_title("Fan-In per Neuron (how many neurons send to each)")
    ax.set_xlabel("Neuron ID")
    ax.set_ylabel("# incoming connections")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    p = argparse.ArgumentParser(description="Plot v8 training diagnostics")
    p.add_argument("run_dir", help="Path to run directory (e.g., outputs/v8/<run_id>/)")
    p.add_argument("--snapshot", type=int, default=None,
                   help="Step number for connectivity snapshot plot")
    args = p.parse_args()

    metrics_path = os.path.join(args.run_dir, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        print(f"No metrics.jsonl found in {args.run_dir}")
        sys.exit(1)

    records = load_metrics(metrics_path)
    print(f"Loaded {len(records)} metric records from {metrics_path}")

    plot_dir = os.path.join(args.run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plot_training_curves(records, os.path.join(plot_dir, "training_curves.png"))
    plot_rl_curves(records, os.path.join(plot_dir, "rl_curves.png"))
    plot_memory_health(records, os.path.join(plot_dir, "memory_health.png"))

    if args.snapshot is not None:
        snap_path = os.path.join(args.run_dir, "snapshots", f"step_{args.snapshot:06d}.pt")
        if os.path.exists(snap_path):
            plot_connectivity_snapshot(snap_path, os.path.join(plot_dir, f"connectivity_{args.snapshot}.png"))
        else:
            print(f"  Snapshot not found: {snap_path}")
            available = os.listdir(os.path.join(args.run_dir, "snapshots")) if os.path.exists(os.path.join(args.run_dir, "snapshots")) else []
            if available:
                print(f"  Available: {sorted(available)}")


if __name__ == "__main__":
    main()
