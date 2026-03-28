"""Plot v9-backprop training curves and memory graph diagnostics.

Usage:
    python -m scripts.plot_training outputs/v9/<run_id>/
    python -m scripts.plot_training outputs/v9/<run_id>/ --snapshot 5000

Produces:
    <run_dir>/plots/training_curves.png  — loss, ppl, LR, throughput
    <run_dir>/plots/memory_health.png    — neuron state, gates, connectivity
    <run_dir>/plots/gradient_health.png  — per-component gradient norms
    <run_dir>/plots/memory_snapshot.png  — (from snapshot) per-neuron stats
"""

import json
import sys
import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Style
# ============================================================================

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#bbb",
    "axes.labelcolor": "#333",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.color": "#555",
    "ytick.color": "#555",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.color": "#ddd",
    "grid.alpha": 0.7,
    "text.color": "#222",
    "legend.fontsize": 9,
}

C = {
    "loss": "#1a73e8",
    "ppl": "#d93025",
    "lr": "#188038",
    "tput": "#7b1fa2",
    "gate": "#e37400",
    "decay": "#d93025",
    "conn": "#1a73e8",
    "hebb": "#188038",
    "usage": "#7b1fa2",
    "prim": "#c2185b",
    "mod": "#e37400",
    "state": "#1a73e8",
    "msg": "#188038",
    "nid": "#7b1fa2",
    "dendrite": "#d93025",
    "pcm": "#616161",
    "lm_grad": "#1a73e8",
    "mem_grad": "#d93025",
}


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


def smooth(values, window=50):
    if len(values) < window * 2:
        window = max(len(values) // 4, 1)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def get(records, key):
    return [r.get(key) for r in records if key in r]


def plot_line(ax, vals, color, label=None, **kwargs):
    arr = np.array(vals)
    steps = np.arange(len(arr))
    if len(arr) > 200:
        ax.plot(steps, arr, alpha=0.15, color=color, linewidth=0.5)
        s = smooth(arr)
        ax.plot(steps[:len(s)], s, color=color, linewidth=2.0, label=label, **kwargs)
    elif len(arr) > 50:
        ax.plot(steps, arr, alpha=0.25, color=color, linewidth=0.8)
        s = smooth(arr, window=max(len(arr) // 8, 3))
        ax.plot(steps[:len(s)], s, color=color, linewidth=2.0, label=label, **kwargs)
    else:
        ax.plot(steps, arr, color=color, linewidth=1.5, alpha=0.9, label=label, **kwargs)


def setup(ax, title, ylabel=None):
    ax.set_title(title, fontweight="bold", pad=8)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("Step")
    ax.grid(True)


# ============================================================================
# Training curves
# ============================================================================

def plot_training_curves(records, output_path):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        n = len(records)
        fig.suptitle(f"Training Curves ({n} steps)", fontsize=14, fontweight="bold")

        # Loss
        loss = get(records, "loss")
        if loss:
            plot_line(axes[0, 0], loss, C["loss"])
            setup(axes[0, 0], "Training Loss", "CE Loss")

        # PPL
        ppl = get(records, "ppl")
        if ppl:
            plot_line(axes[0, 1], ppl, C["ppl"])
            axes[0, 1].set_yscale('log')
            setup(axes[0, 1], "Perplexity", "PPL (log)")

        # LR
        lr = get(records, "lr")
        if lr:
            axes[1, 0].plot(lr, color=C["lr"], linewidth=2.0)
            setup(axes[1, 0], "Learning Rate", "LR")

        # Throughput
        tok_s = get(records, "tok_s")
        if tok_s:
            tok_k = [t / 1000 for t in tok_s]
            plot_line(axes[1, 1], tok_k, C["tput"])
            setup(axes[1, 1], "Throughput", "K tok/s")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Saved: {output_path}")


# ============================================================================
# Memory health
# ============================================================================

def plot_memory_health(records, output_path):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle("Memory Graph Health", fontsize=14, fontweight="bold")

        # mem_gate
        v = get(records, "mem_gate_mean")
        if v:
            plot_line(axes[0, 0], v, C["gate"], "mean")
            v_min = get(records, "mem_gate_min")
            v_max = get(records, "mem_gate_max")
            if v_min and v_max:
                axes[0, 0].fill_between(range(len(v_min)),
                                         v_min, v_max, alpha=0.2, color=C["gate"])
            setup(axes[0, 0], "Memory Gate", "sigmoid(gate)")

        # w_conn
        v = get(records, "mem_w_conn_mean")
        if v:
            plot_line(axes[0, 1], v, C["conn"], "mean")
            v2 = get(records, "mem_w_conn_std")
            if v2:
                plot_line(axes[0, 1], v2, C["conn"], "std", linestyle="--")
            axes[0, 1].legend()
            setup(axes[0, 1], "Connection Weights", "sigmoid(w_conn)")

        # decay
        v = get(records, "mem_decay_mean")
        if v:
            plot_line(axes[0, 2], v, C["decay"], "mean")
            v2 = get(records, "mem_decay_std")
            if v2:
                plot_line(axes[0, 2], v2, C["decay"], "std", linestyle="--")
            axes[0, 2].legend()
            setup(axes[0, 2], "Decay Rate", "sigmoid(decay)")

        # hebbian
        v = get(records, "mem_hebbian_mean")
        if v:
            plot_line(axes[1, 0], v, C["hebb"], "mean")
            v2 = get(records, "mem_hebbian_std")
            if v2:
                plot_line(axes[1, 0], v2, C["hebb"], "std", linestyle="--")
            axes[1, 0].legend()
            setup(axes[1, 0], "Hebbian Traces", "correlation")

        # neuron usage
        v = get(records, "mem_usage_frac")
        if v:
            plot_line(axes[1, 1], v, C["usage"])
            setup(axes[1, 1], "Active Neurons", "fraction > 0.01")

        # message magnitude
        v = get(records, "mem_msg_mag_mean")
        if v:
            plot_line(axes[1, 2], v, C["msg"])
            setup(axes[1, 2], "Message Magnitude", "mean |msg|")

        # h and msg norms
        v = get(records, "mem_h_norm")
        if v:
            plot_line(axes[2, 0], v, C["state"], "h norm")
            v2 = get(records, "mem_msg_norm")
            if v2:
                plot_line(axes[2, 0], v2, C["msg"], "msg norm")
            axes[2, 0].legend()
            setup(axes[2, 0], "State / Message Norms", "L2")

        # primitives norm
        v = get(records, "mem_prim_norm")
        if v:
            plot_line(axes[2, 1], v, C["prim"])
            setup(axes[2, 1], "Primitives Norm", "L2")

        # PCM surprise
        v = get(records, "pcm_surprise_mean")
        if v:
            plot_line(axes[2, 2], v, C["pcm"], "mean")
            v2 = get(records, "pcm_surprise_std")
            if v2:
                plot_line(axes[2, 2], v2, C["pcm"], "std", linestyle="--")
            axes[2, 2].legend()
            setup(axes[2, 2], "PCM Surprise", "prediction error")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Saved: {output_path}")


# ============================================================================
# Gradient health
# ============================================================================

def plot_gradient_health(records, output_path):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Gradient Health", fontsize=14, fontweight="bold")

        # LM vs Memory grad norms
        lm = get(records, "lm_grad_norm")
        mem = get(records, "mem_grad_norm")
        if lm:
            plot_line(axes[0, 0], lm, C["lm_grad"], "LM")
        if mem:
            plot_line(axes[0, 0], mem, C["mem_grad"], "Memory")
        axes[0, 0].legend()
        setup(axes[0, 0], "Gradient Norms (LM vs Memory)", "L2 norm")

        # Per-component memory grads
        components = [
            ("grad_mod_w1", "mod_w1", C["mod"]),
            ("grad_mod_w2", "mod_w2", C["mod"]),
            ("grad_state_w2", "state_w2", C["state"]),
            ("grad_msg_w2", "msg_w2", C["msg"]),
            ("grad_neuron_id", "neuron_id", C["nid"]),
            ("grad_dendrite", "dendrite", C["dendrite"]),
        ]
        for key, label, color in components:
            v = get(records, key)
            if v and any(x > 0 for x in v):
                plot_line(axes[0, 1], v, color, label)
        axes[0, 1].legend(fontsize=7)
        setup(axes[0, 1], "Per-Component Memory Grads", "L2 norm")

        # Weight norms
        for key, label, color in [
            ("mod_w1_norm", "mod_w1", C["mod"]),
            ("mod_w2_norm", "mod_w2", C["mod"]),
            ("state_mlp_w1_norm", "state_w1", C["state"]),
            ("msg_mlp_w1_norm", "msg_w1", C["msg"]),
        ]:
            v = get(records, key)
            if v:
                plot_line(axes[1, 0], v, color, label, linestyle="--" if "w2" in key else "-")
        axes[1, 0].legend(fontsize=7)
        setup(axes[1, 0], "Weight Norms", "L2 norm")

        # aux_loss (PCM)
        v = get(records, "aux_loss")
        if v:
            plot_line(axes[1, 1], v, C["pcm"])
            setup(axes[1, 1], "PCM Aux Loss", "MSE")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Saved: {output_path}")


# ============================================================================
# Snapshot visualization
# ============================================================================

def plot_snapshot(snap_path, output_path):
    import torch
    snap = torch.load(snap_path, map_location="cpu", weights_only=False)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        step = snap.get("step", "?")
        fig.suptitle(f"Memory Snapshot (step {step})", fontsize=14, fontweight="bold")

        # Per-neuron h norm
        v = snap.get("h_norm_per_neuron")
        if v is not None:
            axes[0, 0].bar(range(len(v)), v.numpy(), color=C["state"], alpha=0.7)
            setup(axes[0, 0], "Hidden State Norm per Neuron", "L2")

        # Per-neuron message norm
        v = snap.get("msg_norm_per_neuron")
        if v is not None:
            axes[0, 1].bar(range(len(v)), v.numpy(), color=C["msg"], alpha=0.7)
            setup(axes[0, 1], "Message Norm per Neuron", "L2")

        # Per-neuron decay
        v = snap.get("decay_per_neuron")
        if v is not None:
            axes[0, 2].bar(range(len(v)), v.numpy(), color=C["decay"], alpha=0.7)
            setup(axes[0, 2], "Decay per Neuron", "sigmoid(decay)")

        # Per-neuron activation magnitude
        v = snap.get("msg_magnitude_per_neuron")
        if v is not None:
            axes[1, 0].bar(range(len(v)), v.numpy(), color=C["usage"], alpha=0.7)
            setup(axes[1, 0], "Activation Magnitude per Neuron")

        # w_conn mean per neuron
        v = snap.get("w_conn_mean_per_neuron")
        if v is not None:
            axes[1, 1].bar(range(len(v)), v.numpy(), color=C["conn"], alpha=0.7)
            setup(axes[1, 1], "Mean Connection Weight per Neuron")

        # Connectivity visualization (sparse adjacency)
        conn = snap.get("conn_indices")
        if conn is not None:
            N = conn.shape[0]
            K = conn.shape[1]
            # Show as sparse matrix
            img = np.zeros((N, N))
            for n in range(N):
                for k in range(K):
                    img[n, conn[n, k].item()] = 1
            axes[1, 2].imshow(img, cmap='Blues', aspect='auto', interpolation='nearest')
            setup(axes[1, 2], f"Connectivity ({N} neurons, K={K})")
            axes[1, 2].set_xlabel("Target")
            axes[1, 2].set_ylabel("Source")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Saved: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot v9-backprop training")
    parser.add_argument("run_dir", help="Path to run directory")
    parser.add_argument("--snapshot", type=int, default=None,
                        help="Step number for snapshot visualization")
    args = parser.parse_args()

    metrics_path = os.path.join(args.run_dir, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        print(f"No metrics found at {metrics_path}")
        sys.exit(1)

    records = load_metrics(metrics_path)
    print(f"Loaded {len(records)} records from {metrics_path}")

    plots_dir = os.path.join(args.run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_training_curves(records, os.path.join(plots_dir, "training_curves.png"))
    plot_memory_health(records, os.path.join(plots_dir, "memory_health.png"))
    plot_gradient_health(records, os.path.join(plots_dir, "gradient_health.png"))

    if args.snapshot is not None:
        snap_path = os.path.join(args.run_dir, "snapshots",
                                 f"step_{args.snapshot:06d}.pt")
        if os.path.exists(snap_path):
            plot_snapshot(snap_path, os.path.join(plots_dir, "memory_snapshot.png"))
        else:
            print(f"  Snapshot not found: {snap_path}")


if __name__ == "__main__":
    main()
