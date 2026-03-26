"""Plot v8 training curves and memory graph diagnostics.

Usage:
    python -m scripts.plot_training outputs/v8/<run_id>/
    python -m scripts.plot_training outputs/v8/<run_id>/ --snapshot 5000

Produces:
    <run_dir>/plots/training_curves.png     — loss, ppl, LR, throughput
    <run_dir>/plots/rl_curves.png           — RL policy loss, GRPO, advantages, exploration
    <run_dir>/plots/memory_health.png       — memory graph state, coupling, plasticity
    <run_dir>/plots/memory_connectivity.png — (from snapshot) neuron graph visualization
"""

import json
import sys
import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

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
    "legend.facecolor": "white",
    "legend.edgecolor": "#ccc",
}

# Colors — high contrast on white, distinguishable pairs
C_LOSS = "#1a73e8"
C_PPL = "#d93025"
C_LR = "#188038"
C_LR2 = "#e37400"
C_TPUT = "#7b1fa2"
C_RL = "#e37400"
C_GRPO = "#1a73e8"
C_ADV = "#1a73e8"
C_ENT = "#c2185b"
C_GRAD = "#616161"
C_GATE = "#e37400"
C_PRIM = "#1a73e8"
C_DECAY = "#d93025"
C_FIRE = "#188038"
C_FIRE2 = "#0d904f"
C_PHI = "#7b1fa2"
C_USAGE = "#1a73e8"
C_KEY = "#e37400"
C_SAT = "#c2185b"
C_TRAJ_BEST = "#188038"
C_TRAJ_WORST = "#d93025"


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


def get_field(records, key):
    return [r[key] for r in records if key in r]


def get_steps(records, key):
    return [r["step"] for r in records if key in r]


def _plot_line(ax, steps, vals, color, label=None, linewidth=2.5):
    """Plot raw + smoothed line with good visibility."""
    arr = np.array(vals)
    if len(arr) > 200:
        ax.plot(steps, arr, alpha=0.15, color=color, linewidth=0.5)
        s = smooth(arr)
        ax.plot(steps[:len(s)], s, color=color, linewidth=linewidth, label=label)
    elif len(arr) > 50:
        ax.plot(steps, arr, alpha=0.25, color=color, linewidth=0.8)
        s = smooth(arr, window=max(len(arr) // 8, 3))
        ax.plot(steps[:len(s)], s, color=color, linewidth=linewidth, label=label)
    else:
        ax.plot(steps, arr, color=color, linewidth=linewidth, alpha=0.9, label=label)


def _setup_ax(ax, title, ylabel=None, xlabel="Step"):
    ax.set_title(title, fontweight="bold", pad=8)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True)


# ============================================================================
# Plot functions
# ============================================================================

def plot_training_curves(records, output_path):
    """Loss, PPL, LR (both), throughput."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Training Curves", fontsize=16, fontweight="bold")

        # Loss
        ax = axes[0, 0]
        steps = get_steps(records, "loss")
        loss = get_field(records, "loss")
        if steps:
            _plot_line(ax, steps, loss, C_LOSS)
            _setup_ax(ax, "Training Loss", "CE Loss")

        # PPL (log scale)
        ax = axes[0, 1]
        ppl = get_field(records, "ppl")
        if steps:
            _plot_line(ax, steps, ppl, C_PPL)
            ax.set_yscale('log')
            _setup_ax(ax, "Perplexity", "PPL (log)")

        # LR (both schedules)
        ax = axes[1, 0]
        lr = get_field(records, "lr")
        lr_steps = get_steps(records, "lr")
        nm_lr = get_field(records, "nm_lr")
        nm_lr_steps = get_steps(records, "nm_lr")
        if lr_steps:
            ax.plot(lr_steps, lr, color=C_LR, linewidth=2.0, label="LM")
        if nm_lr_steps:
            ax.plot(nm_lr_steps, nm_lr, color=C_LR2, linewidth=2.0,
                    linestyle="--", label="Neuromod")
        ax.legend()
        _setup_ax(ax, "Learning Rate Schedule", "LR")

        # Throughput (smoothed only — raw alternates between collect/RL steps)
        ax = axes[1, 1]
        tok_s = get_field(records, "tok_s")
        tok_steps = get_steps(records, "tok_s")
        if tok_steps:
            tok_k = np.array([t / 1000 for t in tok_s])
            if len(tok_k) > 100:
                # Wide window to average out collect/RL alternation
                s = smooth(tok_k, window=min(100, len(tok_k) // 4))
                ax.plot(tok_steps[:len(s)], s, color=C_TPUT, linewidth=2.5,
                        label="avg")
                ax.legend()
            else:
                ax.plot(tok_steps, tok_k, color=C_TPUT, linewidth=2.0)
            _setup_ax(ax, "Throughput", "K tok/s")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved: {output_path}")


def plot_rl_curves(records, output_path):
    """GRPO loss, trajectory spread, exploration, entropy."""
    rl_records = [r for r in records if "rl_grpo_loss" in r]
    if not rl_records:
        print("  No RL data found, skipping rl_curves")
        return

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("RL / Neuromodulator", fontsize=16, fontweight="bold")

        steps = [r["step"] for r in rl_records]

        # GRPO loss
        ax = axes[0, 0]
        vals = [r["rl_grpo_loss"] for r in rl_records]
        _plot_line(ax, steps, vals, C_GRPO)
        _setup_ax(ax, "GRPO Loss")

        # Trajectory spread (best vs worst)
        ax = axes[0, 1]
        has_traj = [r for r in rl_records if "rl_traj_loss_best" in r]
        if has_traj:
            t_steps = [r["step"] for r in has_traj]
            best = [r["rl_traj_loss_best"] for r in has_traj]
            worst = [r["rl_traj_loss_worst"] for r in has_traj]
            _plot_line(ax, t_steps, best, C_TRAJ_BEST, "best trajectory")
            _plot_line(ax, t_steps, worst, C_TRAJ_WORST, "worst trajectory")
            ax.legend()
        _setup_ax(ax, "GRPO Trajectory Spread", "z-score advantage")

        # Trajectory advantage std (signal strength)
        ax = axes[0, 2]
        vals = [r.get("rl_traj_adv_std", 0) for r in rl_records]
        _plot_line(ax, steps, vals, C_ADV)
        _setup_ax(ax, "Trajectory Adv Std (signal strength)")

        # Entropy
        ax = axes[1, 0]
        vals = [r.get("rl_entropy", 0) for r in rl_records]
        _plot_line(ax, steps, vals, C_ENT)
        _setup_ax(ax, "Policy Entropy", "entropy")

        # Exploration: logstd values
        ax = axes[1, 1]
        all_records = records
        s = get_steps(all_records, "nm_logstd_prim")
        if s:
            _plot_line(ax, s, get_field(all_records, "nm_logstd_prim"), C_PRIM,
                       "primitives", linewidth=2.5)
            _plot_line(ax, s, get_field(all_records, "nm_logstd_key"), C_KEY,
                       "key", linewidth=2.5)
            _plot_line(ax, s, get_field(all_records, "nm_logstd_decay"), C_DECAY,
                       "decay", linewidth=2.0)
            ax.legend()
        _setup_ax(ax, "Policy Log-Std (exploration)", "log_std")

        # Neuromod grad norm
        ax = axes[1, 2]
        vals = [r.get("rl_nm_grad_norm", 0) for r in rl_records]
        _plot_line(ax, steps, vals, C_GRAD)
        _setup_ax(ax, "Neuromod Gradient Norm")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved: {output_path}")


def plot_memory_health(records, output_path):
    """Memory graph state, LM coupling, plasticity."""
    mem_records = [r for r in records if "mem_h_norm" in r]
    if not mem_records:
        print("  No memory metrics found, skipping memory_health")
        return

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle("Memory Graph Health", fontsize=16, fontweight="bold")

        steps = [r["step"] for r in mem_records]

        # Row 1: State + coupling
        ax = axes[0, 0]
        _plot_line(ax, steps, [r.get("mem_h_norm", 0) for r in mem_records],
                   C_PRIM, "h norm")
        _plot_line(ax, steps, [r.get("mem_msg_norm", 0) for r in mem_records],
                   C_KEY, "msg norm")
        ax.legend()
        _setup_ax(ax, "State Norms", "L2 norm")

        ax = axes[0, 1]
        _plot_line(ax, steps,
                   [r.get("mem_gate_mean", 0.5) for r in mem_records],
                   C_GATE, "mean")
        vals_min = [r.get("mem_gate_min", 0.5) for r in mem_records]
        vals_max = [r.get("mem_gate_max", 0.5) for r in mem_records]
        ax.fill_between(steps, vals_min, vals_max, alpha=0.15, color=C_GATE)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="#999", linestyle="--", linewidth=0.8)
        _setup_ax(ax, "Memory Gate (sigmoid)", "gate value")

        ax = axes[0, 2]
        _plot_line(ax, steps, [r.get("mem_prim_std", 0) for r in mem_records],
                   C_PRIM)
        _setup_ax(ax, "Primitive Diversity", "std across neurons")

        # Row 2: Decay, firing, saturation
        ax = axes[1, 0]
        _plot_line(ax, steps,
                   [r.get("mem_decay_mean", 0) for r in mem_records],
                   C_DECAY, "mean")
        vals_std = [r.get("mem_decay_std", 0) for r in mem_records]
        mean_arr = np.array([r.get("mem_decay_mean", 0) for r in mem_records])
        std_arr = np.array(vals_std)
        ax.fill_between(steps, mean_arr - std_arr, mean_arr + std_arr,
                        alpha=0.12, color=C_DECAY)
        ax.set_ylim(-0.05, 1.05)
        _setup_ax(ax, "Decay (higher = longer memory)", "sigmoid(decay_logit)")

        ax = axes[1, 1]
        _plot_line(ax, steps,
                   [r.get("mem_msg_mag_port", 0) for r in mem_records],
                   C_FIRE, "port neurons")
        _plot_line(ax, steps,
                   [r.get("mem_msg_mag_nonport", 0) for r in mem_records],
                   C_FIRE2, "internal neurons")
        ax.legend()
        _setup_ax(ax, "Message Magnitude (port vs internal)", "magnitude")

        ax = axes[1, 2]
        _plot_line(ax, steps,
                   [r.get("mem_tanh_saturated", 0) for r in mem_records],
                   C_SAT)
        _setup_ax(ax, "tanh Saturation", "fraction |msg| > 0.95")

        # Row 3: Key/primitive diversity, usage, plasticity
        ax = axes[2, 0]
        _plot_line(ax, steps,
                   [r.get("mem_key_diversity", 0) for r in mem_records],
                   C_KEY, "key diversity")
        _plot_line(ax, steps,
                   [r.get("mem_prim_std", 0) for r in mem_records],
                   C_PRIM, "prim diversity")
        ax.legend()
        _setup_ax(ax, "Key & Primitive Diversity")

        ax = axes[2, 1]
        _plot_line(ax, steps,
                   [r.get("mem_usage_frac", 0) for r in mem_records],
                   C_USAGE, "alive neurons")
        _plot_line(ax, steps,
                   [r.get("mem_phi_pos_frac", 0) for r in mem_records],
                   C_PHI, "phi > 0 frac")
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        _setup_ax(ax, "Neuron Usage & Co-activation")

        ax = axes[2, 2]
        rewires = [r.get("mem_plasticity_rewires", 0) for r in mem_records]
        ax.plot(steps, rewires, color=C_PHI, linewidth=2.0)
        _setup_ax(ax, "Cumulative Plasticity Rewires", "total rewires")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved: {output_path}")


def plot_pcm_health(records, output_path):
    """PCM surprise, prediction loss, and aux_loss contribution."""
    pcm_records = [r for r in records if "pcm_surprise_mean" in r]
    if not pcm_records:
        print("  No PCM metrics found, skipping pcm_health")
        return

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("PCM Health", fontsize=16, fontweight="bold")

        steps = [r["step"] for r in pcm_records]

        # Surprise magnitude
        ax = axes[0, 0]
        _plot_line(ax, steps,
                   [r["pcm_surprise_mean"] for r in pcm_records],
                   C_PRIM, "mean")
        _plot_line(ax, steps,
                   [r["pcm_surprise_max"] for r in pcm_records],
                   C_DECAY, "max")
        ax.legend()
        _setup_ax(ax, "Surprise Magnitude (L2 norm)", "norm")

        # Surprise spread across CCs
        ax = axes[0, 1]
        _plot_line(ax, steps,
                   [r.get("pcm_surprise_cc_min", 0) for r in pcm_records],
                   C_FIRE, "CC min")
        _plot_line(ax, steps,
                   [r.get("pcm_surprise_cc_max", 0) for r in pcm_records],
                   C_DECAY, "CC max")
        ax.legend()
        _setup_ax(ax, "Surprise per CC (min/max)", "norm")

        # Prediction loss per CC
        ax = axes[1, 0]
        _plot_line(ax, steps,
                   [r.get("pcm_pred_loss_mean", 0) for r in pcm_records],
                   C_LOSS, "mean")
        vals_min = [r.get("pcm_pred_loss_min", 0) for r in pcm_records]
        vals_max = [r.get("pcm_pred_loss_max", 0) for r in pcm_records]
        ax.fill_between(steps, vals_min, vals_max, alpha=0.15, color=C_LOSS)
        _setup_ax(ax, "PCM Prediction Loss (per CC range)", "MSE")

        # Aux loss (PCM contribution to total loss)
        ax = axes[1, 1]
        _plot_line(ax, steps,
                   [r.get("aux_loss", 0) for r in pcm_records],
                   C_GATE, "aux_loss")
        _setup_ax(ax, "PCM Aux Loss (weighted)", "loss")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved: {output_path}")


def plot_connectivity_snapshot(snapshot_path, output_path):
    """Visualize neuron graph connectivity from a snapshot."""
    snap = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    for k, v in snap.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
            snap[k] = v.float()
    cfg = snap["config"]
    N = cfg["N"]
    C = cfg["C"]
    K = cfg["K"]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Memory Graph Snapshot (step {snap['step']})",
                     fontsize=16, fontweight="bold")

        x = np.arange(N)

        # Per-neuron state norms
        ax = axes[0, 0]
        h_norm = snap["h_norm_per_neuron"].numpy()
        msg_norm = snap["msg_norm_per_neuron"].numpy()
        ax.bar(x, h_norm, alpha=0.7, label="h norm", width=1.0, color=C_PRIM)
        ax.bar(x, msg_norm, alpha=0.5, label="msg norm", width=1.0, color=C_KEY)
        ax.axvline(x=C - 0.5, color='red', linestyle='--', alpha=0.7,
                   label=f"port neurons (0-{C-1})")
        ax.legend()
        _setup_ax(ax, "Per-Neuron State Norms", xlabel="Neuron ID")

        # Decay per neuron
        ax = axes[0, 1]
        decay = snap["decay_per_neuron"].numpy()
        ax.bar(x, decay, width=1.0, alpha=0.7, color=C_DECAY)
        ax.axvline(x=C - 0.5, color='red', linestyle='--', alpha=0.7)
        ax.set_ylim(0, 1)
        _setup_ax(ax, "Decay per Neuron", xlabel="Neuron ID")

        # Key vector heatmap (first 64 neurons)
        ax = axes[1, 0]
        if "key_per_neuron" in snap:
            key_data = snap["key_per_neuron"][:64].numpy()
            im = ax.imshow(key_data, aspect='auto', cmap='RdBu_r')
            plt.colorbar(im, ax=ax)
            _setup_ax(ax, "Key Vectors (neurons 0-63)",
                      ylabel="Neuron ID", xlabel="Dimension")
        else:
            ax.text(0.5, 0.5, "No key data", ha='center', va='center')
            _setup_ax(ax, "Key Vectors")

        # Fan-in histogram
        ax = axes[1, 1]
        conn_idx = snap["conn_indices"].numpy()
        fan_in = np.bincount(conn_idx.flatten(), minlength=N)
        ax.bar(x, fan_in, width=1.0, alpha=0.7, color=C_USAGE)
        ax.axvline(x=C - 0.5, color='red', linestyle='--', alpha=0.7,
                   label="port neurons")
        ax.legend()
        _setup_ax(ax, "Fan-In per Neuron", ylabel="# incoming",
                  xlabel="Neuron ID")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved: {output_path}")


def plot_neuron_graph(snapshot_path, output_path):
    """Visualize neurons using UMAP on primitives with routing weight edges.

    Layout: UMAP projection of primitive vectors (neurons with similar
    broadcast directions cluster together).
    Node color: activation (h_norm). Node size: firing rate.
    Edge color: routing weight intensity (if available in snapshot).
    Port neurons: larger squares with red border.
    """
    try:
        import umap
    except ImportError:
        print("  UMAP not installed (pip install umap-learn), skipping neuron graph")
        return

    snap = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    for k, v in snap.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
            snap[k] = v.float()

    cfg = snap["config"]
    N = cfg["N"]
    C = cfg["C"]
    K = cfg["K"]

    conn_idx = snap["conn_indices"].numpy()  # [N, K]
    h_norm = snap["h_norm_per_neuron"].numpy()  # [N]

    if "msg_magnitude_per_neuron" in snap:
        firing = snap["msg_magnitude_per_neuron"].numpy()
    elif "firing_rate_per_neuron" in snap:
        firing = snap["firing_rate_per_neuron"].numpy()
    else:
        firing = np.ones(N) * 0.1

    # UMAP on primitives — neurons with similar broadcast directions cluster
    primitives = snap["primitives_mean"].numpy()  # [N, D]
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                        random_state=42, metric='cosine')
    coords = reducer.fit_transform(primitives)  # [N, 2]

    # Routing weights (if available)
    has_routing = "routing_weights" in snap
    routing_w = snap["routing_weights"].numpy() if has_routing else None  # [N, K]

    # Build edges with weights
    all_edges = []
    all_weights = []
    for dst in range(N):
        for k_idx in range(K):
            src = int(conn_idx[dst, k_idx])
            w = float(routing_w[dst, k_idx]) if has_routing else 1.0 / K
            all_edges.append((src, dst))
            all_weights.append(w)
    all_weights = np.array(all_weights)

    # Draw top edges by weight (strongest routing connections)
    n_draw = min(len(all_edges), max(N * 5, 5000))
    if has_routing:
        top_idx = np.argsort(all_weights)[-n_draw:]
    else:
        rng = np.random.RandomState(42)
        top_idx = rng.choice(len(all_edges), n_draw, replace=False)
    draw_edges = [all_edges[i] for i in top_idx]
    draw_weights = all_weights[top_idx]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        title = f"Neuron Graph — UMAP on Primitives ({N} neurons, step {snap['step']})"
        if has_routing:
            title += f", top {n_draw} edges by routing weight"
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # Draw edges with weight-based color/alpha
        w_max = max(draw_weights.max(), 1e-6)
        for (src, dst), w in zip(draw_edges, draw_weights):
            intensity = min(w / w_max, 1.0)
            alpha = 0.02 + 0.3 * intensity
            ax.plot([coords[src, 0], coords[dst, 0]],
                    [coords[src, 1], coords[dst, 1]],
                    color=(0.3, 0.3, 0.7, alpha), linewidth=0.3)

        # Node sizes
        node_sizes = 5 + 40 * firing

        # Internal neurons
        internal = np.arange(C, N)
        sc = ax.scatter(coords[internal, 0], coords[internal, 1],
                        c=h_norm[internal], s=node_sizes[internal],
                        cmap='viridis', vmin=0, vmax=max(h_norm.max(), 1e-6),
                        edgecolors='none', zorder=3)

        # Port neurons (larger, square marker, red edge)
        ports = np.arange(C)
        ax.scatter(coords[ports, 0], coords[ports, 1],
                   c=h_norm[ports], s=node_sizes[ports] * 4,
                   cmap='viridis', vmin=0, vmax=max(h_norm.max(), 1e-6),
                   marker='s', edgecolors='#d93025', linewidths=1.5, zorder=4)

        # Port labels
        for i in range(C):
            ax.annotate(f'P{i}', (coords[i, 0], coords[i, 1]),
                        fontsize=6, fontweight='bold', color='#333',
                        ha='center', va='bottom', zorder=5)

        # Colorbar
        cbar = plt.colorbar(sc, ax=ax, shrink=0.3, pad=0.02)
        cbar.set_label("Activation (h norm)", fontsize=9)

        # Legend
        ax.plot([], [], 's', color='#d93025', markersize=10,
                label=f'Port neurons (0-{C-1})')
        ax.plot([], [], 'o', color='#888', markersize=4,
                label='Internal neurons (size=firing rate)')
        if has_routing:
            ax.plot([], [], '-', color=(0.3, 0.3, 0.7, 0.5), linewidth=1,
                    label='Routing weight (stronger = brighter)')
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

        ax.set_xlabel("UMAP 1 (primitive direction)", fontsize=10)
        ax.set_ylabel("UMAP 2 (primitive direction)", fontsize=10)
        ax.grid(True, alpha=0.2)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved: {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Plot v8 training diagnostics")
    p.add_argument("run_dir", help="Path to run directory")
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

    # Snapshot-based plots (connectivity + neuron graph)
    snap_dir = os.path.join(args.run_dir, "snapshots")
    snap_step = args.snapshot

    # Auto-find latest snapshot if none specified
    if snap_step is None and os.path.exists(snap_dir):
        available = sorted(os.listdir(snap_dir))
        if available:
            latest = available[-1]
            try:
                snap_step = int(latest.split("_")[1].split(".")[0])
            except (IndexError, ValueError):
                pass

    if snap_step is not None:
        snap_path = os.path.join(snap_dir, f"step_{snap_step:06d}.pt")
        if os.path.exists(snap_path):
            plot_connectivity_snapshot(
                snap_path,
                os.path.join(plot_dir, f"connectivity_{snap_step}.png"))
            plot_neuron_graph(
                snap_path,
                os.path.join(plot_dir, f"neuron_graph_{snap_step}.png"))
        else:
            print(f"  Snapshot not found: {snap_path}")


if __name__ == "__main__":
    main()
