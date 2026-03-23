"""Plot v8 training curves and memory graph diagnostics.

Usage:
    python -m scripts.plot_training outputs/v8/<run_id>/
    python -m scripts.plot_training outputs/v8/<run_id>/ --snapshot 5000

Produces:
    <run_dir>/plots/training_curves.png     — loss, ppl, LR, throughput
    <run_dir>/plots/rl_curves.png           — RL policy loss, value, advantages, exploration
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
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.color": "#555",
    "ytick.color": "#555",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "grid.color": "#ddd",
    "grid.alpha": 0.7,
    "text.color": "#222",
    "legend.fontsize": 8,
    "legend.facecolor": "white",
    "legend.edgecolor": "#ccc",
}

# Colors that read well on white
C_LOSS = "#1a73e8"
C_PPL = "#d93025"
C_LR = "#188038"
C_LR2 = "#e37400"
C_TPUT = "#7b1fa2"
C_RL = "#e37400"
C_VAL = "#0d904f"
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
C_CW = "#e37400"
C_SAT = "#c2185b"


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


def _plot_line(ax, steps, vals, color, label=None):
    """Plot raw + smoothed line with good visibility."""
    arr = np.array(vals)
    if len(arr) > 200:
        # Enough data: faint raw + bold smooth
        ax.plot(steps, arr, alpha=0.2, color=color, linewidth=0.5)
        s = smooth(arr)
        ax.plot(steps[:len(s)], s, color=color, linewidth=2.5, label=label)
    elif len(arr) > 50:
        # Medium data: lighter raw + thinner smooth
        ax.plot(steps, arr, alpha=0.3, color=color, linewidth=0.8)
        s = smooth(arr, window=max(len(arr) // 8, 3))
        ax.plot(steps[:len(s)], s, color=color, linewidth=2.0, label=label)
    else:
        # Few points: just draw the line directly, fully visible
        ax.plot(steps, arr, color=color, linewidth=2.0, alpha=0.9, label=label)


def _setup_ax(ax, title, ylabel=None, xlabel="Step"):
    ax.set_title(title, fontweight="bold")
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
        fig.suptitle("Training Curves", fontsize=16, fontweight="bold", color="white")

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
            ax.plot(lr_steps, lr, color=C_LR, linewidth=1.5, label="LM")
        if nm_lr_steps:
            ax.plot(nm_lr_steps, nm_lr, color=C_LR2, linewidth=1.5,
                    linestyle="--", label="Neuromod")
        ax.legend()
        _setup_ax(ax, "Learning Rate Schedule", "LR")

        # Throughput
        ax = axes[1, 1]
        tok_s = get_field(records, "tok_s")
        tok_steps = get_steps(records, "tok_s")
        if tok_steps:
            tok_k = [t / 1000 for t in tok_s]
            _plot_line(ax, tok_steps, tok_k, C_TPUT)
            _setup_ax(ax, "Throughput", "K tok/s")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved: {output_path}")


def plot_rl_curves(records, output_path):
    """RL policy loss, value loss, advantages, exploration, explained variance."""
    rl_records = [r for r in records if "rl_policy_loss" in r]
    if not rl_records:
        print("  No RL data found, skipping rl_curves")
        return

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("RL / Neuromodulator", fontsize=16, fontweight="bold", color="white")

        steps = [r["step"] for r in rl_records]

        # Policy loss
        ax = axes[0, 0]
        vals = [r["rl_policy_loss"] for r in rl_records]
        _plot_line(ax, steps, vals, C_RL)
        _setup_ax(ax, "Policy Loss")

        # Value loss
        ax = axes[0, 1]
        vals = [r.get("rl_value_loss", 0) for r in rl_records]
        _plot_line(ax, steps, vals, C_VAL)
        _setup_ax(ax, "Value Loss (critic)")

        # Explained variance
        ax = axes[0, 2]
        vals = [r.get("rl_explained_var", 0) for r in rl_records]
        _plot_line(ax, steps, vals, C_ADV)
        ax.axhline(y=0, color="#666", linestyle="--", linewidth=0.8)
        ax.axhline(y=1, color="#666", linestyle="--", linewidth=0.8)
        _setup_ax(ax, "Value Explained Variance", "EV (0=useless, 1=perfect)")

        # Advantage std (signal strength)
        ax = axes[1, 0]
        vals = [r.get("rl_adv_std", 0) for r in rl_records]
        _plot_line(ax, steps, vals, C_ADV)
        _setup_ax(ax, "Advantage Std (signal strength)")

        # Exploration: logstd values
        ax = axes[1, 1]
        all_records = records  # logstds are logged every step
        s = get_steps(all_records, "nm_logstd_prim")
        if s:
            _plot_line(ax, s, get_field(all_records, "nm_logstd_prim"), C_PRIM,
                       "primitives")
            _plot_line(ax, s, get_field(all_records, "nm_logstd_conn"), C_CW,
                       "conn_weights")
            _plot_line(ax, s, get_field(all_records, "nm_logstd_decay"), C_DECAY,
                       "decay")
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
        fig.suptitle("Memory Graph Health", fontsize=16, fontweight="bold",
                     color="white")

        steps = [r["step"] for r in mem_records]

        # Row 1: State + coupling
        ax = axes[0, 0]
        _plot_line(ax, steps, [r.get("mem_h_norm", 0) for r in mem_records],
                   C_PRIM, "h norm")
        _plot_line(ax, steps, [r.get("mem_msg_norm", 0) for r in mem_records],
                   C_CW, "msg norm")
        ax.legend()
        _setup_ax(ax, "State Norms", "L2 norm")

        ax = axes[0, 1]
        _plot_line(ax, steps,
                   [r.get("mem_gate_mean", 0.5) for r in mem_records],
                   C_GATE, "mean")
        vals_min = [r.get("mem_gate_min", 0.5) for r in mem_records]
        vals_max = [r.get("mem_gate_max", 0.5) for r in mem_records]
        ax.fill_between(steps, vals_min, vals_max, alpha=0.2, color=C_GATE)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="#666", linestyle="--", linewidth=0.8)
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
                        alpha=0.15, color=C_DECAY)
        ax.set_ylim(-0.05, 1.05)
        _setup_ax(ax, "Decay (higher = longer memory)", "sigmoid(decay_logit)")

        ax = axes[1, 1]
        _plot_line(ax, steps,
                   [r.get("mem_firing_rate_port", 0) for r in mem_records],
                   C_FIRE, "port neurons")
        _plot_line(ax, steps,
                   [r.get("mem_firing_rate_nonport", 0) for r in mem_records],
                   C_FIRE2, "internal neurons")
        ax.legend()
        _setup_ax(ax, "Firing Rate (port vs internal)", "rate")

        ax = axes[1, 2]
        _plot_line(ax, steps,
                   [r.get("mem_tanh_saturated", 0) for r in mem_records],
                   C_SAT)
        _setup_ax(ax, "tanh Saturation", "fraction |msg| > 0.95")

        # Row 3: Connectivity, usage, plasticity
        ax = axes[2, 0]
        _plot_line(ax, steps,
                   [r.get("mem_cw_std", 0) for r in mem_records],
                   C_CW, "weight std")
        _plot_line(ax, steps,
                   [r.get("mem_cw_near_zero", 0) for r in mem_records],
                   C_GRAD, "near-zero frac")
        ax.legend()
        _setup_ax(ax, "Connection Weights", "value")

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
        ax.plot(steps, rewires, color=C_PHI, linewidth=1.5)
        _setup_ax(ax, "Cumulative Plasticity Rewires", "total rewires")

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
                     fontsize=16, fontweight="bold", color="white")

        x = np.arange(N)

        # Per-neuron state norms
        ax = axes[0, 0]
        h_norm = snap["h_norm_per_neuron"].numpy()
        msg_norm = snap["msg_norm_per_neuron"].numpy()
        ax.bar(x, h_norm, alpha=0.7, label="h norm", width=1.0, color=C_PRIM)
        ax.bar(x, msg_norm, alpha=0.5, label="msg norm", width=1.0, color=C_CW)
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

        # Connection weight heatmap
        ax = axes[1, 0]
        cw = snap["cw_per_neuron"][:64].numpy()
        im = ax.imshow(cw, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        plt.colorbar(im, ax=ax)
        _setup_ax(ax, f"Connection Weights (neurons 0-63)",
                  ylabel="Neuron ID", xlabel=f"Connection (K={K})")

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
    """Visualize the full neuron graph as a network.

    All neurons shown. Node color = activation magnitude, node size = firing rate.
    Only the top edges by |weight| are drawn to keep the plot readable.
    Edge color: red = positive weight, blue = negative weight, intensity = magnitude.
    Port neurons drawn as larger squares with red border.
    """
    import networkx as nx

    snap = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    for k, v in snap.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
            snap[k] = v.float()

    cfg = snap["config"]
    N = cfg["N"]
    C = cfg["C"]
    K = cfg["K"]

    conn_idx = snap["conn_indices"].numpy()    # [N, K]
    cw = snap["cw_per_neuron"].numpy()         # [N, K] batch-averaged weights
    h_norm = snap["h_norm_per_neuron"].numpy()  # [N]
    # Handle both old (usage_per_neuron) and new (firing_rate_per_neuron) formats
    if "firing_rate_per_neuron" in snap:
        firing = snap["firing_rate_per_neuron"].numpy()
    elif "usage_per_neuron" in snap:
        firing = snap["usage_per_neuron"].numpy()
    else:
        firing = np.ones(N) * 0.1  # fallback

    # Build full graph
    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    all_edges = []
    all_weights = []
    for dst in range(N):
        for k_idx in range(K):
            src = int(conn_idx[dst, k_idx])
            w = float(cw[dst, k_idx])
            all_edges.append((src, dst))
            all_weights.append(w)

    all_weights = np.array(all_weights)

    # Only draw top edges by |weight| to keep plot readable
    # With N=1024 × K=96 = 98K edges, show top ~5K (top 5%)
    n_draw = min(len(all_weights), max(N * 5, 5000))
    top_idx = np.argsort(np.abs(all_weights))[-n_draw:]
    draw_edges = [all_edges[i] for i in top_idx]
    draw_weights = all_weights[top_idx]

    for src, dst in draw_edges:
        G.add_edge(src, dst)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(24, 24))
        fig.suptitle(
            f"Neuron Graph ({N} neurons, top {n_draw} edges by |weight|, "
            f"step {snap['step']})",
            fontsize=16, fontweight="bold")

        # Layout: spring with more space
        pos = nx.spring_layout(G, k=3.0 / np.sqrt(N), iterations=80, seed=42)

        # Node sizes: scale by firing rate
        node_sizes = 8 + 60 * firing  # small base, bigger = more active

        # Separate port vs non-port
        port_nodes = list(range(C))
        internal_nodes = list(range(C, N))

        # Edge colors by weight
        w_abs_max = max(np.abs(draw_weights).max(), 1e-6)
        edge_colors = []
        for w in draw_weights:
            intensity = min(abs(w) / w_abs_max, 1.0)
            alpha = 0.05 + 0.4 * intensity
            if w >= 0:
                edge_colors.append((*plt.cm.Reds(0.3 + 0.7 * intensity)[:3], alpha))
            else:
                edge_colors.append((*plt.cm.Blues(0.3 + 0.7 * intensity)[:3], alpha))

        # Draw edges
        nx.draw_networkx_edges(
            G, pos, edgelist=draw_edges, edge_color=edge_colors,
            width=0.3, arrows=False, ax=ax)

        # Draw internal neurons
        if internal_nodes:
            nx.draw_networkx_nodes(
                G, pos, nodelist=internal_nodes,
                node_color=h_norm[C:N],
                node_size=node_sizes[C:N],
                cmap=plt.cm.viridis, vmin=0,
                vmax=max(h_norm.max(), 1e-6),
                edgecolors="none", ax=ax)

        # Draw port neurons (larger, square, red border)
        if port_nodes:
            nx.draw_networkx_nodes(
                G, pos, nodelist=port_nodes,
                node_color=h_norm[:C],
                node_size=node_sizes[:C] * 4,
                node_shape="s",
                cmap=plt.cm.viridis, vmin=0,
                vmax=max(h_norm.max(), 1e-6),
                edgecolors="#d93025", linewidths=1.5, ax=ax)

        # Port neuron labels
        port_labels = {i: f"P{i}" for i in port_nodes}
        nx.draw_networkx_labels(G, pos, port_labels, font_size=6,
                                font_color="#333", font_weight="bold", ax=ax)

        # Colorbar for node activation
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis,
            norm=plt.Normalize(vmin=0, vmax=max(h_norm.max(), 1e-6)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.3, pad=0.01)
        cbar.set_label("Activation (h norm)", fontsize=9)

        # Legend
        ax.plot([], [], 's', color='#d93025', markersize=10,
                label=f'Port neurons (0-{C-1})')
        ax.plot([], [], 'o', color='#888', markersize=4,
                label='Internal neurons (size=firing rate)')
        ax.plot([], [], '-', color='#d93025', alpha=0.6, linewidth=2,
                label='Positive weight')
        ax.plot([], [], '-', color='#1a73e8', alpha=0.6, linewidth=2,
                label='Negative weight')
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

        ax.axis('off')
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
            # Parse step from filename "step_001000.pt"
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
