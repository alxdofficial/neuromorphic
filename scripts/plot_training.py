"""Plot training curves for the current v12 architecture (phase 1 and phase 2).

Usage:
    # Phase 1 (src.train or cycle phase 1)
    python -m scripts.plot_training outputs/v12/

    # Phase 2
    python -m scripts.plot_training outputs/v12/cycle_00/ --phase 2

Expects `{run_dir}/metrics.jsonl` (phase 1) or `{run_dir}/phase2_metrics.jsonl`
(phase 2). Writes PNG plots into `{run_dir}/plots/`.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    "aux": "#c2185b",
    "lm_grad": "#1a73e8",
    "mem_grad": "#d93025",
    "mod_grad": "#e37400",
    "mod": "#e37400",
    "state": "#1a73e8",
    "msg": "#188038",
    "inject": "#7b1fa2",
    "nid": "#c2185b",
    "h": "#1a73e8",
    "msg2": "#188038",
    "W": "#7b1fa2",
    "decay": "#d93025",
    "surprise": "#e37400",
    "drift": "#c2185b",
    "reward": "#188038",
    "log_pi": "#1a73e8",
    "codes": "#7b1fa2",
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
    if window < 2:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def get(records, key):
    return [r[key] for r in records if key in r and r[key] is not None]


def plot_line(ax, vals, color, label=None, **kwargs):
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return
    steps = np.arange(arr.size)
    if arr.size > 200:
        ax.plot(steps, arr, alpha=0.15, color=color, linewidth=0.5)
        s = smooth(arr)
        ax.plot(steps[: len(s)], s, color=color, linewidth=2.0, label=label, **kwargs)
    elif arr.size > 50:
        ax.plot(steps, arr, alpha=0.25, color=color, linewidth=0.8)
        s = smooth(arr, window=max(arr.size // 8, 3))
        ax.plot(steps[: len(s)], s, color=color, linewidth=2.0, label=label, **kwargs)
    else:
        ax.plot(steps, arr, color=color, linewidth=1.5, alpha=0.9, label=label, **kwargs)


def setup(ax, title, ylabel=None):
    ax.set_title(title, fontweight="bold", pad=8)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("Step")
    ax.grid(True)


# ============================================================================
# Phase 1
# ============================================================================


def plot_phase1_training(records, output_path):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"Phase 1 Training ({len(records)} steps)",
                     fontsize=14, fontweight="bold")

        plot_line(axes[0, 0], get(records, "loss"), C["loss"], "ce_loss")
        plot_line(axes[0, 0], get(records, "aux_loss"), C["aux"], "mem_pred_loss")
        axes[0, 0].legend()
        setup(axes[0, 0], "Losses", "nats")

        ppl = get(records, "ppl")
        if ppl:
            plot_line(axes[0, 1], ppl, C["ppl"])
            axes[0, 1].set_yscale("log")
            setup(axes[0, 1], "Perplexity", "PPL (log)")

        lr = get(records, "lr")
        if lr:
            axes[1, 0].plot(lr, color=C["lr"], linewidth=2.0)
            setup(axes[1, 0], "Learning Rate", "LR")

        tok_s = get(records, "tok_s")
        if tok_s:
            plot_line(axes[1, 1], [t / 1000 for t in tok_s], C["tput"])
            setup(axes[1, 1], "Throughput", "K tok/s")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Saved: {output_path}")


def plot_phase1_gradients(records, output_path):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Phase 1 Gradient Health", fontsize=14, fontweight="bold")

        plot_line(axes[0, 0], get(records, "lm_grad_norm"), C["lm_grad"], "LM")
        plot_line(axes[0, 0], get(records, "mem_grad_norm"), C["mem_grad"], "Memory")
        plot_line(axes[0, 0], get(records, "mod_grad_norm"), C["mod_grad"], "Modulator")
        axes[0, 0].legend()
        setup(axes[0, 0], "Group Gradient Norms (post-clip)", "L2")

        for key, label, color in [
            ("grad_mod_w1", "mod_w1", C["mod"]),
            ("grad_mod_w2", "mod_w2", C["mod"]),
            ("grad_state_w1", "state_w1", C["state"]),
            ("grad_state_w2", "state_w2", C["state"]),
            ("grad_msg_w1", "msg_w1", C["msg"]),
            ("grad_msg_w2", "msg_w2", C["msg"]),
            ("grad_inject_w", "inject_w", C["inject"]),
            ("grad_neuron_id", "neuron_id", C["nid"]),
        ]:
            vals = get(records, key)
            if vals and any(v > 0 for v in vals):
                ls = "--" if "w2" in key else "-"
                plot_line(axes[0, 1], vals, color, label, linestyle=ls)
        axes[0, 1].legend(fontsize=7)
        setup(axes[0, 1], "Per-Component Memory Grads (pre-clip)", "L2")

        for key, label, color in [
            ("mod_w1_norm", "mod_w1", C["mod"]),
            ("mod_w2_norm", "mod_w2", C["mod"]),
            ("state_w1_norm", "state_w1", C["state"]),
            ("state_w2_norm", "state_w2", C["state"]),
            ("msg_w1_norm", "msg_w1", C["msg"]),
            ("msg_w2_norm", "msg_w2", C["msg"]),
            ("inject_w_norm", "inject_w", C["inject"]),
            ("neuron_id_norm", "neuron_id", C["nid"]),
        ]:
            vals = get(records, key)
            if vals:
                ls = "--" if "w2" in key else "-"
                plot_line(axes[1, 0], vals, color, label, linestyle=ls)
        axes[1, 0].legend(fontsize=7)
        setup(axes[1, 0], "Weight Norms", "L2")

        plot_line(axes[1, 1], get(records, "mod_action_norm"), C["mod"], "action_norm")
        plot_line(axes[1, 1], get(records, "mod_action_var"), C["mod_grad"], "action_var")
        axes[1, 1].legend()
        setup(axes[1, 1], "Modulator Action Stats", "magnitude")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Saved: {output_path}")


def plot_phase1_memory(records, output_path):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle("Phase 1 Memory Health", fontsize=14, fontweight="bold")

        plot_line(axes[0, 0], get(records, "h_norm"), C["h"], "h")
        plot_line(axes[0, 0], get(records, "msg_norm"), C["msg2"], "msg")
        axes[0, 0].legend()
        setup(axes[0, 0], "Per-Element State Norms", "L2 / sqrt(N)")

        plot_line(axes[0, 1], get(records, "h_max"), C["h"], "h_max")
        plot_line(axes[0, 1], get(records, "msg_max"), C["msg2"], "msg_max")
        axes[0, 1].legend()
        setup(axes[0, 1], "State Max |abs|", "magnitude")

        plot_line(axes[1, 0], get(records, "W_norm"), C["W"], "W_norm")
        plot_line(axes[1, 0], get(records, "W_max"), C["W"], "W_max", linestyle="--")
        axes[1, 0].legend()
        setup(axes[1, 0], "W Norm / Max", "magnitude")

        plot_line(axes[1, 1], get(records, "W_sparsity"), C["W"])
        setup(axes[1, 1], "W Sparsity (|w| < 1e-4)", "fraction")

        plot_line(axes[2, 0], get(records, "decay_mean"), C["decay"], "mean")
        plot_line(axes[2, 0], get(records, "decay_std"), C["decay"], "std", linestyle="--")
        axes[2, 0].legend()
        setup(axes[2, 0], "Decay σ(decay_logit)", "probability")

        plot_line(axes[2, 1], get(records, "s_mem_live"), C["surprise"], "s_mem_live")
        plot_line(axes[2, 1], get(records, "s_mem_ema_fast"), C["surprise"],
                  "s_mem_ema_fast", linestyle="--")
        plot_line(axes[2, 1], get(records, "readout_drift_mean"), C["drift"], "drift")
        axes[2, 1].legend()
        setup(axes[2, 1], "Surprise / Drift", "nats / L1")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Saved: {output_path}")


# ============================================================================
# Phase 2
# ============================================================================


def plot_phase2_grpo(records, output_path):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Phase 2 GRPO ({len(records)} steps)",
                     fontsize=14, fontweight="bold")

        # Policy loss
        plot_line(axes[0, 0], get(records, "loss"), C["loss"])
        setup(axes[0, 0], "GRPO Loss", "-(A * log π)")

        # Reward distribution
        plot_line(axes[0, 1], get(records, "reward_mean"), C["reward"], "mean")
        rmin = get(records, "reward_min")
        rmax = get(records, "reward_max")
        if rmin and rmax:
            plot_line(axes[0, 1], rmin, C["reward"], "min", linestyle=":")
            plot_line(axes[0, 1], rmax, C["reward"], "max", linestyle="--")
        axes[0, 1].legend()
        setup(axes[0, 1], "Reward (−windowed CE)", "reward")

        # Log pi
        plot_line(axes[0, 2], get(records, "log_pi_mean"), C["log_pi"])
        setup(axes[0, 2], "Mean log π", "log-prob")

        # Modulator grad norm
        plot_line(axes[1, 0], get(records, "mod_grad_norm"), C["mod_grad"])
        setup(axes[1, 0], "Modulator Grad Norm", "L2")

        # Unique codes used per step
        plot_line(axes[1, 1], get(records, "n_unique_codes"), C["codes"])
        setup(axes[1, 1], "Unique Code Tuples / Step", "count")

        # Stage window over time
        windows = get(records, "stage_window")
        if windows:
            axes[1, 2].plot(windows, color=C["codes"], linewidth=2.0)
            setup(axes[1, 2], "Curriculum Window W", "tokens")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Saved: {output_path}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Plot training curves (v12)")
    parser.add_argument("run_dir", help="Run directory containing metrics.jsonl")
    parser.add_argument("--phase", choices=["1", "2"], default="1")
    args = parser.parse_args()

    if args.phase == "1":
        metrics_path = os.path.join(args.run_dir, "metrics.jsonl")
    else:
        metrics_path = os.path.join(args.run_dir, "phase2_metrics.jsonl")

    if not os.path.exists(metrics_path):
        print(f"No metrics at {metrics_path}")
        sys.exit(1)

    records = load_metrics(metrics_path)
    print(f"Loaded {len(records)} records from {metrics_path}")
    if not records:
        print("  (empty)")
        sys.exit(0)

    plots_dir = os.path.join(args.run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if args.phase == "1":
        plot_phase1_training(records, os.path.join(plots_dir, "phase1_training.png"))
        plot_phase1_gradients(records, os.path.join(plots_dir, "phase1_gradients.png"))
        plot_phase1_memory(records, os.path.join(plots_dir, "phase1_memory.png"))
    else:
        plot_phase2_grpo(records, os.path.join(plots_dir, "phase2_grpo.png"))


if __name__ == "__main__":
    main()
