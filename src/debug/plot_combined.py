"""
Combined multi-phase training plots and per-phase plot orchestration.

Usage (standalone):
    python -m src.debug.plot_combined checkpoints/metrics.jsonl
    python -m src.debug.plot_combined checkpoints/metrics.jsonl checkpoints/

Programmatic:
    from src.debug.plot_combined import generate_phase_plots, generate_combined_plot
    generate_phase_plots("metrics.jsonl", "A", "checkpoints/")
    generate_combined_plot("metrics.jsonl", ["A", "B", "C"], "checkpoints/")
"""

import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Per-phase plot generation (delegates to existing plot scripts)
# ============================================================================

PLOT_SCRIPTS = [
    "plot_training",
    "plot_pm",
    "plot_em",
    "plot_gates",
    "plot_wm",
    "plot_gradients",
]


def generate_phase_plots(metrics_file: str, phase: str, save_dir: str):
    """Run all 6 existing plot scripts after a phase completes.

    Note: plots are cumulative (all phases up through the named phase),
    because the underlying scripts read the full JSONL without filtering.
    """
    os.makedirs(save_dir, exist_ok=True)
    phase = phase.upper()
    for name in PLOT_SCRIPTS:
        out_path = os.path.join(save_dir, f"through_phase_{phase}_{name}.png")
        cmd = [sys.executable, "-m", f"src.debug.{name}", metrics_file, out_path]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
            print(f"  Plot: {out_path}")
        except subprocess.CalledProcessError as e:
            print(f"  Warning: {name} failed: {e.stderr.strip()[:200] if e.stderr else 'unknown error'}")
        except subprocess.TimeoutExpired:
            print(f"  Warning: {name} timed out")


# ============================================================================
# JSONL helpers
# ============================================================================

def load_metrics(path: str) -> list[dict]:
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


def _safe_float(val, default=float("nan")):
    """Convert a value to float, mapping None to NaN."""
    if val is None:
        return default
    return float(val)


def smooth(values, window):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# ============================================================================
# Combined multi-phase plot
# ============================================================================

SMOOTH_WINDOW = 50
PHASE_COLORS = {
    "A": "#1f77b4",
    "B": "#ff7f0e",
    "C": "#d62728",
}


def _detect_phase_boundaries(records: list[dict]) -> list[dict]:
    """Extract phase_start/phase_end markers from JSONL records."""
    boundaries = []
    for r in records:
        mode = r.get("mode", "")
        if mode in ("phase_start", "phase_end"):
            boundaries.append(r)
    return boundaries


def generate_combined_plot(metrics_file: str, phases: list[str], save_dir: str):
    """Generate a 2x2 combined plot with phase boundaries."""
    os.makedirs(save_dir, exist_ok=True)
    all_records = load_metrics(metrics_file)
    if not all_records:
        print("No records for combined plot.")
        return

    train_records = [r for r in all_records if r.get("mode", "train") == "train"]
    val_records = [r for r in all_records if r.get("mode") == "val"]
    boundaries = _detect_phase_boundaries(all_records)

    if not train_records:
        print("No training records for combined plot.")
        return

    # Extract boundary steps for vertical lines
    phase_starts = {}
    for b in boundaries:
        if b.get("mode") == "phase_start":
            p = b.get("phase", "?")
            s = b.get("step", 0)
            if p not in phase_starts:
                phase_starts[p] = s

    steps = np.array([r.get("step", i) for i, r in enumerate(train_records)], dtype=float)
    loss = np.array([_safe_float(r.get("loss")) for r in train_records])
    ppl = np.array([_safe_float(r.get("ppl")) for r in train_records])
    lr = np.array([_safe_float(r.get("lr")) for r in train_records])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Combined Training: Phases {', '.join(phases)}", fontsize=14)

    def _add_phase_boundaries(ax, y_range=None):
        """Add vertical phase boundary lines and labels."""
        for phase_name, step in phase_starts.items():
            ax.axvline(x=step, color=PHASE_COLORS.get(phase_name, "gray"),
                       linestyle="--", alpha=0.6, linewidth=1.5)
            if y_range is not None:
                y_pos = y_range[1] * 0.95 if y_range[1] > 0 else 1.0
            else:
                y_pos = ax.get_ylim()[1] * 0.95
            ax.text(step + (steps[-1] - steps[0]) * 0.005, y_pos,
                    f"Phase {phase_name}", fontsize=9, alpha=0.7,
                    color=PHASE_COLORS.get(phase_name, "gray"),
                    verticalalignment="top")

    # --- Panel 1: Loss (raw + smoothed) ---
    ax = axes[0, 0]
    ax.plot(steps, loss, alpha=0.15, color="blue", linewidth=0.5)
    if len(loss) > SMOOTH_WINDOW:
        s = smooth(loss, SMOOTH_WINDOW)
        ax.plot(steps[SMOOTH_WINDOW - 1:], s, color="blue",
                label=f"smooth({SMOOTH_WINDOW})", linewidth=1.5)
    if val_records:
        v_steps = [r.get("step", 0) for r in val_records]
        v_loss = [_safe_float(r.get("val_loss")) for r in val_records]
        ax.scatter(v_steps, v_loss, color="black", s=10, alpha=0.6, label="val", zorder=5)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _add_phase_boundaries(ax)

    # --- Panel 2: Perplexity (log scale) ---
    ax = axes[0, 1]
    ax.plot(steps, ppl, alpha=0.15, color="orange", linewidth=0.5)
    if len(ppl) > SMOOTH_WINDOW:
        s = smooth(ppl, SMOOTH_WINDOW)
        ax.plot(steps[SMOOTH_WINDOW - 1:], s, color="orange",
                label=f"smooth({SMOOTH_WINDOW})", linewidth=1.5)
    ax.set_yscale("log")
    if val_records:
        v_steps = [r.get("step", 0) for r in val_records]
        v_ppl = [_safe_float(r.get("val_ppl")) for r in val_records]
        ax.scatter(v_steps, v_ppl, color="black", s=10, alpha=0.6, label="val", zorder=5)
    ax.set_xlabel("step")
    ax.set_ylabel("perplexity")
    ax.set_title("Perplexity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _add_phase_boundaries(ax)

    # --- Panel 3: Learning rate ---
    ax = axes[1, 0]
    ax.plot(steps, lr, color="green", linewidth=1.0)
    ax.set_xlabel("step")
    ax.set_ylabel("lr")
    ax.set_title("Learning Rate")
    ax.grid(True, alpha=0.3)
    _add_phase_boundaries(ax)

    # --- Panel 4: Validation PPL ---
    ax = axes[1, 1]
    if val_records:
        v_steps = [r.get("step", 0) for r in val_records]
        v_ppl = [_safe_float(r.get("val_ppl")) for r in val_records]
        ax.plot(v_steps, v_ppl, color="darkred", marker="o", markersize=3,
                linewidth=1.0, label="val PPL")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No validation data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, alpha=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("perplexity")
    ax.set_title("Validation Perplexity")
    ax.grid(True, alpha=0.3)
    _add_phase_boundaries(ax)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "plot_combined.png")
    plt.savefig(out_path, dpi=150)
    print(f"Combined plot: {out_path}")
    plt.close()


# ============================================================================
# Standalone entry point
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.debug.plot_combined <metrics.jsonl> [save_dir]")
        sys.exit(1)

    metrics_path = sys.argv[1]
    save_directory = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(metrics_path) or "."

    # Auto-detect phases from JSONL
    records = load_metrics(metrics_path)
    detected_phases = []
    for r in records:
        if r.get("mode") == "phase_start":
            p = r.get("phase")
            if p and p not in detected_phases:
                detected_phases.append(p)

    if not detected_phases:
        # Fall back: look at phase field in training records
        for r in records:
            p = r.get("phase")
            if p and p not in detected_phases:
                detected_phases.append(p)

    if not detected_phases:
        detected_phases = ["A"]  # fallback

    print(f"Detected phases: {detected_phases}")
    generate_combined_plot(metrics_path, detected_phases, save_directory)
