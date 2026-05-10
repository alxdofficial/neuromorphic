"""Live training plot — multi-panel PNG saved to a fixed path so a
viewer can keep it open and watch it refresh.

Designed for a 3-min refresh cadence during long runs — call
`save_training_plots(history, path)` from the trainer loop every N
seconds. Overwrites in place.

History dict schema (all lists indexed by step except `val_*` which
are indexed by `val_step`):

    step:                 list[int]
    loss:                 list[float]
    grad_norm:            list[float]
    grad_norm_<comp>:     list[float]    one per component label
    param_norm_<comp>:    list[float]    optional, less critical
    surprise_mean:        list[float]
    surprise_std:         list[float]
    lr:                   list[list[float]]   per-param-group LRs
    tok_per_sec:          list[float]
    vram_peak_gb:         list[float]
    val_step:             list[int]
    val_loss:             dict[str, list[float]]    per-source val loss
                                                     (e.g., needle, fineweb)

Missing fields are skipped silently — partial history still plots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_training_plots(history: dict[str, Any], output_path: Path) -> None:
    """Render multi-panel training plot to `output_path` (PNG).

    Imports matplotlib lazily so the dependency is optional. If
    matplotlib isn't installed, the call is a no-op (the JSON dump
    still happens via `dump_history_json`).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend — no $DISPLAY needed
        import matplotlib.pyplot as plt
    except ImportError:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    steps = history.get("step", [])

    # ── 1. Train loss ────────────────────────────────────────────────
    ax = axes[0, 0]
    if steps and "loss" in history:
        ax.plot(steps, history["loss"], alpha=0.3, label="loss", linewidth=0.8)
        # Running window-32 mean for readability over noisy per-step loss.
        if len(steps) >= 8:
            window = max(8, len(steps) // 50)
            avg = _running_mean(history["loss"], window)
            ax.plot(steps, avg, label=f"avg{window}", color="C0")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Train loss")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

    # ── 2. Val loss (per-source) ─────────────────────────────────────
    ax = axes[0, 1]
    val_step = history.get("val_step", [])
    val_loss_dict = history.get("val_loss", {})
    if val_step and val_loss_dict:
        for source, vals in val_loss_dict.items():
            if not vals:
                continue
            label = f"{source}"
            if "needle" in source.lower():
                label += " (memory probe)"
            ax.plot(val_step[:len(vals)], vals, marker="o", markersize=3,
                    label=label,
                    linewidth=2.0 if "needle" in source.lower() else 1.0)
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Val loss (needle = memory-bridging probe)")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

    # ── 3. Inject SNR (memory contribution diagnostic) ───────────────
    # Replaces the prior "grad_norm overall" panel which was redundant
    # with panel 4 (per-component). B8 fix — surfaces the
    # MemInjectLayer._last_inj_norm / _last_hidden_norm ratio so we can
    # see if memory module silently collapses (scale → 0 → snr → 0).
    ax = axes[0, 2]
    if steps and "inject_snr" in history:
        snr = history["inject_snr"]
        ax.plot(steps[:len(snr)], snr, color="C2", label="inject/hidden")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Inject SNR (||scale·W_out(readout)|| / ||hidden||)")
    ax.set_xlabel("step")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")

    # ── 4. Per-component grad norm ───────────────────────────────────
    ax = axes[1, 0]
    plotted = False
    component_keys = sorted(
        k for k in history if k.startswith("grad_norm_") and k != "grad_norm"
    )
    for key in component_keys:
        comp = key[len("grad_norm_"):]
        vals = history[key]
        if not vals:
            continue
        ax.plot(steps[:len(vals)], vals, label=comp, linewidth=1.0)
        plotted = True
    if plotted:
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Grad norm by component")
    ax.set_xlabel("step")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")

    # ── 5. LR schedule ───────────────────────────────────────────────
    ax = axes[1, 1]
    if steps and history.get("lr"):
        lr_history = history["lr"]
        if lr_history:
            n_groups = len(lr_history[0])
            labels = ["memory_lr", "adapter_lr"][:n_groups]
            for g in range(n_groups):
                ys = [lrs[g] if g < len(lrs) else 0 for lrs in lr_history]
                ax.plot(steps[:len(ys)], ys,
                        label=labels[g] if g < len(labels) else f"group {g}")
            ax.legend(loc="upper right", fontsize=8)
    ax.set_title("LR schedule")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

    # ── 6. Surprise distribution ─────────────────────────────────────
    ax = axes[1, 2]
    if steps and "surprise_mean" in history:
        means = history["surprise_mean"]
        stds = history.get("surprise_std", [0.0] * len(means))
        ax.plot(steps[:len(means)], means, label="mean", color="C3")
        if any(s > 0 for s in stds):
            lo = [m - s for m, s in zip(means, stds)]
            hi = [m + s for m, s in zip(means, stds)]
            ax.fill_between(steps[:len(means)], lo, hi, alpha=0.2, color="C3")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Per-window surprise (NTP CE)")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

    # ── 7. Trajectory diversity ──────────────────────────────────────
    ax = axes[2, 0]
    if steps and "read_unique_frac" in history:
        ax.plot(steps[:len(history["read_unique_frac"])],
                history["read_unique_frac"], label="read")
    if steps and "write_unique_frac" in history:
        ax.plot(steps[:len(history["write_unique_frac"])],
                history["write_unique_frac"], label="write")
    if "read_unique_frac" in history or "write_unique_frac" in history:
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(0, 1)
    ax.set_title("Trajectory diversity (unique concepts visited / N)")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

    # ── 8. Throughput ─────────────────────────────────────────────────
    ax = axes[2, 1]
    if steps and "tok_per_sec" in history and history["tok_per_sec"]:
        ax.plot(steps[:len(history["tok_per_sec"])],
                [t / 1000 for t in history["tok_per_sec"]], color="C4")
    ax.set_title("Throughput (k tok/s)")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

    # ── 9. VRAM peak ─────────────────────────────────────────────────
    ax = axes[2, 2]
    if steps and "vram_peak_gb" in history:
        ax.plot(steps[:len(history["vram_peak_gb"])],
                history["vram_peak_gb"], color="C5")
    ax.set_title("VRAM peak (GB)")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"trajectory-memory training — step {steps[-1] if steps else 0}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def _running_mean(values: list[float], window: int) -> list[float]:
    """Right-aligned running mean — out[i] = mean(values[max(0,i-w+1):i+1])."""
    if not values or window <= 1:
        return list(values)
    out = []
    cum = 0.0
    for i, v in enumerate(values):
        cum += v
        if i >= window:
            cum -= values[i - window]
        out.append(cum / min(i + 1, window))
    return out


def dump_history_json(history: dict[str, Any], output_path: Path) -> None:
    """Persist history to JSON for offline re-plot / analysis. Atomic
    write (write to .tmp, rename) so the file is never half-written
    during a refresh."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2, default=_json_default)
    tmp.replace(output_path)


def _json_default(obj):
    """Coerce common non-JSON-native types (tensors, numpy scalars) to floats."""
    try:
        return float(obj)
    except Exception:
        return str(obj)
