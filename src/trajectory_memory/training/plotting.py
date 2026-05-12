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


def save_training_plots(
    history: dict[str, Any],
    output_path: Path,
    *,
    routing_uniform_baseline: float | None = None,
    vanilla_floor: float | None = None,
) -> None:
    """Render multi-panel training plot to `output_path` (PNG).

    Imports matplotlib lazily so the dependency is optional. If
    matplotlib isn't installed, the call is a no-op (the JSON dump
    still happens via `dump_history_json`).

    Args:
        routing_uniform_baseline: expected `read_unique_frac` under
            uniform-random routing. Drawn as a dashed horizontal line
            on the trajectory-diversity panel. If `r_uf` flatlines at
            this value, routing isn't depending on input (the Gumbel-
            noise bug). Compute as `1 - ((N-1)/N) ** (BS*J*K*D)` for
            the trainer's config.
        vanilla_floor: vanilla Llama bulk-token NTP CE on the same
            data. Drawn as a dashed horizontal line on val_loss. Tells
            us how close memory is to "not hurting."
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend — no $DISPLAY needed
        import matplotlib.pyplot as plt
    except ImportError:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 4×3 grid (12 panels) — added per-source train loss + needle-answer-only
    # val + per-source surprise (Tier 3 follow-up to land actual diagnostics
    # for the memory-bridging behavior we care about).
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    steps = history.get("step", [])

    # ── 1. Train loss — total + component breakdown ─────────────────
    # The 3 components (NTP CE, aux load_balance, aux z_loss) are
    # already coefficient-multiplied, so they sum to total `loss`.
    # Watching the breakdown catches: aux losses dominating CE
    # (over-regularized to uniform routing) or CE diverging while aux
    # stays flat (memory injecting noise that Llama can't suppress).
    ax = axes[0, 0]
    if steps and "loss" in history:
        if len(steps) >= 8:
            window = max(8, len(steps) // 50)
            avg = _running_mean(history["loss"], window)
            ax.plot(steps, avg, label=f"total (avg{window})", color="black",
                    linewidth=1.5)
        else:
            ax.plot(steps, history["loss"], label="total", color="black")
        for key, color, label in [
            ("ntp_ce_loss", "C0", "NTP CE"),
            ("aux_lb_loss", "C1", "load_balance"),
            ("aux_z_loss", "C2", "z_loss"),
        ]:
            vals = history.get(key, [])
            if not vals:
                continue
            if len(steps) >= 8:
                window = max(8, len(steps) // 50)
                vals_smooth = _running_mean(vals, window)
            else:
                vals_smooth = vals
            ax.plot(steps[:len(vals_smooth)], vals_smooth,
                    label=label, color=color, alpha=0.8)
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Train loss — components sum to total")
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
        if vanilla_floor is not None:
            ax.axhline(y=vanilla_floor, color="gray", linestyle="--",
                       alpha=0.6,
                       label=f"vanilla Llama floor ({vanilla_floor:.2f})")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Val loss — distance to vanilla floor = memory cost")
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
    # Healthy specialization: r_uf BELOW the uniform-random baseline
    # (concepts are re-visited because they're meaningful), not at it.
    # Above the baseline = routing more diverse than random (rare).
    # Right at the baseline = the Gumbel-noise bug (routing is random).
    ax = axes[2, 0]
    if steps and "read_unique_frac" in history:
        ax.plot(steps[:len(history["read_unique_frac"])],
                history["read_unique_frac"], label="read")
    if steps and "write_unique_frac" in history:
        ax.plot(steps[:len(history["write_unique_frac"])],
                history["write_unique_frac"], label="write")
    if routing_uniform_baseline is not None:
        ax.axhline(y=routing_uniform_baseline, color="red", linestyle="--",
                   alpha=0.6,
                   label=f"uniform-random ({routing_uniform_baseline:.3f})")
    if "read_unique_frac" in history or "write_unique_frac" in history:
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(0, 1)
    ax.set_title("Trajectory diversity — at red line = random routing")
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

    # ── 9. Architectural-health scalars ──────────────────────────────
    # Three params that must move from init for training to work:
    #   - bridge scale_raw mean (should drift from `scale_init` as the
    #     bridge learns whether/how much to use memory)
    #   - read/write logit_scale (cosine→softmax temperature; if stuck
    #     near exp(init), softmax may still be too flat to specialize)
    # Flat traces here = the Gumbel-noise / aux-loss bugs re-emerging.
    ax = axes[2, 2]
    if steps and "bridge_scale_raw_mean" in history:
        ax.plot(steps[:len(history["bridge_scale_raw_mean"])],
                history["bridge_scale_raw_mean"],
                label="bridge scale_raw mean", color="C0")
    if steps and "read_logit_scale" in history:
        ax.plot(steps[:len(history["read_logit_scale"])],
                history["read_logit_scale"],
                label="read logit_scale (exp)", color="C1")
    if steps and "write_logit_scale" in history:
        ax.plot(steps[:len(history["write_logit_scale"])],
                history["write_logit_scale"],
                label="write logit_scale (exp)", color="C2")
    if (steps and ("bridge_scale_raw_mean" in history
                   or "read_logit_scale" in history)):
        ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Bridge gate + routing logit_scale (must move from init)")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

    # ── 10. Per-source TRAIN loss (the actual memory-bridging diagnostic) ─
    # If memory works, the needle source's per-step loss should drop
    # FASTER than fineweb/wiki/slimpajama (because needle docs require
    # memory retrieval at the answer position; other sources don't).
    # Tracking this in train (not just val) gives signal every step
    # rather than every save.
    ax = axes[3, 0]
    by_source = history.get("loss_by_source", {})
    if by_source:
        for src in sorted(by_source.keys()):
            entries = by_source[src]
            if not entries:
                continue
            xs = [e[0] for e in entries]
            ys = [e[1] for e in entries]
            # Smooth with a running window for readability.
            window = max(8, len(ys) // 50) if len(ys) >= 8 else 1
            smoothed = _running_mean(ys, window)
            label = src
            if "needle" in src.lower():
                label = f"{src} (memory probe)"
            ax.plot(xs, smoothed,
                    label=label,
                    linewidth=2.0 if "needle" in src.lower() else 1.0,
                    alpha=0.9 if "needle" in src.lower() else 0.7)
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Train loss by source (smoothed)")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

    # ── 11. Needle-answer-only val loss ──────────────────────────────
    # Computed in `eval_wave1` from the answer_span tokens only — NOT
    # the diluted full-doc average (which is dominated by 30K filler
    # tokens whose loss barely moves regardless of memory). This is
    # THE metric to track for memory-bridging — should drop sharply
    # if memory is doing its job, regardless of filler loss trends.
    ax = axes[3, 1]
    val_step = history.get("val_step", [])
    answer_only = history.get("val_answer_loss", {})  # {source: [vals]}
    if val_step and answer_only:
        for source, vals in answer_only.items():
            if not vals:
                continue
            ax.plot(val_step[:len(vals)], vals, marker="o", markersize=4,
                    label=source, linewidth=2.0)
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Val ANSWER-only loss (memory probe)")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

    # ── 12. Per-source surprise (per-window CE, by source) ───────────
    # Same idea as panel 10 but for the writer surprise signal: does
    # surprise (per-window mean CE) drop more on needle than fineweb?
    # Surprise ≠ loss in the prior_loss_weight=0 W1 case (here they
    # match) but it's a separate diagnostic that comes from the writer's
    # input, not the optimizer's loss.
    ax = axes[3, 2]
    surprise_by_source = history.get("surprise_by_source", {})
    if surprise_by_source:
        for src in sorted(surprise_by_source.keys()):
            entries = surprise_by_source[src]
            if not entries:
                continue
            xs = [e[0] for e in entries]
            ys = [e[1] for e in entries]
            window = max(8, len(ys) // 50) if len(ys) >= 8 else 1
            smoothed = _running_mean(ys, window)
            label = src
            if "needle" in src.lower():
                label = f"{src} (memory probe)"
            ax.plot(xs, smoothed,
                    label=label,
                    linewidth=2.0 if "needle" in src.lower() else 1.0,
                    alpha=0.9 if "needle" in src.lower() else 0.7)
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Per-window surprise by source (writer's CE input)")
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
