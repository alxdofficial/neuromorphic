#!/usr/bin/env python3
"""Plot v2 training metrics from a JSONL log.

Auto-detects wave (1 / 2 / 3) from the metric keys present. Produces
one multi-panel figure per wave covering loss components, routing
health, R∩W overlap (Wave 1), edge stats, bank health, and per-module
gradient norms. Wave 3 swaps in reward distribution + advantage stats.

Usage:
    # One-shot:
    python scripts/diagnostics/plot_v2.py \\
        --jsonl outputs/wave1_v2/train.jsonl \\
        --out outputs/wave1_v2/plot.png

    # Live (re-render every 180s; ideal for tail-mode monitoring):
    python scripts/diagnostics/plot_v2.py \\
        --jsonl outputs/wave1_v2/train.jsonl \\
        --out outputs/wave1_v2/plot.png \\
        --watch 180

The training scripts can also fire this in a background subprocess on
a wall-clock cadence to act as a live dashboard. See `--plot-every-secs`
on train_wave[123]_v2.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> tuple[list[dict], list[dict]]:
    """Return (train_rows, val_rows). val_rows is rows where phase == 'val'."""
    train, val = [], []
    if not path.exists() or path.stat().st_size == 0:
        return train, val
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("phase") == "val":
            val.append(row)
        else:
            train.append(row)
    return train, val


def detect_wave(rows: list[dict]) -> str:
    if not rows:
        return "unknown"
    keys = set(rows[-1].keys()) | set(rows[0].keys())
    if "mean_reward" in keys or "pg_loss" in keys:
        return "wave3"
    if "rw_overlap_target" in keys or "rw_overlap" in keys:
        return "wave1"
    if "l_contrast_per_step" in keys or "answer_loss" in keys:
        return "wave2"
    return "unknown"


def _series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    """Extract (steps, values) for a key, skipping rows that don't have it."""
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] is not None and isinstance(r[key], (int, float)):
            xs.append(r["step"])
            ys.append(float(r[key]))
    return xs, ys


def _ema(values: list[float], alpha: float = 0.1) -> list[float]:
    """Lightweight EMA for smoothing noisy training curves."""
    if not values:
        return values
    out = [values[0]]
    for v in values[1:]:
        out.append((1 - alpha) * out[-1] + alpha * v)
    return out


def _plot_lines(ax, rows, keys: list[tuple[str, str]], *, log_y: bool = False, smooth: bool = True):
    """Plot multiple metric series on one axis. keys = list of (jsonl_key, label)."""
    for key, label in keys:
        xs, ys = _series(rows, key)
        if not xs:
            continue
        if smooth and len(ys) > 5:
            ax.plot(xs, ys, alpha=0.25, linewidth=0.8)
            ax.plot(xs, _ema(ys), label=label, linewidth=1.4)
        else:
            ax.plot(xs, ys, label=label, marker=".", markersize=3)
    if log_y:
        ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)


def plot_wave1(train: list[dict], val: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    fig.suptitle(f"Wave 1 v2 (retrieval) — {len(train)} train, {len(val)} val rows",
                 fontsize=12)

    # Row 0: losses + accuracy
    _plot_lines(axes[0, 0], train, [
        ("loss", "total"), ("answer_loss", "answer (CE)"),
        ("l_contrast_entry", "contrast_entry"),
        ("l_contrast_per_step", "contrast_per_step"),
    ])
    axes[0, 0].set_title("loss components (train)")

    _plot_lines(axes[0, 1], train, [("answer_acc", "train acc")])
    if val:
        _plot_lines(axes[0, 1], val, [("answer_acc", "val acc")], smooth=False)
    axes[0, 1].set_title("answer accuracy")
    axes[0, 1].set_ylim(0, 1)

    _plot_lines(axes[0, 2], train, [
        ("aux_lb", "load_balance"), ("aux_z", "z_loss"),
    ], log_y=True)
    axes[0, 2].set_title("aux losses (raw)")

    # Row 1: R∩W overlap + routing entropy
    _plot_lines(axes[1, 0], train, [
        ("rw_overlap_target", "vs target"),
        ("rw_overlap_all", "vs all writes"),
        ("rw_overlap_entry", "entry only"),
        ("rw_overlap_hop", "hops only"),
    ])
    axes[1, 0].set_title("R∩W overlap (train)")
    axes[1, 0].set_ylim(0, 1)

    _plot_lines(axes[1, 1], train, [
        ("read_entry_entropy", "read entry H"),
        ("write_entry_entropy", "write entry H"),
    ])
    axes[1, 1].set_title("entry routing entropy")

    _plot_lines(axes[1, 2], train, [
        ("r_unique_per_window", "read unique/win"),
        ("w_unique_per_window", "write unique/win"),
        ("r_unique_per_traj", "read unique/traj"),
        ("w_unique_per_traj", "write unique/traj"),
    ])
    axes[1, 2].set_title("unique cells touched")

    # Row 2: per-module grad norms + total grad
    _plot_lines(axes[2, 0], train, [
        ("grad_norm", "total"),
        ("grad_norm_read", "read"),
        ("grad_norm_write", "write"),
        ("grad_norm_entry_proj", "entry_proj"),
    ], log_y=True)
    axes[2, 0].set_title("grad norms (log)")

    _plot_lines(axes[2, 1], train, [
        ("grad_norm_concept_ids", "concept_ids"),
        ("grad_norm_mem_inject", "mem_inject"),
        ("grad_norm_read_attn", "read_attn"),
        ("grad_norm_lambda_edge", "lambda_edge"),
    ], log_y=True)
    axes[2, 1].set_title("grad norms — adapter (log)")

    _plot_lines(axes[2, 2], train, [
        ("concept_ids_norm_mean", "norm mean"),
        ("concept_ids_norm_cv", "norm CV"),
        ("concept_ids_pairwise_cos", "pairwise cos"),
    ])
    axes[2, 2].set_title("vocab bank health")

    # Row 3: edge stats
    _plot_lines(axes[3, 0], train, [
        ("n_active_edges", "active edges"),
        ("mean_fan_out", "mean fan_out"),
    ])
    axes[3, 0].set_title("edge graph size")

    _plot_lines(axes[3, 1], train, [
        ("mean_edge_state_norm", "state norm"),
        ("mean_edge_specificity", "spec"),
    ])
    axes[3, 1].set_title("edge state magnitudes")

    _plot_lines(axes[3, 2], train, [
        ("mean_visit_count", "visit_count"),
        ("mean_edge_age", "age"),
    ])
    axes[3, 2].set_title("edge lifecycle")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_wave2(train: list[dict], val: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.suptitle(f"Wave 2 v2 (streaming SFT) — {len(train)} train, {len(val)} val rows",
                 fontsize=12)

    _plot_lines(axes[0, 0], train, [
        ("loss", "total"), ("answer_loss", "answer (CE)"),
        ("l_contrast_per_step", "contrast_per_step"),
    ])
    axes[0, 0].set_title("loss components")

    _plot_lines(axes[0, 1], train, [("answer_loss", "train")])
    if val:
        _plot_lines(axes[0, 1], val, [("answer_loss", "val")], smooth=False)
    axes[0, 1].set_title("answer CE (train vs val)")

    _plot_lines(axes[0, 2], train, [
        ("aux_lb", "load_balance"), ("aux_z", "z_loss"),
    ], log_y=True)
    axes[0, 2].set_title("aux losses (raw)")

    _plot_lines(axes[1, 0], train, [
        ("grad_norm", "total"),
    ], log_y=True)
    axes[1, 0].set_title("gradient norm")

    _plot_lines(axes[1, 1], train, [
        ("n_active_edges", "active edges"),
        ("mean_fan_out", "mean fan_out"),
    ])
    axes[1, 1].set_title("edge graph size")

    _plot_lines(axes[1, 2], train, [
        ("mean_edge_state_norm", "state norm"),
        ("mean_edge_specificity", "specificity"),
    ])
    axes[1, 2].set_title("edge state")

    _plot_lines(axes[2, 0], train, [
        ("mean_visit_count", "visit_count"),
        ("mean_edge_age", "age"),
    ])
    axes[2, 0].set_title("edge lifecycle")

    _plot_lines(axes[2, 1], train, [
        ("step_s", "step (s)"),
    ])
    axes[2, 1].set_title("step time")

    axes[2, 2].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_wave3(train: list[dict], val: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.suptitle(f"Wave 3 v2 (GRPO) — {len(train)} train rows", fontsize=12)

    _plot_lines(axes[0, 0], train, [
        ("loss", "total"), ("pg_loss", "policy grad"), ("kl_loss", "KL"),
    ])
    axes[0, 0].set_title("loss components")

    _plot_lines(axes[0, 1], train, [
        ("mean_reward", "mean R"),
        ("max_reward", "max R"),
        ("min_reward", "min R"),
    ])
    axes[0, 1].set_title("reward stats")
    axes[0, 1].set_ylim(0, 1)

    _plot_lines(axes[0, 2], train, [
        ("reward_std", "reward std"),
    ])
    axes[0, 2].set_title("rollout variance (learning signal)")

    _plot_lines(axes[1, 0], train, [
        ("grad_norm", "total"),
    ], log_y=True)
    axes[1, 0].set_title("gradient norm")

    _plot_lines(axes[1, 1], train, [
        ("mean_response_len", "tokens"),
    ])
    axes[1, 1].set_title("mean response length")

    _plot_lines(axes[1, 2], train, [
        ("n_active_edges", "active edges"),
    ])
    axes[1, 2].set_title("edge graph size")

    _plot_lines(axes[2, 0], train, [
        ("mean_effectiveness", "mean eff"),
        ("mean_visit_count", "visit_count"),
    ])
    axes[2, 0].set_title("edge health (W-TinyLFU)")

    _plot_lines(axes[2, 1], train, [
        ("step_s", "step (s)"),
    ])
    axes[2, 1].set_title("step time")

    # Per-source rolling avg reward
    source_series: dict[str, list[tuple[int, float]]] = {}
    for r in train:
        ps = r.get("per_source") or {}
        for src, val_r in ps.items():
            source_series.setdefault(src, []).append((r["step"], float(val_r)))
    if source_series:
        for src, pts in source_series.items():
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            axes[2, 2].plot(xs, _ema(ys), label=src, linewidth=1.4)
        axes[2, 2].legend(loc="best", fontsize=8)
        axes[2, 2].set_xlabel("step")
        axes[2, 2].grid(True, alpha=0.3)
        axes[2, 2].set_ylim(0, 1)
        axes[2, 2].set_title("per-source mean reward")
    else:
        axes[2, 2].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def render(jsonl: Path, out_path: Path) -> str:
    """Render the plot once; return the detected wave string."""
    train, val = load_jsonl(jsonl)
    if not train:
        return "empty"
    wave = detect_wave(train)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if wave == "wave1":
        plot_wave1(train, val, out_path)
    elif wave == "wave2":
        plot_wave2(train, val, out_path)
    elif wave == "wave3":
        plot_wave3(train, val, out_path)
    else:
        return f"unknown (keys: {sorted(train[-1].keys())[:5]}...)"
    return wave


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="output PNG path")
    ap.add_argument("--watch", type=float, default=0.0,
                    help="re-render every N seconds (0 = one-shot)")
    args = ap.parse_args()

    if args.watch > 0:
        print(f"[plot_v2] watching {args.jsonl} → {args.out} every {args.watch}s",
              flush=True)
        while True:
            try:
                wave = render(args.jsonl, args.out)
                print(f"[plot_v2] rendered {wave} @ {time.strftime('%H:%M:%S')}",
                      flush=True)
            except Exception as e:
                print(f"[plot_v2] render failed: {e}", file=sys.stderr, flush=True)
            time.sleep(args.watch)
    else:
        wave = render(args.jsonl, args.out)
        print(f"rendered {wave} → {args.out}")


if __name__ == "__main__":
    main()
