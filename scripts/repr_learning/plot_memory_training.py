#!/usr/bin/env python3
"""Plot memory-training health from repr_learning JSONL logs.

This is the pre-smoke dashboard for the active memory-experiment suite. It reads
the trainer's per-variant JSONL files and renders:

  - train loss vs validation loss
  - train/validation top-1
  - memory-vs-LoRA gradient norms
  - REAL/SHUF/OFF validation binding gaps
  - graph_v8 collapse/state/coactivation telemetry when present

Examples:
    .venv/bin/python scripts/repr_learning/plot_memory_training.py \
        --jsonl outputs/repr_learning/run_graph_v8/jsonl/graph_v8_baseline.jsonl \
        --out outputs/repr_learning/run_graph_v8/plots/training_health.png

    .venv/bin/python scripts/repr_learning/plot_memory_training.py \
        --jsonl-dir outputs/repr_learning/run_graph_v8/jsonl \
        --out outputs/repr_learning/run_graph_v8/plots/training_health.png \
        --watch 60
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "graph_v8_baseline": "#bf1f2f",
    "icae_baseline": "#1f5aa6",
    "ccm_baseline": "#2f7d32",
    "autocompressor_baseline": "#6f4ab3",
    "beacon_baseline": "#d77b1f",
}

LABELS = {
    "graph_v8_baseline": "Graph V8",
    "icae_baseline": "ICAE",
    "ccm_baseline": "CCM",
    "autocompressor_baseline": "AutoCompressor",
    "beacon_baseline": "Beacon",
}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists() or path.stat().st_size == 0:
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def discover_jsonl(args) -> list[Path]:
    paths: list[Path] = []
    if args.jsonl:
        paths.extend(args.jsonl)
    if args.jsonl_dir:
        paths.extend(sorted(args.jsonl_dir.glob("*.jsonl")))
    # De-duplicate while preserving order.
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        p = p.resolve()
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    train, val = [], []
    for r in rows:
        if r.get("phase") == "val":
            val.append(r)
        else:
            train.append(r)
    return train, val


def series(rows: Iterable[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            xs.append(int(r.get("step", len(xs))))
            ys.append(float(v))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def smooth(y: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or y.size < max(3, width):
        return y
    width = min(width, y.size)
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(y, kernel, mode="same")


def label_for(name: str) -> str:
    return LABELS.get(name, name)


def color_for(name: str) -> str:
    return COLORS.get(name, "#555555")


def plot_train_val(ax, datasets, train_key, val_key, title, ylabel, smooth_width):
    had = False
    for name, (train, val) in datasets.items():
        color = color_for(name)
        xs, ys = series(train, train_key)
        if ys.size:
            had = True
            if ys.size > 4:
                ax.plot(xs, ys, color=color, alpha=0.18, linewidth=0.8)
                ax.plot(xs, smooth(ys, smooth_width), color=color, linewidth=1.6,
                        label=f"{label_for(name)} train")
            else:
                ax.plot(xs, ys, color=color, marker=".", linewidth=1.3,
                        label=f"{label_for(name)} train")
        xv, yv = series(val, val_key)
        if yv.size:
            had = True
            ax.plot(xv, yv, color=color, marker="o", linestyle="--",
                    linewidth=1.2, markersize=4, label=f"{label_for(name)} val")
    style_axis(ax, title, ylabel)
    if had:
        ax.legend(loc="best", fontsize=7)


def plot_metric(ax, datasets, keys, title, ylabel, *, log_y=False, smooth_width=15):
    had = False
    for name, (train, _val) in datasets.items():
        color = color_for(name)
        for key, linestyle, suffix in keys:
            xs, ys = series(train, key)
            if not ys.size:
                continue
            had = True
            yplot = smooth(ys, smooth_width) if ys.size > 4 else ys
            marker = "." if ys.size < 3 else None
            ax.plot(xs, yplot, color=color, linestyle=linestyle, linewidth=1.3,
                    marker=marker, markersize=5,
                    label=f"{label_for(name)} {suffix}")
    if log_y and had:
        ax.set_yscale("log")
    style_axis(ax, title, ylabel)
    if had:
        ax.legend(loc="best", fontsize=7)
    else:
        ax.text(0.5, 0.5, "no matching metrics yet", ha="center", va="center",
                transform=ax.transAxes, color="#777777")


def plot_val_controls(ax, datasets):
    had = False
    for name, (_train, val) in datasets.items():
        color = color_for(name)
        xr, yr = series(val, "val_loss_recon")
        xo, yo = series(val, "val_loss_recon_off")
        xs, ys = series(val, "val_loss_recon_shuf")
        if yr.size:
            had = True
            ax.plot(xr, yr, color=color, marker="o", linewidth=1.3,
                    label=f"{label_for(name)} REAL")
        if yo.size:
            had = True
            ax.plot(xo, yo, color=color, linestyle=":", marker="^", linewidth=1.2,
                    label=f"{label_for(name)} OFF")
        if ys.size:
            had = True
            ax.plot(xs, ys, color=color, linestyle="--", marker="s", linewidth=1.2,
                    label=f"{label_for(name)} SHUF")
    style_axis(ax, "validation binding controls", "CE loss")
    if had:
        ax.legend(loc="best", fontsize=7)


def plot_v8_layer_metric(ax, datasets, stem, title, ylabel, *, sources=False):
    had = False
    layers = set()
    for train, val in datasets.values():
        for row in [*train, *val]:
            for key in row:
                if not key.startswith(stem):
                    continue
                try:
                    layers.add(int(key[len(stem):]))
                except ValueError:
                    pass
    for name, (train, _val) in datasets.items():
        if name != "graph_v8_baseline":
            continue
        color = color_for(name)
        for i, layer in enumerate(sorted(layers)):
            key = f"{stem}{layer}"
            xs, ys = series(train, key)
            if not ys.size:
                continue
            had = True
            marker = "." if ys.size < 3 else None
            ax.plot(xs, smooth(ys, 15), color=color, alpha=0.95 - 0.18 * i,
                    linewidth=1.4, marker=marker, markersize=5,
                    label=f"L{layer}")
    style_axis(ax, title, ylabel)
    if had:
        ax.legend(loc="best", fontsize=7)
    else:
        ax.text(0.5, 0.5, "no graph_v8 telemetry yet", ha="center", va="center",
                transform=ax.transAxes, color="#777777")


def style_axis(ax, title: str, ylabel: str):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("step")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render(paths: list[Path], out: Path, smooth_width: int):
    datasets = {}
    for p in paths:
        rows = load_jsonl(p)
        if not rows:
            print(f"[warn] no rows in {p}", file=sys.stderr)
            continue
        variant = p.stem
        datasets[variant] = split_rows(rows)
    if not datasets:
        raise SystemExit("no JSONL rows found")

    fig, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)
    fig.suptitle("memory training health", fontsize=13, fontweight="bold")

    plot_train_val(axes[0, 0], datasets, "loss_recon", "val_loss_recon",
                   "train vs validation loss", "CE loss", smooth_width)
    plot_train_val(axes[0, 1], datasets, "top1_acc", "val_top1_acc",
                   "train vs validation top-1", "accuracy", smooth_width)
    plot_metric(
        axes[0, 2],
        datasets,
        [("grad_norm_memory", "-", "memory"), ("grad_norm_lora", "--", "LoRA")],
        "gradient norms by parameter group",
        "norm",
        log_y=True,
        smooth_width=smooth_width,
    )

    plot_val_controls(axes[1, 0], datasets)
    plot_v8_layer_metric(axes[1, 1], datasets, "graph_v8_key_collapse_cos_L",
                         "Graph V8 key collapse", "mean |off-diag cosine|")
    plot_v8_layer_metric(axes[1, 2], datasets, "graph_v8_value_collapse_cos_L",
                         "Graph V8 value collapse", "mean |off-diag cosine|")

    plot_v8_layer_metric(axes[2, 0], datasets, "graph_v8_state_effect_L",
                         "Graph V8 state effect", "RMS delta from base")
    plot_v8_layer_metric(axes[2, 1], datasets, "graph_v8_coact_eff_edge_frac_L",
                         "Graph V8 coactivation spread", "effective edge fraction",
                         sources=True)
    plot_metric(
        axes[2, 2],
        datasets,
        [("graph_v8_reader_gate_mean", "-", "gate mean"),
         ("graph_v8_reader_gate_abs_max", "--", "gate abs max")],
        "Graph V8 reader gates",
        "gate value",
        smooth_width=smooth_width,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="*", type=Path,
                    help="One or more trainer JSONL files.")
    ap.add_argument("--jsonl-dir", type=Path,
                    help="Directory containing per-variant JSONL files.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--smooth", type=int, default=15)
    ap.add_argument("--watch", type=float, default=0,
                    help="If >0, re-render every N seconds.")
    args = ap.parse_args()

    paths = discover_jsonl(args)
    if not paths:
        raise SystemExit("pass --jsonl or --jsonl-dir")

    if args.watch > 0:
        while True:
            try:
                render(paths, args.out, args.smooth)
            except Exception as exc:
                print(f"[plot] error: {exc}", file=sys.stderr)
            time.sleep(args.watch)
    else:
        render(paths, args.out, args.smooth)


if __name__ == "__main__":
    main()
