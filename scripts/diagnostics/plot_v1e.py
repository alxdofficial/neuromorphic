#!/usr/bin/env python3
"""Plot v1e HSM training health diagnostics.

Reads outputs/repr_learning/v1e_<variant>/jsonl/<variant>.jsonl and renders:
  - Train loss_hsm (smoothed) for all variants on one axis
  - Val loss_hsm for all variants on one axis
  - Per-variant grad_norm (last 1000 steps)
  - LR schedule
  - Loss components (aux/orth/z) for V2.1 specifically

Usage:
    python scripts/diagnostics/plot_v1e.py
    python scripts/diagnostics/plot_v1e.py --watch 60  # refresh every 60s
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
VARIANTS = [
    "v21", "flat_baseline", "continuous_baseline",
    "memorizing_baseline", "recurrent_baseline", "vanilla_llama",
]
LABELS = {
    "v21": "V2.1",
    "flat_baseline": "A (flat)",
    "continuous_baseline": "B (slot)",
    "memorizing_baseline": "MT",
    "recurrent_baseline": "Mamba",
    "vanilla_llama": "vanilla (floor)",
}
COLORS = {
    "v21": "#d62728",
    "flat_baseline": "#1f77b4",
    "continuous_baseline": "#2ca02c",
    "memorizing_baseline": "#ff7f0e",
    "recurrent_baseline": "#9467bd",
    "vanilla_llama": "#7f7f7f",
}


def load_jsonl(variant: str) -> tuple[list, list]:
    path = REPO / f"outputs/repr_learning/v1e_{variant}/jsonl/{variant}.jsonl"
    if not path.exists():
        return [], []
    train, val = [], []
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("phase") == "val":
                val.append(row)
            else:
                train.append(row)
    return train, val


def smooth(xs: list[float], window: int = 50) -> list[float]:
    if not xs:
        return xs
    arr = np.asarray(xs, dtype=np.float64)
    if len(arr) < window:
        return arr.tolist()
    kernel = np.ones(window) / window
    sm = np.convolve(arr, kernel, mode="same")
    # fix edges
    for i in range(window // 2):
        sm[i] = arr[: i + window // 2 + 1].mean()
        sm[-(i + 1)] = arr[-(i + window // 2 + 1):].mean()
    return sm.tolist()


def render(out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax_train, ax_val, ax_gn = axes[0]
    ax_lr, ax_v21_loss, ax_disp = axes[1]

    # ── Train loss_hsm (smoothed) ──
    ax_train.set_title("Train loss_hsm (smoothed)")
    ax_train.set_xlabel("step")
    ax_train.set_ylabel("MSE")
    for v in VARIANTS:
        train, _ = load_jsonl(v)
        if not train:
            continue
        steps = [r["step"] for r in train]
        losses = [r["loss_hsm"] for r in train]
        sm = smooth(losses, window=50)
        ax_train.plot(steps, sm, label=LABELS[v], color=COLORS[v], linewidth=1.5)
    ax_train.legend(loc="upper right", fontsize=8)
    ax_train.grid(alpha=0.3)
    ax_train.set_yscale("log")

    # ── Val loss_hsm ──
    ax_val.set_title("Val loss_hsm (every 1K steps)")
    ax_val.set_xlabel("step")
    ax_val.set_ylabel("MSE")
    for v in VARIANTS:
        _, val = load_jsonl(v)
        if not val:
            continue
        steps = [r["step"] for r in val]
        losses = [r.get("val_loss_hsm", float("nan")) for r in val]
        ax_val.plot(steps, losses, marker="o", label=LABELS[v], color=COLORS[v],
                     linewidth=1.5, markersize=4)
    ax_val.legend(loc="upper right", fontsize=8)
    ax_val.grid(alpha=0.3)

    # ── Gradient norm ──
    ax_gn.set_title("grad_norm (smoothed)")
    ax_gn.set_xlabel("step")
    ax_gn.set_ylabel("‖∇‖")
    for v in VARIANTS:
        train, _ = load_jsonl(v)
        if not train:
            continue
        steps = [r["step"] for r in train]
        gn = [r.get("grad_norm", 0.0) for r in train]
        sm = smooth(gn, window=50)
        ax_gn.plot(steps, sm, label=LABELS[v], color=COLORS[v], linewidth=1.0, alpha=0.8)
    ax_gn.legend(loc="upper right", fontsize=8)
    ax_gn.grid(alpha=0.3)
    ax_gn.set_yscale("log")

    # ── Learning rate ──
    ax_lr.set_title("learning rate schedule")
    ax_lr.set_xlabel("step")
    ax_lr.set_ylabel("lr")
    for v in VARIANTS:
        train, _ = load_jsonl(v)
        if not train:
            continue
        steps = [r["step"] for r in train]
        lrs = [r.get("lr", 0.0) for r in train]
        ax_lr.plot(steps, lrs, label=LABELS[v], color=COLORS[v], linewidth=1.0)
    ax_lr.legend(loc="upper right", fontsize=8)
    ax_lr.grid(alpha=0.3)

    # ── V2.1 loss components ──
    ax_v21_loss.set_title("V2.1 loss components (smoothed)")
    ax_v21_loss.set_xlabel("step")
    ax_v21_loss.set_ylabel("value")
    train, _ = load_jsonl("v21")
    if train:
        steps = [r["step"] for r in train]
        for field, color, label in [
            ("loss_hsm", "#d62728", "loss_hsm"),
            ("loss_aux", "#1f77b4", "load_balance (raw)"),
            ("loss_orth", "#2ca02c", "codebook_orth"),
            ("loss_z", "#ff7f0e", "z_loss"),
        ]:
            vals = [r.get(field, 0.0) for r in train]
            sm = smooth(vals, window=50)
            ax_v21_loss.plot(steps, sm, label=label, color=color, linewidth=1.2)
        ax_v21_loss.legend(loc="upper right", fontsize=8)
    ax_v21_loss.grid(alpha=0.3)
    ax_v21_loss.set_yscale("symlog")

    # ── Final val comparison (bar chart) ──
    ax_disp.set_title("final val_loss_hsm (lower is better)")
    ax_disp.set_xlabel("variant")
    ax_disp.set_ylabel("MSE")
    names, finals, colors = [], [], []
    for v in VARIANTS:
        _, val = load_jsonl(v)
        if not val:
            continue
        final = val[-1].get("val_loss_hsm", float("nan"))
        names.append(LABELS[v])
        finals.append(final)
        colors.append(COLORS[v])
    if names:
        bars = ax_disp.bar(names, finals, color=colors)
        for b, f in zip(bars, finals):
            ax_disp.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001,
                          f"{f:.3f}", ha="center", fontsize=8)
        ax_disp.tick_params(axis="x", rotation=30)
    ax_disp.grid(alpha=0.3, axis="y")

    fig.suptitle("v1e — HSM cross-chunk training health", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="outputs/repr_learning/v1e_plot.png")
    ap.add_argument("--watch", type=int, default=0, help="Refresh every N sec (0 = once)")
    args = ap.parse_args()

    out_path = REPO / args.out
    if args.watch:
        while True:
            try:
                render(out_path)
            except Exception as e:
                print(f"[error] {e}")
            time.sleep(args.watch)
    else:
        render(out_path)


if __name__ == "__main__":
    main()
