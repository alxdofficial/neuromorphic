#!/usr/bin/env python3
"""Multi-panel diagnostic plot for repr_learning training.

Reads JSONL logs written by scripts/repr_learning/train_smoke.py (one
file per variant, each row is a ReprMetrics asdict()). Renders a 3x3
figure: loss / dispersion / routing / codebook health / grad norms /
modifier delta / throughput.

Usage:
    python scripts/diagnostics/plot_repr.py \\
        --jsonl-dir outputs/repr_learning/jsonl \\
        --out outputs/repr_learning/plot.png

    # Live mode (re-renders every 60s for tail-mode monitoring):
    python scripts/diagnostics/plot_repr.py \\
        --jsonl-dir outputs/repr_learning/jsonl \\
        --out outputs/repr_learning/plot.png \\
        --watch 60
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


VARIANT_COLOR = {
    "v21":                  "#1f77b4",   # blue
    "flat_baseline":        "#ff7f0e",   # orange
    "continuous_baseline":  "#2ca02c",   # green
    "memorizing_baseline":  "#d62728",   # red
    "recurrent_baseline":   "#9467bd",   # purple
}
VARIANT_LABEL = {
    "v21":                  "V2.1",
    "flat_baseline":        "A: flat",
    "continuous_baseline":  "B: cont",
    "memorizing_baseline":  "MT",
    "recurrent_baseline":   "Mamba",
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _ema(values: list[float], alpha: float = 0.2) -> list[float]:
    if not values:
        return values
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def _series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if v is None or not isinstance(v, (int, float)):
            continue
        xs.append(r["step"])
        ys.append(float(v))
    return xs, ys


def _plot_metric(ax, all_rows: dict, key: str, *, log_y: bool = False,
                 ema: bool = True, ylabel: str = "", title: str = ""):
    for variant, rows in all_rows.items():
        xs, ys = _series(rows, key)
        if not xs:
            continue
        color = VARIANT_COLOR.get(variant, "gray")
        label = VARIANT_LABEL.get(variant, variant)
        if ema and len(ys) > 5:
            ax.plot(xs, ys, color=color, alpha=0.2, linewidth=0.8)
            ax.plot(xs, _ema(ys), color=color, linewidth=1.4, label=label)
        else:
            ax.plot(xs, ys, color=color, linewidth=1.4, marker=".",
                    markersize=3, label=label)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("step")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7)


def render(jsonl_dir: Path, out_path: Path):
    variants = sorted(jsonl_dir.glob("*.jsonl"))
    if not variants:
        print(f"No JSONL files in {jsonl_dir}", file=sys.stderr)
        return

    all_rows = {p.stem: load_jsonl(p) for p in variants}
    all_rows = {k: v for k, v in all_rows.items() if v}
    if not all_rows:
        print(f"All JSONL files empty in {jsonl_dir}", file=sys.stderr)
        return

    n_steps = max(len(v) for v in all_rows.values())
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(
        f"repr_learning — {len(all_rows)} variants, "
        f"up to {n_steps} steps",
        fontsize=12,
    )

    # Row 0: loss + dispersion + grad norm
    _plot_metric(axes[0, 0], all_rows, "loss_recon",
                 title="reconstruction CE (train)",
                 ylabel="nats")
    _plot_metric(axes[0, 1], all_rows, "mem_dispersion",
                 title="memory token dispersion (lower = better)",
                 ylabel="mean off-diag cos")
    axes[0, 1].set_ylim(0, 1.05)
    _plot_metric(axes[0, 2], all_rows, "grad_norm", log_y=True,
                 title="total grad norm (log)")

    # Row 1: routing health (V2.1 + A only) + codebook health
    _plot_metric(axes[1, 0], all_rows, "routing_entropy",
                 title="codebook routing entropy",
                 ylabel="nats (max ≈ log(4096) ≈ 8.3)")
    _plot_metric(axes[1, 1], all_rows, "unique_codes_per_batch",
                 title="distinct codes picked per batch",
                 ylabel="count")
    _plot_metric(axes[1, 2], all_rows, "codebook_pairwise_cos",
                 title="codebook pairwise cosine",
                 ylabel="mean off-diag cos")

    # Row 2: per-module grads, V2.1 specifics, throughput
    # Per-module grads on one shared subplot
    ax = axes[2, 0]
    for variant, rows in all_rows.items():
        color = VARIANT_COLOR.get(variant, "gray")
        for key, ls in [
            ("grad_norm_encoder", "-"),
            ("grad_norm_codebook", "--"),
            ("grad_norm_modifier", ":"),
            ("grad_norm_proj", "-."),
        ]:
            xs, ys = _series(rows, key)
            if xs and any(y > 0 for y in ys):
                ax.plot(xs, _ema(ys), color=color, linestyle=ls,
                        linewidth=1.0,
                        label=f"{VARIANT_LABEL.get(variant, variant)} {key.split('_')[-1]}",
                        alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("grad norm")
    ax.set_title("per-module grad norms (log)", fontsize=10)
    ax.legend(loc="best", fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    _plot_metric(axes[2, 1], all_rows, "modifier_delta_norm",
                 title="V2.1 modifier delta norm (= 0 at init)",
                 ylabel="mean ||delta||")
    _plot_metric(axes[2, 2], all_rows, "text_tok_per_sec",
                 title="throughput",
                 ylabel="text tok / s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--watch", type=float, default=0,
                    help="If >0, re-render every N seconds")
    args = ap.parse_args()

    if args.watch > 0:
        while True:
            try:
                render(args.jsonl_dir, args.out)
            except Exception as e:
                print(f"[plot_repr] error: {e}", file=sys.stderr)
            time.sleep(args.watch)
    else:
        render(args.jsonl_dir, args.out)


if __name__ == "__main__":
    main()
