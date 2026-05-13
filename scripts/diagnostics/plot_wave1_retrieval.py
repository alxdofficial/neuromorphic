#!/usr/bin/env python3
"""Plot training-curve diagnostics for Wave 1 v4 retrieval pretraining.

Reads the JSONL log emitted by `train_wave1_retrieval.py --log-jsonl` and
produces a small set of matplotlib figures. The figures are saved to disk
(no GUI required).

Run:
    python3 scripts/diagnostics/plot_wave1_retrieval.py \
        --log outputs/wave1_retrieval/train.jsonl \
        --out-dir outputs/wave1_retrieval/plots/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_log(path: str) -> tuple[list[dict], list[dict]]:
    train_rows, val_rows = [], []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("phase") == "val":
                val_rows.append(r)
            else:
                train_rows.append(r)
    return train_rows, val_rows


def _running_mean(xs: list[float], window: int) -> list[float]:
    out, run = [], 0.0
    buf: list[float] = []
    for x in xs:
        buf.append(x)
        if len(buf) > window:
            buf.pop(0)
        out.append(sum(buf) / len(buf))
    return out


def plot_loss(train_rows, val_rows, out_path):
    train_steps = [r["step"] for r in train_rows]
    train_loss = [r["loss"] for r in train_rows]
    smoothed = _running_mean(train_loss, 50)
    val_steps = [r["step"] for r in val_rows]
    val_loss = [r["loss"] for r in val_rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_steps, train_loss, "-", alpha=0.25, label="train (raw)")
    ax.plot(train_steps, smoothed, "-", label="train (50-step EMA)")
    if val_rows:
        ax.plot(val_steps, val_loss, "o-", label="val", markersize=4)
    ax.set_xlabel("step")
    ax.set_ylabel("answer-token CE loss")
    ax.set_title("Wave 1 v4 retrieval — loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_routing(train_rows, out_path):
    steps = [r["step"] for r in train_rows]
    r_uf = [r["r_uf"] for r in train_rows]
    w_uf = [r["w_uf"] for r in train_rows]
    r_ent = [r["r_ent"] for r in train_rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(steps, _running_mean(r_uf, 30), label="r_uf (read)")
    ax1.plot(steps, _running_mean(w_uf, 30), label="w_uf (write)")
    ax1.set_ylabel("routing uniformity\n(0=uniform, 1=peaked)")
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax1.set_title("Routing uniformity")
    ax2.plot(steps, _running_mean(r_ent, 30), color="purple", label="r_ent")
    ax2.set_ylabel("read entropy (nats)")
    ax2.set_xlabel("step")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_grads(train_rows, out_path):
    steps = [r["step"] for r in train_rows]
    grad = [r["grad_norm"] for r in train_rows]
    wgn = [r["w_gn"] for r in train_rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, _running_mean(grad, 30), label="total grad_norm")
    ax.plot(steps, _running_mean(wgn, 30), label="write_module grad_norm")
    ax.set_xlabel("step")
    ax.set_ylabel("gradient norm")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    ax.set_title("Gradient norms (log scale)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_scales(train_rows, out_path):
    steps = [r["step"] for r in train_rows]
    mi = [r["mem_inject_scale"] for r in train_rows]
    r_logit = [r["read_logit_scale"] for r in train_rows]
    w_logit = [r["write_logit_scale"] for r in train_rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(steps, mi, color="C2")
    ax1.set_ylabel("mem_inject scale\n(effective)")
    ax1.set_title("Learnable scales")
    ax1.grid(alpha=0.3)
    ax2.plot(steps, r_logit, label="read logit_scale")
    ax2.plot(steps, w_logit, label="write logit_scale")
    ax2.set_xlabel("step")
    ax2.set_ylabel("logit_scale (exp)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows = load_log(args.log)
    print(f"Loaded {len(train_rows)} train + {len(val_rows)} val records")
    if not train_rows:
        print("No train records — nothing to plot.")
        return
    plot_loss(train_rows, val_rows, out / "loss.png")
    plot_routing(train_rows, out / "routing.png")
    plot_grads(train_rows, out / "gradients.png")
    plot_scales(train_rows, out / "scales.png")
    print(f"Wrote 4 plots to {out}/")


if __name__ == "__main__":
    main()
