#!/usr/bin/env python3
"""Plot training-curve diagnostics for Wave 1 v4 retrieval pretraining.

Reads the JSONL log emitted by `train_wave1_retrieval.py --log-jsonl` and
produces a set of matplotlib figures. The figures are saved to disk
(no GUI required).

Run:
    python3 scripts/diagnostics/plot_wave1_retrieval.py \
        --log outputs/wave1_v4/train.jsonl \
        --out-dir outputs/wave1_v4/plots/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Vanilla Llama-3.2-1B loss floors on the v8 val split (measured 2026-05-13
# via /tmp/measure_llama_floor.py). Drawn as horizontal reference lines on
# the loss plot. Update if val split changes.
LLAMA_FLOOR_NO_CONTEXT = 2.9194   # A: Llama sees only the QA window
LLAMA_FLOOR_FULL_CONTEXT = 1.0421  # B: Llama sees 8 facts + Q + A in one sequence


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
    out, buf = [], []
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
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(train_steps, train_loss, "-", alpha=0.25, label="train (raw)", color="C0")
    ax.plot(train_steps, smoothed, "-", label="train (50-step EMA)", color="C0")
    if val_rows:
        ax.plot(val_steps, val_loss, "o-", label="val", markersize=4, color="C1")
    # Vanilla Llama floors as horizontal reference lines.
    ax.axhline(
        LLAMA_FLOOR_NO_CONTEXT, color="red", linestyle="--", alpha=0.6,
        label=f"Llama (no context) = {LLAMA_FLOOR_NO_CONTEXT:.2f}",
    )
    ax.axhline(
        LLAMA_FLOOR_FULL_CONTEXT, color="green", linestyle="--", alpha=0.6,
        label=f"Llama (8 facts in ctx) = {LLAMA_FLOOR_FULL_CONTEXT:.2f}",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("answer-token CE loss")
    ax.set_title("Wave 1 v4 retrieval — loss (lower=better; below red = memory works; near green = matches long-ctx)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_accuracy(train_rows, val_rows, out_path):
    train_steps = [r["step"] for r in train_rows if "answer_acc" in r]
    train_acc = [r["answer_acc"] for r in train_rows if "answer_acc" in r]
    val_steps = [r["step"] for r in val_rows if "answer_acc" in r]
    val_acc = [r["answer_acc"] for r in val_rows if "answer_acc" in r]
    if not train_acc:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_steps, train_acc, "-", alpha=0.25, color="C0", label="train (raw)")
    ax.plot(train_steps, _running_mean(train_acc, 50), "-", color="C0", label="train (EMA)")
    if val_acc:
        ax.plot(val_steps, val_acc, "o-", markersize=4, color="C1", label="val")
    ax.set_xlabel("step")
    ax.set_ylabel("answer-token argmax accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Answer-token argmax accuracy (fraction predicted correctly)")
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
    rgn = [r.get("r_gn", 0.0) for r in train_rows]
    mign = [r.get("mi_gn", 0.0) for r in train_rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, _running_mean(grad, 30), label="total grad_norm", color="black")
    ax.plot(steps, _running_mean(wgn, 30), label="write_module", color="C0")
    ax.plot(steps, _running_mean(rgn, 30), label="read_module", color="C1")
    ax.plot(steps, _running_mean(mign, 30), label="mem_inject", color="C2")
    ax.set_xlabel("step")
    ax.set_ylabel("gradient norm")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    ax.set_title("Per-module gradient norms (log scale) — all should be non-zero")
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


def plot_overlap(train_rows, val_rows, out_path):
    """Read trajectory vs target's write trajectory — concept overlap.
    Random baseline ≈ 0.01. > 0.1 means read is preferentially finding target."""
    train_steps = [r["step"] for r in train_rows if "read_target_overlap" in r]
    train_o = [r["read_target_overlap"] for r in train_rows if "read_target_overlap" in r]
    val_steps = [r["step"] for r in val_rows if "read_target_overlap" in r]
    val_o = [r["read_target_overlap"] for r in val_rows if "read_target_overlap" in r]
    if not train_o:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_steps, train_o, "-", alpha=0.25, color="C0", label="train (raw)")
    ax.plot(train_steps, _running_mean(train_o, 50), "-", color="C0", label="train (EMA)")
    if val_o:
        ax.plot(val_steps, val_o, "o-", markersize=4, color="C1", label="val")
    # Random baseline.
    ax.axhline(0.01, color="gray", linestyle="--", alpha=0.5, label="random baseline (≈0.01)")
    ax.axhline(0.1, color="green", linestyle=":", alpha=0.5, label="meaningful threshold (0.1)")
    ax.set_xlabel("step")
    ax.set_ylabel("fraction of read concepts in target's write set")
    ax.set_title("Read↔target concept overlap (mechanistic: is the read finding the target's memory?)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_state(train_rows, out_path):
    """Manifold state norm — catches state explosion/collapse."""
    rows = [r for r in train_rows if "state_norm_mean" in r]
    if not rows:
        return
    steps = [r["step"] for r in rows]
    mean = [r["state_norm_mean"] for r in rows]
    std = [r["state_norm_std"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, mean, label="state norm (mean over concepts)", color="C3")
    ax.fill_between(
        steps, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],
        alpha=0.2, color="C3", label="±1 σ (across concepts)",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("||concept_state||₂")
    ax.set_title("Manifold concept-state norm (catches collapse / explosion)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_class_val(val_rows, out_path):
    """Bar chart of per-(class, attr) val loss + accuracy at last val cycle."""
    rows_with_pk = [r for r in val_rows if r.get("per_key_loss")]
    if not rows_with_pk:
        return
    last = rows_with_pk[-1]
    keys = sorted(last["per_key_loss"].keys())
    losses = [last["per_key_loss"][k] for k in keys]
    accs = [last["per_key_acc"].get(k, 0.0) for k in keys]
    ns = [last["per_key_n"].get(k, 0) for k in keys]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    x = list(range(len(keys)))
    bars1 = ax1.bar(x, losses, color="C1")
    ax1.axhline(LLAMA_FLOOR_NO_CONTEXT, color="red", linestyle="--", alpha=0.6,
                label=f"Llama no-ctx = {LLAMA_FLOOR_NO_CONTEXT:.2f}")
    ax1.axhline(LLAMA_FLOOR_FULL_CONTEXT, color="green", linestyle="--", alpha=0.6,
                label=f"Llama full-ctx = {LLAMA_FLOOR_FULL_CONTEXT:.2f}")
    ax1.set_ylabel("val loss")
    ax1.set_title(f"Per-(class, attr) val metrics at step {last['step']}")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3, axis="y")
    bars2 = ax2.bar(x, accs, color="C2")
    ax2.set_ylabel("val accuracy")
    ax2.set_ylim(0, 1)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{k}\nn={n}" for k, n in zip(keys, ns)], rotation=45, ha="right", fontsize=8)
    ax2.grid(alpha=0.3, axis="y")
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
    plot_accuracy(train_rows, val_rows, out / "accuracy.png")
    plot_routing(train_rows, out / "routing.png")
    plot_grads(train_rows, out / "gradients.png")
    plot_scales(train_rows, out / "scales.png")
    plot_overlap(train_rows, val_rows, out / "read_target_overlap.png")
    plot_state(train_rows, out / "state_norm.png")
    plot_per_class_val(val_rows, out / "per_class_val.png")
    print(f"Wrote 8 plots to {out}/")


if __name__ == "__main__":
    main()
