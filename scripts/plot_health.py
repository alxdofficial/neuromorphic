"""Unified training health dashboard across bootstrap + all cycles.

Stitches phase 1 (`metrics.jsonl`) and phase 2 (`phase2_metrics.jsonl`) logs
from `<work_dir>/metrics.jsonl` and `<work_dir>/cycle_NN/*.jsonl` onto a
single cumulative-token x-axis, with cycle/phase boundaries annotated.

Usage:
    python -m scripts.plot_health outputs/v12/
    python -m scripts.plot_health outputs/v12/ --bs 80 --T 128 --out health.png
"""

import argparse
import json
import os
import re
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P1_COLOR = "#1a73e8"
P2_COLOR = "#d93025"
BOOT_COLOR = "#555555"
P1_SHADE = "#e8f0fe"
P2_SHADE = "#fce8e6"
BOOT_SHADE = "#f1f3f4"


def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def smooth(y, window=50):
    y = np.asarray(y, dtype=float)
    if y.size < window * 2:
        window = max(y.size // 4, 1)
    if window < 2:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def collect_segments(work_dir, bs, T):
    """Return list of dicts: {label, phase, cycle, records, x_tok (np.ndarray)}."""
    tokens_per_p1_step = bs * T
    segments = []
    cum = 0.0

    boot = load_jsonl(os.path.join(work_dir, "metrics.jsonl"))
    if boot:
        steps = np.array([r["step"] for r in boot], dtype=float)
        s0 = steps[0]
        x = cum + (steps - s0 + 1) * tokens_per_p1_step
        segments.append(dict(
            label="boot", phase="p1", cycle=-1, records=boot, x_tok=x,
        ))
        cum = float(x[-1])

    cycle_dirs = sorted(glob(os.path.join(work_dir, "cycle_*")))
    for cd in cycle_dirs:
        m = re.search(r"cycle_(\d+)", cd)
        if not m:
            continue
        ci = int(m.group(1))

        p1 = load_jsonl(os.path.join(cd, "metrics.jsonl"))
        if p1:
            steps = np.array([r["step"] for r in p1], dtype=float)
            s0 = steps[0]
            x = cum + (steps - s0 + 1) * tokens_per_p1_step
            segments.append(dict(
                label=f"C{ci}-P1", phase="p1", cycle=ci, records=p1, x_tok=x,
            ))
            cum = float(x[-1])

        p2 = load_jsonl(os.path.join(cd, "phase2_metrics.jsonl"))
        if p2:
            # tokens_seen resets periodically within each stage; compute true
            # phase-2 total as sum of per-stage maxima, then spread records
            # linearly by step index so x-axis is monotone and scaled correctly.
            per_stage_max = {}
            for r in p2:
                sw = r.get("stage_window", 0)
                t = r.get("tokens_seen", 0) or 0
                if t > per_stage_max.get(sw, 0):
                    per_stage_max[sw] = t
            total_p2 = float(sum(per_stage_max.values())) or (len(p2) * bs * T)
            steps = np.array([r.get("step", i + 1) for i, r in enumerate(p2)], dtype=float)
            if steps[-1] > 0:
                x = cum + (steps / steps[-1]) * total_p2
            else:
                x = cum + np.linspace(0, total_p2, len(p2))
            segments.append(dict(
                label=f"C{ci}-P2", phase="p2", cycle=ci, records=p2, x_tok=x,
            ))
            cum = float(x[-1])

    return segments


def plot_panel(ax, segments, p1_key, p2_key=None, p2_transform=None,
               label=None, log=False, scatter_p2=False, smoothing=True):
    for seg in segments:
        xs = seg["x_tok"]
        if seg["phase"] == "p1":
            ys = [r.get(p1_key) for r in seg["records"]]
            ys = np.array([y if y is not None else np.nan for y in ys], dtype=float)
            if np.all(np.isnan(ys)):
                continue
            color = BOOT_COLOR if seg["cycle"] == -1 else P1_COLOR
            if smoothing and ys.size > 100:
                ax.plot(xs, ys, color=color, alpha=0.12, linewidth=0.5)
                ax.plot(xs, smooth(ys, 50), color=color, linewidth=1.6)
            else:
                ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.9)
        else:
            if p2_key is None:
                continue
            ys = [r.get(p2_key) for r in seg["records"]]
            ys = np.array([y if y is not None else np.nan for y in ys], dtype=float)
            if np.all(np.isnan(ys)):
                continue
            if p2_transform is not None:
                ys = p2_transform(ys)
            if scatter_p2:
                ax.scatter(xs, ys, color=P2_COLOR, s=4, alpha=0.5)
            else:
                if smoothing and ys.size > 50:
                    ax.plot(xs, ys, color=P2_COLOR, alpha=0.15, linewidth=0.5)
                    ax.plot(xs, smooth(ys, 20), color=P2_COLOR, linewidth=1.6)
                else:
                    ax.plot(xs, ys, color=P2_COLOR, linewidth=1.2, alpha=0.9)
    if log:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if label:
        ax.set_ylabel(label)


def shade_phases(ax, segments):
    for seg in segments:
        xs = seg["x_tok"]
        if xs.size == 0:
            continue
        x0, x1 = xs[0], xs[-1]
        if seg["phase"] == "p2":
            ax.axvspan(x0, x1, color=P2_SHADE, alpha=0.6, zorder=0)
        elif seg["cycle"] == -1:
            ax.axvspan(x0, x1, color=BOOT_SHADE, alpha=0.6, zorder=0)
        else:
            ax.axvspan(x0, x1, color=P1_SHADE, alpha=0.6, zorder=0)


def annotate_segments(ax, segments, total_tokens, x_lo=0.0):
    """Label one tag per cycle inside the top of the axes, plus a dashed
    vertical line at each cycle boundary."""
    seen_cycles = set()
    for seg in segments:
        xs = seg["x_tok"]
        if xs.size == 0:
            continue
        if seg["cycle"] == -1 and -1 not in seen_cycles:
            x0_vis = max(xs[0], x_lo)
            ax.text(0.5 * (x0_vis + xs[-1]), 0.97, "boot",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color="#333",
                    fontweight="bold")
            seen_cycles.add(-1)
        elif seg["cycle"] >= 0 and seg["cycle"] not in seen_cycles:
            ax.axvline(xs[0], color="#888", linestyle="--", linewidth=0.7,
                       alpha=0.8, zorder=1)
            cycle_end = max(s["x_tok"][-1] for s in segments if s["cycle"] == seg["cycle"])
            ax.text(0.5 * (xs[0] + cycle_end), 0.97,
                    f"C{seg['cycle']}",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color="#333",
                    fontweight="bold")
            seen_cycles.add(seg["cycle"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir", type=str)
    ap.add_argument("--bs", type=int, default=80, help="phase 1 batch size")
    ap.add_argument("--T", type=int, default=128, help="phase 1 sequence length")
    ap.add_argument("--bootstrap-tail", type=float, default=0.1,
                    help="fraction of bootstrap to show (0=hide, 1=all, default 0.1)")
    ap.add_argument("--out", type=str, default=None,
                    help="output path (default: <work_dir>/plots/health.png)")
    args = ap.parse_args()

    segments = collect_segments(args.work_dir, args.bs, args.T)
    if not segments:
        print(f"no metrics found under {args.work_dir}")
        return
    total_tokens = max(seg["x_tok"][-1] for seg in segments if seg["x_tok"].size)

    # compute x-axis lower bound: show only the tail of the bootstrap segment
    x_lo = 0.0
    boot_segs = [s for s in segments if s["cycle"] == -1]
    if boot_segs and args.bootstrap_tail < 1.0:
        boot_end = boot_segs[-1]["x_tok"][-1]
        boot_start = boot_segs[0]["x_tok"][0]
        x_lo = boot_end - (boot_end - boot_start) * max(args.bootstrap_tail, 0.0)

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(7, 1, figsize=(16, 20), sharex=True)

    ax = axes[0]
    def p2_neg_ce(records):
        ys = []
        for r in records:
            rm = r.get("reward_mean")
            if rm is None:
                ys.append(np.nan)
                continue
            rmc = r.get("reward_mean_complete")
            if rmc is not None and rmc != 0.0:
                ys.append(-rmc)
                continue
            cf = r.get("complete_fraction")
            if cf and cf > 0:
                ys.append(-rm / cf)
            else:
                ys.append(-rm * 2.0)
        return np.array(ys, dtype=float)

    # plot phase 1 loss (blue) and phase 2 -CE (red) manually so we can use
    # the complete-window-corrected reward
    for seg in segments:
        xs = seg["x_tok"]
        if seg["phase"] == "p1":
            ys = np.array([r.get("loss", np.nan) for r in seg["records"]], dtype=float)
            color = BOOT_COLOR if seg["cycle"] == -1 else P1_COLOR
            if ys.size > 100:
                ax.plot(xs, ys, color=color, alpha=0.12, linewidth=0.5)
                ax.plot(xs, smooth(ys, 50), color=color, linewidth=1.6)
            else:
                ax.plot(xs, ys, color=color, linewidth=1.2)
        else:
            ys = p2_neg_ce(seg["records"])
            if ys.size > 50:
                ax.plot(xs, ys, color=P2_COLOR, alpha=0.15, linewidth=0.5)
                ax.plot(xs, smooth(ys, 20), color=P2_COLOR, linewidth=1.6)
            else:
                ax.plot(xs, ys, color=P2_COLOR, linewidth=1.2)
    ax.set_ylabel("LM CE — P1 loss / P2 −CE (complete-window)")
    ax.grid(True, alpha=0.3)
    # y-range using only data within the visible x-window
    vals = []
    for s in segments:
        if s["phase"] == "p1":
            ys = np.array([r.get("loss", np.nan) for r in s["records"]], dtype=float)
        else:
            ys = p2_neg_ce(s["records"])
        for xi, y in zip(s["x_tok"], ys):
            if xi >= x_lo and np.isfinite(y):
                vals.append(y)
    if vals:
        all_vis = np.array(vals, dtype=float)
        y_hi = float(np.nanpercentile(all_vis, 99.5))
        y_lo = float(np.nanmin(all_vis))
        pad = (y_hi - y_lo) * 0.08 if y_hi > y_lo else 0.5
        ax.set_ylim(y_lo - pad, y_hi + pad)

    # Panel 1: held-out eval CE — train with memory (line), no-memory (dots),
    # P2 quant-argmax eval (red dots = deterministic-policy CE)
    ax = axes[1]
    for seg in segments:
        xs = seg["x_tok"]
        if seg["phase"] == "p1":
            em = np.array([r.get("eval_ce_loss", np.nan) for r in seg["records"]], dtype=float)
            en = np.array([r.get("eval_ce_loss_no_mem", np.nan) for r in seg["records"]], dtype=float)
            valid = np.isfinite(em)
            if valid.any():
                ax.scatter(xs[valid], em[valid], s=10, color=P1_COLOR, alpha=0.8,
                           label="P1 eval_ce (w/ mem)" if seg["cycle"] == 0 else None)
            valid_nm = np.isfinite(en)
            if valid_nm.any():
                ax.scatter(xs[valid_nm], en[valid_nm], s=10, color="#fbbc04",
                           marker="x", alpha=0.8,
                           label="P1 eval_ce_no_mem" if seg["cycle"] == 0 else None)
        else:
            qe = np.array([r.get("quant_eval_ce", np.nan) for r in seg["records"]], dtype=float)
            ev = np.array([r.get("eval_ce_loss", np.nan) for r in seg["records"]], dtype=float)
            valid = np.isfinite(qe)
            if valid.any():
                ax.scatter(xs[valid], qe[valid], s=10, color=P2_COLOR, alpha=0.8,
                           label="P2 quant_eval_ce" if seg["cycle"] == 0 else None)
            valid_ev = np.isfinite(ev)
            if valid_ev.any():
                ax.scatter(xs[valid_ev], ev[valid_ev], s=10, color="#9334e6",
                           marker="^", alpha=0.7,
                           label="P2 eval_ce" if seg["cycle"] == 0 else None)
    ax.set_ylabel("eval CE (held-out)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", frameon=True)
    # clip: use percentile-based ylim to ignore early bootstrap outliers
    _all_eval = []
    for seg in segments:
        for k in ("eval_ce_loss", "eval_ce_loss_no_mem", "quant_eval_ce"):
            _all_eval.extend(r[k] for r in seg["records"] if r.get(k) is not None)
    if _all_eval:
        arr = np.array(_all_eval, dtype=float)
        y_lo = float(np.nanpercentile(arr, 1)) - 0.1
        y_hi = float(np.nanpercentile(arr, 99)) + 0.1
        ax.set_ylim(y_lo, y_hi)

    # Panel 2: memory leverage (P1) — eval_ce − eval_ce_no_mem
    # Negative = memory HELPS at inference. This is the smoking-gun metric.
    ax = axes[2]
    for seg in segments:
        if seg["phase"] != "p1":
            continue
        xs = seg["x_tok"]
        em = np.array([r.get("eval_ce_loss", np.nan) for r in seg["records"]], dtype=float)
        en = np.array([r.get("eval_ce_loss_no_mem", np.nan) for r in seg["records"]], dtype=float)
        leverage = em - en
        valid = np.isfinite(leverage)
        if valid.any():
            color = BOOT_COLOR if seg["cycle"] == -1 else P1_COLOR
            ax.scatter(xs[valid], leverage[valid], s=15, color=color, alpha=0.8)
    ax.axhline(0, color="#888", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("memory leverage\n(eval_CE − eval_CE_no_mem)\nnegative = mem helps")
    ax.grid(True, alpha=0.3)
    # clip memory leverage y-range too
    _lev = []
    for seg in segments:
        for r in seg["records"]:
            em = r.get("eval_ce_loss"); en = r.get("eval_ce_loss_no_mem")
            if em is not None and en is not None:
                _lev.append(em - en)
    if _lev:
        arr = np.array(_lev, dtype=float)
        y_lo = float(np.nanpercentile(arr, 1)) - 0.05
        y_hi = float(np.nanpercentile(arr, 99)) + 0.05
        ax.set_ylim(y_lo, y_hi)

    ax = axes[3]
    plot_panel(ax, segments, p1_key="mod_grad_norm", p2_key="mod_grad_norm",
               label="mod_grad_norm (log)", log=True)

    ax = axes[4]
    plot_panel(ax, segments, p1_key=None, p2_key="k_spread_mean",
               label="k_spread_mean (P2)", scatter_p2=True)

    ax = axes[5]
    plot_panel(ax, segments, p1_key="h_norm", p2_key=None,
               label="h_norm + decay_mean", smoothing=True)
    plot_panel(ax, segments, p1_key="decay_mean", p2_key=None, smoothing=True)

    ax = axes[6]
    plot_panel(ax, segments, p1_key="tok_s", p2_key=None, label="tok/s (P1)")
    ax.set_xlabel("cumulative tokens")

    for ax in axes:
        shade_phases(ax, segments)
        ax.set_xlim(x_lo, total_tokens)

    annotate_segments(axes[0], segments, total_tokens, x_lo=x_lo)

    # legend for phase shading
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=BOOT_SHADE, label="bootstrap (P1)"),
        mpatches.Patch(color=P1_SHADE, label="cycle phase 1"),
        mpatches.Patch(color=P2_SHADE, label="cycle phase 2 (GRPO)"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.99, 0.985))

    fig.suptitle(f"Training health — {args.work_dir}", fontweight="bold",
                 y=0.995, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    out = args.out or os.path.join(args.work_dir, "plots", "health.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)

    n_p1 = sum(1 for s in segments if s["phase"] == "p1")
    n_p2 = sum(1 for s in segments if s["phase"] == "p2")
    print(f"wrote {out}  ({n_p1} phase-1 segments, {n_p2} phase-2 segments, "
          f"{total_tokens/1e6:.1f}M tokens)")


if __name__ == "__main__":
    main()
