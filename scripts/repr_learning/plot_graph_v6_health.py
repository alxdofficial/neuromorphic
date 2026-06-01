#!/usr/bin/env python3
"""graph_v6 training-health plot — time series of the v6 telemetry from a run's
jsonl, so you can diagnose at a glance whether the mechanism is learning healthily.

Panels (all read from the `phase=="val"` rows, which carry the eval-only probes
as ``val_graph_v6_*`` keys; write-side gates are also logged on train rows):
  1. loss context        — val_loss_recon (+ top1 on twin axis)
  2. read alive/no-op     — rezero_scale_eff (read injection magnitude; ~0 = dead read)
                             state_effect    (||fact - fact|state=0||; ~0 = edge-state ignored,
                                              i.e. the no-op-free principle is violated)
  3. node-bank health     — node_collapse_cos (off-diag cosine; →1 = collapsed bank)
                             node_active_frac  (fraction of nodes any edge points at; →0 = hub collapse)
  4. read-pointer sharpness — read_src_entropy / read_dst_entropy (low = sharp picks, high = diffuse)
  5. write gates + facts  — node_gate_mean_avg / edge_gate_mean_avg (are slots updating?) + fact_norm

Usage:
  python scripts/repr_learning/plot_graph_v6_health.py --jsonl <path/to/graph_v6_baseline.jsonl>
  python scripts/repr_learning/plot_graph_v6_health.py --run-dir outputs/repr_learning/<tag>_graph_v6_baseline
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_val_rows(jsonl_path: Path) -> list[dict]:
    rows = []
    for line in jsonl_path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("phase") == "val" and "val_loss_recon" in r:
            rows.append(r)
    return rows


def _series(rows, key):
    """(steps, values) for rows that have `key` (non-None)."""
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if v is not None:
            xs.append(r["step"]); ys.append(float(v))
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None,
                    help="output dir; expects jsonl/graph_v6_baseline.jsonl under it")
    ap.add_argument("--out", type=str, default=None, help="PNG path (default: next to jsonl)")
    args = ap.parse_args()

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
    elif args.run_dir:
        rd = Path(args.run_dir)
        cands = list(rd.glob("**/graph_v6_baseline.jsonl")) or list(rd.glob("**/*.jsonl"))
        if not cands:
            raise SystemExit(f"no jsonl under {rd}")
        jsonl_path = cands[0]
    else:
        raise SystemExit("pass --jsonl or --run-dir")

    rows = _load_val_rows(jsonl_path)
    if not rows:
        raise SystemExit(f"no val rows in {jsonl_path}")
    out = Path(args.out) if args.out else jsonl_path.with_name("graph_v6_health.png")

    fig, axes = plt.subplots(5, 1, figsize=(11, 16), sharex=True)
    P = "val_graph_v6_"

    # 1. loss context
    ax = axes[0]
    xs, ys = _series(rows, "val_loss_recon")
    ax.plot(xs, ys, "-o", ms=3, color="#333", label="val_loss_recon")
    ax.set_ylabel("val_loss_recon"); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    xt, yt = _series(rows, "val_top1_acc")
    if yt:
        ax2 = ax.twinx(); ax2.plot(xt, [v * 100 for v in yt], "-s", ms=3, color="#1f77b4", label="top1 %")
        ax2.set_ylabel("top1 %", color="#1f77b4"); ax2.legend(loc="upper right")
    ax.set_title(f"graph_v6 health — {jsonl_path.parent.parent.name}")

    # 2. read alive / no-op-free
    ax = axes[1]
    for key, c, lab in [(P + "rezero_scale_eff", "#d62728", "rezero_scale_eff (read magnitude)"),
                        (P + "state_effect", "#2ca02c", "state_effect (no-op-free Δ)")]:
        xs, ys = _series(rows, key)
        if ys: ax.plot(xs, ys, "-o", ms=3, color=c, label=lab)
    ax.axhline(0.0, color="grey", lw=0.8, ls="--")
    ax.set_ylabel("read alive / no-op"); ax.legend(loc="best"); ax.grid(alpha=0.3)

    # 3. node-bank health
    ax = axes[2]
    for key, c, lab in [(P + "node_collapse_cos", "#9467bd", "node_collapse_cos (→1 bad)"),
                        (P + "node_active_frac", "#ff7f0e", "node_active_frac (→0 bad)")]:
        xs, ys = _series(rows, key)
        if ys: ax.plot(xs, ys, "-o", ms=3, color=c, label=lab)
    ax.set_ylabel("node bank"); ax.set_ylim(-0.05, 1.05); ax.legend(loc="best"); ax.grid(alpha=0.3)

    # 4. read-pointer entropy
    ax = axes[3]
    for key, c, lab in [(P + "read_src_entropy", "#1f77b4", "src entropy"),
                        (P + "read_dst_entropy", "#e377c2", "dst entropy")]:
        xs, ys = _series(rows, key)
        if ys: ax.plot(xs, ys, "-o", ms=3, color=c, label=lab)
    ax.set_ylabel("read-pointer entropy"); ax.legend(loc="best"); ax.grid(alpha=0.3)

    # 5. write gates + fact norm
    ax = axes[4]
    for key, c, lab in [(P + "node_gate_mean_avg", "#8c564b", "node_gate_mean"),
                        (P + "edge_gate_mean_avg", "#17becf", "edge_gate_mean")]:
        xs, ys = _series(rows, key)
        if ys: ax.plot(xs, ys, "-o", ms=3, color=c, label=lab)
    ax.set_ylabel("write gate mean"); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    xs, ys = _series(rows, P + "fact_norm")
    if ys:
        ax3 = ax.twinx(); ax3.plot(xs, ys, "-^", ms=3, color="#bcbd22", label="fact_norm")
        ax3.set_ylabel("fact_norm", color="#bcbd22"); ax3.legend(loc="upper right")
    ax.set_xlabel("step")

    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"wrote {out}  ({len(rows)} val points from {jsonl_path})")


if __name__ == "__main__":
    main()
