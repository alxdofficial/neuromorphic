"""Two plots for the cross-variant scoreboard:

1. ``--mode 3d``: interactive Plotly HTML, 3D scatter (val_recon, top1, best_step).
   One dot per variant. Color-codes graph lineage vs other architectures.

2. ``--mode family``: static matplotlib PNG, grouped bar plot.
   Per-family val_recon across all variants. Cells missing in a variant's mix
   (e.g. NarrativeQA absent from v5 runs) are left blank, not zeroed.

Reads each variant's jsonl, finds the row with the lowest ``val_loss_recon``
(the trainer-protocol best.pt), and extracts (val_loss_recon, val_top1_acc,
val_per_family, step) at that point.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np


# (variant_label, variant_key, color, group) — the 7 live v2.1 arms. The run
# dir is derived as f"{out_tag}_{variant_key}" and the jsonl as f"{variant_key}.jsonl",
# matching the trainer's output layout, so pass --out-tag for a given sweep.
VARIANTS = [
    ("graph_v6 (primary)",        "graph_v6_baseline",     "#d62728", "graph_v6"),
    ("flat (VQ codebook)",        "flat_baseline",         "#8c564b", "baseline"),
    ("continuous (slot-attn)",    "continuous_baseline",   "#2ca02c", "baseline"),
    ("mamba (recurrent SSM)",     "recurrent_baseline",    "#1f77b4", "baseline"),
    ("memorizing (top-K KV)",     "memorizing_baseline",   "#9467bd", "baseline"),
    ("vanilla (no context)",      "vanilla_llama",         "#7f7f7f", "vanilla"),
    ("vanilla (full context)",    "vanilla_full_context",  "#17becf", "vanilla"),
]


def _read_best_row(jsonl_path: Path) -> Optional[dict]:
    if not jsonl_path.exists():
        return None
    best = None
    for line in jsonl_path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "val_loss_recon" not in row:
            continue
        if best is None or row["val_loss_recon"] < best["val_loss_recon"]:
            best = row
    return best


def _collect_outputs_root(out_root: Path, out_tag: str) -> dict:
    out = OrderedDict()
    for label, variant, color, group in VARIANTS:
        subdir, jsonl_name = f"{out_tag}_{variant}", f"{variant}.jsonl"
        path = out_root / subdir / "jsonl" / jsonl_name
        row = _read_best_row(path)
        if row is None:
            print(f"[warn] {label}: no val rows in {path}")
            continue
        if int(row.get("step", 0)) == 0:
            # A single step-0 row = an untrained snapshot (e.g. the full-context
            # ceiling logged its val once at init); not a real best.pt. Skip so
            # it doesn't masquerade as a converged point.
            print(f"[warn] {label}: only an untrained step-0 row — skipping")
            continue
        out[label] = {
            "color": color,
            "group": group,
            "step": int(row.get("step", 0)),
            "val_recon": float(row["val_loss_recon"]),
            "top1": float(row.get("val_top1_acc", 0.0)),
            "per_family": row.get("val_per_family", {}),
        }
    return out


def plot_3d(data: dict, out_path: Path) -> None:
    # A clean 2D bubble chart conveys all three quantities far more legibly than
    # a matplotlib 3D scatter: x = steps-to-best (← faster), y = answer-NLL
    # (↓ lower = better), bubble AREA ∝ top-1. (Kept the function name/filename
    # for callers; the old 3D render was unreadable.)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    for label, d in data.items():
        top1 = d["top1"] * 100.0
        marker = "D" if d["group"] == "graph_v6" else "o"
        ax.scatter(d["step"], d["val_recon"], s=140 + 26 * top1,
                   c=[d["color"]], marker=marker, edgecolors="#222",
                   linewidths=0.9, alpha=0.88, zorder=3)
        ax.annotate(f"{label.split(' (')[0]}  ·  top-1 {top1:.0f}%",
                    (d["step"], d["val_recon"]), textcoords="offset points",
                    xytext=(9, 9), fontsize=9, zorder=4)
    ax.set_xlabel("steps to best.pt  (← faster convergence)")
    ax.set_ylabel("answer-NLL at best.pt  (↓ lower = better)")
    ax.invert_yaxis()
    ax.grid(alpha=0.22, zorder=0)
    ax.set_title("Scoreboard — convergence × answer-NLL × top-1 (bubble area)\n"
                 "v2.1 fair sweep, hard families")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[output] wrote {out_path}")


def _collapse_families(per_family: dict) -> dict:
    """Aggregate the 10 babilong_qaN sub-tasks into one n-weighted ``babilong``
    bar (the v2.1 mix logs them split); keep every other family as-is."""
    out, bab_loss, bab_n = {}, 0.0, 0
    for fam, v in per_family.items():
        ml = v.get("mean_loss")
        if ml is None:
            continue
        if fam.startswith("babilong"):
            n = v.get("n", 1)
            bab_loss += ml * n
            bab_n += n
        else:
            out[fam] = ml
    if bab_n:
        out["babilong"] = bab_loss / bab_n
    return out


def plot_per_family(data: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    # The actual v2.1 hard-family mix (NOT the old composite-subtask list).
    canonical_order = ["biographical", "hotpot_qa", "musique", "narrative_qa", "babilong"]
    collapsed = {lbl: _collapse_families(d["per_family"]) for lbl, d in data.items()}
    all_families = set().union(*collapsed.values()) if collapsed else set()
    families = [f for f in canonical_order if f in all_families]
    for f in sorted(all_families - set(canonical_order)):
        families.append(f)

    variants = list(data.keys())
    n_var = len(variants)
    n_fam = len(families)
    x = np.arange(n_fam)
    bar_w = 0.8 / max(n_var, 1)

    fig, ax = plt.subplots(figsize=(max(11, 1.6 * n_fam), 5.5))
    for vi, vlabel in enumerate(variants):
        d = data[vlabel]
        ys = [collapsed[vlabel].get(f, np.nan) for f in families]
        offset = (vi - (n_var - 1) / 2) * bar_w
        bars = ax.bar(x + offset, ys, bar_w,
                      label=vlabel, color=d["color"],
                      edgecolor="#222", linewidth=0.3)
        # Mark NaN families (mix didn't include them) with a faint dash so
        # the absence is visible rather than a missing bar.
        for xi, y in zip(x, ys):
            if np.isnan(y):
                ax.text(xi + offset, 0.05, "—",
                        ha="center", va="bottom", color="#999",
                        fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("val_recon (lower = better)")
    ax.set_title("Per-family val_recon across variants (best.pt)")
    ax.axhline(0, color="#aaa", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"[output] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["3d", "family", "both"], default="both")
    ap.add_argument("--outputs-root", type=Path,
                    default=Path("outputs/repr_learning"))
    ap.add_argument("--out-dir", type=Path, default=Path("docs/plots"))
    ap.add_argument("--out-tag", type=str, required=True,
                    help="Sweep tag: run dirs are <out-tag>_<variant> (e.g. v2_1).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = _collect_outputs_root(args.outputs_root, args.out_tag)
    print(f"[loaded] {len(data)} variants")
    for label, d in data.items():
        print(f"  {label:35s}  step={d['step']:>5}  val={d['val_recon']:.3f}  "
              f"top1={d['top1']*100:.1f}%  families={len(d['per_family'])}")

    if args.mode in ("3d", "both"):
        plot_3d(data, args.out_dir / "scoreboard_3d.png")
    if args.mode in ("family", "both"):
        plot_per_family(data, args.out_dir / "scoreboard_per_family.png")


if __name__ == "__main__":
    main()
