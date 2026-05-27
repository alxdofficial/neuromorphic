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


# (variant_label, output_dir_under_outputs/repr_learning, jsonl_filename, color, group)
# Order here = legend order. group is used for marker / color family in 3D.
VARIANTS = [
    ("v5.4 (graph + graph MP)",
     "v5_4_first_graph_v5_baseline", "graph_v5_baseline.jsonl", "#d62728", "graph_v5"),
    ("v5.1-first (graph + xfmr)",
     "v5_1_first_graph_v5_baseline", "graph_v5_baseline.jsonl", "#ff7f0e", "graph_v5"),
    ("mamba (recurrent SSM)",
     "v1h_t4k_v3_recurrent_baseline", "recurrent_baseline.jsonl", "#1f77b4", "baseline"),
    ("continuous (slot-attn)",
     "v1h_t4k_v3_continuous_baseline", "continuous_baseline.jsonl", "#2ca02c", "baseline"),
    ("memorizing (top-K KV)",
     "v1h_t4k_v3_memorizing_baseline", "memorizing_baseline.jsonl", "#9467bd", "baseline"),
    ("flat (VQ codebook)",
     "v1h_t4k_v3_flat_baseline", "flat_baseline.jsonl", "#8c564b", "baseline"),
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


def _collect_outputs_root(out_root: Path) -> dict:
    out = OrderedDict()
    for label, subdir, jsonl_name, color, group in VARIANTS:
        path = out_root / subdir / "jsonl" / jsonl_name
        row = _read_best_row(path)
        if row is None:
            print(f"[warn] {label}: no val rows in {path}")
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
    import plotly.graph_objects as go

    # Single 3D scatter: x=best_step, y=val_recon, z=top1.
    # The "good corner" is (low x, low y, high z): fast to converge, low loss,
    # high accuracy. Markers stay one trace per variant so the legend lists
    # them individually instead of being collapsed by group.
    fig = go.Figure()

    for label, d in data.items():
        symbol = "diamond" if d["group"] == "graph_v5" else "circle"
        fig.add_trace(go.Scatter3d(
            x=[d["step"]],
            y=[d["val_recon"]],
            z=[d["top1"] * 100.0],
            mode="markers+text",
            marker=dict(size=10, color=d["color"], symbol=symbol,
                        line=dict(color="#222", width=0.8)),
            text=[label.split(" (")[0]],
            textposition="top center",
            textfont=dict(size=11),
            name=label,
            hovertext=(f"{label}<br>"
                       f"best_step: {d['step']}<br>"
                       f"val_recon: {d['val_recon']:.3f}<br>"
                       f"top-1: {d['top1']*100:.1f}%"),
            hoverinfo="text",
        ))

    fig.update_layout(
        title=("Scoreboard: best_step × val_recon × top-1 "
               "(good corner = front-left-top)"),
        scene=dict(
            xaxis=dict(title="steps to best.pt", autorange="reversed"),
            yaxis=dict(title="val_recon (lower better)", autorange="reversed"),
            zaxis=dict(title="top-1 accuracy (%)"),
            aspectmode="cube",
        ),
        legend=dict(itemsizing="constant"),
        height=750,
    )
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
    print(f"[output] wrote {out_path}")


def plot_per_family(data: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    # Union of families across variants. Preserve roughly stable order with
    # composite_v1's 9 first, then external corpora (hotpot_qa, narrative_qa).
    canonical_order = [
        "biographical", "hotpot_qa", "boxes", "calendar", "preferences",
        "theory_of_mind", "knights", "triage", "revisions", "passphrase",
        "narrative_qa",
    ]
    all_families = set()
    for d in data.values():
        all_families.update(d["per_family"].keys())
    families = [f for f in canonical_order if f in all_families]
    # Catch any unexpected new family.
    for f in sorted(all_families - set(canonical_order)):
        families.append(f)

    variants = list(data.keys())
    n_var = len(variants)
    n_fam = len(families)
    x = np.arange(n_fam)
    bar_w = 0.8 / max(n_var, 1)

    fig, ax = plt.subplots(figsize=(max(12, 1.0 * n_fam), 5.5))
    for vi, vlabel in enumerate(variants):
        d = data[vlabel]
        ys = [d["per_family"].get(f, {}).get("mean_loss", np.nan) for f in families]
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
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = _collect_outputs_root(args.outputs_root)
    print(f"[loaded] {len(data)} variants")
    for label, d in data.items():
        print(f"  {label:35s}  step={d['step']:>5}  val={d['val_recon']:.3f}  "
              f"top1={d['top1']*100:.1f}%  families={len(d['per_family'])}")

    if args.mode in ("3d", "both"):
        plot_3d(data, args.out_dir / "scoreboard_3d.html")
    if args.mode in ("family", "both"):
        plot_per_family(data, args.out_dir / "scoreboard_per_family.png")


if __name__ == "__main__":
    main()
