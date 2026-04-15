"""Retrospective probe across cycle checkpoints.

Walks a `outputs/v12/` work dir, loads each `cycle_NN/phase2.pt` + `codebook.pt`,
and computes per-cycle statistics that are hard to reconstruct from metrics
alone — modulator weight drift from the first cycle, codebook centroid drift
between consecutive cycles, action database summary stats.

Output: `<work_dir>/retrospective.jsonl` + `<work_dir>/plots/retrospective.png`.

Usage:
    python -m scripts.retrospective_probe outputs/v12/
"""

import argparse
import json
import math
import os
import re
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


MOD_KEYS = ["memory.mod_w1", "memory.mod_b1", "memory.mod_w2", "memory.mod_b2"]


def _load_state(path):
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _mod_weights(ckpt):
    sd = ckpt["model_state_dict"]
    return {k: sd[k].float() for k in MOD_KEYS if k in sd}


def _mod_distance(a, b):
    if not a or not b:
        return float("nan"), float("nan")
    diff_sq = sum((a[k] - b[k]).pow(2).sum().item() for k in a if k in b)
    ref_sq = sum(b[k].pow(2).sum().item() for k in b)
    return math.sqrt(diff_sq), math.sqrt(diff_sq) / max(math.sqrt(ref_sq), 1e-8)


def _codebook_centroids(cb_state):
    # 'state_dict' contains "rvq.codebooks" normally, or it might be flat
    sd = cb_state.get("state_dict", cb_state)
    for k in ("rvq.codebooks", "codebooks"):
        if k in sd:
            return sd[k].float()
    return None


def probe(work_dir):
    cycle_dirs = sorted(glob(os.path.join(work_dir, "cycle_*")))
    records = []

    # Reference modulator = phase 1 end of cycle 0 (first available)
    ref_mod = None
    prev_codebook = None

    for cd in cycle_dirs:
        m = re.search(r"cycle_(\d+)", cd)
        if not m:
            continue
        ci = int(m.group(1))
        p2 = _load_state(os.path.join(cd, "phase2.pt"))
        p1e = _load_state(os.path.join(cd, "phase1_end.pt"))
        cb = _load_state(os.path.join(cd, "codebook.pt"))

        # Skip cycles without completed phase 2
        if p2 is None:
            continue

        mod_now = _mod_weights(p2)
        mod_p1 = _mod_weights(p1e) if p1e is not None else None

        if ref_mod is None:
            ref_mod = mod_now  # first completed cycle is the baseline

        # Drift from first cycle's phase-2-end modulator
        drift_abs, drift_rel = _mod_distance(mod_now, ref_mod)
        # Drift from this cycle's own phase-1 end (how much did GRPO move?)
        p2_abs, p2_rel = _mod_distance(mod_now, mod_p1) if mod_p1 else (float("nan"), float("nan"))

        # Codebook centroid drift
        cb_now = _codebook_centroids(cb) if cb is not None else None
        if cb_now is not None and prev_codebook is not None:
            cb_diff = (cb_now - prev_codebook).pow(2).sum().sqrt().item()
            cb_ref = prev_codebook.pow(2).sum().sqrt().item()
            cb_drift_rel = cb_diff / max(cb_ref, 1e-8)
        else:
            cb_diff = float("nan")
            cb_drift_rel = float("nan")

        # Codebook structure (spacing)
        if cb_now is not None and cb_now.ndim == 3:
            L, K, D = cb_now.shape
            # pairwise L2 per level
            pairwise = []
            for l in range(L):
                c = cb_now[l]
                pd = (c.unsqueeze(0) - c.unsqueeze(1)).pow(2).sum(-1).sqrt()
                mask = ~torch.eye(K, dtype=torch.bool)
                pairwise.append(pd[mask].mean().item())
            cb_spacing_mean = float(np.mean(pairwise))
        else:
            cb_spacing_mean = float("nan")

        rec = {
            "cycle": ci,
            "mod_drift_from_ref_abs": drift_abs,
            "mod_drift_from_ref_rel": drift_rel,
            "mod_drift_p1_to_p2_abs": p2_abs,
            "mod_drift_p1_to_p2_rel": p2_rel,
            "codebook_drift_abs": cb_diff,
            "codebook_drift_rel": cb_drift_rel,
            "codebook_spacing_mean": cb_spacing_mean,
        }
        records.append(rec)

        if cb_now is not None:
            prev_codebook = cb_now.clone()

        print(f"cycle {ci}: mod_drift_rel={drift_rel:.4f} "
              f"(p1→p2: {p2_rel:.4f})  cb_drift={cb_drift_rel:.4f} "
              f"cb_spacing={cb_spacing_mean:.3f}")

    # Write jsonl
    out_jsonl = os.path.join(work_dir, "retrospective.jsonl")
    with open(out_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Plot
    if records:
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        cycles = [r["cycle"] for r in records]

        axes[0].plot(cycles, [r["mod_drift_from_ref_rel"] for r in records],
                     "o-", label="from cycle 0", color="#1a73e8")
        axes[0].plot(cycles, [r["mod_drift_p1_to_p2_rel"] for r in records],
                     "s-", label="P1→P2 (per cycle)", color="#d93025")
        axes[0].set_ylabel("modulator drift (rel)")
        axes[0].set_title("Modulator weight drift", fontweight="bold", loc="left")
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.3)

        axes[1].plot(cycles, [r["codebook_drift_rel"] for r in records],
                     "o-", color="#7b1fa2")
        axes[1].set_ylabel("codebook drift vs prev cycle (rel)")
        axes[1].set_title("Codebook centroid drift (cycle-to-cycle)",
                          fontweight="bold", loc="left")
        axes[1].grid(alpha=0.3)

        axes[2].plot(cycles, [r["codebook_spacing_mean"] for r in records],
                     "o-", color="#188038")
        axes[2].set_ylabel("mean pairwise L2")
        axes[2].set_title("Codebook code spacing", fontweight="bold", loc="left")
        axes[2].set_xlabel("cycle")
        axes[2].grid(alpha=0.3)

        fig.suptitle(f"Retrospective probe — {work_dir}", fontweight="bold")
        fig.tight_layout()
        plot_path = os.path.join(work_dir, "plots", "retrospective.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig.savefig(plot_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {plot_path}")

    print(f"wrote {out_jsonl} ({len(records)} cycles)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir", type=str)
    args = ap.parse_args()
    probe(args.work_dir)


if __name__ == "__main__":
    main()
