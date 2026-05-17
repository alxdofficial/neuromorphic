#!/usr/bin/env python3
"""viz_manifold.py — Tier 1 concept-manifold visualizations.

Six-panel structural inspection of the concept manifold from a checkpoint.
Reads `outputs/wave{N}/ckpt.pt` and produces one PNG. Answers:
  - Have concepts organized into clusters? (UMAP of ids + states)
  - How many concepts are actually being used? (drift, usage curves)
  - Where do high-usage concepts sit on the topology? (ring overlay)

Panels:
  1. UMAP(concept_ids)      colored by log(usage_ema)
  2. UMAP(state_init)       colored by log(usage_ema)
  3. Ring topology          nodes colored by usage, sample rewired edges
  4. id ↔ state_init cosine histogram  (are these two roles distinct?)
  5. Usage Zipf log-log     usage_ema sorted descending
  6. Text summary           N_active, N_dominant, norm stats

Note: `concept_states` is non-persistent (runtime-only volatile buffer),
so it's NOT in the checkpoint. We inspect `state_init` instead — the
learnable reset state, which captures what the model has decided "good
default content" looks like for each concept.

**SAFE TO RUN DURING TRAINING:**
- All torch.load calls use map_location="cpu" — no CUDA allocation.
- Reads only the manifold tensors (~16 MB once unpacked from the ~3 GB ckpt).
- Refuses to read a ckpt written less than 30 s ago (mid-save guard).
- Matplotlib uses the Agg backend — no display required.

Usage:
    python scripts/diagnostics/viz_manifold.py                # auto wave
    python scripts/diagnostics/viz_manifold.py --wave 1
    python scripts/diagnostics/viz_manifold.py --ckpt path/to/ckpt.pt --out viz.png
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# ── ckpt loading ────────────────────────────────────────────────────────


def load_manifold(ckpt_path: Path) -> dict[str, torch.Tensor]:
    """Pull manifold tensors out of the ckpt, on CPU.

    `concept_states` is intentionally NOT in the ckpt — it's a
    non-persistent runtime buffer that gets reset at sequence start.
    We inspect `state_init` (the learnable reset) instead.
    """
    age_s = time.time() - ckpt_path.stat().st_mtime
    if age_s < 30:
        raise RuntimeError(
            f"ckpt {ckpt_path.name} is only {age_s:.1f}s old — may be "
            f"mid-save. Rerun in a minute."
        )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    return {
        "concept_ids":  sd["manifold.concept_ids"].float(),
        "state_init":   sd["manifold.state_init"].float(),
        "edge_indices": sd["manifold.edge_indices"].long(),
        "usage_ema":    sd["manifold.usage_ema"].float(),
        "step":         ck.get("step", None),
    }


# ── projection helper ──────────────────────────────────────────────────


def project_2d(X: np.ndarray, label: str) -> tuple[np.ndarray, str]:
    """UMAP if available, else PCA. Returns (2D embedding, method string)."""
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(
            n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0,
        )
        return reducer.fit_transform(X), "UMAP"
    except Exception as e:
        print(f"  UMAP unavailable for {label} ({e!r}); falling back to PCA")
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=0).fit_transform(X), "PCA"


# ── panels ──────────────────────────────────────────────────────────────


def panel_projection(ax, X: np.ndarray, usage: np.ndarray, title: str):
    coords, method = project_2d(X, title)
    log_usage = np.log10(usage + 1e-10)
    sc = ax.scatter(
        coords[:, 0], coords[:, 1], c=log_usage, s=7,
        cmap="viridis", alpha=0.75, linewidths=0,
    )
    ax.set_title(f"{method}({title})  —  color = log10(usage)", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)


def panel_ring_topology(ax, m: dict[str, torch.Tensor], cfg_radius: int = 32):
    """Ring layout: nodes colored by usage. Sample a few rewired edges
    to illustrate the small-world structure without drowning in 262K lines."""
    N = m["concept_ids"].shape[0]
    K = m["edge_indices"].shape[1]
    usage = m["usage_ema"].numpy()
    log_usage = np.log10(usage + 1e-10)

    angles = 2 * np.pi * np.arange(N) / N
    xs, ys = np.cos(angles), np.sin(angles)
    edges = m["edge_indices"].numpy()

    # Vectorized edge rendering: collect line segments, draw with
    # LineCollection (much faster than 1K matplotlib .plot calls + lets
    # us afford to draw more edges for legibility).
    from matplotlib.collections import LineCollection
    rng = np.random.default_rng(0)
    sample_nodes = rng.choice(N, size=min(600, N), replace=False)
    local_segs = []
    rewired_segs = []
    for i in sample_nodes:
        for k in range(K):
            tgt = int(edges[i, k])
            d = min(abs(tgt - int(i)), N - abs(tgt - int(i)))
            seg = [(xs[i], ys[i]), (xs[tgt], ys[tgt])]
            if d > cfg_radius:
                rewired_segs.append(seg)
            else:
                local_segs.append(seg)
    if local_segs:
        ax.add_collection(LineCollection(
            local_segs[:1200], colors="grey", alpha=0.04, linewidths=0.4))
    if rewired_segs:
        ax.add_collection(LineCollection(
            rewired_segs[:1600], colors="crimson", alpha=0.10, linewidths=0.5))

    sc = ax.scatter(xs, ys, c=log_usage, s=10, cmap="plasma",
                    alpha=0.95, linewidths=0, zorder=2)
    ax.set_title("Small-world ring  —  crimson = rewired, grey = local",
                 fontsize=12)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-1.18, 1.18); ax.set_ylim(-1.18, 1.18)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)


def panel_id_state_cosine(ax, m: dict[str, torch.Tensor]):
    """Per-concept cosine similarity between concept_ids[i] and state_init[i].

    These are two independently learnable D-dim vectors per concept,
    serving different roles (routing key vs reset content). If they end
    up uncorrelated (centered near 0), the model has used them as
    distinct slots — healthy. If centered near +1 or -1, the model
    collapsed the distinction and they're carrying the same signal.
    """
    ids = m["concept_ids"]
    init = m["state_init"]
    cos = torch.nn.functional.cosine_similarity(ids, init, dim=-1).numpy()
    mean = float(np.mean(cos))
    median = float(np.median(cos))
    ax.hist(cos, bins=80, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(0.0, color="grey", linestyle=":", label="orthogonal")
    ax.axvline(mean, color="crimson", linestyle="--",
               label=f"mean = {mean:+.3f}")
    ax.set_xlabel("cos(concept_ids[i], state_init[i])", fontsize=11)
    ax.set_ylabel("# concepts", fontsize=11)
    ax.set_title(f"id ↔ state_init alignment per concept "
                 f"(median = {median:+.3f})", fontsize=12)
    ax.set_xlim(-1.05, 1.05)
    ax.legend(fontsize=10)


def panel_usage_zipf(ax, m: dict[str, torch.Tensor]):
    usage = m["usage_ema"].numpy() + 1e-12
    sorted_u = np.sort(usage)[::-1]
    ranks = np.arange(1, len(sorted_u) + 1)
    ax.loglog(ranks, sorted_u, color="steelblue", linewidth=1.5)
    ax.set_xlabel("concept rank (by usage)", fontsize=11)
    ax.set_ylabel("usage_ema", fontsize=11)
    ax.set_title("Usage distribution (log-log)", fontsize=12)
    cumulative = np.cumsum(sorted_u) / sorted_u.sum()
    for pct in (0.5, 0.8, 0.95):
        idx = int(np.searchsorted(cumulative, pct))
        idx = max(1, min(idx, len(sorted_u) - 1))
        ax.axvline(idx, color="grey", alpha=0.4, linestyle=":")
        ax.text(idx, sorted_u[idx], f" {pct:.0%}→#{idx}", fontsize=9,
                verticalalignment="bottom")


def panel_text_summary(ax, m: dict[str, torch.Tensor], ckpt_path: Path):
    ax.axis("off")
    N = m["concept_ids"].shape[0]
    D = m["concept_ids"].shape[1]
    K = m["edge_indices"].shape[1]
    usage = m["usage_ema"]
    n_alive = int((usage > 0).sum())
    n_active = int((usage > 1.0 / (100 * N)).sum())
    n_dominant = int((usage > 10.0 / N).sum())

    id_norm = m["concept_ids"].norm(dim=-1)
    init_norm = m["state_init"].norm(dim=-1)
    cos = torch.nn.functional.cosine_similarity(
        m["concept_ids"], m["state_init"], dim=-1,
    )
    age_min = (time.time() - ckpt_path.stat().st_mtime) / 60
    step = m.get("step")

    lines = [
        f"ckpt:           {ckpt_path}",
        f"step:           {step}" if step is not None else "step: ?",
        f"age:            {age_min:.1f} min",
        "",
        f"manifold shape: N={N}, D={D}, K={K}",
        "",
        "── usage ──",
        f"  alive       : {n_alive:5d} ({100*n_alive/N:5.1f}%)",
        f"  active      : {n_active:5d} ({100*n_active/N:5.1f}%)  (>1/100N)",
        f"  dominant    : {n_dominant:5d} ({100*n_dominant/N:5.1f}%)  (>10/N)",
        f"  max usage   : {float(usage.max()):.5f}",
        "",
        "── ||concept_ids|| ──",
        f"  mean        : {float(id_norm.mean()):.4f}",
        f"  std         : {float(id_norm.std()):.4f}",
        "",
        "── ||state_init|| ──",
        f"  mean        : {float(init_norm.mean()):.4f}",
        f"  std         : {float(init_norm.std()):.4f}",
        "",
        "── id ↔ state_init cosine ──",
        f"  mean        : {float(cos.mean()):+.4f}  (0 = orthogonal)",
        f"  median      : {float(cos.median()):+.4f}",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), ha="left", va="top",
            family="monospace", fontsize=11)


# ── main ────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wave", type=int, default=None)
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="ckpt path (default: outputs/wave{wave}/ckpt.pt)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output PNG (default: alongside the ckpt)")
    ap.add_argument("--dpi", type=int, default=180,
                    help="output DPI (default 180; bump for sharper text)")
    ap.add_argument("--figsize", type=str, default="24,14",
                    help='"W,H" inches. Default 24,14 → ~4300×2500 px at dpi=180')
    args = ap.parse_args()

    if args.wave is None and args.ckpt is None:
        # Auto-pick by most-recent log mtime.
        w1 = Path("outputs/wave1/training.log")
        w2 = Path("outputs/wave2/training.log")
        if w2.exists() and (not w1.exists()
                            or w2.stat().st_mtime > w1.stat().st_mtime):
            args.wave = 2
        else:
            args.wave = 1
    ckpt = args.ckpt or Path(f"outputs/wave{args.wave}/ckpt.pt")
    out = args.out or ckpt.parent / "viz_manifold.png"

    print(f"Loading manifold from {ckpt} ...")
    m = load_manifold(ckpt)

    W, H = (float(x) for x in args.figsize.split(","))
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22,
                          height_ratios=[1.0, 1.0])
    ax_ids = fig.add_subplot(gs[0, 0])
    ax_states = fig.add_subplot(gs[0, 1])
    ax_ring = fig.add_subplot(gs[0, 2])
    ax_drift = fig.add_subplot(gs[1, 0])
    ax_zipf = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[1, 2])

    usage_np = m["usage_ema"].numpy()
    print("Panel 1/6: UMAP(concept_ids) ...")
    panel_projection(ax_ids, m["concept_ids"].numpy(), usage_np,
                     title="concept_ids")
    print("Panel 2/6: UMAP(state_init) ...")
    panel_projection(ax_states, m["state_init"].numpy(), usage_np,
                     title="state_init")
    print("Panel 3/6: ring topology ...")
    panel_ring_topology(ax_ring, m)
    print("Panel 4/6: id↔state_init cosine histogram ...")
    panel_id_state_cosine(ax_drift, m)
    print("Panel 5/6: usage Zipf ...")
    panel_usage_zipf(ax_zipf, m)
    print("Panel 6/6: text summary ...")
    panel_text_summary(ax_text, m, ckpt)

    fig.suptitle(
        f"Tier 1 concept-manifold inspection  —  {ckpt}",
        fontsize=15,
    )
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved: {out}  ({int(W*args.dpi)}×{int(H*args.dpi)} px)")


if __name__ == "__main__":
    main()
