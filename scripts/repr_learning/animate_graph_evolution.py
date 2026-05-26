#!/usr/bin/env python3
"""Animate graph_baseline edge evolution over one HotpotQA sample.

Loads the v1h_t4k_v3 LB-fixed graph checkpoint, runs a single chunk
forward through 4 streaming windows, captures per-window edge state,
projects all 136 endpoints across all frames into shared 2D UMAP, and
renders an interactive plotly HTML with:

  - Per-frame: 68 src + 68 dst nodes as colored markers (k-means cluster
    on state vector), 68 line segments connecting src->dst, line opacity
    = update_alpha proxy (post-hoc: cos delta vs previous frame),
    halo on markers sized by u (saliency).
  - Trail: previous 3 frames drawn faded in gray underneath.
  - Overwrites: edges whose endpoints jumped > 0.5 cos distance flash red.
  - Connectivity panel: per-frame # connected components + largest
    component size, computed by union-find with cos > 0.7 endpoint
    similarity threshold.

Output: docs/plots/graph_evolution.html (self-contained, openable in any
browser, slider + play/pause built-in).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import umap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from transformers import AutoTokenizer

from src.repr_learning.config import ReprConfig
from src.repr_learning.encoder import GraphBaselineEncoder
from src.repr_learning.data_qa import HotpotQADataset, collate_qa


ROOT = Path("/home/alex/code/neuromorphic")
CKPT = ROOT / "outputs/repr_learning/v1h_t4k_v3_lb_graph_baseline/ckpts/graph_baseline.best.pt"
OUT = ROOT / "docs/plots/graph_evolution.html"


def load_encoder(ckpt_path: Path) -> tuple[GraphBaselineEncoder, ReprConfig, int]:
    cfg = ReprConfig()
    cfg.n_flat_codes = 36
    cfg.d_mamba = 768
    enc = GraphBaselineEncoder(cfg)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc_state = {k.removeprefix("encoder."): v for k, v in sd["model_state_dict"].items()
                 if k.startswith("encoder.")}
    missing, unexpected = enc.load_state_dict(enc_state, strict=False)
    print(f"[ckpt] step={sd.get('step')} missing={len(missing)} unexpected={len(unexpected)}")
    enc.train(False)
    return enc, cfg, int(sd.get("step", -1))


def grab_sample(tokenizer, cfg: ReprConfig, seed: int = 7, skip: int = 0):
    """Pull one HotpotQA val sample and tokenize as a single-chunk batch."""
    ds = HotpotQADataset(
        split="validation", tokenizer=tokenizer, chunk_size=4096,
        sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id, seed=seed,
    )
    it = iter(ds)
    sample = None
    for _ in range(skip + 1):
        sample = next(it)
    batch = collate_qa([sample], pad_token_id=cfg.pad_token_id)
    return batch


def run_chunk(enc, batch, window_size: int = 1024) -> list[dict]:
    """Forward one chunk window-by-window, returning per-frame edge state."""
    with torch.no_grad():
        embed = enc  # GraphBaselineEncoder doesn't own llama; use pin_encoder dtype
        # Use Llama embed table? Not loaded here. Use a frozen random projection
        # of token IDs into d_llama space as a placeholder. To stay faithful to
        # the trained model, we instead load the Llama embedding lazily.
        from transformers import AutoModel
        # Lighter: load Llama embed_tokens only.
        llama_embed = _load_llama_embed(enc.cfg.d_llama)
        ctx_embeds = llama_embed(batch.context_ids)
    B, T_ctx = batch.context_ids.shape
    device = ctx_embeds.device
    n_windows = (T_ctx + window_size - 1) // window_size

    state = enc.init_streaming_state(B, device, ctx_embeds.dtype)
    frames = [_snapshot_state(state, window_idx=-1)]  # frame 0 = init
    with torch.no_grad():
        for w in range(n_windows):
            s = w * window_size
            e = min(s + window_size, T_ctx)
            state, _ = enc.streaming_write(
                state, ctx_embeds[:, s:e, :], batch.context_mask[:, s:e],
                chunk_offset=s,
            )
            frames.append(_snapshot_state(state, window_idx=w))
    print(f"[forward] captured {len(frames)} frames ({n_windows} writes + 1 init)")
    return frames


_LLAMA_EMBED = None
def _load_llama_embed(d_llama: int):
    global _LLAMA_EMBED
    if _LLAMA_EMBED is None:
        print("[llama] loading embed_tokens only (no full model)")
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B", torch_dtype=torch.float32,
        )
        _LLAMA_EMBED = m.get_input_embeddings()
        del m
    return _LLAMA_EMBED


def _snapshot_state(state, window_idx: int) -> dict:
    e = state["edges"]
    def _np(t):
        return t.detach().float().cpu().numpy()
    return {
        "window": window_idx,
        "src": _np(e["src"][0]),    # [K, d_node]
        "dst": _np(e["dst"][0]),
        "state": _np(e["state"][0]),
        "u": _np(e["u"][0]),         # [K]
        "age": _np(e["age"][0]),     # [K]
    }


def detect_overwrites(frames: list[dict], cos_jump_thresh: float = 0.5) -> list[np.ndarray]:
    """For each frame after the first, mark slots whose src or dst jumped
    by cosine distance > thresh from the previous frame.

    Returns: list of bool arrays [K], one per frame (frame 0 is all False).
    """
    overwrites = [np.zeros(frames[0]["src"].shape[0], dtype=bool)]
    for i in range(1, len(frames)):
        prev, cur = frames[i - 1], frames[i]
        cos_src = _row_cos(prev["src"], cur["src"])
        cos_dst = _row_cos(prev["dst"], cur["dst"])
        ow = (cos_src < (1.0 - cos_jump_thresh)) | (cos_dst < (1.0 - cos_jump_thresh))
        overwrites.append(ow)
    return overwrites


def _row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (an * bn).sum(axis=1)


def connected_components(src: np.ndarray, dst: np.ndarray,
                          cos_thresh: float = 0.7) -> tuple[int, int, np.ndarray]:
    """Cluster endpoints by cosine; chain edges; return component count,
    largest size, and per-edge component-id array."""
    K = src.shape[0]
    endpoints = np.concatenate([src, dst], axis=0)              # [2K, d]
    en = endpoints / (np.linalg.norm(endpoints, axis=1, keepdims=True) + 1e-8)
    sim = en @ en.T
    # Union-find
    parent = list(range(2 * K))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y);
        if rx != ry: parent[rx] = ry
    # Endpoints same node if cos > thresh
    for i in range(2 * K):
        for j in range(i + 1, 2 * K):
            if sim[i, j] > cos_thresh:
                union(i, j)
    # Slot edges: src_i (idx i) and dst_i (idx K+i) connected
    for i in range(K):
        union(i, K + i)
    roots = [find(i) for i in range(2 * K)]
    unique_roots = set(roots)
    sizes = {r: roots.count(r) for r in unique_roots}
    largest = max(sizes.values())
    # Per-edge component: edge i's component = component of its src
    edge_comp = np.array([roots[i] for i in range(K)])
    # Relabel roots to 0..n-1 for color stability
    root_to_id = {r: idx for idx, r in enumerate(sorted(unique_roots))}
    edge_comp = np.array([root_to_id[r] for r in edge_comp])
    return len(unique_roots), largest, edge_comp


def _effective_rank(X: np.ndarray, variance_frac: float = 0.95) -> int:
    """Min # of SVD components needed to explain `variance_frac` of variance."""
    Xc = X - X.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    var = (s ** 2)
    cumfrac = np.cumsum(var) / var.sum()
    return int(np.searchsorted(cumfrac, variance_frac) + 1)


def fit_global_umap(frames: list[dict], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fit UMAP on all endpoints across all frames. Returns:
       src_xy [F, K, 2], dst_xy [F, K, 2]."""
    K = frames[0]["src"].shape[0]
    all_pts = np.concatenate(
        [np.concatenate([f["src"], f["dst"]], axis=0) for f in frames], axis=0,
    )                                                            # [F * 2K, d]
    print(f"[umap] fitting on {all_pts.shape[0]} endpoints (dim={all_pts.shape[1]})")
    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.25, metric="cosine",
        random_state=seed,
    )
    xy = reducer.fit_transform(all_pts)                          # [F * 2K, 2]
    F_n = len(frames)
    xy = xy.reshape(F_n, 2 * K, 2)
    return xy[:, :K, :], xy[:, K:, :]


def cluster_states(frames: list[dict], n_clusters: int = 8, seed: int = 0) -> np.ndarray:
    """K-means on stacked state vectors. Returns labels [F, K]."""
    K = frames[0]["state"].shape[0]
    all_states = np.concatenate([f["state"] for f in frames], axis=0)   # [F * K, d]
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=4)
    labels = km.fit_predict(all_states)
    return labels.reshape(len(frames), K)


def build_animation(frames, src_xy, dst_xy, state_labels, overwrites,
                    out_path: Path, n_trail: int = 3):
    K = frames[0]["src"].shape[0]
    F_n = len(frames)

    # Connectivity time series — tight thresholds (endpoints are densely
    # packed; sub-0.99 collapses everything to one component, masking
    # structure). Also report SVD-based "effective number of directions"
    # which is threshold-free.
    thresholds = [0.97, 0.99, 0.999]
    series = {t: {"n_comp": [], "largest": []} for t in thresholds}
    eff_rank_series = []
    for f in frames:
        for t in thresholds:
            nc, lg, _ = connected_components(f["src"], f["dst"], cos_thresh=t)
            series[t]["n_comp"].append(nc)
            series[t]["largest"].append(lg)
        eff_rank_series.append(_effective_rank(
            np.concatenate([f["src"], f["dst"]], axis=0), variance_frac=0.95,
        ))
    n_comp_series = series[0.99]["n_comp"]
    largest_series = series[0.99]["largest"]

    # Cluster palette (8 hues)
    palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                "#42d4f4", "#f032e6", "#bfef45"]

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.78, 0.22],
        vertical_spacing=0.08,
        subplot_titles=("Graph edges in 2D UMAP", "Connectivity over windows"),
    )

    # ── Connectivity traces (static, all frames) ──
    # Three tight thresholds + threshold-free effective rank (SVD@95%).
    thresh_colors = {0.97: "#bbbbbb", 0.99: "#4363d8", 0.999: "#e6194b"}
    for t in thresholds:
        fig.add_trace(go.Scatter(
            x=list(range(F_n)), y=series[t]["n_comp"], mode="lines+markers",
            name=f"# components @ cos>{t}",
            line=dict(color=thresh_colors[t], width=2),
            marker=dict(size=7),
        ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=list(range(F_n)), y=eff_rank_series, mode="lines+markers",
        name="effective rank @ 95% var",
        line=dict(color="#3cb44b", width=2, dash="dash"),
        marker=dict(size=8, symbol="square"),
    ), row=2, col=1)
    fig.update_xaxes(title_text="window (0=init, 1..4=writes)", row=2, col=1,
                     tickmode="array", tickvals=list(range(F_n)))
    fig.update_yaxes(title_text="# distinct nodes (lower = more connected)",
                      row=2, col=1)

    # ── Frame builder ──
    n_static = len(thresholds) + 1   # +1 for effective-rank line
    plotly_frames = []
    for fi in range(F_n):
        traces = []
        # Trail (previous n_trail frames, faded gray)
        for back in range(1, n_trail + 1):
            ti = fi - back
            if ti < 0: continue
            opacity = 0.15 * (1.0 - back / (n_trail + 1))
            edge_xs, edge_ys = [], []
            for k in range(K):
                edge_xs += [src_xy[ti, k, 0], dst_xy[ti, k, 0], None]
                edge_ys += [src_xy[ti, k, 1], dst_xy[ti, k, 1], None]
            traces.append(go.Scatter(
                x=edge_xs, y=edge_ys, mode="lines",
                line=dict(color="rgba(120,120,120,1)", width=1),
                opacity=opacity, showlegend=False, hoverinfo="skip",
                xaxis="x", yaxis="y",
            ))

        # Current edges — split into normal vs overwritten
        ow = overwrites[fi]
        labels = state_labels[fi]
        u = frames[fi]["u"]
        # One line trace per cluster color (for legibility); one combined for overwrites
        for cluster_id in range(state_labels.max() + 1):
            mask = (labels == cluster_id) & (~ow)
            if not mask.any(): continue
            edge_xs, edge_ys = [], []
            for k in np.where(mask)[0]:
                edge_xs += [src_xy[fi, k, 0], dst_xy[fi, k, 0], None]
                edge_ys += [src_xy[fi, k, 1], dst_xy[fi, k, 1], None]
            traces.append(go.Scatter(
                x=edge_xs, y=edge_ys, mode="lines",
                line=dict(color=palette[cluster_id % len(palette)], width=1.5),
                opacity=0.85, showlegend=False,
                hoverinfo="skip", xaxis="x", yaxis="y",
            ))
        # Overwritten edges — red, thicker
        if ow.any():
            edge_xs, edge_ys = [], []
            for k in np.where(ow)[0]:
                edge_xs += [src_xy[fi, k, 0], dst_xy[fi, k, 0], None]
                edge_ys += [src_xy[fi, k, 1], dst_xy[fi, k, 1], None]
            traces.append(go.Scatter(
                x=edge_xs, y=edge_ys, mode="lines",
                line=dict(color="#ff1744", width=3),
                opacity=1.0, name="overwritten", showlegend=(fi == 1),
                hoverinfo="skip", xaxis="x", yaxis="y",
            ))

        # Endpoint markers — size by u (halo)
        u_norm = (u - u.min()) / max(u.max() - u.min(), 1e-6)
        marker_size = 8 + u_norm * 18   # 8..26
        marker_colors = [palette[c % len(palette)] for c in labels]
        traces.append(go.Scatter(
            x=src_xy[fi, :, 0], y=src_xy[fi, :, 1], mode="markers",
            marker=dict(size=marker_size, color=marker_colors,
                        symbol="circle", line=dict(color="black", width=0.8),
                        opacity=0.95),
            name="src", showlegend=(fi == 0),
            hovertext=[f"slot {k}<br>u={u[k]:.3f}<br>cluster={labels[k]}"
                       for k in range(K)],
            hoverinfo="text", xaxis="x", yaxis="y",
        ))
        traces.append(go.Scatter(
            x=dst_xy[fi, :, 0], y=dst_xy[fi, :, 1], mode="markers",
            marker=dict(size=marker_size, color=marker_colors,
                        symbol="diamond", line=dict(color="black", width=0.8),
                        opacity=0.85),
            name="dst", showlegend=(fi == 0),
            hovertext=[f"slot {k}<br>u={u[k]:.3f}<br>cluster={labels[k]}"
                       for k in range(K)],
            hoverinfo="text", xaxis="x", yaxis="y",
        ))

        # Connectivity marker (vertical line at current frame, on bottom panel)
        y_max = max(max(max(series[t]["n_comp"]) for t in thresholds),
                     max(eff_rank_series))
        traces.append(go.Scatter(
            x=[fi, fi], y=[0, y_max],
            mode="lines", line=dict(color="rgba(0,0,0,0.4)", width=2, dash="dot"),
            showlegend=False, hoverinfo="skip", xaxis="x2", yaxis="y2",
        ))

        plotly_frames.append(go.Frame(
            data=traces, name=f"w{fi}",
            layout=go.Layout(title_text=_frame_title(fi, frames, n_comp_series, largest_series)),
        ))

    # Initial frame
    fig.add_traces(plotly_frames[0].data)
    fig.frames = plotly_frames

    # Slider + play button
    steps = [dict(method="animate",
                   args=[[f.name], dict(mode="immediate",
                                         frame=dict(duration=800, redraw=True),
                                         transition=dict(duration=300))],
                   label=f.name) for f in plotly_frames]
    sliders = [dict(active=0, currentvalue=dict(prefix="frame: "),
                     pad=dict(t=40), steps=steps)]
    fig.update_layout(
        title=_frame_title(0, frames, n_comp_series, largest_series),
        height=900, width=1100,
        sliders=sliders,
        updatemenus=[dict(
            type="buttons", direction="right", showactive=False, x=0.0, y=-0.12,
            buttons=[dict(label="▶ Play", method="animate",
                          args=[None, dict(frame=dict(duration=900, redraw=True),
                                            transition=dict(duration=300),
                                            fromcurrent=True, mode="immediate")]),
                     dict(label="⏸ Pause", method="animate",
                          args=[[None], dict(frame=dict(duration=0, redraw=False),
                                              mode="immediate")])],
        )],
    )
    fig.update_xaxes(showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, scaleanchor="x", row=1, col=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"[output] wrote {out_path}")
    for t in thresholds:
        print(f"[connectivity cos>{t}] components:  {series[t]['n_comp']}"
              f"   largest: {series[t]['largest']}")
    print(f"[effective rank @ 95% var]: {eff_rank_series}  "
          f"(out of {2 * frames[0]['src'].shape[0]} endpoints; lower = more clustered)")


def _frame_title(fi, frames, n_comp, largest):
    win = frames[fi]["window"]
    label = "init" if win < 0 else f"after write {win}"
    return (f"Graph evolution — {label} (frame {fi}/{len(frames)-1})"
            f" — components={n_comp[fi]}, largest={largest[fi]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip", type=int, default=0,
                    help="skip N HotpotQA val samples before grabbing one")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    enc, cfg, step = load_encoder(Path(args.ckpt))
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tok.pad_token_id is None:
        tok.pad_token_id = cfg.pad_token_id
    batch = grab_sample(tok, cfg, seed=args.seed, skip=args.skip)
    print(f"[sample] ctx={batch.context_ids.shape} q={batch.question_ids.shape}"
          f" a={batch.answer_ids.shape}")

    frames = run_chunk(enc, batch, window_size=1024)
    src_xy, dst_xy = fit_global_umap(frames, seed=args.seed)
    state_labels = cluster_states(frames, n_clusters=8, seed=args.seed)
    overwrites = detect_overwrites(frames, cos_jump_thresh=0.5)

    build_animation(frames, src_xy, dst_xy, state_labels, overwrites,
                    Path(args.out), n_trail=3)


if __name__ == "__main__":
    main()
