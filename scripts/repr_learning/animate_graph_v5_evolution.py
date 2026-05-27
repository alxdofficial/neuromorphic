#!/usr/bin/env python3
"""v5.1 graph evolution — interactive HTML animation.

Port of `animate_graph_evolution.py` to the v5 substrate. Loads the
v5.1-fair best.pt, streams one HotpotQA chunk through 4 windows, and
renders a self-contained plotly HTML with:

  - Per-frame: K_node bank slots positioned in 2D UMAP, sized by pick
    count and colored by hub rank. Edges drawn as arrows from each edge's
    src-argmax slot to its dst-argmax slot (thickness ∝ how many edges
    share that (src→dst) pair).
  - Trail: previous 2 frames' edges faded gray underneath.
  - Connectivity panel: # unique slots touched, cross-role overlap, and
    soft-pointer entropy over windows.

Output: docs/plots/v5_graph_evolution.html (self-contained, openable in
any browser, slider + play/pause built in).

Usage:
  python -m scripts.repr_learning.animate_graph_v5_evolution
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import AutoTokenizer

from src.repr_learning.config import ReprConfig
from src.repr_learning.encoder import GraphV5BaselineEncoder
from src.repr_learning.graph_substrate_v5 import materialize_endpoints
from src.repr_learning.data_qa import HotpotQADataset, collate_qa


ROOT = Path("/home/alex/code/neuromorphic")
CKPT = ROOT / "outputs/repr_learning/v5_4_first_graph_v5_baseline/ckpts/graph_v5_baseline.best.pt"
OUT = ROOT / "docs/plots/v5_4_graph_evolution.html"


def load_encoder(ckpt_path: Path) -> tuple[GraphV5BaselineEncoder, ReprConfig, int]:
    cfg = ReprConfig()
    enc = GraphV5BaselineEncoder(cfg)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc_state = {k.removeprefix("encoder."): v for k, v in sd["model_state_dict"].items()
                 if k.startswith("encoder.")}
    enc.load_state_dict(enc_state, strict=False)
    step = sd.get("step", -1)
    enc.train(False)
    return enc, cfg, step


def grab_sample(tokenizer, cfg: ReprConfig, chunk_size: int = 4096):
    ds = HotpotQADataset(
        tokenizer=tokenizer, split="validation",
        chunk_size=chunk_size,
        pad_token_id=tokenizer.pad_token_id or 128_001,
    )
    it = iter(ds)
    sample = next(it)
    batch = collate_qa([sample], pad_token_id=tokenizer.pad_token_id or 128_001)
    return batch


def stream_with_capture(enc, cfg, llama_embed, batch, device, window_size=1024):
    """Stream + capture (N, q_src, q_dst, edge state, materialized α) per window."""
    enc = enc.to(device)
    input_ids = batch.context_ids.to(device)
    attention_mask = batch.context_mask.to(device)

    frames = []
    with torch.no_grad():
        token_embeds = llama_embed(input_ids)
        T = token_embeds.shape[1]
        state = enc.init_streaming_state(1, device=device, dtype=token_embeds.dtype)

        # Capture chunk-init state (window -1, "before any write")
        frames.append(_snapshot(enc, state, window=-1))

        w = 0
        for s in range(0, T, window_size):
            e = min(s + window_size, T)
            state, _ = enc.streaming_write(
                state, token_embeds[:, s:e, :],
                attention_mask=attention_mask[:, s:e],
                chunk_offset=s,
            )
            frames.append(_snapshot(enc, state, window=w))
            w += 1
    return frames


def _snapshot(enc, state, window: int) -> dict:
    """Capture per-frame state needed for visualization."""
    N = state["N"]; q_src = state["q_src"]; q_dst = state["q_dst"]
    edge_state = state["state"]
    # v5.4: use the trained soft_pointer (W_k + learnable τ) — matches the
    # α the model actually uses internally. Falls back to stateless function
    # for older ckpts that pre-date soft_pointer.
    if hasattr(enc, "soft_pointer"):
        _, attn_src = enc.soft_pointer(q_src, N)
        _, attn_dst = enc.soft_pointer(q_dst, N)
    else:
        _, attn_src = materialize_endpoints(q_src, N, enc.read_temperature)
        _, attn_dst = materialize_endpoints(q_dst, N, enc.read_temperature)
    src_argmax = attn_src[0].argmax(dim=-1).cpu().numpy()
    dst_argmax = attn_dst[0].argmax(dim=-1).cpu().numpy()
    src_ent = (-(attn_src[0].clamp_min(1e-8) *
                  attn_src[0].clamp_min(1e-8).log()).sum(-1)).cpu().numpy()
    dst_ent = (-(attn_dst[0].clamp_min(1e-8) *
                  attn_dst[0].clamp_min(1e-8).log()).sum(-1)).cpu().numpy()
    return {
        "window": window,
        "N": N[0].float().cpu().numpy(),
        "edge_state": edge_state[0].float().cpu().numpy(),
        "src_argmax": src_argmax,
        "dst_argmax": dst_argmax,
        "src_ent": src_ent,
        "dst_ent": dst_ent,
    }


def umap_edge_state_to_hsv(frames, seed: int = 42) -> dict:
    """UMAP edge state vectors to 3D, then map (x, y, z) → (H, S, V) so each
    edge gets a perceptually-meaningful color reflecting its state content.

    Edges with similar relational content land near each other in UMAP space
    → similar HSV color. The UMAP is fit ACROSS ALL non-init frames so a
    given (state direction → color) mapping is consistent over time —
    watching the animation, you can track which "kind" of edges emerge where.

    Returns dict mapping fi → list of N rgb strings, one per edge.
    """
    import umap
    import matplotlib.colors as mcolors

    valid = [(fi, f) for fi, f in enumerate(frames) if f["window"] != -1]
    K_e = frames[0]["edge_state"].shape[0]
    if not valid:
        return {fi: ["rgb(128,128,128)"] * K_e for fi in range(len(frames))}

    stacked = np.concatenate([f["edge_state"] for _, f in valid], axis=0)
    stacked_n = stacked / (np.linalg.norm(stacked, axis=-1, keepdims=True) + 1e-6)
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.3,
                         metric="cosine", random_state=seed)
    emb = reducer.fit_transform(stacked_n)

    # Per-axis normalize to [0, 1]
    emb_min = emb.min(axis=0)
    emb_max = emb.max(axis=0)
    emb_norm = (emb - emb_min) / (emb_max - emb_min + 1e-9)

    # Map UMAP → HSV. Hue gets full range (most distinguishable axis).
    # Saturation + Value constrained to mid-high so colors stay vivid +
    # legible on a light background.
    h = emb_norm[:, 0]
    s = 0.55 + 0.35 * emb_norm[:, 1]   # 0.55 – 0.90
    v = 0.65 + 0.25 * emb_norm[:, 2]   # 0.65 – 0.90
    rgb = mcolors.hsv_to_rgb(np.stack([h, s, v], axis=-1))
    rgb_strs = [
        f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        for r, g, b in rgb
    ]

    out = {}
    offset = 0
    for fi, _ in valid:
        out[fi] = rgb_strs[offset:offset + K_e]
        offset += K_e
    for fi in range(len(frames)):
        if fi not in out:
            out[fi] = ["rgba(180,180,180,0.4)"] * K_e
    return out


def fit_global_pca(frames, n_components: int = 2):
    """Run PCA on union of N states across all frames.
    Linear, deterministic — every run gives the same layout for a given
    ckpt, no init-sensitivity. The fraction of variance captured by
    top-k PCs is data-dependent (typically 40-55% for k=3 in our regime).

    Honest geometric view: if PCA shows nodes clustered, they really are
    similar in the top-k variance directions. UMAP can fabricate clusters
    that don't exist in the original space.
    """
    all_N = []
    sizes = []
    for f in frames:
        all_N.append(f["N"])
        sizes.append(f["N"].shape[0])
    stacked = np.concatenate(all_N, axis=0)
    # Center the data — PCA assumes mean=0
    stacked_centered = stacked - stacked.mean(axis=0, keepdims=True)
    # SVD: stacked = U·diag(s)·Vt; top-k principal directions are rows of Vt
    _, _, Vt = np.linalg.svd(stacked_centered, full_matrices=False)
    emb = stacked_centered @ Vt[:n_components].T
    layouts = []
    offset = 0
    for sz in sizes:
        layouts.append(emb[offset:offset + sz])
        offset += sz
    return layouts


def fit_global_umap(frames, n_components: int = 2):
    """Run UMAP on union of N states across all frames for stable layout.
    n_components=2 → 2D layouts (default, animations work cleanly);
    n_components=3 → 3D (interactive rotation but plotly's Scatter3d
    doesn't smoothly transition between animation frames the way Scatter
    does, so animation is more "jump-cut" than fade).

    min_dist=0.8 (up from default 0.3): the trained bank vectors cluster
    moderately (cross_node_cos ~0.5–0.8 from training telemetry), so a
    lower min_dist would crunch them together in UMAP space. 0.8 forces
    more visible separation between distinct nodes.
    """
    import umap
    all_N = []
    sizes = []
    for f in frames:
        all_N.append(f["N"])
        sizes.append(f["N"].shape[0])
    stacked = np.concatenate(all_N, axis=0)
    stacked_n = stacked / (np.linalg.norm(stacked, axis=-1, keepdims=True) + 1e-6)
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.8, metric="cosine",
                        random_state=42, n_components=n_components)
    emb = reducer.fit_transform(stacked_n)
    layouts = []
    offset = 0
    for sz in sizes:
        layouts.append(emb[offset:offset + sz])
        offset += sz
    return layouts


def compute_frame_stats(frames, K_node):
    """Per-frame: # unique slots touched, cross-role overlap, mean entropy."""
    n_unique, cross_role, mean_ent_src, mean_ent_dst = [], [], [], []
    for f in frames:
        if f["src_argmax"] is None or f["window"] == -1:
            # Init frame — no edges materialized
            n_unique.append(0)
            cross_role.append(0)
            mean_ent_src.append(np.nan)
            mean_ent_dst.append(np.nan)
            continue
        src_set = set(f["src_argmax"].tolist())
        dst_set = set(f["dst_argmax"].tolist())
        n_unique.append(len(src_set | dst_set))
        cross_role.append(len(src_set & dst_set))
        mean_ent_src.append(float(f["src_ent"].mean()))
        mean_ent_dst.append(float(f["dst_ent"].mean()))
    return n_unique, cross_role, mean_ent_src, mean_ent_dst


def _wrap(text: str, width: int = 130) -> str:
    """Hand-wrap a long line into <br>-joined chunks for plotly annotations
    (which don't auto-wrap on width)."""
    import textwrap
    lines = textwrap.wrap(text, width=width, break_long_words=False,
                          replace_whitespace=False, drop_whitespace=False)
    return "<br>".join(lines)


def _decode_window_texts(batch, tokenizer, window_size: int, max_chars: int = 900) -> list[str]:
    """Decode the input tokens per window into readable text snippets.
    Length-cap each window's text to keep the HTML manageable. The init
    frame gets empty text (no window seen yet)."""
    ids = batch.context_ids[0].cpu().tolist()
    mask = batch.context_mask[0].cpu().tolist()
    T = len(ids)
    texts = [""]  # init frame placeholder
    for s in range(0, T, window_size):
        e = min(s + window_size, T)
        window_ids = [tid for tid, m in zip(ids[s:e], mask[s:e]) if m]
        if not window_ids:
            texts.append("[empty / all-padded window]")
            continue
        text = tokenizer.decode(window_ids, skip_special_tokens=True)
        text = text.replace("\n", " ").strip()
        if len(text) > max_chars:
            text = text[:max_chars] + " ... [truncated]"
        texts.append(_wrap(text, width=130))
    return texts



def build_animation(frames, K_node, K_edge, out_path: Path,
                    panels: list,
                    window_texts: Optional[list[str]] = None,
                    metric_axes: Optional[dict] = None,
                    metric_x_domain=(0.0, 1.0),
                    metric_y_domain=(0.04, 0.22),
                    height: int = 1100,
                    text_y_offset: float = -0.18,
                    text_width: int = 1100):
    """Render the per-window v5 graph evolution as an interactive HTML.

    `panels` is a list of dicts, one per graph subplot:
        {
            "title": str,                    # subplot title
            "dim": 2 or 3,                   # dimension
            "layouts": list[np.ndarray],     # per-frame [K_node, dim]
            "axis_kwargs": dict,             # trace axis binding, e.g. {"xaxis":"x","yaxis":"y"} or {"scene":"scene"}
            "x_domain": tuple[float, float], # paper-coord x extent
            "y_domain": tuple[float, float], # paper-coord y extent
        }

    `metric_axes` defaults to {"xaxis": "x2", "yaxis": "y2"} when 1 panel,
    or chosen based on panel count (next available axis IDs after panels).
    """
    F_n = len(frames)
    n_unique, cross_role, ent_src, ent_dst = compute_frame_stats(frames, K_node)
    max_ent = float(np.log(K_node))
    x_idx = list(range(F_n))
    frame_labels = ["init" if f["window"] == -1 else f"w{f['window']}" for f in frames]

    # Choose metric panel axis IDs based on which xy axes the panels used.
    # Count xy panels among panels; metric panel gets the next xy axis IDs.
    n_xy_panels = sum(1 for p in panels if p["dim"] == 2)
    if metric_axes is None:
        if n_xy_panels == 0:
            metric_axes = {"xaxis": "x", "yaxis": "y"}
        else:
            metric_axes = {"xaxis": f"x{n_xy_panels + 1}", "yaxis": f"y{n_xy_panels + 1}"}

    # Build a "naked" Figure (no make_subplots); we'll set axes/scenes by hand
    # via layout_kwargs. This is simpler than juggling make_subplots specs
    # for arbitrary panel counts.
    fig = go.Figure()

    # ── helpers ─────────────────────────────────────────────────────────
    def _metric_traces(current_fi):
        out = []
        ax = metric_axes
        out.append(go.Scatter(
            x=x_idx, y=n_unique, mode="lines+markers",
            name=f"# unique slots touched (max={K_node})",
            line=dict(color="#3cb44b", width=2), marker=dict(size=8),
            showlegend=True, legendgroup="metric_unique",
            **ax,
        ))
        out.append(go.Scatter(
            x=x_idx, y=cross_role, mode="lines+markers",
            name="# slots cross-role (both src AND dst)",
            line=dict(color="#e6194b", width=2),
            marker=dict(size=8, symbol="diamond"),
            showlegend=True, legendgroup="metric_xrole",
            **ax,
        ))
        out.append(go.Scatter(
            x=x_idx, y=ent_src, mode="lines+markers",
            name=f"mean src entropy (max={max_ent:.2f})",
            line=dict(color="#4363d8", width=2, dash="dash"),
            marker=dict(size=7),
            showlegend=True, legendgroup="metric_ent",
            **ax,
        ))
        y_top = max(K_node, max_ent + 0.5)
        out.append(go.Scatter(
            x=[current_fi, current_fi], y=[0, y_top], mode="lines",
            line=dict(color="gray", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
            **ax,
        ))
        return out

    def _graph_traces(panel, frame, edge_colors, pc, show_legend_nodes):
        """Emit graph traces for a single panel given a frame's state."""
        layout_arr = panel["layouts"][_frame_idx_lookup[frame_uid(frame)]]
        dim = panel["dim"]
        axis_kwargs = panel["axis_kwargs"]
        out_traces = []
        for ei in range(K_edge):
            if frame["src_argmax"] is None or frame["window"] == -1:
                if dim == 3:
                    out_traces.append(go.Scatter3d(
                        x=[], y=[], z=[], mode="lines",
                        line=dict(color=edge_colors[ei], width=2),
                        showlegend=False, hoverinfo="skip",
                        **axis_kwargs,
                    ))
                else:
                    out_traces.append(go.Scatter(
                        x=[], y=[], mode="lines",
                        line=dict(color=edge_colors[ei], width=1.5),
                        showlegend=False, hoverinfo="skip",
                        **axis_kwargs,
                    ))
                continue
            s = int(frame["src_argmax"][ei]); d = int(frame["dst_argmax"][ei])
            if dim == 3:
                x0, y0, z0 = layout_arr[s]; x1, y1, z1 = layout_arr[d]
                out_traces.append(go.Scatter3d(
                    x=[x0, x1], y=[y0, y1], z=[z0, z1], mode="lines",
                    line=dict(color=edge_colors[ei], width=3),
                    opacity=0.75, showlegend=False,
                    hovertext=f"edge {ei}: slot {s} → slot {d}",
                    hoverinfo="text",
                    **axis_kwargs,
                ))
            else:
                x0, y0 = layout_arr[s]; x1, y1 = layout_arr[d]
                out_traces.append(go.Scatter(
                    x=[x0, x1], y=[y0, y1], mode="lines",
                    line=dict(color=edge_colors[ei], width=1.5),
                    opacity=0.75, showlegend=False,
                    hovertext=f"edge {ei}: slot {s} → slot {d}",
                    hoverinfo="text",
                    **axis_kwargs,
                ))
        # bank nodes
        hover_texts = [
            f"node {k}<br>picks this window: {int(pc[k])}"
            for k in range(K_node)
        ]
        labels = [str(k) for k in range(K_node)]
        if dim == 3:
            out_traces.append(go.Scatter3d(
                x=layout_arr[:, 0], y=layout_arr[:, 1], z=layout_arr[:, 2],
                mode="markers+text",
                marker=dict(size=5, color="white",
                            line=dict(color="#222222", width=1)),
                text=labels, textposition="top center",
                textfont=dict(size=9, color="#222222", family="Arial"),
                name=f"bank node 0–{K_node - 1}", showlegend=show_legend_nodes,
                legendgroup="bank_node",
                hovertext=hover_texts, hoverinfo="text",
                **axis_kwargs,
            ))
        else:
            out_traces.append(go.Scatter(
                x=layout_arr[:, 0], y=layout_arr[:, 1], mode="markers+text",
                marker=dict(size=18, color="white", symbol="circle",
                            line=dict(color="#222222", width=1.2)),
                text=labels, textposition="middle center",
                textfont=dict(size=8, color="#222222", family="Arial"),
                name=f"bank node 0–{K_node - 1}", showlegend=show_legend_nodes,
                legendgroup="bank_node",
                hovertext=hover_texts, hoverinfo="text",
                **axis_kwargs,
            ))
        return out_traces

    # Each frame holds an ordered list of traces — metric panel + each panel's
    # graph traces. Order must be stable across frames so plotly's index-based
    # animation update works.
    def frame_uid(f): return ("init" if f["window"] == -1 else f["window"])
    _frame_idx_lookup = {frame_uid(f): i for i, f in enumerate(frames)}

    pick_counts_per_frame = []
    for f in frames:
        pc = np.zeros(K_node)
        if f["src_argmax"] is not None and f["window"] != -1:
            for k in f["src_argmax"].tolist() + f["dst_argmax"].tolist():
                pc[k] += 1
        pick_counts_per_frame.append(pc)
    edge_colors_per_frame = umap_edge_state_to_hsv(frames)

    plotly_frames = []
    for fi in range(F_n):
        traces = list(_metric_traces(fi))  # 4 metric traces
        f = frames[fi]
        pc = pick_counts_per_frame[fi]
        edge_colors = edge_colors_per_frame[fi]
        for pi, panel in enumerate(panels):
            # Show legend bank-node entry only on the FIRST panel to avoid duplicates.
            traces.extend(_graph_traces(panel, f, edge_colors, pc,
                                         show_legend_nodes=(pi == 0)))

        # Per-frame text annotation
        if window_texts is not None and fi < len(window_texts):
            text = window_texts[fi]
        else:
            text = ""
        text_anno = dict(
            name="window_text_panel",
            x=0.0, y=text_y_offset, xref="paper", yref="paper",
            xanchor="left", yanchor="top", showarrow=False,
            align="left",
            text=(f"<b>Window text:</b><br>{text}" if text
                  else "<b>Window text:</b> <i>[init — no tokens consumed yet]</i>"),
            font=dict(size=11, family="monospace", color="#222"),
            bgcolor="rgba(245,245,245,1.0)",
            bordercolor="#888", borderwidth=1,
            width=text_width,
        )
        plotly_frames.append(go.Frame(
            data=traces, name=frame_labels[fi],
            layout=go.Layout(
                title_text=_frame_title(fi, frames, n_unique, cross_role,
                                        ent_src, max_ent, K_node, K_edge),
                annotations=[text_anno],
            ),
        ))

    initial_frame_idx = 1 if F_n > 1 else 0
    fig.add_traces(plotly_frames[initial_frame_idx].data)
    fig.frames = plotly_frames

    steps = [dict(method="animate",
                   args=[[f.name], dict(mode="immediate",
                                         frame=dict(duration=800, redraw=True),
                                         transition=dict(duration=300))],
                   label=f.name) for f in plotly_frames]
    sliders = [dict(active=initial_frame_idx,
                     currentvalue=dict(prefix="frame: "),
                     pad=dict(t=40), steps=steps)]

    # Build layout: per-panel axis/scene domains, metric axes, title, sliders.
    layout_kwargs = dict(
        title=dict(
            text=_frame_title(initial_frame_idx, frames, n_unique, cross_role,
                                ent_src, max_ent, K_node, K_edge),
            x=0.02, xanchor="left",
        ),
        height=height,
        autosize=True,
        margin=dict(l=60, r=180, t=80, b=320),
        sliders=sliders,
        legend=dict(
            x=1.02, y=1.0, xanchor="left", yanchor="top",
            bgcolor="rgba(245,245,245,0.85)",
            bordercolor="#888", borderwidth=1,
            font=dict(size=10),
        ),
        updatemenus=[dict(
            type="buttons", direction="right", showactive=False,
            x=0.0, y=-0.10, xanchor="left",
            buttons=[dict(label="▶ Play", method="animate",
                          args=[None, dict(frame=dict(duration=900, redraw=True),
                                            transition=dict(duration=300),
                                            fromcurrent=True, mode="immediate")]),
                     dict(label="⏸ Pause", method="animate",
                          args=[[None], dict(frame=dict(duration=0, redraw=False),
                                              mode="immediate")])],
        )],
    )

    # Set per-panel axes/scenes. Track which xy axis number we're on.
    xy_counter = 0
    for panel in panels:
        if panel["dim"] == 2:
            xy_counter += 1
            ax_suffix = "" if xy_counter == 1 else str(xy_counter)
            layout_kwargs[f"xaxis{ax_suffix}"] = dict(
                domain=list(panel["x_domain"]),
                anchor=f"y{ax_suffix}",
                showgrid=False, zeroline=False,
                title=dict(text=panel.get("title"), font=dict(size=12)),
            )
            layout_kwargs[f"yaxis{ax_suffix}"] = dict(
                domain=list(panel["y_domain"]),
                anchor=f"x{ax_suffix}",
                showgrid=False, zeroline=False,
                scaleanchor=f"x{ax_suffix}",
            )
        else:  # 3D
            scene_name = panel["axis_kwargs"]["scene"]
            layout_kwargs[scene_name] = dict(
                domain=dict(x=list(panel["x_domain"]),
                            y=list(panel["y_domain"])),
                aspectmode="data",
                dragmode="orbit",
            )
            # Add subplot title as a paper-coord annotation since 3D scenes
            # don't have native titles in plotly.
            if panel.get("title"):
                layout_kwargs.setdefault("annotations", []).append(dict(
                    text=f"<b>{panel['title']}</b>",
                    x=(panel["x_domain"][0] + panel["x_domain"][1]) / 2,
                    y=panel["y_domain"][1] + 0.015,
                    xref="paper", yref="paper",
                    showarrow=False, font=dict(size=12),
                    xanchor="center", yanchor="bottom",
                ))

    # Metric panel axes. Trace-axis name like "x3" maps to layout property
    # "xaxis3" (with the digits appended after "axis"). For the base axis
    # "x"/"y", the layout key is "xaxis"/"yaxis" with no suffix.
    mx = metric_axes["xaxis"]; my = metric_axes["yaxis"]
    mx_layout_key = "xaxis" if mx == "x" else f"xaxis{mx[1:]}"
    my_layout_key = "yaxis" if my == "y" else f"yaxis{my[1:]}"
    layout_kwargs[mx_layout_key] = dict(
        domain=list(metric_x_domain),
        anchor=my,
        tickmode="array", tickvals=x_idx, ticktext=frame_labels,
        title_text="frame",
    )
    layout_kwargs[my_layout_key] = dict(
        domain=list(metric_y_domain),
        anchor=mx,
        title_text="count / entropy",
    )

    fig.update_layout(**layout_kwargs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    initial_frame_name = plotly_frames[initial_frame_idx].name
    post_script = f"""
    setTimeout(function() {{
      var div = document.getElementsByClassName('plotly-graph-div')[0];
      if (div && typeof Plotly !== 'undefined') {{
        Plotly.animate(div, ['{initial_frame_name}'],
          {{frame: {{duration: 0, redraw: true}}, mode: 'immediate',
            transition: {{duration: 0}}}});
      }}
    }}, 200);
    """
    fig.write_html(str(out_path), include_plotlyjs="cdn",
                    config={"responsive": True},
                    post_script=post_script)
    print(f"[output] wrote {out_path}")

    print(f"\n[per-frame stats]")
    print(f"  unique slots touched: {n_unique}")
    print(f"  cross-role overlap:   {cross_role}")
    print(f"  mean src entropy:     {[f'{e:.2f}' if not np.isnan(e) else 'init' for e in ent_src]}")
    print(f"  (max entropy = log({K_node}) = {max_ent:.2f})")



def _frame_title(fi, frames, n_unique, cross_role, ent_src, max_ent, K_node, K_edge):
    f = frames[fi]
    if f["window"] == -1:
        return (f"<b>Chunk init</b> — N sampled from learned (μ, σ); no writes yet • "
                f"K_node={K_node}, K_edge={K_edge}")
    return (f"<b>After window {f['window']}</b> — "
            f"{n_unique[fi]} unique slots touched ({100*n_unique[fi]/K_node:.0f}% of bank), "
            f"{cross_role[fi]} cross-role • "
            f"src entropy {ent_src[fi]:.2f}/{max_ent:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--window-size", type=int, default=1024)
    ap.add_argument("--layout", choices=["2d", "3d", "both", "quad"], default="quad",
                    help="2d: 2D UMAP only. 3d: 3D UMAP only. both: 2D+3D UMAP "
                         "side-by-side. quad (default): 2D UMAP | 3D UMAP on top "
                         "row, 2D PCA | 3D PCA on second row, shared metrics + text.")
    args = ap.parse_args()

    enc, cfg, step = load_encoder(args.ckpt)
    print(f"[ckpt] step={step}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.llama_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = grab_sample(tokenizer, cfg, args.chunk_size)

    from transformers import AutoModelForCausalLM
    llama = AutoModelForCausalLM.from_pretrained(cfg.llama_model, torch_dtype=torch.float32)
    llama.train(False)
    llama_embed = llama.get_input_embeddings()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    llama_embed = llama_embed.to(device)

    print(f"[forward] streaming chunk ({args.chunk_size // args.window_size} windows) on {device}")
    frames = stream_with_capture(enc, cfg, llama_embed, batch, device,
                                  window_size=args.window_size)

    # Fit the projection(s) we'll need.
    umap_2d = umap_3d = pca_2d = pca_3d = None
    if args.layout in ("2d", "both", "quad"):
        print("[layout] UMAP 2D")
        umap_2d = fit_global_umap(frames, n_components=2)
    if args.layout in ("3d", "both", "quad"):
        print("[layout] UMAP 3D")
        umap_3d = fit_global_umap(frames, n_components=3)
    if args.layout == "quad":
        print("[layout] PCA 2D + PCA 3D")
        pca_2d = fit_global_pca(frames, n_components=2)
        pca_3d = fit_global_pca(frames, n_components=3)

    # Decode per-window text for the side panel (init frame gets empty).
    window_texts = _decode_window_texts(batch, tokenizer, args.window_size)

    # Build the panels list — one per graph subplot. Each carries its own
    # axis binding and paper-coord domain. The metric panel and text
    # annotation are appended automatically by build_animation.
    panels: list[dict] = []
    height = 1100
    if args.layout == "2d":
        panels.append(dict(title="v5 graph (2D UMAP)", dim=2,
                            layouts=umap_2d,
                            axis_kwargs={"xaxis": "x", "yaxis": "y"},
                            x_domain=(0.0, 1.0), y_domain=(0.32, 0.98)))
    elif args.layout == "3d":
        panels.append(dict(title="v5 graph (3D UMAP)", dim=3,
                            layouts=umap_3d,
                            axis_kwargs={"scene": "scene"},
                            x_domain=(0.0, 1.0), y_domain=(0.32, 0.98)))
    elif args.layout == "both":
        panels.append(dict(title="v5 graph (2D UMAP)", dim=2,
                            layouts=umap_2d,
                            axis_kwargs={"xaxis": "x", "yaxis": "y"},
                            x_domain=(0.0, 0.46), y_domain=(0.32, 0.98)))
        panels.append(dict(title="v5 graph (3D UMAP)", dim=3,
                            layouts=umap_3d,
                            axis_kwargs={"scene": "scene"},
                            x_domain=(0.54, 1.0), y_domain=(0.32, 0.98)))
    elif args.layout == "quad":
        height = 1500
        # Row 1 (top): UMAP-2D | UMAP-3D, y in [0.66, 0.98]
        # Row 2 (mid): PCA-2D  | PCA-3D,  y in [0.30, 0.62]
        # Metric panel: y in [0.04, 0.24]
        panels.append(dict(title="v5 graph (2D UMAP)", dim=2,
                            layouts=umap_2d,
                            axis_kwargs={"xaxis": "x", "yaxis": "y"},
                            x_domain=(0.0, 0.46), y_domain=(0.66, 0.98)))
        panels.append(dict(title="v5 graph (3D UMAP)", dim=3,
                            layouts=umap_3d,
                            axis_kwargs={"scene": "scene"},
                            x_domain=(0.54, 1.0), y_domain=(0.66, 0.98)))
        panels.append(dict(title="v5 graph (2D PCA)", dim=2,
                            layouts=pca_2d,
                            axis_kwargs={"xaxis": "x2", "yaxis": "y2"},
                            x_domain=(0.0, 0.46), y_domain=(0.30, 0.62)))
        panels.append(dict(title="v5 graph (3D PCA)", dim=3,
                            layouts=pca_3d,
                            axis_kwargs={"scene": "scene2"},
                            x_domain=(0.54, 1.0), y_domain=(0.30, 0.62)))

    metric_y_domain = (0.04, 0.24) if args.layout == "quad" else (0.04, 0.22)
    build_animation(frames, cfg.graph_v5_K_node, cfg.graph_v5_K_edge, args.out,
                     panels=panels,
                     window_texts=window_texts,
                     height=height,
                     metric_y_domain=metric_y_domain)


if __name__ == "__main__":
    main()
