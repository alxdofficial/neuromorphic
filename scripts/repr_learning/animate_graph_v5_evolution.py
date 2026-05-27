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


def fit_global_umap(frames):
    """Run UMAP on union of N states across all frames for stable layout."""
    import umap
    all_N = []
    sizes = []
    for f in frames:
        all_N.append(f["N"])
        sizes.append(f["N"].shape[0])
    stacked = np.concatenate(all_N, axis=0)
    stacked_n = stacked / (np.linalg.norm(stacked, axis=-1, keepdims=True) + 1e-6)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine",
                        random_state=42, n_components=2)
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


def build_animation(frames, layouts, K_node, K_edge, out_path: Path,
                    window_texts: Optional[list[str]] = None, n_trail: int = 2):
    F_n = len(frames)
    n_unique, cross_role, ent_src, ent_dst = compute_frame_stats(frames, K_node)
    max_ent = float(np.log(K_node))

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.78, 0.22],
        vertical_spacing=0.08,
        subplot_titles=("v5 graph: bank nodes + edges in 2D UMAP",
                         "Per-window structure metrics"),
    )

    # Per-frame metric traces are included in every frame's data (NOT added
    # statically) so that plotly's animation — which replaces fig.data by
    # index — doesn't wipe them when stepping between frames.
    x_idx = list(range(F_n))
    frame_labels = ["init" if f["window"] == -1 else f"w{f['window']}" for f in frames]
    fig.update_xaxes(title_text="frame", row=2, col=1,
                     tickmode="array", tickvals=x_idx, ticktext=frame_labels)
    fig.update_yaxes(title_text="count / entropy", row=2, col=1)

    def _bottom_panel_traces(current_fi):
        """Three metric lines + a vertical marker at current frame.
        Legend entries are always-shown (legendgroup dedupes across frames)."""
        out = []
        out.append(go.Scatter(
            x=x_idx, y=n_unique, mode="lines+markers",
            name=f"# unique slots touched (max={K_node})",
            line=dict(color="#3cb44b", width=2), marker=dict(size=8),
            showlegend=True, legendgroup="metric_unique",
            xaxis="x2", yaxis="y2",
        ))
        out.append(go.Scatter(
            x=x_idx, y=cross_role, mode="lines+markers",
            name="# slots cross-role (both src AND dst)",
            line=dict(color="#e6194b", width=2),
            marker=dict(size=8, symbol="diamond"),
            showlegend=True, legendgroup="metric_xrole",
            xaxis="x2", yaxis="y2",
        ))
        out.append(go.Scatter(
            x=x_idx, y=ent_src, mode="lines+markers",
            name=f"mean src entropy (max={max_ent:.2f})",
            line=dict(color="#4363d8", width=2, dash="dash"),
            marker=dict(size=7),
            showlegend=True, legendgroup="metric_ent",
            xaxis="x2", yaxis="y2",
        ))
        y_top = max(K_node, max_ent + 0.5)
        out.append(go.Scatter(
            x=[current_fi, current_fi], y=[0, y_top],
            mode="lines",
            line=dict(color="rgba(0,0,0,0.4)", width=2, dash="dot"),
            showlegend=False, hoverinfo="skip",
            xaxis="x2", yaxis="y2",
        ))
        return out

    # Compute pick counts per frame (used for hover only — node sizing/coloring
    # is uniform now since edge density already conveys popularity).
    pick_counts_per_frame = []
    for f in frames:
        pc = np.zeros(K_node)
        if f["src_argmax"] is not None and f["window"] != -1:
            for k in f["src_argmax"].tolist() + f["dst_argmax"].tolist():
                pc[k] += 1
        pick_counts_per_frame.append(pc)

    # Per-edge color from UMAP-3D(edge_state) → HSV. Each edge's color
    # reflects its state-vector content; edges with similar relational
    # content get similar colors (consistent across frames).
    edge_colors_per_frame = umap_edge_state_to_hsv(frames)

    # ── Frame builder ──
    # Each frame includes ALL traces (top-panel viz + bottom-panel metrics).
    # plotly's animation replaces fig.data by index when stepping between
    # frames — include metric traces per-frame to persist. Per-edge colors
    # mean we emit K_edge edge traces (one each), giving each its own color.
    plotly_frames = []
    for fi in range(F_n):
        traces = list(_bottom_panel_traces(fi))   # 4 traces: 3 metric lines + vertical marker
        f = frames[fi]
        layout = layouts[fi]
        pc = pick_counts_per_frame[fi]
        edge_colors = edge_colors_per_frame[fi]

        # One trace per edge (so each gets its own UMAP→HSV color).
        # Self-loops drawn as a tiny circle at the slot position.
        for ei in range(K_edge):
            if f["src_argmax"] is None or f["window"] == -1:
                # init frame — emit empty placeholder to keep trace-index stable
                traces.append(go.Scatter(
                    x=[], y=[], mode="lines",
                    line=dict(color=edge_colors[ei], width=1.5),
                    showlegend=False, hoverinfo="skip",
                    xaxis="x", yaxis="y",
                ))
                continue
            s = int(f["src_argmax"][ei]); d = int(f["dst_argmax"][ei])
            x0, y0 = layout[s]; x1, y1 = layout[d]
            traces.append(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(color=edge_colors[ei], width=1.8),
                opacity=0.75,
                showlegend=False,
                hovertext=f"edge {ei}: slot {s} → slot {d}",
                hoverinfo="text",
                xaxis="x", yaxis="y",
            ))

        # Bank nodes — circles with persistent numeric labels (0..K_node-1).
        # The number is purely positional ID, not semantic content (slot
        # identities are arbitrary per-chunk). Same k in window 1 vs window 4
        # lets you watch how that specific node drifts through semantic
        # (UMAP) space as its content evolves window-to-window.
        hover_texts = [
            f"node {k}<br>picks this window: {int(pc[k])}"
            for k in range(K_node)
        ]
        labels = [str(k) for k in range(K_node)]
        traces.append(go.Scatter(
            x=layout[:, 0], y=layout[:, 1], mode="markers+text",
            marker=dict(size=24, color="white",
                        symbol="circle",
                        line=dict(color="#222222", width=1.5)),
            text=labels, textposition="middle center",
            textfont=dict(size=9, color="#222222", family="Arial"),
            name=f"bank node 0–{K_node - 1}", showlegend=True,
            legendgroup="bank_node",
            hovertext=hover_texts, hoverinfo="text",
            xaxis="x", yaxis="y",
        ))

        # Per-frame text annotation: shows the actual tokens consumed in
        # this window. Sits below the metric panel so users can visually
        # correlate "this passage came in → these nodes/edges shifted."
        if window_texts is not None and fi < len(window_texts):
            text = window_texts[fi]
        else:
            text = ""
        text_anno = dict(
            name="window_text_panel",  # named so plotly replaces (not stacks) across frames
            x=0.0, y=-0.18, xref="paper", yref="paper",
            xanchor="left", yanchor="top", showarrow=False,
            align="left",
            text=(f"<b>Window text:</b><br>{text}" if text
                  else "<b>Window text:</b> <i>[init — no tokens consumed yet]</i>"),
            font=dict(size=11, family="monospace", color="#222"),
            bgcolor="rgba(245,245,245,1.0)",  # full opacity to prevent blend with stale frames
            bordercolor="#888", borderwidth=1,
            width=1100,
        )
        plotly_frames.append(go.Frame(
            data=traces, name=frame_labels[fi],
            layout=go.Layout(
                title_text=_frame_title(fi, frames, n_unique, cross_role,
                                        ent_src, max_ent, K_node, K_edge),
                annotations=[text_anno],
            ),
        ))

    # Default the initial visible frame to w0 (window 0, not chunk init) so
    # the user sees edges immediately on load. Init frame is still in the
    # animation, just not the landing frame.
    initial_frame_idx = 1 if F_n > 1 else 0
    fig.add_traces(plotly_frames[initial_frame_idx].data)
    fig.frames = plotly_frames
    # Initial annotation = the initial frame's annotation. Per-frame
    # annotations from frame.layout replace this on slider/play because
    # plotly merges layout dicts by KEY, and we use the same "annotations"
    # list-key everywhere, so the most recent layout's annotations list wins.
    # (Earlier attempts that left annotations in both static AND frame layouts
    # caused STACKING — two text panels visible per frame, the static one
    # stuck at w0's text.)
    initial_annos = list(plotly_frames[initial_frame_idx].layout.annotations or ())

    # 300ms transition gives smooth node/edge interpolation between frames.
    # (The "opacity blend on text" bug from an earlier session was actually
    # a static-layout-annotation stacking issue, not a transition issue —
    # fixed separately by removing the static annotation. Annotations
    # don't fade through transitions; they just replace on the next frame.)
    steps = [dict(method="animate",
                   args=[[f.name], dict(mode="immediate",
                                         frame=dict(duration=800, redraw=True),
                                         transition=dict(duration=300))],
                   label=f.name) for f in plotly_frames]
    sliders = [dict(active=initial_frame_idx,
                     currentvalue=dict(prefix="frame: "),
                     pad=dict(t=40), steps=steps)]

    # Responsive layout: NO fixed width. Title carries the key context.
    # Plotly's native legend (right side, INSIDE the figure) replaces the
    # external paper-coordinate annotation that was getting cropped off.
    fig.update_layout(
        title=dict(
            text=_frame_title(initial_frame_idx, frames, n_unique, cross_role,
                                ent_src, max_ent, K_node, K_edge),
            x=0.02, xanchor="left",
        ),
        height=1050,
        autosize=True,
        margin=dict(l=60, r=180, t=80, b=320),
        # No annotations in the static layout — plotly's animation merge
        # APPENDS frame.layout.annotations rather than replacing, so a static
        # text panel would stack on top of each frame's text panel. The
        # post_script below fires Plotly.animate on load to render the
        # initial frame's text immediately (no static annotation needed).
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
    fig.update_xaxes(showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, scaleanchor="x", row=1, col=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # responsive=True → fills container, useful when the HTML is loaded into
    # a wider viewport than the default 700px plotly fallback.
    # post_script: fire Plotly.animate to initial frame on load so the
    # initial render shows the initial frame's text annotation. Without
    # this the page loads blank-text until the user moves the slider.
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

    # Console summary
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
    print(f"[layout] global UMAP over {len(frames)} snapshots")
    layouts = fit_global_umap(frames)

    # Decode per-window text for the side panel (init frame gets empty).
    window_texts = _decode_window_texts(batch, tokenizer, args.window_size)

    build_animation(frames, layouts, cfg.graph_v5_K_node, cfg.graph_v5_K_edge,
                     args.out, window_texts=window_texts)


if __name__ == "__main__":
    main()
