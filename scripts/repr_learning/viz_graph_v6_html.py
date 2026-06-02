#!/usr/bin/env python3
"""Pretty (Plotly) version of the graph_v6 write-structure viz + headless screenshot.

Reuses the proven extraction in probe_graph_v6_viz.py (node bank → 2D, soft-pointer
edges, per-node usage) but renders it as a styled interactive Plotly figure (HTML) and
screenshots a static PNG via headless chromium (playwright). Two biographical contexts
side-by-side so the *generic-across-inputs* topology is visible.

Outputs:
  docs/plots/graph_v6_structure.html   (interactive)
  docs/plots/graph_v6_structure_pretty.png  (screenshot, for the doc)
  docs/plots/graph_v6_structure.npz    (raw arrays, re-render without GPU)

Usage: python scripts/repr_learning/viz_graph_v6_html.py [--tag v6_1] [--n-ctx 2]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from transformers import AutoTokenizer

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.repr_learning.config import ReprConfig
import scripts.repr_learning.eval_per_family as E
import scripts.repr_learning.probe_graph_v6_viz as P


def extract(model, sample, dev):
    enc = model.encoder
    st = P.stream_to_state(model, sample, dev)
    N = st["N"][0].float()
    with torch.no_grad():
        spk, spv = enc.read_pointer.project_kv(st["N"])
        _, a_src = enc.read_pointer.attend(st["q_src"], spk, spv)
        _, a_dst = enc.read_pointer.attend(st["q_dst"], spk, spv)
    src_n = a_src[0].argmax(-1).cpu()
    dst_n = a_dst[0].argmax(-1).cpu()
    Kn = N.shape[0]
    usage = torch.zeros(Kn)
    for v in torch.cat([src_n, dst_n]):
        usage[v] += 1
    used = int((usage > 0).sum())
    p = usage / usage.sum().clamp_min(1)
    eff = float((-(p[p > 0] * p[p > 0].log()).sum()).exp())
    Nn = torch.nn.functional.normalize(N, dim=-1)
    offcos = float((Nn @ Nn.t())[~torch.eye(Kn, dtype=torch.bool)].mean())
    distinct = len({(int(a), int(b)) for a, b in zip(src_n.tolist(), dst_n.tolist())})
    XY = P.project2d(N.cpu().numpy())
    return dict(XY=np.asarray(XY), usage=usage.numpy(), src=src_n.numpy(),
                dst=dst_n.numpy(), Kn=Kn, used=used, eff=eff, offcos=offcos,
                distinct=distinct, n_edges=int(src_n.shape[0]))


def build_fig(ctxs):
    titles = [f"context {i}: {c['used']}/{c['Kn']} nodes used · "
              f"eff≈{c['eff']:.0f} · {c['distinct']}/{c['n_edges']} distinct edges · "
              f"collapse cos={c['offcos']:.2f}" for i, c in enumerate(ctxs)]
    fig = make_subplots(rows=1, cols=len(ctxs), subplot_titles=titles,
                        horizontal_spacing=0.06)
    for ci, c in enumerate(ctxs, start=1):
        XY, usage, src, dst = c["XY"], c["usage"], c["src"], c["dst"]
        ex, ey = [], []
        for a, b in zip(src, dst):
            ex += [XY[a, 0], XY[b, 0], None]
            ey += [XY[a, 1], XY[b, 1], None]
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                                 line=dict(color="rgba(120,120,140,0.18)", width=1),
                                 hoverinfo="skip", showlegend=False), row=1, col=ci)
        fig.add_trace(go.Scatter(
            x=XY[:, 0], y=XY[:, 1], mode="markers",
            marker=dict(size=6 + 3.2 * usage, color=usage, colorscale="Viridis",
                        line=dict(color="rgba(20,20,30,0.6)", width=0.7),
                        colorbar=dict(title="edge<br>endpoints", thickness=12,
                                      len=0.8) if ci == len(ctxs) else None,
                        showscale=(ci == len(ctxs))),
            text=[f"node {i}<br>{int(u)} endpoints" for i, u in enumerate(usage)],
            hoverinfo="text", showlegend=False), row=1, col=ci)
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=ci)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False,
                         scaleanchor=f"x{ci if ci > 1 else ''}", row=1, col=ci)
    fig.update_layout(
        template="plotly_white",
        title=dict(text="graph_v6 write structure — node bank + soft-pointer edges "
                        "(two different biographies → the same generic topology)",
                   x=0.5, font=dict(size=16)),
        margin=dict(l=20, r=20, t=90, b=20), width=1480, height=720,
        paper_bgcolor="white", plot_bgcolor="white",
    )
    for ann in fig.layout.annotations:
        ann.font.size = 11
    return fig


def screenshot(html_path, png_path):
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        print(f"[skip screenshot] playwright unavailable: {e}")
        return False
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 1500, "height": 760}, device_scale_factor=2)
        pg.goto(f"file://{html_path}")
        pg.wait_for_timeout(1800)
        pg.screenshot(path=str(png_path))
        b.close()
    print(f"[output] wrote {png_path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v6_1")
    ap.add_argument("--n-ctx", type=int, default=2)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ReprConfig(fixed_window_size=1024, max_window_size=8192)
    tok = AutoTokenizer.from_pretrained(cfg.llama_model)
    ck = ROOT / f"outputs/repr_learning/{args.tag}_graph_v6_baseline/ckpts/graph_v6_baseline.best.pt"
    model, _ = E.load_variant("graph_v6_baseline", ck, cfg, None)
    model = model.to(dev)
    samples = E.collect_samples(["biographical"], args.n_ctx, tokenizer=tok, cfg=cfg,
                                chunk_size=8192, passages_per_chunk=600)
    ctxs = [extract(model, s, dev) for s in samples]
    for i, c in enumerate(ctxs):
        print(f"ctx {i}: nodes_used={c['used']}/{c['Kn']} eff={c['eff']:.1f} "
              f"distinct_edges={c['distinct']}/{c['n_edges']} collapse_cos={c['offcos']:.3f}")
    outdir = ROOT / "docs/plots"
    np.savez(outdir / "graph_v6_structure.npz",
             **{f"ctx{i}_{k}": v for i, c in enumerate(ctxs)
                for k, v in c.items() if isinstance(v, np.ndarray)})
    fig = build_fig(ctxs)
    html = outdir / "graph_v6_structure.html"
    fig.write_html(str(html), include_plotlyjs="cdn")
    print(f"[output] wrote {html}")
    screenshot(html, outdir / "graph_v6_structure_pretty.png")


if __name__ == "__main__":
    main()
