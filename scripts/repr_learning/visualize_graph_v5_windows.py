#!/usr/bin/env python3
"""v5.1 per-window graph visualization.

Loads v5.1-fair best.pt, streams one chunk through 4 windows, and draws
one PNG per window showing the actual GRAPH at that point: bank nodes
positioned by UMAP, edges drawn as arrows from each edge's src-argmax
slot to its dst-argmax slot.

A stable layout is computed by running UMAP on the UNION of N states
across all 4 windows + the chunk-start init — gives consistent node
positions across frames so you can see the graph evolve.

Usage:
  python -m scripts.repr_learning.visualize_graph_v5_windows
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from transformers import AutoTokenizer

from src.repr_learning.config import ReprConfig
from src.repr_learning.encoder import GraphV5BaselineEncoder
from src.repr_learning.graph_substrate_v5 import materialize_endpoints
from src.repr_learning.data_qa import HotpotQADataset, collate_qa


ROOT = Path("/home/alex/code/neuromorphic")
CKPT = ROOT / "outputs/repr_learning/v5_4_first_graph_v5_baseline/ckpts/graph_v5_baseline.best.pt"
OUT_DIR = ROOT / "docs/plots/v5_4_per_window"


def load_encoder(ckpt_path: Path) -> tuple[GraphV5BaselineEncoder, ReprConfig, int]:
    cfg = ReprConfig()
    enc = GraphV5BaselineEncoder(cfg)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc_state = {k.removeprefix("encoder."): v for k, v in sd["model_state_dict"].items()
                 if k.startswith("encoder.")}
    enc.load_state_dict(enc_state, strict=False)
    step = sd.get("step", -1)
    enc.train(False)
    print(f"[ckpt] step={step}")
    return enc, cfg, step


def get_one_chunk(tokenizer, chunk_size: int):
    ds = HotpotQADataset(
        tokenizer=tokenizer, split="validation",
        chunk_size=chunk_size,
        pad_token_id=tokenizer.pad_token_id or 128_001,
    )
    it = iter(ds)
    sample = next(it)
    batch = collate_qa([sample], pad_token_id=tokenizer.pad_token_id or 128_001)
    return batch, sample


def stream_with_capture(enc, cfg, llama_embed, batch, device, window_size=1024):
    """Stream the chunk, capturing N, q_src, q_dst, and materialized attn
    after every window. Returns list of (window_idx, state_snapshot)."""
    enc = enc.to(device)
    input_ids = batch.context_ids.to(device)
    attention_mask = batch.context_mask.to(device)

    snapshots = []
    with torch.no_grad():
        token_embeds = llama_embed(input_ids)                        # [1, T, d_llama]
        T = token_embeds.shape[1]
        state = enc.init_streaming_state(1, device=device, dtype=token_embeds.dtype)

        # Capture INITIAL state (window 0 = chunk-start, before any write)
        N0 = state["N"][0].float().cpu().numpy()
        qs0 = state["q_src"][0].float().cpu().numpy()
        qd0 = state["q_dst"][0].float().cpu().numpy()
        snapshots.append({
            "window": -1, "label": "chunk init",
            "N": N0, "q_src": qs0, "q_dst": qd0,
            "src_argmax": None, "dst_argmax": None,
            "attn_src": None, "attn_dst": None,
        })

        w = 0
        for s in range(0, T, window_size):
            e = min(s + window_size, T)
            state, _ = enc.streaming_write(
                state, token_embeds[:, s:e, :],
                attention_mask=attention_mask[:, s:e],
                chunk_offset=s,
            )
            # Materialize at this window's state.
            # v5.4: use the trained soft_pointer (W_k + learnable τ) for accurate
            # α matching what the model uses internally. The stateless
            # materialize_endpoints would diverge from W_k-projected scoring.
            N = state["N"]; q_src = state["q_src"]; q_dst = state["q_dst"]
            if hasattr(enc, "soft_pointer"):
                _, attn_src = enc.soft_pointer(q_src, N)
                _, attn_dst = enc.soft_pointer(q_dst, N)
            else:
                _, attn_src = materialize_endpoints(q_src, N, enc.read_temperature)
                _, attn_dst = materialize_endpoints(q_dst, N, enc.read_temperature)
            snapshots.append({
                "window": w, "label": f"after window {w}",
                "N": N[0].float().cpu().numpy(),
                "q_src": q_src[0].float().cpu().numpy(),
                "q_dst": q_dst[0].float().cpu().numpy(),
                "src_argmax": attn_src[0].argmax(dim=-1).cpu().numpy(),
                "dst_argmax": attn_dst[0].argmax(dim=-1).cpu().numpy(),
                "attn_src": attn_src[0].float().cpu().numpy(),
                "attn_dst": attn_dst[0].float().cpu().numpy(),
            })
            w += 1
    return snapshots


def compute_stable_layout(snapshots):
    """Run UMAP on the UNION of N states across all snapshots so bank
    node positions are consistent across frames. Returns dict mapping
    snapshot index → [K_node, 2] coords."""
    import umap
    all_N = []
    sizes = []
    for s in snapshots:
        all_N.append(s["N"])
        sizes.append(s["N"].shape[0])
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


def draw_frame(snap, layout, out_path, K_node, K_edge, top1_color_by="hub"):
    """Draw one window's graph: bank nodes sized by pick count, edges
    drawn as arrows src_slot → dst_slot."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Pick counts per slot (for sizing nodes)
    if snap["src_argmax"] is not None:
        pick_counts = np.zeros(K_node)
        for k in snap["src_argmax"].tolist() + snap["dst_argmax"].tolist():
            pick_counts[k] += 1
    else:
        pick_counts = np.zeros(K_node)

    # Color bank nodes by hub rank (most-picked = red, idle = light gray)
    sort_idx = np.argsort(pick_counts)[::-1]
    rank = np.zeros(K_node, dtype=int)
    for r, idx in enumerate(sort_idx):
        rank[idx] = r
    # Red gradient for top 10, gray for rest
    colors = []
    for k in range(K_node):
        if pick_counts[k] == 0:
            colors.append("#e0e0e0")
        elif rank[k] < 5:
            t = rank[k] / 5
            colors.append((1.0, 0.2 + 0.5 * t, 0.2 + 0.5 * t))
        elif rank[k] < 15:
            colors.append("#88aacc")
        else:
            colors.append("#cccccc")

    # Draw edges first (so nodes draw on top)
    if snap["src_argmax"] is not None:
        # Group edges by (src, dst) pair to bundle with line thickness
        edge_pairs = {}
        for src_slot, dst_slot in zip(snap["src_argmax"], snap["dst_argmax"]):
            key = (int(src_slot), int(dst_slot))
            edge_pairs[key] = edge_pairs.get(key, 0) + 1

        # Draw each pair, thickness ∝ # edges
        max_count = max(edge_pairs.values()) if edge_pairs else 1
        for (s, d), count in edge_pairs.items():
            x0, y0 = layout[s]
            x1, y1 = layout[d]
            if s == d:
                # Self-loop — draw small circle
                ax.add_patch(plt.Circle((x0, y0 + 0.15), 0.1,
                                          fill=False, edgecolor="purple",
                                          alpha=0.6, lw=1 + 2 * count / max_count))
            else:
                width = 0.5 + 3 * count / max_count
                alpha = 0.3 + 0.5 * count / max_count
                arrow = FancyArrowPatch(
                    (x0, y0), (x1, y1),
                    arrowstyle="-|>", mutation_scale=12,
                    color="#444444", alpha=alpha, lw=width,
                    connectionstyle="arc3,rad=0.1",
                )
                ax.add_patch(arrow)

    # Draw bank nodes
    sizes = 100 + (pick_counts / max(pick_counts.max(), 1)) * 800
    ax.scatter(layout[:, 0], layout[:, 1], s=sizes, c=colors,
                edgecolors="black", linewidths=1.5, zorder=3)
    # Annotate hub nodes
    for k in sort_idx[:5]:
        if pick_counts[k] > 0:
            ax.annotate(f"{k}\n({int(pick_counts[k])})",
                         xy=layout[k], ha="center", va="center",
                         fontsize=8, weight="bold", zorder=4)

    # Title + legend
    if snap["window"] == -1:
        title = f"Chunk init — N sampled from learned (μ, σ); no writes yet"
    else:
        n_edges_drawn = K_edge
        n_unique_nodes = len(set(snap["src_argmax"].tolist() + snap["dst_argmax"].tolist()))
        cross_role = len(set(snap["src_argmax"].tolist()) & set(snap["dst_argmax"].tolist()))
        title = (
            f"After window {snap['window']}: {K_edge} edges → "
            f"{n_unique_nodes} unique slots ({100*n_unique_nodes/K_node:.0f}% of bank), "
            f"{cross_role} slots cross-role"
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal")

    # Custom legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="red", edgecolor="black", label="top-5 hub nodes"),
        Patch(facecolor="#88aacc", edgecolor="black", label="warm nodes (rank 6-15)"),
        Patch(facecolor="#cccccc", edgecolor="black", label="cold nodes"),
        Patch(facecolor="#e0e0e0", edgecolor="black", label="unused (0 picks)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[output] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--window-size", type=int, default=1024)
    args = ap.parse_args()

    enc, cfg, step = load_encoder(args.ckpt)

    tokenizer = AutoTokenizer.from_pretrained(cfg.llama_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch, sample = get_one_chunk(tokenizer, args.chunk_size)

    from transformers import AutoModelForCausalLM
    llama = AutoModelForCausalLM.from_pretrained(cfg.llama_model, torch_dtype=torch.float32)
    llama.train(False)
    llama_embed = llama.get_input_embeddings()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    llama_embed = llama_embed.to(device)

    print(f"[forward] streaming chunk in {args.chunk_size // args.window_size} windows on {device}")
    snapshots = stream_with_capture(enc, cfg, llama_embed, batch, device,
                                     window_size=args.window_size)

    print(f"[layout] computing UMAP on N union across {len(snapshots)} snapshots")
    layouts = compute_stable_layout(snapshots)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, (snap, layout) in enumerate(zip(snapshots, layouts)):
        name = f"v5_window_{snap['window']:+d}.png" if snap["window"] >= 0 else "v5_window_init.png"
        out_path = args.out_dir / name
        draw_frame(snap, layout, out_path, cfg.graph_v5_K_node, cfg.graph_v5_K_edge)

    print(f"\nDone. {len(snapshots)} PNGs in {args.out_dir}")


if __name__ == "__main__":
    main()
