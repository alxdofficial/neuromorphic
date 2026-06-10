#!/usr/bin/env python3
"""Visualize Graph V8 memory state for one concrete input passage.

The script encodes either a generated EMAT-bio key/value sample or a user-provided
text passage, then writes:

  - coact_heatmaps.png: top-node coactivation matrices for source layers L0..L2
  - node_pca_edges.png: PCA layout of node keys/values with strongest coact edges
  - token_routes.png: top read-routed nodes for one token at the reader layers
  - graph_snapshot.json: machine-readable metadata and top route/edge tables

This intentionally uses PCA, not UMAP, so the script has no new dependency before
the smoke run. Use --project values to project values instead of keys.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.repr_learning.config import ReprConfig
from src.repr_learning.data_emat_bio import EMATBioDataset
from src.repr_learning.encoder import GraphV8ColumnEncoder


def load_encoder(cfg: ReprConfig, ckpt: Path | None, device: str) -> GraphV8ColumnEncoder:
    enc = GraphV8ColumnEncoder(cfg).to(device)
    enc.eval()
    if ckpt is not None:
        payload = torch.load(ckpt, map_location="cpu")
        sd = payload.get("model_state_dict", payload)
        enc_sd = {k.removeprefix("encoder."): v for k, v in sd.items()
                  if k.startswith("encoder.")}
        missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
        print(f"[ckpt] loaded encoder keys from {ckpt}")
        print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")
    return enc


def make_sample(args, tok, cfg: ReprConfig) -> dict:
    if args.text:
        ids = tok(args.text, add_special_tokens=False).input_ids[:args.chunk_size]
        valid = len(ids)
        ids = ids + [cfg.pad_token_id] * (args.chunk_size - valid)
        return {
            "context_ids": torch.tensor(ids, dtype=torch.long),
            "context_mask": torch.tensor([True] * valid + [False] * (args.chunk_size - valid),
                                         dtype=torch.bool),
            "question": "",
            "answer": "",
            "context_text": args.text,
        }

    ds = EMATBioDataset(
        tok,
        context_len=args.chunk_size,
        n_pairs=args.n_pairs,
        n_query=args.n_query,
        n_facts=args.n_facts,
        world_seed=args.world_seed,
        stream_seed=args.stream_seed,
        pad_token_id=cfg.pad_token_id,
    )
    sample = next(iter(ds))
    valid_ids = sample["context_ids"][sample["context_mask"]].tolist()
    return {
        "context_ids": sample["context_ids"],
        "context_mask": sample["context_mask"],
        "question": tok.decode(sample["question_ids"].tolist()),
        "answer": tok.decode(sample["answer_ids"].tolist()),
        "context_text": tok.decode(valid_ids),
    }


def encode_sample(enc: GraphV8ColumnEncoder, sample: dict, window_size: int, device: str):
    ids = sample["context_ids"].unsqueeze(0).to(device)
    mask = sample["context_mask"].unsqueeze(0).to(device)
    embed = enc.base.get_input_embeddings()
    with torch.no_grad():
        ctx_embeds = embed(ids)
        surprise = enc.context_surprise(ids, mask)
        state = enc.init_streaming_state(1, device, ctx_embeds.dtype)
        for start in range(0, ids.shape[1], window_size):
            end = min(start + window_size, ids.shape[1])
            state, _ = enc.streaming_write(
                state,
                ctx_embeds[:, start:end],
                mask[:, start:end],
                chunk_offset=start,
                surprise=surprise[:, start:end],
            )
        memory, aux = enc.finalize_memory(state)
    return ids, mask, ctx_embeds, state["sub"], memory, aux


def pca2(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def coact_top_nodes(co: np.ndarray, n: int) -> np.ndarray:
    score = co.sum(axis=0) + co.sum(axis=1)
    n = min(n, score.shape[0])
    return np.argsort(score)[-n:][::-1]


def edge_table(co: np.ndarray, nodes: np.ndarray, edge_top_k: int) -> list[dict]:
    sub = co[np.ix_(nodes, nodes)].copy()
    np.fill_diagonal(sub, 0.0)
    flat = sub.reshape(-1)
    if flat.size == 0 or flat.max() <= 0:
        return []
    k = min(edge_top_k, flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    out = []
    n = len(nodes)
    for f in idx:
        i, j = divmod(int(f), n)
        if flat[f] <= 0:
            continue
        out.append({"src": int(nodes[i]), "dst": int(nodes[j]), "weight": float(flat[f])})
    return out


def plot_coact_heatmaps(sub: dict, out_dir: Path, top_nodes: int) -> list[Path]:
    depth = len(sub["coact"])
    fig, axes = plt.subplots(1, depth, figsize=(5 * depth, 4.6), constrained_layout=True)
    if depth == 1:
        axes = [axes]
    written = []
    for src, ax in enumerate(axes):
        co = sub["coact"][src][0].detach().float().cpu().numpy()
        nodes = coact_top_nodes(co, top_nodes)
        mat = np.log1p(co[np.ix_(nodes, nodes)])
        im = ax.imshow(mat, cmap="magma", aspect="auto")
        ax.set_title(f"L{src} coactivation, top {len(nodes)} nodes")
        ax.set_xlabel("node rank by traffic")
        ax.set_ylabel("node rank by traffic")
        fig.colorbar(im, ax=ax, shrink=0.82, label="log1p(coact)")
    path = out_dir / "coact_heatmaps.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)
    return written


def plot_node_pca_edges(sub: dict, enc: GraphV8ColumnEncoder, out_dir: Path,
                        top_nodes: int, edge_top_k: int, project: str):
    depth = enc.sub.depth
    fig, axes = plt.subplots(1, depth, figsize=(5 * depth, 4.8), constrained_layout=True)
    if depth == 1:
        axes = [axes]
    edge_json = {}
    for src, ax in enumerate(axes):
        co = sub["coact"][src][0].detach().float().cpu().numpy()
        nodes = coact_top_nodes(co, top_nodes)
        if src == 0:
            bank = enc.sub.atom_values if project == "values" else enc.sub.atom_keys
            x = bank.detach().float().cpu().numpy()
        else:
            bank = sub["values"][src] if project == "values" else sub["keys"][src]
            x = bank[0].detach().float().cpu().numpy()
        coords = pca2(x)
        score = co.sum(axis=0) + co.sum(axis=1)
        score_sel = score[nodes]
        edges = edge_table(co, nodes, edge_top_k)
        edge_json[f"L{src}"] = edges
        c = np.log1p(score_sel)
        ax.scatter(coords[nodes, 0], coords[nodes, 1], c=c, cmap="viridis",
                   s=18, alpha=0.9, linewidths=0)
        max_w = max((e["weight"] for e in edges), default=1.0)
        for e in edges:
            a, b = e["src"], e["dst"]
            lw = 0.25 + 1.75 * math.sqrt(e["weight"] / max_w)
            ax.plot([coords[a, 0], coords[b, 0]], [coords[a, 1], coords[b, 1]],
                    color="#222222", alpha=0.18, linewidth=lw)
        ax.set_title(f"L{src} {project} PCA + strongest coact edges")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    path = out_dir / "node_pca_edges.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, edge_json


def token_routes(enc: GraphV8ColumnEncoder, ids: torch.Tensor, mask: torch.Tensor,
                 token_idx: int, device: str, top_k: int):
    valid = int(mask[0].sum().item())
    if token_idx < 0:
        token_idx = max(0, min(valid - 1, valid // 2))
    token_idx = max(0, min(token_idx, ids.shape[1] - 1))
    embed = enc.base.get_input_embeddings()
    with torch.no_grad():
        emb = embed(ids)
        out = enc.base.model(
            inputs_embeds=emb,
            attention_mask=mask.long(),
            output_hidden_states=True,
            use_cache=False,
        )
    return token_idx, out.hidden_states


def plot_token_routes(enc: GraphV8ColumnEncoder, sub: dict, hidden_states,
                      token_idx: int, tok, token_id: int, out_dir: Path, top_k: int):
    read_layers = tuple(enc.cfg.graph_v8_reader_layers)[1:]
    depth = enc.sub.depth
    fig, axes = plt.subplots(1, depth, figsize=(5 * depth, 4.2), constrained_layout=True)
    if depth == 1:
        axes = [axes]
    route_json = {}
    token_text = tok.decode([int(token_id)])
    for point, ax in enumerate(axes):
        layer = read_layers[point]
        keys = sub["keys"][point + 1]
        h = hidden_states[layer + 1][:, token_idx:token_idx + 1, :]
        with torch.no_grad():
            r = enc.sub.route(point + 1, h, keys)[0, 0].detach().float().cpu()
        vals, idx = torch.topk(r, k=min(top_k, r.numel()))
        x = np.arange(len(idx))
        ax.bar(x, vals.numpy(), color="#1f5aa6")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(i)) for i in idx], rotation=60, ha="right", fontsize=8)
        ax.set_title(f"token {token_idx} route at read L{point + 1}")
        ax.set_xlabel("node id")
        ax.set_ylabel("route weight")
        ax.grid(axis="y", alpha=0.22)
        route_json[f"L{point + 1}"] = [
            {"node": int(i), "weight": float(v)} for i, v in zip(idx, vals)
        ]
    fig.suptitle(f"Top routed nodes for token {token_idx}: {token_text!r}", fontsize=11)
    path = out_dir / "token_routes.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, route_json, token_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="Optional trained checkpoint; loads encoder.* keys only.")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/repr_learning/graph_v8_viz"))
    ap.add_argument("--text", type=str, default=None,
                    help="Optional raw input passage. Default generates one EMAT-bio sample.")
    ap.add_argument("--chunk-size", type=int, default=1024)
    ap.add_argument("--window-size", type=int, default=1024)
    ap.add_argument("--n-pairs", type=int, default=64)
    ap.add_argument("--n-query", type=int, default=1)
    ap.add_argument("--n-facts", type=int, default=3)
    ap.add_argument("--world-seed", type=int, default=0)
    ap.add_argument("--stream-seed", type=int, default=0)
    ap.add_argument("--token-index", type=int, default=-1,
                    help="-1 chooses the middle real token.")
    ap.add_argument("--top-nodes", type=int, default=96)
    ap.add_argument("--edge-top-k", type=int, default=240)
    ap.add_argument("--route-top-k", type=int, default=16)
    ap.add_argument("--project", choices=["keys", "values"], default="keys")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = ReprConfig(max_window_size=args.chunk_size, fixed_window_size=args.window_size)
    tok = AutoTokenizer.from_pretrained(cfg.llama_model)
    enc = load_encoder(cfg, args.ckpt, args.device)
    sample = make_sample(args, tok, cfg)
    ids, mask, _embeds, sub, memory, aux = encode_sample(enc, sample, args.window_size, args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    coact_paths = plot_coact_heatmaps(sub, args.out_dir, args.top_nodes)
    pca_path, edge_json = plot_node_pca_edges(
        sub, enc, args.out_dir, args.top_nodes, args.edge_top_k, args.project)
    token_idx, hidden_states = token_routes(enc, ids, mask, args.token_index,
                                            args.device, args.route_top_k)
    route_path, route_json, token_text = plot_token_routes(
        enc, sub, hidden_states, token_idx, tok, int(ids[0, token_idx].item()),
        args.out_dir, args.route_top_k)

    summary = {
        "graph_v8": {
            "n_layers": int(cfg.graph_v8_n_layers),
            "n_nodes": int(cfg.graph_v8_n_nodes),
            "d_mem": int(cfg.graph_v8_d_mem),
            "memory_shape": list(memory.shape),
            "final_read_floats": int(cfg.graph_v8_n_layers * cfg.graph_v8_n_nodes * 2 * cfg.graph_v8_d_mem),
        },
        "input": {
            "valid_tokens": int(mask[0].sum().item()),
            "token_index": int(token_idx),
            "token_text": token_text,
            "question": sample.get("question", ""),
            "answer": sample.get("answer", ""),
            "context_preview": sample.get("context_text", "")[:4000],
        },
        "routes": route_json,
        "edges": edge_json,
        "telemetry": {
            k: float(v.detach().cpu()) for k, v in aux.items()
            if torch.is_tensor(v) and v.numel() == 1
        },
        "files": [str(p) for p in [*coact_paths, pca_path, route_path]],
    }
    js = args.out_dir / "graph_snapshot.json"
    js.write_text(json.dumps(summary, indent=2))
    print(f"[viz] wrote {coact_paths[0]}")
    print(f"[viz] wrote {pca_path}")
    print(f"[viz] wrote {route_path}")
    print(f"[viz] wrote {js}")


if __name__ == "__main__":
    main()
