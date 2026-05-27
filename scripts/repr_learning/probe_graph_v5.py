#!/usr/bin/env python3
"""v5.1 graph_v5_baseline topology probe.

Loads the v5.1-fair checkpoint, runs it on a batch of real HotpotQA
chunks, and answers: IS GRAPH TOPOLOGY ACTUALLY FORMING?

With K_node=32 bank entries and K_edge=60 edges (120 endpoint picks
must resolve to 32 nodes — pigeonhole forces ~3.75 picks per node):

  1. Bank usage IS concentrated on a subset of "hub" nodes (not uniform,
     not collapsed to one). Healthy: 15-25 nodes do most of the work.
  2. Cross-edge sharing IS happening — pairs of edges materialize src/dst
     endpoints onto the SAME bank entry (cos ≈ 1).
  3. Cross-role overlap IS happening — same N[k] appears as src for some
     edges and dst for others. Structurally impossible in v4.
  4. Soft pointers are not maximally uniform.

Outputs docs/plots/v5_topology_diagnostic.png + console summary.

Usage:
  python -m scripts.repr_learning.probe_graph_v5 --batch-size 8
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
from transformers import AutoTokenizer

from src.repr_learning.config import ReprConfig
from src.repr_learning.encoder import GraphV5BaselineEncoder
from src.repr_learning.graph_substrate_v5 import materialize_endpoints
from src.repr_learning.data_qa import HotpotQADataset, collate_qa


ROOT = Path("/home/alex/code/neuromorphic")
CKPT = ROOT / "outputs/repr_learning/v5_1_fair_graph_v5_baseline/ckpts/graph_v5_baseline.best.pt"
OUT = ROOT / "docs/plots/v5_topology_diagnostic.png"


def load_encoder(ckpt_path: Path) -> tuple[GraphV5BaselineEncoder, ReprConfig, int]:
    cfg = ReprConfig()
    enc = GraphV5BaselineEncoder(cfg)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc_state = {k.removeprefix("encoder."): v for k, v in sd["model_state_dict"].items()
                 if k.startswith("encoder.")}
    missing, unexpected = enc.load_state_dict(enc_state, strict=False)
    step = sd.get("step", -1)
    print(f"[ckpt] step={step} | missing={len(missing)} | unexpected={len(unexpected)}")
    if missing:
        print(f"  missing examples: {missing[:5]}")
    enc.train(False)
    return enc, cfg, step


def get_batch(tokenizer, cfg, batch_size: int, chunk_size: int):
    ds = HotpotQADataset(
        tokenizer=tokenizer, split="validation",
        chunk_size=chunk_size,
        pad_token_id=tokenizer.pad_token_id or 128_001,
    )
    it = iter(ds)
    samples = [next(it) for _ in range(batch_size)]
    return collate_qa(samples, pad_token_id=tokenizer.pad_token_id or 128_001)


def run_forward(enc, cfg, llama_embed, batch, device):
    """Stream the chunk through enc, then materialize endpoints + grab N."""
    enc = enc.to(device)
    # QABatch dataclass: context_ids / context_mask are the chunk fields
    input_ids = batch.context_ids.to(device)
    attention_mask = batch.context_mask.to(device)

    with torch.no_grad():
        token_embeds = llama_embed(input_ids)                        # [B, T, d_llama]
        T = token_embeds.shape[1]
        W = 1024
        state = enc.init_streaming_state(token_embeds.shape[0],
                                          device=device, dtype=token_embeds.dtype)
        for s in range(0, T, W):
            e = min(s + W, T)
            state, _ = enc.streaming_write(
                state, token_embeds[:, s:e, :],
                attention_mask=attention_mask[:, s:e],
                chunk_offset=s,
            )

    N = state["N"]
    q_src = state["q_src"]
    q_dst = state["q_dst"]
    edge_state = state["state"]

    # Materialize at the encoder's configured temperature
    temp = enc.read_temperature
    endpoint_src, attn_src = materialize_endpoints(q_src, N, temperature=temp)
    endpoint_dst, attn_dst = materialize_endpoints(q_dst, N, temperature=temp)
    return {
        "N": N, "q_src": q_src, "q_dst": q_dst, "state": edge_state,
        "endpoint_src": endpoint_src, "endpoint_dst": endpoint_dst,
        "attn_src": attn_src, "attn_dst": attn_dst,
    }


def analyze(probe_out, K_node, K_edge):
    """Print numerical summary + return arrays for plotting."""
    attn_src = probe_out["attn_src"].float().cpu()                   # [B, K_edge, K_node]
    attn_dst = probe_out["attn_dst"].float().cpu()
    B = attn_src.shape[0]

    # ── Bank usage ──────────────────────────────────────────────────────
    src_argmax = attn_src.argmax(dim=-1)                              # [B, K_edge]
    dst_argmax = attn_dst.argmax(dim=-1)
    all_picks = torch.cat([src_argmax, dst_argmax], dim=-1)           # [B, 2K_edge]

    pick_counts = torch.zeros(K_node)
    for b in range(B):
        for k in all_picks[b].tolist():
            pick_counts[k] += 1
    pick_counts = pick_counts / B                                     # avg per chunk

    n_active = (pick_counts > 0.5).int().sum().item()
    top5_share = (pick_counts.sort(descending=True).values[:5].sum() /
                  pick_counts.sum().clamp_min(1)).item()
    print(f"\n── Bank usage (K_node={K_node}) ──")
    print(f"  Slots that receive >=1 pick on average: {n_active} of {K_node} ({100*n_active/K_node:.0f}%)")
    print(f"  Top-5 slots claim {100*top5_share:.0f}% of all picks  (1.0 = monopoly; {100*5/K_node:.0f}% = uniform)")
    print(f"  Pick-count: min={pick_counts.min():.1f}, max={pick_counts.max():.1f}, mean={pick_counts.mean():.1f}, std={pick_counts.std():.1f}")

    # ── Cross-role overlap ──────────────────────────────────────────────
    cross_role = []
    for b in range(B):
        src_set = set(src_argmax[b].tolist())
        dst_set = set(dst_argmax[b].tolist())
        overlap = len(src_set & dst_set)
        cross_role.append(overlap)
    cr = np.array(cross_role)
    print(f"\n── Cross-role overlap (slots that serve BOTH src AND dst in same chunk) ──")
    print(f"  Per-chunk mean: {cr.mean():.1f} of {K_node} slots ({100*cr.mean()/K_node:.0f}%)")
    print(f"  Per-chunk range: {cr.min()} to {cr.max()}")

    # ── Soft-pointer entropy ────────────────────────────────────────────
    def _ent(p):
        return -(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum(-1)
    src_ent = _ent(attn_src).numpy().flatten()
    dst_ent = _ent(attn_dst).numpy().flatten()
    max_ent = float(np.log(K_node))
    print(f"\n── Soft-pointer entropy ──")
    print(f"  max possible: log({K_node}) = {max_ent:.3f}")
    print(f"  src entropy: mean={src_ent.mean():.3f}, min={src_ent.min():.3f}, max={src_ent.max():.3f}")
    print(f"  dst entropy: mean={dst_ent.mean():.3f}, min={dst_ent.min():.3f}, max={dst_ent.max():.3f}")
    frac_sharp = ((src_ent < max_ent * 0.5).sum() + (dst_ent < max_ent * 0.5).sum()) / (len(src_ent) + len(dst_ent))
    print(f"  Fraction below 50% of max entropy: {frac_sharp:.2f}  (sharp = good)")

    # ── Cross-edge endpoint sharing ──────────────────────────────────────
    endpoint_src = probe_out["endpoint_src"].float().cpu()
    endpoint_dst = probe_out["endpoint_dst"].float().cpu()
    ep_bank = torch.cat([endpoint_src, endpoint_dst], dim=1)          # [B, 2K_edge, d_node]
    ep_n = F.normalize(ep_bank, dim=-1, eps=1e-6)
    cos_mat = ep_n @ ep_n.transpose(-1, -2)
    K2 = ep_bank.shape[1]
    off_diag = ~torch.eye(K2, dtype=torch.bool)
    cos_off = cos_mat[:, off_diag].numpy()
    print(f"\n── Cross-edge endpoint sharing (cos between materialized endpoints) ──")
    print(f"  Mean cos: {cos_off.mean():.3f}, max: {cos_off.max():.3f}")
    print(f"  Pairs with cos > 0.9: {100*(cos_off > 0.9).mean():.1f}%  (would be 0% in v4)")
    print(f"  Pairs with cos > 0.99: {100*(cos_off > 0.99).mean():.2f}%  (exact reuse)")

    # ── Effective reuse ──────────────────────────────────────────────────
    edge_reuse = []
    for b in range(B):
        slot_counter = {}
        for k in src_argmax[b].tolist() + dst_argmax[b].tolist():
            slot_counter[k] = slot_counter.get(k, 0) + 1
        n_shared = sum(c for c in slot_counter.values() if c >= 2)
        edge_reuse.append(n_shared / (2 * K_edge))
    print(f"\n── Effective reuse (fraction of picks on slots picked >=2 times) ──")
    print(f"  Per-chunk mean: {np.mean(edge_reuse):.2f}  (1.0 = full reuse)")

    return {
        "pick_counts": pick_counts.numpy(),
        "cross_role": cr,
        "src_ent": src_ent, "dst_ent": dst_ent,
        "cos_off_flat": cos_off,
        "cos_mat_first": cos_mat[0].numpy(),
        "src_argmax_first": src_argmax[0].numpy(),
        "dst_argmax_first": dst_argmax[0].numpy(),
        "N_first": probe_out["N"][0].float().cpu().numpy(),
        "endpoint_src_first": endpoint_src[0].numpy(),
        "endpoint_dst_first": endpoint_dst[0].numpy(),
        "max_ent": max_ent,
        "K_node": K_node,
        "K_edge": K_edge,
    }


def plot(ana, out_path: Path, step: int):
    K_node = ana["K_node"]
    K_edge = ana["K_edge"]
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    # Panel 1: Bank usage histogram
    ax = fig.add_subplot(gs[0, 0])
    sorted_counts = np.sort(ana["pick_counts"])[::-1]
    ax.bar(range(K_node), sorted_counts, color="steelblue", edgecolor="navy")
    ax.set_xlabel(f"Bank slot (sorted, K_node={K_node})")
    ax.set_ylabel("Avg picks per chunk")
    ax.set_title("Bank usage — hubs vs idle slots")
    uniform = (2 * K_edge) / K_node
    ax.axhline(uniform, color="red", ls="--", lw=1, label=f"uniform={uniform:.1f}")
    ax.legend()

    # Panel 2: Soft-pointer entropy
    ax = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, ana["max_ent"] * 1.05, 30)
    ax.hist(ana["src_ent"], bins=bins, alpha=0.6, label="src", color="tab:blue")
    ax.hist(ana["dst_ent"], bins=bins, alpha=0.6, label="dst", color="tab:orange")
    ax.axvline(ana["max_ent"], color="red", ls="--", label=f"max=log({K_node})")
    ax.axvline(ana["max_ent"] * 0.5, color="green", ls=":", label="half-max (sharp)")
    ax.set_xlabel("Soft-pointer entropy")
    ax.set_ylabel("# (edge, role) pairs")
    ax.set_title("Pointer sharpness distribution")
    ax.legend(fontsize=8)

    # Panel 3: Cross-role overlap
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(ana["cross_role"], bins=range(0, K_node + 2), color="darkgreen",
             edgecolor="black", alpha=0.7)
    ax.set_xlabel("# slots used as BOTH src AND dst (same chunk)")
    ax.set_ylabel("# chunks in batch")
    ax.set_title(f"Cross-role overlap (mean={ana['cross_role'].mean():.1f}/{K_node})")
    ax.axvline(ana["cross_role"].mean(), color="red", ls="--",
                label=f"mean={ana['cross_role'].mean():.1f}")
    ax.legend()

    # Panel 4: Cross-edge endpoint cosine heatmap (first chunk)
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(ana["cos_mat_first"], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xlabel(f"endpoint idx (first {K_edge}=src, next {K_edge}=dst)")
    ax.set_ylabel("endpoint idx")
    ax.set_title("Cross-edge endpoint cos (chunk 0)")
    ax.axhline(K_edge, color="black", lw=0.5)
    ax.axvline(K_edge, color="black", lw=0.5)
    plt.colorbar(im, ax=ax, shrink=0.7)

    # Panel 5: Endpoint cos distribution
    ax = fig.add_subplot(gs[1, 1])
    flat = ana["cos_off_flat"].flatten()
    ax.hist(flat, bins=80, color="purple", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Pairwise cos between materialized endpoints")
    ax.set_ylabel("# pairs")
    ax.set_title(f"Endpoint cosine distribution\nmean={flat.mean():.2f}, "
                  f"frac>0.9: {100*(flat>0.9).mean():.1f}%")
    ax.axvline(0.9, color="red", ls="--", lw=1, label="cos=0.9 (strong reuse)")
    ax.legend()

    # Panel 6: Per-edge picks (first chunk)
    ax = fig.add_subplot(gs[1, 2])
    src_picks = ana["src_argmax_first"]
    dst_picks = ana["dst_argmax_first"]
    edge_ids = np.arange(K_edge)
    ax.scatter(edge_ids, src_picks, color="tab:blue", label="src pick",
                s=24, alpha=0.7)
    ax.scatter(edge_ids, dst_picks, color="tab:orange", label="dst pick",
                marker="x", s=24, alpha=0.7)
    ax.set_xlabel("Edge index")
    ax.set_ylabel(f"Picked bank slot [0, {K_node})")
    ax.set_title("Per-edge src/dst picks (chunk 0)\nhorizontal bands = hubs")
    ax.legend()

    # Panel 7-9: UMAP of N + endpoints
    try:
        import umap
        ax = fig.add_subplot(gs[2, :])
        all_pts = np.concatenate([
            ana["N_first"], ana["endpoint_src_first"], ana["endpoint_dst_first"]
        ], axis=0)
        all_pts_n = all_pts / (np.linalg.norm(all_pts, axis=-1, keepdims=True) + 1e-6)
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, metric="cosine",
                            random_state=42, n_components=2)
        emb = reducer.fit_transform(all_pts_n)

        n_e = ana["endpoint_src_first"].shape[0]
        N_pts = emb[:K_node]
        src_pts = emb[K_node:K_node + n_e]
        dst_pts = emb[K_node + n_e:]

        sizes = (ana["pick_counts"] / max(ana["pick_counts"].max(), 1)) * 200 + 30
        ax.scatter(N_pts[:, 0], N_pts[:, 1], s=sizes, c="black", marker="s",
                    edgecolors="yellow", linewidths=2,
                    label="N bank slots (size ∝ pick count)", zorder=3)
        ax.scatter(src_pts[:, 0], src_pts[:, 1], s=20, c="tab:blue", alpha=0.5,
                    label="src endpoints", zorder=2)
        ax.scatter(dst_pts[:, 0], dst_pts[:, 1], s=20, c="tab:orange", alpha=0.5,
                    label="dst endpoints", marker="^", zorder=2)
        top5 = np.argsort(ana["pick_counts"])[::-1][:5]
        for rank, idx in enumerate(top5):
            ax.annotate(f"#{rank+1}({int(ana['pick_counts'][idx])})",
                         xy=N_pts[idx], xytext=(5, 5),
                         textcoords="offset points", fontsize=9,
                         color="darkred", weight="bold")
        ax.set_title("UMAP (cosine) — bank slots (squares) + endpoints\n"
                      "Endpoints clustering on slots → sharp pointers; "
                      "Endpoints spread → diffuse soft pointers")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    except Exception as e:
        ax = fig.add_subplot(gs[2, :])
        ax.text(0.5, 0.5, f"UMAP failed: {e}", ha="center", va="center",
                 transform=ax.transAxes)

    fig.suptitle(
        f"graph_v5_baseline v5.1-fair topology probe — ckpt step {step}\n"
        f"K_node={K_node}, K_edge={K_edge}; thesis test: is graph structure forming?",
        fontsize=13, y=1.00,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\n[output] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=4096)
    args = ap.parse_args()

    enc, cfg, step = load_encoder(args.ckpt)

    print(f"[data] loading HotpotQA val + Llama tokenizer/embedding...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.llama_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = get_batch(tokenizer, cfg, args.batch_size, args.chunk_size)

    from transformers import AutoModelForCausalLM
    llama = AutoModelForCausalLM.from_pretrained(cfg.llama_model, torch_dtype=torch.float32)
    llama.train(False)
    llama_embed = llama.get_input_embeddings()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    llama_embed = llama_embed.to(device)

    print(f"\n[forward] batch_size={args.batch_size}, chunk_size={args.chunk_size}, device={device}")
    probe = run_forward(enc, cfg, llama_embed, batch, device)

    ana = analyze(probe, cfg.graph_v5_K_node, cfg.graph_v5_K_edge)
    plot(ana, args.out, step)


if __name__ == "__main__":
    main()
