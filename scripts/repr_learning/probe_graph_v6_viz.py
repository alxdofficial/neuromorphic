#!/usr/bin/env python3
"""Diagnostic C: visualize graph_v6's learned write structure. For one context, stream
the encoder to its persistent state, project the K_node node bank to 2D, and draw the
K_edge soft-pointer edges (src→dst) on top. Node size/color = how many edge-endpoints
point at it (usage). Answers: is the write forming a real graph (distinct, reused nodes +
varied edges) or a low-rank hub-collapsed blur?

Outputs a PNG to docs/plots/graph_v6_structure.png + prints structure stats.
Usage: python scripts/repr_learning/probe_graph_v6_viz.py [--tag v6_1] [--n-ctx 2]
"""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from src.repr_learning.config import ReprConfig
import scripts.repr_learning.eval_per_family as E


def stream_to_state(model, sample, dev, window=1024):
    enc = model.encoder
    embed = model.decoder.llama.get_input_embeddings()
    ctx = sample.context_ids.unsqueeze(0).to(dev)
    mask = sample.context_mask.unsqueeze(0).to(dev)
    with torch.no_grad():
        te = embed(ctx)
        state = enc.init_streaming_state(1, device=dev, dtype=te.dtype)
        for s in range(0, ctx.shape[1], window):
            state, _ = enc.streaming_write(state, te[:, s:s + window, :],
                                           attention_mask=mask[:, s:s + window], chunk_offset=s)
    return state


def project2d(X):
    try:
        import umap
        return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X)
    except Exception:
        Xc = X - X.mean(0)
        U, S, V = torch.pca_lowrank(torch.tensor(Xc), q=2)
        return (torch.tensor(Xc) @ V).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v6_1")
    ap.add_argument("--n-ctx", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "docs/plots/graph_v6_structure.png"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ReprConfig(fixed_window_size=1024, max_window_size=8192)
    tok = AutoTokenizer.from_pretrained(cfg.llama_model)
    ck = ROOT / f"outputs/repr_learning/{args.tag}_graph_v6_baseline/ckpts/graph_v6_baseline.best.pt"
    model, _ = E.load_variant("graph_v6_baseline", ck, cfg, None)
    model = model.to(dev)
    enc = model.encoder
    samples = E.collect_samples(["biographical"], args.n_ctx, tokenizer=tok, cfg=cfg,
                                chunk_size=8192, passages_per_chunk=600)

    fig, axes = plt.subplots(1, args.n_ctx, figsize=(7 * args.n_ctx, 6.5), squeeze=False)
    for ci, s in enumerate(samples):
        st = stream_to_state(model, s, dev)
        N = st["N"][0].float()                                   # [Kn, d_node]
        with torch.no_grad():
            spk, spv = enc.read_pointer.project_kv(st["N"])
            _, a_src = enc.read_pointer.attend(st["q_src"], spk, spv)   # [1, Ke, Kn]
            _, a_dst = enc.read_pointer.attend(st["q_dst"], spk, spv)
        src_n = a_src[0].argmax(-1).cpu()                        # [Ke] node idx per edge src
        dst_n = a_dst[0].argmax(-1).cpu()
        Kn = N.shape[0]
        usage = torch.zeros(Kn)
        for v in torch.cat([src_n, dst_n]):
            usage[v] += 1
        # stats
        used = (usage > 0).sum().item()
        p = usage / usage.sum()
        eff_nodes = (-(p[p > 0] * p[p > 0].log()).sum()).exp().item()  # perplexity of usage
        Nn = torch.nn.functional.normalize(N, dim=-1)
        offcos = (Nn @ Nn.t())[~torch.eye(Kn, dtype=torch.bool)].mean().item()
        distinct_edges = len({(int(a), int(b)) for a, b in zip(src_n.tolist(), dst_n.tolist())})
        XY = project2d(N.cpu().numpy())

        ax = axes[0][ci]
        for e in range(src_n.shape[0]):
            a, b = src_n[e].item(), dst_n[e].item()
            ax.plot([XY[a, 0], XY[b, 0]], [XY[a, 1], XY[b, 1]], "-", color="#999",
                    alpha=0.10, lw=0.6, zorder=1)
        sc = ax.scatter(XY[:, 0], XY[:, 1], s=10 + 30 * usage.numpy(),
                        c=usage.numpy(), cmap="viridis", edgecolors="#222",
                        linewidths=0.4, zorder=2)
        plt.colorbar(sc, ax=ax, label="edge-endpoints pointing here")
        ax.set_title(f"ctx {ci}: {used}/{Kn} nodes used, eff≈{eff_nodes:.0f}, "
                     f"{distinct_edges}/{src_n.shape[0]} distinct edges\n"
                     f"node collapse cos={offcos:.2f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        print(f"ctx {ci}: nodes_used={used}/{Kn}  eff_nodes≈{eff_nodes:.1f}  "
              f"distinct_edges={distinct_edges}/{src_n.shape[0]}  collapse_cos={offcos:.3f}")
    fig.suptitle(f"graph_v6 ({args.tag}) write structure — node bank (UMAP/PCA) + soft-pointer edges",
                 fontsize=12)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"\n[output] wrote {args.out}")
    print("interpretation: few big nodes + few distinct edges = hub collapse / blur; "
          "many used nodes + many distinct edges = real graph structure")


if __name__ == "__main__":
    main()
