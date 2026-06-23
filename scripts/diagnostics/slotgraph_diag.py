"""slotgraph diagnostics — did the emergent graph structure actually form, or collapse?

Loads the TRAINED slotgraph checkpoint, runs it on real bAbI (the binding task), extracts the
learned per-slot structure (node/edge role + edge endpoints + soft distributions), and reports:
  - structure: edge fraction, src/dst SHARPNESS (entropy vs max), node-target usage diversity
  - collapse: memory eff-rank, slot pairwise cosine, role/endpoint concentration
  - VISUALS: (1) UMAP of one example's 32 slots, node/edge-colored, with the predicted edges drawn;
    (2) a 2x2 histogram panel (edge_frac, per-edge endpoint entropy, node-target usage, mem eff-rank).

Verdict heuristic: edges are "sharp/real" if endpoint entropy ≪ ln(M) and node-usage is spread;
"smeared/collapsed" if entropy ≈ ln(M) (near-uniform = effectively random edges).

Usage: python scripts/diagnostics/slotgraph_diag.py
"""
from __future__ import annotations
import sys, math, dataclasses
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from transformers import AutoTokenizer
from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel
from scripts.train.train import make_mixed_val_sets, to_device

DEV = "cuda"
CKPT = REPO / "outputs/memory/mixed4k_bio_slotgraph_baseline/ckpts/slotgraph_baseline.best.pt"
OUT = REPO / "outputs/memory/slotgraph_diag"
OUT.mkdir(parents=True, exist_ok=True)


def load_model():
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    cd = sd["metadata"]["cfg_dict"]
    valid = {f.name for f in dataclasses.fields(ReprConfig)}        # drop fields removed since training
    cfg = ReprConfig(**{k: v for k, v in cd.items() if k in valid})
    m = ReprLearningModel(cfg, variant="slotgraph_baseline", llama_model=None).to(DEV)
    res = m.load_state_dict(sd["model_state_dict"], strict=False)
    bad = [k for k in res.missing_keys if "llama" not in k.lower() and not k.startswith(("encoder.base.", "decoder.llama."))]
    if bad:
        print(f"  [warn] missing non-base keys: {bad[:8]}")
    m.eval()
    return m, cfg, AutoTokenizer.from_pretrained(cfg.llama_model)


@torch.no_grad()
def structure(enc, ctx, mask):
    """Replicate slotgraph's encode and read out slot_final + soft role/src/dst (the final-layer
    structure). Returns slot_final [B,M,d], role_soft [B,M,2], src_soft/dst_soft [B,M,M]."""
    B, T, d = ctx.shape; M = enc.M
    slots0 = (enc.slot_init.unsqueeze(0).expand(B, M, d) + enc.id_embed.unsqueeze(0)).to(ctx.dtype)
    inp = torch.cat([ctx, slots0], 1)
    attn = torch.cat([mask, torch.ones(B, M, device=mask.device, dtype=mask.dtype)], 1).long()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):   # no injection hooks (structure is read-only now)
        h = enc.base.model(inputs_embeds=inp, attention_mask=attn, use_cache=False).last_hidden_state
    slot_final = h[:, -M:].float()
    hhin = enc._head_in(slot_final)                       # [struct_norm(slot) ; scaled id]
    temp = float(enc.log_temp.exp().clamp_min(1e-2))
    role = enc.role_fixed.unsqueeze(0).expand(B, -1, -1).float()   # FIXED partition (0=node, 1=edge)
    s_logits, d_logits = enc._endpoint_logits(hhin)      # masked to the fixed node pool (edges→nodes)
    src = (s_logits / temp).softmax(-1)
    dst = (d_logits / temp).softmax(-1)
    return slot_final, role, src, dst


def ent(p, dim=-1):
    return -(p.clamp_min(1e-12).log() * p).sum(dim)


def main():
    print(f"loading {CKPT.name} ...")
    m, cfg, tok = load_model()
    enc = m.encoder; M = enc.M
    lnM = math.log(enc.K)   # endpoints are masked to the K-node pool → max entropy is ln(K), not ln(M)
    vs = make_mixed_val_sets(["babi"], tok, cfg, 8, ctx_len=1024, m_slots=32,
                             mae_src_tok="meta-llama/Llama-3.2-1B",
                             babi_tasks=(1, 2, 3, 7, 8, 11, 12, 13, 14), predict_len=64)["babi"]

    # ── aggregate over several batches for the histograms ──
    edge_frac, edge_ent, node_usage, mem_er, role_ent_all = [], [], [], [], []
    edge_valid = []   # per-edge: are BOTH endpoints node slots? (else invalid: edge→non-node)
    keep_example = None
    for bi, batch in enumerate(vs):
        batch = to_device(batch, DEV)
        ctx = m.decoder.llama.get_input_embeddings()(batch.context_ids)
        slot_final, role, src, dst = structure(enc, ctx, batch.context_mask)
        B = slot_final.shape[0]
        role_hard = role.argmax(-1)                       # [B,M] 0=node 1=edge
        is_edge = role_hard == 1
        edge_frac.extend(is_edge.float().mean(1).tolist())
        role_ent_all.extend(ent(role).mean(1).tolist())
        # per-edge endpoint entropy (only for slots predicted edge)
        ee = 0.5 * (ent(src) + ent(dst))                 # [B,M] avg src/dst entropy
        for b in range(B):
            em = is_edge[b]
            if em.any():
                edge_ent.extend(ee[b][em].tolist())
                s_arg = src[b].argmax(-1); d_arg = dst[b].argmax(-1)   # [M] endpoint positions
                node_usage.append(torch.cat([s_arg[em], d_arg[em]]).cpu())
                isnode = ~em                                            # [M] True where slot is a node
                for e in em.nonzero(as_tuple=True)[0]:                 # edge is VALID iff both endpoints are nodes
                    edge_valid.append(bool(isnode[s_arg[e]] and isnode[d_arg[e]]))
            # memory eff-rank per example (slot vectors)
            x = slot_final[b] - slot_final[b].mean(0, keepdim=True)
            C = x.t() @ x
            mem_er.append(float((torch.diagonal(C).sum() ** 2 / (C * C).sum().clamp_min(1e-9))))
        if keep_example is None:   # stash batch 0, example 0 for the UMAP
            keep_example = (slot_final[0].cpu().numpy(), role_hard[0].cpu().numpy(),
                            src[0].argmax(-1).cpu().numpy(), dst[0].argmax(-1).cpu().numpy())

    edge_frac = np.array(edge_frac); edge_ent = np.array(edge_ent)
    mem_er = np.array(mem_er); role_ent_all = np.array(role_ent_all)
    all_tgts = torch.cat(node_usage).numpy() if node_usage else np.array([])
    # node-usage concentration: fraction of all edge-endpoints landing on the top-2 slots
    if all_tgts.size:
        counts = np.bincount(all_tgts, minlength=M)
        top2_frac = np.sort(counts)[::-1][:2].sum() / counts.sum()
        n_used = int((counts > 0).sum())
    else:
        top2_frac, n_used = float("nan"), 0

    print(f"\n{'='*64}\nslotgraph structure diagnostics (bAbI, {len(edge_frac)} examples)\n{'='*64}")
    print(f"  edge_frac:        mean={edge_frac.mean():.3f}  (fraction of slots that are edges)")
    print(f"  role entropy:     mean={role_ent_all.mean():.3f} / max ln2={math.log(2):.3f}")
    print(f"  endpoint entropy: mean={edge_ent.mean():.3f} / max lnK(={enc.K})={lnM:.3f}  "
          f"({100*edge_ent.mean()/lnM:.0f}% of max → {'SMEARED/near-random' if edge_ent.mean()>0.8*lnM else 'has structure'})")
    print(f"  node-target usage: {n_used}/{M} slots ever targeted; top-2 capture {top2_frac*100:.0f}% of endpoints "
          f"({'COLLAPSED' if top2_frac>0.5 else 'spread'})")
    print(f"  memory eff-rank:  mean={mem_er.mean():.2f} / {M}  ({'collapsed' if mem_er.mean()<3 else 'healthy'})")
    inv = (1 - np.mean(edge_valid)) if edge_valid else float("nan")
    print(f"  invalid edges:    {inv*100:.0f}% of edges point to a NON-node slot (no edges→nodes constraint)")

    # ── PLOT 1: UMAP of NODE slots only; edges are arrows between node endpoints (not points) ──
    sv, rh, sh, dh = keep_example                       # slot vecs, role(0=node), src/dst argmax positions
    node_idx = np.where(rh == 0)[0]; node_set = set(int(i) for i in node_idx)
    edge_idx = np.where(rh == 1)[0]
    import umap
    node2d = umap.UMAP(n_neighbors=min(8, max(2, len(node_idx) - 1)), min_dist=0.3,
                       random_state=0).fit_transform(sv[node_idx])   # ONLY the node entities
    pos = {int(p): node2d[i] for i, p in enumerate(node_idx)}        # slot-position → 2D (nodes only)
    fig, ax = plt.subplots(figsize=(8, 8))
    n_valid = n_invalid = 0
    for e in edge_idx:                                                # each edge slot = an ARROW node→node
        s, d = int(sh[e]), int(dh[e])
        if s in node_set and d in node_set:
            ax.annotate("", xy=pos[d], xytext=pos[s],
                        arrowprops=dict(arrowstyle="->", color="crimson", alpha=0.55, lw=1.3)); n_valid += 1
        else:
            n_invalid += 1                                            # edge points to a non-node slot → can't draw
    ax.scatter(node2d[:, 0], node2d[:, 1], c="steelblue", s=200, zorder=3, edgecolors="k")
    for i, p in enumerate(node_idx):
        ax.text(node2d[i, 0], node2d[i, 1], str(int(p)), fontsize=7, ha="center", va="center", zorder=4)
    ax.set_title(f"slotgraph — one bAbI example: NODE slots in UMAP, edges as src→dst arrows\n"
                 f"{len(node_idx)} nodes; edges: {n_valid} valid (node→node), {n_invalid} INVALID (→ non-node, not drawn)")
    ax.set_xticks([]); ax.set_yticks([])
    p1 = OUT / "slotgraph_umap_example.png"; fig.tight_layout(); fig.savefig(p1, dpi=130); plt.close(fig)

    # ── PLOT 2: histogram panel ──
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    axs[0, 0].hist(edge_frac, bins=20, color="slateblue"); axs[0, 0].set_title("edge fraction per example"); axs[0, 0].set_xlabel("frac slots = edge")
    axs[0, 1].hist(edge_ent, bins=30, color="crimson"); axs[0, 1].axvline(lnM, color="k", ls="--", label=f"max lnK({enc.K})={lnM:.2f}")
    axs[0, 1].set_title("per-edge endpoint entropy (↓=sharp, →max=random)"); axs[0, 1].set_xlabel("entropy"); axs[0, 1].legend()
    if all_tgts.size:
        axs[1, 0].bar(range(M), np.bincount(all_tgts, minlength=M), color="seagreen")
    axs[1, 0].set_title("node-target usage (which slots edges point to)"); axs[1, 0].set_xlabel("slot position")
    axs[1, 1].hist(mem_er, bins=20, color="darkorange"); axs[1, 1].set_title(f"memory eff-rank (max {M})"); axs[1, 1].set_xlabel("eff-rank")
    p2 = OUT / "slotgraph_histograms.png"; fig.tight_layout(); fig.savefig(p2, dpi=120); plt.close(fig)
    print(f"\nwrote:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
