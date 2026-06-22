"""Graph-model internals diagnostic — node-bank usage, routing diversity, grad flow.

Loads the trained graph_baseline checkpoint (config rebuilt from metadata.cfg_dict)
and probes the relational parser over MANY real val samples (all 4 mixed tasks):

  (A) NODE-BANK USAGE  — marginal over all pointer selections (src+dst, all windows):
      - node-use %      : fraction of the N-node bank ever hard-selected (argmax)
      - eff-vocab (PPL) : exp(H(usage)) for hard counts AND soft mass — "how many
                          nodes are effectively in play" (N = perfect spread, 1 = hub)
      - top-k share     : mass in the top-1% / top-10 nodes (hub-collapse signature)
      - a sorted usage curve PNG

  (B) ROUTING DIVERSITY — per selection / per sample:
      - mean NORMALISED pointer entropy H/log(N) per edge (0 = one-hot, 1 = uniform):
        how confident each pointer is
      - mean within-sample DISTINCT-node fraction across the E edges: do different
        edges pick different nodes, or collapse to the same few?

  (C) EFF-RANK         — node_bank, edge_state, emitted memory (participation ratio).

  (D) GRAD FLOW        — REAL bf16 fwd+bwd (the training path, not eager-fp32): total
      grad L2 and grad/param ratio per MODULE GROUP, so we can see whether the WRITE
      machinery (node_bank, q_src/q_dst pointer heads, parser blocks) is starved
      relative to the READ (Perceiver) and the decoder LoRA.

Usage:
  python scripts/diagnostics/graph_internals_diag.py [--val-batches 12] [--variant graph_baseline]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer

from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel, _participation_ratio
from scripts.train.train import (
    make_mixed_val_sets, to_device, MIXED_TASK_MODE, MIXED_TASKS_DEFAULT,
    BABI_DEFAULT_TASKS,
)

MIXED_CTX, MIXED_M, WINDOW_SIZE, PREDICT_LEN = 1024, 32, 1024, 64
MAE_SRC_TOK = "meta-llama/Llama-3.2-1B"


def _ppl(counts: torch.Tensor) -> float:
    """exp(entropy) of a (nonneg) count/mass vector = effective number of nodes."""
    p = counts.double()
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    nz = p[p > 0]
    return float(torch.exp(-(nz * nz.log()).sum()).item())


def _grad_group(name: str) -> str:
    if "mask_embed" in name:
        return "decoder.mask_embed"
    if "decoder" in name and ("lora" in name.lower()):
        return "decoder.lora"
    if "encoder.parser." in name:
        tail = name.split("encoder.parser.", 1)[1]
        if tail.startswith("node_bank"):
            return "parser.node_bank (vocab)"
        if tail.startswith("bank_key"):
            return "parser.bank_key"
        if tail.startswith("q_src"):
            return "parser.q_src (ptr)"
        if tail.startswith("q_dst"):
            return "parser.q_dst (ptr)"
        if tail.startswith("obs_proj"):
            return "parser.obs_proj"
        if tail.startswith("edge_head"):
            return "parser.edge_head"
        if tail.startswith("log_temp"):
            return "parser.log_temp"
        if tail.startswith("blocks"):
            return "parser.write_blocks"
        if any(tail.startswith(p) for p in ("role", "tag", "part", "init_graph", "node_role_avail")):
            return "parser.embeds"
        return "parser.other"
    if "encoder.reader." in name:
        tail = name.split("encoder.reader.", 1)[1]
        if tail.startswith("queries"):
            return "reader.queries (latents)"
        if tail.startswith("blocks"):
            return "reader.perceiver_blocks"
        if tail.startswith("out") or tail.startswith("norm"):
            return "reader.out"
        if tail.startswith("role_emb") or tail.startswith("tag"):
            return "reader.embeds"
        return "reader.other"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="graph_baseline")
    ap.add_argument("--out-tag", default="mixed4k_bio")
    ap.add_argument("--val-batches", type=int, default=12)
    ap.add_argument("--tasks", nargs="+", default=list(MIXED_TASKS_DEFAULT))
    args = ap.parse_args()
    device = "cuda"

    ckpt = REPO / f"outputs/memory/{args.out_tag}_{args.variant}/ckpts/{args.variant}.last.pt"
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ReprConfig(**sd["metadata"]["cfg_dict"])
    N, E = cfg.graph_n_nodes, cfg.graph_n_edges
    print(f"[graph-internals] {args.variant}: N={N} bank, E={E} edges, "
          f"d_graph={cfg.graph_d_graph}, window={cfg.graph_window}, ctx={MIXED_CTX}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.llama_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = ReprLearningModel(cfg, variant=args.variant, llama_model=None).to(device)
    res = model.load_state_dict(sd["model_state_dict"], strict=False)
    if res.unexpected_keys:
        print(f"  [warn] {len(res.unexpected_keys)} unexpected keys")

    parser = model.encoder.parser

    val_sets = make_mixed_val_sets(
        args.tasks, tokenizer, cfg, args.val_batches, ctx_len=MIXED_CTX,
        m_slots=MIXED_M, mae_src_tok=MAE_SRC_TOK, babi_tasks=BABI_DEFAULT_TASKS,
        predict_len=PREDICT_LEN)

    # ── parser hook: capture src_ptr/dst_ptr for every window parse ──
    cap = []
    def _hook(mod, inp, out):
        if isinstance(out, dict) and "src_ptr" in out:
            cap.append((out["src_ptr"].detach(), out["dst_ptr"].detach()))
    h = parser.register_forward_hook(_hook)

    results = {}
    overall_hard = torch.zeros(N, dtype=torch.float64, device=device)   # hard argmax counts
    overall_soft = torch.zeros(N, dtype=torch.float64, device=device)   # soft pointer mass

    print("\n" + "=" * 78 + "\n(A,B,C) NODE-BANK USAGE / ROUTING / EFF-RANK (no grad)\n" + "=" * 78)
    model.train(False)
    for t in args.tasks:
        model.task_mode = MIXED_TASK_MODE[t]
        hard = torch.zeros(N, dtype=torch.float64, device=device)
        soft = torch.zeros(N, dtype=torch.float64, device=device)
        ents, distinct_fracs = [], []
        edge_rank, mem_rank, n_sel = [], [], 0
        for i, batch in enumerate(val_sets[t]):
            if i >= args.val_batches:
                break
            cap.clear()
            batch = to_device(batch, device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.compute_loss(batch, window_size=WINDOW_SIZE)
            if "graph_edge_effrank" in out:
                edge_rank.append(float(out["graph_edge_effrank"]))
            if "graph_mem_effrank" in out:
                mem_rank.append(float(out["graph_mem_effrank"]))
            for src_ptr, dst_ptr in cap:                      # each [B,E,N], one per window
                both = torch.cat([src_ptr, dst_ptr], dim=1).float()   # [B,2E,N]
                sel = both.argmax(-1)                          # [B,2E]
                hard += torch.bincount(sel.reshape(-1), minlength=N).double()
                soft += both.sum(dim=(0, 1)).double()
                n_sel += sel.numel()
                # per-edge normalised entropy
                ent = -(both * both.clamp_min(1e-12).log()).sum(-1) / torch.log(torch.tensor(float(N)))
                ents.append(ent.reshape(-1).cpu())
                # within-sample distinct-node fraction (across the 2E selections)
                for b in range(sel.shape[0]):
                    distinct_fracs.append(sel[b].unique().numel() / sel.shape[1])
        ents = torch.cat(ents) if ents else torch.zeros(1)
        srt = torch.sort(hard, descending=True).values
        top1pct = float(srt[:max(1, N // 100)].sum() / srt.sum().clamp_min(1))
        top10 = float(srt[:10].sum() / srt.sum().clamp_min(1))
        results[t] = {
            "n_selections": n_sel,
            "nodes_used": int((hard > 0).sum().item()),
            "node_use_pct": 100.0 * float((hard > 0).sum().item()) / N,
            "effvocab_hard_ppl": _ppl(hard),
            "effvocab_soft_ppl": _ppl(soft),
            "top1pct_share": top1pct,
            "top10_share": top10,
            "ptr_entropy_norm_mean": float(ents.mean()),
            "ptr_entropy_norm_p50": float(ents.median()),
            "within_sample_distinct_frac": float(sum(distinct_fracs) / max(len(distinct_fracs), 1)),
            "edge_effrank": sum(edge_rank) / max(len(edge_rank), 1),
            "mem_effrank": sum(mem_rank) / max(len(mem_rank), 1),
        }
        overall_hard += hard
        overall_soft += soft
        r = results[t]
        print(f"\n[{t}]  ({n_sel:,} selections)")
        print(f"  node-use %        : {r['node_use_pct']:.1f}%  ({r['nodes_used']}/{N} bank nodes ever picked)")
        print(f"  eff-vocab (PPL)   : hard={r['effvocab_hard_ppl']:.1f}   soft={r['effvocab_soft_ppl']:.1f}   (of {N})")
        print(f"  hub concentration : top-10 nodes={100*r['top10_share']:.1f}%   top-1% nodes={100*r['top1pct_share']:.1f}% of all selections")
        print(f"  ptr entropy /logN : mean={r['ptr_entropy_norm_mean']:.3f}  p50={r['ptr_entropy_norm_p50']:.3f}  (0=one-hot, 1=uniform)")
        print(f"  within-sample div : {100*r['within_sample_distinct_frac']:.1f}% of the {2*E} edges pick distinct nodes")
        print(f"  eff-rank          : edge_state={r['edge_effrank']:.2f}  emitted_mem={r['mem_effrank']:.2f}  (d_graph={cfg.graph_d_graph}, M={MIXED_M})")

    h.remove()
    bank_rank = _participation_ratio(parser.node_bank)
    print(f"\n[overall]  node_bank participation-ratio (eff-rank) = {bank_rank:.2f} / {cfg.graph_d_graph}")
    print(f"[overall]  node-use % = {100*float((overall_hard>0).sum())/N:.1f}%   "
          f"eff-vocab hard PPL = {_ppl(overall_hard):.1f} / {N}")

    # sorted usage curve
    png = REPO / f"outputs/memory/{args.out_tag}_{args.variant}_nodeuse.png"
    _plot_usage(overall_hard.cpu(), overall_soft.cpu(), N, png, args.variant)

    # ── (D) GRAD FLOW — real bf16 fwd+bwd ──
    print("\n" + "=" * 78 + "\n(D) GRAD FLOW per module group (real bf16 fwd+bwd)\n" + "=" * 78)
    model.train(True)
    name2group = {n: _grad_group(n) for n, _ in model.named_parameters()}
    grad_sq = defaultdict(float)
    param_sq = defaultdict(float)
    n_back = 0
    for t in args.tasks:
        model.task_mode = MIXED_TASK_MODE[t]
        for i, batch in enumerate(val_sets[t]):
            if i >= max(2, args.val_batches // 4):      # a few backward passes per task
                break
            batch = to_device(batch, device)
            model.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.compute_loss(batch, window_size=WINDOW_SIZE)
            out["loss"].backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    grad_sq[name2group[n]] += float(p.grad.detach().float().pow(2).sum())
            n_back += 1
    # param norms (once)
    for n, p in model.named_parameters():
        if p.requires_grad:
            param_sq[name2group[n]] += float(p.detach().float().pow(2).sum())

    import math
    rows = []
    for g in sorted(grad_sq, key=lambda k: -grad_sq[k]):
        gnorm = math.sqrt(grad_sq[g] / max(n_back, 1))
        pnorm = math.sqrt(param_sq.get(g, 0.0))
        ratio = gnorm / pnorm if pnorm > 0 else float("nan")
        rows.append((g, gnorm, pnorm, ratio))
        results.setdefault("_gradflow", {})[g] = {"grad_l2": gnorm, "param_l2": pnorm, "grad_over_param": ratio}
    print(f"\n{'module group':<28} {'grad L2':>11} {'param L2':>11} {'grad/param':>11}")
    print("-" * 64)
    for g, gn, pn, r in rows:
        print(f"{g:<28} {gn:>11.4e} {pn:>11.4e} {r:>11.3e}")

    out_json = REPO / f"outputs/memory/{args.out_tag}_{args.variant}_internals.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[graph-internals] wrote {out_json}")
    print(f"[graph-internals] wrote {png}")


def _plot_usage(hard, soft, N, png, variant):
    hs = torch.sort(hard, descending=True).values
    ss = torch.sort(soft, descending=True).values
    hp = (hs / hs.sum().clamp_min(1)).numpy()
    sp = (ss / ss.sum().clamp_min(1)).numpy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for a, y, ttl in ((ax[0], hp, "hard (argmax) node usage"), (ax[1], sp, "soft (pointer-mass) node usage")):
        a.plot(range(1, N + 1), y, lw=1.2)
        a.set_yscale("log"); a.set_xscale("log")
        a.set_xlabel("node rank (sorted)"); a.set_ylabel("share of selections")
        a.set_title(ttl)
        a.axhline(1.0 / N, ls="--", c="gray", lw=0.8, label=f"uniform 1/{N}")
        a.legend(fontsize=8)
    fig.suptitle(f"{variant} — node-bank usage (N={N})  [flat=spread, steep=hub-collapse]")
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
