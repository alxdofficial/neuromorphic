"""Collapse-contributor sweep for the graph model — what co-causes the node-bank
collapse BESIDES the bag-of-edges read?

The read-side probe (graph_internals_diag) showed the bank collapses to ~2 used nodes
and the pointer/bank gradient is starved. But the read may not be the only cause. This
sweep interrogates the WRITE side (parser → selection) on the trained checkpoint:

  (1) PARSER OUTPUT DIVERSITY  — are the per-edge pointer QUERIES (q_src/q_dst) and the
      part-2 slot outputs distinct across the E edges, or has the parser collapsed every
      edge to ~the same vector (→ every edge points at the same node, independent of the
      read)? eff-rank across edges + within-sample edge-query cosine.

  (2) INPUT SENSITIVITY        — do the queries depend on the OBSERVATION, or are they
      input-invariant (parser ignores obs → structural collapse)? eff-rank of a fixed
      edge slot's query across many inputs.

  (3) POINTER-KEY DISTINGUISHABILITY — are the bank match-keys bank_key(node_bank)
      separable, or do many nodes look identical to a query (→ unselectable)? key
      eff-rank + pairwise-cosine, vs the raw node_bank eff-rank.

  (4) TEMPERATURE LOCK         — learned softmax temps (pointer + parser/reader attn):
      any saturated at the clamp (sharp lock or washed-out)?

  (5) IDENTITY EMBEDS          — relative norms of per-edge tag vs role/part/init: do
      edges carry enough distinct identity to differentiate?

Static + a few real forward passes; NO training.
Usage: python scripts/diagnostics/graph_collapse_sweep.py [--val-batches 6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from transformers import AutoTokenizer

from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel, _participation_ratio
from scripts.train.train import (
    make_mixed_val_sets, to_device, MIXED_TASK_MODE, MIXED_TASKS_DEFAULT,
    BABI_DEFAULT_TASKS,
)

MIXED_CTX, MIXED_M, WINDOW_SIZE, PREDICT_LEN = 1024, 32, 1024, 64
MAE_SRC_TOK = "meta-llama/Llama-3.2-1B"


def _effrank(x):  # x: [n, d]
    return _participation_ratio(x)


def _mean_offdiag_cosine(x):
    """mean pairwise cosine of rows of x [n,d] (n small)."""
    xn = torch.nn.functional.normalize(x.float(), dim=-1)
    c = xn @ xn.t()
    n = c.shape[0]
    off = (c.sum() - c.diag().sum()) / (n * (n - 1))
    return float(off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="graph_baseline")
    ap.add_argument("--out-tag", default="mixed4k_bio")
    ap.add_argument("--val-batches", type=int, default=6)
    ap.add_argument("--tasks", nargs="+", default=list(MIXED_TASKS_DEFAULT))
    args = ap.parse_args()
    device = "cuda"

    ckpt = REPO / f"outputs/memory/{args.out_tag}_{args.variant}/ckpts/{args.variant}.last.pt"
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ReprConfig(**sd["metadata"]["cfg_dict"])
    N, E, dg = cfg.graph_n_nodes, cfg.graph_n_edges, cfg.graph_d_graph
    print(f"[collapse-sweep] {args.variant}: N={N}, E={E}, d_graph={dg}")

    tok = AutoTokenizer.from_pretrained(cfg.llama_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = ReprLearningModel(cfg, variant=args.variant, llama_model=None).to(device)
    model.load_state_dict(sd["model_state_dict"], strict=False)
    model.train(False)
    parser = model.encoder.parser

    # ── (3) STATIC: pointer-key distinguishability + (5) identity embeds + (4) temps ──
    print("\n" + "=" * 78 + "\n(3) POINTER-KEY DISTINGUISHABILITY (static)\n" + "=" * 78)
    with torch.no_grad():
        bank = parser.node_bank.float()                      # [N,d]
        keys = parser.bank_key(parser.node_bank).float()     # [N,d] match-keys
        bank_er = _effrank(bank)
        key_er = _effrank(keys)
        # sample 512 nodes for a cosine estimate
        idx = torch.randperm(N)[:512]
        key_cos = _mean_offdiag_cosine(keys[idx])
        bank_cos = _mean_offdiag_cosine(bank[idx])
    print(f"  node_bank   eff-rank = {bank_er:6.2f} / {dg}   mean pairwise cos = {bank_cos:+.3f}")
    print(f"  bank_KEYS   eff-rank = {key_er:6.2f} / {dg}   mean pairwise cos = {key_cos:+.3f}")
    print(f"  → keys {'COLLAPSED vs bank (key proj is a bottleneck)' if key_er < 0.6*bank_er else 'track the bank (keys separable)'}")

    print("\n" + "=" * 78 + "\n(5) IDENTITY EMBEDS — relative RMS norm (static)\n" + "=" * 78)
    def _rms(p): return float(p.float().pow(2).mean().sqrt())
    embeds = {"tag[E]": parser.tag, "role[3]": parser.role, "part[2]": parser.part,
              "init_graph[3,E]": parser.init_graph}
    if hasattr(parser, "node_role_avail"):
        embeds["node_role_avail"] = parser.node_role_avail
    for k, p in embeds.items():
        print(f"  {k:<18} RMS = {_rms(p):.4f}")
    print(f"  → per-edge tag {'SWAMPED by role/part (edges lack identity)' if _rms(parser.tag) < 0.5*_rms(parser.role) else 'comparable to role/part (edges identifiable)'}")

    print("\n" + "=" * 78 + "\n(4) TEMPERATURE LOCK — learned softmax temps (static)\n" + "=" * 78)
    pt = parser.log_temp.detach().clamp(-3, 3).exp().tolist()
    print(f"  pointer temp (src,dst) = {[round(x,3) for x in pt]}   (clamp range 0.05–20; <0.1 = sharp lock)")
    sat = []
    for name, m in model.named_modules():
        lt = getattr(m, "log_temp", None)
        if isinstance(lt, torch.nn.Parameter) and "parser" in name or (lt is not None and ("reader" in name or "parser" in name)):
            if isinstance(lt, torch.nn.Parameter):
                raw = lt.detach()
                near = (raw.abs() >= 2.9).float().mean().item()
                if near > 0:
                    sat.append((name, round(near, 2)))
    print(f"  attn temps saturated at clamp (|log_temp|≥2.9): {sat if sat else 'none'}")

    # ── forward probes: capture q_src/q_dst (output) + slot inputs ──
    cap = {"q_src": [], "q_dst": [], "src_t": [], "dst_t": [], "edge_state": []}
    def _mk(name, which):
        def hook(mod, inp, out):
            cap[name].append(out.detach())
            if which:
                cap[which].append(inp[0].detach())
        return hook
    hs = [parser.q_src.register_forward_hook(_mk("q_src", "src_t")),
          parser.q_dst.register_forward_hook(_mk("q_dst", "dst_t")),
          parser.edge_head.register_forward_hook(_mk("edge_state", None))]

    val_sets = make_mixed_val_sets(
        args.tasks, tok, cfg, args.val_batches, ctx_len=MIXED_CTX, m_slots=MIXED_M,
        mae_src_tok=MAE_SRC_TOK, babi_tasks=BABI_DEFAULT_TASKS, predict_len=PREDICT_LEN)

    print("\n" + "=" * 78 + "\n(1) PARSER OUTPUT DIVERSITY  +  (2) INPUT SENSITIVITY\n" + "=" * 78)
    print(f"{'task':<16} {'q_src ER(all)':>13} {'edge-cos(WS)':>13} {'q ER/edge':>11} {'q ER/input':>11} {'edge_state ER':>13}")
    print("-" * 80)
    for t in args.tasks:
        for k in cap: cap[k].clear()
        model.task_mode = MIXED_TASK_MODE[t]
        for i, batch in enumerate(val_sets[t]):
            if i >= args.val_batches:
                break
            batch = to_device(batch, device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model.compute_loss(batch, window_size=WINDOW_SIZE)
        q = torch.cat([x.float() for x in cap["q_src"]], dim=0)      # [n_fwd*B, E, d]
        es = torch.cat([x.float() for x in cap["edge_state"]], dim=0)
        n = q.shape[0]
        er_all = _effrank(q.reshape(-1, dg))
        # within-sample edge cosine: mean over rows of the E-edge cosine
        ws_cos = sum(_mean_offdiag_cosine(q[b]) for b in range(min(n, 64))) / min(n, 64)
        # eff-rank across EDGES within a sample (avg) — do edges differ?
        er_edge = sum(_effrank(q[b]) for b in range(min(n, 64))) / min(n, 64)
        # eff-rank across INPUTS for a fixed edge slot (avg over a few slots) — input sensitivity
        er_input = sum(_effrank(q[:, e]) for e in range(min(E, 16))) / min(E, 16)
        er_es = _effrank(es.reshape(-1, dg))
        print(f"{t:<16} {er_all:>13.2f} {ws_cos:>+13.3f} {er_edge:>11.2f} {er_input:>11.2f} {er_es:>13.2f}")
    for h in hs:
        h.remove()
    print(f"\n  legend: ER=participation-ratio (eff-rank) out of d_graph={dg}.")
    print(f"  edge-cos(WS)→1 ⇒ all E edges emit the same query (parser collapse, read-independent).")
    print(f"  q ER/edge≈1 ⇒ edges identical within a sample; q ER/input≈1 ⇒ query ignores the observation.")


if __name__ == "__main__":
    main()
