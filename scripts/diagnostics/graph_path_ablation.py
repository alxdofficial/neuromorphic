"""Which path carries the memory — the FREE edge_state, or the NODE composition?

Tests H1 (addressing failure: nodes wanted but unaddressable) vs H2 (nodes unnecessary:
the model uses each edge_state as a flat free-vector slot and ignores the node vocabulary).

On the trained graph checkpoint, ablate the reader's input per task:
  REAL       : full graph (src_value + dst_value + edge_state)
  edge_only  : zero src_value/dst_value (kill NODE content), keep edge_state (free slots)
  node_only  : zero edge_state (kill the free payload), keep node content
  OFF        : zero memory (no-memory floor)

Read the gaps:
  edge_only ≈ REAL  &  node_only ≈ OFF   ⇒ H2: it's all in edge_state, nodes are dead weight.
  node_only ≈ REAL  &  edge_only ≈ OFF   ⇒ nodes carry it (H2 rejected).
  both ablations hurt a lot                ⇒ the two paths are jointly used.

Usage: python scripts/diagnostics/graph_path_ablation.py [--val-batches 8]
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
from src.memory.model import ReprLearningModel
from scripts.train.train import (
    make_mixed_val_sets, to_device, MIXED_TASK_MODE, MIXED_TASKS_DEFAULT, BABI_DEFAULT_TASKS,
)

MIXED_CTX, MIXED_M, WINDOW_SIZE, PREDICT_LEN = 1024, 32, 1024, 64
MAE_SRC_TOK = "meta-llama/Llama-3.2-1B"
_MODE = {"v": "real"}   # mutated per pass; read by the reader pre-hook


def _prehook(module, args, kwargs):
    """Zero components of the graph dict before the reader consumes it."""
    graph = args[0] if args else kwargs.get("graph")
    g = dict(graph)
    m = _MODE["v"]
    if m == "edge_only":     # kill node content
        g["src_value"] = torch.zeros_like(g["src_value"])
        g["dst_value"] = torch.zeros_like(g["dst_value"])
    elif m == "node_only":   # kill the free edge payload
        g["edge_state"] = torch.zeros_like(g["edge_state"])
    if args:
        return (g, *args[1:]), kwargs
    kwargs = dict(kwargs); kwargs["graph"] = g
    return args, kwargs


@torch.no_grad()
def _val(model, val_set, device, n):
    model.train(False)
    tot = 0.0
    for i, b in enumerate(val_set):
        if i >= n:
            break
        b = to_device(b, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.compute_loss(b, window_size=WINDOW_SIZE)
        tot += float(out["loss_recon"])
    return tot / max(min(n, len(val_set)), 1)


@torch.no_grad()
def _val_off(model, val_set, device, n):
    model.train(False)
    tot = 0.0
    for i, b in enumerate(val_set):
        if i >= n:
            break
        b = to_device(b, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.compute_loss(b, window_size=WINDOW_SIZE, zero_memory=True)
        tot += float(out["loss_recon"])
    return tot / max(min(n, len(val_set)), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="graph_baseline")
    ap.add_argument("--out-tag", default="mixed4k_bio")
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--tasks", nargs="+", default=list(MIXED_TASKS_DEFAULT))
    args = ap.parse_args()
    device = "cuda"

    ckpt = REPO / f"outputs/memory/{args.out_tag}_{args.variant}/ckpts/{args.variant}.last.pt"
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ReprConfig(**sd["metadata"]["cfg_dict"])
    tok = AutoTokenizer.from_pretrained(cfg.llama_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = ReprLearningModel(cfg, variant=args.variant, llama_model=None).to(device)
    model.load_state_dict(sd["model_state_dict"], strict=False)
    model.encoder.reader.register_forward_pre_hook(_prehook, with_kwargs=True)

    vs = make_mixed_val_sets(args.tasks, tok, cfg, args.val_batches, ctx_len=MIXED_CTX,
                             m_slots=MIXED_M, mae_src_tok=MAE_SRC_TOK,
                             babi_tasks=BABI_DEFAULT_TASKS, predict_len=PREDICT_LEN)
    n = args.val_batches
    print(f"\n{'task':<16} {'REAL':>8} {'edge_only':>10} {'node_only':>10} {'OFF':>8}   interpretation")
    print("-" * 86)
    for t in args.tasks:
        model.task_mode = MIXED_TASK_MODE[t]
        _MODE["v"] = "real";      real = _val(model, vs[t], device, n)
        _MODE["v"] = "edge_only"; eo = _val(model, vs[t], device, n)
        _MODE["v"] = "node_only"; no = _val(model, vs[t], device, n)
        _MODE["v"] = "real";      off = _val_off(model, vs[t], device, n)
        band = max(off - real, 1e-6)
        # how much of REAL's benefit each path retains (1.0 = fully carries it, 0 = useless)
        edge_keep = (off - eo) / band
        node_keep = (off - no) / band
        if edge_keep > 0.7 and node_keep < 0.3:
            verd = "H2: edge_state carries it; nodes dead"
        elif node_keep > 0.7 and edge_keep < 0.3:
            verd = "nodes carry it (H2 rejected)"
        elif edge_keep < 0.3 and node_keep < 0.3:
            verd = "neither alone — jointly used / fragile"
        else:
            verd = f"mixed (edge {edge_keep:.0%} / node {node_keep:.0%})"
        print(f"{t:<16} {real:>8.3f} {eo:>10.3f} {no:>10.3f} {off:>8.3f}   {verd}")
    print("\n  edge_keep=(OFF−edge_only)/(OFF−REAL): fraction of memory benefit the FREE edge path retains.")
    print("  node_keep likewise for node content. ~1 = that path carries the memory; ~0 = that path is dead.")


if __name__ == "__main__":
    main()
