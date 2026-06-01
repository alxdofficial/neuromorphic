#!/usr/bin/env python3
"""Decisive read-utilization probe for graph_v6 (why does it score BELOW a
no-memory floor?). Runs the TRAINED graph_v6 ckpt on the same questions in 3
conditions and compares EM/containment:

  REAL    — normal per-token fact inject (what eval did).
  OFF     — inject disabled (inject=None): graph_v6 decodes with NO memory read.
  SHUFFLE — each sample gets ANOTHER sample's facts (value rolled along batch).

Readout:
  OFF > REAL      → the inject is NET-HARMFUL (corrupts decode).
  REAL ≈ OFF      → the inject is inert (no contribution).
  REAL ≈ SHUFFLE  → the read is NOT content-sensitive (generic perturbation,
                    not retrieving answer-relevant facts).
  REAL > SHUFFLE  → the read IS using the right facts (good; deficit is elsewhere).

Usage: python scripts/repr_learning/probe_graph_v6_inject.py [--tag v2_1] [--n 32]
"""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoTokenizer
from src.repr_learning.config import ReprConfig
import scripts.repr_learning.eval_per_family as E


def macro(rows):
    em = sum(r["em"] for r in rows) / max(1, len(rows))
    con = sum(r["con"] for r in rows) / max(1, len(rows))
    return 100 * em, 100 * con


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2_1")
    ap.add_argument("--variant", default="graph_v6_baseline")
    ap.add_argument("--families", nargs="+", default=["biographical", "hotpot_qa", "musique"])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--window-size", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = ReprConfig(fixed_window_size=args.window_size, max_window_size=args.chunk_size)
    tok = AutoTokenizer.from_pretrained(cfg.llama_model)
    ckpt = ROOT / f"outputs/repr_learning/{args.tag}_{args.variant}/ckpts/{args.variant}.best.pt"
    if not ckpt.exists():
        ckpt = ROOT / f"outputs/repr_learning/{args.tag}_{args.variant}/ckpts/{args.variant}.last.pt"
    print(f"[probe] loading {ckpt}")
    model, step = E.load_variant(args.variant, ckpt, cfg, None)
    model = model.to(dev)
    pp = max(75, (args.chunk_size // 1024) * 75)
    samples = E.collect_samples(args.families, args.n, tokenizer=tok, cfg=cfg,
                                chunk_size=args.chunk_size, passages_per_chunk=pp)
    by_fam = {}
    for s in samples:
        by_fam.setdefault(s.family, []).append(s)

    conds = {"REAL": [], "OFF": [], "SHUFFLE": []}
    per_fam = {c: {} for c in conds}
    for fam, fs in by_fam.items():
        for c in conds:
            per_fam[c][fam] = []
        for start in range(0, len(fs), args.batch_size):
            batch = fs[start:start + args.batch_size]
            memory, faux = E._stream_encode_batch(model, batch, dev, args.window_size)
            mem_mask = faux.get("memory_mask")
            facts = faux.get("graph_v6_facts")
            lidx = model.encoder.inject_layer_idx
            ct = getattr(model, "chat_template", None)
            specs = {
                "REAL": {"encoder": model.encoder, "facts": facts, "layer_idx": lidx},
                "OFF": None,
                "SHUFFLE": {"encoder": model.encoder, "layer_idx": lidx,
                            "facts": {k: (v.roll(1, 0) if torch.is_tensor(v) else v)
                                      for k, v in facts.items()}},
            }
            for c, inj in specs.items():
                _, clean = E.generate_answers(
                    model.decoder.llama, tok, memory.detach(), batch,
                    args.max_new_tokens, dev, memory_mask=mem_mask,
                    chat_template=ct, inject=inj)
                for i, s in enumerate(batch):
                    em = E.max_over_refs(clean[i], s.answer_refs, E.em_score)
                    con = E.max_over_refs(clean[i], s.answer_refs, E.containment_score)
                    rec = {"em": em, "con": con}
                    conds[c].append(rec); per_fam[c][fam].append(rec)

    print(f"\n=== graph_v6 read-utilization probe (step {step}, n={len(conds['REAL'])} samples) ===")
    print(f"{'condition':10s} {'EM':>7s} {'Contain':>9s}")
    for c in ["REAL", "OFF", "SHUFFLE"]:
        em, con = macro(conds[c])
        print(f"{c:10s} {em:7.1f} {con:9.1f}")
    print("\nper-family containment:")
    fams = list(by_fam.keys())
    print(f"{'cond':10s} " + " ".join(f"{f[:10]:>11s}" for f in fams))
    for c in ["REAL", "OFF", "SHUFFLE"]:
        print(f"{c:10s} " + " ".join(f"{macro(per_fam[c][f])[1]:11.1f}" for f in fams))
    print("\nreadout: OFF>REAL ⇒ inject harmful | REAL≈SHUFFLE ⇒ read not content-sensitive | REAL>SHUFFLE ⇒ read uses right facts")


if __name__ == "__main__":
    main()
