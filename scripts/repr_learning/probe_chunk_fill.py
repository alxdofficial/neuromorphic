#!/usr/bin/env python3
"""Probe fill-rate per source for each tranche {4096, 8192, 16384}.

Measures (real-tokens / chunk_size) across a sample of batches per source.
A fill-rate of 1.0 means the chunk is completely filled with real tokens
(no padding waste). Below 0.85 → packing isn't doing its job and we're
training on padding (bad signal + wasted compute).

Usage:
  python scripts/repr_learning/probe_chunk_fill.py [--tranches 4096 8192 16384]
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.repr_learning.config import ReprConfig
from src.repr_learning.data_qa import make_mixed_qa_dataloader


def probe_one(chunk_size: int, n_batches: int, passages_per_chunk: int,
              tokenizer, cfg: ReprConfig) -> dict:
    print(f"\n=== chunk_size={chunk_size} (passages_per_chunk={passages_per_chunk}) ===", flush=True)

    dl = make_mixed_qa_dataloader(
        cfg, tokenizer,
        composite_passages_path=REPO / "data/wave1/composite_v1/val/passages.jsonl",
        composite_questions_path=REPO / "data/wave1/composite_v1/val/questions.jsonl",
        use_hotpot=True, use_narrative=True,
        split="validation", chunk_size=chunk_size,
        passages_per_chunk=passages_per_chunk,
        weights=(0.5, 0.25, 0.25), batch_size=4,
    )

    per_src = defaultdict(list)
    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        real_per_row = batch.context_mask.sum(dim=-1).tolist()
        for fam, real in zip(batch.task_family, real_per_row):
            # task_family for HotpotQA is "hotpot_qa", NarrativeQA is "narrative_qa",
            # composite is one of the composite task families.
            if fam == "hotpot_qa":
                src = "hotpot"
            elif fam == "narrative_qa":
                src = "narrative"
            else:
                src = "composite"
            per_src[src].append(real / chunk_size)

    print(f"{'source':<12} {'n':>5} {'fill avg':>10} {'fill min':>10} {'fill max':>10}")
    print("-" * 50)
    results = {}
    for src in ["composite", "hotpot", "narrative"]:
        vals = per_src.get(src, [])
        if not vals:
            print(f"{src:<12} {0:>5}  (no samples)")
            continue
        avg = sum(vals) / len(vals)
        mn = min(vals)
        mx = max(vals)
        print(f"{src:<12} {len(vals):>5} {avg:>10.3f} {mn:>10.3f} {mx:>10.3f}")
        results[src] = {"n": len(vals), "avg": avg, "min": mn, "max": mx}
        if avg < 0.80:
            print(f"  ⚠️  {src} under-fills at chunk_size={chunk_size} (avg {avg:.2f} < 0.80)")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranches", type=int, nargs="+", default=[4096, 8192, 16384])
    ap.add_argument("--n-batches", type=int, default=20)
    args = ap.parse_args()

    print("Loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = ReprConfig()

    all_results = {}
    for cs in args.tranches:
        # Match the trainer's auto-scaling: 75 passages per 1024 tokens
        ppc = max(75, (cs // 1024) * 75)
        all_results[cs] = probe_one(cs, args.n_batches, ppc, tok, cfg)

    print("\n" + "=" * 60)
    print("SUMMARY (avg fill rate per source per tranche)")
    print("=" * 60)
    print(f"{'chunk_size':<12} {'composite':>10} {'hotpot':>10} {'narrative':>10}")
    for cs in args.tranches:
        r = all_results[cs]
        c = r.get("composite", {}).get("avg", float("nan"))
        h = r.get("hotpot", {}).get("avg", float("nan"))
        n = r.get("narrative", {}).get("avg", float("nan"))
        print(f"{cs:<12} {c:>10.3f} {h:>10.3f} {n:>10.3f}")


if __name__ == "__main__":
    main()
