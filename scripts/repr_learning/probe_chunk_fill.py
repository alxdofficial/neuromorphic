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


def _classify_src(fam: str) -> str:
    """Map task_family string to high-level source bucket."""
    if fam == "hotpot_qa":
        return "hotpot"
    if fam == "narrative_qa":
        return "narrative"
    if fam == "musique":
        return "musique"
    if fam.startswith("babilong_"):
        return "babilong"
    return "composite"


def probe_one(chunk_size: int, n_batches: int, passages_per_chunk: int,
              tokenizer, cfg: ReprConfig,
              use_musique: bool, use_babilong: bool,
              babilong_config: str) -> dict:
    enabled = ["composite", "hotpot", "narrative"]
    weights = [0.4, 0.2, 0.2]
    if use_musique:
        enabled.append("musique"); weights.append(0.1)
    if use_babilong:
        enabled.append("babilong"); weights.append(0.1)
    # Renormalize the 5-tuple expected by the loader
    weights5 = [
        weights[0],
        weights[1],
        weights[2],
        weights[3] if use_musique else 0.0,
        weights[-1] if use_babilong else 0.0,
    ]

    print(f"\n=== chunk_size={chunk_size} (passages_per_chunk={passages_per_chunk}, "
          f"sources={enabled}) ===", flush=True)

    dl = make_mixed_qa_dataloader(
        cfg, tokenizer,
        composite_passages_path=REPO / "data/wave1/composite_v1/val/passages.jsonl",
        composite_questions_path=REPO / "data/wave1/composite_v1/val/questions.jsonl",
        use_hotpot=True, use_narrative=True,
        use_musique=use_musique, use_babilong=use_babilong,
        babilong_config=babilong_config,
        split="validation", chunk_size=chunk_size,
        passages_per_chunk=passages_per_chunk,
        weights=tuple(weights5), batch_size=4,
    )

    per_src = defaultdict(list)
    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        real_per_row = batch.context_mask.sum(dim=-1).tolist()
        for fam, real in zip(batch.task_family, real_per_row):
            per_src[_classify_src(fam)].append(real / chunk_size)

    print(f"{'source':<12} {'n':>5} {'fill avg':>10} {'fill min':>10} {'fill max':>10}")
    print("-" * 50)
    results = {}
    for src in enabled:
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
    ap.add_argument("--musique", action="store_true",
                    help="Also probe MuSiQue-Ans fill rates.")
    ap.add_argument("--babilong", action="store_true",
                    help="Also probe BABILong fill rates.")
    ap.add_argument("--babilong-config", type=str, default="auto",
                    help="BABILong length config or 'auto' to match each tranche.")
    args = ap.parse_args()

    print("Loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = ReprConfig()

    def auto_babilong_config(cs: int) -> str:
        if cs >= 16384: return "16k"
        if cs >= 8192:  return "8k"
        if cs >= 4096:  return "4k"
        if cs >= 2048:  return "2k"
        if cs >= 1024:  return "1k"
        return "0k"

    all_results = {}
    for cs in args.tranches:
        # Match the trainer's auto-scaling: 75 passages per 1024 tokens
        ppc = max(75, (cs // 1024) * 75)
        bcfg = auto_babilong_config(cs) if args.babilong_config == "auto" else args.babilong_config
        all_results[cs] = probe_one(
            cs, args.n_batches, ppc, tok, cfg,
            use_musique=args.musique,
            use_babilong=args.babilong,
            babilong_config=bcfg,
        )

    print("\n" + "=" * 80)
    print("SUMMARY (avg fill rate per source per tranche)")
    print("=" * 80)
    cols = ["composite", "hotpot", "narrative"]
    if args.musique:  cols.append("musique")
    if args.babilong: cols.append("babilong")
    header = "  ".join(f"{c:>10}" for c in cols)
    print(f"{'chunk_size':<12} {header}")
    for cs in args.tranches:
        r = all_results[cs]
        row = "  ".join(f"{r.get(c, {}).get('avg', float('nan')):>10.3f}" for c in cols)
        print(f"{cs:<12} {row}")


if __name__ == "__main__":
    main()
