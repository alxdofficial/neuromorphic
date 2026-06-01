"""Visual + quantitative data-sample audit for the v2.1 joint sweep.

For each data source we use (training and eval), pull a few real samples at the
production chunk size and check the things that actually break QA training/eval:

  1. input length     — does the packed context fill ~8k? how much is padding?
  2. answer present   — does the gold answer string actually appear in the
                        VISIBLE (non-pad) context the encoder sees?
  3. distractors/pads — how many passages are packed (sep-delimited)?
  4. coherence        — print question + gold + a context snippet so a human
                        can eyeball that the QA pair makes sense and the
                        supporting facts are really there.

Usage:
  python scripts/repr_learning/audit_data_samples.py --sources biographical ruler --n 4
  python scripts/repr_learning/audit_data_samples.py --sources hotpot musique narrative babilong --n 4 --split val

Sources: biographical hotpot narrative musique babilong ruler
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.repr_learning.config import ReprConfig  # noqa: E402
from src.repr_learning.data_qa import (  # noqa: E402
    QADataset, HotpotQADataset, NarrativeQADataset, MuSiQueDataset,
    BABILongDataset, RULERNIAHDataset, LoCoMoQADataset,
)

COMPOSITE_TRAIN_P = REPO / "data/wave1/composite_v1/train/passages.jsonl"
COMPOSITE_TRAIN_Q = REPO / "data/wave1/composite_v1/train/questions.jsonl"
COMPOSITE_VAL_P = REPO / "data/wave1/composite_v1/val/passages.jsonl"
COMPOSITE_VAL_Q = REPO / "data/wave1/composite_v1/val/questions.jsonl"


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_source(name: str, split: str, tok, cfg, chunk: int, bab_config: str):
    """Return an iterator over per-sample dicts for the named source."""
    if name == "biographical":
        pp = COMPOSITE_TRAIN_P if split == "train" else COMPOSITE_VAL_P
        qp = COMPOSITE_TRAIN_Q if split == "train" else COMPOSITE_VAL_Q
        ppc = max(75, (chunk // 1024) * 75)
        ds = QADataset(pp, qp, chunk_size=chunk, passages_per_chunk=ppc,
                       sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id,
                       task_weights={"biographical": 1.0}, seed=0)
    elif name == "hotpot":
        hp = "train" if split == "train" else "validation"
        ds = HotpotQADataset(split=hp, tokenizer=tok, chunk_size=chunk,
                             sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id, seed=1)
    elif name == "narrative":
        nq = "train" if split == "train" else "validation"
        ds = NarrativeQADataset(split=nq, tokenizer=tok, chunk_size=chunk,
                                sep_token_id=cfg.sep_token_id,
                                pad_token_id=cfg.pad_token_id, seed=2)
    elif name == "musique":
        mq = "train" if split == "train" else "validation"
        ds = MuSiQueDataset(split=mq, tokenizer=tok, chunk_size=chunk,
                            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id, seed=3)
    elif name == "babilong":
        ds = BABILongDataset(split=split, tokenizer=tok, chunk_size=chunk,
                             config_name=bab_config, sep_token_id=cfg.sep_token_id,
                             pad_token_id=cfg.pad_token_id, seed=4)
    elif name == "ruler":
        ds = RULERNIAHDataset(split=split, tokenizer=tok, chunk_size=chunk,
                              sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id, seed=5)
    elif name == "locomo":
        ds = LoCoMoQADataset(split=split, tokenizer=tok, chunk_size=chunk,
                             sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id, seed=6)
    else:
        raise ValueError(f"unknown source {name}")
    return iter(ds)


def audit_source(name: str, it, tok, cfg, chunk: int, n: int, pad_id: int, sep_id: int):
    print("\n" + "=" * 90)
    print(f"SOURCE: {name}   (chunk={chunk}, n={n})")
    print("=" * 90)
    fills, present_flags, n_passages = [], [], []
    for k in range(n):
        s = next(it)
        ctx = s["context_ids"]
        mask = s["context_mask"]
        valid_len = int(mask.sum().item())
        fill = valid_len / chunk
        valid_ids = ctx[:valid_len].tolist()
        n_seps = sum(1 for t in valid_ids if t == sep_id)
        ctx_text = tok.decode(valid_ids, skip_special_tokens=False)
        ctx_norm = _norm(ctx_text)

        q_text = tok.decode(s["question_ids"].tolist(), skip_special_tokens=True)
        refs = s.get("answer_refs") or [tok.decode(s["answer_ids"].tolist(), skip_special_tokens=True)]
        # answer present if ANY ref appears (normalized) in the visible context
        present = any(_norm(r) and _norm(r) in ctx_norm for r in refs)

        fills.append(fill)
        present_flags.append(present)
        n_passages.append(n_seps + 1)

        # locate first ref occurrence for a focused snippet
        snippet = ""
        for r in refs:
            rn = _norm(r)
            if rn and rn in ctx_norm:
                pos = ctx_text.lower().find(r.lower())
                if pos >= 0:
                    snippet = ctx_text[max(0, pos - 180):pos + len(r) + 120]
                break

        print(f"\n  --- sample {k+1} | family={s.get('task_family')} "
              f"qtype={s.get('question_type')} ---")
        print(f"  valid_len={valid_len}/{chunk}  fill={fill*100:.1f}%  "
              f"pad={100*(1-fill):.1f}%  passages(sep+1)={n_seps+1}")
        print(f"  Q:    {q_text.strip()[:300]}")
        print(f"  GOLD: {refs}")
        print(f"  ANSWER IN VISIBLE CONTEXT: {'YES' if present else '*** NO ***'}")
        if snippet:
            print(f"  …context@answer…: {snippet.strip()[:360]!r}")
        print(f"  context HEAD: {ctx_text[:240].strip()!r}")
        print(f"  context TAIL: {ctx_text[-160:].strip()!r}")

    nn = max(1, n)
    print(f"\n  >>> {name} SUMMARY: mean_fill={sum(fills)/nn*100:.1f}%  "
          f"answer_in_ctx={sum(present_flags)}/{n}  "
          f"mean_passages={sum(n_passages)/nn:.1f}")
    return {"source": name, "mean_fill": sum(fills) / nn,
            "answer_in_ctx": sum(present_flags), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+",
                    default=["biographical", "ruler"],
                    help="biographical hotpot narrative musique babilong ruler")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--babilong-config", default="8k")
    args = ap.parse_args()

    cfg = ReprConfig()
    print(f"Loading tokenizer {cfg.llama_model}...")
    tok = AutoTokenizer.from_pretrained(cfg.llama_model)

    summaries = []
    for name in args.sources:
        it = build_source(name, args.split, tok, cfg, args.chunk_size, args.babilong_config)
        summaries.append(audit_source(name, it, tok, cfg, args.chunk_size,
                                       args.n, cfg.pad_token_id, cfg.sep_token_id))

    print("\n" + "#" * 90)
    print("OVERALL AUDIT SUMMARY")
    print("#" * 90)
    print(f"  {'source':<16}{'mean_fill':>12}{'answer_in_ctx':>16}")
    for s in summaries:
        ratio = f"{s['answer_in_ctx']}/{s['n']}"
        print(f"  {s['source']:<16}{s['mean_fill']*100:>11.1f}%{ratio:>16}")


if __name__ == "__main__":
    main()
