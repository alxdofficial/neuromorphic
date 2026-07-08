#!/usr/bin/env python
"""Ingest a BOUNDED LMSYS-Chat-1M sample -> ``data/lmsys_chat/{train,val}.jsonl`` (best-effort).

Streams a small N-conversation sample of LONG (``--min-turns``, default 6 user<->assistant exchanges)
real chat conversations from HuggingFace ``lmsys/lmsys-chat-1m`` and writes each as a slim
``{"conversation": [{"role","content"}, ...]}`` jsonl row under ``data/lmsys_chat/`` so
``LmsysChatSource`` can then load it fully offline (local jsonl takes priority over HF streaming).

**GATED DATASET**: ``lmsys/lmsys-chat-1m`` requires accepting the use-policy agreement on the Hub
(https://huggingface.co/datasets/lmsys/lmsys-chat-1m) and a logged-in token (``huggingface-cli login``
or ``HF_TOKEN``) before it can be streamed. If that hasn't been done this exits with a clear error —
no fake/synthetic data is ever substituted.

Usage:
    python scripts/data_build/ingest/lmsys_chat/download.py [--n-train 3000] [--n-val 500]
                                                             [--min-turns 6] [--language English]

If HF is unreachable/gated this exits with a clear error (no partial/half file); rerun once the
agreement is accepted + a token is set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
HF_NAME = "lmsys/lmsys-chat-1m"


def _slim_turns(turns: list) -> list:
    out = []
    for t in turns:
        role = (t.get("role") or "").lower()
        content = (t.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out


def stream_split(n_docs: int, skip: int, min_turns: int, language: str):
    from datasets import load_dataset
    ds = load_dataset(HF_NAME, split="train", streaming=True)   # lmsys-chat-1m ships one 'train' split
    out, seen = [], 0
    for ex in ds:
        if language and (ex.get("language") or "") != language:
            continue
        turns = _slim_turns(ex.get("conversation") or [])
        if len(turns) < 2 * min_turns:
            continue
        if seen < skip:
            seen += 1
            continue
        out.append({"conversation": turns})
        if len(out) >= n_docs:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--min-turns", type=int, default=6)
    ap.add_argument("--language", default="English")
    args = ap.parse_args()

    out_dir = REPO / "data" / "lmsys_chat"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        train = stream_split(args.n_train, 0, args.min_turns, args.language)
        val = stream_split(args.n_val, args.n_train, args.min_turns, args.language)
    except Exception as e:
        raise SystemExit(
            f"[lmsys_chat] HF dataset {HF_NAME!r} unreachable ({type(e).__name__}: {str(e)[:160]}). "
            f"This dataset is GATED — visit https://huggingface.co/datasets/{HF_NAME} to accept the "
            f"use-policy agreement, run `huggingface-cli login` (or set HF_TOKEN), then rerun.")

    for split, rows in (("train", train), ("val", val)):
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as fp:
            for r in rows:
                fp.write(json.dumps(r) + "\n")
        print(f"[lmsys_chat] wrote {len(rows)} long conversations (min_turns={args.min_turns}) -> {path}")


if __name__ == "__main__":
    main()
