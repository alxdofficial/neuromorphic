#!/usr/bin/env python
"""Ingest a BOUNDED WildChat sample -> ``data/wildchat/{train,val}.jsonl`` (best-effort).

Streams a small N-conversation sample of LONG (``--min-turns``, default 6 user<->assistant exchanges)
real chat conversations from HuggingFace (``allenai/WildChat-1M``, falling back to the smaller ungated
``allenai/WildChat`` mirror if the -1M dump is unreachable) and writes each as a slim
``{"conversation": [{"role","content"}, ...]}`` jsonl row under ``data/wildchat/`` so ``WildChatSource``
can then load it fully offline (local jsonl takes priority over HF streaming). Only ``role in
{"user","assistant"}`` turns with non-empty content are kept; per-turn PII/moderation/IP metadata is
dropped (not needed downstream).

Usage:
    python scripts/data_build/ingest/wildchat/download.py [--n-train 3000] [--n-val 500]
                                                           [--min-turns 6] [--language English]

If HF is unreachable (both names) this exits with a clear error (no partial/half file); rerun once
online (or once you've accepted access, if the -1M dump ever becomes gated).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
HF_NAME = "allenai/WildChat-1M"
HF_NAME_FALLBACK = "allenai/WildChat"


def _slim_turns(turns: list) -> list:
    out = []
    for t in turns:
        role = (t.get("role") or "").lower()
        content = (t.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out


def stream_split(hf_name: str, n_docs: int, skip: int, min_turns: int, language: str):
    from datasets import load_dataset
    ds = load_dataset(hf_name, split="train", streaming=True)   # WildChat ships one 'train' split
    out, seen = [], 0
    for ex in ds:
        if language and (ex.get("language") or "") != language:
            continue
        if ex.get("toxic"):
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
    ap.add_argument("--hf-name", default=HF_NAME)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--min-turns", type=int, default=6)
    ap.add_argument("--language", default="English")
    args = ap.parse_args()

    out_dir = REPO / "data" / "wildchat"
    out_dir.mkdir(parents=True, exist_ok=True)

    errs, train, val, used_name = [], None, None, None
    for hf_name in (args.hf_name, HF_NAME_FALLBACK):
        try:
            train = stream_split(hf_name, args.n_train, 0, args.min_turns, args.language)
            val = stream_split(hf_name, args.n_val, args.n_train, args.min_turns, args.language)
            used_name = hf_name
            break
        except Exception as e:
            errs.append(f"{hf_name}: {type(e).__name__}: {str(e)[:160]}")
    if train is None:
        raise SystemExit(f"[wildchat] no usable HF source — " + " | ".join(errs) +
                          ". Restore network access and rerun.")

    for split, rows in (("train", train), ("val", val)):
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as fp:
            for r in rows:
                fp.write(json.dumps(r) + "\n")
        print(f"[wildchat] wrote {len(rows)} long conversations (from {used_name}, "
              f"min_turns={args.min_turns}) -> {path}")


if __name__ == "__main__":
    main()
