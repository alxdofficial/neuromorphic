#!/usr/bin/env python
"""Ingest a BOUNDED, ASCII-filtered LongCite sample -> ``data/longcite/{train,val}.jsonl`` (best-effort).

Streams ``zai-org/LongCite-45k`` (single upstream "train" split; ~half Chinese / half English SFT
distillation data), parses each row's templated ``prompt`` (chunk-tagged document ``<C0>...<Cn>`` plus
a trailing free-text question) and ``response`` (``<statement>...<cite>[s-e]</cite></statement>``
sentences) into ``{"chunks": [...], "question": ..., "answer": ..., "cited": [...]}`` jsonl, keeping
only roughly-English rows (ASCII-ratio heuristic -- cheap language-ID, see ``longcite.py`` source
docstring for the reasoning) so ``LongCiteSource`` can load it fully offline. Disjoint train/val =
train[0:n]/train[n:2n] valid rows (no genuine upstream val split to carve from instead).

Usage:
    python scripts/data_build/ingest/longcite/download.py [--n-train 2000] [--n-val 400]
                                                          [--hf-name zai-org/LongCite-45k]
                                                          [--min-ascii-ratio 0.85]

If HF is unreachable this exits with a clear error (no partial file); rerun once online.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

_DOC_RE = re.compile(r"\[Document Start\]\n(.*)\n\[Document End\]\n\n(.*)", re.S)
_CHUNK_SPLIT = re.compile(r"<C\d+>")
_STMT_RE = re.compile(r"<statement>(.*?)<cite>(.*?)</cite></statement>", re.S)
_CITE_RANGE_RE = re.compile(r"\[(\d+)-(\d+)\]")


def _ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) < 128) / len(text)


def _prompt_response_fields(ex: dict):
    """Canonical ``prompt``/``response``, or the English-filtered mirror's ``INSTRUCTION``/``RESPONSE``."""
    if "prompt" in ex:
        return ex.get("prompt") or "", ex.get("response") or ""
    return ex.get("INSTRUCTION") or "", ex.get("RESPONSE") or ""


def _parse_row(prompt: str, response: str):
    m = _DOC_RE.search(prompt or "")
    if not m:
        return None
    chunks = [c.strip() for c in _CHUNK_SPLIT.split(m.group(1))[1:] if c.strip()]
    question = m.group(2).strip()
    if not chunks or not question:
        return None
    statements, cited = [], set()
    for sm in _STMT_RE.finditer(response or ""):
        text = sm.group(1).strip()
        if text:
            statements.append(text)
        for a, b in _CITE_RANGE_RE.findall(sm.group(2)):
            for i in range(int(a), int(b) + 1):
                cited.add(i)
    answer = " ".join(statements).strip()
    if not answer:
        return None
    cited = sorted(i for i in cited if 0 <= i < len(chunks))
    return {"chunks": chunks, "question": question, "answer": answer, "cited": cited}


def stream_rows(hf_name: str, n_docs: int, skip: int, min_ascii_ratio: float):
    from datasets import load_dataset
    ds = load_dataset(hf_name, split="train", streaming=True)
    out, kept_before_skip, scanned = [], 0, 0
    scan_cap = max(50 * n_docs, 5000) + skip
    for ex in ds:
        scanned += 1
        prompt, response = _prompt_response_fields(ex)
        row = _parse_row(prompt, response)
        if row is None:
            continue
        # Language filter on QUESTION+ANSWER (clean natural-language text), not the raw document -- a
        # numeric/table-heavy Chinese document can score high-ASCII on the full prompt while its
        # question/answer are unambiguously Chinese, so filtering post-parse is more reliable.
        if min_ascii_ratio > 0 and _ascii_ratio(row["question"] + " " + row["answer"]) < min_ascii_ratio:
            continue
        if kept_before_skip < skip:            # disjoint val slice: skip the first `skip` valid rows
            kept_before_skip += 1
            continue
        out.append(row)
        if len(out) >= n_docs or scanned >= scan_cap:
            break
    return out, scanned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-name", default="zai-org/LongCite-45k")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--min-ascii-ratio", type=float, default=0.85)
    args = ap.parse_args()

    out_dir = REPO / "data" / "longcite"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        train, n_scanned_train = stream_rows(args.hf_name, args.n_train, 0, args.min_ascii_ratio)
        val, n_scanned_val = stream_rows(args.hf_name, args.n_val, args.n_train, args.min_ascii_ratio)
    except Exception as e:
        raise SystemExit(
            f"[longcite] HF dataset {args.hf_name!r} unreachable ({type(e).__name__}: {str(e)[:160]}). "
            f"Restore network access and rerun.")

    for split, rows, n_scanned in (("train", train, n_scanned_train), ("val", val, n_scanned_val)):
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as fp:
            for r in rows:
                fp.write(json.dumps(r) + "\n")
        print(f"[longcite] wrote {len(rows)} rows ({n_scanned} scanned, "
              f"ascii>={args.min_ascii_ratio} filter) -> {path}")


if __name__ == "__main__":
    main()
