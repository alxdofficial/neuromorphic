#!/usr/bin/env python
"""Ingest a BOUNDED GovReport sample -> ``data/govreport/{train,val}.jsonl`` (best-effort).

Streams a small N-doc sample of ``ccdv/govreport-summarization`` (long US government reports, ~9k
backbone tokens) and writes ``{"text": report, "summary": summary}`` jsonl under ``data/govreport/`` --
the ``report`` field is renamed to ``text`` so ``GovReportSource``'s local-cache path can reuse
``_corpus.py``'s generic loader unmodified (its live-HF fallback reads ``report`` directly, since
GovReport's raw HF column name isn't one ``_corpus.py``'s generic loader recognizes). ``summary`` rides
along unused for now, in case a future summarization-QA framing wants it without re-ingesting.

Usage:
    python scripts/data_build/ingest/govreport/download.py [--n-train 2000] [--n-val 400]

``ccdv/govreport-summarization`` has real train/validation splits, so val here is a genuine held-out
slice. If HF is unreachable this exits with a clear error (no partial file); rerun once online.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
HF_NAME = "ccdv/govreport-summarization"


def stream_split(hf_split: str, n_docs: int):
    from datasets import load_dataset
    ds = load_dataset(HF_NAME, split=hf_split, streaming=True)
    out = []
    for ex in ds:
        report = (ex.get("report") or "").strip()
        summary = (ex.get("summary") or "").strip()
        if not report:
            continue
        out.append({"text": report, "summary": summary})
        if len(out) >= n_docs:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=400)
    args = ap.parse_args()

    out_dir = REPO / "data" / "govreport"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        train = stream_split("train", args.n_train)
        val = stream_split("validation", args.n_val)
    except Exception as e:
        raise SystemExit(
            f"[govreport] HF dataset {HF_NAME!r} unreachable ({type(e).__name__}: {str(e)[:160]}). "
            f"Restore network access and rerun.")

    for split, docs in (("train", train), ("val", val)):
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as fp:
            for r in docs:
                fp.write(json.dumps(r) + "\n")
        print(f"[govreport] wrote {len(docs)} docs -> {path}")


if __name__ == "__main__":
    main()
