#!/usr/bin/env python
"""Ingest a BOUNDED, timestep-ORDERED WikiBigEdit sample → ``data/wikibigedit/{train,val}.jsonl``.

Streams a small per-timestep sample from HuggingFace (``lukasthede/WikiBigEdit`` — 8 chronological
JSON files, one per edit-timestep) and stages it as normalized jsonl rows (one per fact edit, with
all 5 probe fields) so ``WikiBigEditSource`` can load it fully offline. Timestep ORDER is preserved:
each row keeps ``"timestep"`` (0..7, chronological) so a Curriculum can place a lag between the edit
and its probe — the direct data fit for the T2 bounded-capacity forced-forgetting gate.

Usage:
    python scripts/data_build/ingest/wikibigedit/download.py [--n-train 3000] [--n-val 500]

train = the first ``ceil(n_train/8)`` rows of each timestep's stream; val = the NEXT
``ceil(n_val/8)`` rows of the SAME per-timestep stream (disjoint from train, order preserved).
If HF is unreachable this exits with a clear error (no partial/half file); rerun once online.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
HF_NAME = "lukasthede/WikiBigEdit"

TIMESTEP_FILES = [
    "wiki_big_edit_20240201_20240220.json",
    "wiki_big_edit_20240220_20240301.json",
    "wiki_big_edit_20240301_20240320.json",
    "wiki_big_edit_20240320_20240401.json",
    "wiki_big_edit_20240401_20240501.json",
    "wiki_big_edit_20240501_20240601.json",
    "wiki_big_edit_20240601_20240620.json",
    "wiki_big_edit_20240620_20240701.json",
]

_FIELDS = ("subject", "relation", "object", "ans", "rephrase", "loc", "loc_ans", "mhop", "mhop_ans",
           "update", "personas", "tag")
_CORE = ("subject", "relation", "object", "ans")


def _clean(v):
    return (v or "").strip() if isinstance(v, str) else (str(v) if v is not None else "")


def _normalize(ex: dict, timestep: int):
    row = {k: _clean(ex.get(k)) for k in _FIELDS}
    if not all(row[k] for k in _CORE):
        return None
    row["timestep"] = timestep
    return row


def stream_timestep(fname: str, timestep: int, skip: int, take: int):
    from datasets import load_dataset
    ds = load_dataset(HF_NAME, data_files=fname, split="train", streaming=True)
    out = []
    seen = 0
    for ex in ds:
        if seen < skip:
            seen += 1
            continue
        r = _normalize(ex, timestep)
        if r is not None:
            out.append(r)
        seen += 1
        if seen >= skip + take:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=500)
    args = ap.parse_args()

    n_ts = len(TIMESTEP_FILES)
    per_ts_train = -(-args.n_train // n_ts)      # ceil
    per_ts_val = -(-args.n_val // n_ts)

    out_dir = REPO / "data" / "wikibigedit"
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val = [], []
    try:
        for ts, fname in enumerate(TIMESTEP_FILES):
            train.extend(stream_timestep(fname, ts, skip=0, take=per_ts_train))
            val.extend(stream_timestep(fname, ts, skip=per_ts_train, take=per_ts_val))
    except Exception as e:
        raise SystemExit(
            f"[wikibigedit] HF dataset {HF_NAME!r} unreachable ({type(e).__name__}: {str(e)[:160]}). "
            f"Restore network access and rerun.")

    train = train[:args.n_train]
    val = val[:args.n_val]

    for split, rows in (("train", train), ("val", val)):
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as fp:
            for r in rows:
                fp.write(json.dumps(r) + "\n")
        n_mhop = sum(1 for r in rows if r.get("mhop"))
        print(f"[wikibigedit] wrote {len(rows)} edits ({n_mhop} with multi-hop probe) → {path}")


if __name__ == "__main__":
    main()
