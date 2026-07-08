#!/usr/bin/env python
"""Ingest a BOUNDED MultiWOZ 2.2 sample -> ``data/multiwoz/{train,val}.jsonl`` (best-effort).

The HF ``multi_woz_v22`` dataset ships as a loading *script* that ``datasets>=3`` refuses to run, so
we read the raw MultiWOZ 2.2 dialogue JSON the script points at (GitHub ``budzianowski/multiwoz``) and
write SLIM ``{"dialogue_id","lines","state"}`` records (the same shape ``MultiWOZSource`` produces from
GitHub) so the source loads them fully offline. ``lines`` = ``User:/Assistant:`` turn strings;
``state`` = accumulated ``{"domain-slot": value}`` belief state. Slot-recall QA (value stated verbatim
in the dialogue = un-guessable exact recall) is constructed by the source at sample time.

Usage:
    python scripts/data_build/ingest/multiwoz/download.py [--n-train 3000] [--n-val 500]

If GitHub is unreachable this exits with a clear error; rerun once online.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=500)
    args = ap.parse_args()

    # Reuse the source's fetch+parse (single source of truth). No local cache exists yet on a fresh
    # ingest, so iter_slim_records fetches the raw GitHub shards.
    from src.memory.data.sources.multiwoz import iter_slim_records

    out_dir = REPO / "data" / "multiwoz"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        train = list(iter_slim_records("train", args.n_train))
        val = list(iter_slim_records("validation", args.n_val))
    except Exception as e:
        raise SystemExit(f"[multiwoz] raw MultiWOZ 2.2 unreachable ({type(e).__name__}: "
                         f"{str(e)[:160]}). Restore network access and rerun.")

    for split, recs in (("train", train), ("val", val)):
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as fp:
            for r in recs:
                fp.write(json.dumps(r) + "\n")
        print(f"[multiwoz] wrote {len(recs)} dialogues -> {path}")


if __name__ == "__main__":
    main()
