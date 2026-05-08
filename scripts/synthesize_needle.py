"""Generate synthetic needle-in-haystack documents for Wave 1 memory pretraining.

Per plan §4.5: 10% of Wave 1 data is synthetic — a fact planted at position
X, queried at position Y > X+2K. Forces the manifold to bridge the LM's
2K context cap or the NTP loss can't be reduced.

Pulls filler text from a long-doc source (FineWeb-Edu or PG19) and wraps
it with statement-then-query template. Cycles through several distance
buckets (3K / 8K / 16K / 32K) to provide curriculum.

Usage:
    python scripts/synthesize_needle.py \\
        --filler-source pg19 --output data/wave1/needle.parquet \\
        --num-docs 1000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from src.trajectory_memory.data.needle_haystack import (
    NeedleDoc, generate_needle_docs,
)
from src.trajectory_memory.data.tokenizer import get_tokenizer


_FILLER_SOURCES = {
    "fineweb-edu": {
        "id": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT",
        "split": "train", "text_col": "text",
    },
    "pg19": {
        "id": "deepmind/pg19", "config": None, "split": "train",
        "text_col": "text",
    },
    "wikipedia-en": {
        "id": "wikimedia/wikipedia", "config": "20231101.en",
        "split": "train", "text_col": "text",
    },
}


def iter_filler_text(
    source: str, *, streaming: bool, min_chars: int = 16000,
) -> Iterable[str]:
    info = _FILLER_SOURCES[source]
    kwargs = {"path": info["id"], "split": info["split"], "streaming": streaming}
    if info["config"]:
        kwargs["name"] = info["config"]
    ds = load_dataset(**kwargs)
    for ex in ds:
        text = ex.get(info["text_col"], "")
        if isinstance(text, str) and len(text) >= min_chars:
            yield text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--filler-source", choices=list(_FILLER_SOURCES.keys()),
                    default="pg19")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--num-docs", type=int, default=1000)
    ap.add_argument("--streaming", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--distances", nargs="+", type=int,
                    default=[3000, 8000, 16000, 32000],
                    help="target needle→query token distances (curriculum)")
    args = ap.parse_args()

    tok = get_tokenizer()
    rng = random.Random(args.seed)

    fillers = []
    print(f"Collecting {args.num_docs} fillers from {args.filler_source}...")
    for text in iter_filler_text(args.filler_source, streaming=args.streaming):
        fillers.append(text)
        if len(fillers) >= args.num_docs:
            break
    if len(fillers) < args.num_docs:
        print(f"  only got {len(fillers)} fillers (wanted {args.num_docs}); proceeding")

    rows_input_ids = []
    rows_num_tokens = []
    rows_target_distance = []
    rows_answer = []
    rows_needle_pos = []
    rows_query_pos = []

    for needle_doc in generate_needle_docs(
        fillers,
        target_distances=args.distances,
        seed=args.seed,
    ):
        ids = tok.encode(needle_doc.text, add_special_tokens=True)
        rows_input_ids.append(ids)
        rows_num_tokens.append(len(ids))
        rows_target_distance.append(needle_doc.target_distance)
        rows_answer.append(needle_doc.answer)
        # Convert char positions to approximate token positions (4 chars/tok).
        rows_needle_pos.append(needle_doc.needle_pos_chars // 4)
        rows_query_pos.append(needle_doc.query_pos_chars // 4)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "input_ids": rows_input_ids,
        "num_tokens": rows_num_tokens,
        "target_distance": rows_target_distance,
        "answer": rows_answer,
        "needle_pos_token": rows_needle_pos,
        "query_pos_token": rows_query_pos,
        "source": ["needle-haystack"] * len(rows_input_ids),
    })
    pq.write_table(table, args.output)
    print(f"  wrote {args.output} — {len(rows_input_ids)} docs, "
          f"{args.output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
