"""Preprocess a long-doc dataset (FineWeb-Edu, PG19, etc.) into
pre-tokenized parquet for Wave 1 TF-NTP training.

Output: parquet file with columns:
    - input_ids: List[int]   tokens (BOS-prefixed)
    - num_tokens: int        len(input_ids)
    - source: str            dataset name
    - source_id: str         original example id (best-effort)

Wave 1 trainer streams this file, packs documents into chunks of
`D * T_window` tokens, drops trailing partials.

Usage:
    python scripts/preprocess_longdoc.py fineweb-edu \\
        --output data/wave1/fineweb_edu.parquet \\
        --max-examples 1000 --min-tokens 4096 --streaming
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from src.trajectory_memory.data.tokenizer import get_tokenizer


# Map source name → HF metadata.
#
# Verified schemas (2026-05-08):
# - FineWeb-Edu sample-10BT: {text, id, dump, url, file_path, language,
#   language_score, token_count, score, int_score}. Already filtered for
#   English + Edu-classifier-score >= 3. Use `score` for further quality.
# - Wikipedia (en, 2023-11-01): {text, id, url, title}.
# - The Stack-dedup: GATED — needs HF token. Disabled by default.
# - PG19 (deepmind/pg19): BROKEN on current `datasets` (script-based dataset
#   support removed). Use `wikipedia-en` or `slimpajama-6b` for long-doc filler.
# - SlimPajama-6B (DKYoon): {text, meta} — mixed-domain, includes books.
_SOURCES = {
    "fineweb-edu": {
        "id": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "text_col": "text",
        "id_col": "id",
        "score_col": "score",       # Edu classifier; sample-10BT is pre-filtered to >= 3.
        "language_col": "language", # always "en" in sample-10BT.
    },
    "wikipedia-en": {
        "id": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "text_col": "text",
        "id_col": "id",
        "score_col": None,
        "language_col": None,
    },
    "slimpajama-6b": {
        "id": "DKYoon/SlimPajama-6B",
        "config": None,
        "split": "train",
        "text_col": "text",
        "id_col": "__index_level_0__",
        "score_col": None,
        "language_col": None,
    },
}


def iterate_examples(
    source: str, *, streaming: bool, max_examples: int | None,
) -> Iterable[dict]:
    info = _SOURCES[source]
    kwargs = {"path": info["id"], "split": info["split"], "streaming": streaming}
    if info["config"]:
        kwargs["name"] = info["config"]
    ds = load_dataset(**kwargs)
    for i, ex in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            return
        yield ex


def preprocess(
    source: str,
    *,
    output: Path,
    max_examples: int | None,
    min_tokens: int,
    max_tokens: int,
    streaming: bool,
    min_score: float | None = None,        # FineWeb-Edu: filter `score` column
    language_filter: str | None = None,    # e.g. "en" — only matters if dataset has it
    batch_size: int = 200,
) -> None:
    info = _SOURCES[source]
    tok = get_tokenizer()

    rows_input_ids = []
    rows_num_tokens = []
    rows_source = []
    rows_source_id = []

    n_seen = 0
    n_filtered_score = 0
    n_filtered_lang = 0
    n_kept = 0
    for ex in iterate_examples(
        source, streaming=streaming, max_examples=max_examples,
    ):
        n_seen += 1
        text = ex.get(info["text_col"], "")
        if not isinstance(text, str) or not text.strip():
            continue

        # Optional FineWeb-Edu score filter (default config has score >= 3
        # already; bump to 4+ for highest quality).
        if min_score is not None and info.get("score_col"):
            score = ex.get(info["score_col"])
            if score is None or score < min_score:
                n_filtered_score += 1
                continue

        # Optional language filter.
        if language_filter and info.get("language_col"):
            if ex.get(info["language_col"]) != language_filter:
                n_filtered_lang += 1
                continue

        ids = tok.encode(text, add_special_tokens=True)
        if len(ids) < min_tokens:
            continue
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]

        rows_input_ids.append(ids)
        rows_num_tokens.append(len(ids))
        rows_source.append(source)
        rows_source_id.append(str(ex.get(info["id_col"], "")))
        n_kept += 1

        if n_kept % 100 == 0:
            print(f"  [{source}] processed {n_seen:>6}, kept {n_kept:>6}, "
                  f"(filtered: score={n_filtered_score} lang={n_filtered_lang})")

    print(f"  [{source}] total: seen={n_seen}, kept={n_kept}, "
          f"(filtered: score={n_filtered_score} lang={n_filtered_lang})")
    if n_kept == 0:
        print("  [warn] no documents passed the length filter — output will be empty.")
        return

    # Write parquet.
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "input_ids": rows_input_ids,
        "num_tokens": rows_num_tokens,
        "source": rows_source,
        "source_id": rows_source_id,
    })
    pq.write_table(table, output)
    print(f"  [{source}] wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", choices=list(_SOURCES.keys()),
                    help="data source name")
    ap.add_argument("--output", type=Path, required=True,
                    help="output parquet path")
    ap.add_argument("--max-examples", type=int, default=None,
                    help="max examples to process (None = all)")
    ap.add_argument("--min-tokens", type=int, default=4096,
                    help="drop docs shorter than this (default 4096 — "
                         "filter for memory-stress training, plan §4.5)")
    ap.add_argument("--max-tokens", type=int, default=131072,
                    help="truncate docs longer than this (default 128K)")
    ap.add_argument("--streaming", action="store_true",
                    help="stream the dataset instead of fully loading "
                         "(necessary for fineweb-edu / wikipedia)")
    ap.add_argument("--min-score", type=float, default=None,
                    help="FineWeb-Edu only: filter `score` column (Edu classifier). "
                         "Default config sample-10BT is pre-filtered to >= 3; "
                         "pass 4 for highest quality, 3.5 for stricter than default.")
    ap.add_argument("--language", default=None,
                    help="filter by language code (FineWeb-Edu sample-10BT is "
                         "always 'en' so this is a no-op for that source).")
    args = ap.parse_args()

    preprocess(
        source=args.source,
        output=args.output,
        max_examples=args.max_examples,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        streaming=args.streaming,
        min_score=args.min_score,
        language_filter=args.language,
    )


if __name__ == "__main__":
    main()
