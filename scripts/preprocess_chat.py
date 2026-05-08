"""Preprocess a multi-turn chat dataset (UltraChat / WildChat / etc.) into
pre-tokenized TurnPair parquet for Wave 2 + Wave 4 training.

Output: parquet with columns:
    - prior_ids:    List[int]
    - response_ids: List[int]
    - num_prior:    int
    - num_response: int
    - source:       str
    - session_id:   str | None
    - turn_index:   int

Filters: only TurnPairs with prior_length >= min_prior_tokens (defaults
to 4096, per plan §4.5 — drops short pairs that wouldn't exercise memory).

Usage:
    python scripts/preprocess_chat.py wildchat-1m \\
        --output data/wave2/wildchat_long.parquet \\
        --max-sessions 5000 --min-prior 4096 --streaming
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from src.trajectory_memory.data.tokenizer import get_tokenizer
from src.trajectory_memory.data.turn_pair import session_to_turn_pairs


# source → (HF id, config, split, message-extractor, optional filter fields).
#
# Note on schema specifics (verified 2026-05-08 against live datasets):
# - UltraChat-200k: 207865 train_sft, 23110 test_sft (also train_gen / test_gen).
#   `messages` is strictly alternating user/assistant, 4-12 turns typical.
# - WildChat-1M: single "train" split (~1M sessions). MULTILINGUAL —
#   `language` is e.g. "English" / "Spanish" (NOT ISO codes). Has per-session
#   `toxic` and `redacted` flags; only ~30% are English + non-toxic + ≥4 turns.
# - SmolTalk: mostly 2-turn instruction-tuning data, no stable session id.
_SOURCES = {
    "ultrachat-200k": {
        "id": "HuggingFaceH4/ultrachat_200k",
        "config": None,
        "split": "train_sft",
        "messages_col": "messages",
        "session_id_col": "prompt_id",
        "language_col": None,           # already English-only
        "toxic_col": None,
    },
    "wildchat-1m": {
        "id": "allenai/WildChat-1M",
        "config": None,
        "split": "train",
        "messages_col": "conversation",
        "session_id_col": "conversation_hash",
        "language_col": "language",     # values like "English", "Spanish"
        "toxic_col": "toxic",           # bool
    },
    "smoltalk": {
        "id": "HuggingFaceTB/smoltalk",
        "config": "all",
        "split": "train",
        "messages_col": "messages",
        "session_id_col": None,
        "language_col": None,
        "toxic_col": None,
    },
}


def iterate_sessions(
    source: str, *, streaming: bool, max_sessions: int | None,
) -> Iterable[dict]:
    info = _SOURCES[source]
    kwargs = {"path": info["id"], "split": info["split"], "streaming": streaming}
    if info["config"]:
        kwargs["name"] = info["config"]
    ds = load_dataset(**kwargs)
    for i, ex in enumerate(ds):
        if max_sessions is not None and i >= max_sessions:
            return
        yield ex


def normalize_messages(raw: list, source: str) -> list[dict]:
    """Normalize various dataset message schemas into [{"role", "content"}]."""
    out = []
    for m in raw:
        if isinstance(m, dict) and "role" in m and "content" in m:
            out.append({"role": m["role"], "content": m["content"]})
        elif isinstance(m, dict) and "from" in m and "value" in m:
            # vicuna-style {"from": "human", "value": "..."}
            role = "user" if m["from"] in ("human", "user") else "assistant"
            out.append({"role": role, "content": m["value"]})
    return out


def preprocess(
    source: str,
    *,
    output: Path,
    max_sessions: int | None,
    min_prior_tokens: int,
    max_response_tokens: int,
    streaming: bool,
    language_filter: str | None = None,    # e.g. "English" for WildChat
    exclude_toxic: bool = True,
) -> None:
    info = _SOURCES[source]
    tok = get_tokenizer()

    rows_prior = []
    rows_response = []
    rows_num_prior = []
    rows_num_response = []
    rows_source = []
    rows_session = []
    rows_turn = []

    n_sessions = 0
    n_filtered_lang = 0
    n_filtered_toxic = 0
    n_pairs = 0
    for ex in iterate_sessions(
        source, streaming=streaming, max_sessions=max_sessions,
    ):
        n_sessions += 1

        # Filter: language (only WildChat has this column; UltraChat / SmolTalk
        # are mono-lingual already).
        if language_filter and info["language_col"]:
            if ex.get(info["language_col"]) != language_filter:
                n_filtered_lang += 1
                continue

        # Filter: toxic flag (WildChat only).
        if exclude_toxic and info["toxic_col"]:
            if ex.get(info["toxic_col"]):
                n_filtered_toxic += 1
                continue

        raw_msgs = ex.get(info["messages_col"]) or []
        msgs = normalize_messages(raw_msgs, source)
        if not msgs:
            continue
        session_id = (
            str(ex.get(info["session_id_col"])) if info["session_id_col"]
            else None
        )

        pairs = session_to_turn_pairs(
            msgs, tokenizer=tok, source=source,
            session_id=session_id,
            min_prior_tokens=min_prior_tokens,
            max_response_tokens=max_response_tokens,
        )
        for tp in pairs:
            rows_prior.append(tp.prior_ids)
            rows_response.append(tp.response_ids)
            rows_num_prior.append(len(tp.prior_ids))
            rows_num_response.append(len(tp.response_ids))
            rows_source.append(tp.source)
            rows_session.append(tp.session_id or "")
            rows_turn.append(tp.turn_index or 0)
            n_pairs += 1

        if n_sessions % 200 == 0:
            print(f"  [{source}] sessions={n_sessions:>6} pairs={n_pairs:>6} "
                  f"(filtered: lang={n_filtered_lang} toxic={n_filtered_toxic})")

    print(f"  [{source}] total sessions={n_sessions} pairs={n_pairs} "
          f"(filtered: lang={n_filtered_lang} toxic={n_filtered_toxic})")
    if n_pairs == 0:
        print("  [warn] no pairs passed the length filter — output empty.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "prior_ids": rows_prior,
        "response_ids": rows_response,
        "num_prior": rows_num_prior,
        "num_response": rows_num_response,
        "source": rows_source,
        "session_id": rows_session,
        "turn_index": rows_turn,
    })
    pq.write_table(table, output)
    print(f"  [{source}] wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", choices=list(_SOURCES.keys()))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--min-prior", type=int, default=4096,
                    help="filter: drop pairs with prior_tokens < this (plan §4.5)")
    ap.add_argument("--max-response", type=int, default=2048,
                    help="truncate response to this many tokens")
    ap.add_argument("--language", default="English",
                    help="filter to this language (WildChat only; "
                         "UltraChat/SmolTalk ignore this). "
                         "WildChat values: 'English', 'Spanish', etc.")
    ap.add_argument("--include-toxic", action="store_true",
                    help="include sessions flagged toxic (WildChat only). "
                         "Default: exclude toxic.")
    ap.add_argument("--streaming", action="store_true")
    args = ap.parse_args()

    preprocess(
        source=args.source,
        output=args.output,
        max_sessions=args.max_sessions,
        min_prior_tokens=args.min_prior,
        max_response_tokens=args.max_response,
        streaming=args.streaming,
        language_filter=args.language if args.language else None,
        exclude_toxic=not args.include_toxic,
    )


if __name__ == "__main__":
    main()
