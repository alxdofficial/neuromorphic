#!/usr/bin/env python
"""Ingest a BOUNDED Multi-Session-Chat sample -> ``data/msc/{train,val}.jsonl`` (best-effort).

HF ``nayohan/multi_session_chat`` ships each conversation as several session ROWS sharing a
``dialoug_id`` [sic — spelled that way upstream], ordered by ``session_id``. This groups rows into
conversations (a group finalizes when the id changes — the live stream arrives already grouped/ordered,
verified empirically), keeps only conversations with >= ``--min-sessions`` sessions (true multi-session
recall needs at least an early + a later session), sorts each conversation's sessions by ``session_id``,
and writes one jsonl row per conversation: ``{"sessions": [{"session_id","persona1","persona2","speaker",
"dialogue"}, ...]}`` under ``data/msc/`` so ``MSCSource`` can then load it fully offline.

``nayohan/multi_session_chat`` exposes real ``train``/``validation`` splits (used directly — no
skip-carved val needed).

Usage:
    python scripts/data_build/ingest/msc/download.py [--n-train 3000] [--n-val 500] [--min-sessions 2]

If HF is unreachable this exits with a clear error (no partial/half file); rerun once online.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
HF_NAME = "nayohan/multi_session_chat"


def _slim_session(ex: dict) -> dict:
    return {
        "session_id": ex.get("session_id"),
        "persona1": [str(s).strip() for s in (ex.get("persona1") or []) if str(s).strip()],
        "persona2": [str(s).strip() for s in (ex.get("persona2") or []) if str(s).strip()],
        "speaker": [str(s) for s in (ex.get("speaker") or [])],
        "dialogue": [str(u).strip() for u in (ex.get("dialogue") or [])],
    }


def _group_conversations(rows_iter, max_conv: int):
    """Group session rows by ``dialoug_id``, finalizing a group once the id changes."""
    cur_id = None
    cur = []
    n_yielded = 0
    for ex in rows_iter:
        did = ex.get("dialoug_id")
        if cur_id is not None and did != cur_id:
            yield cur_id, cur
            n_yielded += 1
            cur = []
            if n_yielded >= max_conv:
                return
        cur_id = did
        cur.append(ex)
    if cur and n_yielded < max_conv:
        yield cur_id, cur


def stream_conversations(hf_split: str, n_conv: int, min_sessions: int):
    from datasets import load_dataset
    ds = load_dataset(HF_NAME, split=hf_split, streaming=True)
    out = []
    scan_cap = max(8 * n_conv, 500)                    # generous — ~35% of groups are single-session
    for _did, rows in _group_conversations(ds, max_conv=scan_cap):
        sessions = sorted((_slim_session(r) for r in rows), key=lambda s: s["session_id"])
        if len(sessions) < min_sessions:
            continue
        out.append(sessions)
        if len(out) >= n_conv:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--min-sessions", type=int, default=2)
    args = ap.parse_args()

    out_dir = REPO / "data" / "msc"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        train = stream_conversations("train", args.n_train, args.min_sessions)
        val = stream_conversations("validation", args.n_val, args.min_sessions)
    except Exception as e:
        raise SystemExit(
            f"[msc] HF dataset {HF_NAME!r} unreachable ({type(e).__name__}: {str(e)[:160]}). "
            f"Restore network access and rerun.")

    for split, convs in (("train", train), ("val", val)):
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as fp:
            for sessions in convs:
                fp.write(json.dumps({"sessions": sessions}) + "\n")
        print(f"[msc] wrote {len(convs)} multi-session conversations "
              f"(>= {args.min_sessions} sessions) -> {path}")


if __name__ == "__main__":
    main()
