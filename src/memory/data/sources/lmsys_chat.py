"""LMSYS-Chat-1M real long multi-turn conversation source — a turn-position recall QA probe.

Same shaper/rendering as ``wildchat.py`` (see that module's docstring for the full design rationale —
"recall the user's earlier message #k" as a verbatim, EM-gradeable target) applied to a second,
independently-collected real-conversation corpus (Chatbot-Arena-style multi-model chat logs rather
than WildChat's ChatGPT-only logs), for source variety on the same task family.

HF ``lmsys/lmsys-chat-1m`` is a **GATED** dataset — it requires accepting the dataset's use-policy
agreement on the Hub (and a logged-in ``HF_TOKEN``) before it can be streamed. This module builds the
correct loader either way; if the agreement hasn't been accepted it raises a clear, actionable error
(no fake/synthetic data) pointing at the dataset page + the ingest script.

Resolution order (BEST-EFFORT, never hangs):
  1. Local ``data/lmsys_chat/<split>.jsonl`` (``{"conversation":[{"role","content"},...]}`` per line) —
     ingest cache (works fully offline once staged, even without ever accepting the gate yourself, as
     long as whoever ran the ingest script had access).
  2. Else HF-stream a BOUNDED sample of ``lmsys/lmsys-chat-1m`` (needs the accepted agreement + token).
  3. Else (gated / offline) raise a clear "accept the agreement, then run ingest" error.

Ingest: ``scripts/data_build/ingest/lmsys_chat/download.py``. See DATASETS.md / docs/data_arch_plan.md (L1).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

from .base import Source, QAItem
from ._corpus import local_jsonl
from ._chat import render_turns, cap_by_tokens, pick_recall_target

HF_NAME = "lmsys/lmsys-chat-1m"


def _local_jsonl(split: str) -> Optional[Path]:
    return local_jsonl("lmsys_chat", split)


def _stream_rows(n_docs: int, skip: int, min_turns: int, language: Optional[str]) -> List[list]:
    from datasets import load_dataset
    ds = load_dataset(HF_NAME, split="train", streaming=True)   # lmsys-chat-1m ships one 'train' split
    rows: List[list] = []
    seen = 0
    for ex in ds:
        if language and (ex.get("language") or "") != language:
            continue
        turns = ex.get("conversation") or []
        if len(turns) < 2 * min_turns:                          # 2× — "turn" pairs a user+assistant msg
            continue
        if seen < skip:
            seen += 1
            continue
        rows.append(turns)
        if len(rows) >= n_docs:
            break
    return rows


def _iter_hf_rows(split: str, n_docs: int, min_turns: int, language: Optional[str]) -> List[list]:
    """Bounded HF stream. Disjoint train/val via a skip count (lmsys-chat-1m exposes only 'train')."""
    skip = 0 if split == "train" else n_docs
    try:
        rows = _stream_rows(n_docs, skip, min_turns, language)
    except Exception as e:                                       # gated (401/403) / offline
        raise RuntimeError(
            f"[lmsys_chat] HF dataset {HF_NAME!r} unreachable ({type(e).__name__}: {str(e)[:160]}). "
            f"This dataset is GATED — visit https://huggingface.co/datasets/{HF_NAME} to accept the "
            f"use-policy agreement, run `huggingface-cli login` (or set HF_TOKEN) so the accepted "
            f"agreement is visible, then run  python scripts/data_build/ingest/lmsys_chat/download.py  "
            f"to stage a local data/lmsys_chat/{{train,val}}.jsonl sample (loads fully offline once "
            f"staged, no further access needed)."
        ) from e
    if not rows:
        raise ValueError(f"[lmsys_chat] 0 conversations matched from HF:{HF_NAME} "
                          f"(min_turns={min_turns}, language={language!r}, split={split!r}).")
    return rows


def _load_rows(split: str, n_docs: int, min_turns: int, language: Optional[str]) -> Tuple[List[list], str]:
    local = _local_jsonl(split)
    if local is not None:
        rows: List[list] = []
        with open(local) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                turns = o.get("conversation") or []
                if len(turns) < 2 * min_turns:
                    continue
                rows.append(turns)
                if len(rows) >= n_docs:
                    break
        origin = f"data/lmsys_chat/{local.name}"
        if not rows:
            raise ValueError(f"[lmsys_chat] {origin} has no conversation with >= {min_turns} turns each "
                              f"way (re-run the ingest script with a lower --min-turns, or raise n_docs).")
    else:
        rows = _iter_hf_rows(split, n_docs, min_turns, language)
        origin = f"HF:{HF_NAME}"
    print(f"[data.lmsys_chat] {split}: {len(rows)} long conversations from {origin} "
          f"(min_turns={min_turns})", flush=True)
    return rows, origin


class LmsysChatSource(Source):
    """Yields real lmsys-chat-1m conversations as QAItems (turn-position recall, see module docstring)."""

    kind = "qa"
    pack_n_queries = (1, 2)

    def __init__(self, tokenizer, *, split: str = "train", seed: int = 0, n_docs: int = 3000,
                 min_turns: int = 6, language: Optional[str] = "English",
                 max_dialogue_tokens: int = 900, max_turn_chars: int = 320,
                 pool_cap: int = 2000, **kw):
        self.tok = tokenizer
        self.seed = seed
        self.max_dialogue_tokens = max_dialogue_tokens
        self.max_turn_chars = max_turn_chars
        self.rows, _origin = _load_rows(split, n_docs, min_turns, language)

        pool: List[str] = []
        for turns in self.rows:
            pool.extend(line for line, _role, _content in render_turns(turns, max_turn_chars))
        if len(pool) > pool_cap:
            pool = random.Random(seed).sample(pool, pool_cap)
        self._pool = pool

    def _episode_lines(self, turns: list) -> List[Tuple[str, str, str]]:
        rendered = render_turns(turns, self.max_turn_chars)
        lines = [ln for ln, _r, _c in rendered]
        kept_lines = cap_by_tokens(lines, self.tok, self.max_dialogue_tokens)
        return rendered[:len(kept_lines)]

    def sample(self, rng, n: int) -> list:
        out: List[QAItem] = []
        tries = 0
        while len(out) < n and tries < n * 8:
            tries += 1
            turns = self.rows[rng.randrange(len(self.rows))]
            kept = self._episode_lines(turns)
            target = pick_recall_target(kept, rng)
            if target is None:
                continue
            k, content = target
            q = f"In this conversation, what was the user's message #{k}?"
            out.append(QAItem(facts=[ln for ln, _r, _c in kept], question=q, answer=content,
                              task_id=0, meta={"dataset": "lmsys_chat"}))
        return out

    def distractor_pool(self) -> list:
        return self._pool
