"""WildChat real long multi-turn conversation source — a turn-position recall QA probe.

Source half of the ``qa`` task for the Phase-1 gap this line is missing: real, long, in-the-wild
user<->assistant conversations (not a synthetic story), so memory must bind content to a specific
EARLIER turn across many intervening turns — exactly the "recall across turns/sessions" regime
``project_data_plan`` calls the dominant gap. Each ``WildChat`` conversation becomes a ``QAItem``:
``facts`` = one line per turn ("User: ..."/"Assistant: ...", flattened+capped, see ``_chat.py``);
``question`` asks for a specific EARLY user turn by its ordinal position ("what was the user's message
#k?"); ``answer`` = that turn's exact (flattened) text — always verbatim-recoverable from the packed
facts, never a paraphrase, so it grades by plain EM/containment.

Filtered to LONG conversations only (``min_turns``, default 6 user<->assistant exchanges) — short
one-shot chats don't exercise cross-turn memory.

Resolution order (BEST-EFFORT, never hangs):
  1. Local ``data/wildchat/<split>.jsonl`` (``{"conversation":[{"role","content"},...]}`` per line) —
     ingest cache.
  2. Else HF-stream a BOUNDED sample of ``allenai/WildChat-1M`` (may need HF access — falls back to
     the smaller, ungated ``allenai/WildChat`` if the -1M dump is unreachable).
  3. Else (offline / both unreachable) raise a clear "run ingest first" error — no silent hang.

Ingest: ``scripts/data_build/ingest/wildchat/download.py``. See DATASETS.md / docs/history/docs/history/data_arch_plan.md (L1).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

from .base import Source, QAItem
from ._corpus import local_jsonl
from ._chat import render_turns, cap_by_tokens, pick_recall_target

HF_NAME = "allenai/WildChat-1M"
HF_NAME_FALLBACK = "allenai/WildChat"          # smaller, ungated mirror — used if the -1M dump 401s


def _local_jsonl(split: str) -> Optional[Path]:
    return local_jsonl("wildchat", split)


def _stream_rows(hf_name: str, n_docs: int, skip: int, min_turns: int, language: Optional[str]) -> List[list]:
    from datasets import load_dataset
    ds = load_dataset(hf_name, split="train", streaming=True)   # WildChat ships one 'train' split
    rows: List[list] = []
    seen = 0
    for ex in ds:
        if language and (ex.get("language") or "") != language:
            continue
        if ex.get("toxic"):
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


def _iter_hf_rows(split: str, n_docs: int, min_turns: int, language: Optional[str]) -> Tuple[List[list], str]:
    """Bounded HF stream, primary dataset then fallback. Disjoint train/val via a skip count (WildChat
    exposes only a 'train' split)."""
    skip = 0 if split == "train" else n_docs
    errs: List[str] = []
    for hf_name in (HF_NAME, HF_NAME_FALLBACK):
        try:
            rows = _stream_rows(hf_name, n_docs, skip, min_turns, language)
        except Exception as e:                                  # network / offline / gated
            errs.append(f"{hf_name}: {type(e).__name__}: {str(e)[:120]}")
            continue
        if rows:
            return rows, hf_name
        errs.append(f"{hf_name}: 0 conversations matched (min_turns={min_turns}, language={language!r})")
    raise RuntimeError(
        "[wildchat] no usable HF source — " + " | ".join(errs) + ". Run  python "
        "scripts/data_build/ingest/wildchat/download.py  to stage a local data/wildchat/"
        "{train,val}.jsonl sample, then retry (works fully offline once staged)."
    )


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
        origin = f"data/wildchat/{local.name}"
        if not rows:
            raise ValueError(f"[wildchat] {origin} has no conversation with >= {min_turns} turns each way "
                              f"(re-run the ingest script with a lower --min-turns, or raise n_docs).")
    else:
        rows, hf_name = _iter_hf_rows(split, n_docs, min_turns, language)
        origin = f"HF:{hf_name}"
    print(f"[data.wildchat] {split}: {len(rows)} long conversations from {origin} "
          f"(min_turns={min_turns})", flush=True)
    return rows, origin


class WildChatSource(Source):
    """Yields real WildChat conversations as QAItems (turn-position recall, see module docstring)."""

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
        """Rendered+capped (line, role, content) triples for one conversation (prefix-capped by
        token budget, so an answer target picked from the KEPT triples is always in-context)."""
        rendered = render_turns(turns, self.max_turn_chars)
        lines = [ln for ln, _r, _c in rendered]
        kept_lines = cap_by_tokens(lines, self.tok, self.max_dialogue_tokens)
        return rendered[:len(kept_lines)]

    def sample(self, rng, n: int) -> list:
        out: List[QAItem] = []
        tries = 0
        while len(out) < n and tries < n * 8:                 # bounded retries (some convos yield no target)
            tries += 1
            turns = self.rows[rng.randrange(len(self.rows))]
            kept = self._episode_lines(turns)
            target = pick_recall_target(kept, rng)
            if target is None:
                continue
            k, content = target
            q = f"In this conversation, what was the user's message #{k}?"
            out.append(QAItem(facts=[ln for ln, _r, _c in kept], question=q, answer=content,
                              task_id=0, meta={"dataset": "wildchat"}))
        return out

    def distractor_pool(self) -> list:
        return self._pool
