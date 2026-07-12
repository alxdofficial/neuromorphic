"""TriviaQA (reading-comprehension) source — a real question + answer-containing web/wiki evidence.

Source half of the ``qa`` task for open-domain RC: each TriviaQA-``rc`` example is a trivia question,
a canonical answer ``value`` (+ ``aliases`` for EM), and evidence text (Wikipedia ``wiki_context`` and/or
web ``search_context``). Evidence is long and noisy, so we DON'T emit it whole: we keep an
answer-centred window (~``max_ctx_tokens``) of the first evidence document that actually CONTAINS the
answer value, sentence-split so the answer sits intact inside one fact. Examples whose evidence never
contains the answer value are skipped at load time (so the answer-in-context invariant holds by
construction). ``aliases`` ride along in ``meta`` for downstream exact-match scoring.

Resolution order (BEST-EFFORT, never hangs):
  1. Local ``data/triviaqa/<split>.jsonl`` (``{"context","question","answer","aliases"}``) — ingest cache.
  2. Else HF-stream a BOUNDED, answer-filtered sample of ``mandarjoshi/trivia_qa`` config ``rc``.
  3. Else (offline) raise a clear "run ingest first" error — no silent hang.

Ingest: ``scripts/data_build/ingest/triviaqa/download.py``. See DATASETS.md / docs/DATA.md (L1).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple

from .base import Source, QAItem
from ._corpus import local_jsonl

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

HF_NAME = "mandarjoshi/trivia_qa"
HF_CONFIG = "rc"

Row = Tuple[List[str], str, str, List[str]]  # (facts, question, answer, aliases)


def _local_jsonl(split: str) -> Optional[Path]:
    return local_jsonl("triviaqa", split)


def _sent_spans(text: str) -> List[List[int]]:
    """Char (start, end) spans of the sentences in ``text`` (positions preserved, unlike split())."""
    spans: List[List[int]] = []
    start = 0
    for m in _SENT_SPLIT.finditer(text):
        spans.append([start, m.start()])
        start = m.end()
    spans.append([start, len(text)])
    return spans


def _sent_split_answer_safe(snippet: str, value: str) -> List[str]:
    """Sentence-split ``snippet`` but MERGE the sentences the answer straddles, so ``value`` survives
    intact (with original spacing) inside a single fact — otherwise ``St. Louis`` / ``J. K. Rowling``
    would be torn across sentences and no longer match as a contiguous substring."""
    spans = _sent_spans(snippet)
    p = snippet.lower().find(value.lower())
    if p >= 0:
        q = p + len(value)
        lo = hi = None
        for i, (a, b) in enumerate(spans):
            if a < q and b > p:            # sentence overlaps the answer span
                lo = i if lo is None else lo
                hi = i
        if lo is not None and hi > lo:     # answer straddles a boundary → merge lo..hi (raw slice)
            spans = spans[:lo] + [[spans[lo][0], spans[hi][1]]] + spans[hi + 1:]
    return [snippet[a:b].strip() for a, b in spans if snippet[a:b].strip()]


def _cap_facts(facts: List[str], ai: int, tok, max_tokens: int) -> List[str]:
    """Grow a sentence window outward from the answer fact ``ai`` while it fits ``max_tokens``."""
    lens = [len(tok(f + "\n", add_special_tokens=False).input_ids) for f in facts]
    lo = hi = ai
    total = lens[ai]
    while True:
        grew = False
        if hi + 1 < len(facts) and total + lens[hi + 1] <= max_tokens:
            hi += 1
            total += lens[hi]
            grew = True
        if lo - 1 >= 0 and total + lens[lo - 1] <= max_tokens:
            lo -= 1
            total += lens[lo]
            grew = True
        if not grew:
            break
    return facts[lo:hi + 1]


def _build_facts(text: str, value: str, tok, max_tokens: int) -> Optional[List[str]]:
    """Answer-centred, sentence-split, token-capped facts from one evidence ``text`` — or None if the
    answer value isn't present (caller then tries the next evidence doc / skips the example)."""
    low, vl = (text or "").lower(), value.lower()
    p = low.find(vl)
    if p < 0:
        return None
    char_pad = max_tokens * 2               # ~4·max_tokens chars total → snippet safely < 1024 tok
    start = max(0, p - char_pad)
    end = min(len(text), p + len(value) + char_pad)
    snippet = text[start:end].strip()
    facts = _sent_split_answer_safe(snippet, value)
    if not facts:
        return None
    ai = next((i for i, f in enumerate(facts) if vl in f.lower()), None)
    if ai is None:                          # answer-safe split failed (shouldn't happen) → skip
        return None
    facts = _cap_facts(facts, ai, tok, max_tokens)
    if not any(vl in f.lower() for f in facts):
        return None
    return facts


def _evidence_texts(ex: dict) -> List[str]:
    """All evidence documents for an example: entity (wiki) pages first, then web search contexts."""
    ep = ex.get("entity_pages") or {}
    sr = ex.get("search_results") or {}
    return list(ep.get("wiki_context") or []) + list(sr.get("search_context") or [])


def _row_from_ex(ex: dict, tok, max_tokens: int) -> Optional[Row]:
    """Build a Row from a raw HF example: window the FIRST evidence doc that contains the answer value."""
    q = (ex.get("question") or "").strip()
    ans = ex.get("answer") or {}
    value = (ans.get("value") or "").strip()
    aliases = [a for a in (ans.get("aliases") or []) if a]
    if not q or not value:
        return None
    for text in _evidence_texts(ex):
        if value.lower() in (text or "").lower():
            facts = _build_facts(text, value, tok, max_tokens)
            if facts:
                return (facts, q, value, aliases)
    return None


def _iter_hf_rows(split: str, n_docs: int, tok, max_tokens: int) -> List[Row]:
    """Stream + answer-filter up to ``n_docs`` valid rows from HF trivia_qa/rc (bounded scan)."""
    hf_split = "train" if split == "train" else "validation"
    try:
        from datasets import load_dataset
        ds = load_dataset(HF_NAME, HF_CONFIG, split=hf_split, streaming=True)
    except Exception as e:  # network / offline / missing dataset
        raise RuntimeError(
            f"[triviaqa] HF dataset {HF_NAME!r}/{HF_CONFIG} unreachable ({type(e).__name__}: "
            f"{str(e)[:120]}). Run  python scripts/data_build/ingest/triviaqa/download.py  to stage a "
            f"local data/triviaqa/{{train,val}}.jsonl sample, then retry (offline once staged)."
        ) from e
    rows: List[Row] = []
    scanned, scan_cap = 0, max(50 * n_docs, 5000)   # bounded: don't scan forever if evidence is thin
    for ex in ds:
        scanned += 1
        r = _row_from_ex(ex, tok, max_tokens)
        if r is not None:
            rows.append(r)
        if len(rows) >= n_docs or scanned >= scan_cap:
            break
    print(f"[data.triviaqa] {split}: kept {len(rows)}/{scanned} scanned "
          f"(answer-in-evidence filter) from HF:{HF_NAME}/{HF_CONFIG}", flush=True)
    return rows


def _iter_local_rows(path: Path, n_docs: int, tok, max_tokens: int) -> List[Row]:
    """Load staged ``{context,question,answer,aliases}`` jsonl → re-window to answer-centred facts."""
    rows: List[Row] = []
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            value = (o.get("answer") or "").strip()
            q = (o.get("question") or "").strip()
            ctx = o.get("context") or ""
            aliases = [a for a in (o.get("aliases") or []) if a]
            if not value or not q:
                continue
            facts = _build_facts(ctx, value, tok, max_tokens)
            if facts:
                rows.append((facts, q, value, aliases))
            if len(rows) >= n_docs:
                break
    print(f"[data.triviaqa] loaded {len(rows)} rows from {path}", flush=True)
    return rows


class TriviaQASource(Source):
    """Yields TriviaQA questions with answer-containing evidence facts + a small distractor pool."""

    kind = "qa"

    def __init__(self, tokenizer, *, split: str = "train", seed: int = 0, n_docs: int = 3000,
                 max_ctx_tokens: int = 800, pool_cap: int = 2000, **kw):
        self.tok = tokenizer
        self.split = split
        self.seed = seed
        self.max_ctx_tokens = max_ctx_tokens
        local = _local_jsonl(split)
        if local is not None:
            self.rows = _iter_local_rows(local, n_docs, tokenizer, max_ctx_tokens)
        else:
            self.rows = _iter_hf_rows(split, n_docs, tokenizer, max_ctx_tokens)
        if not self.rows:
            raise ValueError(f"[triviaqa] no answer-containing rows for split={split}.")
        # Small distractor pool from OTHER rows' evidence sentences (the qa task NEEDS a non-empty pool
        # to pad short-tail episodes; caps-name overlap with the gold facts is rejected there).
        pool: List[str] = []
        seen = set()
        for facts, _q, _a, _al in self.rows:
            for s in facts:
                if len(s) < 25 or s in seen:
                    continue
                seen.add(s)
                pool.append(s)
        if len(pool) > pool_cap:
            pool = random.Random(seed).sample(pool, pool_cap)
        self._pool = pool

    def sample(self, rng, n: int) -> list:
        out = []
        for _ in range(n):
            facts, q, a, aliases = self.rows[rng.randrange(len(self.rows))]
            out.append(QAItem(facts=list(facts), question=q, answer=a, task_id=0,
                              meta={"dataset": "triviaqa", "aliases": aliases}))
        return out

    def distractor_pool(self) -> list:
        return self._pool
