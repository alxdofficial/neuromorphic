"""LongCite source — 128k-word contexts with sentence-level citations, rendered as QAItems.

Source half of the ``qa`` task for citation-grounded long-context QA: each LongCite example is a
document broken into numbered evidence CHUNKS (``<C0>``…``<Cn>``), a question, and a model-generated
answer whose sentences are each tagged with the citing chunk range(s) — the templated format
``<statement>{sentence}<cite>[s1-e1][s2-e2]...</cite></statement>``. We split the chunk-tagged document
into ``facts`` (one per ``<Ci>`` chunk, order preserved), take the trailing free-text question, and
reduce the tagged answer to plain text (``answer``) while keeping the CITED CHUNK INDICES in ``meta``
— the load-bearing evidence: which facts the answer actually depends on (unlike squad/triviaqa's single
gold paragraph, a LongCite answer can cite several non-adjacent chunks scattered across a huge document).

LANGUAGE: the canonical ``zai-org/LongCite-45k`` (mirrored as ``THUDM/LongCite-45k``) is roughly half
Chinese / half English (bilingual SFT distillation data). Our backbone tokenizer is English-centric, so
we keep only ROUGHLY-ASCII rows (``min_ascii_ratio``, default 0.85) — a cheap language-ID heuristic
(CJK text is overwhelmingly non-ASCII once encoded to unicode codepoints), not a perfect filter. Checked
on the parsed QUESTION+ANSWER text (NOT the raw document) — a numeric/table-heavy Chinese report can
score high-ASCII on the full document while its question/answer are unambiguously Chinese, so filtering
post-parse on the clean natural-language fields is the more reliable signal. Pass ``min_ascii_ratio=0``
to disable (e.g. for a bilingual backbone), or override ``hf_name`` to a pre-filtered community mirror
such as ``abideen/LongCite-English-Filtered-13K`` (uses ``INSTRUCTION``/``RESPONSE`` columns instead of
``prompt``/``response`` — both are auto-detected).

Resolution order (BEST-EFFORT, never hangs):
  1. Local ``data/longcite/<split>.jsonl`` (``{"chunks","question","answer","cited"}``) — ingest cache.
  2. Else HF-stream a BOUNDED, ascii-filtered sample of ``zai-org/LongCite-45k`` (single upstream
     "train" split → carved train[0:n_docs] / train[n_docs:2n_docs], scanning past non-English rows).
  3. Else (offline) raise a clear "run ingest first" error — no silent hang.

Ingest: ``scripts/data_build/ingest/longcite/download.py``. See DATASETS.md / docs/data_arch_plan.md (L1).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple

from .base import Source, QAItem
from ._corpus import local_jsonl

HF_NAME = "zai-org/LongCite-45k"
MIN_ASCII_RATIO = 0.85     # documents below this ASCII fraction are treated as non-English, skipped

_DOC_RE = re.compile(r"\[Document Start\]\n(.*)\n\[Document End\]\n\n(.*)", re.S)
_CHUNK_SPLIT = re.compile(r"<C\d+>")
_STMT_RE = re.compile(r"<statement>(.*?)<cite>(.*?)</cite></statement>", re.S)
_CITE_RANGE_RE = re.compile(r"\[(\d+)-(\d+)\]")

Row = Tuple[List[str], str, str, List[int]]   # (chunks, question, answer, cited chunk indices)


def _local_jsonl(split: str) -> Optional[Path]:
    return local_jsonl("longcite", split)


def _ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) < 128) / len(text)


def _split_chunks(doc: str) -> List[str]:
    """The chunk-tagged document (``<C0>chunk0<C1>chunk1...``) → ordered chunk texts."""
    parts = _CHUNK_SPLIT.split(doc)
    return [p.strip() for p in parts[1:] if p.strip()]   # parts[0] = preamble before <C0> (empty)


def _parse_prompt(prompt: str) -> Optional[Tuple[List[str], str]]:
    """``prompt`` → (chunks, question), or None if the template markers aren't present."""
    m = _DOC_RE.search(prompt or "")
    if not m:
        return None
    chunks = _split_chunks(m.group(1))
    question = m.group(2).strip()
    if not chunks or not question:
        return None
    return chunks, question


def _parse_response(resp: str) -> Tuple[str, List[int]]:
    """Cited-answer ``response`` → (plain answer text, sorted cited chunk indices).

    Falls back to a tag-stripped plain string (no citations) if the ``<statement><cite>`` template
    doesn't parse (malformed row) — never crashes on a single bad example.
    """
    statements: List[str] = []
    cited = set()
    for sm in _STMT_RE.finditer(resp or ""):
        text = sm.group(1).strip()
        if text:
            statements.append(text)
        for a, b in _CITE_RANGE_RE.findall(sm.group(2)):
            for i in range(int(a), int(b) + 1):
                cited.add(i)
    if statements:
        return " ".join(statements), sorted(cited)
    plain = re.sub(r"</?statement>|<cite>.*?</cite>|</?cite>", "", resp or "", flags=re.S).strip()
    return plain, sorted(cited)


def _row_from_raw(prompt_text: str, response_text: str) -> Optional[Row]:
    parsed = _parse_prompt(prompt_text)
    if parsed is None:
        return None
    chunks, question = parsed
    answer, cited = _parse_response(response_text)
    if not answer:
        return None
    cited = sorted(i for i in cited if 0 <= i < len(chunks))   # drop any out-of-range cite (malformed)
    return chunks, question, answer, cited


def _prompt_response_fields(ex: dict) -> Tuple[str, str]:
    """LongCite mirrors expose either ``prompt``/``response`` (canonical) or ``INSTRUCTION``/
    ``RESPONSE`` (the ``abideen/LongCite-English-Filtered-13K`` re-export) — normalize either way."""
    if "prompt" in ex:
        return ex.get("prompt") or "", ex.get("response") or ""
    return ex.get("INSTRUCTION") or "", ex.get("RESPONSE") or ""


def _iter_hf_rows(hf_name: str, n_docs: int, skip: int, min_ascii_ratio: float) -> List[Row]:
    """Stream + ascii-filter up to ``n_docs`` valid rows from HF (bounded scan, disjoint val skip)."""
    try:
        from datasets import load_dataset
        ds = load_dataset(hf_name, split="train", streaming=True)
    except Exception as e:  # network / offline / missing dataset
        raise RuntimeError(
            f"[longcite] HF dataset {hf_name!r} unreachable ({type(e).__name__}: {str(e)[:120]}). "
            f"Run  python scripts/data_build/ingest/longcite/download.py  to stage a local "
            f"data/longcite/{{train,val}}.jsonl sample, then retry (works offline once staged)."
        ) from e
    rows: List[Row] = []
    scanned, kept_before_skip = 0, 0
    scan_cap = max(50 * n_docs, 5000) + skip   # bounded: don't scan forever past a thin val skip
    for ex in ds:
        scanned += 1
        p, r = _prompt_response_fields(ex)
        row = _row_from_raw(p, r)
        if row is None:
            continue
        _chunks, question, answer, _cited = row
        # Language filter on QUESTION+ANSWER (clean natural-language text), not the raw document --
        # a numeric/table-heavy Chinese document can score high-ASCII on the full prompt while its
        # question/answer are unambiguously Chinese, so filtering post-parse on those is more reliable.
        if min_ascii_ratio > 0 and _ascii_ratio(question + " " + answer) < min_ascii_ratio:
            continue
        if kept_before_skip < skip:            # disjoint val slice: skip the first `skip` valid rows
            kept_before_skip += 1
            continue
        rows.append(row)
        if len(rows) >= n_docs or scanned >= scan_cap:
            break
    print(f"[data.longcite] kept {len(rows)}/{scanned} scanned (ascii>={min_ascii_ratio} + template "
          f"filter) from HF:{hf_name}", flush=True)
    return rows


def _iter_local_rows(path: Path, n_docs: int) -> List[Row]:
    rows: List[Row] = []
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            chunks = [c for c in (o.get("chunks") or []) if c]
            question = (o.get("question") or "").strip()
            answer = (o.get("answer") or "").strip()
            cited = [i for i in (o.get("cited") or []) if isinstance(i, int)]
            if chunks and question and answer:
                rows.append((chunks, question, answer, cited))
            if len(rows) >= n_docs:
                break
    print(f"[data.longcite] loaded {len(rows)} rows from {path}", flush=True)
    return rows


class LongCiteSource(Source):
    """Yields LongCite citation-grounded long documents as ``QAItem``s (facts = the chunk list)."""

    kind = "qa"

    def __init__(self, tokenizer, *, split: str = "train", seed: int = 0, n_docs: int = 2000,
                 hf_name: str = HF_NAME, min_ascii_ratio: float = MIN_ASCII_RATIO,
                 pool_cap: int = 2000, **kw):
        self.tok = tokenizer
        self.split = split
        self.seed = seed
        local = _local_jsonl(split)
        if local is not None:
            self.rows = _iter_local_rows(local, n_docs)
        else:
            skip = 0 if split == "train" else n_docs   # single upstream "train" split → skip-carve val
            self.rows = _iter_hf_rows(hf_name, n_docs, skip, min_ascii_ratio)
        if not self.rows:
            raise ValueError(f"[longcite] no rows loaded for split={split!r}.")
        # Distractor pool = OTHER rows' chunks (cross-document padding noise for short episodes).
        pool: List[str] = []
        seen = set()
        for chunks, _q, _a, _c in self.rows:
            for s in chunks:
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
            chunks, q, a, cited = self.rows[rng.randrange(len(self.rows))]
            out.append(QAItem(facts=list(chunks), question=q, answer=a, task_id=0,
                              meta={"dataset": "longcite", "cited_chunks": cited}))
        return out

    def distractor_pool(self) -> list:
        return self._pool
