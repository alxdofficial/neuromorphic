"""QuALITY source — long-document multiple-choice reading comprehension as QAItems.

Source half for the ``qa`` task: loads QuALITY stories (HF ``emozilla/quality``) — a long ``article``
(~4.7k-7.6k backbone tokens), a ``question``, ~4 ``options`` and a gold ``answer`` index — and yields
one ``QAItem`` per (story, question). The whole STORY is the context (split into paragraph "facts");
the question carries the lettered option menu; the answer is the correct option's *text* (EM-scorable
and un-guessable as free-form generation, unlike a bare A/B/C/D letter).

LONG-CONTEXT regime. QuALITY articles all EXCEED 4096 tokens, so this source is meant for
``total_len >= 4096`` (ideally >= 8192). The qa Task packs facts at the front and TAIL-truncates the
overflow, so at ``total_len=1024`` most of the story is dropped — that is expected; run QuALITY with a
large ``total_len``. The article tail (which contains the ending) is what survives truncation.

Answer-in-context caveat: QuALITY answers are ABSTRACTIVE option sentences (median ~11 words), so the
gold option text is a verbatim substring of the article only ~6% of the time — a substring
"answer-in-context" check is near-zero *by construction* for this dataset (it measures extractive
recall, which QuALITY is not). The meaningful invariant here is that the answer-*supporting story* is
present in the context (true at large total_len); the option menu is always present in the question.

``answer`` indexing: for ``emozilla/quality`` the ``answer`` field is 0-indexed (verified: values seen
are {0,1,2,3} over 4 options), so the gold option is ``options[answer]`` directly.

Bounded local-cache-first: ``data/quality/<split>.jsonl`` (written by the ingest script) is used if
present, else a bounded HF stream (``n_docs`` rows). Ingest:
``scripts/data_build/ingest/quality/download.py``. See DATASETS.md / docs/data_arch_plan.md (Layer L1).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

from .base import Source, QAItem
from ._corpus import local_jsonl

_PARA_SPLIT = re.compile(r"\n{2,}")


def _local_jsonl(split: str) -> Optional[Path]:
    return local_jsonl("quality", split)          # ingest writes train.jsonl / val.jsonl


def _hf_split(split: str) -> str:
    if split == "train":
        return "train"
    if split in ("validation", "val"):
        return "validation"
    raise ValueError(f"quality: unrecognized split {split!r} (expected 'train'/'validation'/'val')")


def _iter_rows(split: str, n_docs: int, hf_name: str):
    """Yield up to ``n_docs`` raw rows dict(article, question, options, answer). Local jsonl first
    (fully offline), else a bounded HF stream. Raises a clear ingest-first error if HF is unreachable."""
    local = _local_jsonl(split)
    if local is not None:
        with open(local) as fp:
            count = 0
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
                count += 1
                if count >= n_docs:
                    break
        return

    try:
        from datasets import load_dataset
        ds = load_dataset(hf_name, split=_hf_split(split), streaming=True)
    except Exception as e:  # network / offline / gated
        raise RuntimeError(
            f"[quality] HF dataset {hf_name!r} unreachable ({type(e).__name__}: {str(e)[:120]}). "
            f"Run  python scripts/data_build/ingest/quality/download.py  to stage a local "
            f"data/quality/{{train,val}}.jsonl sample, then retry (works fully offline once staged).") from e
    count = 0
    for ex in ds:
        yield {"article": ex["article"], "question": ex["question"],
               "options": list(ex["options"]), "answer": int(ex["answer"])}
        count += 1
        if count >= n_docs:
            break


def _split_paragraphs(article: str) -> List[str]:
    """Split a story into paragraph "facts" (order preserved). Falls back to the whole article as a
    single fact if it has no blank-line paragraph breaks — the qa Task tail-truncates either way."""
    paras = [p.strip() for p in _PARA_SPLIT.split(article) if p.strip()]
    return paras or [article.strip()]


def _format_question(question: str, options: List[str]) -> str:
    lines = [question.strip(), "", "Options:"]
    for i, opt in enumerate(options):
        lines.append(f"({chr(ord('A') + i)}) {opt.strip()}")
    return "\n".join(lines)


class QualitySource(Source):
    """Yields QuALITY long-document MC-RC stories as ``QAItem``s (whole story = facts)."""

    kind = "qa"

    def __init__(self, tokenizer, *, split: str = "train", seed: int = 0, n_docs: int = 1500,
                 hf_name: str = "emozilla/quality", **kw):
        self.tok = tokenizer
        self.seed = seed
        origin = ("data/quality/%s.jsonl" % ("train" if split == "train" else "val")
                  if _local_jsonl(split) is not None else f"HF:{hf_name}")
        self.rows: List[QAItem] = []
        skipped = 0
        for r in _iter_rows(split, n_docs, hf_name):
            opts = [str(o) for o in r["options"]]
            ans = int(r["answer"])
            if not (0 <= ans < len(opts)):        # guard a mis-indexed cache (emozilla is 0-indexed)
                skipped += 1
                continue
            facts = _split_paragraphs(str(r["article"]))
            self.rows.append(QAItem(
                facts=facts,
                question=_format_question(str(r["question"]), opts),
                answer=opts[ans].strip(),
                task_id=0,
                meta={"dataset": "quality", "answer_idx": ans, "n_options": len(opts)}))
        if not self.rows:
            raise ValueError(f"[quality] no rows loaded for split={split!r} from {origin}")
        # Bounded distractor pool of OTHER stories' paragraphs. At the intended total_len (>=4096) a
        # single QuALITY story overflows the budget, so the qa Task tail-truncates and never reaches
        # the distractor loop — the pool is unused. It exists only so a run at total_len LARGER than
        # the longest article (~7.6k tok) still has padding material (an empty pool would crash the
        # Task's rng.choice). Capped so construction stays light.
        self._pool: List[str] = []
        for it in self.rows:
            self._pool.extend(it.facts)
            if len(self._pool) >= 20000:
                break
        print(f"[quality] {split}: {len(self.rows)} stories from {origin} "
              f"(skipped {skipped} mis-indexed)", flush=True)

    def sample(self, rng, n: int) -> list:
        # QAItem.facts are large lists — reuse the stored items (the qa Task only reads them).
        # Draw uniformly with replacement.
        return [self.rows[rng.randrange(len(self.rows))] for _ in range(n)]

    def distractor_pool(self) -> list:
        return self._pool
