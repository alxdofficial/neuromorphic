"""GovReport corpus source — long US government reports (~9k backbone tokens) as CorpusItem docs.

Source half of the continuation/mae/reconstruction Tasks for LONG natural-text documents that are NOT
gist-gameable (a real GAO/CRS report, not a synthetic story). ``ccdv/govreport-summarization`` is a
document+summary SUMMARIZATION dataset; here we use only the ``report`` (document) half and frame it as
a plain ``kind="corpus"`` source — simpler, and it fits the existing continuation/mae tasks directly
(a ``kind="qa"`` "Summarize the report." framing was the alternative; picked corpus per the build brief).

GOTCHA: ``ccdv/govreport-summarization`` rows are keyed ``"report"``/``"summary"``, NOT the
``"text"``/``"content"`` keys ``_corpus.py``'s generic HF-stream loader (``_text_of``) recognizes — so
this Source can't just delegate to ``load_corpus_docs`` wholesale for the live-HF path (it would
silently see empty text and raise "no doc >= min_len"). The ingest script normalizes to
``{"text": report, "summary": summary}`` so the LOCAL-cache path reuses ``_corpus.py``'s helpers
unmodified; only the live-HF fallback here reads ``report`` directly.

Resolution order (BEST-EFFORT, never hangs):
  1. Local ``data/govreport/<split>.jsonl`` (``{"text","summary"}`` per line) — ingest cache.
  2. Else HF-stream a BOUNDED sample of ``ccdv/govreport-summarization`` (real train / validation
     splits upstream — no skip-carving needed).
  3. Else (offline) raise a clear "run ingest first" error — no silent hang.

Ingest: ``scripts/data_build/ingest/govreport/download.py``. See DATASETS.md / docs/DATA.md (L1).
"""
from __future__ import annotations

from typing import Iterator, List

import numpy as np

from .base import Source, CorpusItem
from ._corpus import _CorpusSampleMixin, local_jsonl, _iter_local_texts, _tokenize_cached

HF_NAME = "ccdv/govreport-summarization"


def _hf_split(split: str) -> str:
    if split == "train":
        return "train"
    if split in ("validation", "val"):
        return "validation"
    raise ValueError(f"govreport: unrecognized split {split!r} (expected 'train'/'validation'/'val')")


def _iter_hf_reports(hf_split: str, n_docs: int) -> Iterator[str]:
    """Stream up to ``n_docs`` report texts — reads the ``report`` field directly since
    ``_corpus.py``'s generic ``_text_of`` doesn't recognize GovReport's raw column names."""
    try:
        from datasets import load_dataset
        ds = load_dataset(HF_NAME, split=hf_split, streaming=True)
    except Exception as e:  # network / offline / missing dataset
        raise RuntimeError(
            f"[govreport] HF dataset {HF_NAME!r} unreachable ({type(e).__name__}: {str(e)[:120]}). "
            f"Run  python scripts/data_build/ingest/govreport/download.py  to stage a local "
            f"data/govreport/{{train,val}}.jsonl sample, then retry (works offline once staged)."
        ) from e
    count = 0
    for ex in ds:
        t = (ex.get("report") or "").strip()
        if not t:
            continue
        yield t
        count += 1
        if count >= n_docs:
            break


def _load_govreport_docs(tokenizer, *, split: str, min_len: int, n_docs: int) -> List[np.ndarray]:
    """Load → re-tokenize (backbone) → keep docs >= min_len. Local jsonl first, else HF stream."""
    from transformers.utils import logging as _hf_logging
    _prev = _hf_logging.get_verbosity()
    _hf_logging.set_verbosity_error()
    try:
        local = local_jsonl("govreport", split)
        if local is not None:
            docs_all = _tokenize_cached(local, tokenizer, lambda: _iter_local_texts(local))  # disk-cached
            origin = f"data/govreport/{split}.jsonl"
        else:
            texts = _iter_hf_reports(_hf_split(split), n_docs)
            origin = f"HF:{HF_NAME}"
            docs_all = [np.asarray(tokenizer(t, add_special_tokens=False).input_ids, dtype=np.int64)
                        for t in texts]
    finally:
        _hf_logging.set_verbosity(_prev)

    n_total = len(docs_all)
    docs = [a for a in docs_all if a.shape[0] >= min_len]
    if not docs:
        raise ValueError(
            f"[govreport] no doc >= {min_len} tok from {origin} ({n_total} scanned). "
            f"Lower the task's total_len, or stage more docs via the ingest script.")
    print(f"[data.govreport] {split}: {len(docs)}/{n_total} docs >= {min_len} tok from {origin}",
          flush=True)
    return docs


class GovReportSource(_CorpusSampleMixin, Source):
    """Yields whole GovReport documents (backbone-tokenized) as ``CorpusItem`` token arrays."""

    kind = "corpus"

    def __init__(self, tokenizer, *, split: str = "train", min_len: int = 1024, seed: int = 0,
                 n_docs: int = 2000, **kw):
        self.tok = tokenizer
        self.split = split
        self.min_len = min_len
        self.seed = seed
        self.docs = _load_govreport_docs(tokenizer, split=split, min_len=min_len, n_docs=n_docs)
