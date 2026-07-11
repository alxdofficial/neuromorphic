"""PG19 corpus source — full-length Project Gutenberg BOOKS (very long documents, genuine long-range
horizon) as ``CorpusItem`` token arrays.

Mirrors ``PileSource``/``RedpajamaSource``'s corpus interface for the cleanest "long horizon" natural-
text source we have — a whole PRE-1919 book (many streaming windows per document, unlike a short web
doc). ``deepmind/pg19`` itself is a loading-script dataset (unsupported by ``datasets`` >= 4.x — its
loader fetches raw text from a GCS bucket at runtime, so HF can't auto-convert it to parquet either); we
default to the parquet-native ``emozilla/pg19`` mirror, which exposes the IDENTICAL schema
(``short_book_title``/``publication_date``/``url``/``text``) and streams cleanly.

``min_len`` should be set HIGH (well above the 1024 default used by the other corpus sources) — PG19
books run tens to hundreds of thousands of tokens, so a low ``min_len`` wastes the point of this source
(many streaming windows per book); pass e.g. ``min_len=8192`` or higher for a genuine long-horizon regime.

Reuses ``_corpus.py``'s shared ``load_corpus_docs`` loader (local-jsonl-first, else a bounded HF sample,
else a clear ingest-first error) exactly like Pile/RedPajama — including its skip-carved train/val
convention (``train[0:n_docs]`` / ``train[n_docs:2n_docs]``, both disjoint book sets). NOTE:
``emozilla/pg19`` DOES expose genuine upstream train/validation/test splits, but the shared loader (by
design, shared with Pile/RedPajama's single-split HF mirrors) always reads "train" live and carves val
by skipping past it — so pass ``split="train"``/``"val"`` per that convention, not ``"validation"``; a
local ``data/pg19/val.jsonl`` staged from the real upstream validation split (see the ingest script)
takes priority over the skip-carve once ingested.

Ingest: ``scripts/data_build/ingest/pg19/download.py`` (bounded to ~500 books by default — full books
are huge, so even a few hundred is already a sizeable local cache). See DATASETS.md / docs/history/docs/history/data_arch_plan.md.
"""
from __future__ import annotations

from .base import Source
from ._corpus import _CorpusSampleMixin, load_corpus_docs


class PG19Source(_CorpusSampleMixin, Source):
    """Yields whole PG19 books (backbone-tokenized) as ``CorpusItem`` token arrays."""

    kind = "corpus"

    def __init__(self, tokenizer, *, split: str = "train", min_len: int = 8192, seed: int = 0,
                 hf_name: str = "emozilla/pg19", hf_config: str | None = None,
                 n_docs: int = 500, **kw):
        self.tok = tokenizer
        self.split = split
        self.min_len = min_len
        self.seed = seed
        self.docs = load_corpus_docs(
            tokenizer, name="pg19", hf_name=hf_name, hf_config=hf_config,
            split=split, min_len=min_len, n_docs=n_docs)
