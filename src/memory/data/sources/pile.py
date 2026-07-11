"""Pile corpus source — bucket-1 natural text (best-effort HF sample).

Mirrors ``FinewebSource``'s corpus interface (yields whole documents as ``CorpusItem`` token arrays,
backbone-tokenized, ≥ ``min_len``) for the diverse-web Pile bucket. Loads a BOUNDED local sample from
``data/pile/<split>.jsonl`` if the ingest script has staged one; otherwise HF-streams a small sample
(default ``NeelNanda/pile-10k``); otherwise (offline) raises a clear "run ingest first" error — it
never hangs. Windowing / span placement is the Task's job.

Ingest: ``scripts/data_build/ingest/pile/download.py``. See DATASETS.md / docs/history/docs/history/data_arch_plan.md.
"""
from __future__ import annotations

from .base import Source
from ._corpus import _CorpusSampleMixin, load_corpus_docs


class PileSource(_CorpusSampleMixin, Source):
    """Yields whole Pile documents (backbone-tokenized) as ``CorpusItem`` token arrays."""

    kind = "corpus"

    def __init__(self, tokenizer, *, split: str = "train", min_len: int = 1024, seed: int = 0,
                 hf_name: str = "NeelNanda/pile-10k", hf_config: str | None = None,
                 n_docs: int = 2000, **kw):
        self.tok = tokenizer
        self.split = split
        self.min_len = min_len
        self.seed = seed
        self.docs = load_corpus_docs(
            tokenizer, name="pile", hf_name=hf_name, hf_config=hf_config,
            split=split, min_len=min_len, n_docs=n_docs)
