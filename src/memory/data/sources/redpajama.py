"""RedPajama corpus source — bucket-1 natural text (best-effort HF sample).

Mirrors ``FinewebSource``'s corpus interface (yields whole documents as ``CorpusItem`` token arrays,
backbone-tokenized, ≥ ``min_len``) for the RedPajama mixture. Loads a BOUNDED local sample from
``data/redpajama/<split>.jsonl`` if the ingest script has staged one; otherwise HF-streams a small
sample; otherwise (offline) raises a clear "run ingest first" error — it never hangs. Windowing /
span placement is the Task's job.

Default HF sample = ``DKYoon/SlimPajama-6B`` (SlimPajama is the deduplicated RedPajama mixture,
parquet-native). The classic ``togethercomputer/RedPajama-Data-1T-Sample`` is a *loading-script*
dataset, which ``datasets`` ≥ 4.x no longer supports — pass ``hf_name=`` to override if you have a
parquet-native RedPajama mirror.

Ingest: ``scripts/data_build/ingest/redpajama/download.py``. See DATASETS.md / docs/DATA.md.
"""
from __future__ import annotations

from .base import Source
from ._corpus import _CorpusSampleMixin, load_corpus_docs


class RedpajamaSource(_CorpusSampleMixin, Source):
    """Yields whole RedPajama documents (backbone-tokenized) as ``CorpusItem`` token arrays."""

    kind = "corpus"

    def __init__(self, tokenizer, *, split: str = "train", min_len: int = 1024, seed: int = 0,
                 hf_name: str = "DKYoon/SlimPajama-6B",
                 hf_config: str | None = None, n_docs: int = 2000, **kw):
        self.tok = tokenizer
        self.split = split
        self.min_len = min_len
        self.seed = seed
        self.docs = load_corpus_docs(
            tokenizer, name="redpajama", hf_name=hf_name, hf_config=hf_config,
            split=split, min_len=min_len, n_docs=n_docs)
