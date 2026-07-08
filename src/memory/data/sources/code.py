"""Code corpus source — source code as natural text with UN-GUESSABLE exact-recall binding.

Code is the "hard middle": a natural distribution whose long-range dependencies are exact-token
(a `def foo(x, y):` / `const KEY = …` bound early and referenced hundreds of tokens later can only
be recalled verbatim, not paraphrased) — the key→value binding stress test bio/fineweb under-supply,
and white space no prior compressor trained on (only CoMem, on agent trajectories). Whole-file
corpus source (like fineweb): load files → backbone-tokenize → continuation / masked reconstruction.

Default `codeparrot/codeparrot-clean` (self-contained parquet, no gating/S3; `bigcode/the-stack-smol`
is gated — see the `hf_name` note in `__init__` below). Local sample:
`scripts/data_build/ingest/code/download.py`. See DATASETS.md.
"""
from __future__ import annotations

from .base import Source
from ._corpus import _CorpusSampleMixin, load_corpus_docs


class CodeSource(_CorpusSampleMixin, Source):
    """Yields whole source-code files (backbone-tokenized) as ``CorpusItem`` token arrays."""

    kind = "corpus"

    def __init__(self, tokenizer, *, split: str = "train", min_len: int = 1024, seed: int = 0,
                 hf_name: str = "codeparrot/codeparrot-clean", hf_data_dir: str | None = None,
                 n_docs: int = 2000, **kw):
        # default codeparrot/codeparrot-clean (python, parquet, OPEN) — bigcode/the-stack* are gated
        # (need an HF agreement); pass hf_name="bigcode/the-stack-smol", hf_data_dir="data/python"
        # after accepting the ToS if you want multi-language.
        self.tok = tokenizer
        self.split = split
        self.min_len = min_len
        self.seed = seed
        self.docs = load_corpus_docs(
            tokenizer, name="code", hf_name=hf_name, hf_config=None,
            split=split, min_len=min_len, n_docs=n_docs, hf_data_dir=hf_data_dir)
