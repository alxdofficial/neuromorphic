"""Multi-corpus source — continuation/mae VARIETY over several natural-text corpora at once.

Unions the doc pools of several corpus sources (fineweb + pile + redpajama + code) into ONE
corpus source, so a single `continuation`/`mae` task sees varied text instead of fineweb-only.
ROBUST: a corpus that can't load (HF unreachable, not yet ingested) is skipped with a warning
rather than failing the whole source — you get variety from whatever is available (fineweb is
always local). See DATASETS.md / docs/history/docs/history/data_arch_plan.md.
"""
from __future__ import annotations

from .base import Source, CorpusItem
from ._corpus import _CorpusSampleMixin

# default variety pool. fineweb = always-local anchor; the rest add distribution variety
# (pile/redpajama = diverse web/books; code = un-guessable exact-recall binding).
DEFAULT_CORPORA = ("fineweb", "pile", "redpajama", "code")


def _build_one(name, tokenizer, *, split, min_len, seed, src_tokenizer_name):
    """Build a single corpus source by name (direct class import — no registry cycle)."""
    if name == "fineweb":
        from .fineweb import FinewebSource
        return FinewebSource(tokenizer, split=split, src_tokenizer_name=src_tokenizer_name,
                             min_len=min_len, seed=seed)
    if name == "pile":
        from .pile import PileSource
        return PileSource(tokenizer, split=split, min_len=min_len, seed=seed)
    if name == "redpajama":
        from .redpajama import RedpajamaSource
        return RedpajamaSource(tokenizer, split=split, min_len=min_len, seed=seed)
    if name == "code":
        from .code import CodeSource
        return CodeSource(tokenizer, split=split, min_len=min_len, seed=seed)
    raise ValueError(f"multicorpus: unknown corpus {name!r} (have {DEFAULT_CORPORA})")


class MultiCorpusSource(_CorpusSampleMixin, Source):
    """Yields docs drawn CORPUS-uniformly (each loaded corpus equal weight), then doc-uniformly within
    the chosen corpus — so a large corpus (fineweb ~14k docs) doesn't drown the smaller variety corpora
    (pile/redpajama/code) by raw doc count. (Concatenate-then-doc-uniform would make continuation ~92%
    fineweb.) Small variety pools repeat, which is fine for training-time variety."""

    kind = "corpus"

    def __init__(self, tokenizer, *, split: str = "train", min_len: int = 1024, seed: int = 0,
                 corpora=DEFAULT_CORPORA, src_tokenizer_name: str = "meta-llama/Llama-3.2-1B", **kw):
        self.tok = tokenizer
        self.pools = []          # [(name, docs)] — sampled corpus-uniform then doc-uniform
        self.docs = []           # concatenated (kept for any consumer that reads .docs directly)
        loaded, skipped = [], []
        for i, name in enumerate(corpora):
            try:
                src = _build_one(name, tokenizer, split=split, min_len=min_len,
                                 seed=seed + i, src_tokenizer_name=src_tokenizer_name)
                self.pools.append((name, src.docs))
                self.docs.extend(src.docs)
                loaded.append(f"{name}:{len(src.docs)}")
            except Exception as e:                       # unreachable / not-ingested → skip, keep variety
                skipped.append(f"{name} ({type(e).__name__})")
        if not self.pools:
            raise ValueError(
                f"[multicorpus] no corpus loaded from {list(corpora)} — at least fineweb should be "
                f"local. Skipped: {skipped}")
        print(f"[data.multicorpus] {split}: {len(self.docs)} docs from [{', '.join(loaded)}] "
              f"(corpus-uniform: each ≈{100//len(self.pools)}%)"
              + (f"  (skipped: {', '.join(skipped)} — run their ingest for variety)" if skipped else ""),
              flush=True)

    def sample(self, rng, n: int) -> list:
        out = []
        for _ in range(n):
            _name, docs = self.pools[rng.randrange(len(self.pools))]   # corpus-uniform
            out.append(CorpusItem(tokens=docs[rng.randrange(len(docs))]))
        return out
