"""Retrievers for the RAG reference baseline: sparse (BM25) and dense (sentence-transformers, CPU).

Both take a list of candidate passages (for LongMemEval: one per haystack session) + a query, and return
the indices of the top-k, kept in original (chronological) order. Dense retrieval runs on CPU by default so
it never contends with local Phase-0 GPU work.
"""
from __future__ import annotations

import functools


# Per-context BM25 index cache. MemoryAgentBench has only 36 distinct contexts but 3,071 questions, and the
# loader reuses ONE `sessions` list object per context across its questions — so keying by id(passages) (with
# a length guard) memoizes the expensive tokenize+index build and reuses it across the ~85 questions sharing
# a context, instead of rebuilding it every call (the same reuse win the dense retriever already has).
_BM25_CACHE: dict = {}


def bm25_topk(passages: list[str], query: str, k: int) -> list[int]:
    """Top-k passage indices by BM25 (Okapi), returned in original (chronological) order."""
    if len(passages) <= k:
        return list(range(len(passages)))
    key = id(passages)
    entry = _BM25_CACHE.get(key)
    if entry is None or entry[1] != len(passages):
        from rank_bm25 import BM25Okapi
        tokenized = [p.lower().split() for p in passages]
        entry = ((BM25Okapi(tokenized) if any(tokenized) else None), len(passages))
        _BM25_CACHE[key] = entry
    bm25 = entry[0]
    q = query.lower().split()
    if bm25 is None or not q:
        # all-empty passages ⇒ BM25 avgdl=0 (ZeroDivisionError); empty query ⇒ all scores 0.
        # Either way there is no retrieval signal → keep the first k (chronological) rather than crash.
        return list(range(k))
    scores = bm25.get_scores(q)
    top = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)[:k]
    return sorted(top)


class DenseRetriever:
    """Cosine-similarity retrieval with a small sentence-transformer on CPU (cached MiniLM by default).

    The model is loaded once (lazily) and reused across queries. CPU-pinned to avoid GPU contention.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._emb_cache: dict = {}          # passages-fingerprint -> passage embeddings

    @functools.cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self.model_name, device=self.device)

    def _encode_passages(self, passages: list[str]):
        """Encode a passage list ONCE per unique context. MemoryAgentBench reuses ~36 contexts across ~3071
        questions ("inject once, query many"); without this cache dense retrieval re-embeds every context per
        question (~1.4M chunk encodings instead of ~16k) and blocks the async loop. Keyed by a content hash."""
        key = hash(tuple(passages))
        emb = self._emb_cache.get(key)
        if emb is None:
            emb = self._model.encode(passages, normalize_embeddings=True, show_progress_bar=False)
            self._emb_cache[key] = emb
        return emb

    def topk(self, passages: list[str], query: str, k: int) -> list[int]:
        if len(passages) <= k:
            return list(range(len(passages)))
        emb = self._encode_passages(passages)   # cached across questions sharing this context
        q = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = emb @ q
        top = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)[:k]
        return sorted(top)


def retrieve(passages: list[str], query: str, k: int, method: str,
             dense: DenseRetriever | None = None) -> list[int]:
    """Dispatch to bm25 or dense. `dense` is a reusable DenseRetriever (create once, pass in)."""
    if method == "bm25":
        return bm25_topk(passages, query, k)
    if method == "dense":
        return (dense or DenseRetriever()).topk(passages, query, k)
    raise ValueError(f"unknown retrieval method {method!r}")
