"""Regression tests for the FOURTH review's fixes, no network.

Dense-embedding cache (redundant re-encoding), Tier-2 length-cutoff exclusion, MAB source-filter under-fill.
"""
import importlib.util
from pathlib import Path

import numpy as np

from src.memory.eval.retrieval import DenseRetriever

REPO = Path(__file__).resolve().parents[1]


# ---------- dense retriever caches passage embeddings per context ----------
class _FakeModel:
    def __init__(self):
        self.encoded = []

    def encode(self, texts, **kw):
        self.encoded.append(list(texts))
        return np.ones((len(texts), 4), dtype=float)


def test_dense_retriever_caches_passages():
    r = DenseRetriever.__new__(DenseRetriever)     # bypass real-model load
    r._emb_cache = {}
    fake = _FakeModel()
    r.__dict__["_model"] = fake                    # satisfy the cached_property
    passages = ["ctx chunk a", "ctx chunk b", "ctx chunk c"]
    r.topk(passages, "first query", 1)
    r.topk(passages, "second query", 1)            # SAME context → passages must not re-encode
    passage_encodes = [e for e in fake.encoded if e == passages]
    assert len(passage_encodes) == 1               # encoded once, cached for the 2nd question
    query_encodes = [e for e in fake.encoded if e == ["first query"] or e == ["second query"]]
    assert len(query_encodes) == 2                 # queries still encoded per-item (cheap)


# ---------- Tier-2 length-cutoffs excluded from scoring ----------
def _load(mod_name, rel):
    spec = importlib.util.spec_from_file_location(mod_name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tier2_valid_excludes_length_error_and_content_filter():
    # the Tier-2 runners now share src/memory/eval/tier2_common.valid_for_scoring (audit #2 tightened it to
    # also drop terminal error/content_filter finish reasons).
    from src.memory.eval.tier2_common import valid_for_scoring
    assert valid_for_scoring({"finish_reason": "stop"}) is True
    assert valid_for_scoring({"finish_reason": "length"}) is False           # truncated ⇒ not scored
    assert valid_for_scoring({"error": "boom", "finish_reason": "error"}) is False
    assert valid_for_scoring({"finish_reason": "content_filter"}) is False   # provider refusal ⇒ not scored


def test_tier2_make_record_carries_finish_reason():
    # make_record carries finish_reason through so the scoring filter can drop length-cutoffs
    from src.memory.eval.tier2_common import make_record
    rec = make_record({"question_id": "1", "question": "q", "answer": "a", "question_type": "t"},
                      hyp="partial", finish_reason="length")
    assert rec["finish_reason"] == "length" and rec["question_id"] == "1"
