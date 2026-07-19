"""Regression tests for the adversarial-review fixes (no network)."""

from src.memory.eval.longmemeval_score import _containment, LongMemEvalScorer
from src.memory.eval.retrieval import bm25_topk
from src.memory.eval.results import store_path


# --- containment boundary (was: '1' matched inside '2001') ---
def test_containment_rejects_short_gold_inside_larger_token():
    assert not _containment("1", "It happened in 2001")
    assert not _containment("yes", "yesterday I went out")
    assert not _containment("10", "the temperature was 105 degrees")


def test_containment_still_accepts_whole_token_and_phrase():
    assert _containment("Johnson", "The answer is Johnson.")
    assert _containment("25 minutes", "it took 25 minutes total")
    assert _containment("1", "the count was 1 that day")     # boundary-delimited whole token → matches


# --- abstention false-positive guard (long substantive answer that merely hedges is NOT an abstention) ---
def test_abstention_true_for_concise_refusal():
    s = LongMemEvalScorer(use_bem=False)
    r = s.score_item("Where did I move?", "insufficient information",
                     "I don't have enough information to answer that.", "abstention", "q_abs")
    assert r["correct"] is True


def test_abstention_false_for_long_answer_with_stray_hedge():
    s = LongMemEvalScorer(use_bem=False)
    long_answer = ("Based on the history you moved to Portland in April, then to Seattle, then back, and "
                   "later to Denver where you took a new job at a bakery, though I'm not sure about the exact dates "
                   "and there were several other cities mentioned across the many sessions of the conversation.")
    r = s.score_item("Where did I move?", "insufficient information", long_answer, "abstention", "q_abs")
    assert r["correct"] is False                              # committed to an answer → not a valid abstention


# --- bm25 no longer crashes on degenerate input ---
def test_bm25_all_empty_passages_no_crash():
    assert bm25_topk(["", "  ", "\n"], "some query", k=2) == [0, 1]   # falls back to first-k, no ZeroDivision


def test_bm25_empty_query_returns_first_k():
    assert bm25_topk(["a b c", "d e f", "g h i"], "", k=2) == [0, 1]


# --- config-aware cache key: changing a generation knob → different store file ---
def test_store_path_config_sig_changes_filename(tmp_path):
    a = store_path(tmp_path, "longmemeval", "meta/llama", "rag_bm25", "aaaa1111")
    b = store_path(tmp_path, "longmemeval", "meta/llama", "rag_bm25", "bbbb2222")
    assert a != b
    assert a.name.endswith("__aaaa1111.jsonl")
