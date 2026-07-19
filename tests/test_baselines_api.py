"""CPU-only, no-network tests for the Phase-2 API baseline building blocks (prompts, retrieval, budgeting).

The live API path (OpenRouterClient) is not exercised here — it needs a key and is validated via --dry-run.
"""
import pytest

from src.memory.eval.baselines import build_messages, char_budget_for, MODES
from src.memory.eval.retrieval import bm25_topk
from src.memory.eval.api_client import cost_usd, PRICING, DEFAULT_MODELS


SESSIONS = [
    "[Session 1 — 2024-01-01]\nUser: I adopted a beagle named Cooper.",
    "[Session 2 — 2024-02-01]\nUser: I started a new job at a bakery.",
    "[Session 3 — 2024-03-01]\nUser: Cooper learned to sit and roll over.",
    "[Session 4 — 2024-04-01]\nUser: I moved to Portland.",
]
HISTORY = "\n\n".join(SESSIONS)


def test_floor_has_no_history():
    msgs, info = build_messages("floor", question="What is my dog's name?")
    assert not info["truncated"] and info["retrieved_idx"] is None
    assert all("Cooper" not in m["content"] for m in msgs)      # floor sees no history
    assert "What is my dog" in msgs[-1]["content"]


def test_full_context_includes_history():
    msgs, info = build_messages("full_context", question="Where did I move?", full_history=HISTORY)
    assert "Portland" in msgs[-1]["content"] and not info["truncated"]


def test_full_context_truncates_to_budget_keeping_recent():
    long_hist = "OLDEST\n" + ("x" * 500) + "\nNEWEST_MARKER"
    msgs, info = build_messages("full_context", question="q", full_history=long_hist, char_budget=50)
    assert info["truncated"]
    assert "NEWEST_MARKER" in msgs[-1]["content"]               # tail kept
    assert "OLDEST" not in msgs[-1]["content"]                  # head dropped


def test_bm25_retrieves_relevant_session_in_order():
    idx = bm25_topk(SESSIONS, "what tricks did Cooper learn", k=2)
    assert idx == sorted(idx)                                   # chronological order preserved
    assert 2 in idx                                             # the "roll over" session must be retrieved


def test_rag_bm25_prompt_contains_only_topk():
    msgs, info = build_messages("rag_bm25", question="what tricks did Cooper learn",
                                sessions=SESSIONS, bm25_topk=1)
    body = msgs[-1]["content"]
    assert "roll over" in body                                 # the relevant session made it in
    assert body.count("[Session") == 1                         # exactly top-1
    assert info["retrieved_idx"] == [2]                        # retrieval indices persisted for analysis


def test_rag_requires_sessions():
    with pytest.raises(ValueError):
        build_messages("rag_bm25", question="q")               # no sessions -> error


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        build_messages("telepathy", question="q")


def test_char_budget_monotonic_and_positive():
    assert char_budget_for(131072) > char_budget_for(32768) > 0
    assert char_budget_for(1000) == 0 or char_budget_for(6000) == 0   # reserve can zero out tiny windows


def test_cost_uses_price_table():
    m = "meta-llama/llama-3.1-8b-instruct"
    pin, pout = PRICING[m]
    assert cost_usd(m, 1_000_000, 0) == pytest.approx(pin * 1_000_000)
    assert cost_usd("unknown/model", 10**9, 10**9) == 0.0       # unknown -> 0, never crashes


def test_default_panel_all_priced():
    assert MODES == ("floor", "full_context", "rag_bm25", "rag_dense")
    for m in DEFAULT_MODELS:
        assert m in PRICING, f"{m} missing from PRICING"
