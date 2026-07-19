"""Offline integration test of the async runner `run_one` (no network, stub client).

Exercises the parts NOT reachable by the pure-function unit tests: coverage/error/cutoff accounting (#9),
partial-cutoff exclusion + retryability (#10), selection-scoped scoring so a bigger cached run doesn't
leak in (#11), and competency threading for MemoryAgentBench (#12).
"""
import asyncio
import importlib.util
from pathlib import Path

from src.memory.eval.api_client import CallResult
from src.memory.eval import score_longmemeval, score_memoryagentbench
from src.memory.eval.results import ResultStore

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_api_eval", REPO / "scripts/baselines/run_api_eval.py")
run_api_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_api_eval)


class StubClient:
    """Returns a canned CallResult per question (matched by substring of the user message). Counts calls."""
    def __init__(self, by_q: dict):
        self.by_q = by_q
        self.calls = 0

    async def chat(self, model, msgs, max_tokens=0):
        self.calls += 1
        content = msgs[-1]["content"]
        for needle, result in self.by_q.items():
            if needle in content:
                return result
        return CallResult("", error="no stub match")


def _lme_items():
    mk = lambda qid, q, a, t: {"question_id": qid, "question": q, "answer": a, "question_type": t,
                               "question_date": "2024-01-01", "full_history": f"User: {a}",
                               "sessions": [f"User: {a}"]}
    return [mk("1", "What is my dog's name?", "Cooper", "single-session-user"),
            mk("2", "Where did I move?", "Portland", "multi-session"),
            mk("3", "What car did I buy?", "Tesla", "temporal"),
            mk("4", "What is my job?", "baker", "knowledge-update")]


def test_run_one_coverage_cutoff_and_scoping(tmp_path):
    items = _lme_items()
    responses = {
        "dog's name": CallResult("Cooper", 10, 2, None, "stop"),        # correct
        "move": CallResult("Portland", 10, 2, None, "stop"),            # correct
        "car": CallResult("", 10, 0, "Timeout 500", None),              # transient error → excluded, retry
        "job": CallResult("bak", 10, 5, None, "length"),                # partial cutoff → excluded, retry
    }
    client = StubClient(responses)
    store = ResultStore(tmp_path / "s.jsonl")
    # #11: a stray record from a LARGER earlier run must NOT be scored by this (4-item) selection.
    store.append({"question_id": "999", "question": "stray", "gold": "x", "question_type": "multi-session",
                  "hypothesis": "totally wrong", "finish_reason": "stop", "error": None, "correct": None})

    agg, meta = asyncio.run(run_api_eval.run_one(
        client, "stub/model", "floor", items, None, 440_000, 5, None, 256, store, score_longmemeval, False))

    # coverage/accounting (#9/#10)
    assert meta["n"] == 4 and meta["n_scored"] == 2 and meta["coverage"] == 0.5
    assert meta["n_errors"] == 1 and meta["n_gen_cutoff"] == 1
    # only the 2 valid, correct answers count — stray '999' excluded by selection scoping (#11)
    assert agg["overall_accuracy"] == 1.0 and agg["n_nonabstention"] == 2
    # error + partial-cutoff are retryable → not "done" (of the 4 SELECTED items, only 1 & 2 are done;
    # the stray 999 is also 'done' but that's about resumption, not this run's scoring scope).
    assert store.done_ids() & {"1", "2", "3", "4"} == {"1", "2"}

    # resume: a second run re-requests ONLY the 2 unfinished (3=error, 4=cutoff), not the 2 done
    client.calls = 0
    asyncio.run(run_api_eval.run_one(
        client, "stub/model", "floor", items, None, 440_000, 5, None, 256, store, score_longmemeval, False))
    assert client.calls == 2


def test_run_one_threads_competency_for_mab(tmp_path):
    items = [
        {"question_id": "a", "question": "Map: foo", "answer": ["3"], "question_type": "Test_Time_Learning",
         "competency": "Test_Time_Learning", "source": "icl_banking77", "metric": "exact_match",
         "system": "sys", "question_template": "{question}\nlabel:", "context_header": "# Context",
         "full_history": "examples", "sessions": ["examples"]},
        {"question_id": "b", "question": "Who?", "answer": ["France"], "question_type": "Accurate_Retrieval",
         "competency": "Accurate_Retrieval", "source": "ruler_qa1", "metric": "substring_exact_match",
         "system": "sys", "question_template": "Question: {question}\nAnswer:", "context_header": "# Context",
         "full_history": "docs", "sessions": ["docs"]},
    ]
    responses = {"Map: foo": CallResult("3", 5, 1, None, "stop"),
                 "Who?": CallResult("the answer is France", 5, 3, None, "stop")}
    store = ResultStore(tmp_path / "m.jsonl")
    agg, meta = asyncio.run(run_api_eval.run_one(
        StubClient(responses), "stub/model", "full_context", items, None, 440_000, 5, None, 256,
        store, score_memoryagentbench, False))

    # #12: competency really flows through → per_competency has the real buckets, not one 'unknown'
    assert set(agg["per_competency"]) == {"Test_Time_Learning", "Accurate_Retrieval"}
    assert "unknown" not in agg["per_competency"]
    assert agg["overall_accuracy"] == 1.0 and meta["coverage"] == 1.0
