"""Regression tests for the THIRD review's fixes (Batches A/B/C), no network.

A (Tier-1 scoring): #1 negation-guard + paren-note drop, #2 preference polarity + low-confidence flag,
#3 non-official disclaimer. B (MAB): #4 source-diverse stratification, #5 faithful Recall@5. C (Tier-2):
#6 M+ field names, #7 512-token injection blocks.
"""
import importlib.util
from pathlib import Path

from src.memory.eval.longmemeval_score import LongMemEvalScorer, _gold_candidates, _avoidance_tokens
from src.memory.eval import score_longmemeval
from src.memory.eval.memoryagentbench_score import recall_at_k, extract_movie_name, score_recsys
from src.memory.data.memoryagentbench import _round_robin_stratified

REPO = Path(__file__).resolve().parents[1]


# ---------- A#1: negation guard ----------
def test_negation_vetoes_wrong_answer():
    s = LongMemEvalScorer(use_bem=False)
    for gold, hyp in [("2", "The answer is 3, not 2."), ("Paris", "It was London, not Paris."),
                      ("yes", "No, not yes."), ("four", "It was five, not four.")]:
        assert s.score_item("q", gold, hyp, "multi-session", "x")["correct"] is False


def test_non_negated_still_matches():
    s = LongMemEvalScorer(use_bem=False)
    for gold, hyp in [("2", "You worked on 2 projects"), ("Paris", "The capital is Paris."),
                      ("25 minutes", "it took 25 minutes total")]:
        assert s.score_item("q", gold, hyp, "multi-session", "x")["correct"] is True


# ---------- A#1: parenthetical note dropped, real answers kept ----------
def test_paren_note_dropped_but_answers_kept():
    gold = "7 days. 8 days (including the last day) is also acceptable."
    cands = [c.strip() for c in _gold_candidates(gold)]
    assert "including the last day" not in cands       # explanatory note is NOT an acceptable answer
    assert "7 days" in cands and "8 days" in cands     # the real alternates still exposed
    s = LongMemEvalScorer(use_bem=False)
    assert s.score_item("q", gold, "sure, including the last day of the trip", "temporal", "x")["correct"] is False
    assert s.score_item("q", gold, "7 days", "temporal", "x")["correct"] is True


def test_paren_alternate_still_kept():
    # "(or 25:50)" IS an alternate → keep; a short atomic "(25:50)" too
    cands = [c.strip() for c in _gold_candidates("25 minutes and 50 seconds (or 25:50)")]
    assert "25:50" in cands


# ---------- A#2: preference polarity ----------
def test_preference_avoidance_tokens():
    assert _avoidance_tokens("prefers Adobe Premiere, avoid generic advice.") == {"generic", "advice"}


def test_preference_rejects_prohibited_only_response():
    s = LongMemEvalScorer(use_bem=False)
    rubric = "The user prefers detailed Adobe Premiere tutorials; avoid generic advice."
    assert s.score_item("q", rubric, "Here is some generic advice.", "single-session-preference", "p")["correct"] is False


# ---------- A#2/#3: aggregate flags ----------
def test_aggregate_flags_preference_and_discloses_metric():
    recs = [{"question": "q", "answer": "prefers detailed Adobe Premiere tutorials",
             "hypothesis": "detailed Adobe Premiere tutorials", "question_type": "single-session-preference",
             "question_id": "p1"}]
    agg = score_longmemeval(recs, use_bem=False)
    assert agg["per_subtask"]["single-session-preference"].get("low_confidence") is True
    assert "not the official" in agg["scoring_note"].lower()


# ---------- B#4: stratification spans SOURCES within a competency ----------
def test_stratified_spans_sources_within_competency():
    bcs = {"Accurate_Retrieval": {"ruler_qa1": [{"src": "ruler_qa1"} for _ in range(10)],
                                  "ruler_qa2": [{"src": "ruler_qa2"} for _ in range(10)],
                                  "eventqa": [{"src": "eventqa"} for _ in range(10)]}}
    sample = _round_robin_stratified(bcs, 6)
    assert len(sample) == 6
    assert {x["src"] for x in sample} == {"ruler_qa1", "ruler_qa2", "eventqa"}   # not 6 from one source


# ---------- B#5: faithful Recall@5 ----------
_NAME_TO_ID = {"The Matrix (1999)": 1, "Inception": 2, "Titanic": 3, "Avatar": 4,
               "Interstellar": 5, "Gladiator": 6}
_PRED = "Recommendations: 1. The Matrix\n2. Inception\n3. Titanic\n4. Avatar\n5. Interstellar\n6. Gladiator"


def test_extract_movie_name():
    assert extract_movie_name("path/to/The_Matrix (1999)") == "The Matrix"


def test_recall_at_5_faithful():
    assert recall_at_k(_PRED, ["1", "2"], _NAME_TO_ID, 5) == 1.0     # both in the top-5
    assert recall_at_k(_PRED, ["6"], _NAME_TO_ID, 5) == 0.0          # Gladiator is 6th → out of top-5
    assert recall_at_k(_PRED, ["1", "6"], _NAME_TO_ID, 5) == 0.5     # one of two


def test_score_recsys_mean():
    recs = [{"hypothesis": _PRED, "answer": ["1", "2"], "metric": "recall@5"},
            {"hypothesis": _PRED, "answer": ["6"], "metric": "recall@5"}]
    out = score_recsys(recs, _NAME_TO_ID, 5)
    assert out["n"] == 2 and out["recsys_recall@5"] == 0.5           # mean of 1.0 and 0.0


# ---------- C#6/#7: M+ runner (import via importlib — it's a script) ----------
def _load_memoryllm():
    spec = importlib.util.spec_from_file_location(
        "run_memoryllm", REPO / "scripts/baselines/tier2/run_memoryllm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _StubTok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}        # 1 id per word


def test_mplus_real_field_names_present():
    mod = _load_memoryllm()
    for real in ("ltm_recall_frequencies", "cached_dropped_memories",
                 "cached_dropped_memory_ages", "cached_dropped_keys"):
        assert real in mod._LTM_CANDIDATE_ATTRS


def test_inject_blocks_512():
    mod = _load_memoryllm()
    tok = _StubTok()
    hist = " ".join(["w"] * 1000)                                    # 1000 tokens
    blocks = mod._inject_blocks(tok, hist, block_tokens=512, min_tokens=17)
    assert [len(b) for b in blocks] == [512, 488]                    # fixed 512-token blocks
    # short trailing block (< min) is merged into the previous
    blocks2 = mod._inject_blocks(tok, " ".join(["w"] * 520), block_tokens=512, min_tokens=17)
    assert [len(b) for b in blocks2] == [520]
