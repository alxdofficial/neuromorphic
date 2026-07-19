"""Regression tests for the SECOND audit's fixes (no network).

Covers: #1 question_date injection, #2 token-budget char fallback, #3 gold-alternate splitting,
#4 bare-number containment guard, #7 MemoryAgentBench per-competency prompts, #10 partial-cutoff retry,
#12 competency fallback, #14 recsys dropped, #15 stratified sampling.
"""

from src.memory.eval.baselines import build_messages, fit_history, _truncate_tail
from src.memory.eval.longmemeval_score import LongMemEvalScorer, _is_bare_number, _is_list_like
from src.memory.eval.results import ResultStore
from src.memory.eval.memoryagentbench_score import MemoryAgentBenchScorer
from src.memory.data.longmemeval import _stratified_sample, _text_question_type
from src.memory.data.memoryagentbench import (_query_template, _metric_competency, MAB_SYSTEM,
                                              _round_robin_stratified)


# --- #1 question_date injection ---
def test_build_messages_injects_question_date():
    msgs, _ = build_messages("floor", question="How long ago?", question_date="2023/04/10 (Mon) 23:07")
    assert "Current Date: 2023/04/10 (Mon) 23:07" in msgs[-1]["content"]


def test_build_messages_no_date_line_when_absent():
    msgs, _ = build_messages("floor", question="q")
    assert "Current Date:" not in msgs[-1]["content"]


# --- #2 token-budget char fallback (no tokenizer needed) ---
def test_fit_history_char_fallback_keeps_tail():
    text = "OLDEST" + ("x" * 100) + "NEWEST"
    kept, trunc = fit_history(text, token_budget=None, char_budget=10)
    assert trunc and kept.endswith("NEWEST") and "OLDEST" not in kept


def test_fit_history_no_truncation_when_short():
    kept, trunc = fit_history("short", token_budget=None, char_budget=10_000)
    assert kept == "short" and trunc is False


# --- fix-of-fix: budget == 0 must keep NOTHING (guard the negative-zero slice ids[-0:] == whole list) ---
def test_zero_budget_keeps_nothing_not_everything():
    text = "OLD " * 500 + "NEWEST"
    assert _truncate_tail(text, 0) == ("", True)                      # char path
    assert fit_history(text, token_budget=0, char_budget=999) == ("", True)  # token path (would overflow!)


# --- #3 gold-alternate splitting: whitespace-guarded slash + verbose "... is also acceptable" ---
def test_ratio_gold_not_split_on_bare_slash():
    s = LongMemEvalScorer(use_bem=False)
    # '1/48' must stay atomic — a hypothesis of just '1' must NOT match
    assert s.score_item("q", "1/48", "1", "multi-session", "q")["correct"] is False


def test_spaced_slash_is_a_real_alternation():
    s = LongMemEvalScorer(use_bem=False)
    assert s.score_item("q", "cat / dog", "I think it was a dog", "multi-session", "q")["correct"] is True


def test_verbose_acceptable_gold_exposes_both_answers():
    s = LongMemEvalScorer(use_bem=False)
    gold = "7 days. 8 days (including the last day) is also acceptable."
    assert s.score_item("q", gold, "7 days", "temporal", "q1")["correct"] is True
    assert s.score_item("q", gold, "8 days", "temporal", "q2")["correct"] is True


# --- #4 bare-number containment guard ---
def test_bare_number_helpers():
    assert _is_bare_number("2") and _is_bare_number("two") and not _is_bare_number("2 dogs")
    assert _is_list_like("1. A, 2. B, 3. C") and not _is_list_like("You worked on 2 projects")


def test_bare_number_not_matched_in_enumeration():
    s = LongMemEvalScorer(use_bem=False)
    hyp = "You led 3 projects: 1. Alpha, 2. Beta, 3. Gamma"
    assert s.score_item("how many?", "2", hyp, "multi-session", "q")["correct"] is False


def test_bare_number_matched_in_plain_answer():
    s = LongMemEvalScorer(use_bem=False)
    assert s.score_item("how many?", "2", "You worked on 2 projects", "multi-session", "q")["correct"] is True


def test_enumerated_but_correct_number_is_accepted():
    # fix-of-fix: the answer IS 3 and it's stated ('3 projects') even though the reply also enumerates —
    # must NOT be rejected by the list-index guard (was a false-negative in BEM-off mode).
    s = LongMemEvalScorer(use_bem=False)
    hyp = "You led 3 projects last year: 1. Alpha, 2. Beta, 3. Gamma."
    assert s.score_item("how many?", "3", hyp, "multi-session", "q")["correct"] is True
    # but gold '2' (only present as the '2.' list marker) is still correctly rejected
    assert s.score_item("how many?", "2", hyp, "multi-session", "q")["correct"] is False


# --- fix-of-fix: verb-phrase " or " disjunction must NOT split into a generic prefix that false-matches ---
def test_or_disjunction_in_long_gold_does_not_false_match():
    s = LongMemEvalScorer(use_bem=False)
    gold = ("I have worked on or bought five model kits. The scales of the models are: "
            "Revell F-15 Eagle, Tamiya 1/48 scale Spitfire.")
    wrong = "I have worked on a few unrelated writing assignments recently."
    assert s.score_item("what have you done?", gold, wrong, "multi-session", "q")["correct"] is False


def test_short_or_alternation_still_splits():
    s = LongMemEvalScorer(use_bem=False)
    assert s.score_item("which?", "cat or dog", "I believe it was a dog", "multi-session", "q")["correct"] is True


# --- #7 MemoryAgentBench per-competency prompts (verbatim + literal {label} survives) ---
def test_mab_query_templates_are_task_specific():
    assert 'label: {label}' in _query_template("icl_banking77")               # ICL numeric-label instruction
    assert "larger serial number" in _query_template("factconsolidation_en")  # conflict rule
    assert "strict output format" in _query_template("detective_qa")
    assert _query_template("unknown_src") == ""                               # unknown → default block


def test_build_messages_mab_template_preserves_literal_label():
    tmpl = _query_template("icl_banking77")
    msgs, _ = build_messages("full_context", question="Map this: foo", full_history="the examples",
                             system=MAB_SYSTEM, question_template=tmpl, context_header="# Context")
    body, sysmsg = msgs[-1]["content"], msgs[0]["content"]
    assert sysmsg == MAB_SYSTEM
    assert "Map this: foo" in body           # {question} substituted
    assert "{label}" in body                 # literal {label} NOT consumed by .format (would KeyError)
    assert "# Context" in body and "the examples" in body


# --- #10 partial (non-empty) length cutoff is retryable ---
def test_nonempty_length_cutoff_is_retryable(tmp_path):
    s = ResultStore(tmp_path / "s.jsonl")
    s.append({"question_id": "a", "hypothesis": "partial ans", "finish_reason": "length", "error": None})
    s.append({"question_id": "b", "hypothesis": "done", "finish_reason": "stop", "error": None})
    assert s.done_ids() == {"b"}             # 'a' was cut off mid-answer → retry, don't freeze/score it


# --- #12 competency falls back to question_type (never degenerates to a single 'unknown' bucket) ---
def test_mab_competency_fallback_to_question_type():
    sc = MemoryAgentBenchScorer()
    sc.add({"hypothesis": "France", "answer": ["France"], "metric": "substring_exact_match",
            "source": "ruler_qa", "question_type": "Accurate_Retrieval", "question_id": "x"})  # no competency
    agg = sc.aggregate()
    assert "Accurate_Retrieval" in agg["per_competency"] and "unknown" not in agg["per_competency"]


# --- #14 recsys dropped from the deterministic loader map ---
def test_recsys_and_judged_sources_dropped():
    assert _metric_competency("recsys_redial_full") == (None, None)
    assert _metric_competency("longmemeval_s") == (None, None)
    assert _metric_competency("ruler_qa1_197K")[0] == "substring_exact_match"


# --- #15 stratified sampling spans question types (not a single-type prefix) ---
def test_stratified_sample_spans_types():
    raw = ([{"question_id": f"u{i}", "question_type": "single-session-user"} for i in range(10)] +
           [{"question_id": f"t{i}", "question_type": "temporal-reasoning"} for i in range(10)])
    sample = _stratified_sample(raw, 4, _text_question_type)
    assert len(sample) == 4
    assert {_text_question_type(x) for x in sample} == {"single-session-user", "temporal-reasoning"}


def test_stratified_preference_kept_distinct():
    ex = {"question_id": "p_abs", "question_type": "single-session-preference"}
    assert _text_question_type(ex) == "abstention"   # _abs suffix wins
    ex2 = {"question_id": "p1", "question_type": "single-session-preference"}
    assert _text_question_type(ex2) == "single-session-preference"   # NOT merged to 'single-session'


# --- fix-of-fix: MAB stratification must span COMPETENCIES, not fine-grained sources (a small sample was
# 100% Accurate_Retrieval because it has the most distinct `source` strings and is inserted first) ---
def test_mab_round_robin_spans_competencies_not_sources():
    mk = lambda: [{"x": i} for i in range(20)]
    by_comp_source = {                                   # AR has 3 sources; others have 1 (real shape)
        "Accurate_Retrieval": {"ruler_qa1": mk(), "ruler_qa2": mk(), "eventqa": mk()},
        "Test_Time_Learning": {"icl": mk()},
        "Long_Range_Understanding": {"detective": mk()},
        "Conflict_Resolution": {"factconsolidation": mk()},
    }
    # per the old flat-by-source round-robin, max_examples=4 was 100% Accurate_Retrieval; now it must span all 4.
    sample = _round_robin_stratified(by_comp_source, 4)
    assert len(sample) == 4
    # reconstruct which competency each item came from (impossible from {'x':i}, so re-run with tagged items)
    tagged = {c: [{"comp": c} for _ in range(20)] for c in by_comp_source}
    bcs = {"Accurate_Retrieval": {"a": tagged["Accurate_Retrieval"][:20], "b": [], "c": []},
           "Test_Time_Learning": {"a": tagged["Test_Time_Learning"]},
           "Long_Range_Understanding": {"a": tagged["Long_Range_Understanding"]},
           "Conflict_Resolution": {"a": tagged["Conflict_Resolution"]}}
    s2 = _round_robin_stratified(bcs, 4)
    assert {it["comp"] for it in s2} == set(by_comp_source)   # one from every competency
