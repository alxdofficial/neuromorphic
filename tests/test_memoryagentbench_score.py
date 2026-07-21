"""Offline tests for the MemoryAgentBench deterministic scorer (no network)."""

from src.memory.eval.memoryagentbench_score import (
    normalize_answer, substring_exact_match, exact_match, parse_output,
    score_item, score_memoryagentbench,
)


def test_normalize_drqa_style():
    assert normalize_answer("The, France!") == "france"
    assert normalize_answer("  an  Apple ") == "apple"


def test_substring_direction_is_gold_in_pred():
    # gold ∈ prediction (lenient): a rambling answer containing the gold scores correct
    assert substring_exact_match("I think the answer is France, probably.", "France")
    # but the reverse (pred ∈ gold) must NOT score
    assert not substring_exact_match("France and Germany and Spain", "France is the capital region of Europe")


def test_exact_match_strict_after_normalize():
    assert exact_match("The France", "france")
    assert not exact_match("France and Spain", "France")


def test_parse_output_strips_answer_prefix_and_takes_first_line():
    assert parse_output("Answer: 28\nreasoning: ...") == "28"
    assert parse_output("blah\nAnswer: yes") == "yes"        # regex searches anywhere
    assert parse_output("just text") == "just text"


def test_score_item_max_over_paraphrases_and_parsed():
    # exact_match only succeeds on the parsed form ("28" == "28"), not the raw CoT dump
    assert score_item("Let me think...\nAnswer: 28", ["28"], "exact_match")
    # substring succeeds via a paraphrase gold present in the prediction
    assert score_item("the capital is Paris.", ["London", "Paris"], "substring_exact_match")
    assert not score_item("the capital is Berlin.", ["London", "Paris"], "substring_exact_match")


def test_recall_at_5_metric_is_skipped_not_crashed():
    assert score_item("anything", ["123"], "recall@5") is False   # unsupported metric → False, no crash


def test_aggregate_groups_and_skips():
    recs = [
        {"hypothesis": "France", "answer": ["France"], "metric": "substring_exact_match",
         "source": "ruler_qa1_197K", "competency": "Accurate_Retrieval", "question_id": "a"},
        {"hypothesis": "Answer: 28", "answer": ["28"], "metric": "exact_match",
         "source": "icl_banking77", "competency": "Test_Time_Learning", "question_id": "b"},
        {"hypothesis": "wrong", "answer": ["right"], "metric": "exact_match",
         "source": "detective_qa", "competency": "Long_Range_Understanding", "question_id": "c"},
        {"hypothesis": "x", "answer": ["1"], "metric": "recall@5",
         "source": "recsys_redial_full", "competency": "Recsys", "question_id": "d"},   # skipped
    ]
    agg = score_memoryagentbench(recs)
    assert agg["n_scored"] == 3
    assert agg["n_skipped"] == {"recsys_redial_full": 1}
    assert agg["overall_accuracy"] == 2 / 3                       # France + 28 correct, detective wrong
    assert agg["per_competency"]["Accurate_Retrieval"]["accuracy"] == 1.0
    assert agg["per_competency"]["Long_Range_Understanding"]["accuracy"] == 0.0


# --- regression: two competencies scored 0.000 for EVERY model because our own prompts mandated an output
# --- format the metric rejected. See commit "fix two structural scoring zeros".

def test_parse_output_extracts_json_answer_field():
    """detective_qa is prompted for single-line JSON. `_ANSWER_PREFIX` can't match `"answer":` (the key's
    closing quote sits between `answer` and `:`), so the whole blob — reasoning included — used to be
    compared against a short gold. 0/71 for deepseek, llama and h2o alike."""
    from src.memory.eval.memoryagentbench_score import exact_match, parse_output
    hyp = ('{"answer":"C. The Brandt couple", "reasoning":"The deceased, Quentin Dugastlin, was murdered '
           'by the couple who had been blackmailed for years."}')
    assert parse_output(hyp) == "C. The Brandt couple"
    assert exact_match(parse_output(hyp), "C. The Brandt couple")


def test_parse_output_strips_label_prefix():
    """ICL is prompted 'Only output "label: {label}"' while the gold is a bare number — so every response
    that OBEYED the instruction scored zero."""
    from src.memory.eval.memoryagentbench_score import exact_match, parse_output
    for raw, gold in (("label: 19", "19"), ("label: 0", "0"), ("LABEL:  43", "43")):
        assert parse_output(raw) == gold
        assert exact_match(parse_output(raw), gold)


def test_parse_output_preserves_existing_behaviour():
    """The new extractions must not disturb the plain and `Answer:` paths."""
    from src.memory.eval.memoryagentbench_score import parse_output
    assert parse_output("Answer: Paris") == "Paris"
    assert parse_output("Paris") == "Paris"
    assert parse_output("") == ""
    assert parse_output("first line\nsecond line") == "first line"


def test_parse_output_json_with_escaped_quotes():
    from src.memory.eval.memoryagentbench_score import parse_output
    assert parse_output(r'{"answer": "he said \"hi\" then left"}') == r'he said \"hi\" then left'
