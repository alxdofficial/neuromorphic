"""Tests for the resumable per-question ResultStore (no network)."""

from src.memory.eval.results import ResultStore, store_path


def _rec(qid, hyp="ans", error=None):
    return {"question_id": qid, "question": f"q{qid}", "gold": "g", "hypothesis": hyp, "error": error,
            "prompt_tokens": 10, "completion_tokens": 2, "correct": None, "score_method": None}


def test_append_persists_and_reloads(tmp_path):
    p = tmp_path / "s.jsonl"
    s = ResultStore(p)
    s.append(_rec("a")); s.append(_rec("b"))
    s2 = ResultStore(p)                                # fresh load from disk
    assert set(s2.records) == {"a", "b"}
    assert s2.done_ids() == {"a", "b"}


def test_errors_are_not_done_and_get_retried(tmp_path):
    p = tmp_path / "s.jsonl"
    s = ResultStore(p)
    s.append(_rec("a"))
    s.append(_rec("b", hyp="", error="Timeout"))
    assert s.done_ids() == {"a"}                       # errored 'b' is NOT done → will be retried


def test_length_cutoff_empty_is_retryable(tmp_path):
    p = tmp_path / "s.jsonl"
    s = ResultStore(p)
    s.append({**_rec("a"), "finish_reason": "stop"})                    # good answer → done
    s.append({**_rec("b", hyp=""), "finish_reason": "length"})         # cut off, empty → NOT done (retry)
    s.append({**_rec("c", hyp="Refuse."), "finish_reason": "stop"})    # non-empty → done
    assert s.done_ids() == {"a", "c"}


def test_last_wins_on_retry(tmp_path):
    p = tmp_path / "s.jsonl"
    s = ResultStore(p)
    s.append(_rec("a", hyp="", error="Timeout"))       # first attempt errored
    s.append(_rec("a", hyp="fixed", error=None))       # retry succeeded
    s2 = ResultStore(p)
    assert s2.records["a"]["hypothesis"] == "fixed"     # last-wins
    assert s2.done_ids() == {"a"}


def test_tolerates_torn_final_line(tmp_path):
    p = tmp_path / "s.jsonl"
    s = ResultStore(p); s.append(_rec("a"))
    with open(p, "a") as f:
        f.write('{"question_id": "b", "hypothesis": "x"')   # truncated (crash) line, no newline/close
    s2 = ResultStore(p)                                # must not raise
    assert "a" in s2.records                            # the good record survives


def test_merge_verdicts_and_compact(tmp_path):
    p = tmp_path / "s.jsonl"
    s = ResultStore(p)
    s.append(_rec("a", hyp="", error="Timeout")); s.append(_rec("a", hyp="ok"))  # 2 lines, same qid
    s.merge_verdicts([{"question_id": "a", "correct": True, "method": "exact_match"}])
    s.compact()
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1                              # deduped
    s2 = ResultStore(p)
    assert s2.records["a"]["correct"] is True and s2.records["a"]["score_method"] == "exact_match"


def test_summary_counts(tmp_path):
    p = tmp_path / "s.jsonl"
    s = ResultStore(p)
    s.append({**_rec("a"), "correct": True})
    s.append({**_rec("b"), "correct": False})
    s.append(_rec("c", hyp="", error="boom"))
    summ = s.summary()
    assert summ == {"n_total": 3, "n_answered": 2, "n_errored": 1, "n_scored": 2,
                    "n_correct": 1, "accuracy": 0.5, "errored_ids": ["c"]}


def test_store_path_slugs_colon(tmp_path):
    p = store_path(tmp_path, "longmemeval", "nvidia/nemotron:free", "floor")
    assert p.name == "longmemeval__nemotron-free__floor.jsonl"
    assert p.parent.name == "cache"
