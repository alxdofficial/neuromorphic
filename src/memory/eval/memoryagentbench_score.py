"""Deterministic scorer for MemoryAgentBench (no LLM judge).

Faithful reimplementation of the repo's `utils/eval_other_utils.py` metrics (verified 2026-07-18, see
`docs/baselines/MEMORYAGENTBENCH_SCHEMA.md`): DrQA-style `normalize_answer`, `substring_exact_match`
(gold ∈ prediction), `exact_match` (equality), max-over-paraphrase-golds, and max-over-{raw output,
parse_output(output)}. Per-item metric is chosen by the item's `metric` field (set by the reader from the
sub-dataset). The two LLM-judged subsets (longmemeval, infbench_sum) are dropped by the reader, not here.

Record shape (from the runner): {hypothesis, answer (str | list[str] paraphrases), metric, source,
competency, question_id, question?}. `use_bem` is accepted for a uniform scorer signature but IGNORED
(MemoryAgentBench is pure string-match).
"""
from __future__ import annotations

import re
import string
from collections import defaultdict

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_ANSWER_PREFIX = re.compile(r"answer\s*:\s*(.*)", re.IGNORECASE)   # line-bounded (no DOTALL) — matches the repo
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """DrQA/SQuAD: lowercase, strip all punctuation, drop articles, collapse whitespace."""
    text = (text or "").lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def substring_exact_match(prediction: str, gold: str) -> bool:
    """gold ∈ prediction (the repo's lenient direction)."""
    return normalize_answer(gold) in normalize_answer(prediction)


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def parse_output(text: str) -> str:
    """Take the (rest of the) 'Answer:' line if present (the repo's behavior), else the first non-empty line."""
    text = text or ""
    m = _ANSWER_PREFIX.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return text.strip()


_METRIC_FNS = {"substring_exact_match": substring_exact_match, "exact_match": exact_match}


def _golds(answer) -> list[str]:
    if isinstance(answer, (list, tuple)):
        return [str(a) for a in answer]
    return [str(answer)]


def score_item(hypothesis: str, answer, metric: str) -> bool:
    """max over {raw, parse_output(raw)} × over paraphrase golds, for the item's metric."""
    fn = _METRIC_FNS.get(metric)
    if fn is None:                      # e.g. recall@5 (recsys) — not supported deterministically here
        return False
    cands = [hypothesis or "", parse_output(hypothesis or "")]
    return any(fn(c, g) for c in cands for g in _golds(answer))


class MemoryAgentBenchScorer:
    def __init__(self, use_bem: bool = False):
        del use_bem                     # MemoryAgentBench is pure string-match; kept for signature parity
        self._by_source: dict[str, list[bool]] = defaultdict(list)
        self._by_competency: dict[str, list[bool]] = defaultdict(list)
        self._skipped: dict[str, int] = defaultdict(int)
        self._details: list[dict] = []

    def add(self, rec: dict) -> None:
        metric = rec.get("metric", "")
        if metric not in _METRIC_FNS:                      # recsys / judged / unknown → not scored here
            self._skipped[rec.get("source", "unknown")] += 1
            return
        correct = score_item(rec.get("hypothesis", ""), rec.get("answer", ""), metric)
        src = rec.get("source", "unknown")
        # competency drives competency_averaged_accuracy; fall back to question_type (the reader sets both to
        # the same value) so it never silently degenerates to a single "unknown" bucket = the micro-average.
        comp = rec.get("competency") or rec.get("question_type") or "unknown"
        self._by_source[src].append(correct)
        self._by_competency[comp].append(correct)
        self._details.append({"question_id": rec.get("question_id", ""), "source": src,
                              "competency": comp, "metric": metric, "correct": correct})

    def aggregate(self) -> dict:
        def acc(v):
            return sum(v) / len(v) if v else 0.0
        per_source = {s: {"accuracy": acc(v), "n": len(v)} for s, v in self._by_source.items()}
        per_comp = {c: {"accuracy": acc(v), "n": len(v)} for c, v in self._by_competency.items()}
        allv = [x for v in self._by_source.values() for x in v]
        return {
            "overall_accuracy": acc(allv),
            "competency_averaged_accuracy": (sum(d["accuracy"] for d in per_comp.values()) / len(per_comp)
                                             if per_comp else 0.0),
            "per_competency": per_comp,
            "per_source": per_source,
            "per_subtask": per_source,          # uniform key the report tool reads (subtask = sub-dataset)
            "n_scored": len(allv),
            "n_skipped": dict(self._skipped),
        }


def score_memoryagentbench(records: list[dict], use_bem: bool = False) -> dict:
    """Score MemoryAgentBench prediction records; returns the aggregate with a `details` list attached."""
    scorer = MemoryAgentBenchScorer(use_bem=use_bem)
    for r in records:
        scorer.add(r)
    out = scorer.aggregate()
    out["details"] = scorer._details
    return out
