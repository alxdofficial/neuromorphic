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
# {"answer": "..."} — the detective_qa prompt demands single-line JSON, which _ANSWER_PREFIX cannot match
# because the closing quote of the key sits between `answer` and `:`.
_JSON_ANSWER = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', re.IGNORECASE)
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
    """Take the (rest of the) 'Answer:' line if present (the repo's behavior), else the first non-empty line.

    Two extractions run BEFORE the first-non-empty-line fallback, because our own prompts mandate output
    formats that strict exact_match would otherwise reject 100% of the time:
      - detective_qa is instructed to emit single-line JSON, so the answer arrives as {"answer": "..."}.
        `_ANSWER_PREFIX` cannot match it (the quote sits between `answer` and `:`), and the fallback then
        compares the whole blob — including the `reasoning` field — against a short gold. 0/71 for EVERY
        model, deepseek included, despite character-identical answers.
      - ICL is instructed 'Only output "label: {label}"' while the gold is a bare number, so every compliant
        response fails. 0/500 for every model.
    Both were structurally unsatisfiable, not capability results (deepseek full_context: 0.789 / 0.828
    once parsed). Handled here rather than by loosening the metric, so exact_match stays exact.
    """
    text = text or ""
    m = _JSON_ANSWER.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    m = _ANSWER_PREFIX.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    for line in text.splitlines():
        if line.strip():
            return _LABEL_PREFIX.sub("", line.strip()).strip()
    return _LABEL_PREFIX.sub("", text.strip()).strip()


# substring_exact_match → RULER (gold ∈ RAW output). substring_parsed → EventQA (gold ∈ the PARSED answer
# line): EventQA answers are a single parsed line upstream, so substring-on-raw over-credits a gold that
# merely appears later in an otherwise-wrong/refusal response (audit #5). exact_match → ICL/detective (parsed).
_METRIC_FNS = {"substring_exact_match": substring_exact_match, "substring_parsed": substring_exact_match,
               "exact_match": exact_match}
_LABEL_PREFIX = re.compile(r"^\s*label\s*:\s*", re.IGNORECASE)


# --- Recsys Recall@k (faithful port of the MAB repo's utils/eval_other_utils._process_recsys_dataset) ------
# OPT-IN, excluded from the default judge-free panel: it is deterministic but needs the repo's
# processed_data/Recsys_Redial/entity2id.json (movie-name↔id map) + a ranked-list model output. Provide the
# map to `score_recsys(...)` to enable it; otherwise MAB coverage is 4/5 competencies (Recsys omitted).
def _levenshtein(a: str, b: str) -> int:
    try:
        import editdistance  # fast C ext if available
        return editdistance.eval(a, b)
    except Exception:  # noqa: BLE001 — pure-python fallback (no hard dep)
        if a == b:
            return 0
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def extract_movie_name(name: str) -> str:
    """basename after '/', '_-'->space, drop parenthetical, collapse whitespace (repo extract_movie_name)."""
    name = name.split("/")[-1]
    name = re.sub(r"[_\-]", " ", name)
    name = re.sub(r"\([^()]*\)", "", name)
    return " ".join(name.split())


def _clean_recsys_item(item: str) -> str:
    """strip parens, strip leading numbering '1.'/'2)'/'3、', collapse whitespace (repo clean_text_elements)."""
    item = re.sub(r"\([^()]*\)", "", item)
    item = re.sub(r"^(?:\d+[.)、]?\s*[\-—–]?\s*)?", "", item.strip())
    return " ".join(item.split())


def _find_nearest_movie(target: str, candidates: list[str]) -> str:
    return min(candidates, key=lambda c: _levenshtein(target.lower(), c.lower())) if candidates else target


def extract_recommendation_list(text: str, candidates: list[str]) -> list[str]:
    """Parse a model's numbered recommendation list → fuzzy-snapped candidate names, in output order."""
    text = text or ""
    if "1." in text:
        body = text.split("1.", 1)[1]
    else:
        body = text.replace(",", "\n")
    raw = [_clean_recsys_item(x) for x in body.split("\n") if x.strip()]
    return [_find_nearest_movie(x, candidates) for x in raw]


def recall_at_k(prediction: str, gold_ids, name_to_id: dict, k: int = 5) -> float:
    """Recall@k = |{gold movies in the top-k predicted}| / |gold|. `name_to_id` = repo entity2id.json
    ({movie_name: entity_id}); `gold_ids` = list of id-strings (the item's answer). Faithful to the repo."""
    id_to_name = {int(eid): extract_movie_name(n) for n, eid in name_to_id.items()}
    candidates = list(id_to_name.values())
    predicted = extract_recommendation_list(prediction, candidates)
    gold = [id_to_name[int(str(g).strip())] for g in gold_ids if str(g).strip().lstrip("-").isdigit()]
    if not gold:
        return 0.0
    return sum(g in predicted[:k] for g in gold) / len(gold)


def score_recsys(records: list[dict], name_to_id: dict, k: int = 5) -> dict:
    """Mean Recall@k over recsys records (each: {hypothesis, answer (id-strings)}). OPT-IN — call explicitly
    with the entity2id map; NOT part of the default judge-free aggregate (which is boolean accuracy)."""
    vals = [recall_at_k(r.get("hypothesis", ""), r.get("answer", []), name_to_id, k)
            for r in records if r.get("metric") == "recall@5"]
    return {f"recsys_recall@{k}": (sum(vals) / len(vals) if vals else 0.0), "n": len(vals)}


def _golds(answer) -> list[str]:
    if isinstance(answer, (list, tuple)):
        return [str(a) for a in answer]
    return [str(answer)]


def score_item(hypothesis: str, answer, metric: str) -> bool:
    """SOURCE-FAITHFUL per-metric scoring (audit #14): `substring_exact_match` (RULER/EventQA-retrieval) checks
    gold ∈ the RAW output, as upstream does; `exact_match` (ICL/EventQA/detective) checks the PARSED answer
    line (`parse_output`), NOT the raw output. The previous max-over-{raw, parsed} added leniency exact_match
    upstream does not have (it could credit a gold appearing anywhere in a verbose output). Max over the
    paraphrase golds is faithful."""
    fn = _METRIC_FNS.get(metric)
    if fn is None:                      # e.g. recall@5 (recsys) — not supported deterministically here
        return False
    # RULER substring matches the RAW output; EventQA (substring_parsed) + exact_match match the PARSED line.
    cand = (hypothesis or "") if metric == "substring_exact_match" else parse_output(hypothesis or "")
    return any(fn(cand, g) for g in _golds(answer))


def score_item_lenient(hypothesis: str, answer, metric: str) -> bool:
    """INTENT-PARSED metric (audit #13) — reported SEPARATELY, NOT the reproduction number. The strict
    exact-match produces unintuitive zeros the benchmark itself documents (e.g. `label: 43` vs gold `43`).
    This forgives label-prefix + direction: strip a leading `label:` on both sides and accept bidirectional
    containment / equality of the parsed answer. Use it for ARCHITECTURE interpretation, alongside strict."""
    if metric not in _METRIC_FNS:
        return False
    if score_item(hypothesis, answer, metric):     # lenient is a SUPERSET of strict — never scores below it
        return True
    pred = _LABEL_PREFIX.sub("", parse_output(hypothesis or ""))
    for g in _golds(answer):
        gg = _LABEL_PREFIX.sub("", g)
        if not normalize_answer(gg):
            continue
        if exact_match(pred, gg) or substring_exact_match(pred, gg) or substring_exact_match(gg, pred):
            return True
    return False


class MemoryAgentBenchScorer:
    def __init__(self, use_bem: bool = False):
        del use_bem                     # MemoryAgentBench is pure string-match; kept for signature parity
        self._by_source: dict[str, list[bool]] = defaultdict(list)
        self._by_competency: dict[str, list[bool]] = defaultdict(list)
        self._by_competency_lenient: dict[str, list[bool]] = defaultdict(list)   # audit #13 intent-parsed
        self._skipped: dict[str, int] = defaultdict(int)
        self._details: list[dict] = []

    def add(self, rec: dict) -> None:
        metric = rec.get("metric", "")
        src = rec.get("source", "unknown")
        # audit #5: correct EventQA at score time even for OLD caches whose stored metric predates the split
        # (EventQA was written as substring_exact_match; it must score the PARSED answer → substring_parsed).
        if metric == "substring_exact_match" and str(src).lower().startswith("eventqa"):
            metric = "substring_parsed"
        if metric not in _METRIC_FNS:                      # recsys / judged / unknown → not scored here
            self._skipped[src] += 1
            return
        correct = score_item(rec.get("hypothesis", ""), rec.get("answer", ""), metric)
        lenient = score_item_lenient(rec.get("hypothesis", ""), rec.get("answer", ""), metric)
        src = rec.get("source", "unknown")
        # competency drives competency_averaged_accuracy; fall back to question_type (the reader sets both to
        # the same value) so it never silently degenerates to a single "unknown" bucket = the micro-average.
        comp = rec.get("competency") or rec.get("question_type") or "unknown"
        self._by_source[src].append(correct)
        self._by_competency[comp].append(correct)
        self._by_competency_lenient[comp].append(lenient)
        self._details.append({"question_id": rec.get("question_id", ""), "source": src,
                              "competency": comp, "metric": metric, "correct": correct,
                              "correct_lenient": lenient})

    def aggregate(self) -> dict:
        def acc(v):
            return sum(v) / len(v) if v else 0.0
        per_source = {s: {"accuracy": acc(v), "n": len(v)} for s, v in self._by_source.items()}
        per_comp = {c: {"accuracy": acc(v), "n": len(v)} for c, v in self._by_competency.items()}
        per_comp_len = {c: acc(v) for c, v in self._by_competency_lenient.items()}
        allv = [x for v in self._by_source.values() for x in v]
        all_len = [x for v in self._by_competency_lenient.values() for x in v]
        comp_macro = (sum(d["accuracy"] for d in per_comp.values()) / len(per_comp)) if per_comp else 0.0
        return {
            # PRIMARY reproduction metric = strict per-source micro-average, but report the competency
            # MACRO-average too (audit #15: micro is dominated by Accurate_Retrieval's 1,700 Q). The lenient
            # (intent-parsed) numbers are an ADAPTATION for architecture interpretation, NOT the headline.
            "overall_accuracy": acc(allv),
            "competency_averaged_accuracy": comp_macro,
            "overall_accuracy_lenient": acc(all_len),
            "competency_averaged_accuracy_lenient": (sum(per_comp_len.values()) / len(per_comp_len)
                                                     if per_comp_len else 0.0),
            "per_competency": per_comp,
            "per_competency_lenient": per_comp_len,
            "per_source": per_source,
            "per_subtask": per_source,          # uniform key the report tool reads (subtask = sub-dataset)
            "n_scored": len(allv),
            "n_skipped": dict(self._skipped),
            # DISCLOSURE (audit): the default judge-free panel covers 4/5 competencies — Recsys is EXCLUDED
            # (its deterministic Recall@5 needs entity2id.json + a ranked-list output; use score_recsys(...)
            # opt-in). longmemeval/infbench_sum judge subsets are also dropped upstream of scoring.
            "coverage_note": "4/5 competencies (Accurate_Retrieval, Test_Time_Learning, "
                             "Long_Range_Understanding, Conflict_Resolution); Recsys excluded (opt-in Recall@5)",
        }


def score_memoryagentbench(records: list[dict], use_bem: bool = False) -> dict:
    """Score MemoryAgentBench prediction records; returns the aggregate with a `details` list attached."""
    scorer = MemoryAgentBenchScorer(use_bem=use_bem)
    for r in records:
        scorer.add(r)
    out = scorer.aggregate()
    out["details"] = scorer._details
    return out
