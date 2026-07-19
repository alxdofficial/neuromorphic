"""Deterministic LongMemEval scorer — policy-compliant (NO LLM-as-judge).

LongMemEval (Wu et al., ICLR'25) officially scores free-form answers with a
GPT-4o judge (`src/evaluation/evaluate_qa.py` in xiaowu0162/LongMemEval). That
violates our project policy (no LLM-as-judge; prefer EM + containment/recall,
BEM for paraphrase, NOT BERTScore). This module reproduces a defensible,
deterministic substitute so every baseline panel is scored by identical code.

Per LongMemEval's 5-way taxonomy (+ abstention), answers fall into three
scoring regimes — inspected against the real `longmemeval_oracle.json`:

  FACTUAL  single-session-user / -assistant, multi-session, temporal(-reasoning),
           knowledge-update.  Gold is a short factual string ("Business
           Administration", "the suburbs", "four", "25 minutes and 50 seconds
           (or 25:50)").  Correct if ANY of: normalized exact-match |
           containment (all gold content-tokens present in the hypothesis) |
           BEM paraphrase-equivalence >= threshold.

  PREFERENCE  single-session-preference.  Gold is a RUBRIC string ("The user
           would prefer responses tailored to Adobe Premiere...") — matching a
           rubric verbatim is a category error, so we score keyword-coverage:
           the fraction of the rubric's salient tokens (proper nouns / capitalized
           / non-stopword content words) present in the hypothesis.  Heuristic
           and flagged lower-confidence.

  ABSTENTION  question_id ends `_abs` (~30/500).  The model must SAY it cannot
           answer.  Scored by a fixed refusal-lexicon detector on the hypothesis,
           NOT by answer-matching.

Reporting mirrors the three distinct numbers of the official
`print_qa_metrics.py`:
  - overall_accuracy       micro-average over all NON-abstention items
  - task_averaged_accuracy mean of the per-question-type means (non-abstention)
  - abstention_accuracy    accuracy over the `_abs` items only

BEM (Bulian et al. 2022, "Tomayto Tomahto", arXiv:2202.07654) is loaded lazily
from the HF port `kortukov/answer-equivalence-bem`; if it cannot be loaded the
scorer degrades gracefully to EM+containment (a warning is emitted once) so the
module is always usable. Set `use_bem=False` to skip it entirely.

Comparability caveat (bake into any reported number): this deterministic score
sits somewhat BELOW the official GPT-4o-judge number (the judge credits
paraphrase / superset answers that strict matching penalizes). Report it as
"deterministic (EM+containment+BEM)", not as the official metric, and calibrate
the offset with a one-time judge cross-check on a sample.
"""
from __future__ import annotations

import re
import string
import warnings
from collections import defaultdict
from typing import Optional

# --- canonical question-type buckets (accept both raw and reader-canonicalized) ---
_FACTUAL_TYPES = {
    "single-session-user", "single-session-assistant", "single-session",
    "multi-session", "temporal", "temporal-reasoning", "knowledge-update",
}
_PREFERENCE_TYPES = {"single-session-preference", "preference"}
_ABSTENTION_TYPES = {"abstention"}

# number words -> digits (both directions handled by normalizing words->digits)
_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100", "thousand": "1000",
}

_ARTICLES = {"a", "an", "the"}
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "at",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "that", "this",
    "would", "prefer", "user", "response", "responses", "like", "want", "wants",
    "should", "suggest", "suggestions", "recommend", "specifically", "tailored",
    "some", "more", "about", "their", "your", "you", "they", "them", "it", "its",
}

# refusal / abstention detector — the model correctly signals it cannot answer.
# Broad by design: abstention is the highest-risk category for silent
# under-scoring (a missed refusal reads as a wrong answer), so err toward recall.
_REFUSAL_PATTERNS = [
    r"\bi (don'?t|do not|cannot|can'?t|couldn'?t|am unable to|'?m unable to) "
    r"(know|tell|say|determine|find|answer|recall|remember|provide)",
    r"\b(don'?t|do not|didn'?t|does not|doesn'?t|couldn'?t|cannot|can'?t) have "
    r"(enough |any |sufficient |the |that |access to )?(information|info|details|data|record)",
    r"\b(no|not enough|insufficient|incomplete|lacking|without enough) "
    r"(information|info|details|data|context|record)",
    r"\benough (information|info|details|data|context) to (answer|determine|know|tell|say)",
    r"\bnot (mentioned|stated|specified|provided|available|given|recorded|discussed|clear|certain|sure)",
    r"\bnever (mentioned|stated|specified|discussed|told)",
    r"\b(isn'?t|is not|there'?s no|there is no|there isn'?t|no) "
    r"(any )?(information|mention|record|indication|detail|way to)",
    r"\b(unable|impossible|no way) to (determine|answer|find|tell|know)",
    r"\b(didn'?t|did not|haven'?t|have not|hasn'?t) (mention|specify|state|tell|discuss|provide|say)",
    r"\bcan'?t be (determined|answered|found)",
    r"\bno (record|mention|data|way to know|information)",
    r"\b(i'?m|i am) not (sure|certain)",
    r"\bcannot answer\b",
    r"\bdon'?t see (any|anything|a mention)",
    r"\bcouldn'?t find\b",
    r"\bwasn'?t (shared|mentioned|specified|provided)",
    r"\bnothing (about|in the|indicating|that (mentions|says))",
    r"\bno (mention|reference|indication) (of|to|that)",
    r"\b(isn'?t|is not|aren'?t|are not|wasn'?t|was not) "
    r"(available|mentioned|specified|recorded|provided|there|known|in the (chat|conversation|history))",
    r"\bunclear\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization + number-word canonicalization.

    lowercase -> strip punctuation -> drop articles -> map number words to digits
    -> collapse whitespace. Applied to BOTH gold and hypothesis before matching.
    """
    s = (s or "").lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    toks = [_NUM_WORDS.get(t, t) for t in s.split() if t not in _ARTICLES]
    return " ".join(toks)


def _content_tokens(s: str) -> list[str]:
    return [t for t in normalize_answer(s).split() if t not in _STOPWORDS]


# trailing "... (is) (also) acceptable/correct/fine" clause on a verbose gold (stripped to expose the answer).
# Leading \b so it can't bite mid-word (e.g. 'TikTok' -> 'TikT'). Only APPLIED to golds containing
# "acceptable" (see _gold_candidates), so the generic answer-words below can't clip a normal gold.
_ACCEPT_TAIL_RE = re.compile(
    r"[\.,;:]?\s*(?:is\s+)?(?:also\s+)?(?:an?\s+)?\b(?:acceptable|correct|fine|valid|okay|ok)\b\.?\s*$",
    re.IGNORECASE)


def _gold_candidates(gold: str) -> list[str]:
    """Split a gold answer into acceptable alternates.

    Handles parenthetical alternates — "25 minutes and 50 seconds (or 25:50)" -> ["25 minutes and 50
    seconds", "25:50", full]; " or " and SPACE-DELIMITED " / " alternations (a bare "1/48", a date "5/12",
    or a ratio must NOT be split — only "A / B" with surrounding spaces is an alternation); and the recurring
    verbose form "7 days. 8 days (including the last day) is also acceptable." -> exposes "7 days" and "8 days"
    (split on sentence boundaries + strip the acceptability tail), so a model answering either counts.
    """
    cands: list[str] = [gold]
    # parenthetical content (strip a leading "or"/"i.e."/"aka")
    for m in re.findall(r"\(([^)]*)\)", gold):
        inner = re.sub(r"^\s*(or|i\.?e\.?|aka|=)\s*", "", m.strip(), flags=re.IGNORECASE)
        if inner:
            cands.append(inner)
    # text with the parentheticals removed
    stripped = re.sub(r"\([^)]*\)", "", gold).strip()
    if stripped:
        cands.append(stripped)
    # explicit alternations: " or " and SPACE-padded " / " — but ONLY for SHORT candidates (real answer
    # alternations like "cat / dog", "Business Admin or Business Administration"). Long golds are sentences
    # whose " or " is a verb-phrase disjunction ("I have worked on OR bought five kits") — splitting those
    # yields generic prefixes ("I have worked on") that false-match wrong answers. Never a bare 1/48/date.
    is_accept = bool(re.search(r"\bacceptable\b", gold, re.IGNORECASE))
    parts: list[str] = []
    for c in list(cands):
        if len(c.split()) <= 6:
            parts.extend(re.split(r"\s+or\s+|\s+/\s+", c))
    # verbose "X. Y (...) is also acceptable" golds list several answers across sentence boundaries — split
    # on '. ' ONLY for those (guarded by the acceptability marker so 'Dr. Smith' / '3.5' stay intact).
    if is_accept:
        for c in list(cands) + list(parts):
            parts.extend(re.split(r"\.\s+", c))
    cands.extend(parts)
    # expose "8 days" from "8 days is also acceptable." — ONLY for acceptability golds, so the generic tail
    # words (fine/correct/valid/ok) can't clip a normal gold like "It was fine." -> "It was".
    if is_accept:
        cands = cands + [_ACCEPT_TAIL_RE.sub("", c) for c in cands]
    # de-dup, keep non-empty
    seen, out = set(), []
    for c in cands:
        c = c.strip()
        k = normalize_answer(c)
        if c and k and k not in seen:
            seen.add(k)
            out.append(c)
    return out


_BARE_NUM_RE = re.compile(r"^\d+$")   # normalize_answer strips '.', so list indices are always integers here
# an enumerated-list marker like "1. " / "2) " — detected on the RAW hypothesis (survives normalization).
_LIST_MARKER_RE = re.compile(r"(?:^|\s)\d{1,3}\s*[.)]\s")


def _is_bare_number(s: str) -> bool:
    """True if `s` normalizes to a single numeric token (incl. number-words: 'two' -> '2')."""
    n = normalize_answer(s)
    return bool(n) and _BARE_NUM_RE.match(n) is not None


def _is_list_like(hyp: str) -> bool:
    """True if the hypothesis enumerates items ('... 1. A, 2. B, 3. C') — where a bare-number gold can match
    an incidental list index rather than the actual answer."""
    return len(_LIST_MARKER_RE.findall(hyp or "")) >= 2


def _exact_match(gold: str, hyp: str) -> bool:
    return normalize_answer(gold) == normalize_answer(hyp)


def _containment(gold: str, hyp: str) -> bool:
    """True if the gold answer is contained in the hypothesis: either the
    normalized gold string is a substring of the normalized hypothesis, or every
    gold content-token appears in the hypothesis token set (order-free recall)."""
    g_norm = normalize_answer(gold)
    h_norm = normalize_answer(hyp)
    if not g_norm:
        return False
    # WORD/NUMBER-BOUNDARY substring (pad with spaces): a short gold must match whole tokens, not sit
    # inside a larger one — rejects '1'⊂'2001', 'yes'⊂'yesterday', '10'⊂'105', while still accepting
    # multi-token phrases like '25 minutes' ⊂ 'took 25 minutes total'.
    if f" {g_norm} " in f" {h_norm} ":
        return True
    g_toks = _content_tokens(gold)
    if not g_toks:
        # gold was all stopwords (e.g. "the suburbs") -> fall back to boundary substring only
        return f" {g_norm} " in f" {h_norm} "
    h_toks = set(h_norm.split())
    return all(t in h_toks for t in g_toks)


class _BEM:
    """Lazy singleton wrapper around the BEM answer-equivalence classifier.

    Predicts P(candidate is equivalent to reference | question). Deterministic
    (argmax / fixed threshold, no sampling). CPU-friendly single forward pass.
    """

    _instance: "Optional[_BEM]" = None
    _failed = False

    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        name = "kortukov/answer-equivalence-bem"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name)
        self.model.eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # keep BEM on CPU by default to avoid contending with a training run on
        # the GPU; it is a small BERT-base and fast enough on CPU for 500 items.
        self.dev = "cpu"
        _ = dev  # (cuda availability noted but intentionally unused)
        self.model.to(self.dev)

    @classmethod
    def get(cls) -> "Optional[_BEM]":
        if cls._failed:
            return None
        if cls._instance is None:
            try:
                cls._instance = cls()
            except Exception as e:  # noqa: BLE001 — degrade gracefully
                cls._failed = True
                warnings.warn(
                    f"[longmemeval_score] BEM unavailable ({type(e).__name__}: {e}); "
                    "falling back to EM+containment only. Install/allow "
                    "'kortukov/answer-equivalence-bem' for paraphrase scoring.",
                    RuntimeWarning,
                )
                return None
        return cls._instance

    def equivalent(self, question: str, reference: str, candidate: str,
                   threshold: float = 0.5) -> bool:
        # exact input format per the model card (kortukov/answer-equivalence-bem):
        # text=[CLS] candidate [SEP], text_pair=reference [SEP] question [SEP],
        # add_special_tokens=False. Label index 1 = "equivalent".
        torch = self.torch
        enc = self.tok(text=f"[CLS] {candidate} [SEP]",
                       text_pair=f"{reference} [SEP] {question} [SEP]",
                       add_special_tokens=False, truncation=True, max_length=512,
                       return_tensors="pt").to(self.dev)
        with torch.no_grad():
            logits = self.model(**enc).logits
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
        return prob >= threshold


class LongMemEvalScorer:
    """Deterministic LongMemEval scorer. Call `score(...)` per item, then
    `aggregate()`; or use the module-level `score_longmemeval(records)`.
    """

    def __init__(self, use_bem: bool = True, bem_threshold: float = 0.5,
                 preference_coverage: float = 0.5):
        self.use_bem = use_bem
        self.bem_threshold = bem_threshold
        self.preference_coverage = preference_coverage
        self._per_type: dict[str, list[bool]] = defaultdict(list)
        self._abstention: list[bool] = []
        self._details: list[dict] = []

    # ---- per-item ----
    def score_item(self, question: str, gold: str, hypothesis: str,
                   question_type: str, question_id: str = "") -> dict:
        qtype = (question_type or "").strip().lower()
        is_abs = qtype in _ABSTENTION_TYPES or str(question_id).endswith("_abs")

        if is_abs:
            # abstention = the model DECLINES. Require the refusal to DOMINATE: a concise, refusal-bearing
            # reply. A long substantive answer that merely contains a hedge ("...though I'm not sure") is NOT
            # an abstention (it committed to an answer) — guards the confirmed false-positive without hurting
            # normal refusals (which are brief). 40-token cap is comfortably above real refusal lengths.
            h = hypothesis or ""
            correct = bool(_REFUSAL_RE.search(h)) and len(normalize_answer(h).split()) <= 40
            method = "refusal"
            bucket = "abstention"
        elif qtype in _PREFERENCE_TYPES:
            correct, method = self._score_preference(gold, hypothesis)
            bucket = "single-session-preference"
        else:
            correct, method = self._score_factual(question, gold, hypothesis)
            bucket = qtype or "unknown"

        rec = {"question_id": question_id, "question_type": bucket,
               "correct": bool(correct), "method": method}
        self._details.append(rec)
        if is_abs:
            self._abstention.append(bool(correct))
        else:
            self._per_type[bucket].append(bool(correct))
        return rec

    def _score_factual(self, question: str, gold: str, hyp: str) -> tuple[bool, str]:
        hyp = hyp or ""
        for cand in _gold_candidates(gold):
            if _exact_match(cand, hyp):
                return True, "exact_match"
        for cand in _gold_candidates(gold):
            if _containment(cand, hyp):
                # A bare-number gold matched inside an ENUMERATED hypothesis is unreliable ONLY when the digit
                # appears SOLELY as a list index. Strip the enumeration markers and re-test: if the number
                # still matches (e.g. gold '3' in 'You led 3 projects: 1. A, 2. B, 3. C' — '3 projects'
                # survives) it's the real answer → accept; if it vanishes (gold '2', which was only '2. B')
                # → defer to exact/BEM. Fixes both the false-positive AND the enumerated-but-correct case.
                if _is_bare_number(cand) and _is_list_like(hyp) and not _containment(cand, _LIST_MARKER_RE.sub(" ", hyp)):
                    continue
                return True, "containment"
        if self.use_bem:
            bem = _BEM.get()
            if bem is not None:
                for cand in _gold_candidates(gold):
                    if bem.equivalent(question, cand, hyp, self.bem_threshold):
                        return True, "bem"
        return False, "none"

    def _score_preference(self, rubric: str, hyp: str) -> tuple[bool, str]:
        # salient rubric tokens: proper-noun-ish (originally capitalized) + content words
        proper = {normalize_answer(w) for w in re.findall(r"\b[A-Z][a-zA-Z0-9]+\b", rubric or "")}
        proper = {p for p in proper if p and p not in _STOPWORDS}
        content = set(_content_tokens(rubric))
        keys = (proper | content)
        if not keys:
            return False, "preference_no_keys"
        h_toks = set(normalize_answer(hyp).split())
        # proper nouns are load-bearing: if the rubric names specific entities,
        # require at least one to be present, then check overall coverage.
        if proper and not (proper & h_toks):
            return False, "preference_missing_entity"
        coverage = len(keys & h_toks) / len(keys)
        return (coverage >= self.preference_coverage,
                f"preference_coverage={coverage:.2f}")

    # ---- aggregate ----
    def aggregate(self) -> dict:
        per_type_acc = {t: (sum(v) / len(v) if v else 0.0, len(v))
                        for t, v in self._per_type.items()}
        all_nonabs = [c for v in self._per_type.values() for c in v]
        overall = sum(all_nonabs) / len(all_nonabs) if all_nonabs else 0.0
        task_avg = (sum(a for a, _ in per_type_acc.values()) / len(per_type_acc)
                    if per_type_acc else 0.0)
        abst = (sum(self._abstention) / len(self._abstention)
                if self._abstention else None)
        per_type = {t: {"accuracy": a, "n": n} for t, (a, n) in per_type_acc.items()}
        # per_subtask = uniform key the report tool reads across datasets (question types + abstention).
        per_subtask = dict(per_type)
        if self._abstention:
            per_subtask["abstention"] = {"accuracy": abst, "n": len(self._abstention)}
        return {
            "overall_accuracy": overall,
            "task_averaged_accuracy": task_avg,
            "abstention_accuracy": abst,
            "per_type": per_type,
            "per_subtask": per_subtask,
            "n_nonabstention": len(all_nonabs),
            "n_abstention": len(self._abstention),
            "bem_used": self.use_bem and _BEM.get() is not None,
        }


def score_longmemeval(records: list[dict], use_bem: bool = True) -> dict:
    """Score a list of prediction records.

    Each record: {question, answer (gold), hypothesis, question_type,
    question_id?}. Returns the aggregate dict from `LongMemEvalScorer`, with a
    `details` list of per-item verdicts attached.
    """
    scorer = LongMemEvalScorer(use_bem=use_bem)
    for r in records:
        scorer.score_item(
            question=r.get("question", ""),
            gold=r.get("answer", r.get("gold", "")),
            hypothesis=r.get("hypothesis", r.get("prediction", "")),
            question_type=r.get("question_type", ""),
            question_id=str(r.get("question_id", "")),
        )
    out = scorer.aggregate()
    out["details"] = scorer._details
    return out
