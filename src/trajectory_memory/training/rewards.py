"""Reward functions for Wave 3 / Wave 4 GRPO.

Per project policy (no LLM-as-judge): only exact-match + BERT cosine +
rule-based rewards.

- `exact_match_gsm8k`     — extract final number, compare.
- `exact_match_string`    — direct string equality.
- `bert_cosine`           — BERT-base sentence embedding cosine similarity.
- `narrativeqa_match`     — exact match against any of the gold answers,
                             fallback to BERT cosine.
- `humaneval_check`       — DISABLED in v1 (returns 0.0). Running model
                             code needs sandboxing — wire a Docker / firejail
                             runner before enabling.
- `f1_qa`                  — Multi-hop QA SQuAD-style F1 over tokens.
                             Used by MuSiQue, HotpotQA, 2WikiMultiHopQA.
- `mc_letter`              — Multiple-choice letter match (A/B/C/D).
                             Used by QuALITY.

Reward functions return a float in [0, 1] (1 = perfect, 0 = miss).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from functools import lru_cache


def exact_match_string(candidate: str, gold: str) -> float:
    """1 if candidate == gold (case-insensitive, stripped); else 0."""
    return float(candidate.strip().lower() == gold.strip().lower())


def extract_last_boxed(text: str) -> str | None:
    """Extract content of the LAST `\\boxed{...}` from `text`, handling
    nested braces. Returns None if no `\\boxed` found.

    B3 fix: prior implementation used `re.findall(r'\\\\boxed\\{([^}]+)\\}',...)`
    which broke on nested braces (e.g. `\\boxed{\\frac{1}{2}}` returned
    `\\frac{1` instead of `\\frac{1}{2}`). NumniaMath answers commonly
    include `\\frac{}{}`, `\\sqrt{}`, etc.
    """
    target = "\\boxed{"
    # Find LAST occurrence
    idx = text.rfind(target)
    if idx < 0:
        return None
    start = idx + len(target)
    depth = 1
    i = start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None  # unbalanced


def exact_match_gsm8k(candidate: str, gold_number: str) -> float:
    """Extract the final number from `candidate` and compare to gold_number.

    GSM8K answers typically end with a number; we match the last numeric
    token in the response.
    """
    if gold_number is None:
        return 0.0
    matches = re.findall(r"-?\d[\d,]*\.?\d*", candidate)
    if not matches:
        return 0.0
    final = matches[-1].replace(",", "")
    try:
        return float(float(final) == float(gold_number))
    except ValueError:
        return 0.0


@lru_cache(maxsize=1)
def _bert_model():
    """Lazy-load a small BERT for sentence-embedding cosine."""
    from transformers import AutoModel, AutoTokenizer
    name = "sentence-transformers/all-MiniLM-L6-v2"   # ~22M params
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval()
    return tok, model


def bert_cosine(candidate: str, gold: str) -> float:
    """Sentence-embedding cosine similarity in [0, 1] (clamped from [-1,1])."""
    import torch
    tok, model = _bert_model()
    enc = tok([candidate, gold], padding=True, truncation=True,
              max_length=256, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    a, b = emb[0], emb[1]
    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return max(0.0, (cos + 1.0) / 2.0)


def narrativeqa_match(candidate: str, gold_answers: list[str]) -> float:
    """Exact-match-any against a list of gold answers; fallback BERT cosine."""
    for g in gold_answers:
        if exact_match_string(candidate, g) == 1.0:
            return 1.0
    return max(bert_cosine(candidate, g) for g in gold_answers)


def _normalize_qa_answer(text: str) -> str:
    """SQuAD/MuSiQue/HotpotQA normalization: lowercase, strip articles,
    strip punctuation, collapse whitespace. Standard recipe — matches what
    the original SQuAD eval script does, and what MuSiQue/HotpotQA evals
    use too."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def f1_qa(candidate: str, gold: str) -> float:
    """Token-overlap F1 between candidate and gold, post-normalization.
    Standard SQuAD/MuSiQue/HotpotQA eval metric.

    Returns float in [0, 1]. Empty answers return 0 unless both are empty
    (then 1 — degenerate but matches SQuAD convention)."""
    cand_norm = _normalize_qa_answer(candidate)
    gold_norm = _normalize_qa_answer(gold)
    cand_toks = cand_norm.split()
    gold_toks = gold_norm.split()
    if not cand_toks or not gold_toks:
        # Both empty → 1; one empty → 0.
        return float(not cand_toks and not gold_toks)
    common = Counter(cand_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(cand_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def f1_qa_max(candidate: str, golds: list[str]) -> float:
    """F1 vs the best of a list of gold answers (handles answer_aliases
    in MuSiQue / multiple-reference style)."""
    if not golds:
        return 0.0
    return max(f1_qa(candidate, g) for g in golds)


# ── Multiple-choice (QuALITY) ─────────────────────────────────────────


_MC_LETTER_RE = re.compile(r"^\s*\(?([A-Da-d])[\)\.\s]")


def mc_letter(candidate: str, gold_letter: str) -> float:
    """Parse the FIRST letter from the candidate (A/B/C/D) and compare to
    gold_letter. Tolerant to common formats: 'A', 'A.', 'A)', '(A)',
    '  A  ', 'Answer: A', etc.

    Strategy: find the first occurrence of 'A'-'D' that's followed by a
    delimiter (`.`, `)`, whitespace, or end of string), or that appears
    after the word 'answer'. Matches what eval harnesses (lm-eval-harness,
    LongBench's MC scorer) do for MC tasks."""
    if not candidate:
        return 0.0
    gold = gold_letter.strip().upper()
    if gold not in ("A", "B", "C", "D"):
        return 0.0

    # First pass: look for a delimiter pattern at the start.
    m = _MC_LETTER_RE.match(candidate)
    if m:
        return float(m.group(1).upper() == gold)

    # Second pass: "answer is A" / "Answer: B" style.
    m = re.search(
        r"(?:answer|choice)(?:\s+is)?\s*[:\-]?\s*\(?([A-Da-d])\b",
        candidate, flags=re.IGNORECASE,
    )
    if m:
        return float(m.group(1).upper() == gold)

    # Last resort: any standalone letter A-D in the first 100 chars.
    head = candidate[:100]
    m = re.search(r"\b([A-Da-d])\b", head)
    if m:
        return float(m.group(1).upper() == gold)
    return 0.0


def humaneval_check(candidate_code: str, test_code: str, entry_point: str) -> float:
    """Run candidate code against the gold test; 1 if it passes, 0 otherwise.

    Disabled in v1 — returns 0.0. Running model-generated code requires a
    real sandbox (Docker container, firejail, or similar) plus timeout
    and resource limits. Wire one before enabling for training.
    """
    return 0.0


def compute_reward(
    reward_kind: str,
    candidate: str,
    *,
    gold: str | None = None,
    meta: dict | None = None,
) -> float:
    """Dispatch to the right reward function based on `reward_kind`."""
    if reward_kind == "exact_match":
        if meta and meta.get("gold_number") is not None:
            return exact_match_gsm8k(candidate, meta["gold_number"])
        if meta and meta.get("gold_boxed") is not None:
            # NuminaMath candidates produce chain-of-thought ending with
            # `\boxed{ANSWER}`. B3 fix: use brace-balanced extractor for
            # nested-brace answers like `\boxed{\frac{1}{2}}`.
            box = extract_last_boxed(candidate)
            if box is None:
                return 0.0
            return exact_match_string(box, meta["gold_boxed"])
        return exact_match_string(candidate, gold or "")
    elif reward_kind == "exact_match_or_bert_cosine":
        all_answers = (meta or {}).get("all_answers", [gold] if gold else [])
        return narrativeqa_match(candidate, all_answers)
    elif reward_kind == "rule_based_exec":
        return humaneval_check(
            candidate,
            (meta or {}).get("test", ""),
            (meta or {}).get("entry_point", ""),
        )
    elif reward_kind == "f1_qa":
        # Multi-hop QA F1. Golds is a list (answer + aliases). Used by
        # MuSiQue (answer_aliases), HotpotQA (single answer), and
        # 2WikiMultiHopQA (single answer + supporting_facts).
        golds = (meta or {}).get("all_answers", [gold] if gold else [])
        return f1_qa_max(candidate, golds)
    elif reward_kind == "mc_letter":
        # Multiple-choice (QuALITY). gold_letter = 'A'/'B'/'C'/'D'.
        return mc_letter(candidate, (meta or {}).get("gold_letter", ""))
    raise ValueError(f"unknown reward_kind: {reward_kind}")
