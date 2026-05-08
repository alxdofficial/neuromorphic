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

Reward functions return a float in [0, 1] (1 = perfect, 0 = miss).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable


def exact_match_string(candidate: str, gold: str) -> float:
    """1 if candidate == gold (case-insensitive, stripped); else 0."""
    return float(candidate.strip().lower() == gold.strip().lower())


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


def humaneval_check(candidate_code: str, test_code: str, entry_point: str) -> float:
    """Run candidate code against the gold test; 1 if it passes, 0 otherwise.

    Disabled in v1 — returns 0.0. Running model-generated code requires a
    real sandbox (Docker container, firejail, or similar) plus timeout
    and resource limits. Wire one before enabling for training.
    """
    return 0.0


_REWARD_DISPATCH: dict[str, Callable] = {
    "exact_match": exact_match_string,
    "exact_match_or_bert_cosine": narrativeqa_match,
    "rule_based_exec": humaneval_check,
}


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
            return exact_match_string(candidate, meta["gold_boxed"])
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
    raise ValueError(f"unknown reward_kind: {reward_kind}")
