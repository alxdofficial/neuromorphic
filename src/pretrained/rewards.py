"""Reward functions for phase-2 GRPO on pretrained-LM memory.

All factories return a `Callable[[Tensor, Tensor], Tensor]` matching the
signature `grpo_step` expects:

    generated: [K, gen_length] int64 — sampled continuation tokens
    reference: [gen_length]    int64 — teacher-forcing reference (unused by
                                       string-based rewards; kept for
                                       signature parity with the built-in
                                       `token_match_reward`)
    returns:   [K] float32 in [0, 1], same device as `generated`.

String rewards run on CPU (decode + SequenceMatcher) and only touch the
generated tensor once per step — cheap compared to the rollout itself.

Designed for the passphrase-retrieval task (`passphrase_data.py`). The
`make_passphrase_reward` factory closes over the ground-truth phrase list
for one example; `grpo_step` is called once per prompt, so a fresh reward
function is built per example. Cost is negligible.

No LLM-as-judge, no reward model, no sBERT dependency in the hot path.
Surface-level ratio + order-correctness only. Replace with sentence-BERT
cosine in a later revision for Stage-1 paraphrase-tolerant scoring.
"""

from __future__ import annotations

import difflib
import re
from typing import Callable

import torch
from torch import Tensor


# --------------------------------------------------------------------------
# String-level scoring (tokenizer-free, testable in isolation)
# --------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def normalize(s: str) -> str:
    """Lowercase, strip, collapse whitespace. Used before every comparison
    so trailing spaces / case differences don't penalize the model."""
    return _WS_RE.sub(" ", s.lower()).strip()


def sequence_ratio(pred: str, gt: str) -> float:
    """`difflib.SequenceMatcher.ratio` on normalized strings.

    Returns in [0, 1]. Empty pred vs non-empty gt = 0.0. Identical = 1.0.
    Character-level similarity, so 'hello' vs 'helo' ~ 0.89, vs 'world' ~ 0.2.
    """
    np_pred = normalize(pred)
    np_gt = normalize(gt)
    if not np_gt:
        # Degenerate ground truth — define similarity against empty as 1 iff
        # pred is also empty, else 0. Matches user intuition.
        return 1.0 if not np_pred else 0.0
    return difflib.SequenceMatcher(None, np_pred, np_gt).ratio()


def _strip_trailing_punct(s: str) -> str:
    return s.rstrip(" .!?,;:")


def score_passphrase_ratio(pred_text: str, gt_phrases: list[str]) -> float:
    """Split pred on '.', compare each segment to the k-th gt phrase.

    Missing positions (pred has fewer '.' segments than N) score 0 and
    still divide by N. Extra segments are ignored — the model isn't
    rewarded for padding a short answer with junk, just not penalized
    beyond losing the missing positions.
    """
    N = len(gt_phrases)
    assert N >= 1
    segments = [_strip_trailing_punct(seg) for seg in pred_text.split(".")]
    segments = [s for s in segments if s.strip()]  # drop empties
    total = 0.0
    for i in range(N):
        pred_i = segments[i] if i < len(segments) else ""
        total += sequence_ratio(pred_i, gt_phrases[i])
    return total / N


def score_order_correctness(pred_text: str, gt_phrases: list[str]) -> float:
    """Fraction of gt phrases that appear as substrings of pred in the
    original relative order.

    Tolerates preamble / extra text around the phrases — useful when the
    model ignores the "no preamble" instruction but still got the phrases
    in the right order. A phrase 'matches' if its normalized form is a
    substring of the normalized pred text starting at or after the
    previous match's end position.
    """
    N = len(gt_phrases)
    assert N >= 1
    np_pred = normalize(pred_text)
    matched = 0
    cursor = 0
    for phrase in gt_phrases:
        np_phrase = normalize(phrase)
        if not np_phrase:
            continue
        pos = np_pred.find(np_phrase, cursor)
        if pos >= 0:
            matched += 1
            cursor = pos + len(np_phrase)
    return matched / N


def score_exact_match(pred_text: str, gt_text: str) -> float:
    """1.0 iff normalized(pred) == normalized(gt), else 0.0.

    Trailing punctuation / case / whitespace all normalized away first.
    Useful for synthetic tasks where the ground truth is unambiguous
    (passkey, single-shot classification)."""
    return 1.0 if normalize(pred_text) == normalize(gt_text) else 0.0


# --------------------------------------------------------------------------
# Factories — wrap a tokenizer + ground truth into a grpo_step reward_fn
# --------------------------------------------------------------------------


def _decode_rows(tokenizer, generated: Tensor) -> list[str]:
    """Decode [K, gen_length] to K strings, CPU-side, skipping specials.

    `skip_special_tokens=True` drops EOS/PAD so a short, EOS-terminated
    completion scores on its meaningful prefix (no phantom '</s>' hurting
    the SequenceMatcher ratio)."""
    ids = generated.detach().to("cpu").tolist()
    return [tokenizer.decode(row, skip_special_tokens=True) for row in ids]


def make_exact_match_reward(tokenizer, expected: str) -> Callable[[Tensor, Tensor], Tensor]:
    """reward_fn that returns 1.0 if decoded == expected (normalized).

    For synthetic passkey ("what is the passkey? ▁74921") where the
    answer is a short unambiguous string. Use `make_passphrase_reward`
    for multi-needle passphrase retrieval instead."""

    def reward_fn(generated: Tensor, reference: Tensor) -> Tensor:  # noqa: ARG001
        texts = _decode_rows(tokenizer, generated)
        scores = [score_exact_match(t, expected) for t in texts]
        return torch.tensor(scores, dtype=torch.float32, device=generated.device)

    return reward_fn


def make_passphrase_reward(
    tokenizer,
    expected_phrases: list[str],
    *,
    ratio_weight: float = 0.8,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Composite reward for ordered passphrase retrieval.

    reward = ratio_weight * passphrase_ratio + (1 - ratio_weight) * order_correctness

    `passphrase_ratio` segments pred on '.' and difflib-ratios each
    position against the gt phrase. Strict on surface form but tolerant
    of small typos. `order_correctness` gives partial credit when the
    phrases show up in order somewhere in the output (handles
    instruction-disobeying preambles).

    Default ratio_weight=0.8 makes surface form the dominant signal;
    order_correctness acts as a safety net so a mostly-correct answer
    with a preamble doesn't score zero. For stricter scoring, set
    ratio_weight=1.0.
    """
    assert 0.0 <= ratio_weight <= 1.0
    assert len(expected_phrases) >= 1

    def reward_fn(generated: Tensor, reference: Tensor) -> Tensor:  # noqa: ARG001
        texts = _decode_rows(tokenizer, generated)
        scores: list[float] = []
        for t in texts:
            r = score_passphrase_ratio(t, expected_phrases)
            o = score_order_correctness(t, expected_phrases)
            scores.append(ratio_weight * r + (1.0 - ratio_weight) * o)
        return torch.tensor(scores, dtype=torch.float32, device=generated.device)

    return reward_fn
