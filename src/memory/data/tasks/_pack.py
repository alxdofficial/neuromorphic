"""Shared streaming-episode packer — the common scheduler for the `qa` and `reconstruction` tasks.

Both tasks are "pack a bunch of FACTS into a context, then ask about particular ones" — they differ
only in how a fact is written and how it's queried. This factors that into a `Unit(write, query,
answer, ...)` abstraction + one packer driven by the `EpisodeSpec` knobs:
  - n_inputs      : how many facts the caller offers (fill-to-budget uses <= this)
  - n_queries     : how many facts we ask about
  - query_lag     : WHERE the queried fact sits — "early" (front, max retention lag) / "recent"
                    (back, recency) / "any" (random) / "vary" (sampled per-episode). window_size
                    ties this to the encoder chunks.
  - total_len     : the compression numerator (context is filled to ~this). Distractors fill the
                    budget implicitly (fill-to-budget), so `spec.n_distractors` is not used here.

CAUSALITY is guaranteed by construction: query facts are packed FIRST (so they always fit), the
question is asked at the very END (after the whole context), and every queried fact is therefore
written before it is asked. `filler_ok` lets the caller keep the answer un-guessable (reject a
distractor that contains the answer). Emits the exact per-sample dict `common.collate_qa` consumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class Unit:
    """One fact: `write` goes into the context; `query`→`answer` is what we ask (empty query = pure
    distractor, never asked). `answer_spans` = substrings of `answer` to score (empty → whole
    answer); `answer_exclude` = substrings NEVER scored (e.g. the entity name); `refs` = eval refs."""
    write: str
    query: str = ""
    answer: str = ""
    answer_spans: tuple = ()
    answer_exclude: tuple = ()
    refs: tuple = ()


def _answer_ids_content(tok, answer: str, spans, exclude, lead_space: bool):
    """Tokenize `answer`; content=True ONLY on `spans` char-ranges (minus `exclude`); fall back to
    the whole answer (minus exclude) when no span matches — so a spanless answer is never loss-less."""
    s = (" " + answer) if lead_space else answer
    enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)

    def _spans_of(needles):
        out = []
        for nd in {x for x in needles if x}:
            start = 0
            while True:
                i = s.find(nd, start)
                if i < 0:
                    break
                out.append((i, i + len(nd)))
                start = i + 1
        return out

    pos, exc = _spans_of(spans), _spans_of(exclude)

    def _ov(sp, a, b):
        return any(not (b <= x or a >= y) for x, y in sp)

    if pos:
        content = [_ov(pos, a, b) and not _ov(exc, a, b) for (a, b) in enc.offset_mapping]
        if any(content):
            return enc.input_ids, content
    content = [not _ov(exc, a, b) for (a, b) in enc.offset_mapping]
    return enc.input_ids, content


def pack_streaming_episode(query_units: List[Unit], filler_units: List[Unit], spec, tok,
                           pad_token_id: int, *, task_family: str, rng, filler_ok=None) -> Optional[dict]:
    """Pack `query_units` (the facts we ask about) + `filler_units` (distractors) into a
    total_len context per `spec`, then emit the queries + span-masked answers. Returns None (resample)
    if the query facts alone overflow the budget."""
    T = spec.total_len
    budget = T - 8                                          # margin for BPE join effects

    def ids(s):
        return tok(s, add_special_tokens=False).input_ids

    # Each write is tokenized ONCE and cached; the context (step 4) concatenates the cached ids instead
    # of re-tokenizing the joined string (writes are newline-terminated, so BPE rarely merges across the
    # join — negligible boundary effect, ~half the tokenization cost on big-context episodes).
    tok_cache = {}

    def wids(u):
        w = tok_cache.get(id(u))
        if w is None:
            w = tok_cache[id(u)] = ids(u.write)
        return w

    # 1. query facts must fit (they're always written) — else signal resample.
    used = sum(len(wids(u)) for u in query_units)
    if used > budget or not query_units:
        return None

    # 2. fill the remaining budget with distractors (rejecting any that would leak an answer).
    #    Break when essentially full OR after a run of SIZE-misses (big-context sources can't fill the
    #    tail — stop tokenizing candidates that won't fit rather than scanning the whole pool). An
    #    answer-leak rejection is NOT a fullness signal, so it does not count toward the break — else
    #    small-context sources with common answers (bAbI) stall early and underfill.
    chosen, misses = [], 0
    for u in filler_units:
        if used > budget - 24 or misses >= 12:
            break
        if filler_ok is not None and not filler_ok(u):
            continue                                       # answer-leak skip — orthogonal to fullness
        wl = len(wids(u))
        if used + wl > budget:
            misses += 1
            continue                                       # too big for the remaining budget (fullness signal)
        chosen.append(u)
        used += wl
        misses = 0

    # 3. order the writes by query_lag — WHERE the queried facts sit (the retention-lag axis).
    #    "vary" samples the lag per episode (retention-lag robustness across training).
    lag = spec.query_lag
    if lag == "vary":
        lag = rng.choice(("early", "recent", "any"))
    if lag == "recent":
        seq = chosen + list(query_units)                   # queried facts last = short lag
    elif lag == "any":
        seq = list(query_units) + chosen
        rng.shuffle(seq)
    else:                                                  # "early" / window index → queried facts first = long lag
        seq = list(query_units) + chosen

    # 4. materialize context from the CACHED per-write ids (no re-tokenization). Guaranteed <= budget
    #    < T, so no clamping of query facts → causality holds.
    ctx = [t for u in seq for t in wids(u)][:T]
    valid = len(ctx)
    ctx = ctx + [pad_token_id] * (T - valid)

    # 5. emit question(s) + answer(s) with span masks. Multi-query: inline " next_query" cue.
    question_ids = ids(query_units[0].query)
    answer_ids: List[int] = []
    content: List[bool] = []
    for n, u in enumerate(query_units):
        if n > 0:
            cue = ids(" " + u.query)
            answer_ids += cue
            content += [False] * len(cue)
        a_ids, a_content = _answer_ids_content(tok, u.answer, u.answer_spans, u.answer_exclude, True)
        answer_ids += a_ids
        content += a_content
    if not any(content):                                   # degenerate (no scorable answer token) → resample
        return None

    nq = len(query_units)
    refs = list(query_units[0].refs) or [query_units[0].answer]
    return {
        "context_ids": torch.tensor(ctx, dtype=torch.long),
        "context_mask": torch.tensor([True] * valid + [False] * (T - valid), dtype=torch.bool),
        "question_ids": torch.tensor(question_ids, dtype=torch.long),
        "answer_ids": torch.tensor(answer_ids, dtype=torch.long),
        "answer_content_mask_list": content,
        "task_family": task_family,
        "question_type": "single" if nq == 1 else f"multi{nq}",
        "answer_refs": refs,
    }
