"""Overwrite task — reassign the SAME key, query the LATEST value (forced-forgetting).

Dedicated task for the ``ruler_overwrite`` source (the survey's forced-forgetting probe, a fork of
RULER ``variable_tracking``): pack a variable-assignment sequence in which the queried KEY is bound
to ``v1`` early, then REASSIGNED ``v2`` later — with the stale ``v1`` and other variables'
assignments sitting in between as interference — and query the KEY expecting ONLY the latest value
``v2``. A correct memory must OVERWRITE the stale binding, not average it in or retrieve the earlier
one. EM-scorable; loss on the ``v2`` value span only.

Layout (matches the ``k = v`` line form the reconstruction task uses, so the decoder predicts the
value in-distribution)::

    KEY = v1
    other_a = ...
    other_b = ...          # distractor assignments (other variables)
    ...
    KEY = v2               # the OVERWRITE — the most-recent binding, the answer

Consumes a ``keyed``-kind source (distinct random variables). Emits the exact per-sample dict
``common.collate_qa`` consumes, so ``compute_loss`` + the REAL/SHUF/OFF gate are unchanged.
See docs/data_arch_plan.md (Layer L2).
"""
from __future__ import annotations

from typing import List

import torch

from .base import Task
from ..schedule import EpisodeSpec


class OverwriteTask(Task):
    accepts = ("keyed",)

    def build(self, source, spec: EpisodeSpec, tok, rng, pad_token_id: int) -> dict | None:
        ctx_len = spec.total_len

        def ids(s: str) -> List[int]:
            return tok(s, add_special_tokens=False).input_ids

        # n_inputs assignment lines total; two of them belong to the TARGET (v1 stale, v2 latest),
        # the rest are distractor assignments to OTHER variables. +2 so the pool sources the
        # target's second (latest) value from a different random variable.
        n_lines = max(3, spec.n_inputs)
        pool = source.sample(rng, n_lines + 2)

        key = pool[0].key_text
        v1 = pool[0].value_text                       # stale binding (written first)
        v2 = pool[1].value_text                       # latest binding (written last) — the answer
        # Guarantee v1 != v2 (random collision is astronomically unlikely, but be exact).
        j = 1
        while v2 == v1 and j + 1 < len(pool):
            j += 1
            v2 = pool[j].value_text
        if v2 == v1:
            return None                               # degenerate draw → resample

        distractors = [it for it in pool[2:] if it.key_text != key]

        # Sequence: TARGET=v1  →  distractor assignments (other vars)  →  TARGET=v2 (most recent).
        def line_ids(k: str, v: str) -> List[int]:
            return ids(f"{k} = {v}\n")

        head = line_ids(key, v1)
        tail = line_ids(key, v2)
        if len(head) + len(tail) > ctx_len:           # values too long for even the two target lines
            return None

        cap = spec.n_distractors or len(distractors)  # 0 ⇒ fill to budget; N ⇒ at most N distractors
        body: List[int] = []
        cum = len(head) + len(tail)
        n_distr = 0
        for it in distractors:
            if n_distr >= cap:
                break
            li = line_ids(it.key_text, it.value_text)
            if cum + len(li) > ctx_len - 8:           # keep both target lines inside the budget
                break
            body += li
            cum += len(li)
            n_distr += 1

        ctx_ids = head + body + tail                  # v2 line is last → unambiguously the latest
        valid = len(ctx_ids)
        if valid < ctx_len:
            ctx_ids = ctx_ids + [pad_token_id] * (ctx_len - valid)

        # Query with the "key =" form the context uses so the decoder predicts the value
        # in-distribution; space-prefix the answer to match "= v2".
        question_ids = ids(f"{key} =")
        answer_ids = ids(" " + v2)
        content = [True] * len(answer_ids)            # the whole latest value is load-bearing

        return {
            "context_ids": torch.tensor(ctx_ids, dtype=torch.long),
            "context_mask": torch.tensor([True] * valid + [False] * (ctx_len - valid),
                                         dtype=torch.bool),
            "question_ids": torch.tensor(question_ids, dtype=torch.long),
            "answer_ids": torch.tensor(answer_ids, dtype=torch.long),
            "answer_content_mask_list": content,
            "task_family": "overwrite",
            "question_type": "overwrite",
            "answer_refs": [v2],
        }
