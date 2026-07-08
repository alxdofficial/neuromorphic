"""QA task — story → question → answer, distractor-padded to the budget (bAbI-style).

Task half of the old ``data/babi.py``: pack the gold story's fact sentences at the front
(guaranteed intact), then pad with entity-disjoint distractor sentences up to ``spec.total_len``,
turning every example into a retrieve-among-noise read. Loss on every answer token.

Consumes a ``qa``-kind source (``.sample`` gold QAItems + ``.distractor_pool()`` noise).
"""
from __future__ import annotations

from typing import List

import torch

from .base import Task
from ..schedule import EpisodeSpec
from ..sources.babi import _caps_names


class QATask(Task):
    accepts = ("qa",)

    def build(self, source, spec: EpisodeSpec, tok, rng, pad_token_id: int) -> dict | None:
        ctx_len = spec.total_len
        item = source.sample(rng, 1)[0]
        pool = source.distractor_pool()

        def ids(s: str) -> List[int]:
            return tok(s, add_special_tokens=False).input_ids

        # Real supporting facts first (front, intact), one fact per line.
        ctx_ids: List[int] = []
        for sent in item.facts:
            ctx_ids += ids(sent + "\n")
        # Over-long real story: keep the TAIL (the answer-relevant fact is usually the most recent).
        if len(ctx_ids) > ctx_len:
            ctx_ids = ctx_ids[-ctx_len:]

        # Distractor padding to ~ctx_len; reject distractors sharing a gold entity (would contaminate
        # the label). n_distractors>0 caps the count; 0 (default) fills the budget as bAbI always did.
        gold_names = set()
        for f in item.facts:
            gold_names |= _caps_names(f)
        added, cap = 0, (spec.n_distractors or None)
        guard = 0
        while pool and len(ctx_ids) < ctx_len and guard < 8 * ctx_len:   # empty pool → no padding (pad-tokens fill below)
            guard += 1
            if cap is not None and added >= cap:
                break
            d = rng.choice(pool)
            if _caps_names(d) & gold_names:
                continue
            d_ids = ids(d + "\n")
            if len(ctx_ids) + len(d_ids) > ctx_len:
                room = ctx_len - len(ctx_ids)
                if room > 0:
                    ctx_ids += d_ids[:room]
                break
            ctx_ids += d_ids
            added += 1

        valid = len(ctx_ids)
        if valid < ctx_len:
            ctx_ids = ctx_ids + [pad_token_id] * (ctx_len - valid)

        question_ids = ids(item.question)
        answer_ids = ids(item.answer)               # list tasks carry comma-joined answers; all content
        content = [True] * len(answer_ids)

        return {
            "context_ids": torch.tensor(ctx_ids, dtype=torch.long),
            "context_mask": torch.tensor([True] * valid + [False] * (ctx_len - valid), dtype=torch.bool),
            "question_ids": torch.tensor(question_ids, dtype=torch.long),
            "answer_ids": torch.tensor(answer_ids, dtype=torch.long),
            "answer_content_mask_list": content,
            "task_family": "babi",
            "question_type": f"task{item.task_id}",
            "answer_refs": [item.answer],
        }
