"""Continuation task — compress a prefix span into memory, predict the next span (gist objective).

Task half of the old ``data/continuation.py``: the AutoCompressors / Activation-Beacon / CCM family
objective (gist, not binding). Draw one corpus document, pick a random start offset, compress the
first ``spec.total_len`` tokens into memory (then throw those tokens away), and predict the next
``spec.predict_len`` tokens with the decoder seeing ONLY the memory (closed-book continuation). CE
on the continuation tokens. The predicted window sits immediately after the compressed cutoff, so
the early continuation tokens genuinely depend on the compressed context.

Consumes a ``corpus``-kind source (``.sample`` → CorpusItem token arrays). Emits the same per-sample
dict the old reader did, so ``collate_qa`` + the whole ``compute_loss`` / REAL/SHUF/OFF gate are
unchanged.
"""
from __future__ import annotations

import torch

from .base import Task
from ..schedule import EpisodeSpec


class ContinuationTask(Task):
    accepts = ("corpus",)

    def build(self, source, spec: EpisodeSpec, tok, rng, pad_token_id: int) -> dict | None:
        compress_len = spec.total_len
        predict_len = spec.predict_len
        span_need = compress_len + predict_len

        d = source.sample(rng, 1)[0].tokens
        if d.shape[0] < span_need:                      # unlucky short doc → resample
            return None
        s = rng.randrange(0, len(d) - span_need + 1)
        compress = d[s: s + compress_len]
        predict = d[s + compress_len: s + compress_len + predict_len]

        # continuation cue (the chat-template user turn); the value to predict is raw document text.
        # NOTE: intermediate-horizon targets are sliced from context_ids by the loss (the block right
        # after each window boundary), so no extra tokens are emitted here — answer_ids is the
        # final-horizon (beyond-span) block. Multi-horizon needs predict_len <= window_size to tile.
        trigger_ids = tok("Continue the passage.", add_special_tokens=False).input_ids
        out = {
            "context_ids": torch.tensor(compress, dtype=torch.long),
            "context_mask": torch.ones(compress_len, dtype=torch.bool),   # full span, no pad
            "question_ids": torch.tensor(trigger_ids, dtype=torch.long),
            "answer_ids": torch.tensor(predict, dtype=torch.long),
            "answer_content_mask_list": [True] * predict_len,             # predict every token
            "task_family": "continuation",
            "question_type": "continuation",
            "answer_refs": [],
        }
        if spec.n_horizons is not None:                                   # cap # of scored boundaries
            out["n_horizons"] = int(spec.n_horizons)
        return out
