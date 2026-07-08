"""MAE task — contiguous-passage masked reconstruction (storage / compression objective).

Task half of the old ``data/mae.py``'s ``LongPassageMAEDataset._gen``: draw one corpus document
(≥ ``spec.total_len`` tokens), a random start offset, and emit the contiguous ``spec.total_len``
span as ``context_ids`` with a full (all-True) mask; the answer IS the context (teacher-forced
reconstruction). The masked_reconstruction loss path masks positions internally — we add a trigger
but no real question. Loss on every context token.

Consumes a ``corpus``-kind source (``.sample`` → CorpusItem token arrays). Emits the same per-sample
dict the old reader did, PLUS ``k_slots`` (the fixed memory budget M — the capacity-relative slice
``memory[:, :k]`` keeps all M emitted tokens) and ``n_tokens`` (the span length), which the MAE loss
path reads.

k_slots collation: the old ``make_long_passage_mae_dataloader`` used a special ``_collate_long_passage``
that STAMPS ``batch.k_slots`` / ``batch.n_tokens`` onto the QABatch (plain ``collate_qa`` does not).
In the Source/Task architecture that batch-level stamping is the mix builder's job (phase 4) — the
mixed-path collate reads the ``k_slots`` / ``n_tokens`` keys this task emits. We still emit them here
so nothing downstream loses the budget. ``m_slots`` is the memory budget M: it defaults to 32 (the
old reader default) so the registry can build the task arg-free; the mix layer overrides it either by
constructing ``MaeTask(m_slots=M)`` or setting ``task.m_slots = M`` after construction.
"""
from __future__ import annotations

import torch

from .base import Task
from ..schedule import EpisodeSpec


class MaeTask(Task):
    accepts = ("corpus",)

    def __init__(self, m_slots: int = 32):
        self.m_slots = m_slots

    def build(self, source, spec: EpisodeSpec, tok, rng, pad_token_id: int) -> dict | None:
        ctx_len = spec.total_len

        d = source.sample(rng, 1)[0].tokens
        if d.shape[0] < ctx_len:                        # unlucky short doc → resample
            return None
        s = rng.randrange(0, len(d) - ctx_len + 1)
        span = torch.tensor(d[s: s + ctx_len], dtype=torch.long)

        trigger_ids = tok("Reconstruct the text above.", add_special_tokens=False).input_ids
        out = {
            "context_ids": span,
            "context_mask": torch.ones(ctx_len, dtype=torch.bool),        # full span, no pad
            "question_ids": torch.tensor(trigger_ids, dtype=torch.long),
            "answer_ids": span.clone(),
            "answer_content_mask_list": [True] * ctx_len,
            "task_family": "masked_reconstruction", "question_type": "masked_reconstruction",
            "answer_refs": [], "k_slots": self.m_slots, "n_tokens": ctx_len,
        }
        if spec.mask_ratio is not None:                                   # curriculum mask-% override
            out["mask_ratio"] = float(spec.mask_ratio)
        return out
