"""Shared QA-reader core: the `(context_ids, question_ids, answer_ids, ...)`
contract used by every reader in `src.memory.data`.

Holds the base pieces that all per-dataset readers import:
- `QABatch` — the batched-example dataclass every reader/collate produces.
- `_pack_context` — concat/truncate/pad passages into a fixed-width context.
- `collate_qa` — stack per-sample dicts into a `QABatch`.
- `REPO_ROOT` — repo root (kept on `sys.path`; some readers resolve data paths from it).

The per-dataset readers (hotpot, narrativeqa, musique, babilong, ruler,
locomo, mixed) live in sibling modules and import from here. The Source × Task
training pipeline (`src.memory.data.sources` / `tasks`) also consumes `collate_qa`.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class QABatch:
    """A batched v1h example for QA training."""
    context_ids: Tensor          # [B, T_ctx] long — packed passages
    context_mask: Tensor         # [B, T_ctx] bool — True at valid context positions
    question_ids: Tensor         # [B, T_q] long — right-padded
    question_mask: Tensor        # [B, T_q] bool
    answer_ids: Tensor           # [B, T_a] long — right-padded
    answer_mask: Tensor          # [B, T_a] bool — True at valid (non-pad) answer positions
    answer_content_mask: Tensor  # [B, T_a] bool — True at load-bearing content positions
    task_family: list[str]       # length B — per-example task family for telemetry
    question_type: list[str]     # length B — per-example question type for telemetry
    answer_refs: list = None     # length B — per-example reference answer strings (for generated
                                 # EM/F1 eval on abstractive QA); optional, dropped by loss/CE paths


def _pack_context(
    passage_token_ids_list: list[list[int]],
    chunk_size: int,
    sep_token_id: int,
    pad_token_id: int,
) -> tuple[list[int], int]:
    """Concat passages separated by sep_token_id. Truncate or pad to chunk_size.
    Returns (token_ids, valid_length)."""
    out: list[int] = []
    for i, p in enumerate(passage_token_ids_list):
        if i > 0:
            out.append(sep_token_id)
        out.extend(p)
        if len(out) >= chunk_size:
            out = out[:chunk_size]
            return out, chunk_size
    valid = len(out)
    if valid < chunk_size:
        out = out + [pad_token_id] * (chunk_size - valid)
    return out, valid


def collate_qa(samples: list[dict], pad_token_id: int = 128_001) -> QABatch:
    """Stack per-sample dicts into a QABatch, padding question and answer
    to the max length in the batch."""
    B = len(samples)
    T_q = max(int(s["question_ids"].shape[0]) for s in samples)
    T_a = max(int(s["answer_ids"].shape[0]) for s in samples)

    context_ids = torch.stack([s["context_ids"] for s in samples])
    context_mask = torch.stack([s["context_mask"] for s in samples])

    question_ids = torch.full((B, T_q), pad_token_id, dtype=torch.long)
    question_mask = torch.zeros((B, T_q), dtype=torch.bool)
    answer_ids = torch.full((B, T_a), pad_token_id, dtype=torch.long)
    answer_mask = torch.zeros((B, T_a), dtype=torch.bool)
    answer_content_mask = torch.zeros((B, T_a), dtype=torch.bool)

    for i, s in enumerate(samples):
        tq = int(s["question_ids"].shape[0])
        ta = int(s["answer_ids"].shape[0])
        question_ids[i, :tq] = s["question_ids"]
        question_mask[i, :tq] = True
        answer_ids[i, :ta] = s["answer_ids"]
        answer_mask[i, :ta] = True
        cm = s["answer_content_mask_list"]
        assert len(cm) == ta, (                              # fail loud on a malformed sample
            f"answer_content_mask_list length {len(cm)} != answer_ids length {ta} "
            f"(task={s.get('task_family')!r}); a length-mismatched mask silently mis-scores loss.")
        for j, v in enumerate(cm):
            if j < T_a and v:
                answer_content_mask[i, j] = True

    return QABatch(
        context_ids=context_ids,
        context_mask=context_mask,
        question_ids=question_ids,
        question_mask=question_mask,
        answer_ids=answer_ids,
        answer_mask=answer_mask,
        answer_content_mask=answer_content_mask,
        task_family=[s["task_family"] for s in samples],
        question_type=[s["question_type"] for s in samples],
        answer_refs=[s.get("answer_refs", []) for s in samples],
    )
