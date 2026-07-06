"""Small training-harness utilities: LR schedule, batch→device, val-set materialization.

Extracted verbatim from ``scripts/train/train.py`` (harness reorg phase 2). No logic changes.
"""
from __future__ import annotations

import math

import torch


def lr_at_step(step: int, max_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (0.1 + 0.9 * cos)


def to_device(batch, device):
    for f in ("context_ids", "context_mask", "question_ids", "question_mask",
              "answer_ids", "answer_mask", "answer_content_mask"):
        setattr(batch, f, getattr(batch, f).to(device, non_blocking=True))
    return batch


@torch.no_grad()
def materialize_val_set(val_dl, n_batches: int) -> list:
    """Drain the streaming val_dl ONCE into a fixed list of batches.

    Fixes the val sampling variance bug (#614). Previously each `run_val` call
    advanced the val DataLoader iterator → different random batches per eval →
    0.4-0.7 nat noise between streaming-best and final-eval on the SAME model.
    Materializing the set once means every eval is on the same data → numbers
    become directly comparable and the streaming-best vs final-eval gap closes
    to ~zero (modulo checkpoint differences).

    Memory cost: ~2 MB for 30 batches × 2 examples × 4096 tokens × int64. Trivial.
    """
    fixed = []
    for i, batch in enumerate(val_dl):
        if i >= n_batches:
            break
        fixed.append(batch)
    return fixed
