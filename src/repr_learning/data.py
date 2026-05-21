"""Data pipeline: paragraph windowing + span masking.

For v0, fixed-size 256-token windows. Span masking randomly picks
contiguous mask spans of 5-15 tokens, target mask ratio sampled
uniformly from [0.5, 0.9] per batch.

Two data sources supported:
- HuggingFace dataset name (e.g., "HuggingFaceFW/fineweb-edu")
- Local parquet / jsonl files

For v0 scaffold, we provide a synthetic dataset for the forward-pass
smoke test. Real data loading is added in scripts/repr_learning/train.py.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Iterator, Optional

import torch
from torch import Tensor

from .config import ReprConfig


@dataclass
class WindowBatch:
    """A batch of fixed-size text windows ready for training."""

    input_ids: Tensor          # [B, T] long — Llama token ids
    attention_mask: Tensor     # [B, T] bool — True where real (not padding)
    mask_positions: Tensor     # [B, T] bool — True at positions to mask + predict


def sample_span_mask(
    seq_len: int,
    target_ratio: float,
    span_len_range: tuple[int, int],
    rng: random.Random,
) -> list[int]:
    """Sample contiguous span masks until target mask ratio is reached.

    Returns a list of masked position indices.
    """
    target_n = int(seq_len * target_ratio)
    masked: set[int] = set()
    span_min, span_max = span_len_range
    attempts = 0

    while len(masked) < target_n and attempts < seq_len * 4:
        span_len = rng.randint(span_min, span_max)
        start = rng.randint(0, max(seq_len - span_len, 0))
        # Cap span to remaining budget so we don't overshoot target_ratio.
        # Iterate, count new additions only (existing ones are no-ops).
        for i in range(start, min(start + span_len, seq_len)):
            if i not in masked:
                masked.add(i)
                if len(masked) >= target_n:
                    break
        attempts += 1

    return sorted(masked)


def make_batch_mask_positions(
    batch_size: int,
    seq_len: int,
    cfg: ReprConfig,
    seed: Optional[int] = None,
) -> Tensor:
    """Generate per-sample mask positions for a whole batch.

    Returns: [B, T] bool tensor — True at masked positions.
    """
    rng = random.Random(seed)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    for b in range(batch_size):
        ratio = rng.uniform(cfg.mask_ratio_min, cfg.mask_ratio_max)
        positions = sample_span_mask(
            seq_len, ratio, (cfg.mask_span_min, cfg.mask_span_max), rng,
        )
        for p in positions:
            mask[b, p] = True
    return mask


def synthetic_batch(
    cfg: ReprConfig,
    batch_size: Optional[int] = None,
    seed: int = 0,
) -> WindowBatch:
    """Generate a synthetic batch for smoke-testing the forward pass.

    Tokens are uniformly random from Llama's vocabulary. Lengths fixed at
    cfg.fixed_window_size. Used for verifying the model runs without
    needing real text data.
    """
    bs = batch_size if batch_size is not None else cfg.batch_size
    T = cfg.fixed_window_size

    torch.manual_seed(seed)
    # Skip special tokens (0..255) and stick to ASCII-like range.
    input_ids = torch.randint(
        low=256, high=cfg.llama_vocab_size, size=(bs, T), dtype=torch.long,
    )
    attention_mask = torch.ones(bs, T, dtype=torch.bool)
    mask_positions = make_batch_mask_positions(bs, T, cfg, seed=seed)
    return WindowBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        mask_positions=mask_positions,
    )
