"""
FITB (Fill-In-The-Blank) masking utilities for neuromorphic LM training.

Generates boolean masks indicating which token positions to replace with <FITB>.
Supports random masking, span masking (geometric distribution), and mixed mode.

All mask generation happens on CPU to avoid GPU sync from .item() calls,
then the result is transferred to the target device once.
"""

import random

import torch
from torch import Tensor


def generate_random_mask(BS: int, N: int, mask_rate: float,
                         device: torch.device) -> Tensor:
    """Uniform random masking.

    Returns: [BS, N] bool on ``device`` — True at positions to mask.
    """
    # Random mask is a single vectorized op, safe on any device
    return torch.rand(BS, N, device=device) < mask_rate


def generate_span_mask(BS: int, N: int, mask_rate: float,
                       mean_span_len: int, device: torch.device) -> Tensor:
    """Span masking with geometric span lengths.

    Generates spans whose lengths follow Geometric(p=1/mean_span_len).
    Spans are placed until the target mask_rate is approximately reached.

    Mask is built on CPU (no GPU sync), then transferred to ``device`` once.

    Returns: [BS, N] bool on ``device`` — True at positions to mask.
    """
    target_count = int(mask_rate * N)
    if target_count == 0:
        return torch.zeros(BS, N, dtype=torch.bool, device=device)

    p = 1.0 / max(mean_span_len, 1)
    mask = torch.zeros(BS, N, dtype=torch.bool)  # CPU

    for b in range(BS):
        masked = 0
        attempts = 0
        while masked < target_count and attempts < N:
            start = random.randint(0, N - 1)
            # Geometric length (at least 1): sample via inverse-CDF
            length = int(random.expovariate(p)) + 1
            end = min(start + length, N)
            mask[b, start:end] = True
            masked = int(mask[b].sum())
            attempts += 1

    return mask.to(device, non_blocking=True)


def generate_fitb_mask(BS: int, N: int, mask_rate: float,
                       span_prob: float, mean_span_len: int,
                       device: torch.device) -> Tensor:
    """Mixed FITB masking: stochastically choose random vs span per batch.

    Args:
        mask_rate: target fraction of tokens to mask
        span_prob: probability of using span masking (vs random)
        mean_span_len: mean length for geometric span distribution

    Returns: [BS, N] bool on ``device`` — True at positions to mask.
    """
    if mask_rate <= 0.0:
        return torch.zeros(BS, N, dtype=torch.bool, device=device)

    use_span = random.random() < span_prob

    if use_span:
        return generate_span_mask(BS, N, mask_rate, mean_span_len, device)
    else:
        return generate_random_mask(BS, N, mask_rate, device)


def apply_special_token_protection(fitb_mask: Tensor, seg_ids: Tensor,
                                   eot_id: int, null_id: int) -> Tensor:
    """Never mask EOT or NULL tokens.

    Args:
        fitb_mask: [BS, N] bool
        seg_ids: [BS, N] token IDs (before masking)
        eot_id: end-of-text token ID
        null_id: <NULL> token ID

    Returns: [BS, N] bool with protected positions cleared.
    """
    protected = (seg_ids == eot_id) | (seg_ids == null_id)
    return fitb_mask & ~protected
