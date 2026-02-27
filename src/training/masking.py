"""
FITB (Fill-In-The-Blank) masking utilities for neuromorphic LM training.

Generates boolean masks indicating which token positions to replace with <FITB>.
Supports random masking, span masking (geometric distribution), and mixed mode.
"""

import torch
from torch import Tensor


def generate_random_mask(BS: int, N: int, mask_rate: float,
                         device: torch.device) -> Tensor:
    """Uniform random masking.

    Returns: [BS, N] bool — True at positions to mask.
    """
    return torch.rand(BS, N, device=device) < mask_rate


def generate_span_mask(BS: int, N: int, mask_rate: float,
                       mean_span_len: int, device: torch.device) -> Tensor:
    """Span masking with geometric span lengths.

    Generates spans whose lengths follow Geometric(p=1/mean_span_len).
    Spans are placed until the target mask_rate is approximately reached.

    Returns: [BS, N] bool — True at positions to mask.
    """
    target_count = int(mask_rate * N)
    if target_count == 0:
        return torch.zeros(BS, N, dtype=torch.bool, device=device)

    mask = torch.zeros(BS, N, dtype=torch.bool, device=device)
    p = 1.0 / max(mean_span_len, 1)

    for b in range(BS):
        masked = 0
        # Retry budget to avoid infinite loops on edge cases
        attempts = 0
        while masked < target_count and attempts < N:
            # Random span start
            start = torch.randint(0, N, (1,), device=device).item()
            # Geometric span length (at least 1)
            length = int(torch.distributions.Geometric(probs=p).sample().item()) + 1
            end = min(start + length, N)
            mask[b, start:end] = True
            masked = mask[b].sum().item()
            attempts += 1

    return mask


def generate_fitb_mask(BS: int, N: int, mask_rate: float,
                       span_prob: float, mean_span_len: int,
                       device: torch.device) -> Tensor:
    """Mixed FITB masking: stochastically choose random vs span per batch.

    Args:
        mask_rate: target fraction of tokens to mask
        span_prob: probability of using span masking (vs random)
        mean_span_len: mean length for geometric span distribution

    Returns: [BS, N] bool — True at positions to mask.
    """
    if mask_rate <= 0.0:
        return torch.zeros(BS, N, dtype=torch.bool, device=device)

    use_span = torch.rand(1, device=device).item() < span_prob

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
