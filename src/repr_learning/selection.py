"""Classification-style node selection (replaces VQ-VAE quantization).

Forward: hard argmax over node scores.
Backward: Gumbel-softmax + straight-through estimator.

Also provides the load-balance auxiliary loss (Switch Transformer style)
to prevent classification collapse — where the model would otherwise
always pick the same few nodes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def gumbel_argmax_ste(
    scores: Tensor,
    temperature: float = 0.5,
    training: bool = True,
) -> tuple[Tensor, Tensor]:
    """Hard argmax forward, Gumbel-softmax STE backward.

    Args:
        scores: [..., N] raw logits (unnormalized).
        temperature: τ for Gumbel-softmax. Lower = sharper.
        training: if True, add Gumbel noise; else deterministic argmax.

    Returns:
        idx: [...] long — hard-picked indices.
        one_hot_ste: [..., N] float — soft one-hot with STE gradient.

    Usage:
        idx, one_hot = gumbel_argmax_ste(scores, ...)
        # Lookup embedding:
        emb = one_hot @ codebook   # [..., D] — gradient flows to both scores AND codebook
    """
    if training:
        # Gumbel noise: g = −log(−log(U))
        # U ∈ (0, 1) → -log(U) > 0 → -log(-log(U)) ∈ ℝ
        # Careful with clamps: clamp U inside (0, 1), and clamp inner -log > 0.
        u = torch.rand_like(scores).clamp(min=1e-9, max=1.0 - 1e-9)
        inner = (-torch.log(u)).clamp_min(1e-9)            # > 0
        gumbel = -torch.log(inner)                          # well-defined
        scores_noisy = (scores + gumbel) / temperature
    else:
        scores_noisy = scores / temperature

    # Hard argmax (forward)
    idx = scores_noisy.argmax(dim=-1)

    # Soft softmax (backward path)
    soft = F.softmax(scores_noisy, dim=-1)

    # Straight-through one-hot: forward = one_hot, backward = soft
    one_hot = F.one_hot(idx, num_classes=scores.shape[-1]).to(soft.dtype)
    one_hot_ste = one_hot + soft - soft.detach()

    return idx, one_hot_ste


def load_balance_loss(scores: Tensor) -> Tensor:
    """Switch Transformer-style load-balance aux loss.

    Encourages uniform usage across all N nodes by penalizing the
    correlation between empirical pick frequencies (f) and mean softmax
    probabilities (P).

    Args:
        scores: [batch_dims..., N] raw logits.

    Returns:
        scalar loss. Multiply by α (e.g., 0.01) and add to total loss.
    """
    N = scores.shape[-1]
    flat_scores = scores.reshape(-1, N)  # [B*S, N]
    M = flat_scores.shape[0]

    # Empirical fraction of times each node was picked (top-1)
    picks = flat_scores.argmax(dim=-1)  # [M]
    f = torch.bincount(picks, minlength=N).float() / max(M, 1)  # [N]

    # Mean softmax probability per node across all events
    probs = F.softmax(flat_scores, dim=-1)  # [M, N]
    P = probs.mean(dim=0)  # [N]

    # Switch-style: N · Σ_i (f_i · P_i). Min when uniform.
    return N * (f.to(P.dtype) * P).sum()
