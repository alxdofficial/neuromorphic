"""Classification-style node selection (replaces VQ-VAE quantization).

Forward: hard argmax over node scores.
Backward: Gumbel-softmax + straight-through estimator.

Also provides the load-balance auxiliary loss (Switch Transformer style)
to prevent classification collapse — where the model would otherwise
always pick the same few nodes.
"""
from __future__ import annotations
from typing import Optional

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


def load_balance_loss(
    scores: Tensor,
    picks: Optional[Tensor] = None,
) -> Tensor:
    """Switch Transformer-style load-balance aux loss.

    Encourages uniform usage across all N nodes by penalizing the
    correlation between empirical pick frequencies (f) and mean softmax
    probabilities (P).

    Args:
        scores: [batch_dims..., N] raw logits.
        picks: optional [batch_dims...] long tensor of actually-selected
               indices. If provided, f is computed from these (matches
               the routes actually used in the forward pass). If None,
               falls back to raw-score argmax — fine when scores
               dominate Gumbel noise, but misleading at init when
               argmax(scores) ≠ argmax(scores + gumbel).

    Returns:
        scalar loss. Multiply by α (e.g., 0.01) and add to total loss.
    """
    N = scores.shape[-1]
    flat_scores = scores.reshape(-1, N)  # [B*S, N]
    M = flat_scores.shape[0]

    # Empirical fraction of times each node was picked.
    if picks is None:
        picks_flat = flat_scores.argmax(dim=-1)
    else:
        picks_flat = picks.reshape(-1)
    f = torch.bincount(picks_flat, minlength=N).float() / max(M, 1)  # [N]

    # Mean softmax probability per node across all events
    probs = F.softmax(flat_scores, dim=-1)  # [M, N]
    P = probs.mean(dim=0)  # [N]

    # Switch-style: N · Σ_i (f_i · P_i). Min when uniform.
    return N * (f.to(P.dtype) * P).sum()


def router_z_loss(scores: Tensor) -> Tensor:
    """Mixtral / Switch-Transformer-v2 router z-loss.

    Penalizes (logsumexp(scores))² averaged over routing events. The
    softmax is scale-invariant — adding a constant to every score gives
    the same picks — so without z-loss the network is free to push logits
    arbitrarily large. Large logits in turn make Gumbel noise irrelevant
    (sharp picks), which removes exploration and amplifies any incipient
    routing imbalance. z-loss caps that pressure.

    Args:
        scores: [..., N] raw logits (post-scale, pre-softmax).

    Returns:
        scalar loss. Multiply by ~1e-3 and add to total loss.
    """
    # logsumexp over the routing-dim. Average over all routing events.
    lse = torch.logsumexp(scores.float(), dim=-1)
    return lse.pow(2).mean()
