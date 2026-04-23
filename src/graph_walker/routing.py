"""Gumbel top-1 softmax routing with straight-through estimator.

Design (from docs/graph_walker.md §5.4):

  logits = scores + E_bias           # [B, K]
  # Gumbel-perturbed
  g = -log(-log(U(0,1)))
  logits_g = (logits + g) / τ
  # Soft distribution (differentiable)
  soft = softmax(logits_g, dim=-1)
  # Hard one-hot (non-differentiable)
  hard = one_hot(argmax(soft))
  # Straight-through: forward uses hard, backward uses soft
  ste = hard - soft.detach() + soft

  # With probability ε, replace with uniform random sample (exploration)
  ...

Returns:
  selected_idx: [B] long — the chosen neighbor index (for trajectory recording)
  ste_weights: [B, K] float — straight-through weights (for gradient-carrying
                              contribution in aggregation)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class RoutingOutput:
    """Bundle returned from the routing op."""
    selected_idx: torch.Tensor     # [B] long — argmax of Gumbel-perturbed logits
    ste_weights: torch.Tensor      # [B, K] float — straight-through one-hot
    soft_probs: torch.Tensor       # [B, K] float — for analysis / load-balance loss


def gumbel_top1_softmax(
    scores: torch.Tensor,          # [B, K] — edge logits
    tau: float,                    # Gumbel temperature
    epsilon: float,                # exploration probability (0 = off)
    training: bool,                # True → stochastic; False → argmax
) -> RoutingOutput:
    """Gumbel top-1 softmax with straight-through and optional ε-exploration.

    At training time: sample via Gumbel, return hard argmax forward + soft
    gradient backward. With probability ε, replace with uniform random.

    At inference: hard argmax directly, no noise.
    """
    B, K = scores.shape
    device = scores.device

    if training:
        # Gumbel noise
        u = torch.rand_like(scores).clamp_(min=1e-9, max=1 - 1e-9)
        gumbel = -torch.log(-torch.log(u))
        logits = (scores + gumbel) / max(tau, 1e-3)
        soft = F.softmax(logits, dim=-1)                          # [B, K]

        # Hard argmax
        argmax = soft.argmax(dim=-1)                               # [B]

        # ε-exploration: swap in uniform random sample for εB items
        if epsilon > 0.0:
            explore_mask = (torch.rand(B, device=device) < epsilon)   # [B] bool
            uniform_sample = torch.randint(0, K, (B,), device=device)
            argmax = torch.where(explore_mask, uniform_sample, argmax)

        hard = F.one_hot(argmax, num_classes=K).to(soft.dtype)      # [B, K]

        # Straight-through: forward = hard, backward = soft
        ste = hard - soft.detach() + soft
    else:
        # Inference: hard argmax of raw scores
        soft = F.softmax(scores, dim=-1)
        argmax = scores.argmax(dim=-1)
        ste = F.one_hot(argmax, num_classes=K).to(scores.dtype)

    return RoutingOutput(
        selected_idx=argmax, ste_weights=ste, soft_probs=soft,
    )


def gumbel_schedule(step: int, start: float, end: float, anneal_steps: int) -> float:
    """Linear anneal from `start` to `end` over `anneal_steps`, then hold."""
    if step <= 0:
        return start
    if step >= anneal_steps:
        return end
    frac = step / anneal_steps
    return start + frac * (end - start)
