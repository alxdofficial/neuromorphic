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
    log_pi: torch.Tensor | None = None  # [B] float — log π(selected_idx); only
                                        # populated in phase="phase2" (hard
                                        # Categorical, REINFORCE / GRPO).


def _sample_exploration_mask(
    B: int,
    K: int,
    epsilon: torch.Tensor | float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce (explore_mask [B], uniform_sample [B]).

    Uses rand+argmax to sample categorical uniform over [0, K) instead of
    torch.randint — the latter fuses into the Gumbel+softmax+bmm kernel
    and triggers a Triton 3.6 TritonGPURemoveLayoutConversions crash.
    With rand+argmax, the op pattern is inductor-friendly, so this path
    is now compilable (no @torch._dynamo.disable needed).
    """
    # Epsilon is a tensor under compile — compare element-wise for the mask.
    explore_mask = (torch.rand(B, device=device) < epsilon)
    # Categorical uniform over [0, K) via argmax of K independent uniform values.
    uniform_sample = torch.rand(B, K, device=device).argmax(dim=-1)
    return explore_mask, uniform_sample


def gumbel_top1_softmax(
    scores: torch.Tensor,          # [B, K] — edge logits
    tau: torch.Tensor | float,     # Gumbel temperature (tensor for compile)
    epsilon: torch.Tensor | float, # exploration probability (tensor for compile)
    training: bool,                # True → stochastic; False → argmax
    phase: str = "phase1",         # "phase1" Gumbel-STE | "phase2" hard Categorical
) -> RoutingOutput:
    """Gumbel top-1 softmax with straight-through and optional ε-exploration.

    Phases:
    - `phase1` (default): Gumbel-soft sample, hard one-hot forward, soft
      gradient backward via STE. Used for backprop-driven training.
    - `phase2`: hard Categorical sample from softmax(scores), no temperature
      noise, returns `log_pi` (log of the picked-index probability under
      the policy). For REINFORCE / GRPO. ε-exploration is disabled in this
      mode — exploration in phase 2 is Categorical sampling itself.

    At inference (training=False): hard argmax directly, no noise, no log_pi.

    tau and epsilon are tensors (not Python floats) so dynamo doesn't
    specialise-recompile each time they anneal.
    """
    B, K = scores.shape
    device = scores.device

    if training and phase == "phase2":
        # Hard Categorical sample. log_pi = log p(selected) under softmax(scores).
        soft = F.softmax(scores.float(), dim=-1)
        log_probs = F.log_softmax(scores.float(), dim=-1)
        # `multinomial` requires cpu/cuda probs; numerically clamp to avoid 0s.
        probs = soft.clamp(min=1e-12)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)   # [B]
        # log_pi must remain graph-connected to scores so REINFORCE backprop
        # reaches every parameter that produced `scores` (q_proj, k_all,
        # E_bias, neuromod via active_E_bias).
        log_pi = log_probs.gather(1, sampled.unsqueeze(-1)).squeeze(-1) # [B]
        hard = F.one_hot(sampled, num_classes=K).to(scores.dtype)
        return RoutingOutput(
            selected_idx=sampled,
            ste_weights=hard,                # forward only — gradient comes via log_pi
            soft_probs=soft,
            log_pi=log_pi,
        )

    if training:
        # Gumbel noise
        u = torch.rand_like(scores).clamp_(min=1e-9, max=1 - 1e-9)
        gumbel = -torch.log(-torch.log(u))
        tau_safe = tau.clamp(min=1e-3) if isinstance(tau, torch.Tensor) \
            else max(tau, 1e-3)
        logits = (scores + gumbel) / tau_safe
        soft = F.softmax(logits, dim=-1)                          # [B, K]

        # Hard argmax
        argmax = soft.argmax(dim=-1)                               # [B]

        # ε-exploration: swap in uniform random sample for εB items.
        # Exploration sampling is done in an eager helper to avoid a Triton
        # fusion bug (randint + softmax/bmm → PassManager crash on 3.6).
        explore_mask, uniform_sample = _sample_exploration_mask(
            B, K, epsilon, device,
        )
        argmax = torch.where(explore_mask, uniform_sample, argmax)

        hard = F.one_hot(argmax, num_classes=K).to(soft.dtype)      # [B, K]

        # Straight-through: forward = hard, backward = soft.
        # For exploration samples (where argmax was overridden by a uniform
        # pick), the soft distribution was NOT conditioned on the override
        # so letting gradient flow through it would reward arbitrary random
        # edges. We zero the gradient on those rows by using a detached STE.
        ste_soft = hard - soft.detach() + soft                      # grad via soft
        ste_hard = hard.detach()                                    # no gradient
        ste = torch.where(
            explore_mask.unsqueeze(-1), ste_hard, ste_soft,
        )
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
