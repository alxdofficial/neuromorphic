"""
Shared utilities for the neuromorphic LM.

Pure functions used across memory modules and a StateMixin base class
providing common state management (detach/reset/save/load).
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def unit_normalize(x: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    """L2-normalize along `dim`."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def soft_topk(scores: Tensor, k: int, tau: float = 1.0) -> Tensor:
    """Softmax over top-k entries, zero rest.

    Args:
        scores: [*, N] raw scores
        k: number of entries to keep
        tau: softmax temperature

    Returns:
        weights: [*, N] with top-k softmaxed, rest zero
    """
    N = scores.shape[-1]
    if k >= N:
        return torch.softmax(scores / tau, dim=-1)

    topk_vals, topk_idx = scores.topk(k, dim=-1)
    weights = torch.zeros_like(scores)
    softmaxed = torch.softmax(topk_vals / tau, dim=-1)
    weights.scatter_(-1, topk_idx, softmaxed)
    return weights


def budget_enforce(strengths: Tensor, budget: float) -> Tensor:
    """Scale down strengths if sum exceeds budget (per-stream).

    Args:
        strengths: [BS, N] non-negative strengths
        budget: maximum allowed sum per stream

    Returns:
        strengths: [BS, N] scaled so sum <= budget
    """
    total = strengths.sum(dim=-1, keepdim=True)  # [BS, 1]
    scale = torch.where(total > budget, budget / (total + 1e-8), torch.ones_like(total))
    return strengths * scale


# ---------------------------------------------------------------------------
# StateMixin
# ---------------------------------------------------------------------------

class StateMixin:
    """Mixin providing common state management for stateful modules.

    Convention: runtime state tensors are stored as plain attributes
    (not nn.Parameter, not register_buffer). They are listed in
    `_state_tensor_names` by each subclass.
    """

    # Subclasses should set this to a list of attribute names that are
    # runtime state tensors (e.g. ["pm_K", "pm_V", "pm_a"]).
    _state_tensor_names: list = []

    def _get_state_tensors(self) -> Dict[str, Optional[Tensor]]:
        """Return dict of name -> tensor for all runtime state."""
        result = {}
        for name in self._state_tensor_names:
            result[name] = getattr(self, name, None)
        return result

    def detach_states(self):
        """Detach all runtime state tensors in-place (TBPTT boundary)."""
        for name in self._state_tensor_names:
            t = getattr(self, name, None)
            if t is not None and isinstance(t, Tensor):
                setattr(self, name, t.detach())

    def reset_states(self, mask: Tensor):
        """Zero out runtime state for masked streams.

        Args:
            mask: [BS] bool — True for streams to reset
        """
        for name in self._state_tensor_names:
            t = getattr(self, name, None)
            if t is not None and isinstance(t, Tensor) and t.dim() >= 1:
                # mask is [BS], t is [BS, ...] — zero out masked rows
                expanded = mask
                for _ in range(t.dim() - 1):
                    expanded = expanded.unsqueeze(-1)
                setattr(self, name, t * (~expanded).to(t.dtype))

    def state_dict_runtime(self) -> Dict[str, Optional[Tensor]]:
        """Serialize runtime state for checkpointing."""
        return {name: getattr(self, name, None) for name in self._state_tensor_names}

    def load_state_runtime(self, state: Dict[str, Optional[Tensor]]):
        """Load runtime state from checkpoint."""
        for name, val in state.items():
            if name in self._state_tensor_names:
                setattr(self, name, val)
