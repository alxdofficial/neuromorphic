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
    """L2-normalize along `dim`. Backward-safe for zero inputs.

    Uses squared-norm + eps inside sqrt so the gradient is well-defined
    when x == 0 (avoids 0/0 NaN from torch.norm backward).
    """
    norm_sq = (x * x).sum(dim=dim, keepdim=True)
    return x / (norm_sq + eps).sqrt()



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
        """Load runtime state from checkpoint.

        Shape-safe: skips tensors whose shapes don't match the current
        model's initialized state. This handles tier/config changes
        (e.g. r=8 -> r=16) gracefully — mismatched state is discarded
        and the model falls back to lazy re-initialization.
        """
        for name, val in state.items():
            if name not in self._state_tensor_names:
                continue
            current = getattr(self, name, None)
            if val is None:
                setattr(self, name, None)
            elif current is not None and current.shape != val.shape:
                # Shape mismatch — skip (will be re-initialized lazily)
                print(f"  Runtime state shape mismatch for {type(self).__name__}.{name}: "
                      f"checkpoint {list(val.shape)} vs model {list(current.shape)}, skipping.")
            else:
                setattr(self, name, val)
