"""Shared encoder helpers.

Only genuinely cross-model utilities live here. Each model/baseline keeps its
own private helpers inside its `models/<name>/` subfolder; the one helper that
is shared across several baselines is `_NormMatch`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _NormMatch(nn.Module):
    """Put projected memory tokens in Llama's token-embedding magnitude region.

    The prepend projections used to end in nn.LayerNorm(d_llama), whose output has L2 norm
    ~sqrt(d_llama) ≈ 45 — ~49× the ~0.9 norm of real Llama token embeddings. Prepended
    unmasked, those 49×-loud tokens act as attention distractors the frozen LM cannot route
    around (the dominant reason the prepend baselines fell below the no-memory floor). Keep
    the LayerNorm (centering/stability) but rescale to ~0.9 (learnable). This is a magnitude
    CORRECTION, not added capacity — it does not change what the baseline is.
    """

    def __init__(self, d: int, target: float = 0.9):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.scale = nn.Parameter(torch.tensor(float(target)))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.ln(x), dim=-1) * self.scale
