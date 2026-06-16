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


def resolve_special_ids(tokenizer) -> tuple[int, int]:
    """LLM-AGNOSTIC (pad, sep) ids derived from the ACTIVE tokenizer, so the data
    layer never bakes in a specific model's vocab (the old hard-coded 128001/198 are
    Llama-only → out of range on SmolLM2, etc.). pad = the tokenizer's pad (else eos);
    sep = the tokenizer's newline token. Swap the backbone → these follow it."""
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    nl = tokenizer("\n", add_special_tokens=False)["input_ids"]
    sep = nl[-1] if nl else pad
    return int(pad), int(sep)


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


def beacon_wrap_layers(n_layers: int, n: int = 6) -> tuple:
    """`n` evenly-spaced decoder-layer indices for the Beacon baseline to wrap.

    Beacon's capacity knob is the NUMBER of wrapped layers (each adds full q/k/v
    copies). Derive from the backbone's actual depth so the same call is correct
    on any backbone (SmolLM2-135M's 30 layers → (0,6,12,17,23,29); a 16-layer
    backbone → (0,3,6,9,12,15)). Shared by the trainer, param_count, and tests so
    they can never drift. n=6 is the SmolLM2-135M capacity calibration (≈4.24M).
    """
    qs = [i / (n - 1) for i in range(n)]
    return tuple(sorted({int(round(q * (n_layers - 1))) for q in qs}))
