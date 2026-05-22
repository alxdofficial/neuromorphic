"""JEPA training infrastructure for v1f.

Predicts the encoding of `chunk_2` from the encoding of `chunk_1`, with no
LM in the training loop. Reference: I-JEPA (Assran et al. 2023).

Components:
  - JEPAPredictor: small transformer mapping memory_1 → predicted memory_2
  - EMATarget: clone of online encoder updated as EMA of online weights
  - vicreg_regularizers: variance + covariance penalties (anti-collapse)
"""
from __future__ import annotations
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class JEPAPredictor(nn.Module):
    """Small transformer that maps memory tokens [B, M, d_in] → predicted
    target memory tokens [B, M, d_in].

    Architecture: in_proj → self-attention blocks → out_proj.
    Self-attention lets each query attend to other queries so the predictor
    can produce mutually coherent predictions.

    Defaults: d_in=2048 (d_llama), d_hidden=512, 2 layers, 8 heads.
    Param budget ≈ 7M, kept small relative to encoder (~13-19M each) so the
    representation quality, not the predictor, dominates the loss.
    """

    def __init__(self, d_in: int, d_hidden: int = 512, n_layers: int = 2,
                 n_heads: int = 8, d_ffn_mult: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=d_hidden, nhead=n_heads,
            dim_feedforward=d_hidden * d_ffn_mult,
            dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_hidden)
        self.out_proj = nn.Linear(d_hidden, d_in)

    def forward(self, x: Tensor) -> Tensor:
        h = self.in_proj(x)
        h = self.transformer(h)
        h = self.out_norm(h)
        return self.out_proj(h)


@torch.no_grad()
def init_ema_target(online: nn.Module) -> nn.Module:
    """Create a deep-copy EMA target with parameters frozen.

    Target follows the parent model's train/eval mode. We intentionally
    do NOT force .train(False) here: for Gumbel-STE encoders (V2.1, A),
    deterministic targets remove the only source of batch-level variance
    the online encoder can match, which lets it collapse to a constant
    pick while VicReg (measured on online) sees stochastic spread that
    masks the collapse. Keeping target in train mode means target picks
    are noisy too; the predictor sees varied targets and cannot win with
    a constant output.
    """
    target = copy.deepcopy(online)
    for p in target.parameters():
        p.requires_grad = False
    return target


@torch.no_grad()
def update_ema_target(online: nn.Module, target: nn.Module, tau: float = 0.996):
    """Update target weights as EMA of online: w_t ← τ·w_t + (1−τ)·w_o."""
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)


def vicreg_variance_loss(x: Tensor, eps: float = 1e-4, target_std: float = 1.0) -> Tensor:
    """VicReg variance term: each output dim should have std ≥ target_std.

    Penalizes hinge loss max(0, target_std - std). Flatten [B, M, D] → [B*M, D]
    and compute std per-dim across the B*M samples.
    """
    flat = x.float().reshape(-1, x.shape[-1])
    std = torch.sqrt(flat.var(dim=0) + eps)
    return F.relu(target_std - std).mean()


def vicreg_covariance_loss(x: Tensor) -> Tensor:
    """VicReg covariance term: off-diagonal covariance entries should be 0.

    Penalizes (sum of squared off-diag covariances) / D.
    """
    flat = x.float().reshape(-1, x.shape[-1])
    n, d = flat.shape
    centered = flat - flat.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).sum() / d
