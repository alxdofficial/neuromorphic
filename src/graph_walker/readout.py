"""Multi-horizon readout + ring buffer + per-horizon surprise EMA.

Ported from the gridworld readout with one change: the `motor` vector
now arrives from a cross-attention pool over output-plane column states,
not from a concat of output-neuron slices. The rest of the machinery
(PredictionHead residual, K-factored logits, ring buffer, multi-horizon
CE surprise EMA) is mathematically unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph_walker.config import GraphWalkerConfig


class PredictionHead(nn.Module):
    """Small Linear-residual before the tied unembedding.

    Zero-initialized so at day 0 the head is identity through the residual.
    The thesis says the substrate should produce predictions; this head
    is just a learned re-mix.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = _rmsnorm(dim)
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(self.norm(x))


def _rmsnorm(dim: int) -> nn.Module:
    # Use the autocast-friendly fallback even when nn.RMSNorm exists, because
    # PyTorch's RMSNorm does not auto-cast its weight to input dtype under
    # autocast, which disables its fused kernel and warns every call.
    return _FallbackRMSNorm(dim)


class _FallbackRMSNorm(nn.Module):
    """Autocast-friendly RMSNorm: casts weight to match input dtype so the
    fused kernel path can be selected. Computation is in input dtype to
    avoid the bf16↔fp32 cast warning; this is acceptable for RMSNorm because
    the normalisation itself is scale-invariant."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * w


class MultiHorizonReadout(nn.Module):
    """Given a motor vector [B, D_s], produce [B, K_horizons, V] logits.

    Factored form: `(motor + h_k) @ W^T = motor @ W^T + h_k @ W^T`.
    Tied unembedding (W_unembed = token_emb.weight) is supplied at call time.
    """

    def __init__(self, cfg: GraphWalkerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pred_head = PredictionHead(cfg.D_s)
        self.horizon_emb = nn.Parameter(
            torch.randn(cfg.K_horizons, cfg.D_s) * 0.02
        )

    def forward(
        self, motor: torch.Tensor, unembedding: torch.Tensor
    ) -> torch.Tensor:
        """motor: [B, D_s]; unembedding: [V, D_s]. Returns [B, K_horizons, V]."""
        x = self.pred_head(motor)                          # [B, D_s]
        W_T = unembedding.t()                              # [D_s, V]
        logits_motor = torch.matmul(x, W_T)                # [B, V]
        logits_horizon = torch.matmul(self.horizon_emb, W_T)  # [K_h, V]
        return logits_motor.unsqueeze(1) + logits_horizon   # [B, K_h, V]


# =====================================================================
# Ring buffer of past logits + multi-horizon surprise (unchanged math)
# =====================================================================


def init_prediction_buffer(
    B: int, K_buf: int, K_horizons: int, V: int, device, dtype
) -> torch.Tensor:
    return torch.zeros(B, K_buf, K_horizons, V, device=device, dtype=dtype)


def write_prediction_buffer(
    pred_buf: torch.Tensor,
    cursor: int,
    logits: torch.Tensor,
) -> None:
    """Write this step's [B, K_h, V] logits into the buffer at `cursor`. Detached."""
    with torch.no_grad():
        pred_buf[:, cursor] = logits.detach()


def read_past_prediction(
    pred_buf: torch.Tensor,
    cursor: int,
    K_buf: int,
    k: int,
    filled: int,
) -> torch.Tensor | None:
    """The k-step-ahead prediction emitted k ticks ago, aligned with now.

    Returns `.detach().clone()` to avoid autograd versioning issues when
    the buffer is mutated on a later step.
    """
    if filled < k:
        return None
    idx = (cursor - k) % K_buf
    return pred_buf[:, idx, k - 1].detach().clone()


def multi_horizon_surprise(
    pred_buf: torch.Tensor,
    cursor: int,
    K_buf: int,
    filled: int,
    actual_token: torch.Tensor,
    surprise_ema: torch.Tensor,
    gamma_s: float,
) -> torch.Tensor:
    """Update per-horizon surprise EMA in fp32; return with original dtype.

    For horizons not yet filled (`filled < k`), the EMA entry is unchanged.
    """
    original_dtype = surprise_ema.dtype
    K_horizons = pred_buf.shape[2]
    with torch.autocast(device_type=actual_token.device.type, enabled=False):
        new_ema = surprise_ema.float().clone()
        for k in range(1, K_horizons + 1):
            past = read_past_prediction(pred_buf, cursor, K_buf, k, filled)
            if past is None:
                continue
            ce = F.cross_entropy(past.float(), actual_token, reduction="none")
            new_ema[:, k - 1] = (1.0 - gamma_s) * new_ema[:, k - 1] + gamma_s * ce
    return new_ema.to(original_dtype)
