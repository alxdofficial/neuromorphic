"""Multi-horizon readout for graph-walker.

The dense lexical stack lives here on purpose: it is cheaper to pay for large
model-space capacity once per token or once per block than inside the per-hop
graph core. The graph core therefore returns `motor_state` in D_s, while this
module handles the larger D_model projection and tied unembedding.
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


class _ResidualFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = _rmsnorm(dim)
        self.up = nn.Linear(dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, dim)
        nn.init.normal_(self.down.weight, mean=0.0, std=(2.0 / hidden_dim) ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(self.norm(x))))


class PostModelStack(nn.Module):
    """Model-space residual FFN stack.

    This is the main place to keep capacity after shrinking the graph state:
    it runs once per token on the readout vector, not once per hop inside the
    walker hot loop.
    """

    def __init__(self, dim: int, depth: int, mult: int) -> None:
        super().__init__()
        hidden_dim = mult * dim
        self.blocks = nn.ModuleList([
            _ResidualFFN(dim, hidden_dim) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


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
    """Given motor vectors [..., D_model], produce [..., K_horizons, V] logits.

    Factored form: `(motor + h_k) @ W^T = motor @ W^T + h_k @ W^T`.
    Tied unembedding (W_unembed = token_emb.weight) is supplied at call time.
    """

    def __init__(self, cfg: GraphWalkerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.post_stack = PostModelStack(
            cfg.D_model, cfg.post_model_depth, cfg.post_model_ffn_mult,
        )
        self.pred_head = PredictionHead(cfg.D_model)
        self.horizon_emb = nn.Parameter(
            torch.randn(cfg.K_horizons, cfg.D_model) * 0.02
        )

    def forward(
        self,
        motor: torch.Tensor,
        unembedding: torch.Tensor,
        horizon_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """motor: [..., D_model]; unembedding: [V, D_model].

        Returns: [..., K_horizons, V].

        horizon_logits (optional): precomputed [K_h, V] = horizon_emb @ W^T.
        Pass this in to avoid redoing the K_h × D_model × V matmul every token;
        both operands are Parameters constant within a forward pass.
        """
        x = self.post_stack(motor)
        x = self.pred_head(x)                              # [B, D_model]
        W_T = unembedding.t()                              # [D_model, V]
        logits_motor = torch.matmul(x, W_T)                # [B, V]
        if horizon_logits is None:
            horizon_logits = torch.matmul(self.horizon_emb, W_T)  # [K_h, V]
        return logits_motor.unsqueeze(-2) + horizon_logits  # [..., K_h, V]

    def cross_entropy_factorized(
        self,
        motor: torch.Tensor,         # [B, T, D_model]
        unembedding: torch.Tensor,   # [V, D_model]
        targets: torch.Tensor,       # [B, T, K_h] int64 — target token per horizon
        valid: torch.Tensor,         # [B, T, K_h] bool — horizon has a valid target
        horizon_logits: torch.Tensor | None = None,  # optional [K_h, V] cache
    ) -> torch.Tensor:
        """Memory-efficient CE: avoids the [B, T, K_h, V] broadcast.

        Exploits the factorization
            logits[b,t,k,v] = motor_logits[b,t,v] + horizon_logits[k,v]
        to compute CE per-horizon, iterating over `K_h` in Python. Each
        iteration materializes a [B, T, V] tensor (the summed logits for
        one horizon) instead of the full [B, T, K_h, V].

        At BS=88, T=48, K_h=8, V=32000, bf16: this saves ~3.8 GB of active
        memory compared to the broadcast path, plus a matching save on the
        log_softmax that `F.cross_entropy` would otherwise retain.

        Returns: `[B, T, K_h]` float32 cross-entropy per (position, horizon),
        with invalid entries zeroed per `valid`.
        """
        B, T, _ = motor.shape
        K_h = targets.shape[-1]
        V = unembedding.shape[0]

        x = self.post_stack(motor)
        x = self.pred_head(x)                              # [B, T, D_model]
        W_T = unembedding.t()                              # [D_model, V]
        motor_logits = torch.matmul(x, W_T)                # [B, T, V] — reused

        if horizon_logits is None:
            horizon_logits = torch.matmul(self.horizon_emb, W_T)  # [K_h, V]

        ce_out = torch.empty(B, T, K_h, device=motor.device, dtype=torch.float32)
        for k in range(K_h):
            logits_k = motor_logits + horizon_logits[k]     # [B, T, V]
            ce_k = F.cross_entropy(
                logits_k.reshape(-1, V),
                targets[..., k].reshape(-1),
                reduction="none",
            ).reshape(B, T)                                 # [B, T]
            ce_out[..., k] = ce_k * valid[..., k].float()

        return ce_out
