"""
Cross-pass Predictive Coding Module (v4).

Per-column via grouped ops. Predicts what each token's encoding will look
like next pass. Surprise = how much memory update changed understanding.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GroupedLayerNorm(nn.Module):
    """Per-group LayerNorm. Weight/bias: [C, dim]."""

    def __init__(self, C: int, dim: int, eps: float = 1e-5):
        super().__init__()
        self.C = C
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(C, dim))
        self.bias = nn.Parameter(torch.zeros(C, dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., C, dim]
        var, mean = torch.var_mean(x, dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return x_norm * self.weight + self.bias


class GroupedLinear(nn.Module):
    """Batched linear for C independent column groups.

    Weight: [C, out_features, in_features]
    Input:  [..., C, in_features]
    Output: [..., C, out_features]
    """

    def __init__(self, C: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.C = C
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(C, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(C, out_features))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()

    def _reset_parameters(self):
        for c in range(self.C):
            nn.init.kaiming_uniform_(self.weight[c], a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., C, in_features] -> [..., C, out_features]
        out = torch.einsum("...ci, coi -> ...co", x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class CrossPassPCM(nn.Module):
    """Cross-pass predictive coding.

    Predicts what each token's encoding will look like next pass.
    Surprise = how much memory update changed understanding.
    """

    def __init__(self, C: int, D_col: int, D_pcm: int):
        super().__init__()
        self.C = C
        self.D_col = D_col
        self.D_pcm = D_pcm
        self.inv_sqrt_D = 1.0 / math.sqrt(D_pcm)

        self.encoder_norm = GroupedLayerNorm(C, D_col)
        self.encoder = GroupedLinear(C, D_col, D_pcm)
        self.hyp_up = GroupedLinear(C, D_pcm, D_pcm * 2)
        self.hyp_down = GroupedLinear(C, D_pcm * 2, D_pcm)

    def encode(self, x_col: Tensor) -> Tensor:
        """x_col: [BS,N,C,D_col] -> z: [BS,N,C,D_pcm]"""
        return self.encoder(self.encoder_norm(x_col))

    def predict(self, z: Tensor) -> Tensor:
        """z: [BS,N,C,D_pcm] -> z_hat: [BS,N,C,D_pcm]"""
        return self.hyp_down(F.gelu(self.hyp_up(z)))

    def compute_surprise(self, z: Tensor, z_hat_prev: Tensor | None):
        """Returns (surprise [BS,N,C], delta [BS,N,C,D_pcm])"""
        if z_hat_prev is None:
            surprise = torch.zeros(
                z.shape[:-1], device=z.device, dtype=z.dtype
            )
            delta = torch.zeros_like(z)
            return surprise, delta
        delta = z - z_hat_prev.detach()
        surprise = delta.norm(dim=-1) * self.inv_sqrt_D
        return surprise, delta

    def prediction_loss(self, z_hat: Tensor, z_next: Tensor) -> Tensor:
        """L_pred = MSE(z_hat, z_next.detach()) averaged."""
        return F.mse_loss(z_hat, z_next.detach())
