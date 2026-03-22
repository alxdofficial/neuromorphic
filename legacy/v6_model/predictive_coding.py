"""
Predictive Coding (v5) — Within-scan PCM + grouped utilities.

GroupedLinear and GroupedLayerNorm are general utilities used throughout.
WithinScanPCM: within-scan prediction (predict next token's encoding,
compute vector surprise). Replaces v4's CrossPassPCM.
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
        x_norm = F.layer_norm(x, (self.dim,), None, None, self.eps)
        return x_norm * self.weight + self.bias


class GroupedLinear(nn.Module):
    """Batched linear for C independent column groups.

    Weight: [C, in_features, out_features] (pre-transposed)
    Input:  [..., C, in_features]
    Output: [..., C, out_features]
    """

    def __init__(self, C: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.C = C
        self.in_features = in_features
        self.out_features = out_features
        # Store weight pre-transposed: [C, in_features, out_features]
        self.weight = nn.Parameter(torch.empty(C, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(C, out_features))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()

    def _reset_parameters(self):
        for c in range(self.C):
            # kaiming_uniform_ expects [out, in] layout
            w_t = self.weight.data[c].t()  # [out_features, in_features] view
            nn.init.kaiming_uniform_(w_t, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., C, in_features] -> [..., C, out_features]
        # einsum avoids the transpose-bmm-transpose copy pattern
        out = torch.einsum('...ci,cio->...co', x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class WithinScanPCM(nn.Module):
    """Within-scan predictive coding.

    Predicts next token's encoding from current scan state.
    Surprise = z_hat_{t-1} - z_t (vector, D_col dims).

    Per-column via grouped ops.
    """

    def __init__(self, C: int, D_col: int):
        super().__init__()
        self.C = C
        self.D_col = D_col

        self.W_enc = GroupedLinear(C, D_col, D_col)
        self.W_pcm = GroupedLinear(C, D_col, D_col)
        self.W_gain = GroupedLinear(C, D_col, D_col)

        # Zero-init W_gain so gain starts at 1.0
        nn.init.zeros_(self.W_gain.weight)
        if self.W_gain.bias is not None:
            nn.init.zeros_(self.W_gain.bias)

    def compute_surprise(self, H: Tensor, x_col: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute vector surprise and PCM prediction.

        Args:
            H: [BS, N, C, D_col] — scan hidden states
            x_col: [BS, N, C, D_col] — column input (pre-scan)

        Returns:
            surprise: [BS, N, C, D_col] — vector surprise (zero at position 0)
            z_hat: [BS, N, C, D_col] — prediction for next token
            z: [BS, N, C, D_col] — encoding of current token
        """
        z = self.W_enc(x_col)       # [BS, N, C, D_col]
        z_hat = self.W_pcm(H)       # [BS, N, C, D_col]

        # surprise_t = z_hat_{t-1} - z_t (shifted comparison)
        surprise = torch.zeros_like(z)
        surprise[:, 1:] = z_hat[:, :-1] - z[:, 1:]

        return surprise, z_hat, z

    def apply_gain(self, H: Tensor, surprise: Tensor) -> Tensor:
        """PCM gain modulation: bounded [0.9, 1.1].

        Args:
            H: [BS, N, C, D_col] — scan hidden states
            surprise: [BS, N, C, D_col] — vector surprise

        Returns:
            H_mod: [BS, N, C, D_col] — gain-modulated states
        """
        gain = 1.0 + 0.1 * torch.tanh(self.W_gain(surprise))
        return H * gain

    def prediction_loss(self, z_hat: Tensor, z: Tensor) -> Tensor:
        """Auxiliary prediction loss: MSE(z_hat_t, z_{t+1}.detach()).

        Uses shifted comparison: z_hat[:, :-1] vs z[:, 1:].
        """
        return F.mse_loss(z_hat[:, :-1], z[:, 1:].detach())
