"""Single-column Predictive Coding Module for v8.

Each cortical column has its own PCM with independent weights.
Operates on [BS, T, D_cc] tensors (not grouped [BS, T, C, D_col]).

Encoding and prediction both condition on scan hidden state H AND input x:
  z_t = W_enc(H_t, x_t)           — "what the model sees" at position t
  z_hat_t = W_pcm(H_t, x_t)      — prediction for z_{t+1}
  surprise_t = z_hat_{t-1} - z_t  — how unexpected this position's encoding is
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SingleColumnPCM(nn.Module):
    """Predictive coding for one cortical column.

    Both encoding and prediction condition on the full context (H, x):
      z_t = W_enc(cat(H_t, x_t))       — current position's encoding
      z_hat_t = W_pcm(cat(H_t, x_t))   — prediction of next position's encoding
      surprise = z_hat_{t-1} - z_t      — prediction error

    Uses a hidden layer for richer encoding/prediction when hidden > 0.
    """

    def __init__(self, D_cc: int, hidden: int = 0):
        super().__init__()
        self.D_cc = D_cc
        in_dim = 2 * D_cc  # concatenation of H and x

        if hidden > 0:
            self.W_enc = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, D_cc),
            )
            self.W_pcm = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, D_cc),
            )
        else:
            self.W_enc = nn.Linear(in_dim, D_cc)
            self.W_pcm = nn.Linear(in_dim, D_cc)

        self.W_gain = nn.Linear(D_cc, D_cc)

        # Zero-init W_gain so gain starts at sigmoid(0)*2 = 1.0
        nn.init.zeros_(self.W_gain.weight)
        nn.init.zeros_(self.W_gain.bias)

        # Learnable gain scale: starts at 2.0 (range [0, 2] at init)
        # Can grow during training to allow stronger modulation
        self.gain_scale = nn.Parameter(torch.tensor(2.0))

    def compute_surprise(self, H: Tensor, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute vector surprise and PCM prediction.

        Both encoding and prediction see the full context (H, x) so that
        surprise captures how unexpected the model's representation is,
        not just how unexpected the raw token is.

        Args:
            H: [BS, T, D_cc] — scan hidden states for this column
            x: [BS, T, D_cc] — column input (pre-scan embedding slice)

        Returns:
            surprise: [BS, T, D_cc] — vector surprise (zero at position 0)
            z_hat: [BS, T, D_cc] — prediction for next position's encoding
            z: [BS, T, D_cc] — encoding at current position
        """
        combined = torch.cat([H, x], dim=-1)  # [BS, T, 2*D_cc]
        z = self.W_enc(combined)
        z_hat = self.W_pcm(combined)
        surprise = torch.zeros_like(z)
        surprise[:, 1:] = z_hat[:, :-1] - z[:, 1:]
        return surprise, z_hat, z

    def apply_gain(self, H: Tensor, surprise: Tensor) -> Tensor:
        """PCM gain modulation: learnable range.

        gain = sigmoid(W_gain(surprise)) * gain_scale
        At init: sigmoid(0) * 2.0 = 1.0 (no modulation).
        The network learns how much surprise should amplify/suppress.
        """
        gain = torch.sigmoid(self.W_gain(surprise)) * self.gain_scale
        return H * gain

    def prediction_loss(self, z_hat: Tensor, z: Tensor) -> Tensor:
        """Auxiliary prediction loss: MSE(z_hat_t, z_{t+1}.detach())."""
        return F.mse_loss(z_hat[:, :-1], z[:, 1:].detach())
