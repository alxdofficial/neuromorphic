"""Single-column Predictive Coding Module for v8.

Each cortical column has its own PCM with independent weights.
Operates on [BS, T, D_cc] tensors (not grouped [BS, T, C, D_col]).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SingleColumnPCM(nn.Module):
    """Predictive coding for one cortical column.

    Predicts next token's encoding from current scan state.
    Surprise = z_hat_{t-1} - z_t (vector, D_cc dims).

    Uses a hidden layer for richer encoding/prediction when hidden > 0.
    """

    def __init__(self, D_cc: int, hidden: int = 0):
        super().__init__()
        self.D_cc = D_cc

        if hidden > 0:
            self.W_enc = nn.Sequential(
                nn.Linear(D_cc, hidden), nn.SiLU(), nn.Linear(hidden, D_cc),
            )
            self.W_pcm = nn.Sequential(
                nn.Linear(D_cc, hidden), nn.SiLU(), nn.Linear(hidden, D_cc),
            )
        else:
            self.W_enc = nn.Linear(D_cc, D_cc)
            self.W_pcm = nn.Linear(D_cc, D_cc)

        self.W_gain = nn.Linear(D_cc, D_cc)

        # Zero-init W_gain so gain starts at sigmoid(0)*2 = 1.0
        nn.init.zeros_(self.W_gain.weight)
        nn.init.zeros_(self.W_gain.bias)

        # Learnable gain scale: starts at 2.0 (range [0, 2] at init)
        # Can grow during training to allow stronger modulation
        self.gain_scale = nn.Parameter(torch.tensor(2.0))

    def compute_surprise(self, H: Tensor, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute vector surprise and PCM prediction.

        Args:
            H: [BS, T, D_cc] — scan hidden states for this column
            x: [BS, T, D_cc] — column input (pre-scan embedding slice)

        Returns:
            surprise: [BS, T, D_cc] — vector surprise (zero at position 0)
            z_hat: [BS, T, D_cc] — prediction for next token
            z: [BS, T, D_cc] — encoding of current token
        """
        z = self.W_enc(x)
        z_hat = self.W_pcm(H)
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
