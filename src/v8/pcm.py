"""Predictive Coding Module for v9-backprop.

Predicts the next scan hidden state directly (no learned encoding):
  H_hat_{t+1} = pred_MLP(norm(H_t))
  surprise_t  = H_hat_{t-1} - H_t     — how unexpected this position's state is

Surprise is combined with H_mid via a split-point MLP in lm.py before the
upper scan layers process the memory-enriched representation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BatchedPCM(nn.Module):
    """Predictive coding for all C cortical columns, batched.

    Processes all columns in parallel using batched matrix multiplications
    (C as the batch dim). Predicts next scan hidden state directly — no
    encoder, no learned target. The prediction target is the real scan
    output at the next position.
    """

    def __init__(self, C: int, D_cc: int, hidden: int = 256):
        super().__init__()
        if hidden <= 0:
            raise ValueError(f"PCM hidden must be > 0, got {hidden}")
        self.C = C
        self.D_cc = D_cc
        in_dim = D_cc

        # Per-column RMSNorm weights
        self.norm_weight = nn.Parameter(torch.ones(C, in_dim))
        self.norm_eps = 1e-6

        # Prediction MLP: in_dim → hidden (SiLU) → D_cc, per column
        self.pcm_w1 = nn.Parameter(torch.empty(C, in_dim, hidden))
        self.pcm_b1 = nn.Parameter(torch.zeros(C, 1, hidden))
        self.pcm_w2 = nn.Parameter(torch.empty(C, hidden, D_cc))
        self.pcm_b2 = nn.Parameter(torch.zeros(C, 1, D_cc))

        # Kaiming init
        nn.init.kaiming_uniform_(self.pcm_w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pcm_w2, a=math.sqrt(5))
        # Bias init (matches nn.Linear)
        for b, fan_in in [(self.pcm_b1, in_dim), (self.pcm_b2, hidden)]:
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(b, -bound, bound)

    def forward(self, H: Tensor
                ) -> tuple[Tensor, Tensor, Tensor, list[float]]:
        """Batched surprise computation for all columns.

        Args:
            H: [BS, T, C, D_cc] — scan hidden states per column

        Returns:
            surprise: [BS, T, C, D_cc]
            H_hat: [BS, T, C, D_cc] — predicted next hidden state
            pred_loss: scalar — mean prediction loss across columns
            per_cc_pred_loss: list[float] of length C
        """
        BS, T, C, D_cc = H.shape

        # Per-column RMSNorm on H
        rms = H.pow(2).mean(dim=-1, keepdim=True).add(self.norm_eps).rsqrt()
        normed = H * rms * self.norm_weight  # [BS, T, C, D_cc]

        # Reshape for batched matmul: [C, BS*T, D_cc]
        normed_r = normed.permute(2, 0, 1, 3).reshape(C, BS * T, -1)

        # Prediction MLP: Linear → SiLU → Linear
        p1 = torch.bmm(normed_r, self.pcm_w1) + self.pcm_b1  # [C, BS*T, hidden]
        p1 = F.silu(p1)
        H_hat_r = torch.bmm(p1, self.pcm_w2) + self.pcm_b2   # [C, BS*T, D_cc]

        # Reshape back: [C, BS*T, D_cc] → [BS, T, C, D_cc]
        H_hat = H_hat_r.reshape(C, BS, T, D_cc).permute(1, 2, 0, 3)

        # Surprise: H_hat[t-1] - H[t] (how wrong the prediction was)
        surprise = torch.zeros_like(H)
        surprise[:, 1:] = H_hat[:, :-1] - H[:, 1:]

        # Prediction loss: predict H[t+1] from H[t], target is detached
        pred_err = (H_hat[:, :-1] - H[:, 1:].detach()).pow(2)
        pred_loss = pred_err.mean()
        per_cc_pred_loss = pred_err.mean(dim=(0, 1, 3)).tolist()  # [C]

        return surprise, H_hat, pred_loss, per_cc_pred_loss
