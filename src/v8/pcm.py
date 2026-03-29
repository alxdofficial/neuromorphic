"""Predictive Coding Module for v9-backprop — Dynamic Predictive Coding.

Predicts state TRANSITIONS (deltas) rather than raw states:
  delta_hat[t] = pred_MLP(norm(H[t]))       — predicted change
  delta_actual[t] = H[t+1] - H[t]           — actual change
  surprise[t] = delta_hat[t-1] - delta_actual[t]  — transition prediction error

Deltas are small and stationary (bounded by scan residual structure),
so the predictor can track them even as H_mid evolves during training.
This prevents the divergence seen with raw-state prediction.

Inspired by Dynamic Predictive Coding (Jiang & Rao 2023): higher levels
predict the transition dynamics of lower levels, not raw states.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BatchedPCM(nn.Module):
    """Dynamic predictive coding for all C cortical columns, batched.

    Predicts transitions (H[t+1] - H[t]) rather than raw H[t+1].
    Surprise = predicted_delta - actual_delta.
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
        # Predicts delta (transition), not raw state
        self.pcm_w1 = nn.Parameter(torch.empty(C, in_dim, hidden))
        self.pcm_b1 = nn.Parameter(torch.zeros(C, 1, hidden))
        self.pcm_w2 = nn.Parameter(torch.empty(C, hidden, D_cc))
        self.pcm_b2 = nn.Parameter(torch.zeros(C, 1, D_cc))

        # Kaiming init
        nn.init.kaiming_uniform_(self.pcm_w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pcm_w2, a=math.sqrt(5))
        # Small init on w2: deltas are small, so predictions should start small
        with torch.no_grad():
            self.pcm_w2.mul_(0.1)
        # Bias init (matches nn.Linear)
        for b, fan_in in [(self.pcm_b1, in_dim), (self.pcm_b2, hidden)]:
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(b, -bound, bound)

    def forward(self, H: Tensor
                ) -> tuple[Tensor, Tensor, Tensor, list[float]]:
        """Batched transition-prediction surprise for all columns.

        Args:
            H: [BS, T, C, D_cc] — scan hidden states per column

        Returns:
            surprise: [BS, T, C, D_cc] — transition prediction error
            delta_hat: [BS, T, C, D_cc] — predicted transition
            pred_loss: scalar — mean prediction loss across columns
            per_cc_pred_loss: list[float] of length C
        """
        BS, T, C, D_cc = H.shape

        # Per-column RMSNorm on H
        rms = H.pow(2).mean(dim=-1, keepdim=True).add(self.norm_eps).rsqrt()
        normed = H * rms * self.norm_weight  # [BS, T, C, D_cc]

        # Reshape for batched matmul: [C, BS*T, D_cc]
        normed_r = normed.permute(2, 0, 1, 3).reshape(C, BS * T, -1)

        # Prediction MLP: Linear → SiLU → Linear → predicted delta
        p1 = torch.bmm(normed_r, self.pcm_w1) + self.pcm_b1  # [C, BS*T, hidden]
        p1 = F.silu(p1)
        delta_hat_r = torch.bmm(p1, self.pcm_w2) + self.pcm_b2  # [C, BS*T, D_cc]

        # Reshape back: [C, BS*T, D_cc] → [BS, T, C, D_cc]
        delta_hat = delta_hat_r.reshape(C, BS, T, D_cc).permute(1, 2, 0, 3)

        # Actual transitions: H[t+1] - H[t]
        delta_actual = H[:, 1:] - H[:, :-1]  # [BS, T-1, C, D_cc]

        # Surprise: predicted_delta[t-1] - actual_delta[t]
        # At position t, we check: did the transition from t-1→t match
        # what we predicted at t-1?
        surprise = torch.zeros_like(H)
        surprise[:, 1:] = delta_hat[:, :-1] - delta_actual

        # Prediction loss: predict delta[t→t+1] from H[t], target detached
        pred_err = (delta_hat[:, :-1] - delta_actual.detach()).pow(2)
        pred_loss = pred_err.mean()
        per_cc_pred_loss = pred_err.mean(dim=(0, 1, 3)).tolist()  # [C]

        return surprise, delta_hat, pred_loss, per_cc_pred_loss
