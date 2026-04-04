"""Predictive Coding Module — Dynamic Predictive Coding.

Predicts state TRANSITIONS (deltas) rather than raw states:
  delta_hat[t] = pred_MLP(norm(H[t]))       — predicted change
  delta_actual[t] = H[t+1] - H[t]           — actual change
  surprise[t] = delta_hat[t-1] - delta_actual[t]  — transition prediction error

Inspired by Dynamic Predictive Coding (Jiang & Rao 2023).
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

        self.norm_weight = nn.Parameter(torch.ones(C, in_dim))
        self.norm_eps = 1e-6

        self.pcm_w1 = nn.Parameter(torch.empty(C, in_dim, hidden))
        self.pcm_b1 = nn.Parameter(torch.zeros(C, 1, hidden))
        self.pcm_w2 = nn.Parameter(torch.empty(C, hidden, D_cc))
        self.pcm_b2 = nn.Parameter(torch.zeros(C, 1, D_cc))

        nn.init.kaiming_uniform_(self.pcm_w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pcm_w2, a=math.sqrt(5))
        with torch.no_grad():
            self.pcm_w2.mul_(0.1)
        for b, fan_in in [(self.pcm_b1, in_dim), (self.pcm_b2, hidden)]:
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(b, -bound, bound)

    def forward(self, H: Tensor
                ) -> tuple[Tensor, Tensor, Tensor, list[float]]:
        """
        Args:
            H: [BS, T, C, D_cc]
        Returns:
            surprise, delta_hat, pred_loss, per_cc_pred_loss
        """
        BS, T, C, D_cc = H.shape

        rms = H.pow(2).mean(dim=-1, keepdim=True).add(self.norm_eps).rsqrt()
        normed = H * rms * self.norm_weight

        normed_r = normed.permute(2, 0, 1, 3).reshape(C, BS * T, -1)

        p1 = torch.bmm(normed_r, self.pcm_w1) + self.pcm_b1
        p1 = F.silu(p1)
        delta_hat_r = torch.bmm(p1, self.pcm_w2) + self.pcm_b2

        delta_hat = delta_hat_r.reshape(C, BS, T, D_cc).permute(1, 2, 0, 3)

        delta_actual = H[:, 1:] - H[:, :-1]

        surprise = torch.zeros_like(H)
        surprise[:, 1:] = delta_hat[:, :-1] - delta_actual

        pred_err = (delta_hat[:, :-1] - delta_actual.detach()).pow(2)
        pred_loss = pred_err.mean()
        per_cc_pred_loss = pred_err.mean(dim=(0, 1, 3)).tolist()

        return surprise, delta_hat, pred_loss, per_cc_pred_loss
