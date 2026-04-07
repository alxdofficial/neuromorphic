"""Predictive Coding Module — per-token, memory-aware.

Predicts state TRANSITIONS in the LM hidden state, conditioned on both
the current LM state and the previous memory readout. Runs inside the
per-token memory loop (not batched over T).

  delta_hat[t] = pred_MLP(H_mid[t], prev_readout)
  surprise[t] = delta_hat[t-1] - (H_mid[t] - H_mid[t-1])
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BatchedPCM(nn.Module):
    """Per-token predictive coding across C cortical columns.

    Input: H_mid[t] + prev_readout (concatenated per column).
    Output: delta_hat (predicted transition), used to compute surprise.
    """

    def __init__(self, C: int, D_cc: int, hidden: int = 256):
        super().__init__()
        if hidden <= 0:
            raise ValueError(f"PCM hidden must be > 0, got {hidden}")
        self.C = C
        self.D_cc = D_cc
        in_dim = 2 * D_cc  # H_mid_col + prev_readout_col

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

    def predict(self, H_mid_t: Tensor, prev_readout: Tensor) -> Tensor:
        """Predict the next transition delta.

        Args:
            H_mid_t:      [BS, C, D_cc] — current LM hidden state (per column)
            prev_readout:  [BS, C, D_cc] — previous memory readout (per column)

        Returns:
            delta_hat: [BS, C, D_cc] — predicted transition
        """
        BS = H_mid_t.shape[0]
        C, D_cc = self.C, self.D_cc

        # Concatenate LM state + memory readout per column
        pcm_input = torch.cat([H_mid_t, prev_readout], dim=-1)  # [BS, C, 2*D_cc]

        # RMSNorm
        rms = pcm_input.pow(2).mean(dim=-1, keepdim=True).add(self.norm_eps).rsqrt()
        normed = pcm_input * rms * self.norm_weight  # [BS, C, 2*D_cc]

        # Per-column MLP: [C, BS, 2*D_cc] → [C, BS, D_cc]
        dt = normed.dtype
        normed_r = normed.permute(1, 0, 2)  # [C, BS, 2*D_cc]
        p1 = F.silu(torch.bmm(normed_r, self.pcm_w1.to(dt)) + self.pcm_b1.to(dt))
        delta_hat_r = torch.bmm(p1, self.pcm_w2.to(dt)) + self.pcm_b2.to(dt)

        return delta_hat_r.permute(1, 0, 2)  # [BS, C, D_cc]
