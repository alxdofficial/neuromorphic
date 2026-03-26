"""Predictive Coding Module for v8.

Batched across all C cortical columns for GPU efficiency.
Encoding and prediction both condition on scan hidden state H AND input x:
  z_t = W_enc(norm(cat(H_t, x_t)))   — "what the model sees" at position t
  z_hat_t = W_pcm(norm(cat(H_t, x_t))) — prediction for z_{t+1}
  surprise_t = z_hat_{t-1} - z_t     — how unexpected this position's encoding is

Surprise is passed as a side input to the first upper scan layer, where it
influences both the gate (how much to retain) and the input (what to inject).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BatchedPCM(nn.Module):
    """Predictive coding for all C cortical columns, batched.

    Processes all columns in parallel using batched matrix multiplications
    (C as the batch dim). This replaces the sequential loop over C
    individual SingleColumnPCM modules.

    Both encoding and prediction condition on the full context (H, x),
    with RMSNorm on the concatenated input to prevent magnitude-driven
    gradient explosions.
    """

    def __init__(self, C: int, D_cc: int, hidden: int = 0):
        super().__init__()
        self.C = C
        self.D_cc = D_cc
        in_dim = 2 * D_cc

        # Per-column RMSNorm weights
        self.norm_weight = nn.Parameter(torch.ones(C, in_dim))
        self.norm_eps = 1e-6

        if hidden > 0:
            # Encoding MLP: in_dim → hidden (SiLU) → D_cc, per column
            self.enc_w1 = nn.Parameter(torch.empty(C, in_dim, hidden))
            self.enc_b1 = nn.Parameter(torch.zeros(C, 1, hidden))
            self.enc_w2 = nn.Parameter(torch.empty(C, hidden, D_cc))
            self.enc_b2 = nn.Parameter(torch.zeros(C, 1, D_cc))
            # Prediction MLP: same architecture
            self.pcm_w1 = nn.Parameter(torch.empty(C, in_dim, hidden))
            self.pcm_b1 = nn.Parameter(torch.zeros(C, 1, hidden))
            self.pcm_w2 = nn.Parameter(torch.empty(C, hidden, D_cc))
            self.pcm_b2 = nn.Parameter(torch.zeros(C, 1, D_cc))
            # Kaiming init (matches nn.Linear default)
            for w in [self.enc_w1, self.pcm_w1]:
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            for w in [self.enc_w2, self.pcm_w2]:
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            # Bias init (matches nn.Linear: uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)))
            for b, fan_in in [(self.enc_b1, in_dim), (self.pcm_b1, in_dim),
                              (self.enc_b2, hidden), (self.pcm_b2, hidden)]:
                bound = 1.0 / math.sqrt(fan_in)
                nn.init.uniform_(b, -bound, bound)
            self.has_hidden = True
        else:
            self.enc_w = nn.Parameter(torch.empty(C, in_dim, D_cc))
            self.enc_b = nn.Parameter(torch.zeros(C, 1, D_cc))
            self.pcm_w = nn.Parameter(torch.empty(C, in_dim, D_cc))
            self.pcm_b = nn.Parameter(torch.zeros(C, 1, D_cc))
            for w in [self.enc_w, self.pcm_w]:
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            for b in [self.enc_b, self.pcm_b]:
                bound = 1.0 / math.sqrt(in_dim)
                nn.init.uniform_(b, -bound, bound)
            self.has_hidden = False

    def forward(self, H: Tensor, x: Tensor
                ) -> tuple[Tensor, Tensor, Tensor, Tensor, list[float]]:
        """Batched surprise computation for all columns.

        Args:
            H: [BS, T, C, D_cc] — scan hidden states per column
            x: [BS, T, C, D_cc] — column inputs (pre-scan embedding slices)

        Returns:
            surprise: [BS, T, C, D_cc]
            z_hat: [BS, T, C, D_cc]
            z: [BS, T, C, D_cc]
            pred_loss: scalar — mean prediction loss across columns
            per_cc_pred_loss: list[float] of length C
        """
        BS, T, C, D_cc = H.shape

        # Concatenate H and x: [BS, T, C, 2*D_cc]
        combined = torch.cat([H, x], dim=-1)

        # Per-column RMSNorm: [BS, T, C, 2*D_cc]
        rms = combined.pow(2).mean(dim=-1, keepdim=True).add(self.norm_eps).rsqrt()
        combined = combined * rms * self.norm_weight  # norm_weight broadcasts as [C, 2*D_cc]

        # Reshape for batched matmul: [C, BS*T, in_dim]
        combined_r = combined.permute(2, 0, 1, 3).reshape(C, BS * T, -1)

        if self.has_hidden:
            # Encoding: Linear → SiLU → Linear
            h1 = torch.bmm(combined_r, self.enc_w1) + self.enc_b1  # [C, BS*T, hidden]
            h1 = F.silu(h1)
            z_r = torch.bmm(h1, self.enc_w2) + self.enc_b2  # [C, BS*T, D_cc]
            # Prediction: Linear → SiLU → Linear
            p1 = torch.bmm(combined_r, self.pcm_w1) + self.pcm_b1
            p1 = F.silu(p1)
            z_hat_r = torch.bmm(p1, self.pcm_w2) + self.pcm_b2
        else:
            z_r = torch.bmm(combined_r, self.enc_w) + self.enc_b
            z_hat_r = torch.bmm(combined_r, self.pcm_w) + self.pcm_b

        # Reshape back: [C, BS*T, D_cc] → [BS, T, C, D_cc]
        z = z_r.reshape(C, BS, T, D_cc).permute(1, 2, 0, 3)
        z_hat = z_hat_r.reshape(C, BS, T, D_cc).permute(1, 2, 0, 3)

        # Surprise: z_hat[t-1] - z[t]
        surprise = torch.zeros_like(z)
        surprise[:, 1:] = z_hat[:, :-1] - z[:, 1:]

        # Prediction loss (overall + per-CC for diagnostics)
        pred_err = (z_hat[:, :-1] - z[:, 1:].detach()).pow(2)
        pred_loss = pred_err.mean()
        per_cc_pred_loss = pred_err.mean(dim=(0, 1, 3)).tolist()  # [C]

        return surprise, z_hat, z, pred_loss, per_cc_pred_loss
