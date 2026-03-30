"""v10-gnn Lower Scan — sensory cortex.

Lower scan layers produce H_mid. PCM computes transition-based surprise.
split_mlp combines H_mid + RMSNorm(surprise) → inject signal for memory graph.

No upper scan — the decoder (frontal cortex) handles output prediction.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..model.scan import ScanLayer, RMSNorm


class LowerScan(nn.Module):
    """Lower scan (sensory cortex) + PCM + inject preparation."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        D = config.D_scan
        D_embed = config.D_embed

        # Embedding
        self.embedding = nn.Embedding(config.vocab_size, D_embed)
        if D_embed != D:
            self.proj_up = nn.Linear(D_embed, D)
        else:
            self.proj_up = None
        self.pos_embed = nn.Parameter(torch.randn(config.T, D) * 0.02)

        # Scan layers
        self.layers = nn.ModuleList()
        for _ in range(config.L_scan):
            self.layers.append(
                ScanLayer(D, config.d_inner_scan, config.dropout,
                          n_layers=config.L_scan, glu_output=True)
            )

        # PCM
        if config.pcm_enabled:
            from .pcm import BatchedPCM
            C = config.C
            D_cc = config.D_cc
            self.pcm = BatchedPCM(C, D_cc, hidden=config.pcm_hidden)
            # split_mlp: combines H_mid + surprise → inject signal
            self.split_mlp = nn.Sequential(
                nn.Linear(2 * D, config.d_inner_scan),
                nn.SiLU(),
                nn.Linear(config.d_inner_scan, D),
            )
            # Depth-scaled init
            with torch.no_grad():
                self.split_mlp[2].weight.mul_(
                    1.0 / math.sqrt(2 * config.L_scan))
        else:
            self.pcm = None
            self.split_mlp = None

        # Carries for scan layers
        self._carries = [None] * config.L_scan

    def forward(self, input_ids: Tensor,
                reset_mask: Tensor | None = None,
                ) -> tuple[Tensor, Tensor]:
        """Forward pass: embedding → scan → PCM → inject signal.

        Args:
            input_ids: [BS, T]
            reset_mask: [BS, T] bool — True at document boundaries

        Returns:
            H_inject: [BS, T, D_scan] — combined signal for memory graph inject
            aux_loss: scalar — PCM prediction loss
        """
        BS, T = input_ids.shape
        D = self.config.D_scan

        # Embed
        x = self.embedding(input_ids)
        if self.proj_up is not None:
            x = self.proj_up(x)
        x = x + self.pos_embed[:T]

        # Lower scan layers
        H = x
        for i, layer in enumerate(self.layers):
            carry = self._carries[i]
            H, h_last = layer(H, carry, reset_mask=reset_mask)
            self._carries[i] = h_last

        # PCM: predict transitions, compute surprise
        aux_loss = torch.tensor(0.0, device=H.device)
        if self.pcm is not None:
            C = self.config.C
            D_cc = self.config.D_cc
            H_cols = H.view(BS, T, C, D_cc)
            surprise, delta_hat, pred_loss, per_cc = self.pcm(H_cols)
            aux_loss = pred_loss * self.config.pcm_pred_weight

            # Cache stats for diagnostics
            surp_norms = surprise.detach().norm(dim=-1)
            self._pcm_stats = {
                "surprise_mean": surp_norms.mean().item(),
                "surprise_std": surp_norms.std().item(),
            }

            # RMSNorm on surprise, combine with H_mid via split_mlp
            surprise_flat = surprise.reshape(BS, T, D)
            surp_rms = surprise_flat.pow(2).mean(
                dim=-1, keepdim=True).add(1e-6).rsqrt()
            surprise_normed = surprise_flat * surp_rms

            H_inject = H + self.split_mlp(
                torch.cat([H, surprise_normed], dim=-1))
        else:
            H_inject = H
            self._pcm_stats = None

        return H_inject, aux_loss

    def initialize_carries(self):
        self._carries = [None] * self.config.L_scan

    def detach_carries(self):
        self._carries = [
            h.detach() if h is not None else None for h in self._carries
        ]
