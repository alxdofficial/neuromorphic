"""Two-level neuromod: global trunk + per-column heads.

Simplified from the initial design — the trunk now only consumes the
multi-horizon surprise EMA and its derivative. Everything spatial
(tile-level aggregates of magnitude, variance, traffic) and everything
input-side (input_ctx_ema) has been dropped because per-column heads
already see richer local observables.

Per-column head inputs (everything observable *at column B*):
  - id[B]              [D_id]   — column identity, fixed
  - post[B]            [1]      — receiver activity norm at this tick
  - w_out_proxy[B, :]  [K]      — current outgoing edge weights (proxy for
                                   incoming; see comment in plasticity step)
  - pre_at_nbrs[B, :]  [K]      — activity of B's out-neighbours
  - g (broadcast)      [D_trunk]— global context from surprise trunk

Outputs:
  - η_global  — scalar, global plasticity rate (softplus)
  - η[B]      — per-column plasticity rate (softplus)
  - β[B]      — per-column LTP/LTD threshold (tanh, in [-1, +1])

Precision: forward runs in fp32 under autocast(enabled=False) by caller.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.column_graph.config import ColumnGraphConfig


class Neuromod(nn.Module):
    def __init__(self, cfg: ColumnGraphConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Global trunk — consumes only surprise observables.
        # Input: surprise_ema (K_h) + Δsurprise (K_h)
        K_h = cfg.K_horizons
        trunk_in = 2 * K_h
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, cfg.trunk_hidden),
            nn.GELU(),
            nn.Linear(cfg.trunk_hidden, cfg.D_trunk),
        )

        # Per-column head inputs: g + id + post + w_out + pre_at_nbrs
        col_in = cfg.D_trunk + cfg.D_id + 1 + 2 * cfg.K

        self.head_global = nn.Linear(cfg.D_trunk, 1)
        self.head_eta = nn.Sequential(
            nn.Linear(col_in, cfg.head_hidden),
            nn.GELU(),
            nn.Linear(cfg.head_hidden, 1),
        )
        self.head_beta = nn.Sequential(
            nn.Linear(col_in, cfg.head_hidden),
            nn.GELU(),
            nn.Linear(cfg.head_hidden, 1),
        )

        # Zero-init final heads so at day 0:
        #   η_global = softplus(0) ≈ 0.69
        #   η[c]     = softplus(0) ≈ 0.69
        #   β[c]     = tanh(0) = 0
        # → gentle plasticity rate, no LTP/LTD bias. Good-neighbour default.
        for final in [self.head_global, self.head_eta[-1], self.head_beta[-1]]:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(
        self,
        surprise_ema: torch.Tensor,        # [B, K_h]
        surprise_delta: torch.Tensor,      # [B, K_h]
        col_id_embed: torch.Tensor,        # [N, D_id]
        post: torch.Tensor,                # [B, N]
        w_out_proxy: torch.Tensor,         # [B, N, K]
        pre_at_nbrs: torch.Tensor,         # [B, N, K]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (η_global [B], η [B, N], β [B, N])."""
        B = surprise_ema.shape[0]
        N = col_id_embed.shape[0]

        global_feats = torch.cat([surprise_ema, surprise_delta], dim=-1)  # [B, 2·K_h]
        g = self.trunk(global_feats)                                      # [B, D_trunk]
        eta_global = F.softplus(self.head_global(g).squeeze(-1))           # [B]

        # Broadcast context + combine with per-column local features
        g_exp = g.unsqueeze(1).expand(B, N, g.shape[-1])                  # [B, N, D_trunk]
        id_exp = col_id_embed.unsqueeze(0).expand(B, N, -1)               # [B, N, D_id]
        col_feats = torch.cat(
            [g_exp, id_exp, post.unsqueeze(-1), w_out_proxy, pre_at_nbrs],
            dim=-1,
        )                                                                  # [B, N, col_in]
        eta = F.softplus(self.head_eta(col_feats).squeeze(-1))             # [B, N]
        beta = torch.tanh(self.head_beta(col_feats).squeeze(-1))           # [B, N]

        return eta_global, eta, beta
