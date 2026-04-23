"""Two-level neuromod: global trunk + per-column heads.

Global trunk fires once per plasticity step. Per-column heads fire in
parallel, one per column, each seeing broadcast trunk context + its
own local observables. Output: η[c], β[c] per column (plus η_global
scalar).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.column_graph.config import ColumnGraphConfig


def _rmsnorm(dim: int) -> nn.Module:
    from src.column_graph.readout import _FallbackRMSNorm
    return _FallbackRMSNorm(dim)


class Neuromod(nn.Module):
    """Global trunk + per-column heads producing plasticity gates.

    Precision: all forward computation forced fp32 by autocast(enabled=False)
    at the call site. The module's parameters are fp32 master weights.
    """

    def __init__(self, cfg: ColumnGraphConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Global trunk — consumes aggregate observables.
        # Input: surprise_ema (K_h) + Δsurprise (K_h) + input_ctx_ema (D_s)
        #      + per-plane mag_ema + var_ema + traffic (3 * num_tiles)
        K_h = cfg.K_horizons
        D_ctx = cfg.D_s
        num_tiles = cfg.num_tiles
        trunk_in = 2 * K_h + D_ctx + 3 * num_tiles
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, cfg.trunk_hidden),
            nn.GELU(),
            nn.Linear(cfg.trunk_hidden, cfg.D_trunk),
        )

        # η_global: single scalar from the trunk
        self.head_global = nn.Linear(cfg.D_trunk, 1)

        # Per-column heads: input is trunk-context + id[c] + post[c] (1)
        # + w_in[c] (K) + pre_at_in_nbrs[c] (K)  =  D_trunk + D_id + 1 + 2K
        col_in = cfg.D_trunk + cfg.D_id + 1 + 2 * cfg.K
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
        # Zero-init the final layer so η≈softplus(0)≈0.69 and β≈tanh(0)=0
        # at day 0 — gentle plasticity at init.
        nn.init.zeros_(self.head_eta[-1].weight)
        nn.init.zeros_(self.head_eta[-1].bias)
        nn.init.zeros_(self.head_beta[-1].weight)
        nn.init.zeros_(self.head_beta[-1].bias)
        nn.init.zeros_(self.head_global.weight)
        nn.init.zeros_(self.head_global.bias)

    def forward(
        self,
        surprise_ema: torch.Tensor,        # [B, K_h]
        surprise_delta: torch.Tensor,      # [B, K_h]
        input_ctx_ema: torch.Tensor,       # [B, D_ctx]
        mag_ema: torch.Tensor,             # [B, num_tiles]
        var_ema: torch.Tensor,             # [B, num_tiles]
        traffic: torch.Tensor,             # [B, num_tiles]
        # Per-column local features
        col_id_embed: torch.Tensor,        # [N, D_id]  — fixed column identities
        post: torch.Tensor,                # [B, N]
        w_in: torch.Tensor,                # [B, N, K]
        pre_at_in_nbrs: torch.Tensor,      # [B, N, K]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (η_global, η[B, N], β[B, N])."""
        B = surprise_ema.shape[0]
        N = col_id_embed.shape[0]

        global_feats = torch.cat(
            [surprise_ema, surprise_delta, input_ctx_ema, mag_ema, var_ema, traffic],
            dim=-1,
        )                                                    # [B, trunk_in]
        g = self.trunk(global_feats)                         # [B, D_trunk]
        eta_global = F.softplus(self.head_global(g).squeeze(-1))  # [B]
        # scalar per batch item — plastic state is averaged over batch later

        # Per-column heads: broadcast g across N columns
        g_exp = g.unsqueeze(1).expand(B, N, g.shape[-1])     # [B, N, D_trunk]
        id_exp = col_id_embed.unsqueeze(0).expand(B, N, -1)  # [B, N, D_id]
        col_feats = torch.cat(
            [g_exp, id_exp, post.unsqueeze(-1), w_in, pre_at_in_nbrs],
            dim=-1,
        )                                                    # [B, N, col_in]
        eta = F.softplus(self.head_eta(col_feats).squeeze(-1))  # [B, N]
        beta = torch.tanh(self.head_beta(col_feats).squeeze(-1))  # [B, N]

        return eta_global, eta, beta
