"""Flat-bank read/write modules — the architectural ablation for Wave 1 v4.

Drop-in replacements for ReadTrajectoryGenerator / WriteTrajectoryGenerator
that REMOVE the trajectory machinery: no graph walks, no multi-hop step_mlp,
no history_attn, no positional encodings. Each module just picks the top-K
cells from the manifold via a single softmax over all N concepts.

Same external shape contract — return `(visited, visited_ids, aux)` with
`visited: [BS, J, K, D]`, `visited_ids: [BS, J, K]`. Mem_inject downstream
sees the same flat_traj shape and works unchanged.

The point: if the trajectory machinery is doing real work, the original
modules should outperform these flat-bank modules at the same parameter /
state budget. If flat-bank matches, the graph topology + multi-hop
trajectory is overhead and we should switch.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.trajectory_memory.config import TrajMemConfig


class FlatReadModule(nn.Module):
    """Pick top-K cells via softmax over all N concepts, gather their states.

    Returns the same `(visited, visited_ids, aux)` shape as the trajectory
    read module, so the downstream mem_inject cross-attn works unchanged.
    """

    def __init__(self, cfg: TrajMemConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.D_concept
        d_lm = cfg.d_lm
        # J queries per step (matches the trajectory module's J=4 trajectories).
        # Each query independently picks K_read concepts → J*K_read=32 total slots.
        self.entry_proj = nn.Sequential(
            nn.Linear(d_lm, D * 2, bias=True),
            nn.GELU(),
            nn.Linear(D * 2, D * cfg.J, bias=True),
        )
        # CLIP-style learnable scale (matches trajectory module).
        self.logit_scale_raw = nn.Parameter(
            torch.tensor(float(cfg.logit_scale_init))
        )

    def forward(
        self,
        prev_window_hiddens: Tensor,
        prev_states: Tensor,
        manifold: Any,
        *,
        hard: bool = True,
        tau: Any = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        cfg = self.cfg
        BS, T, d_lm = prev_window_hiddens.shape
        D = cfg.D_concept
        J = cfg.J
        K = cfg.K_read
        N = cfg.N

        # Pool window hiddens; project to J queries.
        pooled = prev_window_hiddens.mean(dim=1)              # [BS, d_lm]
        Q = self.entry_proj(pooled).reshape(BS, J, D)         # [BS, J, D]

        # Cosine vs ALL concepts (no neighbor restriction — that's the diff).
        ids_normed = manifold.concept_ids_normed              # [N, D]
        Q_normed = F.normalize(Q, dim=-1)
        scores_raw = torch.einsum(
            "bjd,nd->bjn", Q_normed, ids_normed,
        )                                                     # [BS, J, N] ∈ [-1,1]
        eff_scale = self.logit_scale_raw.exp().clamp(max=20.0)
        scores = scores_raw * eff_scale                       # [BS, J, N]

        # Softmax for differentiable weighting.
        soft = F.softmax(scores, dim=-1)                      # [BS, J, N]

        # Hard top-K selection.
        top_vals, top_idx = soft.topk(K, dim=-1)              # [BS, J, K]

        # STE for the K-hot selection: forward weight = 1 (so cross-attn
        # downstream sees full-magnitude states), backward gradient through
        # soft weights.
        ones = torch.ones_like(top_vals)
        weights_ste = ones.detach() + top_vals - top_vals.detach()

        # Gather states at top_idx.
        idx_expanded = top_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # [BS, J, K, D]
        prev_states_expanded = prev_states.unsqueeze(1).expand(-1, J, -1, -1)  # [BS, J, N, D]
        visited = torch.gather(prev_states_expanded, dim=2, index=idx_expanded)  # [BS, J, K, D]

        # Apply STE weights (broadcast over D).
        visited = visited * weights_ste.unsqueeze(-1)

        # Aux losses: load-balance over selected concepts (encourage spread),
        # z-loss on raw cosine to prevent logit explosion. Mirror the
        # trajectory module's pattern.
        mean_select_per_concept = soft.mean(dim=(0, 1))       # [N]
        target_uniform = 1.0 / N
        load_balance = ((mean_select_per_concept - target_uniform) ** 2).sum() * N
        z_loss = (scores_raw ** 2).mean()
        aux = {"load_balance": load_balance, "z_loss": z_loss}

        return visited, top_idx, aux


class FlatWriteModule(nn.Module):
    """Pick top-K cells via softmax over all N concepts, propose mutations
    for each, scatter-update the manifold state.

    Same shape contract as WriteTrajectoryGenerator's forward.
    """

    def __init__(self, cfg: TrajMemConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.D_concept
        d_lm = cfg.d_lm
        J = cfg.J

        # Query projection — bigger than read's because write needs more
        # capacity to encode "what to write" (not just "what to retrieve").
        self.entry_proj = nn.Sequential(
            nn.Linear(d_lm, D * 2, bias=True),
            nn.GELU(),
            nn.Linear(D * 2, D * J, bias=True),
        )

        # Mutation generator — GRUCell: input is [pooled_hidden, surprise]
        # (d_lm+1), hidden is current_state (D). Output: new_state.
        self.mutate = nn.GRUCell(input_size=d_lm + 1, hidden_size=D)

        # Learnable logit scale.
        self.logit_scale_raw = nn.Parameter(
            torch.tensor(float(cfg.logit_scale_init))
        )

    def forward(
        self,
        current_window_hiddens: Tensor,
        surprise: Tensor,
        prev_states: Tensor,
        manifold: Any,
        *,
        hard: bool = True,
        tau: Any = None,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        cfg = self.cfg
        BS, T, d_lm = current_window_hiddens.shape
        D = cfg.D_concept
        J = cfg.J
        K = cfg.K_write
        N = cfg.N
        assert prev_states.shape == (BS, N, D)

        # Pool and project to J queries.
        pooled = current_window_hiddens.mean(dim=1)           # [BS, d_lm]
        Q = self.entry_proj(pooled).reshape(BS, J, D)         # [BS, J, D]

        # Cosine vs all concepts.
        ids_normed = manifold.concept_ids_normed              # [N, D]
        Q_normed = F.normalize(Q, dim=-1)
        scores_raw = torch.einsum(
            "bjd,nd->bjn", Q_normed, ids_normed,
        )                                                     # [BS, J, N]
        eff_scale = self.logit_scale_raw.exp().clamp(max=20.0)
        scores = scores_raw * eff_scale

        soft = F.softmax(scores, dim=-1)                      # [BS, J, N]
        top_vals, top_idx = soft.topk(K, dim=-1)              # [BS, J, K]
        ones = torch.ones_like(top_vals)
        weights_ste = ones.detach() + top_vals - top_vals.detach()

        # Gather current states at picked indices.
        idx_expanded = top_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        prev_states_expanded = prev_states.unsqueeze(1).expand(-1, J, -1, -1)
        current_states = torch.gather(
            prev_states_expanded, dim=2, index=idx_expanded,
        )                                                     # [BS, J, K, D]

        # Mutate: for each picked cell, run GRU update.
        # GRU input: [pooled_hidden, surprise], state: current_state.
        pooled_bjk = pooled.view(BS, 1, 1, d_lm).expand(BS, J, K, d_lm)
        surprise_bjk = surprise.view(BS, 1, 1, 1).expand(BS, J, K, 1)
        gru_input = torch.cat([pooled_bjk, surprise_bjk], dim=-1)  # [BS, J, K, d_lm+1]
        gru_input_flat = gru_input.reshape(-1, d_lm + 1)
        current_states_flat = current_states.reshape(-1, D)
        proposed_flat = self.mutate(gru_input_flat, current_states_flat)  # [BS·J·K, D]
        proposed = proposed_flat.reshape(BS, J, K, D)

        # Apply STE weights.
        proposed = proposed * weights_ste.unsqueeze(-1)

        # Scatter-mean write to manifold (use the same helper as the
        # trajectory module — manifold.write_states with flat-indexed args).
        flat_ids = top_idx.reshape(BS, J * K)                 # [BS, J*K]
        flat_proposed = proposed.reshape(BS, J * K, D)        # [BS, J*K, D]
        new_states = manifold.write_states(prev_states, flat_ids, flat_proposed)

        # Aux losses.
        mean_select_per_concept = soft.mean(dim=(0, 1))
        target_uniform = 1.0 / N
        load_balance = ((mean_select_per_concept - target_uniform) ** 2).sum() * N
        z_loss = (scores_raw ** 2).mean()
        aux = {"load_balance": load_balance, "z_loss": z_loss}

        return new_states, top_idx, proposed, aux
