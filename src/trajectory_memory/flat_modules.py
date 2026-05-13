"""Flat-bank read/write modules - the architectural ablation for Wave 1 v4.

Drop-in replacements for ReadTrajectoryGenerator / WriteTrajectoryGenerator
that REMOVE the trajectory machinery: no graph walks, no multi-hop step_mlp,
no history_attn, no positional encodings. Each module picks the top-K cells
from the manifold via softmax over all N concepts.

Same external shape contract: return (visited, visited_ids, aux) with
visited: [BS, J, K, D], visited_ids: [BS, J, K]. Mem_inject downstream
sees the same flat_traj shape and works unchanged.

Routing follows the Mixtral / Switch Transformer / GShard conventions for
robust top-K MoE gating:

  1. Renormalize over the top-K picks (Mixtral): gather softmax weights at
     top-K indices, divide by their sum so they form a K-element prob
     distribution. Gradient through these renormed weights is O(1/K), not
     O(1/N), so backward signal survives the K-of-N selection.
  2. Small router init (Switch / GShard convention, std~0.01): keeps the
     routing distribution near-uniform at start so no single cell wins
     early and starves the rest.
  3. Optional Noisy top-K (Shazeer 2017): during training add Gaussian
     noise to logits to encourage exploration of new cells.
  4. Load-balance + z-loss auxiliary losses, same coefficients as the
     trajectory modules.

The point: if the trajectory machinery is doing real work, the original
modules should outperform these flat-bank modules at the same parameter /
state budget. If flat-bank matches, the graph topology + multi-hop
trajectory is overhead and we should switch.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.trajectory_memory.config import TrajMemConfig


# Switch / GShard convention: small router init keeps initial routing
# near-uniform so no single cell wins early.
_ROUTER_INIT_STD = 0.01

# Shazeer 2017 noisy top-K: Gaussian noise stdev added to logits during
# training. Encourages cells to be explored. Disabled at eval.
_NOISE_STD = 0.5


def _init_router_(linear: nn.Linear) -> None:
    """In-place small init for a router linear layer."""
    nn.init.normal_(linear.weight, std=_ROUTER_INIT_STD)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _topk_with_mixtral_renorm(
    scores: Tensor,
    k: int,
    *,
    training_noise: bool = False,
) -> tuple[Tensor, Tensor]:
    """Top-K routing with Mixtral-style renormalization.

    scores: [..., N]
    Returns:
        weights_ste: [..., K]  with forward=ones, backward through renormed top-K softmax.
        top_idx:     [..., K]  int64 selected indices.

    The trick: softmax over N cells produces tiny weights (~1/N), so STE
    through those values gives a vanishing gradient. Gathering only the
    top-K softmax weights and renormalizing them to sum to 1 gives
    K-sized weights (~1/K), much larger gradient.
    """
    if training_noise:
        # Shazeer 2017 noisy gating: Gaussian noise during training only.
        scores = scores + _NOISE_STD * torch.randn_like(scores)
    soft = F.softmax(scores, dim=-1)                          # [..., N]
    top_idx = soft.topk(k, dim=-1).indices                    # [..., K]
    top_soft = torch.gather(soft, dim=-1, index=top_idx)      # [..., K]
    # Renormalize the K picks to sum to 1 (Mixtral pattern).
    top_soft_renormed = top_soft / top_soft.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    # STE: forward = ones (so cross-attn sees full-magnitude states),
    # backward through the renormed top-K weights.
    ones = torch.ones_like(top_soft_renormed)
    weights_ste = ones.detach() + top_soft_renormed - top_soft_renormed.detach()
    return weights_ste, top_idx


class FlatReadModule(nn.Module):
    """Pick top-K cells via softmax-over-all-N + Mixtral renorm.

    Returns the same (visited, visited_ids, aux) shape as the trajectory
    read module so the downstream mem_inject cross-attn works unchanged.
    """

    def __init__(self, cfg: TrajMemConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.D_concept
        d_lm = cfg.d_lm
        # J queries per step (matches the trajectory module's J=4).
        # Each query independently picks K_read concepts -> J*K_read=32 slots.
        self.entry_proj = nn.Sequential(
            nn.Linear(d_lm, D * 2, bias=True),
            nn.GELU(),
            nn.Linear(D * 2, D * cfg.J, bias=True),
        )
        # Small router init (Switch / GShard).
        _init_router_(self.entry_proj[0])
        _init_router_(self.entry_proj[-1])
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

        pooled = prev_window_hiddens.mean(dim=1)              # [BS, d_lm]
        Q = self.entry_proj(pooled).reshape(BS, J, D)         # [BS, J, D]

        ids_normed = manifold.concept_ids_normed              # [N, D]
        Q_normed = F.normalize(Q, dim=-1)
        scores_raw = torch.einsum(
            "bjd,nd->bjn", Q_normed, ids_normed,
        )                                                     # [BS, J, N]
        eff_scale = self.logit_scale_raw.exp().clamp(max=20.0)
        scores = scores_raw * eff_scale

        weights_ste, top_idx = _topk_with_mixtral_renorm(
            scores, K, training_noise=self.training,
        )                                                     # [BS, J, K]

        # Gather states.
        idx_expanded = top_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        prev_states_expanded = prev_states.unsqueeze(1).expand(-1, J, -1, -1)
        visited = torch.gather(prev_states_expanded, dim=2, index=idx_expanded)
        visited = visited * weights_ste.unsqueeze(-1)

        # Aux losses (post-noise, pre-renorm).
        with torch.no_grad():
            soft_clean = F.softmax(scores, dim=-1)
        mean_select = soft_clean.mean(dim=(0, 1))             # [N]
        load_balance = ((mean_select - 1.0 / N) ** 2).sum() * N
        z_loss = (scores_raw ** 2).mean()
        aux = {"load_balance": load_balance, "z_loss": z_loss}

        return visited, top_idx, aux


class FlatWriteModule(nn.Module):
    """Pick top-K cells via softmax-over-all-N + Mixtral renorm, propose
    mutations for each, scatter back.

    Same shape contract as WriteTrajectoryGenerator's forward.
    """

    def __init__(self, cfg: TrajMemConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.D_concept
        d_lm = cfg.d_lm
        J = cfg.J

        self.entry_proj = nn.Sequential(
            nn.Linear(d_lm, D * 2, bias=True),
            nn.GELU(),
            nn.Linear(D * 2, D * J, bias=True),
        )
        _init_router_(self.entry_proj[0])
        _init_router_(self.entry_proj[-1])

        # Mutation GRUCell: input=[pooled_hidden, surprise], hidden=current_state.
        self.mutate = nn.GRUCell(input_size=d_lm + 1, hidden_size=D)

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

        pooled = current_window_hiddens.mean(dim=1)           # [BS, d_lm]
        Q = self.entry_proj(pooled).reshape(BS, J, D)         # [BS, J, D]

        ids_normed = manifold.concept_ids_normed
        Q_normed = F.normalize(Q, dim=-1)
        scores_raw = torch.einsum(
            "bjd,nd->bjn", Q_normed, ids_normed,
        )
        eff_scale = self.logit_scale_raw.exp().clamp(max=20.0)
        scores = scores_raw * eff_scale

        weights_ste, top_idx = _topk_with_mixtral_renorm(
            scores, K, training_noise=self.training,
        )

        # Gather current states at picked indices.
        idx_expanded = top_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        prev_states_expanded = prev_states.unsqueeze(1).expand(-1, J, -1, -1)
        current_states = torch.gather(
            prev_states_expanded, dim=2, index=idx_expanded,
        )                                                     # [BS, J, K, D]

        # Mutate each picked cell via GRU update.
        pooled_bjk = pooled.view(BS, 1, 1, d_lm).expand(BS, J, K, d_lm)
        surprise_bjk = surprise.view(BS, 1, 1, 1).expand(BS, J, K, 1)
        gru_input = torch.cat([pooled_bjk, surprise_bjk], dim=-1)
        gru_input_flat = gru_input.reshape(-1, d_lm + 1)
        current_states_flat = current_states.reshape(-1, D)
        proposed_flat = self.mutate(gru_input_flat, current_states_flat)
        proposed = proposed_flat.reshape(BS, J, K, D)

        proposed = proposed * weights_ste.unsqueeze(-1)

        # Scatter-mean to manifold (same API as trajectory module).
        flat_ids = top_idx.reshape(BS, J * K)
        flat_proposed = proposed.reshape(BS, J * K, D)
        new_states = manifold.write_states(prev_states, flat_ids, flat_proposed)

        with torch.no_grad():
            soft_clean = F.softmax(scores, dim=-1)
        mean_select = soft_clean.mean(dim=(0, 1))
        load_balance = ((mean_select - 1.0 / N) ** 2).sum() * N
        z_loss = (scores_raw ** 2).mean()
        aux = {"load_balance": load_balance, "z_loss": z_loss}

        return new_states, top_idx, proposed, aux
