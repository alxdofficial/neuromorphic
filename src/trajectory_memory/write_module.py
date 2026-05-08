"""WriteTrajectoryGenerator — J parallel hops at end of window with persistent
state mutation via functional scatter_mean.

Same shape as ReadTrajectoryGenerator but:
- conditions on `current_window_hiddens` (now available) + `surprise`
- separate parameters (no weight sharing with read)
- includes `mutate_write` MLP — the only mutation function in the system
- proposes new states; commits them via `manifold.write_states` (functional
  scatter_mean returning a NEW tensor, autograd-safe across TBPTT windows)

See docs/plan_trajectory_memory.md §2.3, §3.2, §3.3, §4.8.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.manifold import Manifold
from src.trajectory_memory.read_module import (
    _CrossAttn,
    gumbel_top1_ste,
    make_pos_enc,
    per_j_attn,
)


class WriteTrajectoryGenerator(nn.Module):
    """Generates J parallel write trajectories per window and proposes
    state mutations. The actual scatter_mean lives on Manifold for clean
    autograd semantics.

    Forward:
        current_window_hiddens: [BS, T_window, d_lm]
        surprise:               [BS] window-level scalar (mean per-token CE)
        prev_states:            [BS, N, D_concept]
        manifold:               Manifold

        returns: new_states:    [BS, N, D_concept]   (for next window's read)
                 visited_ids:   [BS, J, K_write] int64
                 proposed:      [BS, J, K_write, D]  proposed states (debug/telemetry)
    """

    def __init__(self, cfg: TrajMemConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.D_concept
        d_lm = cfg.d_lm

        # J head queries — separate from read's, since these specialize
        # for encoding (not retrieval).
        self.head_query = nn.Parameter(torch.empty(cfg.J, D))
        nn.init.normal_(self.head_query, std=0.02)

        # Entry-point MLP: takes pooled current_hiddens (d_lm) + head_query (D)
        # + surprise (1) → query in id-space (D).
        self.entry_mlp = nn.Sequential(
            nn.Linear(d_lm + D + 1, D, bias=True),
            nn.GELU(),
            nn.Linear(D, D, bias=True),
        )

        # Per-hop attention.
        d_attn = D
        self.history_attn = _CrossAttn(d_q=D, d_kv=D, d_attn=d_attn)
        self.cross_attn = _CrossAttn(d_q=D, d_kv=d_lm, d_attn=d_attn)

        # Per-hop step MLP — same inputs as read plus surprise scalar.
        self.step_mlp = nn.Sequential(
            nn.Linear(D * 3 + 1, D, bias=True),
            nn.GELU(),
            nn.Linear(D, D, bias=True),
        )

        # mutate_write MLP — produces the proposed new state.
        # `new = state + mutation_init_scale * MLP(state, cross_attn, history_attn, surprise)`
        # mutation_init_scale keeps writes small at init so the manifold
        # doesn't drift wildly in the first few training steps.
        self.mutate_mlp = nn.Sequential(
            nn.Linear(D * 3 + 1, D, bias=True),
            nn.GELU(),
            nn.Linear(D, D, bias=True),
        )
        self.mutation_scale = cfg.mutation_init_scale

        # Positional encoding for visited list inside trajectory.
        # Size K_write: indexed up to pos_enc[:K_write] in the longest hop.
        pe = make_pos_enc(cfg.K_write, D) * cfg.pos_enc_scale
        self.register_buffer("pos_enc", pe, persistent=False)

    def forward(
        self,
        current_window_hiddens: Tensor,
        surprise: Tensor,
        prev_states: Tensor,
        manifold: Manifold,
        *,
        hard: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run write trajectories and return new manifold state.

        Args:
            current_window_hiddens: [BS, T_window, d_lm]
            surprise:               [BS] window-level scalar
            prev_states:            [BS, N, D_concept]
            manifold:               Manifold instance
            hard:                   Gumbel-STE if True, argmax if False

        Returns:
            new_states:  [BS, N, D] — manifold state after scatter_mean
            visited_ids: [BS, J, K_write] int64
            proposed:    [BS, J, K_write, D] proposed new states (for telemetry)
        """
        cfg = self.cfg
        BS, T, d_lm = current_window_hiddens.shape
        D = cfg.D_concept
        J, K = cfg.J, cfg.K_write
        N = cfg.N
        assert prev_states.shape == (BS, N, D)
        assert surprise.shape == (BS,), f"surprise shape {surprise.shape} != ({BS},)"

        surprise_b = surprise.view(BS, 1, 1)                      # [BS, 1, 1]
        surprise_bj = surprise.view(BS, 1).expand(BS, J).unsqueeze(-1)  # [BS, J, 1]

        # ── 1. Entry-point selection ──────────────────────────────────
        pooled = current_window_hiddens.mean(dim=1)               # [BS, d_lm]
        pooled_j = pooled.unsqueeze(1).expand(BS, J, d_lm)
        hq = self.head_query.unsqueeze(0).expand(BS, J, D)
        entry_input = torch.cat([pooled_j, hq, surprise_bj], dim=-1)
        Q_entry = self.entry_mlp(entry_input)                     # [BS, J, D]

        ids = manifold.concept_ids
        entry_logits = torch.einsum("bjd,nd->bjn", Q_entry, ids)
        _, entry_idx = gumbel_top1_ste(entry_logits, cfg.gumbel_tau, hard=hard)

        # ── 2. K_write hops + mutation proposals ─────────────────────
        current = entry_idx                                       # [BS, J]
        current_state = manifold.gather_states(prev_states, current)

        proposed_new_list: list[Tensor] = []         # mutated states (used for both history_attn and scatter)
        visited_id_list: list[Tensor] = []

        for t in range(K):
            # history_attn over previously-proposed states (note: we use the
            # MUTATED proposals, because that's what carries the trajectory's
            # own etching history within a single trajectory; raw states
            # would lose the in-trajectory accumulation).
            if t == 0:
                # No history yet at hop 0; use a zero placeholder so
                # history_attn returns something well-defined.
                hist_kv = current_state.unsqueeze(2)              # [BS, J, 1, D]
                # Pos enc for the placeholder doesn't matter much; use pe[0].
                pos = self.pos_enc[:1].unsqueeze(0).unsqueeze(1)
                hist_kv = hist_kv + pos
            else:
                hist_kv = torch.stack(proposed_new_list, dim=2)   # [BS, J, t, D]
                pos = self.pos_enc[:t].unsqueeze(0).unsqueeze(1)
                hist_kv = hist_kv + pos
            history_attn_out = per_j_attn(
                self.history_attn, current_state, hist_kv,
            )

            prev_hid_bjt = current_window_hiddens.unsqueeze(1).expand(
                BS, J, T, d_lm,
            )
            cross_attn_out = per_j_attn(
                self.cross_attn, current_state, prev_hid_bjt,
            )

            # Step query for next-hop selection.
            step_input = torch.cat(
                [current_state, history_attn_out, cross_attn_out, surprise_bj],
                dim=-1,
            )
            Q_t = self.step_mlp(step_input)                       # [BS, J, D]

            nbr_idx = manifold.get_neighbor_indices(current)
            nbr_ids = manifold.concept_ids[nbr_idx]               # [BS, J, K_max, D]
            nbr_logits = torch.einsum("bjd,bjkd->bjk", Q_t, nbr_ids)
            _, next_local = gumbel_top1_ste(
                nbr_logits, cfg.gumbel_tau, hard=hard,
            )
            next_global = torch.gather(
                nbr_idx, dim=2, index=next_local.unsqueeze(-1),
            ).squeeze(-1)

            # mutate_write — proposed new state for this concept.
            mutate_input = torch.cat(
                [current_state, history_attn_out, cross_attn_out, surprise_bj],
                dim=-1,
            )
            delta = self.mutate_mlp(mutate_input)                 # [BS, J, D]
            new_state = current_state + self.mutation_scale * delta  # [BS, J, D]

            visited_id_list.append(current)
            proposed_new_list.append(new_state)

            # Move to next concept, fetch its raw (pre-mutation) state.
            current = next_global
            current_state = manifold.gather_states(prev_states, current)

        proposed = torch.stack(proposed_new_list, dim=2)          # [BS, J, K, D]
        visited_ids = torch.stack(visited_id_list, dim=2)         # [BS, J, K]

        # ── 3. Functional scatter_mean — returns new tensor ──────────
        flat_ids = visited_ids.reshape(BS, J * K)                 # [BS, J*K]
        flat_proposed = proposed.reshape(BS, J * K, D)            # [BS, J*K, D]
        new_states = manifold.write_states(prev_states, flat_ids, flat_proposed)

        return new_states, visited_ids, proposed
