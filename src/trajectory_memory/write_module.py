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
import torch.nn.functional as F
from torch import Tensor

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.manifold import Manifold
from src.trajectory_memory.read_module import (
    _CrossAttn,
    EntryProjector,
    gumbel_top1_ste,
    softmax_top1_ste,
    make_pos_enc,
    per_j_attn,
    routing_aux_losses,
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

    def __init__(
        self, cfg: TrajMemConfig, *, entry_proj: "EntryProjector | None" = None,
    ):
        super().__init__()
        self.cfg = cfg
        D = cfg.D_concept
        d_lm = cfg.d_lm

        # Entry projection: shared with the read module (Hopfield-tied keys)
        # when constructed via IntegratedLM. Surprise is intentionally NOT
        # an input — it controls write strength downstream (step_mlp,
        # mutate_mlp), not address. Routing surprise into the address would
        # re-introduce read/write divergence and starve write of gradient.
        self.entry_proj = entry_proj if entry_proj is not None else EntryProjector(cfg)

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

        # State update — `nn.GRUCell` replaces the previous
        # mutate_mlp + decay_gate combo. GRUCell does:
        #
        #     z       = sigmoid(W_z · [step_input, current_state])   # update gate
        #     r       = sigmoid(W_r · [step_input, current_state])   # reset gate
        #     candidate = tanh(W_h · [step_input, r * current_state])  # bounded
        #     new     = (1 - z) * current_state + z * candidate
        #
        # The previous additive-then-blend formulation had an unbounded
        # candidate (`mutation_scale * mlp(...)`) — over many writes the
        # convex combination still accumulated drift since `current_state`
        # could grow unboundedly. GRUCell's tanh bound on the candidate
        # makes `||current_state|| <= 1` by induction (convex combination
        # of bounded with bounded). Direct fix for the step-14564
        # drift bug in the Wave 1 run.
        #
        # Input size: same as before — concat[current_state(D),
        # history_attn(D), cross_attn(D), surprise(1)] = 3D + 1.
        self.state_update = nn.GRUCell(input_size=D * 3 + 1, hidden_size=D)

        # Positional encoding for visited list inside trajectory.
        # Size K_write: indexed up to pos_enc[:K_write] in the longest hop.
        pe = make_pos_enc(cfg.K_write, D) * cfg.pos_enc_scale
        self.register_buffer("pos_enc", pe, persistent=False)

        # Learnable logit scale (CLIP-style). Same rationale as read_module:
        # cosine routing produces logits in [-1, 1]; need scaling for the
        # softmax to be selective. See read_module for full reasoning.
        self.logit_scale_raw = nn.Parameter(
            torch.tensor(float(cfg.logit_scale_init))
        )

    def forward(
        self,
        current_window_hiddens: Tensor,
        surprise: Tensor,
        prev_states: Tensor,
        manifold: Manifold,
        *,
        hard: bool = True,
        tau: Tensor | float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Run write trajectories and return new manifold state.

        Args:
            current_window_hiddens: [BS, T_window, d_lm]
            surprise:               [BS] window-level scalar
            prev_states:            [BS, N, D_concept]
            manifold:               Manifold instance
            hard:                   Gumbel-STE if True, argmax if False
            tau:                    Gumbel temperature override; None → cfg.gumbel_tau.

        Returns:
            new_states:  [BS, N, D] — manifold state after scatter_mean
            visited_ids: [BS, J, K_write] int64
            proposed:    [BS, J, K_write, D] proposed new states (for telemetry)
            aux:         {'load_balance': scalar, 'z_loss': scalar} routing
                         aux losses summed over (entry + K_write) decisions.
        """
        cfg = self.cfg
        BS, T, d_lm = current_window_hiddens.shape
        D = cfg.D_concept
        J, K = cfg.J, cfg.K_write
        N = cfg.N
        assert prev_states.shape == (BS, N, D)
        assert surprise.shape == (BS,), f"surprise shape {surprise.shape} != ({BS},)"
        # Tensor pass-through (same rationale as read_module — avoids the
        # per-step Dynamo recompile triggered by Python-scalar specialization
        # of an exp-decay-schedule tau).
        tau_eff = cfg.gumbel_tau if tau is None else tau

        surprise_bj = surprise.view(BS, 1).expand(BS, J).unsqueeze(-1)  # [BS, J, 1]

        # ── 1. Entry-point selection ──────────────────────────────────
        # Hopfield-tied: shares EntryProjector with read_module. Pool of
        # current window hiddens here matches what next window's read will
        # pool as its prev_window_hiddens, so write deposits at the slot
        # the next read will retrieve from.
        pooled = current_window_hiddens.mean(dim=1)               # [BS, d_lm]
        Q_entry = self.entry_proj(pooled)                         # [BS, J, D]

        # Precompute cross_attn K,V on current_window_hiddens — identical
        # for all j and all K_write hops. Avoids per-hop re-projection and
        # the J-broadcast materialization.
        cross_K, cross_V = self.cross_attn.precompute_kv(current_window_hiddens)

        # Cosine routing for entry (was raw dot product; this lets the same
        # learnable logit_scale govern signal-to-noise as in read_module).
        ids_normed = manifold.concept_ids_normed
        entry_logits_raw = torch.einsum(
            "bjd,nd->bjn", F.normalize(Q_entry, dim=-1), ids_normed,
        )                                                         # ∈ [-1, 1]
        eff_scale = self.logit_scale_raw.exp().clamp(max=20.0)
        entry_logits = entry_logits_raw * eff_scale
        entry_one_hot, entry_idx = softmax_top1_ste(
            entry_logits, hard=hard,
        )                                                         # [BS, J, N], [BS, J]
        # Routing aux losses — z_loss on the unscaled cosine to avoid
        # fighting logit_scale_raw growth.
        _aux_e = routing_aux_losses(
            entry_logits, entry_one_hot, z_loss_logits=entry_logits_raw,
        )
        aux_lb_total = _aux_e["load_balance"]
        aux_z_total = _aux_e["z_loss"]

        # ── 2. K_write hops + mutation proposals ─────────────────────
        current = entry_idx                                       # [BS, J] int
        # Differentiable initial-state gather via entry one-hot.
        current_state = torch.einsum(
            "bjn,bnd->bjd", entry_one_hot, prev_states,
        )                                                         # [BS, J, D]

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

            # Shared-KV fast path: no J broadcast, K/V already projected.
            cross_attn_out = self.cross_attn.forward_with_kv(
                current_state, cross_K, cross_V,
            )

            # Step query for next-hop selection.
            step_input = torch.cat(
                [current_state, history_attn_out, cross_attn_out, surprise_bj],
                dim=-1,
            )
            Q_t = self.step_mlp(step_input)                       # [BS, J, D]

            nbr_idx = manifold.get_neighbor_indices(current)
            # L2-normalized routing: cosine similarity, bounded ∈ [-1, 1].
            # Decouples next-hop selection from concept_ids magnitude
            # so the Gumbel τ schedule operates on a stable scale.
            nbr_ids = manifold.concept_ids_normed[nbr_idx]        # [BS, J, K_max, D]
            nbr_logits_raw = torch.einsum(
                "bjd,bjkd->bjk", F.normalize(Q_t, dim=-1), nbr_ids,
            )                                                     # ∈ [-1, 1]
            nbr_logits = nbr_logits_raw * eff_scale
            nbr_one_hot, next_local = softmax_top1_ste(
                nbr_logits, hard=hard,
            )                                                     # [BS, J, K_max], [BS, J]
            _aux_h = routing_aux_losses(
                nbr_logits, nbr_one_hot, z_loss_logits=nbr_logits_raw,
            )
            aux_lb_total = aux_lb_total + _aux_h["load_balance"]
            aux_z_total = aux_z_total + _aux_h["z_loss"]
            next_global = torch.gather(
                nbr_idx, dim=2, index=next_local.unsqueeze(-1),
            ).squeeze(-1)                                         # [BS, J]

            # mutate_write — GRUCell state update.
            # GRUCell expects 2D tensors (batch, features), so flatten
            # [BS, J, ...] → [BS*J, ...] and reshape back after.
            BS_, J_, D_ = current_state.shape
            new_state = self.state_update(
                step_input.reshape(BS_ * J_, -1),
                current_state.reshape(BS_ * J_, D_),
            ).view(BS_, J_, D_)

            visited_id_list.append(current)
            proposed_new_list.append(new_state)

            # Move to next concept; differentiable soft gather over the
            # neighbors of `current` by the Gumbel-STE one-hot. Gradient
            # flows back to nbr_logits → Q_t → step_mlp.
            current = next_global                                 # int, for next neighbor lookup
            nbr_states_full = manifold.gather_states(
                prev_states, nbr_idx,
            )                                                     # [BS, J, K_max, D]
            current_state = torch.einsum(
                "bjk,bjkd->bjd", nbr_one_hot, nbr_states_full,
            )                                                     # [BS, J, D]

        proposed = torch.stack(proposed_new_list, dim=2)          # [BS, J, K, D]
        visited_ids = torch.stack(visited_id_list, dim=2)         # [BS, J, K]

        # ── 3. Functional scatter_mean — returns new tensor ──────────
        flat_ids = visited_ids.reshape(BS, J * K)                 # [BS, J*K]
        flat_proposed = proposed.reshape(BS, J * K, D)            # [BS, J*K, D]
        new_states = manifold.write_states(prev_states, flat_ids, flat_proposed)

        n_routes = 1 + K
        aux = {
            "load_balance": aux_lb_total / n_routes,
            "z_loss": aux_z_total / n_routes,
        }
        return new_states, visited_ids, proposed, aux
