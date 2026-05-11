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
    EntryProjector,
    gumbel_top1_ste,
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

        # mutate_write MLP — produces the candidate write.
        # Original: `new = state + mutation_init_scale * mutate_mlp(...)`
        # Replaced with gated convex combination (see decay_gate below):
        #   `new = α·state + (1−α)·(mutation_scale * mutate_mlp(...))`
        # This bounds state magnitude — the original additive update had
        # NO bound on accumulated state and collapsed routing at step ~900
        # under TXL-style continuous training (see commit msg for fix #4
        # follow-up). Mutation_scale on the candidate keeps the early-
        # training behavior close to the original (small writes at init)
        # while α handles forgetting.
        self.mutate_mlp = nn.Sequential(
            nn.Linear(D * 3 + 1, D, bias=True),
            nn.GELU(),
            nn.Linear(D, D, bias=True),
        )
        self.mutation_scale = cfg.mutation_init_scale

        # Learnable decay gate — per-trajectory-hop sigmoid α ∈ [0, 1].
        # α = 1 → fully retain old state; α = 0 → fully overwrite with
        # candidate. The bias is initialized to reproduce the
        # pre-decay-gate update rate: legacy was `new = old + s·mlp(...)`
        # which is equivalent (in steady-state magnitude terms) to
        # `new = α_target · old + (1-α_target) · candidate` with
        #     α_target = 1 / (1 + s).
        # Inverting the sigmoid gives the right bias:
        #     bias = logit(α_target) = log(α_target / (1 - α_target))
        #          = log(1/s) = -log(s).
        # For s = mutation_init_scale = 0.1 this gives bias = log(10) ≈ 2.30.
        # Decay-gate output sits near α_target at init; over training,
        # the model can lower α (forget more on surprising windows) or
        # raise it (preserve state where doing so helps the LM loss).
        self.decay_gate = nn.Sequential(
            nn.Linear(D * 3 + 1, D, bias=True),
            nn.GELU(),
            nn.Linear(D, 1, bias=True),
        )
        import math as _math
        _decay_bias_init = -_math.log(self.mutation_scale)
        nn.init.constant_(self.decay_gate[-1].bias, _decay_bias_init)
        # Zero out the final-layer weights so the bias dominates at init
        # (otherwise random initialization produces α with high variance
        # across hops, destabilizing the first few training steps).
        nn.init.zeros_(self.decay_gate[-1].weight)

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
        tau: float | None = None,
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
        tau_eff = cfg.gumbel_tau if tau is None else float(tau)

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

        ids = manifold.concept_ids
        entry_logits = torch.einsum("bjd,nd->bjn", Q_entry, ids)
        entry_one_hot, entry_idx = gumbel_top1_ste(
            entry_logits, tau_eff, hard=hard,
        )                                                         # [BS, J, N], [BS, J]
        # Routing aux losses — accumulated across entry + K_write hops.
        _aux_e = routing_aux_losses(entry_logits, entry_one_hot)
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
            nbr_ids = manifold.concept_ids[nbr_idx]               # [BS, J, K_max, D]
            nbr_logits = torch.einsum("bjd,bjkd->bjk", Q_t, nbr_ids)
            nbr_one_hot, next_local = gumbel_top1_ste(
                nbr_logits, tau_eff, hard=hard,
            )                                                     # [BS, J, K_max], [BS, J]
            _aux_h = routing_aux_losses(nbr_logits, nbr_one_hot)
            aux_lb_total = aux_lb_total + _aux_h["load_balance"]
            aux_z_total = aux_z_total + _aux_h["z_loss"]
            next_global = torch.gather(
                nbr_idx, dim=2, index=next_local.unsqueeze(-1),
            ).squeeze(-1)                                         # [BS, J]

            # mutate_write — proposed new state for the *current* concept.
            # Same input as step_mlp; compute once.
            delta = self.mutate_mlp(step_input)                   # [BS, J, D]
            candidate = self.mutation_scale * delta               # [BS, J, D]
            # Learnable decay gate: convex combination of old state and
            # candidate. Bounds state magnitude by the candidate magnitude
            # at equilibrium (≈mutation_scale * mlp_output_scale), rather
            # than letting it accumulate unboundedly.
            alpha = torch.sigmoid(self.decay_gate(step_input))    # [BS, J, 1]
            new_state = alpha * current_state + (1.0 - alpha) * candidate

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
