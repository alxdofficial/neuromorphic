"""ReadTrajectoryGenerator — J parallel autoregressive hops at start of window.

Conditions only on `prev_window_hiddens: [BS, T_window, D_lm]` (Llama hiddens
from the previous window). Outputs `[BS, J*K_read, D_concept]` — J·K_read
visited concept states, flattened, ready for cross-attention into Llama.

Per-hop Q construction takes `current_state[j], history_attn, cross_attn`
(no `id_vec[current]` — redundant with the K side of the QK match).
Order is encoded by sinusoidal positional encoding added to history-attn
keys/values.

Discrete next-concept selection uses Gumbel-top-1 STE during training
(hard forward, soft backward) and argmax at inference.

See docs/plan_trajectory_memory.md §2.2, §3.1, §4.8.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.manifold import Manifold


def make_pos_enc(K: int, D: int) -> Tensor:
    """Sinusoidal positional encoding [K, D]. Standard Transformer-style."""
    pos = torch.arange(K, dtype=torch.float32).unsqueeze(1)        # [K, 1]
    div = torch.exp(
        torch.arange(0, D, 2, dtype=torch.float32) * (-math.log(10000.0) / D)
    )                                                              # [D/2]
    pe = torch.zeros(K, D)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def gumbel_top1_ste(
    logits: Tensor,
    tau: float,
    *,
    hard: bool,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Gumbel-top-1 with straight-through estimator.

    During training (hard=True): hard one-hot forward, soft backward.
    During inference (hard=False): just argmax (no Gumbel noise).

    Returns:
        one_hot:  [..., V]  one-hot selection (forward); soft probs in backward
        index:    [...]     int64 selected index
    """
    if not hard:
        idx = logits.argmax(dim=-1)
        one_hot = F.one_hot(idx, num_classes=logits.shape[-1]).to(logits.dtype)
        return one_hot, idx

    # Gumbel noise. `torch.rand_like` sometimes lacks a generator kwarg
    # depending on torch version; use empty + uniform_ for portability.
    if generator is None:
        u = torch.rand_like(logits)
    else:
        u = torch.empty_like(logits).uniform_(0.0, 1.0, generator=generator)
    g = -torch.log(-torch.log(u.clamp_min(1e-20)).clamp_min(1e-20))
    y = (logits + g) / tau
    soft = F.softmax(y, dim=-1)
    idx = soft.argmax(dim=-1)
    hard_one_hot = F.one_hot(idx, num_classes=logits.shape[-1]).to(soft.dtype)
    # Straight-through: forward = hard, backward = soft
    one_hot = (hard_one_hot - soft).detach() + soft
    return one_hot, idx


class _CrossAttn(nn.Module):
    """Single-head cross-attention. Q in d_q, KV in d_kv, both project to d.

    Used in two places:
    - history_attn: Q = current state, KV = visited list + pos_enc
    - cross_attn:   Q = current state, KV = window hiddens

    Single-head suffices because trajectory hops are small and we don't
    need multi-head expressivity; if benching shows it's too lossy we can
    bump to multi-head.
    """

    def __init__(self, d_q: int, d_kv: int, d_attn: int):
        super().__init__()
        self.d_attn = d_attn
        self.W_q = nn.Linear(d_q, d_attn, bias=False)
        self.W_k = nn.Linear(d_kv, d_attn, bias=False)
        self.W_v = nn.Linear(d_kv, d_attn, bias=False)
        for m in (self.W_q, self.W_k, self.W_v):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        """
        Args:
            q:  [BS, *Q, d_q]
            kv: [BS, *KV, d_kv]
        Returns:
            [BS, *Q, d_attn]
        """
        Q = self.W_q(q)                                          # [BS, *Q, d_attn]
        K = self.W_k(kv)                                         # [BS, *KV, d_attn]
        V = self.W_v(kv)
        # Flatten Q-extra-dims and KV-extra-dims for bmm-friendly attention.
        BS = q.shape[0]
        Q2 = Q.reshape(BS, -1, self.d_attn)                      # [BS, NQ, d]
        K2 = K.reshape(BS, -1, self.d_attn)                      # [BS, NK, d]
        V2 = V.reshape(BS, -1, self.d_attn)
        scores = torch.bmm(Q2, K2.transpose(1, 2)) / math.sqrt(self.d_attn)
        attn = F.softmax(scores, dim=-1)
        out2 = torch.bmm(attn, V2)                               # [BS, NQ, d]
        return out2.reshape(*Q.shape)                            # [BS, *Q, d]


class ReadTrajectoryGenerator(nn.Module):
    """Generates J parallel read trajectories per window.

    Forward:
        prev_window_hiddens: [BS, T_window, d_lm]
        current_states:      [BS, N, D_concept] — manifold state at window start
        manifold:            Manifold instance (for concept_ids, edge_indices)

        returns visited:     [BS, J, K_read, D_concept]
                visited_ids: [BS, J, K_read] (int64) — for telemetry/debug
    """

    def __init__(self, cfg: TrajMemConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.D_concept
        d_lm = cfg.d_lm

        # J head queries to differentiate the parallel trajectories.
        self.head_query = nn.Parameter(torch.empty(cfg.J, D))
        nn.init.normal_(self.head_query, std=0.02)

        # Entry-point MLP: takes pooled prev_hiddens (d_lm) + head_query (D)
        # and produces a query in concept-id space (D).
        self.entry_mlp = nn.Sequential(
            nn.Linear(d_lm + D, D, bias=True),
            nn.GELU(),
            nn.Linear(D, D, bias=True),
        )

        # Per-hop attention modules:
        d_attn = D
        self.history_attn = _CrossAttn(d_q=D, d_kv=D, d_attn=d_attn)
        self.cross_attn = _CrossAttn(d_q=D, d_kv=d_lm, d_attn=d_attn)

        # Per-hop step MLP: takes (current_state, history_attn_out, cross_attn_out)
        # produces a query in id-space for the neighbor scoring.
        self.step_mlp = nn.Sequential(
            nn.Linear(D * 3, D, bias=True),
            nn.GELU(),
            nn.Linear(D, D, bias=True),
        )

        # Positional encoding for the visited list (used in history_attn keys).
        # Buffer so it moves to GPU with the module.
        pe = make_pos_enc(cfg.K_read + 1, D) * cfg.pos_enc_scale
        self.register_buffer("pos_enc", pe, persistent=False)

    def forward(
        self,
        prev_window_hiddens: Tensor,
        current_states: Tensor,
        manifold: Manifold,
        *,
        hard: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Generate J parallel trajectories.

        Args:
            prev_window_hiddens: [BS, T_window, d_lm]
            current_states:      [BS, N, D_concept]
            manifold:            Manifold (concept_ids + edge_indices)
            hard:                Gumbel-STE if True, argmax if False

        Returns:
            visited:     [BS, J, K_read, D_concept]
            visited_ids: [BS, J, K_read] int64
        """
        cfg = self.cfg
        BS, T, d_lm = prev_window_hiddens.shape
        D = cfg.D_concept
        J, K = cfg.J, cfg.K_read
        N = cfg.N
        assert current_states.shape == (BS, N, D)

        # ── 1. Pick J entry concepts ──────────────────────────────────
        pooled = prev_window_hiddens.mean(dim=1)                  # [BS, d_lm]
        # Broadcast pooled across J, concat head_query.
        pooled_j = pooled.unsqueeze(1).expand(BS, J, d_lm)        # [BS, J, d_lm]
        hq = self.head_query.unsqueeze(0).expand(BS, J, D)        # [BS, J, D]
        entry_input = torch.cat([pooled_j, hq], dim=-1)           # [BS, J, d_lm+D]
        Q_entry = self.entry_mlp(entry_input)                     # [BS, J, D]

        # Score each entry query against all N concept_ids.
        # concept_ids: [N, D] → broadcast over BS, J.
        ids = manifold.concept_ids                                # [N, D]
        entry_logits = torch.einsum("bjd,nd->bjn", Q_entry, ids)  # [BS, J, N]
        _, entry_idx = gumbel_top1_ste(entry_logits, cfg.gumbel_tau, hard=hard)
        # entry_idx: [BS, J] int64

        # ── 2. K_read autoregressive hops ────────────────────────────
        current = entry_idx                                       # [BS, J]
        # Gather initial states. current_state: [BS, J, D].
        current_state = manifold.gather_states(current_states, current)

        # Pre-allocate visited list. We accumulate via list-append for
        # readability; could be replaced with a pre-allocated tensor for
        # torch.compile friendliness if profiling motivates.
        visited_list: list[Tensor] = []
        visited_id_list: list[Tensor] = []

        for t in range(K):
            # Append current to visited (raw state, no mutation in read path).
            visited_list.append(current_state)                    # [BS, J, D]
            visited_id_list.append(current)                       # [BS, J]

            # Build keys/values for history attention from visited so far.
            # visited so far has length t+1 (including the just-appended).
            # We attend on visited[:t+1] + pos_enc[:t+1].
            hist_kv = torch.stack(visited_list, dim=2)            # [BS, J, t+1, D]
            pos = self.pos_enc[: t + 1].unsqueeze(0).unsqueeze(1) # [1, 1, t+1, D]
            hist_kv = hist_kv + pos                               # broadcast

            # history_attn: query = current_state ([BS, J, D]); kv per j.
            # _CrossAttn supports [BS, *Q, ...] and [BS, *KV, ...] but each
            # j has its own visited list, so we run J separate attentions
            # by treating J as part of the leading shape and broadcasting.
            history_attn_out = self._per_j_attn(
                self.history_attn, current_state, hist_kv,
            )                                                     # [BS, J, D]

            # cross_attn: query per j attends over the same prev_window_hiddens.
            # Broadcast prev_window_hiddens across J dim.
            prev_hid_bjt = prev_window_hiddens.unsqueeze(1).expand(
                BS, J, T, d_lm,
            )
            cross_attn_out = self._per_j_attn(
                self.cross_attn, current_state, prev_hid_bjt,
            )                                                     # [BS, J, D]

            # Step MLP → query in id-space.
            step_input = torch.cat(
                [current_state, history_attn_out, cross_attn_out], dim=-1,
            )                                                     # [BS, J, 3D]
            Q_t = self.step_mlp(step_input)                       # [BS, J, D]

            # Score against neighbors of current[j]. neighbor ids: [BS, J, K_max, D].
            nbr_idx = manifold.get_neighbor_indices(current)      # [BS, J, K_max]
            nbr_ids = manifold.concept_ids[nbr_idx]               # [BS, J, K_max, D]
            nbr_logits = torch.einsum(
                "bjd,bjkd->bjk", Q_t, nbr_ids,
            )                                                     # [BS, J, K_max]
            _, next_local = gumbel_top1_ste(
                nbr_logits, cfg.gumbel_tau, hard=hard,
            )                                                     # [BS, J] in [0, K_max)

            # Map local neighbor index to absolute concept id.
            next_global = torch.gather(
                nbr_idx, dim=2, index=next_local.unsqueeze(-1),
            ).squeeze(-1)                                         # [BS, J]

            current = next_global
            current_state = manifold.gather_states(current_states, current)

        visited = torch.stack(visited_list, dim=2)                # [BS, J, K, D]
        visited_ids = torch.stack(visited_id_list, dim=2)         # [BS, J, K]
        return visited, visited_ids

    @staticmethod
    def _per_j_attn(
        attn_module: _CrossAttn, q: Tensor, kv: Tensor,
    ) -> Tensor:
        """Apply attention separately for each j in the J dim.

        q:  [BS, J, D]
        kv: [BS, J, NK, d_kv]

        Returns: [BS, J, D]

        Implementation: fold J into BS, apply attn, fold back. This is
        the per-j-independent attention semantically — attn_module's
        internal flatten won't accidentally mix across j because the
        leading batch is now BS*J.
        """
        BS, J, D = q.shape
        d_kv = kv.shape[-1]
        NK = kv.shape[-2]
        q_flat = q.reshape(BS * J, 1, D)
        kv_flat = kv.reshape(BS * J, NK, d_kv)
        out = attn_module(q_flat, kv_flat)                        # [BS*J, 1, D]
        return out.reshape(BS, J, D)
