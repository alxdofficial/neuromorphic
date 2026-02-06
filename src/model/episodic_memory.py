"""
Episodic Memory — fixed-size per-stream vector store.

One EM instance per block (B total). Provides long-range recall via
novelty-based writes and top-k retrieval with cross-attention aggregation.

State per stream:
    em_K: [BS, M, D_em]   — keys (unit-normalized)
    em_V: [BS, M, D_em]   — values
    em_S: [BS, M]          — strengths (bounded)
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import ModelConfig
from .utils import StateMixin, unit_normalize, soft_topk, budget_enforce


class EpisodicMemory(nn.Module, StateMixin):
    _state_tensor_names = ["em_K", "em_V", "em_S"]

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.M = config.M
        self.D_em = config.D_em
        self.k_ret = config.k_ret
        self.C = config.C_em
        self.k_write = config.k_write
        self.tau = config.tau_em
        self.weakness_weight = config.weakness_weight_em
        self.S_max = config.S_max
        self.budget = config.budget_em
        self.decay = config.decay_em

        D = config.D

        # Query projection for retrieval
        self.W_q_em = nn.Linear(D + D, config.D_em)  # concat(x, y_wm)

        # Cross-attention for aggregating retrieved tokens
        self.W_q_cross = nn.Linear(D, config.D_em)
        self.W_o_cross = nn.Linear(config.D_em, D)
        self.cross_scale = (config.D_em) ** -0.5

        # Candidate proposal projections
        self.W_k_cand = nn.Linear(D + D, config.D_em)    # from (x, y_wm)
        self.W_v_cand = nn.Linear(config.D_h, config.D_em)  # from h_final (per-block)

        # State (lazily initialized)
        self.em_K: Tensor = None
        self.em_V: Tensor = None
        self.em_S: Tensor = None

    def reset_states(self, mask: Tensor):
        """Reset EM state for masked streams — only zero strengths.

        Unlike the default StateMixin.reset_states() which zeros all state
        tensors, EM preserves em_K and em_V on doc boundary resets. Only
        em_S (slot strengths) is zeroed, effectively making old slots
        invisible to retrieval while keeping their content available for
        future overwrites.
        """
        if self.em_S is not None and self.em_S.dim() >= 1:
            expanded = mask
            for _ in range(self.em_S.dim() - 1):
                expanded = expanded.unsqueeze(-1)
            self.em_S = self.em_S * (~expanded).to(self.em_S.dtype)

    def _lazy_init(self, BS: int, device: torch.device):
        """Initialize state on first forward pass."""
        self.em_K = unit_normalize(torch.randn(BS, self.M, self.D_em, device=device))
        self.em_V = torch.randn(BS, self.M, self.D_em, device=device) * 0.01
        self.em_S = torch.zeros(BS, self.M, device=device)

    def retrieve(self, x: Tensor, y_wm: Tensor) -> Tensor:
        """Query EM and return aggregated output.

        Args:
            x: [BS, D] — current token embedding
            y_wm: [BS, D] — working memory output

        Returns:
            y_em: [BS, D]
        """
        BS = x.shape[0]
        device = x.device

        if self.em_K is None:
            self._lazy_init(BS, device)

        # Compute query
        q = unit_normalize(self.W_q_em(torch.cat([x, y_wm], dim=-1)))  # [BS, D_em]

        # Score against all slots
        scores = torch.einsum("bd, bmd -> bm", q, self.em_K)  # [BS, M]

        # Mask inactive slots (S == 0)
        active_mask = self.em_S > 0
        scores = scores.masked_fill(~active_mask, float("-inf"))

        # Top-k retrieval
        k = min(self.k_ret, self.M)
        topk_scores, topk_idx = scores.topk(k, dim=-1)  # [BS, k]

        # Gather values
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, self.D_em)
        V_top = self.em_V.gather(1, topk_idx_exp)  # [BS, k, D_em]

        # Cross-attention aggregation
        q_cross = self.W_q_cross(x)  # [BS, D_em]
        # Attention: q_cross against V_top as both keys and values
        attn = torch.einsum("bd, bkd -> bk", q_cross, V_top) * self.cross_scale
        # Mask out -inf topk positions (no active slots)
        attn = attn.masked_fill(topk_scores == float("-inf"), float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)

        out = torch.einsum("bk, bkd -> bd", attn, V_top)  # [BS, D_em]
        y_em = self.W_o_cross(out)  # [BS, D]

        return y_em

    def propose_candidate(self, x: Tensor, y_wm: Tensor,
                          h_final: Tensor, surprise: Tensor) -> tuple:
        """Per-token candidate proposal.

        Args:
            x: [BS, D] — token embedding
            y_wm: [BS, D] — WM output
            h_final: [BS, D_h] — final layer output for this block
            surprise: [BS, 1] — current surprise

        Returns:
            k_cand: [BS, D_em]
            v_cand: [BS, D_em]
            novelty: [BS]
        """
        k_cand = unit_normalize(self.W_k_cand(torch.cat([x, y_wm], dim=-1)))
        v_cand = self.W_v_cand(h_final)

        # Novelty = surprise + (1 - max cosine similarity to existing keys)
        if self.em_K is not None:
            cos_sim = torch.einsum("bd, bmd -> bm", k_cand, self.em_K)  # [BS, M]
            max_sim = cos_sim.max(dim=-1).values  # [BS]
        else:
            max_sim = torch.zeros(x.shape[0], device=x.device)

        novelty = (0.5 * surprise.squeeze(-1) + 0.5 * (1.0 - max_sim)).clamp(0.0, 1.0)

        return k_cand, v_cand, novelty

    def write_at_boundary(self, cand_K: Tensor, cand_V: Tensor,
                          cand_score: Tensor, write_mask: Tensor,
                          g_em: Tensor):
        """Span-boundary EM write. Top-C candidates, soft multi-slot EMA.

        Args:
            cand_K: [BS, P, D_em] — candidate keys from span
            cand_V: [BS, P, D_em] — candidate values from span
            cand_score: [BS, P] — novelty scores from span
            write_mask: [BS] bool — which streams write
            g_em: [BS] — write strength per stream
        """
        if not write_mask.any() or self.em_K is None:
            return

        BS = write_mask.shape[0]

        with torch.no_grad():
            # Select top-C candidates per stream
            C = min(self.C, cand_score.shape[1])
            topC_scores, topC_idx = cand_score.topk(C, dim=-1)  # [BS, C]

            topC_idx_exp = topC_idx.unsqueeze(-1).expand(-1, -1, self.D_em)
            K_C = cand_K.gather(1, topC_idx_exp)  # [BS, C, D_em]
            V_C = cand_V.gather(1, topC_idx_exp)  # [BS, C, D_em]

            # For each candidate, soft-select slots to update
            for c in range(C):
                k_c = K_C[:, c]   # [BS, D_em]
                v_c = V_C[:, c]   # [BS, D_em]

                # Score against existing slots
                scores_slot = torch.einsum("bd, bmd -> bm", k_c, self.em_K)  # [BS, M]

                # Weakness bias: prefer overwriting weak slots
                scores_slot = scores_slot - self.weakness_weight * self.em_S

                # Soft top-k selection
                w = soft_topk(scores_slot, self.k_write, self.tau)  # [BS, M]

                # Apply write mask and strength
                alpha = w * g_em.unsqueeze(-1) * write_mask.float().unsqueeze(-1)  # [BS, M]

                # EMA update
                alpha_3d = alpha.unsqueeze(-1)  # [BS, M, 1]
                self.em_K = unit_normalize(
                    (1 - alpha_3d) * self.em_K + alpha_3d * k_c.unsqueeze(1)
                )
                self.em_V = (1 - alpha_3d) * self.em_V + alpha_3d * v_c.unsqueeze(1)

                # Strength update
                self.em_S = (self.em_S + alpha * topC_scores[:, c].unsqueeze(-1)).clamp(
                    0.0, self.S_max
                )

            # Decay and budget enforcement
            self.em_S = self.em_S * self.decay
            self.em_S = budget_enforce(self.em_S, self.budget)


class EMController(nn.Module):
    """Heuristic write decisions based on novelty + surprise.

    MVP: writes if mean novelty exceeds a threshold. Future versions
    will use RL-based gating.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.novelty_threshold = 0.3
        self.default_g = 0.3

    def forward(self, span_surprise: Tensor, em_usage: Tensor,
                cand_novelty_mean: Tensor) -> tuple:
        """Decide which streams should write and with what strength.

        Args:
            span_surprise: [BS] — mean surprise over span
            em_usage: [BS] — sum of em_S (current usage)
            cand_novelty_mean: [BS] — mean novelty of candidates

        Returns:
            write_mask: [BS] bool
            g_em: [BS] — write strength
        """
        write_mask = cand_novelty_mean > self.novelty_threshold
        g_em = torch.full_like(span_surprise, self.default_g)
        return write_mask, g_em
