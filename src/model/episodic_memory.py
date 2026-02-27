"""
Episodic Memory (v4) — fixed-size per-stream vector store.

Single instance (batched across B_blocks via BS*B batch dim).
Operates in D_mem space.
Read: top-k retrieval with cross-attention (GroupedLinear for per-block params).
Write: novelty-scored EMA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig
from .predictive_coding import GroupedLinear
from .utils import StateMixin, unit_normalize, budget_enforce


class EpisodicMemory(nn.Module, StateMixin):
    """Fixed-size episodic memory with top-k retrieval and novelty-based writes.

    State (batched across B_blocks):
        em_K: [BS*B, M, D_mem] — keys (unit-normalized)
        em_V: [BS*B, M, D_mem] — values
        em_S: [BS*B, M] — strengths (bounded)
        em_age: [BS*B, M] — tokens since last write
    """

    _state_tensor_names = ["em_K", "em_V", "em_S", "em_age"]

    def __init__(self, D_mem: int, M_slots: int, config: ModelConfig):
        super().__init__()
        self.D_mem = D_mem
        self.M = M_slots
        self.k_ret = config.k_ret
        self.S_max = config.S_max
        self.budget = config.budget_em
        self.B = config.B_blocks

        # Per-block cross-attention projections via GroupedLinear
        self.W_q_cross = GroupedLinear(self.B, D_mem, D_mem)
        self.W_o_cross = GroupedLinear(self.B, D_mem, D_mem)

        # State (lazily allocated)
        self.em_K: Tensor | None = None
        self.em_V: Tensor | None = None
        self.em_S: Tensor | None = None
        self.em_age: Tensor | None = None

    def initialize(self, BS: int, device: torch.device, dtype: torch.dtype):
        self.em_K = torch.zeros(BS, self.M, self.D_mem, device=device, dtype=dtype)
        self.em_V = torch.zeros(BS, self.M, self.D_mem, device=device, dtype=dtype)
        self.em_S = torch.zeros(BS, self.M, device=device, dtype=dtype)
        self.em_age = torch.zeros(BS, self.M, device=device, dtype=dtype)

    def is_initialized(self) -> bool:
        return self.em_K is not None

    def read(self, q: Tensor) -> Tensor:
        """Top-k retrieval + cross-attention (gather-free).

        q: [BS*B, N*C, D_mem] -> [BS*B, N*C, D_mem]

        Uses masked dense attention against all M slots instead of
        topk+gather. Softmax masking ensures only top-k contribute.
        Eliminates expensive GatherBackward and ExpandBackward in backward.

        Cross-attention projections are per-block (GroupedLinear(B, ...)),
        so we reshape to [BS, NC, B, D] for those ops, then back to [BS*B, NC, D].
        """
        BSB = q.shape[0]
        prefix_shape = q.shape[:-1]  # [BS*B, N*C]
        D = q.shape[-1]
        BS = BSB // self.B

        # Flatten to [BSB, NC, D_mem] for retrieval
        q_flat = q.reshape(BSB, -1, D)  # [BSB, NC, D_mem]
        NC = q_flat.shape[1]

        # Dense retrieval scores against all M slots
        scores = torch.einsum(
            "bnd, bmd -> bnm", q_flat, self.em_K
        )  # [BSB, NC, M]

        # Mask inactive slots (S == 0)
        active_mask = (self.em_S > 0).unsqueeze(1)  # [BSB, 1, M]
        scores = scores.masked_fill(~active_mask, -1e9)

        # Top-k mask (no gather — just mask non-topk to -inf)
        k = min(self.k_ret, self.M)
        topk_vals, _ = scores.topk(k, dim=-1)  # [BSB, NC, k]
        threshold = topk_vals[..., -1:].detach()  # [BSB, NC, 1]
        topk_mask = scores >= threshold  # [BSB, NC, M]

        # Cross-attention with per-block GroupedLinear projections
        q_grouped = q_flat.view(BS, self.B, NC, D).permute(0, 2, 1, 3)
        q_cross = self.W_q_cross(q_grouped)  # [BS, NC, B, D]
        q_cross = q_cross.permute(0, 2, 1, 3).reshape(BSB, NC, D)  # [BSB, NC, D]

        # Dense cross-attention logits against all M slots, masked to top-k
        attn_logits = torch.einsum(
            "bnd, bmd -> bnm", q_cross, self.em_K
        )  # [BSB, NC, M]
        attn_logits = attn_logits + scores
        attn_logits = attn_logits.masked_fill(~topk_mask, -1e9)
        attn_weights = F.softmax(attn_logits.float(), dim=-1)  # [BSB, NC, M]

        # Weighted sum from all slots (only top-k have nonzero weight)
        out = torch.einsum(
            "bnm, bmd -> bnd", attn_weights, self.em_V
        )  # [BSB, NC, D_mem]

        # Output projection with per-block GroupedLinear
        out_grouped = out.view(BS, self.B, NC, D).permute(0, 2, 1, 3)
        out = self.W_o_cross(out_grouped)  # [BS, NC, B, D]
        out = out.permute(0, 2, 1, 3).reshape(BSB, NC, D)  # [BSB, NC, D]

        # Reshape back to prefix shape
        return out.reshape(*prefix_shape, D)

    def score_novelty(self, q_nov: Tensor, surprise: Tensor, w_nov: Tensor) -> Tensor:
        """Score novelty for candidate selection.

        q_nov: [BSB,NC,D_mem], surprise: [BSB,NC], w_nov: [BSB,NC]
        Returns: novelty [BSB,NC]
        """
        BSB = q_nov.shape[0]
        prefix_shape = q_nov.shape[:-1]
        q_flat = q_nov.reshape(BSB, -1, self.D_mem)

        # Max cosine similarity against em_K (no branch — masking handles empty memory)
        sim = torch.einsum("bnd, bmd -> bnm", q_flat, self.em_K)  # [BSB, NC, M]
        active_mask = (self.em_S > 0).unsqueeze(1)  # [BSB, 1, M]
        sim = sim.masked_fill(~active_mask, -1.0)
        max_sim = sim.max(dim=-1).values.clamp(min=0.0)  # [BSB, NC]
        max_sim = max_sim.reshape(*prefix_shape)

        novelty = w_nov * surprise + (1 - w_nov) * (1 - max_sim)
        return novelty

    def select_top_candidates(self, q_nov: Tensor, v_nov: Tensor,
                               novelty: Tensor, C_cand: int):
        """Select top-C_cand candidates across all N*C positions.

        Returns: (cand_K, cand_V, cand_scores) each [BSB, C_cand, D_mem] or [BSB, C_cand]
        """
        BSB = q_nov.shape[0]
        D = q_nov.shape[-1]

        # Flatten N*C
        q_flat = q_nov.reshape(BSB, -1, D)   # [BSB, NC, D_mem]
        v_flat = v_nov.reshape(BSB, -1, D)   # [BSB, NC, D_mem]
        nov_flat = novelty.reshape(BSB, -1)   # [BSB, NC]

        # Top-k candidates
        C_cand = min(C_cand, nov_flat.shape[1])
        topk_scores, topk_idx = nov_flat.topk(C_cand, dim=-1)  # [BSB, C_cand]

        # Gather
        topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, D)  # [BSB, C_cand, D_mem]
        cand_K = torch.gather(q_flat, 1, topk_idx_expanded)  # [BSB, C_cand, D_mem]
        cand_V = torch.gather(v_flat, 1, topk_idx_expanded)  # [BSB, C_cand, D_mem]

        return cand_K, cand_V, topk_scores

    def write(self, cand_K: Tensor, cand_V: Tensor, cand_scores: Tensor,
              g_em: Tensor, tau: Tensor, decay: Tensor):
        """Write candidates to EM via EMA.

        cand_K: [BSB, C_cand, D_mem]
        cand_V: [BSB, C_cand, D_mem]
        cand_scores: [BSB, C_cand]
        g_em: [BSB]
        tau: [BSB]
        decay: [BSB]
        """
        BSB, C_cand, D = cand_K.shape

        # Score candidates against all slots for slot selection
        cand_K_norm = unit_normalize(cand_K)
        slot_scores = torch.einsum(
            "bcd, bmd -> bcm", cand_K_norm, self.em_K
        )  # [BSB, C_cand, M]

        # Weakness bias: prefer weaker slots
        weakness = -self.em_S.unsqueeze(1)  # [BSB, 1, M]
        slot_scores = slot_scores + 0.5 * weakness

        # Softmax slot selection
        slot_weights = F.softmax(
            slot_scores / tau.reshape(BSB, 1, 1).clamp(min=0.01), dim=-1
        )  # [BSB, C_cand, M]

        # Write strength per candidate
        cand_weight = cand_scores / (cand_scores.sum(dim=-1, keepdim=True) + 1e-8)
        alpha = g_em.reshape(BSB, 1, 1) * cand_weight.unsqueeze(-1) * slot_weights  # [BSB, C_cand, M]

        # Sum across candidates for each slot
        alpha_per_slot = alpha.sum(dim=1)  # [BSB, M]

        # Weighted candidate blend per slot
        blended_K = torch.einsum("bcm, bcd -> bmd", alpha, cand_K_norm)  # [BSB, M, D]
        blended_V = torch.einsum("bcm, bcd -> bmd", alpha, cand_V)       # [BSB, M, D]

        # Normalize by alpha
        denom = alpha_per_slot.unsqueeze(-1).clamp(min=1e-8)
        blended_K = blended_K / denom
        blended_V = blended_V / denom

        # EMA update (only slots with nonzero alpha)
        update_mask = (alpha_per_slot > 1e-8).unsqueeze(-1)  # [BSB, M, 1]
        alpha_exp = alpha_per_slot.unsqueeze(-1).clamp(max=1.0)

        new_K = (1 - alpha_exp) * self.em_K + alpha_exp * unit_normalize(blended_K)
        new_V = (1 - alpha_exp) * self.em_V + alpha_exp * blended_V

        self.em_K = torch.where(update_mask, new_K, self.em_K)
        self.em_V = torch.where(update_mask, new_V, self.em_V)

        # Strength update
        self.em_S = (self.em_S + alpha_per_slot).clamp(0, self.S_max)

        # Age reset for updated slots
        self.em_age = self.em_age * (1 - alpha_per_slot)

        # Decay
        self.em_S = self.em_S * decay.unsqueeze(-1)

        # Budget enforcement
        self.em_S = budget_enforce(self.em_S, self.budget)

    def age_tick(self, n_tokens: int):
        """Increment age of active slots."""
        if self.em_age is not None:
            active = (self.em_S > 0).float()
            self.em_age = self.em_age + n_tokens * active

    def reset_states(self, mask: Tensor):
        """Custom reset: zero S and age, preserve K/V (makes slots invisible)."""
        if self.em_S is None:
            return
        expanded = mask.unsqueeze(-1)  # [BSB, 1]
        self.em_S = self.em_S * ~expanded
        self.em_age = self.em_age * ~expanded


class EMNeuromodulator(nn.Module):
    """Neuromodulator for EM write decisions.

    Returns: g_em [BSB], tau [BSB], decay [BSB]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.em_enabled = config.em_enabled
        self.default_g = config.g_em_default
        self.default_tau = config.tau_em
        self.default_decay = config.decay_em
        self.g_em_floor = config.g_em_floor
        self.g_em_ceil = config.g_em_ceil
        self.tau_floor = config.tau_em_floor
        self.tau_ceil = config.tau_em_ceil
        self.decay_floor = config.decay_em_floor
        self.decay_ceil = config.decay_em_ceil

        if self.em_enabled:
            H = config.neuromod_hidden
            n_scalar = 2  # novelty_mean, em_usage
            n_content = config.content_proj_dim
            n_features = n_scalar + n_content

            self.content_proj = nn.Linear(config.D_mem, n_content)
            self.backbone = nn.Sequential(
                nn.Linear(n_features, H),
                nn.ReLU(),
            )
            self.g_head = nn.Linear(H, 1)
            self.tau_head = nn.Linear(H, 1)
            self.decay_head = nn.Linear(H, 1)

            nn.init.zeros_(self.backbone[0].bias)
            nn.init.normal_(self.content_proj.weight, std=0.01)
            nn.init.zeros_(self.content_proj.bias)

    def forward(self, novelty_mean: Tensor, em_usage: Tensor,
                content_emb: Tensor | None = None):
        """
        novelty_mean: [BSB]
        em_usage: [BSB]
        content_emb: [BSB, D_mem] (optional)

        Returns: g_em [BSB], tau [BSB], decay [BSB]
        """
        if not self.em_enabled:
            BSB = novelty_mean.shape[0]
            device = novelty_mean.device
            return (
                torch.full((BSB,), self.default_g, device=device),
                torch.full((BSB,), self.default_tau, device=device),
                torch.full((BSB,), self.default_decay, device=device),
            )

        features = [novelty_mean.unsqueeze(-1), em_usage.unsqueeze(-1)]
        if content_emb is not None:
            features.append(self.content_proj(content_emb))
        else:
            features.append(torch.zeros(
                novelty_mean.shape[0], self.content_proj.out_features,
                device=novelty_mean.device, dtype=novelty_mean.dtype,
            ))

        x = torch.cat(features, dim=-1)
        h = self.backbone(x)

        g_raw = self.g_head(h).squeeze(-1)
        g_em = self.g_em_floor + (self.g_em_ceil - self.g_em_floor) * torch.sigmoid(g_raw)

        tau_raw = self.tau_head(h).squeeze(-1)
        tau = self.tau_floor + (self.tau_ceil - self.tau_floor) * torch.sigmoid(tau_raw)

        decay_raw = self.decay_head(h).squeeze(-1)
        decay = self.decay_floor + (self.decay_ceil - self.decay_floor) * torch.sigmoid(decay_raw)

        return g_em, tau, decay
