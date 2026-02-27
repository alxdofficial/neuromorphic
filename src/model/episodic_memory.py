"""
Episodic Memory (v4) — fixed-size per-stream vector store.

Per-block (B_blocks instances). Operates in D_mem space.
Read: top-k retrieval with cross-attention. Write: novelty-scored EMA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig
from .utils import StateMixin, unit_normalize, budget_enforce


class EpisodicMemory(nn.Module, StateMixin):
    """Fixed-size episodic memory with top-k retrieval and novelty-based writes.

    State:
        em_K: [BS, M, D_mem] — keys (unit-normalized)
        em_V: [BS, M, D_mem] — values
        em_S: [BS, M] — strengths (bounded)
        em_age: [BS, M] — tokens since last write
    """

    _state_tensor_names = ["em_K", "em_V", "em_S", "em_age"]

    def __init__(self, D_mem: int, M_slots: int, config: ModelConfig):
        super().__init__()
        self.D_mem = D_mem
        self.M = M_slots
        self.k_ret = config.k_ret
        self.S_max = config.S_max
        self.budget = config.budget_em

        # Cross-attention for retrieval
        self.W_q_cross = nn.Linear(D_mem, D_mem)
        self.W_o_cross = nn.Linear(D_mem, D_mem)

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
        """Top-k retrieval + cross-attention.

        q: [BS,N,C,D_mem] -> [BS,N,C,D_mem]
        """
        BS = q.shape[0]
        prefix_shape = q.shape[:-1]  # [BS, N, C]
        D = q.shape[-1]
        state_dtype = self.em_K.dtype

        # Flatten to [BS, N*C, D_mem] for retrieval
        q_flat = q.reshape(BS, -1, D)  # [BS, N*C, D_mem]
        NC = q_flat.shape[1]

        # Scores against all slots (cast q to state dtype for einsum)
        scores = torch.einsum(
            "bnd, bmd -> bnm", q_flat.to(state_dtype), self.em_K
        )  # [BS, N*C, M]

        # Mask inactive slots (S == 0)
        active_mask = (self.em_S > 0).unsqueeze(1)  # [BS, 1, M]
        scores = scores.masked_fill(~active_mask, -1e9)

        # Top-k retrieval
        k = min(self.k_ret, self.M)
        topk_scores, topk_idx = scores.topk(k, dim=-1)  # [BS, N*C, k]

        # Gather top-k keys and values
        topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # [BS, N*C, k, D]
        em_K_expanded = self.em_K.unsqueeze(1).expand(-1, NC, -1, -1)  # [BS, N*C, M, D]
        em_V_expanded = self.em_V.unsqueeze(1).expand(-1, NC, -1, -1)
        K_top = torch.gather(em_K_expanded, 2, topk_idx_expanded)  # [BS, N*C, k, D]
        V_top = torch.gather(em_V_expanded, 2, topk_idx_expanded)

        # Cross-attention (W_q_cross is fp32, so feed it q_flat in original dtype)
        q_cross = self.W_q_cross(q_flat)  # [BS, N*C, D_mem]
        # K_top is in state dtype; cast q_cross for einsum, then cast back
        attn_logits = torch.einsum(
            "bnd, bnkd -> bnk", q_cross.to(state_dtype), K_top
        )  # [BS, N*C, k]
        attn_logits = attn_logits + topk_scores  # add retrieval score as bias
        attn_weights = F.softmax(attn_logits.float(), dim=-1)  # [BS, N*C, k]
        out = torch.einsum(
            "bnk, bnkd -> bnd", attn_weights.to(state_dtype), V_top
        )  # [BS, N*C, D_mem]
        # W_o_cross is fp32, cast out back to q's dtype
        out = self.W_o_cross(out.to(q.dtype))

        # Reshape back to prefix shape
        return out.reshape(*prefix_shape, D)

    def score_novelty(self, q_nov: Tensor, surprise: Tensor, w_nov: Tensor) -> Tensor:
        """Score novelty for candidate selection.

        q_nov: [BS,N,C,D_mem], surprise: [BS,N,C], w_nov: [BS,N,C]
        Returns: novelty [BS,N,C]
        """
        BS = q_nov.shape[0]
        prefix_shape = q_nov.shape[:-1]
        q_flat = q_nov.reshape(BS, -1, self.D_mem).to(self.em_K.dtype)

        # Max cosine similarity against em_K
        if self.em_S is not None and (self.em_S > 0).any():
            sim = torch.einsum("bnd, bmd -> bnm", q_flat, self.em_K)  # [BS, N*C, M]
            active_mask = (self.em_S > 0).unsqueeze(1)
            sim = sim.masked_fill(~active_mask, -1.0)
            max_sim = sim.max(dim=-1).values  # [BS, N*C]
            max_sim = max_sim.reshape(*prefix_shape)
        else:
            max_sim = torch.zeros(prefix_shape, device=q_nov.device, dtype=q_nov.dtype)

        novelty = w_nov * surprise + (1 - w_nov) * (1 - max_sim)
        return novelty

    def select_top_candidates(self, q_nov: Tensor, v_nov: Tensor,
                               novelty: Tensor, C_cand: int):
        """Select top-C_cand candidates across all N*C positions.

        Returns: (cand_K, cand_V, cand_scores) each [BS, C_cand, D_mem] or [BS, C_cand]
        """
        BS = q_nov.shape[0]
        D = q_nov.shape[-1]

        # Flatten N*C
        q_flat = q_nov.reshape(BS, -1, D)   # [BS, N*C, D_mem]
        v_flat = v_nov.reshape(BS, -1, D)   # [BS, N*C, D_mem]
        nov_flat = novelty.reshape(BS, -1)   # [BS, N*C]

        # Top-k candidates
        C_cand = min(C_cand, nov_flat.shape[1])
        topk_scores, topk_idx = nov_flat.topk(C_cand, dim=-1)  # [BS, C_cand]

        # Gather
        topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, D)  # [BS, C_cand, D_mem]
        cand_K = torch.gather(q_flat, 1, topk_idx_expanded)  # [BS, C_cand, D_mem]
        cand_V = torch.gather(v_flat, 1, topk_idx_expanded)  # [BS, C_cand, D_mem]

        return cand_K, cand_V, topk_scores

    def write(self, cand_K: Tensor, cand_V: Tensor, cand_scores: Tensor,
              g_em: Tensor, tau: Tensor, decay: Tensor):
        """Write candidates to EM via EMA.

        cand_K: [BS, C_cand, D_mem]
        cand_V: [BS, C_cand, D_mem]
        cand_scores: [BS, C_cand]
        g_em: [BS]
        tau: [BS]
        decay: [BS]
        """
        BS, C_cand, D = cand_K.shape

        # Score candidates against all slots for slot selection
        cand_K_norm = unit_normalize(cand_K)
        # Cast to state dtype for einsum (state may be bf16 on CUDA)
        slot_scores = torch.einsum(
            "bcd, bmd -> bcm", cand_K_norm.to(self.em_K.dtype), self.em_K
        ).to(cand_K.dtype)  # [BS, C_cand, M]

        # Weakness bias: prefer weaker slots
        weakness = -self.em_S.unsqueeze(1)  # [BS, 1, M]
        slot_scores = slot_scores + 0.5 * weakness

        # Softmax slot selection
        slot_weights = F.softmax(
            slot_scores / tau.reshape(BS, 1, 1).clamp(min=0.01), dim=-1
        )  # [BS, C_cand, M]

        # Write strength per candidate
        cand_weight = cand_scores / (cand_scores.sum(dim=-1, keepdim=True) + 1e-8)
        alpha = g_em.reshape(BS, 1, 1) * cand_weight.unsqueeze(-1) * slot_weights  # [BS, C_cand, M]

        # Sum across candidates for each slot
        alpha_per_slot = alpha.sum(dim=1)  # [BS, M]

        # Weighted candidate blend per slot
        blended_K = torch.einsum("bcm, bcd -> bmd", alpha, cand_K_norm)  # [BS, M, D]
        blended_V = torch.einsum("bcm, bcd -> bmd", alpha, cand_V)       # [BS, M, D]

        # Normalize by alpha
        denom = alpha_per_slot.unsqueeze(-1).clamp(min=1e-8)
        blended_K = blended_K / denom
        blended_V = blended_V / denom

        # EMA update (only slots with nonzero alpha)
        update_mask = (alpha_per_slot > 1e-8).unsqueeze(-1)  # [BS, M, 1]
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
        expanded = mask.unsqueeze(-1)  # [BS, 1]
        self.em_S = self.em_S * (~expanded).to(self.em_S.dtype)
        self.em_age = self.em_age * (~expanded).to(self.em_age.dtype)


class EMNeuromodulator(nn.Module):
    """Neuromodulator for EM write decisions.

    Returns: g_em [BS], tau [BS], decay [BS]
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
        novelty_mean: [BS]
        em_usage: [BS]
        content_emb: [BS, D_mem] (optional)

        Returns: g_em [BS], tau [BS], decay [BS]
        """
        if not self.em_enabled:
            BS = novelty_mean.shape[0]
            device = novelty_mean.device
            return (
                torch.full((BS,), self.default_g, device=device),
                torch.full((BS,), self.default_tau, device=device),
                torch.full((BS,), self.default_decay, device=device),
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
