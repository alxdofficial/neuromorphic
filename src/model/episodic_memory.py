"""
Episodic Memory (v4) — fixed-size per-stream vector store.

Single instance (batched across B_blocks via BS,B dims).
Operates in D space (block level: C columns concatenated).
Read: top-k retrieval (direct dot-product, no projections).
Write: novelty-scored EMA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig
from .utils import StateMixin, unit_normalize, budget_enforce


class EpisodicMemory(nn.Module, StateMixin):
    """Fixed-size episodic memory with top-k retrieval and novelty-based writes.

    State (with explicit B dim):
        em_K: [BS, B, M, D] — keys (unit-normalized)
        em_V: [BS, B, M, D] — values
        em_S: [BS, B, M] — strengths (bounded)
        em_age: [BS, B, M] — tokens since last write
    """

    _state_tensor_names = ["em_K", "em_V", "em_S", "em_age"]

    def __init__(self, dim: int, M_slots: int, config: ModelConfig):
        super().__init__()
        self.dim = dim
        self.M = M_slots
        self.k_ret = config.k_ret
        self.S_max = config.S_max
        self.budget = config.budget_em
        self.B = config.B_blocks

        # State (lazily allocated)
        self.em_K: Tensor | None = None
        self.em_V: Tensor | None = None
        self.em_S: Tensor | None = None
        self.em_age: Tensor | None = None

    def initialize(self, BS: int, device: torch.device, dtype: torch.dtype):
        self.em_K = torch.zeros(BS, self.B, self.M, self.dim, device=device, dtype=dtype)
        self.em_V = torch.zeros(BS, self.B, self.M, self.dim, device=device, dtype=dtype)
        self.em_S = torch.zeros(BS, self.B, self.M, device=device, dtype=dtype)
        self.em_age = torch.zeros(BS, self.B, self.M, device=device, dtype=dtype)

    def is_initialized(self) -> bool:
        return self.em_K is not None

    def read(self, q: Tensor) -> Tensor:
        """Top-k retrieval (gather-free, no projections).

        q: [BS, N, B, D] -> [BS, N, B, D]
        """
        BS, N, B, D = q.shape

        # Dense scores via batched matmul (strided views, no copy)
        q_bn = q.transpose(1, 2)                                   # [BS, B, N, D]
        scores = torch.matmul(q_bn, self.em_K.transpose(-1, -2))   # [BS, B, N, M]
        scores = scores.transpose(1, 2)                             # [BS, N, B, M]

        # Mask inactive slots
        active_mask = (self.em_S > 0)[:, None, :, :]  # [BS, 1, B, M]
        scores = scores.masked_fill(~active_mask, -1e9)

        # Top-k mask (gather-free)
        k = min(self.k_ret, self.M)
        topk_vals, _ = scores.topk(k, dim=-1)           # [BS, N, B, k]
        threshold = topk_vals[..., -1:].detach()         # [BS, N, B, 1]
        topk_mask = scores >= threshold                  # [BS, N, B, M]

        # Masked softmax attention (bf16 — M is small)
        attn_logits = scores.masked_fill(~topk_mask, -1e9)
        attn_weights = F.softmax(attn_logits, dim=-1)    # [BS, N, B, M]

        # Weighted sum via batched matmul (strided views, no copy)
        attn_bn = attn_weights.transpose(1, 2)                     # [BS, B, N, M]
        out = torch.matmul(attn_bn, self.em_V)                     # [BS, B, N, D]
        out = out.transpose(1, 2)                                   # [BS, N, B, D]

        return out

    def read_sliced(self, q: Tensor) -> Tensor:
        """Top-k retrieval with per-column D_col slices.

        q: [BS, N_C, B, C, D_col] -> [BS, N_C, B, C, D_col]

        EM state [BS, B, M, D] is viewed as [BS, B, M, C, D_col] and
        retrieval operates independently per column at D_col width.
        """
        BS, N_C, B, C, D_col = q.shape

        # View state as [BS, B, M, C, D_col] -> permute to [BS, B, C, M, D_col]
        K = self.em_K.view(BS, B, self.M, C, D_col).permute(0, 1, 3, 2, 4)
        V = self.em_V.view(BS, B, self.M, C, D_col).permute(0, 1, 3, 2, 4)

        # Query: [BS, N_C, B, C, D_col] -> [BS, B, C, N_C, D_col]
        q_r = q.permute(0, 2, 3, 1, 4)

        # Scores: [BS, B, C, N_C, D_col] @ [BS, B, C, D_col, M] -> [BS, B, C, N_C, M]
        scores = torch.matmul(q_r, K.transpose(-1, -2))

        # Mask inactive slots: em_S [BS, B, M] -> [BS, B, 1, 1, M]
        active_mask = (self.em_S > 0)[:, :, None, None, :]
        scores = scores.masked_fill(~active_mask, -1e9)

        # Top-k mask (gather-free)
        k = min(self.k_ret, self.M)
        topk_vals, _ = scores.topk(k, dim=-1)           # [BS, B, C, N_C, k]
        threshold = topk_vals[..., -1:].detach()         # [BS, B, C, N_C, 1]
        topk_mask = scores >= threshold                  # [BS, B, C, N_C, M]

        # Masked softmax attention
        attn_logits = scores.masked_fill(~topk_mask, -1e9)
        attn_weights = F.softmax(attn_logits, dim=-1)    # [BS, B, C, N_C, M]

        # Weighted sum: [BS, B, C, N_C, M] @ [BS, B, C, M, D_col] -> [BS, B, C, N_C, D_col]
        out = torch.matmul(attn_weights, V)

        # Back to [BS, N_C, B, C, D_col]
        return out.permute(0, 3, 1, 2, 4)

    def score_novelty(self, q_nov: Tensor, surprise: Tensor, w_nov: Tensor) -> Tensor:
        """Score novelty for candidate selection.

        q_nov: [BS, N, B, D], surprise: [BS, N, B], w_nov: [BS, N, B]
        Returns: novelty [BS, N, B]
        """
        # Max cosine similarity against em_K via batched matmul (no copy)
        q_bn = q_nov.transpose(1, 2)                               # [BS, B, N, D]
        sim = torch.matmul(q_bn, self.em_K.transpose(-1, -2))      # [BS, B, N, M]
        sim = sim.transpose(1, 2)                                   # [BS, N, B, M]

        active_mask = (self.em_S > 0)[:, None, :, :]  # [BS, 1, B, M]
        sim = sim.masked_fill(~active_mask, -1.0)
        max_sim = sim.max(dim=-1).values.clamp(min=0.0)  # [BS, N, B]

        novelty = w_nov * surprise + (1 - w_nov) * (1 - max_sim)
        return novelty

    def select_top_candidates(self, q_nov: Tensor, v_nov: Tensor,
                               novelty: Tensor, C_cand: int):
        """Select top-C_cand candidates across all N positions per (BS, B).

        q_nov, v_nov: [BS, N, B, D]
        novelty: [BS, N, B]
        Returns: (cand_K, cand_V, cand_scores) each [BS, B, C_cand, D] or [BS, B, C_cand]
        """
        BS, N, B, D = q_nov.shape

        # Permute novelty to [BS, B, N]
        nov_flat = novelty.permute(0, 2, 1)  # [BS, B, N]

        C_cand = min(C_cand, N)
        topk_scores, topk_idx = nov_flat.topk(C_cand, dim=-1)  # [BS, B, C_cand]

        # Gather via torch.gather (compile-friendly, no advanced indexing)
        # Permute to [BS, B, N, D] so gather dim=2 selects positions
        q_bn = q_nov.transpose(1, 2)  # [BS, B, N, D]
        v_bn = v_nov.transpose(1, 2)  # [BS, B, N, D]
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # [BS, B, C_cand, D]
        cand_K = torch.gather(q_bn, 2, idx_exp)  # [BS, B, C_cand, D]
        cand_V = torch.gather(v_bn, 2, idx_exp)  # [BS, B, C_cand, D]

        return cand_K, cand_V, topk_scores

    def write(self, cand_K: Tensor, cand_V: Tensor, cand_scores: Tensor,
              g_em: Tensor, tau: Tensor, ww: Tensor | None = None):
        """Write candidates to EM via EMA.

        cand_K: [BS, B, C_cand, D]
        cand_V: [BS, B, C_cand, D]
        cand_scores: [BS, B, C_cand]
        g_em: [BS, B]
        tau: [BS, B]
        ww: [BS, B] — learned weakness weight (None falls back to 0.5)
        """
        BS, B, C_cand, D = cand_K.shape

        # Score candidates against all slots via batched matmul (no copy)
        cand_K_norm = unit_normalize(cand_K)
        # [BS, B, C_cand, D] @ [BS, B, D, M] → [BS, B, C_cand, M]
        slot_scores = torch.matmul(cand_K_norm, self.em_K.transpose(-1, -2))

        # Weakness bias: prefer weaker slots (learned weight)
        weakness = -self.em_S.unsqueeze(2)  # [BS, B, 1, M]
        if ww is not None:
            slot_scores = slot_scores + ww.reshape(BS, B, 1, 1) * weakness
        else:
            slot_scores = slot_scores + 0.5 * weakness

        # Softmax slot selection
        slot_weights = F.softmax(
            slot_scores / tau.reshape(BS, B, 1, 1).clamp(min=0.01), dim=-1
        )  # [BS, B, C_cand, M]

        # Write strength per candidate
        cand_weight = cand_scores / (cand_scores.sum(dim=-1, keepdim=True) + 1e-8)
        alpha = g_em.reshape(BS, B, 1, 1) * cand_weight.unsqueeze(-1) * slot_weights  # [BS, B, C_cand, M]

        # Sum across candidates for each slot
        alpha_per_slot = alpha.sum(dim=2)  # [BS, B, M]

        # Weighted candidate blend per slot via batched matmul (no copy)
        # [BS, B, C_cand, M] → transpose → [BS, B, M, C_cand] @ [BS, B, C_cand, D]
        alpha_t = alpha.transpose(2, 3)                             # [BS, B, M, C_cand]
        blended_K = torch.matmul(alpha_t, cand_K_norm)             # [BS, B, M, D]
        blended_V = torch.matmul(alpha_t, cand_V)                  # [BS, B, M, D]

        # Normalize by alpha
        denom = alpha_per_slot.unsqueeze(-1).clamp(min=1e-8)
        blended_K = blended_K / denom
        blended_V = blended_V / denom

        # EMA update (em_K/V: reassignment required — saved for backward by em.read())
        update_mask = (alpha_per_slot > 1e-8).unsqueeze(-1)  # [BS, B, M, 1]
        alpha_exp = alpha_per_slot.unsqueeze(-1).clamp(max=1.0)

        new_K = (1 - alpha_exp) * self.em_K + alpha_exp * unit_normalize(blended_K)
        new_V = (1 - alpha_exp) * self.em_V + alpha_exp * blended_V

        self.em_K = torch.where(update_mask, new_K, self.em_K)
        self.em_V = torch.where(update_mask, new_V, self.em_V)

        # Strength update (reassignment: alpha_per_slot carries grad through neuromod params)
        self.em_S = (self.em_S + alpha_per_slot).clamp(0, self.S_max)

        # Age reset for updated slots
        self.em_age = self.em_age * (1 - alpha_per_slot)

        # Budget enforcement (per-stream-per-block: last dim is M)
        self.em_S = budget_enforce(self.em_S, self.budget)

    def base_decay(self, decay: Tensor):
        """Apply strength decay once per segment (called after R loop).

        decay: [BS, B] — neuromodulated decay factor.
        Must use reassignment: em_S is saved for backward by em.read().
        """
        if self.em_S is not None:
            self.em_S = self.em_S * decay.unsqueeze(-1)

    def age_tick(self, n_tokens: int):
        """Increment age of active slots."""
        if self.em_age is not None:
            active = (self.em_S > 0).to(self.em_age.dtype)
            self.em_age = self.em_age + n_tokens * active

    def reset_states(self, mask: Tensor):
        """Full reset: zero S, age, K, and V for masked streams.

        Clears K/V to prevent stale keys from influencing write() slot
        selection via cosine similarity (Bug 4).

        mask: [BS] bool.
        """
        if self.em_S is None:
            return
        expanded = mask[:, None, None]           # [BS, 1, 1]
        expanded_kv = mask[:, None, None, None]  # [BS, 1, 1, 1]
        self.em_S = self.em_S * ~expanded
        self.em_age = self.em_age * ~expanded
        self.em_K = self.em_K * ~expanded_kv
        self.em_V = self.em_V * ~expanded_kv


class EMNeuromodulator(nn.Module):
    """Neuromodulator for EM write decisions.

    Returns: g_em [BSB], tau [BSB], decay [BSB], ww [BSB]
    (operates on flattened BS*B vectors from model.py)
    """

    def __init__(self, D_mem: int, config: ModelConfig):
        super().__init__()
        self.em_enabled = config.em_enabled
        self.default_g = config.g_em_default
        self.default_tau = config.tau_em
        self.default_decay = config.decay_em
        self.default_ww = config.ww_em_default
        self.g_em_floor = config.g_em_floor
        self.g_em_ceil = config.g_em_ceil
        self.tau_floor = config.tau_em_floor
        self.tau_ceil = config.tau_em_ceil
        self.ww_floor = config.ww_em_floor
        self.ww_ceil = config.ww_em_ceil
        self.decay_floor = config.decay_em_floor
        self.decay_ceil = config.decay_em_ceil

        if self.em_enabled:
            H = config.neuromod_hidden
            n_scalar = 2  # novelty_mean, em_usage
            n_content = config.content_proj_dim
            n_features = n_scalar + n_content

            self.content_proj = nn.Linear(D_mem, n_content)
            self.backbone = nn.Sequential(
                nn.Linear(n_features, H),
                nn.ReLU(),
            )
            self.g_head = nn.Linear(H, 1)
            self.tau_head = nn.Linear(H, 1)
            self.decay_head = nn.Linear(H, 1)
            self.ww_head = nn.Linear(H, 1)

            nn.init.zeros_(self.backbone[0].bias)
            nn.init.normal_(self.content_proj.weight, std=0.01)
            nn.init.zeros_(self.content_proj.bias)

            # Bias init: sigmoid(bias) = (default - floor) / (ceil - floor)
            _ww_range = self.ww_ceil - self.ww_floor
            if _ww_range > 0:
                _target_sigmoid = (self.default_ww - self.ww_floor) / _ww_range
                _target_sigmoid = max(1e-4, min(1 - 1e-4, _target_sigmoid))
                import math
                self.ww_head.bias.data.fill_(math.log(_target_sigmoid / (1 - _target_sigmoid)))

    def forward(self, novelty_mean: Tensor, em_usage: Tensor,
                content_emb: Tensor | None = None):
        """
        novelty_mean: [BSB]
        em_usage: [BSB]
        content_emb: [BSB, D] (optional)

        Returns: g_em [BSB], tau [BSB], decay [BSB], ww [BSB]
        """
        if not self.em_enabled:
            BSB = novelty_mean.shape[0]
            device = novelty_mean.device
            return (
                torch.full((BSB,), self.default_g, device=device),
                torch.full((BSB,), self.default_tau, device=device),
                torch.full((BSB,), self.default_decay, device=device),
                torch.full((BSB,), self.default_ww, device=device),
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

        ww_raw = self.ww_head(h).squeeze(-1)
        ww = self.ww_floor + (self.ww_ceil - self.ww_floor) * torch.sigmoid(ww_raw)

        return g_em, tau, decay, ww
