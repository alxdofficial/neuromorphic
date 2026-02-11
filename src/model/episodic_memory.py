"""
Episodic Memory — fixed-size per-stream vector store.

One EM instance per block (B total). Provides long-range recall via
novelty-based writes and top-k retrieval with cross-attention aggregation.

State per stream:
    em_K: [BS, M, D_em]   — keys (unit-normalized)
    em_V: [BS, M, D_em]   — values
    em_S: [BS, M]          — strengths (bounded)
"""

import math

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

        # Learned novelty weight adjuster (Phase C+, backprop only)
        self.W_nov = nn.Linear(D + D, 1) if config.em_enabled else None

        # Post-retrieval MLP: processes the aggregated retrieved memory
        if config.em_readout_ffn:
            self.readout_norm = nn.LayerNorm(config.D_em)
            self.readout_ffn = nn.Sequential(
                nn.Linear(config.D_em, config.D_em * 4),
                nn.GELU(),
                nn.Linear(config.D_em * 4, config.D_em),
            )
        else:
            self.readout_norm = None
            self.readout_ffn = None

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
        # Attention: q_cross against V_top as both keys and values.
        # Adding topk_scores preserves a differentiable path from retrieval query
        # (W_q_em) to downstream loss through selected memory logits.
        attn = torch.einsum("bd, bkd -> bk", q_cross, V_top) * self.cross_scale
        attn = attn + topk_scores
        # Mask out -inf topk positions (no active slots)
        attn = attn.masked_fill(topk_scores == float("-inf"), float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)

        out = torch.einsum("bk, bkd -> bd", attn, V_top)  # [BS, D_em]

        # Post-retrieval processing
        if self.readout_ffn is not None:
            out = out + self.readout_ffn(self.readout_norm(out))

        y_em = self.W_o_cross(out)  # [BS, D]

        return y_em

    def retrieve_batch(self, x_all: Tensor, y_wm_all: Tensor) -> Tensor:
        """Query EM for P tokens in parallel.

        Args:
            x_all: [BS, P, D] — token embeddings
            y_wm_all: [BS, P, D] — working memory outputs

        Returns:
            y_em_all: [BS, P, D]
        """
        BS, P, D = x_all.shape
        device = x_all.device

        if self.em_K is None:
            self._lazy_init(BS, device)

        # Compute queries: [BS, P, D_em]
        q = unit_normalize(self.W_q_em(torch.cat([x_all, y_wm_all], dim=-1)))

        # Score against all slots: [BS, P, M]
        scores = torch.einsum("bpd, bmd -> bpm", q, self.em_K)

        # Mask inactive slots
        active_mask = self.em_S > 0  # [BS, M]
        scores = scores.masked_fill(~active_mask.unsqueeze(1), float("-inf"))

        # Top-k retrieval per token
        k = min(self.k_ret, self.M)
        topk_scores, topk_idx = scores.topk(k, dim=-1)  # [BS, P, k]

        # Gather values: expand indices to [BS, P, k, D_em]
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.D_em)
        em_V_exp = self.em_V.unsqueeze(1).expand(-1, P, -1, -1)  # [BS, P, M, D_em]
        V_top = em_V_exp.gather(2, topk_idx_exp)  # [BS, P, k, D_em]

        # Cross-attention: [BS, P, D_em]
        q_cross = self.W_q_cross(x_all)
        attn = torch.einsum("bpd, bpkd -> bpk", q_cross, V_top) * self.cross_scale
        attn = attn + topk_scores
        attn = attn.masked_fill(topk_scores == float("-inf"), float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)

        out = torch.einsum("bpk, bpkd -> bpd", attn, V_top)  # [BS, P, D_em]

        if self.readout_ffn is not None:
            out = out + self.readout_ffn(self.readout_norm(out))

        y_em_all = self.W_o_cross(out)  # [BS, P, D]
        return y_em_all

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

        # Novelty = surprise + (1 - max cosine similarity to active keys)
        cold_start = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        if self.em_K is not None:
            cos_sim = torch.einsum("bd, bmd -> bm", k_cand, self.em_K)  # [BS, M]
            if self.em_S is not None:
                active_mask = self.em_S > 0
                masked = cos_sim.masked_fill(~active_mask, float("-inf"))
                any_active = active_mask.any(dim=-1)
                max_sim = torch.where(
                    any_active,
                    masked.max(dim=-1).values,
                    torch.zeros_like(masked[..., 0]),
                )
                cold_start = ~any_active
            else:
                max_sim = cos_sim.max(dim=-1).values
                cold_start = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        else:
            max_sim = torch.zeros(x.shape[0], device=x.device)

        surprise_1d = surprise.squeeze(-1)
        if self.W_nov is not None:
            w_nov = torch.sigmoid(self.W_nov(torch.cat([x, y_wm], dim=-1))).squeeze(-1)
            novelty = (w_nov * surprise_1d + (1.0 - w_nov) * (1.0 - max_sim)).clamp(0.0, 1.0)
        else:
            novelty = (0.5 * surprise_1d + 0.5 * (1.0 - max_sim)).clamp(0.0, 1.0)

        # Cold start: when no active slots, similarity term is undefined.
        # Use surprise only — don't get a free +0.5 from max_sim=0.
        novelty = torch.where(cold_start, surprise_1d.clamp(0.0, 1.0), novelty)

        return k_cand, v_cand, novelty

    def propose_candidate_batch(self, x_all: Tensor, y_wm_all: Tensor,
                                h_final_all: Tensor,
                                surprise_all: Tensor) -> tuple:
        """Per-token candidate proposal for P tokens in parallel.

        Args:
            x_all: [BS, P, D] — token embeddings
            y_wm_all: [BS, P, D] — WM outputs
            h_final_all: [BS, P, D_h] — final layer outputs
            surprise_all: [BS, P, 1] — per-token surprise

        Returns:
            k_cand: [BS, P, D_em]
            v_cand: [BS, P, D_em]
            novelty: [BS, P]
        """
        BS, P, D = x_all.shape

        k_cand = unit_normalize(self.W_k_cand(torch.cat([x_all, y_wm_all], dim=-1)))
        v_cand = self.W_v_cand(h_final_all)

        # Novelty = surprise + (1 - max cosine similarity to active keys)
        cold_start = torch.ones(BS, dtype=torch.bool, device=x_all.device)
        if self.em_K is not None:
            # [BS, P, M]
            cos_sim = torch.einsum("bpd, bmd -> bpm", k_cand, self.em_K)
            if self.em_S is not None:
                active_mask = self.em_S > 0  # [BS, M]
                masked = cos_sim.masked_fill(
                    ~active_mask.unsqueeze(1), float("-inf")
                )
                any_active = active_mask.any(dim=-1)  # [BS]
                max_sim = masked.max(dim=-1).values    # [BS, P]
                max_sim = torch.where(
                    any_active.unsqueeze(1),
                    max_sim,
                    torch.zeros_like(max_sim),
                )
                cold_start = ~any_active
            else:
                max_sim = cos_sim.max(dim=-1).values
                cold_start = torch.zeros(BS, dtype=torch.bool, device=x_all.device)
        else:
            max_sim = torch.zeros(BS, P, device=x_all.device)

        surprise_2d = surprise_all.squeeze(-1)  # [BS, P]
        if self.W_nov is not None:
            w_nov = torch.sigmoid(
                self.W_nov(torch.cat([x_all, y_wm_all], dim=-1))
            ).squeeze(-1)  # [BS, P]
            novelty = (w_nov * surprise_2d + (1.0 - w_nov) * (1.0 - max_sim)).clamp(0.0, 1.0)
        else:
            novelty = (0.5 * surprise_2d + 0.5 * (1.0 - max_sim)).clamp(0.0, 1.0)

        # Cold start: use surprise only
        novelty = torch.where(
            cold_start.unsqueeze(1),
            surprise_2d.clamp(0.0, 1.0),
            novelty,
        )

        return k_cand, v_cand, novelty

    def write_at_boundary(self, cand_K: Tensor, cand_V: Tensor,
                          cand_score: Tensor, write_mask: Tensor,
                          g_em: Tensor, tau=None, weakness_weight=None,
                          cand_valid: Tensor = None):
        """Span-boundary EM write. Top-C candidates, soft multi-slot EMA.

        Args:
            cand_K: [BS, P, D_em] — candidate keys from span
            cand_V: [BS, P, D_em] — candidate values from span
            cand_score: [BS, P] — novelty scores from span
            write_mask: [BS] bool — which streams write
            g_em: [BS] — write strength per stream
            tau: [BS] or scalar or None — soft top-k temperature (default: self.tau)
            weakness_weight: [BS] or scalar or None — weakness weight (default: self.weakness_weight)
            cand_valid: [BS, P] bool — optional candidate validity mask
        """
        if self.em_K is None:
            return

        BS = write_mask.shape[0]
        device = cand_score.device

        if cand_valid is None:
            cand_valid = torch.ones_like(cand_score, dtype=torch.bool, device=device)
        else:
            cand_valid = cand_valid.bool()

        # Select top-C candidates per stream
        C = min(self.C, cand_score.shape[1])
        masked_scores = cand_score.masked_fill(~cand_valid, float("-inf"))
        topC_scores, topC_idx = masked_scores.topk(C, dim=-1)  # [BS, C]

        topC_idx_exp = topC_idx.unsqueeze(-1).expand(-1, -1, self.D_em)
        K_C = cand_K.gather(1, topC_idx_exp)  # [BS, C, D_em]
        V_C = cand_V.gather(1, topC_idx_exp)  # [BS, C, D_em]

        _tau = tau if tau is not None else self.tau
        _ww = weakness_weight if weakness_weight is not None else self.weakness_weight

        if write_mask.any():
            # For each candidate, soft-select slots to update
            for c in range(C):
                k_c = K_C[:, c]   # [BS, D_em]
                v_c = V_C[:, c]   # [BS, D_em]

                # Score against existing slots
                scores_slot = torch.einsum("bd, bmd -> bm", k_c, self.em_K)  # [BS, M]

                # Weakness bias: prefer overwriting weak slots
                if isinstance(_ww, Tensor) and _ww.dim() >= 1:
                    scores_slot = scores_slot - _ww.unsqueeze(-1) * self.em_S
                else:
                    scores_slot = scores_slot - _ww * self.em_S

                # Soft top-k selection
                w = soft_topk(scores_slot, self.k_write, _tau)  # [BS, M]

                # Apply write mask and strength
                score_ok = torch.isfinite(topC_scores[:, c])
                cand_write_mask = write_mask & score_ok
                alpha = w * g_em.unsqueeze(-1) * cand_write_mask.float().unsqueeze(-1)  # [BS, M]

                # EMA update
                alpha_3d = alpha.unsqueeze(-1)  # [BS, M, 1]
                self.em_K = unit_normalize(
                    (1 - alpha_3d) * self.em_K + alpha_3d * k_c.unsqueeze(1)
                )
                self.em_V = (1 - alpha_3d) * self.em_V + alpha_3d * v_c.unsqueeze(1)

                # Strength update
                score_strength = torch.where(
                    score_ok, topC_scores[:, c], torch.zeros_like(topC_scores[:, c])
                )
                self.em_S = (self.em_S + alpha * score_strength.unsqueeze(-1)).clamp(
                    0.0, self.S_max
                )

        # Always decay strengths each boundary (even if no writes happened).
        self.em_S = self.em_S * self.decay
        self.em_S = budget_enforce(self.em_S, self.budget)


class EMNeuromodulator(nn.Module):
    """Neuromodulator for EM write strength, temperature, and weakness weight.

    Phase A–B (em_enabled=False): empty shell, no parameters.
    Phase C (em_enabled=True, rl_enabled=False): learned g_em via backbone
        + g_head (main optimizer backprop), with heuristic write gate
        (novelty > threshold). g_em clamped to [g_em_floor, g_em_ceil].
        tau and ww are learned via backbone + heads.
    Phase D+ (em_enabled=True, rl_enabled=True): always write, learned g_em,
        tau, ww (dual-trained: main loss backprop + RL counterfactual).

    Returns:
        write_mask: [BS] bool (always True in Phase D+ learned mode)
        g_em: [BS] write strength
        tau: [BS] soft top-k temperature (or None in heuristic mode)
        ww: [BS] weakness weight (or None in heuristic mode)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.rl_enabled = config.rl_enabled
        self.em_enabled = config.em_enabled
        self.novelty_threshold = config.novelty_threshold
        self.default_g = config.g_em_default
        self.default_tau = config.tau_em
        self.default_ww = config.weakness_weight_em
        self.g_em_floor = config.g_em_floor
        self.g_em_ceil = config.g_em_ceil
        self.tau_floor = config.tau_em_floor
        self.tau_ceil = config.tau_em_ceil
        self.ww_floor = config.ww_em_floor
        self.ww_ceil = config.ww_em_ceil

        # Backbone + heads: created when EM is enabled (Phase C+)
        if self.em_enabled:
            H = config.rl_controller_hidden
            self.backbone = nn.Sequential(
                nn.Linear(3, H),
                nn.ReLU(),
            )
            self.g_head = nn.Linear(H, 1)
            self.tau_head = nn.Linear(H, 1)
            self.ww_head = nn.Linear(H, 1)

            # Zero backbone bias: ensures zero input → zero backbone output
            # → head outputs equal their calibrated biases exactly at init.
            # Backbone weights stay Kaiming-init so non-zero inputs produce signal.
            nn.init.zeros_(self.backbone[0].bias)

            # Init heads so output at init matches heuristic defaults.
            # Zero weights + calibrated bias = constant default until
            # gradients make the output input-dependent.
            # g_em: sigmoid maps to [g_em_floor, g_em_ceil], init to g_em_default
            g_frac = (config.g_em_default - config.g_em_floor) / max(config.g_em_ceil - config.g_em_floor, 1e-8)
            g_frac = max(min(g_frac, 0.999), 0.001)
            nn.init.normal_(self.g_head.weight, std=0.01)
            nn.init.constant_(self.g_head.bias, math.log(g_frac / (1.0 - g_frac)))

            tau_frac = (config.tau_em - config.tau_em_floor) / max(config.tau_em_ceil - config.tau_em_floor, 1e-8)
            tau_frac = max(min(tau_frac, 0.999), 0.001)  # avoid log(0)
            nn.init.normal_(self.tau_head.weight, std=0.01)
            nn.init.constant_(self.tau_head.bias, math.log(tau_frac / (1.0 - tau_frac)))

            ww_frac = (config.weakness_weight_em - config.ww_em_floor) / max(config.ww_em_ceil - config.ww_em_floor, 1e-8)
            ww_frac = max(min(ww_frac, 0.999), 0.001)
            nn.init.normal_(self.ww_head.weight, std=0.01)
            nn.init.constant_(self.ww_head.bias, math.log(ww_frac / (1.0 - ww_frac)))

    def forward(self, span_surprise: Tensor, em_usage: Tensor,
                cand_novelty_mean: Tensor) -> tuple:
        if self.em_enabled:
            if self.rl_enabled:
                return self._forward_learned(span_surprise, em_usage, cand_novelty_mean)
            return self._forward_continuous(span_surprise, em_usage, cand_novelty_mean)
        return self._forward_heuristic(span_surprise, em_usage, cand_novelty_mean)

    def _forward_heuristic(self, span_surprise, em_usage, cand_novelty_mean):
        """No learnable params (Phase A–B fallback)."""
        write_mask = cand_novelty_mean > self.novelty_threshold
        g_em = torch.full_like(span_surprise, self.default_g)
        tau = torch.full_like(span_surprise, self.default_tau)
        ww = torch.full_like(span_surprise, self.default_ww)
        return write_mask, g_em, tau, ww

    def _forward_continuous(self, span_surprise, em_usage, cand_novelty_mean):
        """Heuristic write gate + learned g_em/tau/ww (Phase C)."""
        write_mask = cand_novelty_mean > self.novelty_threshold

        features = torch.stack([span_surprise, em_usage, cand_novelty_mean], dim=-1)
        h = self.backbone(features)

        raw_g = torch.sigmoid(self.g_head(h)).squeeze(-1)
        g_em = self.g_em_floor + (self.g_em_ceil - self.g_em_floor) * raw_g

        raw_tau = torch.sigmoid(self.tau_head(h)).squeeze(-1)
        tau = self.tau_floor + (self.tau_ceil - self.tau_floor) * raw_tau

        raw_ww = torch.sigmoid(self.ww_head(h)).squeeze(-1)
        ww = self.ww_floor + (self.ww_ceil - self.ww_floor) * raw_ww

        return write_mask, g_em, tau, ww

    def _forward_learned(self, span_surprise, em_usage, cand_novelty_mean):
        """Always write + learned g_em/tau/ww (Phase D+)."""
        features = torch.stack([span_surprise, em_usage, cand_novelty_mean], dim=-1)
        h = self.backbone(features)

        raw_g = torch.sigmoid(self.g_head(h)).squeeze(-1)
        g_em = self.g_em_floor + (self.g_em_ceil - self.g_em_floor) * raw_g

        raw_tau = torch.sigmoid(self.tau_head(h)).squeeze(-1)
        tau = self.tau_floor + (self.tau_ceil - self.tau_floor) * raw_tau

        raw_ww = torch.sigmoid(self.ww_head(h)).squeeze(-1)
        ww = self.ww_floor + (self.ww_ceil - self.ww_floor) * raw_ww

        write_mask = torch.ones_like(g_em, dtype=torch.bool)
        return write_mask, g_em, tau, ww
