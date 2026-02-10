"""
Shared span-boundary operations for neuromorphic LM training.

Extracted from trainer.py and validation.py to eliminate ~200 lines of
copy-pasted logic. These are free functions (no shared mutable state).
Used by trainer.py, validation.py, and rl_rollout.py.

Design note: In the parallel (forward_span) path, PM eligibility and EM
candidates are accumulated over the full span, then committed at the boundary.
Mid-span doc-boundary resets zero the surprise accumulators. PM eligibility
IS reset at doc boundaries via the carry mask in update_eligibility_batch
(carry=0 at reset positions zeros the recurrent eligibility state). EM
candidates are masked via cand_valid to exclude tokens before the last
reset within the span.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from ..model.config import ModelConfig
from ..model.model import NeuromorphicLM


# ------------------------------------------------------------------
# Typed containers for per-span state
# ------------------------------------------------------------------

@dataclass
class SpanAccumulator:
    """Mutable per-span state. Created at span start, mutated during token loop."""
    surprise_accum: Tensor      # [BS]
    valid_tokens: Tensor        # [BS]
    last_reset: Tensor          # [BS] long
    em_cand_K: list             # [B] of list[Tensor]
    em_cand_V: list             # [B] of list[Tensor]
    em_cand_score: list         # [B] of list[Tensor]
    em_cand_valid: list         # [B] of list[Tensor]

    @staticmethod
    def create(BS: int, num_blocks: int, device: torch.device) -> SpanAccumulator:
        return SpanAccumulator(
            surprise_accum=torch.zeros(BS, device=device),
            valid_tokens=torch.zeros(BS, device=device),
            last_reset=torch.zeros(BS, dtype=torch.long, device=device),
            em_cand_K=[[] for _ in range(num_blocks)],
            em_cand_V=[[] for _ in range(num_blocks)],
            em_cand_score=[[] for _ in range(num_blocks)],
            em_cand_valid=[[] for _ in range(num_blocks)],
        )

    def reset_span(self):
        """Zero accumulators for a new span (reuses tensors)."""
        self.surprise_accum.zero_()
        self.valid_tokens.zero_()
        self.last_reset.zero_()
        B = len(self.em_cand_K)
        self.em_cand_K = [[] for _ in range(B)]
        self.em_cand_V = [[] for _ in range(B)]
        self.em_cand_score = [[] for _ in range(B)]
        self.em_cand_valid = [[] for _ in range(B)]

    def finalize(self, device: torch.device, config: ModelConfig) -> SpanResult:
        """Compute span_surprise_mean and stack EM candidates."""
        surprise_mean = self.surprise_accum / self.valid_tokens.clamp(min=1)
        if config.em_enabled:
            em_stacked = stack_em_candidates(
                self.em_cand_K, self.em_cand_V,
                self.em_cand_score, self.em_cand_valid,
                self.last_reset, device,
            )
        else:
            em_stacked = {}
        return SpanResult(surprise_mean=surprise_mean, em_stacked=em_stacked)


@dataclass(frozen=True)
class SpanResult:
    """Immutable output of one span's boundary computation."""
    surprise_mean: Tensor       # [BS]
    em_stacked: dict            # {b: (sK, sV, sScore, sValid, novelty)}


# ------------------------------------------------------------------
# Loss / reset mask computation
# ------------------------------------------------------------------

def compute_loss_mask(
    span_ids: Tensor, eot_id: int, reset_on_doc_boundary: bool,
) -> tuple[Tensor, Tensor]:
    """Return (is_eot_all [BS, span_P], loss_mask_all [BS, span_P]).

    When reset_on_doc_boundary is True, EOT input positions are masked out.
    Otherwise all positions are valid.
    """
    is_eot_all = (span_ids == eot_id)
    if reset_on_doc_boundary:
        loss_mask_all = ~is_eot_all
    else:
        loss_mask_all = torch.ones_like(is_eot_all)
    return is_eot_all, loss_mask_all


def compute_reset_mask(
    model: NeuromorphicLM,
    span_ids: Tensor,
    reset_first: Tensor,
    reset_on_doc_boundary: bool,
) -> Tensor:
    """Return reset_mask_all [BS, span_P], zeroed if not reset_on_doc_boundary."""
    reset_mask_all = model._compute_reset_masks(span_ids, reset_first)
    if not reset_on_doc_boundary:
        reset_mask_all = torch.zeros_like(reset_mask_all)
    return reset_mask_all


# ------------------------------------------------------------------
# Surprise accumulation
# ------------------------------------------------------------------

def accumulate_span_surprise(
    token_surprise: Tensor,
    loss_mask_all: Tensor,
    reset_mask_all: Tensor,
    reset_on_doc_boundary: bool,
    span_surprise_accum: Tensor,
    span_valid_tokens: Tensor,
    span_last_reset: Tensor,
) -> Tensor:
    """Per-token loop: accumulate surprise, handle resets.

    Mutates span_surprise_accum, span_valid_tokens, span_last_reset in-place.
    Returns span_surprise_mean [BS].
    """
    span_P = loss_mask_all.shape[1]

    for t_local in range(span_P):
        reset_t = reset_mask_all[:, t_local]
        if reset_t.any() and reset_on_doc_boundary:
            span_last_reset[reset_t] = t_local
            span_surprise_accum[reset_t] = 0
            span_valid_tokens[reset_t] = 0

        lm = loss_mask_all[:, t_local]
        span_surprise_accum.add_(token_surprise[:, t_local, 0] * lm.float())
        span_valid_tokens.add_(lm.float())

    return span_surprise_accum / span_valid_tokens.clamp(min=1)


# ------------------------------------------------------------------
# PM eligibility
# ------------------------------------------------------------------

def apply_pm_eligibility_batch(
    model: NeuromorphicLM,
    x_emb_all: Tensor,
    token_surprise: Tensor,
    reset_mask_all: Tensor,
    config: ModelConfig,
) -> None:
    """Project embeddings via W_in, route to blocks, call update_eligibility_batch.

    Modifies PM eligibility traces in-place.
    """
    BS, span_P = token_surprise.shape[:2]

    x_proj_all = model.W_in(x_emb_all)  # [BS, span_P, D]
    x_blocks_all = x_proj_all.view(
        BS, span_P, config.B, config.D_h
    )  # [BS, span_P, B, D_h]

    for b, block in enumerate(model.blocks):
        for layer in block.layers:
            if layer._last_h_all is None:
                continue

            l_idx = layer.layer_idx
            if l_idx == 0:
                x_in = x_blocks_all[:, :, b]  # [BS, span_P, D_h]
            else:
                x_in = block.layers[l_idx - 1]._last_h_all

            h_out = layer._last_h_all  # [BS, span_P, D_h]

            layer.pm.update_eligibility_batch(
                x_in, h_out, token_surprise, reset_mask_all,
            )


# ------------------------------------------------------------------
# EM candidate building
# ------------------------------------------------------------------

def propose_em_candidates(
    model: NeuromorphicLM,
    x_emb_all: Tensor,
    y_wm_all: Tensor,
    token_surprise: Tensor,
    loss_mask_all: Tensor,
    cand_K: list,
    cand_V: list,
    cand_score: list,
    cand_token_valid: list,
) -> None:
    """Propose EM candidates per block. Appends to the cand_* lists in-place.

    Stacking is deferred to SpanAccumulator.finalize() or stack_em_candidates().
    """
    for b, block in enumerate(model.blocks):
        h_final_all = block.layers[-1]._last_h_all
        if h_final_all is not None:
            k_c, v_c, nov = block.em.propose_candidate_batch(
                x_emb_all, y_wm_all, h_final_all, token_surprise,
            )
            cand_K[b].append(k_c)
            cand_V[b].append(v_c)
            cand_score[b].append(nov)
            cand_token_valid[b].append(loss_mask_all)


def stack_em_candidates(
    cand_K: list,
    cand_V: list,
    cand_score: list,
    cand_token_valid: list,
    span_last_reset: Tensor,
    device: torch.device,
) -> dict[int, tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
    """Stack candidate buffers into per-block tensors.

    Returns {b: (sK, sV, sScore, sValid, novelty_mean)}.
    """
    em_stacked = {}
    B = len(cand_K)
    for b in range(B):
        if len(cand_K[b]) > 0:
            sK = torch.cat(cand_K[b], dim=1)
            sV = torch.cat(cand_V[b], dim=1)
            sScore = torch.cat(cand_score[b], dim=1)
            sTokValid = torch.cat(cand_token_valid[b], dim=1)
            S = sScore.shape[1]
            pos = torch.arange(S, device=device).unsqueeze(0)
            sValid = (
                pos >= span_last_reset.unsqueeze(1)
            ) & sTokValid.bool()
            cvf = sValid.float()
            cc = cvf.sum(dim=-1).clamp(min=1)
            novelty = (sScore * cvf).sum(dim=-1) / cc
            em_stacked[b] = (sK, sV, sScore, sValid, novelty)
    return em_stacked


# ------------------------------------------------------------------
# Span-boundary commits and writes
# ------------------------------------------------------------------

def apply_pm_boundary(
    model: NeuromorphicLM,
    span_surprise_mean: Tensor,
) -> dict:
    """base_decay() all PM instances, then commit_at_boundary().

    Returns commit_info dict.
    """
    for block in model.blocks:
        for layer in block.layers:
            layer.pm.base_decay()

    return model.commit_at_boundary(
        span_surprise=span_surprise_mean.detach()
    )


def apply_em_boundary(
    model: NeuromorphicLM,
    em_stacked: dict,
    span_surprise_mean: Tensor,
    config: ModelConfig,
) -> list[tuple[int, Tensor, float, float]]:
    """For each block: neuromod forward + write_at_boundary.

    Returns list of (block_idx, write_mask, novelty_mean, g_em_mean)
    for collector recording.
    """
    write_info = []
    for b, block in enumerate(model.blocks):
        if b not in em_stacked:
            continue
        sK, sV, sScore, sValid, cand_novelty_mean = em_stacked[b]

        em_usage = (
            block.em.em_S.sum(dim=-1)
            if block.em.em_S is not None
            else torch.zeros_like(span_surprise_mean)
        )

        write_mask, g_em, tau_em, ww_em = block.em_neuromodulator.forward(
            span_surprise_mean,
            em_usage / config.budget_em,
            cand_novelty_mean,
        )

        write_info.append((
            b, write_mask,
            cand_novelty_mean.mean().item(),
            g_em.mean().item(),
        ))

        block.em.write_at_boundary(
            sK, sV, sScore,
            write_mask, g_em,
            tau=tau_em, weakness_weight=ww_em,
            cand_valid=sValid,
        )

    return write_info
