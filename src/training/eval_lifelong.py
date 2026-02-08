"""
Lifelong learning evaluation — Phase E evaluation tasks.

Three evaluation modes:
1. Domain adaptation: stream Wikipedia articles, measure per-chunk perplexity decrease.
2. Drift monitoring: after domain adaptation, measure perplexity on held-out general text.
3. Cross-document recall: stream factual text, probe at increasing distances.

Usage:
    python -m src.training.eval_lifelong --checkpoint path/to/ckpt.pt --eval domain_adaptation
"""

import math
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..model.config import ModelConfig
from ..model.model import NeuromorphicLM
from ..model.state import save_runtime_state, load_runtime_state


@dataclass
class EvalResult:
    """Result from a lifelong evaluation run."""
    task: str
    chunk_perplexities: List[float]
    mean_ppl: float
    final_ppl: float
    ppl_trend: float  # negative = improving (good for domain adaptation)
    elapsed: float


def _compute_chunk_ppl(
    model: NeuromorphicLM,
    tokens: Tensor,
    config: ModelConfig,
    device: torch.device,
    prev_token: Optional[Tensor] = None,
) -> tuple:
    """Run model over a token chunk, return perplexity and last token.

    Args:
        model: the model (with persistent state)
        tokens: [BS, T] token ids for one chunk
        config: model config
        device: compute device
        prev_token: [BS] last token from previous chunk (for doc boundary
                    detection at t=0). If None, no reset at chunk start.

    Returns:
        (perplexity, last_token) — last_token is tokens[:, -1] for chaining.
    """
    BS, T = tokens.shape
    P = config.P
    eot_id = config.eot_id

    total_loss = 0.0
    valid_count = 0

    for span_start in range(0, T, P):
        span_end = min(span_start + P, T)
        span_surprise_accum = torch.zeros(BS, device=device)
        span_valid_tokens = torch.zeros(BS, device=device)  # per-stream count

        B = config.B
        cand_K = [[] for _ in range(B)]
        cand_V = [[] for _ in range(B)]
        cand_score = [[] for _ in range(B)]
        cand_token_valid = [[] for _ in range(B)]
        span_last_reset = torch.zeros(BS, dtype=torch.long, device=device)

        for t in range(span_start, span_end):
            # Doc boundary detection
            if t == 0:
                if prev_token is not None:
                    reset_mask = (prev_token == eot_id)
                else:
                    reset_mask = torch.zeros(BS, dtype=torch.bool, device=device)
            else:
                reset_mask = (tokens[:, t - 1] == eot_id)

            # Clear span-level accumulators at doc boundary for streams that
            # reset in this span (only when reset is enabled).
            if reset_mask.any() and config.reset_on_doc_boundary:
                local_t = t - span_start
                span_last_reset[reset_mask] = local_t
                span_surprise_accum[reset_mask] = 0
                span_valid_tokens[reset_mask] = 0

            logits, x_emb, y_wm = model.forward_one_token(tokens[:, t], reset_mask)

            # Target is next token (shift by 1)
            if t + 1 < T:
                targets = tokens[:, t + 1]
                is_eot = (tokens[:, t] == eot_id)
                if config.reset_on_doc_boundary:
                    loss_mask = ~is_eot
                else:
                    loss_mask = torch.ones_like(is_eot)

                if loss_mask.any():
                    loss = F.cross_entropy(
                        logits[loss_mask], targets[loss_mask], reduction="sum"
                    )
                    total_loss += loss.item()
                    valid_count += loss_mask.sum().item()

            # Update surprise
            if t + 1 < T:
                model.update_surprise(logits, tokens[:, t + 1], mask=loss_mask)
                if model.surprise is not None:
                    span_surprise_accum = (
                        span_surprise_accum
                        + model.surprise.squeeze(-1) * loss_mask.float()
                    )
                span_valid_tokens = span_valid_tokens + loss_mask.float()

                # Buffer EM candidates for span-boundary writes
                if config.em_enabled:
                    for b, block in enumerate(model.blocks):
                        h_final = block.layers[-1].h
                        if h_final is not None:
                            k_c, v_c, nov = block.em.propose_candidate(
                                x_emb,
                                y_wm,
                                h_final,
                                model.surprise,
                            )
                            cand_K[b].append(k_c)
                            cand_V[b].append(v_c)
                            cand_score[b].append(nov)
                            cand_token_valid[b].append(loss_mask)

        # Per-stream surprise mean for this span (used by both PM and EM)
        span_surprise_mean = span_surprise_accum / span_valid_tokens.clamp(min=1)

        # Span boundary commits
        if config.pm_enabled:
            for block in model.blocks:
                for layer in block.layers:
                    layer.pm.base_decay()
            model.commit_at_boundary(span_surprise=span_surprise_mean)

        # Span boundary EM writes
        if config.em_enabled:
            for b, block in enumerate(model.blocks):
                if len(cand_K[b]) > 0:
                    stacked_K = torch.stack(cand_K[b], dim=1)
                    stacked_V = torch.stack(cand_V[b], dim=1)
                    stacked_score = torch.stack(cand_score[b], dim=1)
                    stacked_token_valid = torch.stack(cand_token_valid[b], dim=1)

                    S = stacked_score.shape[1]
                    pos = torch.arange(S, device=device).unsqueeze(0)  # [1, S]
                    cand_valid = (
                        pos >= span_last_reset.unsqueeze(1)
                    ) & stacked_token_valid.bool()                     # [BS, S]

                    em_usage = (
                        block.em.em_S.sum(dim=-1)
                        if block.em.em_S is not None
                        else torch.zeros_like(span_surprise_mean)
                    )
                    cand_valid_f = cand_valid.float()
                    cand_count = cand_valid_f.sum(dim=-1).clamp(min=1)  # [BS]
                    cand_novelty_mean = (stacked_score * cand_valid_f).sum(dim=-1) / cand_count
                    write_mask, g_em, _p_write = block.em_neuromodulator.forward(
                        span_surprise_mean,
                        em_usage / config.budget_em,
                        cand_novelty_mean,
                    )
                    block.em.write_at_boundary(
                        stacked_K, stacked_V, stacked_score, write_mask, g_em,
                        cand_valid=cand_valid,
                    )

    # Detach states after chunk
    model.detach_states()

    last_token = tokens[:, -1]

    if valid_count == 0:
        return float("inf"), last_token
    avg_loss = total_loss / valid_count
    return min(math.exp(avg_loss), 1e6), last_token


def _linear_trend(values: List[float]) -> float:
    """Compute slope of linear fit. Negative = decreasing (improving)."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


def domain_adaptation(
    model: NeuromorphicLM,
    chunks: List[Tensor],
    config: ModelConfig,
    device: torch.device,
) -> EvalResult:
    """Stream document chunks, measure per-chunk perplexity decrease rate.

    In lifelong mode, the model should show decreasing perplexity as PM/EM
    accumulate domain knowledge across chunks. Without lifelong mode,
    perplexity should stay flat (no cross-doc memory).

    Args:
        model: model with state (will be modified in-place)
        chunks: list of [BS, T] token tensors (sequential document chunks)
        config: model config
        device: compute device

    Returns:
        EvalResult with per-chunk perplexities and trend.
    """
    model.eval()
    ppls = []
    prev_token = None
    t_start = time.time()

    with torch.no_grad():
        for chunk in chunks:
            chunk = chunk.to(device)
            ppl, prev_token = _compute_chunk_ppl(
                model, chunk, config, device, prev_token
            )
            ppls.append(ppl)

    elapsed = time.time() - t_start
    trend = _linear_trend(ppls)

    return EvalResult(
        task="domain_adaptation",
        chunk_perplexities=ppls,
        mean_ppl=sum(ppls) / len(ppls) if ppls else float("inf"),
        final_ppl=ppls[-1] if ppls else float("inf"),
        ppl_trend=trend,
        elapsed=elapsed,
    )


def drift_monitoring(
    model: NeuromorphicLM,
    domain_chunks: List[Tensor],
    general_chunks: List[Tensor],
    config: ModelConfig,
    device: torch.device,
    baseline_ppl: Optional[float] = None,
    drift_threshold: float = 0.05,
) -> dict:
    """After domain adaptation, measure perplexity on held-out general text.

    Streams domain_chunks first (to adapt), then evaluates on general_chunks.
    Checks that general perplexity stays within drift_threshold of baseline.

    Args:
        model: model with state (will be modified in-place)
        domain_chunks: list of [BS, T] domain token tensors
        general_chunks: list of [BS, T] general token tensors
        config: model config
        device: compute device
        baseline_ppl: Phase D baseline perplexity (if None, skip threshold check)
        drift_threshold: max acceptable relative increase (default 5%)

    Returns:
        dict with domain_result, general_ppls, drift_ratio, passed.
    """
    # Phase 1: domain adaptation
    domain_result = domain_adaptation(model, domain_chunks, config, device)

    # Phase 2: evaluate on general text (model state carries over)
    model.eval()
    general_ppls = []
    prev_token = None
    with torch.no_grad():
        for chunk in general_chunks:
            chunk = chunk.to(device)
            ppl, prev_token = _compute_chunk_ppl(
                model, chunk, config, device, prev_token
            )
            general_ppls.append(ppl)

    general_mean = sum(general_ppls) / len(general_ppls) if general_ppls else float("inf")

    drift_ratio = None
    passed = None
    if baseline_ppl is not None and baseline_ppl > 0:
        drift_ratio = (general_mean - baseline_ppl) / baseline_ppl
        passed = drift_ratio <= drift_threshold

    return {
        "domain_result": domain_result,
        "general_ppls": general_ppls,
        "general_mean_ppl": general_mean,
        "drift_ratio": drift_ratio,
        "passed": passed,
    }


def cross_document_recall(
    model: NeuromorphicLM,
    chunks: List[Tensor],
    probe_chunks: List[Tensor],
    config: ModelConfig,
    device: torch.device,
) -> dict:
    """Stream factual text, probe at increasing distances.

    Measures how long PM/EM retain useful information from earlier documents.

    Args:
        model: model with state (will be modified in-place)
        chunks: list of [BS, T] source chunks (streamed sequentially)
        probe_chunks: list of [BS, T] probe chunks corresponding to source content.
                      probe_chunks[i] tests recall of chunks[i] content.
        config: model config
        device: compute device

    Returns:
        dict with per-distance perplexities (lower = better recall).
    """
    model.eval()

    # First, stream all source chunks to build up memory
    prev_token = None
    with torch.no_grad():
        for chunk in chunks:
            chunk = chunk.to(device)
            _, prev_token = _compute_chunk_ppl(
                model, chunk, config, device, prev_token
            )

    # Save state after adaptation
    state_after = save_runtime_state(model)

    # Probe at increasing distances from the end
    # Each probe is a separate document — force h reset at t=0 via EOT prev_token
    eot_prev = torch.full((chunks[0].shape[0],), config.eot_id, dtype=torch.long, device=device)
    recall_ppls = []
    with torch.no_grad():
        for probe in probe_chunks:
            # Restore state (so each probe starts from same adapted state)
            load_runtime_state(model, state_after)
            probe = probe.to(device)
            ppl, _ = _compute_chunk_ppl(model, probe, config, device, prev_token=eot_prev)
            recall_ppls.append(ppl)

    return {
        "task": "cross_document_recall",
        "probe_ppls": recall_ppls,
        "mean_recall_ppl": sum(recall_ppls) / len(recall_ppls) if recall_ppls else float("inf"),
        "ppl_trend": _linear_trend(recall_ppls),
    }
