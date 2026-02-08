"""
Validation utilities for neuromorphic LM.

Provides a lightweight held-out evaluation loop that computes masked
cross-entropy/perplexity without optimizer updates.
"""

from typing import Iterator, Optional

import torch

from ..data.streaming import StreamBatch
from ..model.config import ModelConfig
from ..model.model import NeuromorphicLM
from ..model.state import save_runtime_state, load_runtime_state
from .loss import online_cross_entropy


def _clear_runtime_for_eval(model: NeuromorphicLM, batch_size: int,
                            device: torch.device):
    """Hard-reset runtime state so validation starts from a clean slate."""
    mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    if model.surprise is None or model.surprise.shape[0] != batch_size:
        model.surprise = torch.zeros(batch_size, 1, device=device)
    else:
        model.surprise = torch.zeros_like(model.surprise)

    # Reset transient recurrent state.
    for block in model.blocks:
        for layer in block.layers:
            layer.reset_states(mask)         # h
            layer.pm.reset_states(mask)      # pm_K/pm_V/pm_a/elig
        # Keep em_K/em_V but zero strengths so retrieval is inactive.
        block.em.reset_states(mask)

    model.wm.reset_states(mask)
    model.detach_states()


@torch.no_grad()
def evaluate_validation(
    model: NeuromorphicLM,
    dataloader: Iterator[StreamBatch],
    config: ModelConfig,
    device: torch.device,
    num_steps: int = 20,
    pm_enabled: Optional[bool] = None,
    em_enabled: Optional[bool] = None,
) -> dict:
    """Evaluate masked CE/perplexity over validation chunks.

    Runtime state and config toggles are restored after evaluation.
    """
    total_loss = 0.0
    valid_count = 0
    total_tokens = 0
    eot_inputs = 0
    resets = 0
    steps_done = 0

    was_training = model.training
    state_before = save_runtime_state(model)
    pm_before = config.pm_enabled
    em_before = config.em_enabled

    if pm_enabled is not None:
        config.pm_enabled = pm_enabled
    if em_enabled is not None:
        config.em_enabled = em_enabled

    try:
        model.eval()
        cleared = False
        eot_id = config.eot_id
        P = config.P

        for _ in range(num_steps):
            try:
                batch = next(dataloader)
            except StopIteration:
                break

            input_ids = batch.input_ids.to(device)
            target_ids = batch.target_ids.to(device)
            prev_token = batch.prev_token.to(device)
            BS, T = input_ids.shape

            if not cleared:
                _clear_runtime_for_eval(model, BS, device)
                cleared = True

            for span_start in range(0, T, P):
                span_end = min(span_start + P, T)

                span_surprise_accum = torch.zeros(BS, device=device)
                span_valid_tokens = torch.zeros(BS, device=device)
                span_last_reset = torch.zeros(BS, dtype=torch.long, device=device)

                B = config.B
                cand_K = [[] for _ in range(B)]
                cand_V = [[] for _ in range(B)]
                cand_score = [[] for _ in range(B)]
                cand_token_valid = [[] for _ in range(B)]

                for t in range(span_start, span_end):
                    if t == 0:
                        reset_mask = (prev_token == eot_id)
                    else:
                        reset_mask = (input_ids[:, t - 1] == eot_id)

                    if reset_mask.any() and config.reset_on_doc_boundary:
                        local_t = t - span_start
                        span_last_reset[reset_mask] = local_t
                        span_surprise_accum[reset_mask] = 0
                        span_valid_tokens[reset_mask] = 0

                    logits, x_emb, y_wm = model.forward_one_token(
                        input_ids[:, t], reset_mask
                    )
                    if not torch.isfinite(logits).all():
                        raise RuntimeError("Non-finite logits detected during validation.")

                    is_eot = (input_ids[:, t] == eot_id)
                    if config.reset_on_doc_boundary:
                        loss_mask = ~is_eot
                    else:
                        loss_mask = torch.ones_like(is_eot)

                    token_loss, count = online_cross_entropy(
                        logits, target_ids[:, t], loss_mask
                    )
                    total_loss += float(token_loss.item())
                    valid_count += int(count)

                    model.update_surprise(logits, target_ids[:, t], mask=loss_mask)
                    if model.surprise is not None:
                        span_surprise_accum = (
                            span_surprise_accum
                            + model.surprise.squeeze(-1) * loss_mask.float()
                        )
                    span_valid_tokens = span_valid_tokens + loss_mask.float()

                    if config.em_enabled:
                        for b, block in enumerate(model.blocks):
                            h_final = block.layers[-1].h
                            if h_final is not None:
                                k_c, v_c, nov = block.em.propose_candidate(
                                    x_emb, y_wm, h_final, model.surprise
                                )
                                cand_K[b].append(k_c)
                                cand_V[b].append(v_c)
                                cand_score[b].append(nov)
                                cand_token_valid[b].append(loss_mask)

                    total_tokens += BS
                    eot_inputs += int(is_eot.sum().item())
                    resets += int(reset_mask.sum().item())

                span_surprise_mean = span_surprise_accum / span_valid_tokens.clamp(min=1)

                if config.pm_enabled:
                    for block in model.blocks:
                        for layer in block.layers:
                            layer.pm.base_decay()
                    model.commit_at_boundary(span_surprise=span_surprise_mean)

                if config.em_enabled:
                    for b, block in enumerate(model.blocks):
                        if len(cand_K[b]) == 0:
                            continue
                        stacked_K = torch.stack(cand_K[b], dim=1)
                        stacked_V = torch.stack(cand_V[b], dim=1)
                        stacked_score = torch.stack(cand_score[b], dim=1)
                        stacked_token_valid = torch.stack(cand_token_valid[b], dim=1)

                        S = stacked_score.shape[1]
                        pos = torch.arange(S, device=device).unsqueeze(0)
                        cand_valid = (
                            pos >= span_last_reset.unsqueeze(1)
                        ) & stacked_token_valid.bool()

                        em_usage = (
                            block.em.em_S.sum(dim=-1)
                            if block.em.em_S is not None
                            else torch.zeros_like(span_surprise_mean)
                        )
                        cand_valid_f = cand_valid.float()
                        cand_count = cand_valid_f.sum(dim=-1).clamp(min=1)
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

            model.detach_states()
            steps_done += 1
    finally:
        load_runtime_state(model, state_before)
        config.pm_enabled = pm_before
        config.em_enabled = em_before
        if was_training:
            model.train()
        else:
            model.eval()

    if valid_count > 0:
        avg_loss = total_loss / valid_count
        ppl = min(float(torch.exp(torch.tensor(avg_loss)).item()), 1e6)
    else:
        avg_loss = float("inf")
        ppl = float("inf")

    denom = max(total_tokens, 1)
    return {
        "loss": avg_loss,
        "ppl": ppl,
        "valid_tokens": valid_count,
        "steps_done": steps_done,
        "valid_fraction": valid_count / denom,
        "eot_input_fraction": eot_inputs / denom,
        "reset_fraction": resets / denom,
    }
