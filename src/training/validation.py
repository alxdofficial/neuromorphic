"""
Validation utilities for neuromorphic LM.

Provides a lightweight held-out evaluation loop that computes masked
cross-entropy/perplexity without optimizer updates.
"""

from typing import Iterator, Optional

import torch
import torch.nn.functional as F

from ..data.streaming import StreamBatch
from ..model.config import ModelConfig
from ..model.model import NeuromorphicLM
from ..model.state import save_runtime_state, load_runtime_state
from .loss import batched_cross_entropy


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
    """Validate masked CE/perplexity over held-out chunks.

    Runtime state and config toggles are restored after validation.
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
                span_P = span_end - span_start

                span_ids = input_ids[:, span_start:span_end]
                span_targets = target_ids[:, span_start:span_end]

                # Reset mask for first token
                if span_start == 0:
                    reset_first = (prev_token == eot_id)
                else:
                    reset_first = (input_ids[:, span_start - 1] == eot_id)

                # Parallel forward
                logits_all, x_emb_all, y_wm_all = model.forward_span(
                    span_ids, reset_first
                )

                if not torch.isfinite(logits_all).all():
                    raise RuntimeError("Non-finite logits detected during validation.")

                # Loss masking
                is_eot_all = (span_ids == eot_id)
                if config.reset_on_doc_boundary:
                    loss_mask_all = ~is_eot_all
                else:
                    loss_mask_all = torch.ones_like(is_eot_all)

                # Batched loss
                span_loss, span_valid = batched_cross_entropy(
                    logits_all, span_targets, loss_mask_all
                )
                total_loss += float(span_loss.item())
                valid_count += span_valid

                # Compute per-token surprise
                logp = F.log_softmax(logits_all, dim=-1)
                token_surprise = -logp.gather(-1, span_targets.unsqueeze(-1))
                token_surprise = token_surprise * loss_mask_all.unsqueeze(-1).float()
                # Note: model.surprise updated below to span mean (matching trainer)

                # Compute reset masks for accumulators
                reset_mask_all = model._compute_reset_masks(span_ids, reset_first)
                if not config.reset_on_doc_boundary:
                    reset_mask_all = torch.zeros_like(reset_mask_all)

                # Track span surprise for boundary decisions
                span_surprise_accum = torch.zeros(BS, device=device)
                span_valid_tokens = torch.zeros(BS, device=device)
                span_last_reset = torch.zeros(BS, dtype=torch.long, device=device)

                for t_local in range(span_P):
                    reset_t = reset_mask_all[:, t_local]
                    if reset_t.any() and config.reset_on_doc_boundary:
                        span_last_reset[reset_t] = t_local
                        span_surprise_accum[reset_t] = 0
                        span_valid_tokens[reset_t] = 0
                    lm = loss_mask_all[:, t_local]
                    span_surprise_accum = (
                        span_surprise_accum
                        + token_surprise[:, t_local, 0] * lm.float()
                    )
                    span_valid_tokens = span_valid_tokens + lm.float()

                total_tokens += BS * span_P
                eot_inputs += int(is_eot_all.sum().item())
                resets += int(reset_mask_all.sum().item())

                span_surprise_mean = span_surprise_accum / span_valid_tokens.clamp(min=1)

                # Update model surprise to span mean (matching trainer)
                model.surprise = span_surprise_mean.unsqueeze(-1)  # [BS, 1]

                # PM eligibility + commit
                if config.pm_enabled:
                    # Batched eligibility (matching trainer)
                    x_proj_all = model.W_in(x_emb_all)  # [BS, span_P, D]
                    x_blocks_all = x_proj_all.view(
                        BS, span_P, config.B, config.D_h
                    )  # [BS, span_P, B, D_h]

                    for b, block in enumerate(model.blocks):
                        for layer in block.layers:
                            if not hasattr(layer, '_last_h_all') or layer._last_h_all is None:
                                continue

                            l_idx = layer.layer_idx
                            if l_idx == 0:
                                x_in = x_blocks_all[:, :, b]
                            else:
                                x_in = block.layers[l_idx - 1]._last_h_all

                            h_out = layer._last_h_all

                            layer.pm.update_eligibility_batch(
                                x_in, h_out, token_surprise, reset_mask_all,
                            )
                    # Decay + commit
                    for block in model.blocks:
                        for layer in block.layers:
                            layer.pm.base_decay()
                    model.commit_at_boundary(span_surprise=span_surprise_mean)

                # EM candidates + write
                if config.em_enabled:
                    for b, block in enumerate(model.blocks):
                        h_final_all = getattr(block.layers[-1], '_last_h_all', None)
                        if h_final_all is None:
                            continue
                        k_c, v_c, nov = block.em.propose_candidate_batch(
                            x_emb_all, y_wm_all, h_final_all, token_surprise,
                        )
                        S = nov.shape[1]
                        pos = torch.arange(S, device=device).unsqueeze(0)
                        cand_valid = (
                            pos >= span_last_reset.unsqueeze(1)
                        ) & loss_mask_all.bool()

                        em_usage = (
                            block.em.em_S.sum(dim=-1)
                            if block.em.em_S is not None
                            else torch.zeros_like(span_surprise_mean)
                        )
                        cand_valid_f = cand_valid.float()
                        cand_count = cand_valid_f.sum(dim=-1).clamp(min=1)
                        cand_novelty_mean = (nov * cand_valid_f).sum(dim=-1) / cand_count
                        write_mask, g_em, _p_write = block.em_neuromodulator.forward(
                            span_surprise_mean,
                            em_usage / config.budget_em,
                            cand_novelty_mean,
                        )
                        block.em.write_at_boundary(
                            k_c, v_c, nov, write_mask, g_em,
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
