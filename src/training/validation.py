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
from . import span_ops


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

    Runtime state and config toggles (pm_enabled, em_enabled) are temporarily
    overridden if caller passes explicit values, then restored in the finally
    block. Model runtime state is also saved/restored so validation is
    side-effect free.
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
        # Match training dtype so torch.compile doesn't recompile on dtype change
        use_amp = device.type == "cuda"
        amp_ctx = torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
        )

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

                # Wrap entire span (forward + PM + EM) under autocast to match
                # training dtype and avoid torch.compile recompilation
                with amp_ctx:
                    logits_all, x_emb_all, y_wm_all = model.forward_span(
                        span_ids, reset_first
                    )

                    if not torch.isfinite(logits_all).all():
                        raise RuntimeError("Non-finite logits detected during validation.")

                    # Loss masking
                    is_eot_all, loss_mask_all = span_ops.compute_loss_mask(
                        span_ids, eot_id, config.reset_on_doc_boundary
                    )

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

                    # Compute reset masks for accumulators
                    reset_mask_all = span_ops.compute_reset_mask(
                        model, span_ids, reset_first, config.reset_on_doc_boundary
                    )

                    # Track span surprise for boundary decisions
                    span_surprise_accum = torch.zeros(BS, device=device)
                    span_valid_tokens = torch.zeros(BS, device=device)
                    span_last_reset = torch.zeros(BS, dtype=torch.long, device=device)

                    span_ops.accumulate_span_surprise(
                        token_surprise, loss_mask_all, reset_mask_all,
                        config.reset_on_doc_boundary,
                        span_surprise_accum, span_valid_tokens, span_last_reset,
                    )

                    total_tokens += BS * span_P
                    eot_inputs += int(is_eot_all.sum().item())
                    resets += int(reset_mask_all.sum().item())

                    span_surprise_mean = span_surprise_accum / span_valid_tokens.clamp(min=1)

                    # Update model surprise to span mean (matching trainer)
                    model.surprise = span_surprise_mean.unsqueeze(-1)  # [BS, 1]

                    # PM eligibility + commit
                    if config.pm_enabled:
                        span_ops.apply_pm_eligibility_batch(
                            model, x_emb_all, token_surprise,
                            reset_mask_all, config,
                        )
                        span_ops.apply_pm_boundary(model, span_surprise_mean)

                    # EM candidates + write
                    if config.em_enabled:
                        B = config.B
                        cand_K = [[] for _ in range(B)]
                        cand_V = [[] for _ in range(B)]
                        cand_score = [[] for _ in range(B)]
                        cand_token_valid = [[] for _ in range(B)]

                        span_ops.propose_em_candidates(
                            model, x_emb_all, y_wm_all, token_surprise,
                            loss_mask_all, cand_K, cand_V, cand_score,
                            cand_token_valid,
                        )
                        em_stacked = span_ops.stack_em_candidates(
                            cand_K, cand_V, cand_score, cand_token_valid,
                            span_last_reset, device,
                        )
                        span_ops.apply_em_boundary(
                            model, em_stacked, span_surprise_mean, config,
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
