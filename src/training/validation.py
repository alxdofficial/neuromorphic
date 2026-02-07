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
    """Reset runtime state so validation starts from a clean memory state."""
    mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    if model.surprise is None or model.surprise.shape[0] != batch_size:
        model.surprise = torch.zeros(batch_size, 1, device=device)
    else:
        model.surprise = torch.zeros_like(model.surprise)

    model.reset_at_doc_boundary(mask)
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

            for t in range(T):
                if t == 0:
                    reset_mask = (prev_token == eot_id)
                else:
                    reset_mask = (input_ids[:, t - 1] == eot_id)

                logits, _, _ = model.forward_one_token(input_ids[:, t], reset_mask)
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

                total_tokens += BS
                eot_inputs += int(is_eot.sum().item())
                resets += int(reset_mask.sum().item())

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
