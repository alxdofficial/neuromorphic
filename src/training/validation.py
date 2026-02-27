"""
Validation utilities for neuromorphic LM (v4).

Lightweight held-out evaluation loop: masked cross-entropy/perplexity
without optimizer updates.
"""

from typing import Iterator, Optional

import torch
import torch.nn.functional as F

from ..data.streaming import StreamBatch
from ..model.config import ModelConfig
from ..model.model import NeuromorphicLM
from ..model.state import save_runtime_state, load_runtime_state
from ..model.utils import runtime_state_dtype
from .loss import batched_cross_entropy, fitb_cross_entropy
from .masking import generate_fitb_mask, apply_special_token_protection


def _clear_runtime_for_validation(model: NeuromorphicLM, batch_size: int,
                                  device: torch.device):
    """Hard-reset runtime state so validation starts from a clean slate."""
    # PM/EM state has explicit B dim: [BS, B, ...]
    mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    model.pm.reset_content(mask)
    model.em.reset_states(mask)
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
    """Validate masked CE/perplexity over held-out chunks."""
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
        model.set_eval_mode()
        cleared = False
        eot_id = config.eot_id
        N = config.N

        use_amp = device.type == "cuda"
        amp_ctx = torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
        )

        is_fitb = config.mask_rate > 0 and config.fitb_id >= 0

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
                _clear_runtime_for_validation(model, BS, device)
                cleared = True

            for seg_start in range(0, T, N):
                seg_end = min(seg_start + N, T)
                seg_ids = input_ids[:, seg_start:seg_end]
                seg_targets = target_ids[:, seg_start:seg_end]

                if seg_start == 0:
                    reset_mask = (prev_token == eot_id)
                else:
                    reset_mask = (input_ids[:, seg_start - 1] == eot_id)

                is_eot = (seg_ids == eot_id)
                eot_inputs += int(is_eot.sum().item())
                resets += int(reset_mask.sum().item())

                if is_fitb:
                    seg_N = seg_ids.shape[1]
                    fitb_mask = generate_fitb_mask(
                        BS, seg_N, config.mask_rate,
                        config.span_mask_prob,
                        config.span_mask_mean_len,
                        device,
                    )
                    fitb_mask = apply_special_token_protection(
                        fitb_mask, seg_ids, eot_id, config.null_id,
                    )
                    seg_ids_masked = seg_ids.clone()
                    seg_ids_masked[fitb_mask] = config.fitb_id

                    with amp_ctx:
                        per_pass_logits, _ = model.forward_segment(
                            seg_ids_masked, reset_mask, fitb_mask=fitb_mask
                        )

                    # Use final-pass logits for perplexity
                    final_logits = per_pass_logits[-1]
                    if not torch.isfinite(final_logits).all():
                        raise RuntimeError("Non-finite logits during FITB validation.")

                    # CE only at masked positions (final pass only for val metric)
                    flat_mask = fitb_mask.reshape(-1)
                    flat_targets = seg_ids.reshape(-1).clone()
                    flat_targets[~flat_mask] = -100
                    seg_loss = F.cross_entropy(
                        final_logits.reshape(-1, final_logits.shape[-1]),
                        flat_targets, ignore_index=-100, reduction="sum",
                    )
                    seg_valid = flat_mask.sum()
                else:
                    with amp_ctx:
                        logits, _ = model.forward_segment(seg_ids, reset_mask)

                    if not torch.isfinite(logits).all():
                        raise RuntimeError("Non-finite logits during validation.")

                    if config.reset_on_doc_boundary:
                        loss_mask = ~is_eot
                    else:
                        loss_mask = torch.ones_like(is_eot)

                    seg_loss, seg_valid = batched_cross_entropy(
                        logits, seg_targets, loss_mask
                    )

                total_loss += float(seg_loss.item())
                valid_count += int(seg_valid.item()) if torch.is_tensor(seg_valid) else int(seg_valid)
                total_tokens += BS * (seg_end - seg_start)

            model.detach_states()
            steps_done += 1
    finally:
        load_runtime_state(model, state_before)
        config.pm_enabled = pm_before
        config.em_enabled = em_before
        if was_training:
            model.train()

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
