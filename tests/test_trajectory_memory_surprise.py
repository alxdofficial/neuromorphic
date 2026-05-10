"""Unit tests for IntegratedLM._compute_surprise_window (the NTP-CE math).

Note (2026-05-09): renamed + reshaped from `_compute_surprise(full_logits,
lm_input_ids, T_window, target_mask)` to `_compute_surprise_window(
needed_logits[BS, T_window+1, V], target_ids[BS, T_window], target_mask)`.
The new contract pre-slices logits to just the T_window+1 positions we
actually need for the NTP shift; the old version cast a [BS, L_lm, V]
tensor to fp32 wholesale (~2 GB at L_lm=2048, V=128K). Tests cover the
same correctness properties.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.trajectory_memory.integrated_lm import IntegratedLM


def test_compute_surprise_window_no_mask_matches_manual_ce():
    """Without a mask, surprise should equal mean per-token NTP CE on the
    T_window targets predicted by needed_logits[:, :-1, :]."""
    torch.manual_seed(0)
    BS, T_window, V = 2, 8, 100
    needed_logits = torch.randn(BS, T_window + 1, V)
    target_ids = torch.randint(0, V, (BS, T_window))

    surprise, _ce_sum, _ce_count = IntegratedLM._compute_surprise_window(
        needed_logits, target_ids, target_mask=None,
    )

    shift_logits = needed_logits[:, :-1, :].float()
    expected_per_tok = F.cross_entropy(
        shift_logits.reshape(-1, V),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape(BS, T_window)
    expected = expected_per_tok.mean(dim=1)
    assert torch.allclose(surprise, expected, atol=1e-5)


def test_compute_surprise_window_with_mask_filters_correctly():
    """Mask=True positions contribute to the mean; mask=False are excluded."""
    torch.manual_seed(0)
    BS, T_window, V = 1, 8, 100
    needed_logits = torch.randn(BS, T_window + 1, V)
    target_ids = torch.randint(0, V, (BS, T_window))

    target_mask = torch.zeros(BS, T_window, dtype=torch.bool)
    target_mask[:, :4] = True

    surprise, _ce_sum, _ce_count = IntegratedLM._compute_surprise_window(
        needed_logits, target_ids, target_mask=target_mask,
    )

    shift_logits = needed_logits[:, :-1, :].float()
    ce_per_tok = F.cross_entropy(
        shift_logits.reshape(-1, V),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape(BS, T_window)
    expected = ce_per_tok[:, :4].mean(dim=1)
    assert torch.allclose(surprise, expected, atol=1e-5)


def test_compute_surprise_window_zero_mask_does_not_nan():
    """All-False mask should return 0 (mask.sum().clamp_min(1) prevents
    division by zero)."""
    torch.manual_seed(0)
    BS, T_window, V = 2, 4, 50
    needed_logits = torch.randn(BS, T_window + 1, V)
    target_ids = torch.randint(0, V, (BS, T_window))
    target_mask = torch.zeros(BS, T_window, dtype=torch.bool)
    surprise, _ce_sum, _ce_count = IntegratedLM._compute_surprise_window(
        needed_logits, target_ids, target_mask=target_mask,
    )
    assert torch.isfinite(surprise).all()
    assert torch.allclose(surprise, torch.zeros(BS), atol=1e-5)


def test_compute_surprise_window_uses_fp32_for_bf16_logits():
    """bf16 logits get cast to fp32 inside CE. Compare to explicit fp32
    baseline — should match closely (bf16 inputs carry bf16 noise, but
    the CE math is fp32)."""
    torch.manual_seed(0)
    BS, T_window, V = 1, 4, 50
    needed_bf16 = torch.randn(BS, T_window + 1, V).to(torch.bfloat16)
    target_ids = torch.randint(0, V, (BS, T_window))

    surprise_bf16, _, _ = IntegratedLM._compute_surprise_window(
        needed_bf16, target_ids, target_mask=None,
    )
    surprise_fp32, _, _ = IntegratedLM._compute_surprise_window(
        needed_bf16.float(), target_ids, target_mask=None,
    )
    assert torch.allclose(surprise_bf16, surprise_fp32, atol=0.5)


def test_compute_surprise_window_short_context_path():
    """At the very first window (no rolling context), needed_logits has
    T_window positions instead of T_window+1. The function should handle
    this by predicting only targets 1..T_window-1 (target 0 has no
    predecessor). Result shape is still [BS]."""
    torch.manual_seed(0)
    BS, T_window, V = 2, 4, 50
    # T_window positions (no +1)
    needed_logits = torch.randn(BS, T_window, V)
    target_ids = torch.randint(0, V, (BS, T_window))
    surprise, _ce_sum, _ce_count = IntegratedLM._compute_surprise_window(
        needed_logits, target_ids, target_mask=None,
    )
    assert surprise.shape == (BS,)
    assert torch.isfinite(surprise).all()
    # Manual: shift_logits = needed_logits[:, :-1] predicts target_ids[:, 1:]
    shift_logits = needed_logits[:, :-1, :].float()
    expected_per_tok = F.cross_entropy(
        shift_logits.reshape(-1, V),
        target_ids[:, 1:].reshape(-1),
        reduction="none",
    ).reshape(BS, T_window - 1)
    expected = expected_per_tok.mean(dim=1)
    assert torch.allclose(surprise, expected, atol=1e-5)


def test_compute_surprise_window_invalid_shape_raises():
    """Wrong leading dim (not T_window or T_window+1) should raise."""
    BS, T_window, V = 1, 4, 50
    bad_logits = torch.randn(BS, T_window + 5, V)
    target_ids = torch.randint(0, V, (BS, T_window))
    try:
        IntegratedLM._compute_surprise_window(
            bad_logits, target_ids, target_mask=None,
        )
    except AssertionError:
        return
    raise AssertionError("expected shape mismatch to raise AssertionError")
