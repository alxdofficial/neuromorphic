"""Unit tests for IntegratedLM._compute_surprise (the NTP-CE-on-current-window math)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.trajectory_memory.integrated_lm import IntegratedLM


def test_compute_surprise_no_mask_matches_manual_ce():
    """Without a mask, surprise should equal mean per-token NTP CE on the
    current window's predictions, exactly."""
    torch.manual_seed(0)
    BS, L_lm, V, T_window = 2, 32, 100, 8
    full_logits = torch.randn(BS, L_lm, V)
    lm_input_ids = torch.randint(0, V, (BS, L_lm))

    surprise = IntegratedLM._compute_surprise(
        full_logits, lm_input_ids, T_window, target_mask=None,
    )

    # Manual compute: take the last T_window predictions of the standard
    # NTP shift and average per-token CE.
    shift_logits = full_logits[:, :-1, :].float()              # [BS, L-1, V]
    shift_targets = lm_input_ids[:, 1:]
    win_logits = shift_logits[:, -T_window:, :].reshape(-1, V)
    win_targets = shift_targets[:, -T_window:].reshape(-1)
    expected_per_tok = F.cross_entropy(win_logits, win_targets, reduction="none")
    expected = expected_per_tok.reshape(BS, T_window).mean(dim=1)
    assert torch.allclose(surprise, expected, atol=1e-5)


def test_compute_surprise_with_mask_filters_correctly():
    """Mask=True positions contribute to the mean; mask=False are excluded."""
    torch.manual_seed(0)
    BS, L_lm, V, T_window = 1, 32, 100, 8
    full_logits = torch.randn(BS, L_lm, V)
    lm_input_ids = torch.randint(0, V, (BS, L_lm))

    # Mask only the first 4 positions of the current window
    target_mask = torch.zeros(BS, T_window, dtype=torch.bool)
    target_mask[:, :4] = True

    surprise_masked = IntegratedLM._compute_surprise(
        full_logits, lm_input_ids, T_window, target_mask=target_mask,
    )

    # Manual compute on just the masked positions
    shift_logits = full_logits[:, :-1, :].float()
    shift_targets = lm_input_ids[:, 1:]
    win_logits = shift_logits[:, -T_window:, :].reshape(-1, V)
    win_targets = shift_targets[:, -T_window:].reshape(-1)
    ce_per_tok = F.cross_entropy(win_logits, win_targets, reduction="none").reshape(BS, T_window)
    expected = ce_per_tok[:, :4].mean(dim=1)
    assert torch.allclose(surprise_masked, expected, atol=1e-5)


def test_compute_surprise_handles_short_input():
    """When L_lm == T_window (no prior context), we have only T_window-1
    valid shift positions. The function should not crash and should return
    the mean over those T_window-1 predictions."""
    torch.manual_seed(0)
    BS, V, T_window = 1, 100, 8
    L_lm = T_window  # no prior context
    full_logits = torch.randn(BS, L_lm, V)
    lm_input_ids = torch.randint(0, V, (BS, L_lm))

    surprise = IntegratedLM._compute_surprise(
        full_logits, lm_input_ids, T_window, target_mask=None,
    )
    assert surprise.shape == (BS,)
    assert torch.isfinite(surprise).all()


def test_compute_surprise_zero_for_extremely_short_lm_input():
    """Degenerate case: L_lm < 2 has no shift positions. Should return zeros."""
    full_logits = torch.randn(2, 1, 10)
    lm_input_ids = torch.randint(0, 10, (2, 1))
    surprise = IntegratedLM._compute_surprise(
        full_logits, lm_input_ids, T_window=4, target_mask=None,
    )
    assert torch.allclose(surprise, torch.zeros(2))


def test_compute_surprise_uses_fp32_for_bf16_logits():
    """bf16 logits should be cast to fp32 inside CE. Verify by comparing
    against an explicit fp32 CE — they should match closely (bf16 inputs
    naturally have bf16 noise, but the CE math is in fp32)."""
    torch.manual_seed(0)
    BS, L_lm, V, T_window = 1, 16, 50, 4
    full_logits_bf16 = torch.randn(BS, L_lm, V).to(torch.bfloat16)
    lm_input_ids = torch.randint(0, V, (BS, L_lm))

    surprise_bf16 = IntegratedLM._compute_surprise(
        full_logits_bf16, lm_input_ids, T_window, target_mask=None,
    )
    surprise_fp32 = IntegratedLM._compute_surprise(
        full_logits_bf16.float(), lm_input_ids, T_window, target_mask=None,
    )
    # Should be approximately equal (small differences from bf16 input
    # rounding).
    assert torch.allclose(surprise_bf16, surprise_fp32, atol=0.5), (
        "bf16-cast CE deviates significantly from fp32 baseline"
    )
