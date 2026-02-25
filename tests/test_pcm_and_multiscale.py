"""Tests for Predictive Coding Module (PCM)."""

import copy
import math
import torch
import pytest

from src.model.config import ModelConfig
from src.model.model import NeuromorphicLM
from src.model.predictive_coding import PredictiveCodingModule

BS = 2
D = 128
B = 2
L = 2
D_h = D // B  # 64
D_pc = 32
P = 16


# ============================================================================
# Predictive Coding Module tests
# ============================================================================

def _make_pcm_config(**overrides):
    defaults = dict(D=D, B=B, L=L, D_pc=D_pc, pcm_enabled=True,
                    D_em=64, dropout=0.0, pm_enabled=True, em_enabled=True)
    defaults.update(overrides)
    return ModelConfig(**defaults)


class TestPCM:
    def test_encode_shape(self):
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        x = torch.randn(BS, P, D_h)
        z = pcm.encode(x)
        assert z.shape == (BS, P, D_pc)

    def test_surprise_zero_before_valid_hypothesis(self):
        """Before any boundary update, z_hat_valid=False -> surprise is zero."""
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        pcm._lazy_init(BS, torch.device("cpu"))
        z = torch.randn(BS, P, D_pc)
        delta = pcm.compute_surprise(z)
        assert torch.equal(delta, torch.zeros_like(z))

    def test_surprise_zero_when_z_hat_is_none(self):
        """Before lazy init, z_hat=None -> surprise is zero."""
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        z = torch.randn(BS, P, D_pc)
        delta = pcm.compute_surprise(z)
        assert torch.equal(delta, torch.zeros_like(z))

    def test_surprise_nonzero_after_boundary(self):
        """After boundary update, z_hat is valid -> surprise is nonzero."""
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        pcm._lazy_init(BS, torch.device("cpu"))

        z_mean = torch.randn(BS, D_pc)
        ctx_b = torch.randn(BS, D_h)
        pm_s = torch.randn(BS, D_h)
        em_s = torch.randn(BS, config.D_em)
        pcm.boundary_update(z_mean, ctx_b, pm_s, em_s)

        # Now z_hat is valid; surprise should be nonzero
        z = torch.randn(BS, P, D_pc)
        delta = pcm.compute_surprise(z)
        assert delta.abs().max() > 0

    def test_z_hat_valid_set_after_boundary(self):
        """boundary_update marks z_hat as valid."""
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        pcm._lazy_init(BS, torch.device("cpu"))

        assert not pcm.z_hat_valid.any()  # initially invalid

        z_mean = torch.randn(BS, D_pc)
        ctx_b = torch.randn(BS, D_h)
        pm_s = torch.randn(BS, D_h)
        em_s = torch.randn(BS, config.D_em)
        pcm.boundary_update(z_mean, ctx_b, pm_s, em_s)

        assert pcm.z_hat_valid.all()  # now valid

    def test_reset_clears_z_hat_valid(self):
        """reset_states clears z_hat_valid for the reset stream."""
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        pcm._lazy_init(BS, torch.device("cpu"))

        z_mean = torch.randn(BS, D_pc)
        ctx_b = torch.randn(BS, D_h)
        pm_s = torch.randn(BS, D_h)
        em_s = torch.randn(BS, config.D_em)
        pcm.boundary_update(z_mean, ctx_b, pm_s, em_s)

        mask = torch.tensor([True, False])
        pcm.reset_states(mask)

        assert not pcm.z_hat_valid[0]  # reset
        assert pcm.z_hat_valid[1]       # preserved

    def test_l_pred_only_counts_valid_streams(self):
        """L_pred should be zero on the first boundary (no valid hypothesis yet)."""
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        pcm._lazy_init(BS, torch.device("cpu"))

        z_mean = torch.randn(BS, D_pc)
        ctx_b = torch.randn(BS, D_h)
        pm_s = torch.randn(BS, D_h)
        em_s = torch.randn(BS, config.D_em)

        # First boundary: z_hat was zeros, z_hat_valid=False → L_pred=0
        L_pred = pcm.boundary_update(z_mean, ctx_b, pm_s, em_s)
        assert L_pred.item() == 0.0

        # Second boundary: z_hat is now valid → L_pred > 0 (with high probability)
        z_mean2 = torch.randn(BS, D_pc)
        L_pred2 = pcm.boundary_update(z_mean2, ctx_b, pm_s, em_s)
        assert L_pred2.item() > 0

    def test_ffn_gain_shape(self):
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        delta = torch.randn(BS, P, D_pc)
        gain = pcm.compute_ffn_gain(delta)
        assert gain.shape == (BS, P, D_h)

    def test_ffn_gain_bounded(self):
        """FFN gain is bounded to [0.9, 1.1] by tanh."""
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        # Large delta to push tanh toward saturation
        delta = torch.randn(BS, P, D_pc) * 100
        gain = pcm.compute_ffn_gain(delta)
        assert gain.min() >= 0.9 - 1e-6
        assert gain.max() <= 1.1 + 1e-6

    def test_ffn_gain_starts_at_one(self):
        """At init, W_gain is zero -> gain is exactly 1."""
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        delta = torch.randn(BS, P, D_pc)
        gain = pcm.compute_ffn_gain(delta)
        assert torch.allclose(gain, torch.ones_like(gain))

    def test_recon_loss_scalar(self):
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        z = torch.randn(BS, P, D_pc)
        x_block = torch.randn(BS, P, D_h)
        loss = pcm.compute_recon_loss(z, x_block)
        assert loss.dim() == 0  # scalar

    def test_boundary_update_returns_loss(self):
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        pcm._lazy_init(BS, torch.device("cpu"))

        z_mean = torch.randn(BS, D_pc)
        ctx_b = torch.randn(BS, D_h)
        pm_s = torch.randn(BS, D_h)
        em_s = torch.randn(BS, config.D_em)

        L_pred = pcm.boundary_update(z_mean, ctx_b, pm_s, em_s)
        assert L_pred.dim() == 0  # scalar
        assert L_pred.item() >= 0  # MSE is non-negative

    def test_boundary_update_z_hat_has_grad(self):
        """z_hat should retain computation graph for predictor backprop."""
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        pcm._lazy_init(BS, torch.device("cpu"))

        z_mean = torch.randn(BS, D_pc)
        ctx_b = torch.randn(BS, D_h)
        pm_s = torch.randn(BS, D_h)
        em_s = torch.randn(BS, config.D_em)
        pcm.boundary_update(z_mean, ctx_b, pm_s, em_s)

        # z_hat should have a grad_fn (from the predictor forward pass)
        assert pcm.z_hat.grad_fn is not None

    def test_detach_states_preserves_z_hat_graph(self):
        """detach_states is intentionally a no-op for PCM."""
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        pcm._lazy_init(BS, torch.device("cpu"))

        z_mean = torch.randn(BS, D_pc)
        ctx_b = torch.randn(BS, D_h)
        pm_s = torch.randn(BS, D_h)
        em_s = torch.randn(BS, config.D_em)
        pcm.boundary_update(z_mean, ctx_b, pm_s, em_s)

        pcm.detach_states()
        # z_hat should STILL have a grad_fn
        assert pcm.z_hat.grad_fn is not None

    def test_reset_states_zeros_z_hat(self):
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        pcm._lazy_init(BS, torch.device("cpu"))

        z_mean = torch.randn(BS, D_pc)
        ctx_b = torch.randn(BS, D_h)
        pm_s = torch.randn(BS, D_h)
        em_s = torch.randn(BS, config.D_em)
        pcm.boundary_update(z_mean, ctx_b, pm_s, em_s)

        mask = torch.tensor([True, False])  # reset stream 0
        pcm.reset_states(mask)
        assert pcm.z_hat[0].abs().max() < 1e-8
        assert pcm.z_hat[1].abs().max() > 0  # stream 1 preserved


# ============================================================================
# Integration: PCM in full model
# ============================================================================

def _make_pcm_model(pcm_enabled=True, **extra):
    config = ModelConfig(
        D=D, B=B, L=L, D_pc=D_pc,
        pcm_enabled=pcm_enabled,
        D_em=64, D_wm=64, n_heads_wm=2,
        P=P, T=P, dropout=0.0, vocab_size=256,
        wm_enabled=True, pm_enabled=True, em_enabled=True,
        snapshot_enabled=False,
        d_dec=64, n_heads_decoder=2,
        **extra,
    )
    return NeuromorphicLM(config)


class TestPCMIntegration:
    def test_forward_span_with_pcm(self):
        """Model forward_span works with pcm_enabled."""
        model = _make_pcm_model(pcm_enabled=True)
        model.train()

        ids = torch.randint(0, 256, (BS, P))
        reset = torch.zeros(BS, dtype=torch.bool)
        model.initialize_states(BS, torch.device("cpu"))

        logits, x_emb, y_wm = model.forward_span(ids, reset)
        assert logits.shape == (BS, P, 256)

    def test_forward_span_without_pcm(self):
        """Model forward_span works with pcm_enabled=False (backward compat)."""
        model = _make_pcm_model(pcm_enabled=False)
        model.train()

        ids = torch.randint(0, 256, (BS, P))
        reset = torch.zeros(BS, dtype=torch.bool)
        model.initialize_states(BS, torch.device("cpu"))

        logits, x_emb, y_wm = model.forward_span(ids, reset)
        assert logits.shape == (BS, P, 256)

    def test_pcm_boundary_returns_losses(self):
        """apply_pcm_boundary returns scalar losses."""
        model = _make_pcm_model(pcm_enabled=True)
        model.train()

        ids = torch.randint(0, 256, (BS, P))
        reset = torch.zeros(BS, dtype=torch.bool)
        model.initialize_states(BS, torch.device("cpu"))
        model.forward_span(ids, reset)

        L_pred, L_recon = model.apply_pcm_boundary()
        assert L_pred.dim() == 0
        assert L_recon.dim() == 0

    def test_pcm_boundary_noop_when_disabled(self):
        """apply_pcm_boundary returns zeros when pcm_enabled=False."""
        model = _make_pcm_model(pcm_enabled=False)
        ids = torch.randint(0, 256, (BS, P))
        reset = torch.zeros(BS, dtype=torch.bool)
        model.initialize_states(BS, torch.device("cpu"))
        model.forward_span(ids, reset)

        L_pred, L_recon = model.apply_pcm_boundary()
        assert L_pred.item() == 0.0
        assert L_recon.item() == 0.0

    def test_pcm_gradient_flow(self):
        """PCM aux losses have gradients to PCM parameters."""
        model = _make_pcm_model(pcm_enabled=True)
        model.train()

        ids = torch.randint(0, 256, (BS, P))
        reset = torch.zeros(BS, dtype=torch.bool)
        model.initialize_states(BS, torch.device("cpu"))
        model.forward_span(ids, reset)

        L_pred, L_recon = model.apply_pcm_boundary()
        total = L_pred + L_recon
        total.backward()

        # PCM encoder should have gradients (from L_recon)
        pcm = model.blocks[0].pcm
        assert pcm.encoder.weight.grad is not None
        assert pcm.encoder.weight.grad.abs().max() > 0

    def test_pcm_surprise_rms_normalized(self):
        """Block caches RMS-normalized PCM surprise (||δ||/sqrt(D_pc))."""
        model = _make_pcm_model(pcm_enabled=True)
        model.train()

        ids = torch.randint(0, 256, (BS, P))
        reset = torch.zeros(BS, dtype=torch.bool)
        model.initialize_states(BS, torch.device("cpu"))

        # First span: no valid hypothesis → surprise is zero
        model.forward_span(ids, reset)
        # After first boundary, hypothesis becomes valid
        model.apply_pcm_boundary()

        # Second span: now surprise should be nonzero and RMS-normalized
        ids2 = torch.randint(0, 256, (BS, P))
        model.forward_span(ids2, reset)
        for block in model.blocks:
            if block._last_token_surprise is not None:
                surprise = block._last_token_surprise
                assert surprise.shape == (BS, P, 1)
                # RMS-normalized surprise should be reasonable (~1 scale)
                # Not the raw L2 norm (~sqrt(D_pc) ≈ 5.7)
                mean_surprise = surprise.mean().item()
                assert mean_surprise < 10.0, \
                    f"RMS-normalized surprise should be moderate, got {mean_surprise}"


class TestPCMDetachment:
    def test_z_hat_detached_in_surprise(self):
        """δ = z - z_hat.detach(): LM loss should NOT reach predictor through δ."""
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        pcm._lazy_init(BS, torch.device("cpu"))

        # Set z_hat via boundary update
        z_mean = torch.randn(BS, D_pc)
        ctx_b = torch.randn(BS, D_h)
        pm_s = torch.randn(BS, D_h)
        em_s = torch.randn(BS, config.D_em)
        pcm.boundary_update(z_mean, ctx_b, pm_s, em_s)

        # Compute surprise
        z = torch.randn(BS, P, D_pc, requires_grad=True)
        delta = pcm.compute_surprise(z)

        # Simulate LM loss flowing back through δ
        fake_lm_loss = delta.sum()
        fake_lm_loss.backward()

        # Predictor weights should have NO gradients from this path
        for param in pcm.predictor.parameters():
            assert param.grad is None or param.grad.abs().max() == 0, \
                "LM loss should not backprop into predictor through δ"

    def test_z_hat_graph_preserved_for_l_pred(self):
        """z_hat's own graph (for L_pred) is preserved despite detach in δ."""
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        pcm._lazy_init(BS, torch.device("cpu"))

        # First boundary: sets z_hat with a computation graph
        z_mean = torch.randn(BS, D_pc)
        ctx_b = torch.randn(BS, D_h)
        pm_s = torch.randn(BS, D_h)
        em_s = torch.randn(BS, config.D_em)
        pcm.boundary_update(z_mean, ctx_b, pm_s, em_s)

        # Second boundary: L_pred should backprop through z_hat into predictor
        z_mean2 = torch.randn(BS, D_pc)
        L_pred = pcm.boundary_update(z_mean2, ctx_b, pm_s, em_s)
        L_pred.backward()

        # Predictor weights should have gradients from L_pred
        has_grad = any(p.grad is not None and p.grad.abs().max() > 0
                       for p in pcm.predictor.parameters())
        assert has_grad, "L_pred should backprop into predictor"


class TestGateZeroInit:
    def test_delta_columns_zero_at_init(self):
        """Gate δ columns start at zero to prevent distribution shift."""
        config = _make_pcm_config()
        model = _make_pcm_model(pcm_enabled=True)

        for block in model.blocks:
            for layer in block.layers:
                # Last D_pc columns of gate_ab weight should be zero
                delta_cols = layer.gate_ab.weight[:, -D_pc:]
                assert delta_cols.abs().max() == 0, \
                    "Gate δ columns should be zero-initialized"

    def test_non_delta_columns_nonzero(self):
        """Non-δ gate columns should have normal (non-zero) init."""
        config = _make_pcm_config()
        model = _make_pcm_model(pcm_enabled=True)

        for block in model.blocks:
            for layer in block.layers:
                # First 4*D_h columns should be nonzero (normal Kaiming init)
                non_delta_cols = layer.gate_ab.weight[:, :4 * D_h]
                assert non_delta_cols.abs().max() > 0, \
                    "Non-δ gate columns should have normal init"
