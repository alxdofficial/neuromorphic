"""Tests for Predictive Coding Module (PCM) and multi-timescale blocks."""

import copy
import torch
import pytest

from src.model.config import ModelConfig
from src.model.model import NeuromorphicLM
from src.model.predictive_coding import PredictiveCodingModule
from src.model.temporal_pool import TemporalPooler, carry_min_pool

BS = 2
D = 128
B = 2
L = 2
D_h = D // B  # 64
D_pc = 32
P = 16


# ============================================================================
# Temporal pooling tests
# ============================================================================

class TestTemporalPooler:
    def test_scale_1_identity(self):
        """scale=1 is a no-op."""
        pooler = TemporalPooler(D_h, scale=1)
        x = torch.randn(BS, P, D_h)
        assert torch.equal(pooler.downsample(x), x)
        assert torch.equal(pooler.upsample(x, P), x)

    def test_downsample_shape(self):
        """Downsampling reduces sequence length by scale."""
        for scale in [2, 4, 8]:
            pooler = TemporalPooler(D_h, scale=scale)
            x = torch.randn(BS, P, D_h)
            out = pooler.downsample(x)
            assert out.shape == (BS, P // scale, D_h), f"scale={scale}"

    def test_upsample_shape(self):
        """Upsampling restores original length."""
        for scale in [2, 4]:
            pooler = TemporalPooler(D_h, scale=scale)
            x = torch.randn(BS, P, D_h)
            down = pooler.downsample(x)
            up = pooler.upsample(down, P)
            assert up.shape == (BS, P, D_h), f"scale={scale}"

    def test_init_as_average(self):
        """At init, pool_logits=0 → softmax → uniform → average pooling."""
        scale = 4
        pooler = TemporalPooler(D_h, scale=scale)
        x = torch.randn(BS, P, D_h)
        out = pooler.downsample(x)
        # Should match average within each window
        expected = x.reshape(BS, P // scale, scale, D_h).mean(dim=2)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_learnable_weights(self):
        """pool_logits are nn.Parameter and affect output."""
        scale = 4
        pooler = TemporalPooler(D_h, scale=scale)
        x = torch.randn(BS, P, D_h)

        out_before = pooler.downsample(x).clone()
        # Shift weights to favor the last position
        with torch.no_grad():
            pooler.pool_logits.copy_(torch.tensor([0.0, 0.0, 0.0, 10.0]))
        out_after = pooler.downsample(x)
        assert not torch.allclose(out_before, out_after)
        # With weight heavily on position 3, output ≈ x at positions 3,7,11,...
        x_win = x.reshape(BS, P // scale, scale, D_h)
        assert torch.allclose(out_after, x_win[:, :, 3], atol=1e-3)

    def test_boundary_masking(self):
        """Tokens before a boundary within a window are masked out."""
        scale = 4
        pooler = TemporalPooler(D_h, scale=scale)
        x = torch.randn(BS, P, D_h)
        carry = torch.ones(BS, P, 1)
        # Boundary at position 2 (within window 0: positions 0,1,2,3)
        carry[:, 2, :] = 0

        out = pooler.downsample(x, carry)
        # Window 0: boundary at pos 2 → only pos 2,3 contribute
        # With uniform weights, out[0] ≈ mean(x[2], x[3])
        x_win = x.reshape(BS, P // scale, scale, D_h)
        expected_win0 = x_win[:, 0, 2:].mean(dim=1)  # mean of pos 2,3
        assert torch.allclose(out[:, 0], expected_win0, atol=1e-5)

        # Window 1 (positions 4-7): no boundary → normal average
        expected_win1 = x_win[:, 1].mean(dim=1)
        assert torch.allclose(out[:, 1], expected_win1, atol=1e-5)

    def test_no_carry_matches_plain(self):
        """Without carry, downsample gives same result as carry=all_ones."""
        scale = 4
        pooler = TemporalPooler(D_h, scale=scale)
        x = torch.randn(BS, P, D_h)
        carry = torch.ones(BS, P, 1)
        assert torch.allclose(
            pooler.downsample(x), pooler.downsample(x, carry), atol=1e-6
        )

    def test_upsample_repeats(self):
        """Upsample uses repeat_interleave to restore length."""
        scale = 4
        pooler = TemporalPooler(D_h, scale=scale)
        x = torch.randn(BS, P, D_h)
        down = pooler.downsample(x)
        up = pooler.upsample(down, P)
        # Each pooled position is repeated scale times
        for i in range(P // scale):
            for j in range(scale):
                assert torch.equal(up[:, i * scale + j], down[:, i])


class TestCarryMinPool:
    def test_all_ones(self):
        """All carry=1 stays 1."""
        carry = torch.ones(BS, P, 1)
        out = carry_min_pool(carry, 4)
        assert out.shape == (BS, P // 4, 1)
        assert torch.allclose(out, torch.ones_like(out))

    def test_boundary_forces_reset(self):
        """Any 0 in a window produces 0 in output."""
        carry = torch.ones(BS, P, 1)
        carry[:, 2, :] = 0  # boundary at position 2
        out = carry_min_pool(carry, 4)
        # With causal left-pad of 3 ones, padded = [1,1,1,carry[0],...,carry[15]]
        # Output pos 0: padded[0:4] = [1,1,1,carry[0]] -> 1 (no boundary)
        # Output pos 1: padded[4:8] = [carry[1],carry[2],carry[3],carry[4]]
        # carry[2]=0 -> min=0
        assert out[:, 1, :].abs().max() < 1e-6

    def test_scale_1_identity(self):
        carry = torch.ones(BS, P, 1)
        carry[:, 3, :] = 0
        assert torch.equal(carry_min_pool(carry, 1), carry)


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

    def test_surprise_zero_at_init(self):
        """Before any boundary update, z_hat=None -> surprise is zero."""
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        z = torch.randn(BS, P, D_pc)
        delta = pcm.compute_surprise(z)
        assert torch.equal(delta, torch.zeros_like(z))

    def test_surprise_nonzero_after_boundary(self):
        """After boundary update, z_hat is set -> surprise is nonzero."""
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        pcm._lazy_init(BS, torch.device("cpu"))

        z_end = torch.randn(BS, D_pc)
        ctx_b = torch.randn(BS, D_h)
        pm_s = torch.randn(BS, D_h)
        em_s = torch.randn(BS, config.D_em)
        pcm.boundary_update(z_end, ctx_b, pm_s, em_s)

        # Now z_hat is set; surprise should be nonzero
        z = torch.randn(BS, P, D_pc)
        delta = pcm.compute_surprise(z)
        assert delta.abs().max() > 0

    def test_ffn_gain_shape(self):
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        delta = torch.randn(BS, P, D_pc)
        gain = pcm.compute_ffn_gain(delta)
        assert gain.shape == (BS, P, D_h)

    def test_ffn_gain_starts_near_one(self):
        """At init, W_gain is small -> gain approx 1."""
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        delta = torch.zeros(BS, P, D_pc)
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

        z_end = torch.randn(BS, D_pc)
        ctx_b = torch.randn(BS, D_h)
        pm_s = torch.randn(BS, D_h)
        em_s = torch.randn(BS, config.D_em)

        L_pred = pcm.boundary_update(z_end, ctx_b, pm_s, em_s)
        assert L_pred.dim() == 0  # scalar
        assert L_pred.item() >= 0  # MSE is non-negative

    def test_boundary_update_z_hat_has_grad(self):
        """z_hat should retain computation graph for predictor backprop."""
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        pcm._lazy_init(BS, torch.device("cpu"))

        z_end = torch.randn(BS, D_pc)
        ctx_b = torch.randn(BS, D_h)
        pm_s = torch.randn(BS, D_h)
        em_s = torch.randn(BS, config.D_em)
        pcm.boundary_update(z_end, ctx_b, pm_s, em_s)

        # z_hat should have a grad_fn (from the predictor forward pass)
        assert pcm.z_hat.grad_fn is not None

    def test_detach_states_preserves_z_hat_graph(self):
        """detach_states is intentionally a no-op for PCM."""
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        pcm._lazy_init(BS, torch.device("cpu"))

        z_end = torch.randn(BS, D_pc)
        ctx_b = torch.randn(BS, D_h)
        pm_s = torch.randn(BS, D_h)
        em_s = torch.randn(BS, config.D_em)
        pcm.boundary_update(z_end, ctx_b, pm_s, em_s)

        pcm.detach_states()
        # z_hat should STILL have a grad_fn
        assert pcm.z_hat.grad_fn is not None

    def test_reset_states_zeros_z_hat(self):
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        pcm._lazy_init(BS, torch.device("cpu"))

        z_end = torch.randn(BS, D_pc)
        ctx_b = torch.randn(BS, D_h)
        pm_s = torch.randn(BS, D_h)
        em_s = torch.randn(BS, config.D_em)
        pcm.boundary_update(z_end, ctx_b, pm_s, em_s)

        mask = torch.tensor([True, False])  # reset stream 0
        pcm.reset_states(mask)
        assert pcm.z_hat[0].abs().max() < 1e-8
        assert pcm.z_hat[1].abs().max() > 0  # stream 1 preserved


# ============================================================================
# Integration: PCM + multi-timescale in full model
# ============================================================================

def _make_pcm_model(pcm_enabled=True, block_scales=None, **extra):
    config = ModelConfig(
        D=D, B=B, L=L, D_pc=D_pc,
        pcm_enabled=pcm_enabled,
        block_scales=block_scales,
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


class TestMultiTimescaleIntegration:
    def test_forward_span_multiscale(self):
        """Multi-timescale blocks produce correct output shapes."""
        model = _make_pcm_model(pcm_enabled=False, block_scales=(1, 4))
        model.train()

        ids = torch.randint(0, 256, (BS, P))
        reset = torch.zeros(BS, dtype=torch.bool)
        model.initialize_states(BS, torch.device("cpu"))

        logits, x_emb, y_wm = model.forward_span(ids, reset)
        assert logits.shape == (BS, P, 256)

    def test_multiscale_with_pcm(self):
        """PCM + multi-timescale together."""
        model = _make_pcm_model(pcm_enabled=True, block_scales=(1, 4))
        model.train()

        ids = torch.randint(0, 256, (BS, P))
        reset = torch.zeros(BS, dtype=torch.bool)
        model.initialize_states(BS, torch.device("cpu"))

        logits, x_emb, y_wm = model.forward_span(ids, reset)
        assert logits.shape == (BS, P, 256)

        L_pred, L_recon = model.apply_pcm_boundary()
        assert L_pred.dim() == 0
        assert L_recon.dim() == 0

    def test_multiscale_validation(self):
        """Config validation catches bad block_scales."""
        with pytest.raises(ValueError, match="block_scales length"):
            config = ModelConfig(D=D, B=B, L=L, block_scales=(1,))
            config.validate()

        with pytest.raises(ValueError, match="must be >= 1"):
            config = ModelConfig(D=D, B=B, L=L, P=P, block_scales=(1, 3))
            config.validate()

    def test_scale_1_matches_no_scale(self):
        """block_scales=(1,1) gives same output as block_scales=None."""
        torch.manual_seed(42)
        model_a = _make_pcm_model(pcm_enabled=False, block_scales=None)
        torch.manual_seed(42)
        model_b = _make_pcm_model(pcm_enabled=False, block_scales=(1, 1))

        model_a.train()
        model_b.train()

        ids = torch.randint(0, 256, (BS, P))
        reset = torch.zeros(BS, dtype=torch.bool)

        model_a.initialize_states(BS, torch.device("cpu"))
        model_b.initialize_states(BS, torch.device("cpu"))

        logits_a, _, _ = model_a.forward_span(ids, reset)
        logits_b, _, _ = model_b.forward_span(ids, reset)

        assert torch.allclose(logits_a, logits_b, atol=1e-5)

    def test_step_multiscale_hold(self):
        """Slow block holds output on non-update steps."""
        model = _make_pcm_model(pcm_enabled=False, block_scales=(1, 4))
        model.eval()
        model.initialize_states(BS, torch.device("cpu"))

        block = model.blocks[1]  # scale=4
        assert block.scale == 4

        # Test block.step() directly
        x_block = torch.randn(BS, D_h)
        y_wm = torch.randn(BS, D)
        x_proj = torch.randn(BS, D)
        surprise = torch.zeros(BS, 1)
        carry = torch.ones(BS, 1)

        # Reset block counter for direct testing
        block._step_counter = 0
        block._last_step_output = None

        # Step 0: processes
        h0 = block.step(x_block, y_wm, x_proj, surprise, carry)
        assert block._step_counter == 1

        # Step 1: holds
        h1 = block.step(x_block * 999, y_wm, x_proj, surprise, carry)
        assert torch.equal(h0, h1), "Hold step should return cached output"
        assert block._step_counter == 2

    def test_step_multiscale_boundary_forces_process(self):
        """Doc boundary forces slow block to process immediately."""
        model = _make_pcm_model(pcm_enabled=False, block_scales=(1, 4))
        model.eval()
        model.initialize_states(BS, torch.device("cpu"))

        block = model.blocks[1]  # scale=4
        x_block = torch.randn(BS, D_h)
        y_wm = torch.randn(BS, D)
        x_proj = torch.randn(BS, D)
        surprise = torch.zeros(BS, 1)
        carry = torch.ones(BS, 1)

        # First real step
        block._step_counter = 0
        block._last_step_output = None
        _ = block.step(x_block, y_wm, x_proj, surprise, carry)
        assert block._step_counter == 1

        # Now send a boundary (carry=0) — should force processing
        carry_boundary = torch.zeros(BS, 1)
        _ = block.step(x_block, y_wm, x_proj, surprise, carry_boundary)
        # Counter should have been reset to 0 on boundary, then incremented to 1
        assert block._step_counter == 1


class TestPCMDetachment:
    def test_z_hat_detached_in_surprise(self):
        """δ = z - z_hat.detach(): LM loss should NOT reach predictor through δ."""
        config = _make_pcm_config()
        pcm = PredictiveCodingModule(config)
        pcm._lazy_init(BS, torch.device("cpu"))

        # Set z_hat via boundary update
        z_end = torch.randn(BS, D_pc)
        ctx_b = torch.randn(BS, D_h)
        pm_s = torch.randn(BS, D_h)
        em_s = torch.randn(BS, config.D_em)
        pcm.boundary_update(z_end, ctx_b, pm_s, em_s)

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
        z_end = torch.randn(BS, D_pc)
        ctx_b = torch.randn(BS, D_h)
        pm_s = torch.randn(BS, D_h)
        em_s = torch.randn(BS, config.D_em)
        pcm.boundary_update(z_end, ctx_b, pm_s, em_s)

        # Second boundary: L_pred should backprop through z_hat into predictor
        z_end2 = torch.randn(BS, D_pc)
        L_pred = pcm.boundary_update(z_end2, ctx_b, pm_s, em_s)
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
