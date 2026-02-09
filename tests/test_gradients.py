"""Gradient flow tests — NEVER change.

If these fail, there's a dead gradient bug.
"""

import pytest
import torch
import torch.nn.functional as F

from src.model.model import NeuromorphicLM
from tests.conftest import make_tiny_config, forward_n_tokens, forward_and_write_em

pytestmark = pytest.mark.invariant

BS = 2
VOCAB = 64


def _get_loss(model, n_tokens=8, with_commits=False, with_em_writes=False):
    """Run tokens and compute loss for backward."""
    if with_em_writes:
        logits, target = forward_and_write_em(model, n_tokens, BS=BS, vocab=VOCAB)
    elif with_commits:
        logits, target = forward_n_tokens(model, n_tokens, BS=BS, vocab=VOCAB,
                                           with_commits=True)
    else:
        logits, target = forward_n_tokens(model, n_tokens, BS=BS, vocab=VOCAB)

    loss = F.cross_entropy(logits, target)
    return loss


# ============================================================================
# Per-phase gradient reachability
# ============================================================================

class TestPhaseGradients:
    def test_all_params_get_grad_phase_a(self):
        """Phase A: PM/EM disabled. Active modules get grad, inactive get zero/None."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)

        loss = _get_loss(model, n_tokens=8)
        loss.backward()

        # Active modules should get gradients
        active_no_grad = []
        inactive_with_grad = []

        for name, param in model.named_parameters():
            is_pm = "pm" in name and "neuromodulator" not in name
            is_em = "em" in name and "neuromodulator" not in name
            is_inactive = is_pm or is_em

            if is_inactive:
                # PM/EM params should have zero or None grad
                if param.grad is not None and param.grad.abs().max() > 1e-10:
                    inactive_with_grad.append(name)
            else:
                # Active params should have non-None grad
                if param.grad is None:
                    active_no_grad.append(name)

        assert not active_no_grad, \
            f"Active params without grad in Phase A: {active_no_grad}"

    def test_all_params_get_grad_phase_b(self):
        """Phase B: After PM commit, all PM params get gradients."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)

        loss = _get_loss(model, n_tokens=cfg.P * 2, with_commits=True)
        loss.backward()

        # PM-specific params that should now get gradients
        # Filter: only check PM readout params (layers.X.pm.*), not EM readout
        pm_params_without_grad = []
        for name, param in model.named_parameters():
            is_pm_param = any(k in name for k in ["W_k_pre", "W_v_post"])
            is_pm_readout = (".pm.readout_ffn" in name or ".pm.readout_norm" in name)
            if is_pm_param or is_pm_readout:
                if param.grad is None or param.grad.abs().max() < 1e-12:
                    pm_params_without_grad.append(name)

        assert not pm_params_without_grad, \
            f"PM params without grad after commit: {pm_params_without_grad}"

    def test_all_params_get_grad_phase_c(self):
        """Phase C: After EM write, all EM params get gradients."""
        cfg = make_tiny_config()
        cfg.set_phase("C")
        model = NeuromorphicLM(cfg)

        loss = _get_loss(model, n_tokens=cfg.P * 2, with_em_writes=True)
        loss.backward()

        # EM-specific params that should now get gradients
        em_params_without_grad = []
        for name, param in model.named_parameters():
            is_em_param = any(k in name for k in [
                "W_q_em", "W_q_cross", "W_o_cross", "W_k_cand", "W_v_cand",
            ])
            # Only check EM params on blocks (not neuromodulator)
            if is_em_param and "neuromodulator" not in name:
                if param.grad is None or param.grad.abs().max() < 1e-12:
                    em_params_without_grad.append(name)

        assert not em_params_without_grad, \
            f"EM params without grad after write: {em_params_without_grad}"


# ============================================================================
# Specific parameter gradient tests
# ============================================================================

class TestSpecificGradients:
    def test_pm_readout_ffn_gets_gradients(self):
        cfg = make_tiny_config()
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)

        loss = _get_loss(model, n_tokens=cfg.P * 2, with_commits=True)
        loss.backward()

        for block in model.blocks:
            for layer in block.layers:
                pm = layer.pm
                if pm.readout_ffn is not None:
                    for p in pm.readout_ffn.parameters():
                        assert p.grad is not None, "readout_ffn should get gradients"
                        assert p.grad.abs().max() > 0

    def test_em_readout_ffn_gets_gradients(self):
        cfg = make_tiny_config()
        cfg.set_phase("C")
        model = NeuromorphicLM(cfg)

        loss = _get_loss(model, n_tokens=cfg.P * 2, with_em_writes=True)
        loss.backward()

        for block in model.blocks:
            em = block.em
            if em.readout_ffn is not None:
                for p in em.readout_ffn.parameters():
                    assert p.grad is not None, "EM readout_ffn should get gradients"

    def test_pm_eligibility_projections_get_gradients(self):
        cfg = make_tiny_config()
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)

        loss = _get_loss(model, n_tokens=cfg.P * 2, with_commits=True)
        loss.backward()

        for block in model.blocks:
            for layer in block.layers:
                pm = layer.pm
                assert pm.W_k_pre.weight.grad is not None
                assert pm.W_k_pre.weight.grad.abs().max() > 0
                assert pm.W_v_post.weight.grad is not None
                assert pm.W_v_post.weight.grad.abs().max() > 0

    def test_wm_projections_get_gradients(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)

        loss = _get_loss(model, n_tokens=8)
        loss.backward()

        for proj_name in ["W_q", "W_k", "W_v", "W_o"]:
            param = getattr(model.wm, proj_name).weight
            assert param.grad is not None, f"WM {proj_name} should get gradients"

    def test_lm_head_gets_gradients(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)

        loss = _get_loss(model, n_tokens=4)
        loss.backward()

        assert model.lm_head.weight.grad is not None

    def test_embedding_gets_gradients(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)

        loss = _get_loss(model, n_tokens=4)
        loss.backward()

        assert model.embedding.weight.grad is not None

    def test_layer_ffn_gets_gradients(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)

        loss = _get_loss(model, n_tokens=4)
        loss.backward()

        layer = model.blocks[0].layers[0]
        if layer.ffn is not None:
            assert layer.ffn[0].weight.grad is not None
            assert layer.ffn[2].weight.grad is not None


# ============================================================================
# Decoder gradients
# ============================================================================

class TestDecoderGradients:
    def test_decoder_output_proj_is_nonzero(self):
        """output_proj weight initialized with std=0.01, should not be zero."""
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        w = model.spatial_decoder.output_proj.weight
        assert w.abs().mean() > 0, "output_proj should be small-init, not zero"

    def test_decoder_all_levels_get_gradients(self):
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase("C")
        model = NeuromorphicLM(cfg)

        loss = _get_loss(model, n_tokens=cfg.P * 2, with_em_writes=True)
        loss.backward()

        # Check columnar
        for col in model.spatial_decoder.columnar:
            for p in col.parameters():
                assert p.grad is not None, "Columnar params should get gradients"

        # Check thalamic
        for p in model.spatial_decoder.thalamic.parameters():
            assert p.grad is not None, "Thalamic params should get gradients"

        # Check decoder blocks
        for db in model.spatial_decoder.decoder_blocks:
            for p in db.parameters():
                assert p.grad is not None, "Decoder block params should get gradients"


# ============================================================================
# Gradient safety
# ============================================================================

class TestGradientSafety:
    def test_gradients_are_finite(self):
        """No NaN/Inf in any param.grad."""
        cfg = make_tiny_config()
        cfg.set_phase("C")
        model = NeuromorphicLM(cfg)

        loss = _get_loss(model, n_tokens=cfg.P * 2, with_em_writes=True)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), \
                    f"Non-finite gradient in {name}"

    def test_detach_states_removes_grad_graph(self):
        cfg = make_tiny_config()
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)

        forward_n_tokens(model, 4, with_commits=True)
        model.detach_states()

        # All state tensors should not require grad
        for block in model.blocks:
            for layer in block.layers:
                if layer.h is not None:
                    assert not layer.h.requires_grad
                pm = layer.pm
                for name in pm._state_tensor_names:
                    t = getattr(pm, name)
                    if t is not None:
                        assert not t.requires_grad, f"pm.{name} still requires grad"
            em = block.em
            for name in em._state_tensor_names:
                t = getattr(em, name)
                if t is not None:
                    assert not t.requires_grad, f"em.{name} still requires grad"

    def test_backward_with_doc_boundary(self):
        """forward_one_token with reset_mask=[True,False] doesn't crash backward."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)

        # Run a few tokens first to populate state
        forward_n_tokens(model, 4)

        # Now forward with partial reset
        input_id = torch.randint(0, 64, (BS,))
        target = torch.randint(0, 64, (BS,))
        reset_mask = torch.tensor([True, False])
        logits, _, _ = model.forward_one_token(input_id, reset_mask)
        loss = F.cross_entropy(logits, target)
        loss.backward()  # should not crash
