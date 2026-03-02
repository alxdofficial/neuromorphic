"""Gradient flow tests (v5) — NEVER change.

If these fail, there's a dead gradient bug.
"""

import torch
import torch.nn.functional as F
import pytest
from tests.conftest import make_tiny_config

from src.model.model import NeuromorphicLM


BS = 2
VOCAB = 64


def _compute_loss(model, N=None, n_segments=1):
    """Forward n_segments and compute CE loss. Returns loss scalar."""
    if N is None:
        N = model.config.N

    model.initialize_states(BS, torch.device("cpu"))
    total_loss = torch.tensor(0.0)
    for _ in range(n_segments):
        input_ids = torch.randint(0, VOCAB, (BS, N))
        target_ids = torch.randint(0, VOCAB, (BS, N))
        reset_mask = torch.zeros(BS, dtype=torch.bool)
        logits, aux_loss = model.forward_segment(input_ids, reset_mask)
        ce_loss = F.cross_entropy(logits.reshape(-1, VOCAB), target_ids.reshape(-1))
        total_loss = total_loss + ce_loss + aux_loss
    return total_loss


class TestGradientFlow:
    def test_all_params_get_gradient(self):
        """Every parameter should receive a gradient (needs 2 segments for EM)."""
        cfg = make_tiny_config(pcm_enabled=True)
        model = NeuromorphicLM(cfg)

        # 2 segments: first populates EM, second reads from it
        loss = _compute_loss(model, n_segments=2)
        loss.backward()

        no_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                no_grad.append(name)

        assert len(no_grad) == 0, f"Parameters with no gradient: {no_grad}"

    def test_nonzero_gradients(self):
        """Gradients should be nonzero for key parameters."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)

        loss = _compute_loss(model)
        loss.backward()

        for name in ["embedding.weight", "lm_head.weight"]:
            parts = name.split(".")
            obj = model
            for p in parts:
                obj = getattr(obj, p)
            assert obj.grad is not None, f"No gradient for {name}"
            assert obj.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_scan_layer_gradients(self):
        """All scan layers in both stages should get gradients."""
        cfg = make_tiny_config(L_scan=3)
        model = NeuromorphicLM(cfg)

        loss = _compute_loss(model)
        loss.backward()

        for stage_name in ("stage1", "stage3"):
            stage = getattr(model, stage_name)
            for i, layer in enumerate(stage):
                assert layer.proj_in.weight.grad is not None, \
                    f"No gradient for {stage_name}[{i}].proj_in"
                assert layer.proj_in.weight.grad.abs().sum() > 0, \
                    f"Zero gradient for {stage_name}[{i}].proj_in"
                assert layer.proj_out.weight.grad is not None, \
                    f"No gradient for {stage_name}[{i}].proj_out"
                assert layer.proj_out.weight.grad.abs().sum() > 0, \
                    f"Zero gradient for {stage_name}[{i}].proj_out"

    def test_gradient_with_pcm(self):
        """PCM parameters should get gradients when enabled."""
        cfg = make_tiny_config(pcm_enabled=True)
        model = NeuromorphicLM(cfg)

        loss = _compute_loss(model)
        loss.backward()

        pcm = model.pcm
        assert pcm is not None
        assert pcm.W_enc.weight.grad is not None
        assert pcm.W_enc.weight.grad.abs().sum() > 0
        assert pcm.W_pcm.weight.grad is not None
        assert pcm.W_pcm.weight.grad.abs().sum() > 0

    def test_pm_lr_gradient(self):
        """PM learning rate (raw_lr_pm) should get gradient (needs PCM for surprise)."""
        cfg = make_tiny_config(pcm_enabled=True)
        model = NeuromorphicLM(cfg)

        loss = _compute_loss(model)
        loss.backward()

        assert model.pm.raw_lr_pm.grad is not None
        assert model.pm.raw_lr_pm.grad.abs().sum() > 0

    def test_em_trail_params_gradient(self):
        """EM trail parameters (w1, w2, gate_bias, tau, sigma) should get gradients."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)

        # Need 2 segments: first populates EM, second reads
        model.initialize_states(BS, torch.device("cpu"))
        N = cfg.N
        total_loss = torch.tensor(0.0)
        for seg in range(2):
            input_ids = torch.randint(0, VOCAB, (BS, N))
            target_ids = torch.randint(0, VOCAB, (BS, N))
            reset_mask = torch.zeros(BS, dtype=torch.bool)
            logits, aux_loss = model.forward_segment(input_ids, reset_mask)
            ce_loss = F.cross_entropy(logits.reshape(-1, VOCAB), target_ids.reshape(-1))
            total_loss = total_loss + ce_loss + aux_loss

        total_loss.backward()

        em = model.em
        for name in ["gate_alpha", "gate_bias", "raw_tau", "raw_sigma"]:
            param = getattr(em, name)
            assert param.grad is not None, f"No gradient for em.{name}"

    def test_w_seed_w_gradient(self):
        """Fused W_seed_w projection should get gradients (need 2 segments for EM)."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)

        loss = _compute_loss(model, n_segments=2)
        loss.backward()

        assert model.W_seed_w.weight.grad is not None
        assert model.W_seed_w.weight.grad.abs().sum() > 0

    def test_proj_up_down_gradient(self):
        """proj_up/proj_down should receive gradients when D_embed != D."""
        cfg = make_tiny_config(D=64, D_embed=32)
        model = NeuromorphicLM(cfg)

        loss = _compute_loss(model)
        loss.backward()

        assert model.proj_up.weight.grad is not None
        assert model.proj_up.weight.grad.abs().sum() > 0
        assert model.proj_down.weight.grad is not None
        assert model.proj_down.weight.grad.abs().sum() > 0

    def test_pos_embed_gradient(self):
        """Position embedding should receive gradients."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)

        loss = _compute_loss(model)
        loss.backward()

        assert model.pos_embed.grad is not None
        assert model.pos_embed.grad.abs().sum() > 0

    def test_em_neuromod_gradient(self):
        """EM neuromodulator should get gradient (through cross-segment TBPTT)."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)

        # Need 2 segments for cross-segment gradient
        model.initialize_states(BS, torch.device("cpu"))
        N = cfg.N
        total_loss = torch.tensor(0.0)
        for seg in range(2):
            input_ids = torch.randint(0, VOCAB, (BS, N))
            target_ids = torch.randint(0, VOCAB, (BS, N))
            reset_mask = torch.zeros(BS, dtype=torch.bool)
            logits, aux_loss = model.forward_segment(input_ids, reset_mask)
            ce_loss = F.cross_entropy(logits.reshape(-1, VOCAB), target_ids.reshape(-1))
            total_loss = total_loss + ce_loss + aux_loss

        total_loss.backward()

        neuromod = model.em_neuromod
        assert neuromod.backbone[0].weight.grad is not None
