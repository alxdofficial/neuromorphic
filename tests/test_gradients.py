"""Gradient flow tests (v4) — NEVER change.

If these fail, there's a dead gradient bug.
"""

import torch
import torch.nn.functional as F
import pytest
from tests.conftest import make_tiny_config

from src.model.model import NeuromorphicLM


BS = 2
VOCAB = 64


def _compute_loss(model, N=None):
    """Forward one segment and compute CE loss. Returns loss scalar."""
    if N is None:
        N = model.config.N
    input_ids = torch.randint(0, VOCAB, (BS, N))
    target_ids = torch.randint(0, VOCAB, (BS, N))
    reset_mask = torch.zeros(BS, dtype=torch.bool)

    model.initialize_states(BS, torch.device("cpu"))
    logits, aux_loss = model.forward_segment(input_ids, reset_mask)
    ce_loss = F.cross_entropy(logits.reshape(-1, VOCAB), target_ids.reshape(-1))
    return ce_loss + aux_loss


class TestGradientFlow:
    def test_all_params_get_gradient(self):
        """Every parameter should receive a gradient from a single forward+loss."""
        cfg = make_tiny_config(pcm_enabled=True)
        model = NeuromorphicLM(cfg)

        loss = _compute_loss(model)
        loss.backward()

        # decay_head only affects em_S state evolution (detached between segments),
        # so gradient doesn't flow through the loss in a single segment
        expected_no_grad = {"decay_head"}

        no_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                if not any(skip in name for skip in expected_no_grad):
                    no_grad.append(name)

        assert len(no_grad) == 0, f"Parameters with no gradient: {no_grad}"

    def test_nonzero_gradients(self):
        """Gradients should be nonzero for key parameters."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)

        loss = _compute_loss(model)
        loss.backward()

        # Check key structural parameters
        for name in ["embedding.weight", "fan_out.weight", "fan_in.weight",
                      "lm_head.weight", "lambda_logit"]:
            parts = name.split(".")
            obj = model
            for p in parts:
                obj = getattr(obj, p)
            assert obj.grad is not None, f"No gradient for {name}"
            assert obj.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_gradient_with_pcm(self):
        """PCM parameters should get gradients when enabled."""
        cfg = make_tiny_config(pcm_enabled=True)
        model = NeuromorphicLM(cfg)

        # Two passes needed: first pass generates z_hat, second generates loss
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

        # Check PCM encoder gets gradient
        pcm = model.columns.pcm
        assert pcm is not None
        assert pcm.encoder.weight.grad is not None
        assert pcm.encoder.weight.grad.abs().sum() > 0

    def test_gradient_through_pm_read(self):
        """PM read should be differentiable."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)

        # Run 2 segments so PM has content from first
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

        # PM projection weights should get gradient
        col = model.columns
        assert col.W_pm_up.weight.grad is not None

    def test_gradient_through_em_read(self):
        """EM read should be differentiable."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)

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

        col = model.columns
        assert col.W_em_up.weight.grad is not None

    def test_lambda_logit_gradient(self):
        """Damped mixing parameter should get gradient."""
        cfg = make_tiny_config(R=3)  # Need R>1 for damped mixing
        model = NeuromorphicLM(cfg)

        loss = _compute_loss(model)
        loss.backward()

        assert model.lambda_logit.grad is not None
        assert model.lambda_logit.grad.abs() > 0
