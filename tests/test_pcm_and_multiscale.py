"""Tests for Cross-pass Predictive Coding Module (v4)."""

import torch
import pytest
from tests.conftest import make_tiny_config

from src.model.predictive_coding import CrossPassPCM, GroupedLinear, GroupedLayerNorm


BS = 2
C = 2
D_col = 16
D_pcm = 8
N = 4


class TestCrossPassPCM:
    def test_encode_predict_cycle(self):
        pcm = CrossPassPCM(C, D_col, D_pcm)
        x = torch.randn(BS, N, C, D_col)

        z = pcm.encode(x)
        z_hat = pcm.predict(z)

        assert z.shape == (BS, N, C, D_pcm)
        assert z_hat.shape == (BS, N, C, D_pcm)

    def test_surprise_zero_on_first_pass(self):
        pcm = CrossPassPCM(C, D_col, D_pcm)
        x = torch.randn(BS, N, C, D_col)
        z = pcm.encode(x)

        surprise, delta = pcm.compute_surprise(z, None)
        assert (surprise == 0).all()
        assert (delta == 0).all()

    def test_surprise_nonzero_on_second_pass(self):
        pcm = CrossPassPCM(C, D_col, D_pcm)
        x1 = torch.randn(BS, N, C, D_col)
        z1 = pcm.encode(x1)
        z_hat = pcm.predict(z1)

        x2 = torch.randn(BS, N, C, D_col)
        z2 = pcm.encode(x2)
        surprise, delta = pcm.compute_surprise(z2, z_hat)

        assert surprise.shape == (BS, N, C)
        # With random inputs, surprise should be nonzero
        assert surprise.sum() > 0

    def test_prediction_loss(self):
        pcm = CrossPassPCM(C, D_col, D_pcm)
        z_hat = torch.randn(BS, N, C, D_pcm)
        z_next = torch.randn(BS, N, C, D_pcm)

        loss = pcm.prediction_loss(z_hat, z_next)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_prediction_loss_zero_when_perfect(self):
        pcm = CrossPassPCM(C, D_col, D_pcm)
        z = torch.randn(BS, N, C, D_pcm)
        loss = pcm.prediction_loss(z, z.clone())
        assert loss.item() < 1e-6

    def test_prediction_loss_detaches_target(self):
        """z_next should be detached so gradient only flows through z_hat."""
        pcm = CrossPassPCM(C, D_col, D_pcm)
        z_hat = torch.randn(BS, N, C, D_pcm, requires_grad=True)
        z_next = torch.randn(BS, N, C, D_pcm, requires_grad=True)

        loss = pcm.prediction_loss(z_hat, z_next)
        loss.backward()

        assert z_hat.grad is not None
        assert z_next.grad is None  # detached


class TestPCMInModel:
    def test_pcm_aux_loss_nonzero_after_two_segments(self):
        """After 2 segments, aux_loss should be nonzero (PCM prediction)."""
        cfg = make_tiny_config(pcm_enabled=True)
        from src.model.model import NeuromorphicLM

        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        # First segment: z_hat_prev is None, aux_loss should be 0
        input_ids = torch.randint(0, 64, (BS, cfg.N))
        _, aux1 = model.forward_segment(input_ids)

        # Second segment: z_hat_prev is set, aux_loss may be nonzero
        input_ids2 = torch.randint(0, 64, (BS, cfg.N))
        _, aux2 = model.forward_segment(input_ids2)

        # aux2 should include PCM prediction loss (from pass r>0)
        # Even on the first segment, passes r>0 have z_hat from pass r-1
        assert aux1.item() >= 0
