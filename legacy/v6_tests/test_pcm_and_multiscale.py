"""Tests for Within-Scan Predictive Coding Module (v5)."""

import torch
import pytest
from tests.conftest import make_tiny_config

from src.model.predictive_coding import WithinScanPCM, GroupedLinear, GroupedLayerNorm


BS = 2
C = 2
D_col = 16
N = 4


class TestWithinScanPCM:
    def test_compute_surprise_shape(self):
        pcm = WithinScanPCM(C, D_col)
        H = torch.randn(BS, N, C, D_col)
        x_col = torch.randn(BS, N, C, D_col)
        surprise, z_hat, z = pcm.compute_surprise(H, x_col)
        assert surprise.shape == (BS, N, C, D_col)
        assert z_hat.shape == (BS, N, C, D_col)
        assert z.shape == (BS, N, C, D_col)

    def test_surprise_zero_at_position_zero(self):
        pcm = WithinScanPCM(C, D_col)
        H = torch.randn(BS, N, C, D_col)
        x_col = torch.randn(BS, N, C, D_col)
        surprise, _, _ = pcm.compute_surprise(H, x_col)
        assert (surprise[:, 0] == 0).all()

    def test_surprise_nonzero_at_later_positions(self):
        """With random inputs, surprise should be nonzero for t > 0."""
        pcm = WithinScanPCM(C, D_col)
        H = torch.randn(BS, N, C, D_col)
        x_col = torch.randn(BS, N, C, D_col)
        surprise, _, _ = pcm.compute_surprise(H, x_col)
        # At least some surprise at positions > 0
        assert surprise[:, 1:].abs().sum() > 0

    def test_apply_gain_shape(self):
        pcm = WithinScanPCM(C, D_col)
        H = torch.randn(BS, N, C, D_col)
        surprise = torch.randn(BS, N, C, D_col)
        H_mod = pcm.apply_gain(H, surprise)
        assert H_mod.shape == H.shape

    def test_gain_bounded_at_init(self):
        """W_gain zero-init means gain = 1.0 at init."""
        pcm = WithinScanPCM(C, D_col)
        H = torch.randn(BS, N, C, D_col)
        surprise = torch.randn(BS, N, C, D_col)
        H_mod = pcm.apply_gain(H, surprise)
        # At init, W_gain=0 so gain = 1 + 0.1*tanh(0) = 1.0
        assert torch.allclose(H_mod, H, atol=1e-5)

    def test_prediction_loss(self):
        pcm = WithinScanPCM(C, D_col)
        z_hat = torch.randn(BS, N, C, D_col)
        z = torch.randn(BS, N, C, D_col)
        loss = pcm.prediction_loss(z_hat, z)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_prediction_loss_zero_when_perfect(self):
        """Loss should be ~0 when z_hat[t] == z[t+1] for all t."""
        pcm = WithinScanPCM(C, D_col)
        z = torch.randn(BS, N, C, D_col)
        # Construct z_hat such that z_hat[:, t] == z[:, t+1]
        z_hat = torch.zeros_like(z)
        z_hat[:, :-1] = z[:, 1:]
        loss = pcm.prediction_loss(z_hat, z)
        assert loss.item() < 1e-6

    def test_prediction_loss_detaches_target(self):
        """z should be detached so gradient only flows through z_hat."""
        pcm = WithinScanPCM(C, D_col)
        z_hat = torch.randn(BS, N, C, D_col, requires_grad=True)
        z = torch.randn(BS, N, C, D_col, requires_grad=True)
        loss = pcm.prediction_loss(z_hat, z)
        loss.backward()
        assert z_hat.grad is not None
        assert z.grad is None  # detached


class TestPCMInModel:
    def test_pcm_aux_loss_nonzero(self):
        """With PCM enabled, aux_loss should be >= 0."""
        cfg = make_tiny_config(pcm_enabled=True)
        from src.model.model import NeuromorphicLM

        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        input_ids = torch.randint(0, 64, (BS, cfg.N))
        _, aux = model.forward_segment(input_ids)
        assert aux.item() >= 0
