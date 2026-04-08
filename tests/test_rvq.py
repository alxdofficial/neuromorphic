"""Unit tests for Residual VQ-VAE codebook module."""

import torch
import torch.nn.functional as F

from src.codebook import ResidualVQ, ActionVQVAE


class TestResidualVQ:
    def test_quantize_shape(self):
        rvq = ResidualVQ(num_levels=4, codes_per_level=16, code_dim=32)
        z = torch.randn(10, 32)
        z_q, codes = rvq.quantize(z)
        assert z_q.shape == z.shape
        assert codes.shape == (10, 4)
        assert codes.dtype == torch.long
        assert (codes >= 0).all() and (codes < 16).all()

    def test_forward_returns_loss(self):
        rvq = ResidualVQ(num_levels=2, codes_per_level=8, code_dim=16)
        z = torch.randn(5, 16, requires_grad=True)
        z_q_st, codes, commit_loss = rvq(z)
        assert z_q_st.shape == z.shape
        assert commit_loss.item() >= 0
        # Straight-through: gradient should flow z_q_st -> z
        z_q_st.sum().backward()
        assert z.grad is not None
        assert z.grad.abs().sum() > 0

    def test_ema_update_moves_codebook(self):
        rvq = ResidualVQ(num_levels=2, codes_per_level=4, code_dim=8, decay=0.5)
        z = torch.randn(100, 8) * 5.0  # far from init
        cb_before = rvq.codebooks.clone()
        _, codes = rvq.quantize(z)
        rvq.ema_update(z, codes)
        cb_after = rvq.codebooks.clone()
        assert (cb_after - cb_before).abs().sum().item() > 0

    def test_dead_code_resample(self):
        rvq = ResidualVQ(num_levels=1, codes_per_level=16, code_dim=8,
                         dead_code_threshold=0.5)
        # Force most codes to be "dead" by setting cluster_size manually.
        rvq.cluster_size.zero_()
        rvq.cluster_size[0, 0] = 100.0  # only code 0 is "alive"
        z = torch.randn(50, 8)
        rvq.resample_dead_codes(z)
        # After resampling, dead codes should have cluster_size reset to 1.0
        assert rvq.cluster_size[0, 1:].min().item() >= 1.0

    def test_usage_histogram_sums_to_one_per_level(self):
        rvq = ResidualVQ(num_levels=3, codes_per_level=8, code_dim=16)
        # Populate cluster_size
        rvq.cluster_size[0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        rvq.cluster_size[1] = torch.tensor([10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        rvq.cluster_size[2] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        usage = rvq.usage_histogram()
        assert usage.shape == (3, 8)
        for lvl in range(3):
            assert abs(usage[lvl].sum().item() - 1.0) < 1e-5


class TestActionVQVAE:
    def test_forward_end_to_end(self):
        vq = ActionVQVAE(action_dim=1056, latent_dim=32, hidden=128,
                         num_levels=4, codes_per_level=16)
        action = torch.randn(8, 1056)
        out = vq(action)
        assert out["recon"].shape == action.shape
        assert out["codes"].shape == (8, 4)
        assert out["loss"].item() > 0
        assert out["recon_loss"].item() > 0
        assert out["commit_loss"].item() >= 0

    def test_training_step_reduces_recon_loss(self):
        """Sanity check: one optimizer step should measurably improve recon
        loss on a small, fixed batch."""
        torch.manual_seed(0)
        vq = ActionVQVAE(action_dim=128, latent_dim=16, hidden=64,
                         num_levels=2, codes_per_level=8)
        action = torch.randn(64, 128)
        opt = torch.optim.AdamW(vq.parameters(), lr=1e-3)

        initial = vq(action)["recon_loss"].item()
        for _ in range(30):
            out = vq(action)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            opt.step()
            vq.rvq.ema_update(out["z"], out["codes"])
        final = vq(action)["recon_loss"].item()
        assert final < initial, f"recon did not improve: {initial} -> {final}"

    def test_encode_quantize_roundtrip(self):
        vq = ActionVQVAE(action_dim=64, latent_dim=16, hidden=64,
                         num_levels=3, codes_per_level=8)
        action = torch.randn(4, 64)
        z, z_q, codes = vq.encode_quantize(action)
        assert z.shape == (4, 16)
        assert z_q.shape == (4, 16)
        assert codes.shape == (4, 3)
        # z_q should be a sum of selected codebook entries
        expected = torch.zeros_like(z_q)
        for lvl in range(3):
            expected = expected + vq.rvq.codebooks[lvl][codes[:, lvl]]
        assert torch.allclose(z_q, expected, atol=1e-5)
