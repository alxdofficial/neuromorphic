"""Tests for ModelConfig validation and tier presets (v4)."""

import pytest
from tests.conftest import make_tiny_config, TINY_DEFAULTS
from src.model.config import ModelConfig


class TestConfigValidation:
    def test_tiny_defaults_valid(self):
        cfg = make_tiny_config()
        cfg.validate()

    def test_invalid_R(self):
        with pytest.raises(ValueError, match="R"):
            make_tiny_config(R=0)

    def test_invalid_B_blocks(self):
        with pytest.raises(ValueError, match="B_blocks"):
            make_tiny_config(B_blocks=0)

    def test_invalid_C(self):
        with pytest.raises(ValueError, match="C"):
            make_tiny_config(C=0)

    def test_invalid_D_col(self):
        with pytest.raises(ValueError, match="D_col"):
            make_tiny_config(D_col=0)

    def test_invalid_D_mem(self):
        with pytest.raises(ValueError, match="D_mem"):
            make_tiny_config(D_mem=0)

    def test_invalid_D_pcm_when_enabled(self):
        with pytest.raises(ValueError, match="D_pcm"):
            make_tiny_config(pcm_enabled=True, D_pcm=0)

    def test_invalid_N(self):
        with pytest.raises(ValueError, match="N"):
            make_tiny_config(N=0)

    def test_k_ret_exceeds_M(self):
        with pytest.raises(ValueError, match="k_ret"):
            make_tiny_config(k_ret=100, M=8)

    def test_T_property(self):
        cfg = make_tiny_config(N=16, K_segments=3)
        assert cfg.T == 48


class TestTierPresets:
    def test_tier_a(self):
        cfg = ModelConfig.tier_a()
        cfg.validate()
        assert cfg.D == 768
        assert cfg.B_blocks == 6
        assert cfg.C == 4
        assert cfg.D_col == 128

    def test_tier_b(self):
        cfg = ModelConfig.tier_b()
        cfg.validate()
        assert cfg.D == 1536
        assert cfg.B_blocks == 8

    def test_tier_c(self):
        cfg = ModelConfig.tier_c()
        cfg.validate()
        assert cfg.D == 2048
        assert cfg.B_blocks == 12

    def test_tier_overrides(self):
        cfg = ModelConfig.tier_a(R=6, N=256)
        assert cfg.R == 6
        assert cfg.N == 256


class TestPhaseToggle:
    def test_phase_a(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        assert cfg.pm_enabled
        assert cfg.em_enabled
        assert not cfg.lifelong_mode

    def test_phase_b(self):
        cfg = make_tiny_config()
        cfg.set_phase("B")
        assert cfg.pm_enabled
        assert cfg.em_enabled
        assert cfg.lifelong_mode

    def test_invalid_phase(self):
        cfg = make_tiny_config()
        with pytest.raises(ValueError, match="Unknown phase"):
            cfg.set_phase("X")
