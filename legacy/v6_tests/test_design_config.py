"""Tests for ModelConfig validation and tier presets (v5)."""

import pytest
from tests.conftest import make_tiny_config, TINY_DEFAULTS
from src.model.config import ModelConfig


class TestConfigValidation:
    def test_tiny_defaults_valid(self):
        cfg = make_tiny_config()
        cfg.validate()

    def test_invalid_B(self):
        with pytest.raises(ValueError, match="B"):
            make_tiny_config(B=0)

    def test_invalid_C(self):
        with pytest.raises(ValueError, match="C"):
            make_tiny_config(C=0)

    def test_invalid_D_not_divisible_by_C(self):
        with pytest.raises(ValueError, match="divisible"):
            make_tiny_config(D=65, C=2)

    def test_invalid_N(self):
        with pytest.raises(ValueError, match="N"):
            make_tiny_config(N=0)

    def test_invalid_L_total(self):
        with pytest.raises(ValueError, match="L_total"):
            make_tiny_config(L_total=0)

    def test_invalid_scan_expansion(self):
        with pytest.raises(ValueError, match="scan_expansion"):
            make_tiny_config(scan_expansion=0)

    def test_invalid_M(self):
        with pytest.raises(ValueError, match="M"):
            make_tiny_config(M=0)

    def test_invalid_n_trail_steps(self):
        with pytest.raises(ValueError, match="n_trail_steps"):
            make_tiny_config(n_trail_steps=0)

    def test_T_property(self):
        cfg = make_tiny_config(N=16, K_segments=3)
        assert cfg.T == 48

    def test_D_embed_defaults_to_D(self):
        cfg = ModelConfig(D=128, C=4)
        cfg.validate()
        assert cfg.D_embed == 128

    def test_D_embed_explicit(self):
        cfg = ModelConfig(D=128, D_embed=64, C=4)
        cfg.validate()
        assert cfg.D_embed == 64

    def test_invalid_D_embed(self):
        with pytest.raises(ValueError, match="D_embed"):
            cfg = ModelConfig(D=128, D_embed=-2, C=4)
            cfg.validate()

    def test_D_col_derived(self):
        cfg = make_tiny_config(D=64, C=2)
        assert cfg.D_col == 32

    def test_D_col_derived_large(self):
        cfg = make_tiny_config(D=128, C=4)
        assert cfg.D_col == 32


class TestTierPresets:
    def test_tier_a(self):
        cfg = ModelConfig.tier_a()
        cfg.validate()
        assert cfg.D == 2048
        assert cfg.D_embed == 768
        assert cfg.B == 4
        assert cfg.C == 16
        assert cfg.D_col == 128  # D=2048 // C=16
        assert cfg.L_total == 10
        assert cfg.L_mem == 5
        assert cfg.scan_expansion == 8
        assert cfg.d_inner == 1024

    def test_tier_b(self):
        cfg = ModelConfig.tier_b()
        cfg.validate()
        assert cfg.D == 3072
        assert cfg.D_embed == 1024
        assert cfg.B == 6
        assert cfg.C == 16
        assert cfg.L_total == 20
        assert cfg.L_mem == 10
        assert cfg.d_inner == 1024

    def test_tier_c(self):
        cfg = ModelConfig.tier_c()
        cfg.validate()
        assert cfg.D == 4096
        assert cfg.D_embed == 2048
        assert cfg.B == 8
        assert cfg.C == 16
        assert cfg.L_total == 28
        assert cfg.L_mem == 14
        assert cfg.d_inner == 2048

    def test_tier_tiny(self):
        cfg = ModelConfig.tier_tiny()
        cfg.validate()
        assert cfg.D == 64
        assert cfg.B == 2
        assert cfg.C == 2
        assert cfg.L_total == 4
        assert cfg.L_mem == 2
        assert cfg.d_inner == 64

    def test_tier_overrides(self):
        cfg = ModelConfig.tier_a(L_total=12, L_mem=6, N=256)
        cfg.validate()
        assert cfg.L_total == 12
        assert cfg.L_mem == 6
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
