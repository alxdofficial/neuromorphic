"""Design-specific config tests — update design_constants.py when design changes."""

import pytest
from dataclasses import fields

from src.model.config import ModelConfig
from tests.design_constants import (
    TIER_A, TIER_B, TIER_C, DEFAULTS, PHASE_TOGGLES,
)

pytestmark = pytest.mark.design


# ============================================================================
# Tier defaults
# ============================================================================

class TestTierDefaults:
    def test_tier_a_defaults(self):
        cfg = ModelConfig.tier_a()
        for key, val in TIER_A.items():
            assert getattr(cfg, key) == val, f"tier_a.{key}: {getattr(cfg, key)} != {val}"

    def test_tier_b_defaults(self):
        cfg = ModelConfig.tier_b()
        for key, val in TIER_B.items():
            assert getattr(cfg, key) == val, f"tier_b.{key}: {getattr(cfg, key)} != {val}"

    def test_tier_c_defaults(self):
        cfg = ModelConfig.tier_c()
        for key, val in TIER_C.items():
            assert getattr(cfg, key) == val, f"tier_c.{key}: {getattr(cfg, key)} != {val}"

    def test_tier_a_overrides(self):
        cfg = ModelConfig.tier_a(D=1024, L=16)
        assert cfg.D == 1024
        assert cfg.L == 16

    def test_tier_b_overrides(self):
        cfg = ModelConfig.tier_b(D=1024)
        assert cfg.D == 1024

    def test_tier_c_overrides(self):
        cfg = ModelConfig.tier_c(D=2048)
        assert cfg.D == 2048


# ============================================================================
# Default hyperparameters
# ============================================================================

class TestDefaults:
    def test_default_hyperparameters(self):
        cfg = ModelConfig()
        for key, val in DEFAULTS.items():
            assert getattr(cfg, key) == val, \
                f"Default {key}: {getattr(cfg, key)} != {val}"


# ============================================================================
# Phase toggles
# ============================================================================

class TestPhaseToggles:
    @pytest.mark.parametrize("phase", ["A", "B", "C", "D"])
    def test_phase_toggles(self, phase):
        cfg = ModelConfig()
        cfg.set_phase(phase)
        expected = PHASE_TOGGLES[phase]
        for key, val in expected.items():
            assert getattr(cfg, key) == val, \
                f"Phase {phase}: {key} = {getattr(cfg, key)}, expected {val}"

    def test_phase_e_inherits_rl_enabled(self):
        """set_phase('E') does NOT set rl_enabled — inherits prior value."""
        # From Phase D (rl_enabled=True)
        cfg1 = ModelConfig()
        cfg1.set_phase("D")
        assert cfg1.rl_enabled is True
        cfg1.set_phase("E")
        assert cfg1.rl_enabled is True, "Phase E should inherit rl_enabled=True from D"
        assert cfg1.lifelong_mode is True

        # From Phase C (rl_enabled=False)
        cfg2 = ModelConfig()
        cfg2.set_phase("C")
        assert cfg2.rl_enabled is False
        cfg2.set_phase("E")
        assert cfg2.rl_enabled is False, "Phase E should inherit rl_enabled=False from C"
        assert cfg2.lifelong_mode is True

    def test_phase_e_enables_lifelong(self):
        cfg = ModelConfig()
        cfg.set_phase("E")
        assert cfg.lifelong_mode is True

    def test_non_e_disables_lifelong(self):
        cfg = ModelConfig()
        cfg.set_phase("E")
        cfg.set_phase("D")
        assert cfg.lifelong_mode is False

    def test_invalid_phase_raises(self):
        cfg = ModelConfig()
        with pytest.raises(ValueError, match="Unknown phase"):
            cfg.set_phase("X")


# ============================================================================
# D_h property
# ============================================================================

class TestDhProperty:
    def test_d_h_computation(self):
        cfg = ModelConfig(D=512, B=4)
        assert cfg.D_h == 128

    def test_d_h_with_different_B(self):
        cfg = ModelConfig(D=768, B=6)
        assert cfg.D_h == 128


# ============================================================================
# Validation
# ============================================================================

class TestValidation:
    def test_d_not_divisible_by_b_raises(self):
        cfg = ModelConfig(D=100, B=3)
        with pytest.raises(ValueError, match="D.*must be divisible by B"):
            cfg.validate()

    def test_d_wm_not_divisible_by_heads_raises(self):
        cfg = ModelConfig(D_wm=100, n_heads_wm=3)
        with pytest.raises(ValueError, match="D_wm.*must be divisible by.*n_heads_wm"):
            cfg.validate()

    def test_decoder_d_dec_not_divisible_raises(self):
        cfg = ModelConfig(snapshot_enabled=True, d_dec=100, n_heads_decoder=3)
        with pytest.raises(ValueError, match="d_dec.*must be divisible by.*n_heads_decoder"):
            cfg.validate()

    def test_decoder_d_h_too_small_raises(self):
        cfg = ModelConfig(D=8, B=4, snapshot_enabled=True,
                          d_dec=16, n_heads_decoder=4)
        # D_h = 8/4 = 2, n_heads_decoder = 4 -> D_h < n_heads_decoder
        with pytest.raises(ValueError, match="D_h.*must be >= n_heads_decoder"):
            cfg.validate()

    def test_valid_config_passes(self):
        cfg = ModelConfig(D=512, B=4, D_wm=128, n_heads_wm=4)
        cfg.validate()  # should not raise

    def test_valid_decoder_config_passes(self):
        cfg = ModelConfig(D=512, B=4, D_wm=128, n_heads_wm=4,
                          snapshot_enabled=True, d_dec=256, n_heads_decoder=4)
        cfg.validate()  # should not raise
