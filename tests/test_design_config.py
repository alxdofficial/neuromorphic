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

    def test_invalid_D_not_divisible_by_C(self):
        with pytest.raises(ValueError, match="divisible"):
            make_tiny_config(D=65, C=2)

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

    def test_invalid_mask_rate_low(self):
        with pytest.raises(ValueError, match="mask_rate"):
            make_tiny_config(mask_rate=-0.1)

    def test_invalid_mask_rate_high(self):
        with pytest.raises(ValueError, match="mask_rate"):
            make_tiny_config(mask_rate=1.1)

    def test_invalid_span_mask_prob(self):
        with pytest.raises(ValueError, match="span_mask_prob"):
            make_tiny_config(span_mask_prob=1.5)

    def test_valid_fitb_defaults(self):
        cfg = make_tiny_config()
        assert cfg.fitb_id == 63
        assert cfg.null_id == 62
        assert cfg.mask_rate == 0.0

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

    def test_position_attn_dim_auto_derives(self):
        cfg = make_tiny_config()  # D=64, C=2 -> D_col=32
        assert cfg.position_attn_dim == 8  # 32 // 4

    def test_position_attn_dim_explicit_zero_disables(self):
        cfg = make_tiny_config(position_attn_dim=0)
        assert cfg.position_attn_dim == 0

    def test_position_attn_dim_explicit_value(self):
        cfg = make_tiny_config(position_attn_dim=16)
        assert cfg.position_attn_dim == 16

    def test_position_attn_dim_invalid_negative(self):
        with pytest.raises(ValueError, match="position_attn_dim"):
            make_tiny_config(position_attn_dim=-5)


class TestTierPresets:
    def test_tier_a(self):
        cfg = ModelConfig.tier_a()
        cfg.validate()
        assert cfg.D == 2048
        assert cfg.D_embed == 384
        assert cfg.B_blocks == 6
        assert cfg.C == 16
        assert cfg.D_col == 128  # D=2048 // C=16
        assert cfg.R == 4
        assert cfg.ffn_depth == 3
        assert cfg.ffn_expansion == 4

    def test_tier_b(self):
        cfg = ModelConfig.tier_b()
        cfg.validate()
        assert cfg.D == 3072
        assert cfg.D_embed == 512
        assert cfg.B_blocks == 12
        assert cfg.C == 16
        assert cfg.R == 6

    def test_tier_c(self):
        cfg = ModelConfig.tier_c()
        cfg.validate()
        assert cfg.D == 4096
        assert cfg.D_embed == 768
        assert cfg.B_blocks == 16
        assert cfg.C == 16
        assert cfg.R == 8

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
