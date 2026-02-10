"""Design-specific decoder tests — update design_constants.py when design changes."""

import pytest
import torch

from src.model.model import NeuromorphicLM
from src.model.config import ModelConfig
from tests.conftest import make_tiny_config
from tests.design_constants import DEFAULTS, DECODER_OUTPUT_PROJ_INIT_STD

pytestmark = pytest.mark.design


# ============================================================================
# Decoder structure
# ============================================================================

class TestDecoderStructure:
    def test_three_levels_exist(self):
        """Spatial decoder has columnar, thalamic, decoder_blocks."""
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        dec = model.spatial_decoder
        assert dec is not None
        assert hasattr(dec, "columnar")
        assert hasattr(dec, "thalamic")
        assert hasattr(dec, "decoder_blocks")

    def test_columnar_count_equals_B(self):
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        assert len(model.spatial_decoder.columnar) == cfg.B

    def test_thalamic_tokens(self):
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        assert model.spatial_decoder.thalamic.K == cfg.thalamic_tokens

    def test_output_proj_location(self):
        """output_proj is at model.spatial_decoder.output_proj."""
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        assert hasattr(model.spatial_decoder, "output_proj")

    def test_output_proj_init_std(self):
        """output_proj initialized with small std (~0.01)."""
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        w = model.spatial_decoder.output_proj.weight
        # The std should be approximately DECODER_OUTPUT_PROJ_INIT_STD
        actual_std = w.std().item()
        assert abs(actual_std - DECODER_OUTPUT_PROJ_INIT_STD) < 0.01, \
            f"output_proj std {actual_std} not close to {DECODER_OUTPUT_PROJ_INIT_STD}"

    def test_decoder_not_instantiated_when_disabled(self):
        cfg = make_tiny_config(snapshot_enabled=False)
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        assert model.spatial_decoder is None

    def test_default_decoder_layers(self):
        cfg = ModelConfig()
        assert cfg.decoder_layers == DEFAULTS["decoder_layers"]
