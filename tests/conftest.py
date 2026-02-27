"""Shared fixtures for the neuromorphic LM test suite (v4).

Provides a tiny model config (~1000x cheaper than tier_a) that preserves
all architectural constraints while running in milliseconds on CPU.
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.config import ModelConfig


# ---------------------------------------------------------------------------
# Tiny defaults — all architectural constraints preserved, ~1000x cheaper
# ---------------------------------------------------------------------------
TINY_DEFAULTS = dict(
    D=64, B_blocks=2, C=2, D_col=16, D_mem=32, D_pcm=16,
    vocab_size=64, R=2, N=16, K_segments=2,
    r=4, M=8, ffn_expansion=2,
    k_ret=2, C_em=4,
    neuromod_hidden=8, content_proj_dim=4,
    # PCM off by default in tests (dedicated PCM tests enable it explicitly)
    pcm_enabled=False, pcm_pred_weight=0.01,
)


def make_tiny_config(**overrides):
    """Factory for tiny ModelConfig with optional overrides."""
    kw = {**TINY_DEFAULTS, **overrides}
    cfg = ModelConfig(**kw)
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def forward_one_segment(model, BS=2, vocab=64, N=None):
    """Forward one segment through the model.

    Returns (logits [BS, N, vocab], aux_loss scalar).
    """
    if N is None:
        N = model.config.N
    input_ids = torch.randint(0, vocab, (BS, N))
    reset_mask = torch.zeros(BS, dtype=torch.bool)

    model.initialize_states(BS, torch.device("cpu"))
    logits, aux_loss = model.forward_segment(input_ids, reset_mask)
    return logits, aux_loss


def forward_k_segments(model, K=None, BS=2, vocab=64):
    """Forward K segments (full TBPTT chunk).

    Returns list of (logits, aux_loss) per segment.
    """
    if K is None:
        K = model.config.K_segments
    N = model.config.N
    results = []

    model.initialize_states(BS, torch.device("cpu"))

    for seg in range(K):
        input_ids = torch.randint(0, vocab, (BS, N))
        reset_mask = torch.zeros(BS, dtype=torch.bool)
        logits, aux_loss = model.forward_segment(input_ids, reset_mask)
        results.append((logits, aux_loss))

    return results


# ---------------------------------------------------------------------------
# Slow test marker handling
# ---------------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="Run tests marked as slow",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
