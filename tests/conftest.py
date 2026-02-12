"""Shared fixtures for the neuromorphic LM test suite.

Provides a tiny model config (~1000x cheaper than tier_a) that preserves
all architectural constraints while running in milliseconds on CPU.
"""

import sys
import os
import pytest
import torch

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.config import ModelConfig


# ---------------------------------------------------------------------------
# Tiny defaults — all architectural constraints preserved, ~1000x cheaper
# ---------------------------------------------------------------------------
TINY_DEFAULTS = dict(
    D=32, L=2, B=2, vocab_size=64,  # D_h = 32/2 = 16
    # WM (must satisfy D_wm % n_heads_wm == 0)
    W=16, D_wm=16, n_heads_wm=2,     # head_dim = 8
    # PM
    r=4,
    # EM
    M=8, D_em=16, k_ret=4, C_em=4,
    # Decoder (must satisfy d_dec % n_heads_decoder == 0, D_h >= n_heads_decoder)
    d_dec=16, n_heads_decoder=2, decoder_layers=1, columnar_layers=1,
    thalamic_layers=1, thalamic_tokens=2,
    # Training
    T=16, P=4,
    # Neuromodulator
    neuromod_hidden=8,
    content_proj_dim=4,
    # FFN
    ffn_expansion=2,  # d_ff = D_h * 2 = 32
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


def forward_n_tokens(model, n, BS=2, vocab=64, with_commits=False):
    """Run n tokens through model, updating surprise after each.

    When with_commits=True, calls commit_at_boundary(force_mode="force_on")
    every P tokens. This is needed for gradient tests where PM slots
    must be populated.

    Returns (last_logits, last_target).
    """
    P = model.config.P
    reset_mask = torch.zeros(BS, dtype=torch.bool)
    last_logits = None
    last_target = None

    for t in range(n):
        input_id = torch.randint(0, vocab, (BS,))
        target = torch.randint(0, vocab, (BS,))

        logits, x_emb, y_wm = model.forward_one_token(input_id, reset_mask)
        model.update_surprise(logits, target)

        last_logits = logits
        last_target = target

        if with_commits and (t + 1) % P == 0:
            model.commit_at_boundary(force_mode="force_on")

    return last_logits, last_target


def forward_and_write_em(model, n, BS=2, vocab=64):
    """Run n tokens, propose EM candidates, call write_at_boundary.

    Populates EM state for gradient tests. Returns (last_logits, last_target).
    """
    P = model.config.P
    reset_mask = torch.zeros(BS, dtype=torch.bool)

    # Buffers for EM candidates per block
    cand_buffers = {}
    for b_idx, block in enumerate(model.blocks):
        cand_buffers[b_idx] = {"K": [], "V": [], "scores": []}

    last_logits = None
    last_target = None

    for t in range(n):
        input_id = torch.randint(0, vocab, (BS,))
        target = torch.randint(0, vocab, (BS,))

        logits, x_emb, y_wm = model.forward_one_token(input_id, reset_mask)
        model.update_surprise(logits, target)

        last_logits = logits
        last_target = target

        # Propose EM candidates
        for b_idx, block in enumerate(model.blocks):
            h_block = block.layers[-1].h
            if h_block is not None:
                k_c, v_c, novelty = block.em.propose_candidate(
                    x_emb, y_wm, h_block, model.surprise
                )
                cand_buffers[b_idx]["K"].append(k_c)
                cand_buffers[b_idx]["V"].append(v_c)
                cand_buffers[b_idx]["scores"].append(novelty)

        # Commit PM + write EM every P tokens
        if (t + 1) % P == 0:
            if model.config.pm_enabled:
                model.commit_at_boundary(force_mode="force_on")

            # EM write
            for b_idx, block in enumerate(model.blocks):
                buf = cand_buffers[b_idx]
                if buf["K"]:
                    cand_K = torch.stack(buf["K"], dim=1)     # [BS, P, D_em]
                    cand_V = torch.stack(buf["V"], dim=1)     # [BS, P, D_em]
                    cand_score = torch.stack(buf["scores"], dim=1)  # [BS, P]

                    # Neuromodulator decision
                    em_usage = block.em.em_S.sum(dim=-1) if block.em.em_S is not None else torch.zeros(BS)
                    span_surprise = model.surprise.squeeze(-1) if model.surprise is not None else torch.zeros(BS)
                    cand_novelty_mean = cand_score.mean(dim=1)

                    g_em, tau_em, ww_em, decay_em = block.em_neuromodulator.forward(
                        span_surprise, em_usage, cand_novelty_mean
                    )
                    # Force full write strength for test purposes
                    g_em = torch.ones(BS) * 0.5
                    block.em.write_at_boundary(cand_K, cand_V, cand_score,
                                                g_em, tau_em, ww_em,
                                                decay=decay_em)

                    # Reset buffers
                    cand_buffers[b_idx] = {"K": [], "V": [], "scores": []}

    return last_logits, last_target


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
