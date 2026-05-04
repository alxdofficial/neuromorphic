"""Smoke + invariants for `src.data.passphrase_loader`.

Runs CPU-only against the committed stub expansion and FineWeb-edu
parquet. Skips when either is missing (so this test stays green on a
fresh clone before user data is built).
"""

from __future__ import annotations

import os

import pytest
import torch

EXPANDED_STUB = "data/passphrase/expanded_stub.json"
FINEWEB = "data/phase_B/fineweb_edu.parquet"


def _has_data() -> bool:
    return os.path.exists(EXPANDED_STUB) and os.path.exists(FINEWEB)


@pytest.mark.skipif(not _has_data(), reason="stub data + FineWeb-edu parquet required")
def test_passphrase_loader_yields_correct_shape():
    from transformers import AutoTokenizer
    from src.data.passphrase_loader import passphrase_phase1ar_iter

    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    bs, T_pre, T_cont = 2, 256, 32
    it = passphrase_phase1ar_iter(
        expanded_path=EXPANDED_STUB,
        tokenizer=tok,
        filler_parquet=FINEWEB,
        bs=bs, T_pre=T_pre, T_cont=T_cont,
        filler_mid_schedule=[(0, 50)],
        n_heldout=20, device="cpu", seed=0, max_batches=2,
        filler_pool_size=100_000,
    )
    n_seen = 0
    for b in it:
        assert b.prefix_ids.shape == (bs, T_pre)
        assert b.continuation_ids.shape == (bs, T_cont)
        assert b.prefix_ids.dtype == torch.int64
        # No negative ids; all within vocab range
        assert b.prefix_ids.min().item() >= 0
        # Use len(tok) — accounts for special tokens beyond the base vocab.
        assert b.prefix_ids.max().item() < len(tok)
        assert b.continuation_ids.min().item() >= 0
        assert b.continuation_ids.max().item() < len(tok)
        n_seen += 1
    assert n_seen == 2


@pytest.mark.skipif(not _has_data(), reason="stub data + FineWeb-edu parquet required")
def test_passphrase_loader_train_heldout_disjoint():
    """Train and held-out splits must not share fact ids."""
    from src.data.passphrase_loader import _load_facts, _split_train_heldout

    facts = _load_facts(EXPANDED_STUB)
    train, heldout = _split_train_heldout(facts, n_heldout=20, seed=42)
    train_ids = {f.id for f in train}
    heldout_ids = {f.id for f in heldout}
    assert len(train_ids & heldout_ids) == 0
    assert len(heldout_ids) == 20
    assert len(train_ids) + len(heldout_ids) == len(facts)


@pytest.mark.skipif(not _has_data(), reason="stub data + FineWeb-edu parquet required")
def test_passphrase_loader_curriculum_advances():
    """The filler_mid length picked should follow the schedule as steps advance."""
    from transformers import AutoTokenizer
    from src.data.passphrase_loader import passphrase_phase1ar_iter

    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    schedule = [(0, 30), (3, 100)]
    it = passphrase_phase1ar_iter(
        expanded_path=EXPANDED_STUB,
        tokenizer=tok,
        filler_parquet=FINEWEB,
        bs=2, T_pre=512, T_cont=32,
        filler_mid_schedule=schedule,
        n_heldout=20, device="cpu", seed=0, max_batches=5,
        filler_pool_size=100_000,
    )
    # Just verify iteration completes — we don't expose internal step
    # counter for direct inspection, but the schedule logic should not
    # crash across the threshold.
    batches = list(it)
    assert len(batches) == 5
