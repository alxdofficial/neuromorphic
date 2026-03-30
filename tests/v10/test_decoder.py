"""Tests for v10 SlidingWindowDecoder."""

import torch
import pytest

from src.v10.decoder import (
    SlidingWindowDecoder,
    _build_causal_mask,
    _build_sliding_window_cross_mask,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def decoder_cfg():
    return dict(
        D_dec=64,
        D_scan=128,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        W_sliding=4,
        vocab_size=256,
        D_embed=64,
        dropout=0.0,
    )


@pytest.fixture
def small_decoder(decoder_cfg):
    return SlidingWindowDecoder(**decoder_cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDecoderOutputShape:

    def test_basic_shape(self, small_decoder, decoder_cfg):
        BS, T, num_words = 2, 8, 4
        D_scan = decoder_cfg["D_scan"]
        word_states = torch.randn(BS, T, num_words, D_scan)

        logits = small_decoder(word_states)

        assert logits.shape == (BS, T, decoder_cfg["vocab_size"])

    def test_single_token(self, small_decoder, decoder_cfg):
        """Decoder works with T=1."""
        BS, T, num_words = 2, 1, 4
        D_scan = decoder_cfg["D_scan"]
        word_states = torch.randn(BS, T, num_words, D_scan)

        logits = small_decoder(word_states)
        assert logits.shape == (BS, T, decoder_cfg["vocab_size"])

    def test_different_d_dec_d_embed(self):
        """proj_down is used when D_dec != D_embed."""
        dec = SlidingWindowDecoder(
            D_dec=128, D_scan=64, n_heads=4, n_layers=1, d_ff=128,
            W_sliding=4, vocab_size=256, D_embed=32, dropout=0.0,
        )
        assert dec.proj_down is not None

        word_states = torch.randn(2, 8, 4, 64)
        logits = dec(word_states)
        assert logits.shape == (2, 8, 256)

    def test_tied_embeddings(self, decoder_cfg):
        """lm_head weight is shared with provided embedding weight."""
        emb = torch.nn.Embedding(decoder_cfg["vocab_size"], decoder_cfg["D_embed"])
        dec = SlidingWindowDecoder(**decoder_cfg, tie_embeddings=emb.weight)
        assert dec.lm_head.weight is emb.weight


class TestCausalMask:

    def test_shape(self):
        T = 8
        mask = _build_causal_mask(T, torch.device("cpu"))
        assert mask.shape == (T, T)
        assert mask.dtype == torch.bool

    def test_lower_triangular(self):
        T = 6
        mask = _build_causal_mask(T, torch.device("cpu"))
        # Diagonal and below should be True
        for i in range(T):
            for j in range(T):
                if j <= i:
                    assert mask[i, j], f"mask[{i},{j}] should be True"
                else:
                    assert not mask[i, j], f"mask[{i},{j}] should be False"

    def test_no_future_leakage_in_self_attention(self, small_decoder, decoder_cfg):
        """Verify that changing a future token does not affect earlier logits."""
        BS, T, num_words = 1, 8, 4
        D_scan = decoder_cfg["D_scan"]

        word_states_a = torch.randn(BS, T, num_words, D_scan)
        word_states_b = word_states_a.clone()
        # Modify the last 2 timesteps in b
        word_states_b[:, -2:] = torch.randn(BS, 2, num_words, D_scan)

        small_decoder.eval()
        with torch.no_grad():
            logits_a = small_decoder(word_states_a)
            logits_b = small_decoder(word_states_b)

        # First T-W tokens should be identical since the modification is
        # outside their sliding window. With W=4, tokens 0..3 should be
        # unaffected by changes at positions 6,7.
        W = decoder_cfg["W_sliding"]
        safe_end = T - 2 - W + 1  # earliest token affected by change at T-2
        if safe_end > 0:
            assert torch.allclose(logits_a[:, :safe_end], logits_b[:, :safe_end],
                                  atol=1e-5), \
                "Tokens outside the sliding window of the modified positions should be unaffected"


class TestSlidingWindowMask:

    def test_shape(self):
        T, num_words, W = 8, 4, 3
        mask = _build_sliding_window_cross_mask(T, num_words, W, torch.device("cpu"))
        assert mask.shape == (T, T * num_words)
        assert mask.dtype == torch.bool

    def test_causality(self):
        """Token t must not attend to word_states from steps > t."""
        T, num_words, W = 8, 4, 16  # W >= T so only causality matters
        mask = _build_sliding_window_cross_mask(T, num_words, W, torch.device("cpu"))

        for t in range(T):
            # KV positions from step t+1 onward must be masked
            future_start = (t + 1) * num_words
            if future_start < T * num_words:
                assert not mask[t, future_start:].any(), \
                    f"Token {t} can see future word_states"

    def test_window_width(self):
        """Token t should see exactly min(t+1, W) steps."""
        T, num_words, W = 10, 3, 4
        mask = _build_sliding_window_cross_mask(T, num_words, W, torch.device("cpu"))

        for t in range(T):
            expected_steps = min(t + 1, W)
            expected_kv = expected_steps * num_words
            actual_kv = mask[t].sum().item()
            assert actual_kv == expected_kv, \
                f"Token {t}: expected {expected_kv} visible KV, got {actual_kv}"

    def test_window_boundaries(self):
        """Check exact start/end of visible window for each query position."""
        T, num_words, W = 8, 2, 3
        mask = _build_sliding_window_cross_mask(T, num_words, W, torch.device("cpu"))

        for t in range(T):
            s_start = max(0, t - W + 1)
            kv_start = s_start * num_words
            kv_end = (t + 1) * num_words

            # Everything in [kv_start, kv_end) should be True
            assert mask[t, kv_start:kv_end].all(), \
                f"Token {t}: window [{kv_start},{kv_end}) not fully visible"

            # Everything outside should be False
            if kv_start > 0:
                assert not mask[t, :kv_start].any(), \
                    f"Token {t}: pre-window positions visible"
            if kv_end < T * num_words:
                assert not mask[t, kv_end:].any(), \
                    f"Token {t}: post-window positions visible"

    def test_first_token(self):
        """Token 0 should only see step 0 (num_words entries)."""
        T, num_words, W = 8, 5, 4
        mask = _build_sliding_window_cross_mask(T, num_words, W, torch.device("cpu"))

        assert mask[0, :num_words].all()
        assert not mask[0, num_words:].any()

    def test_w_equals_one(self):
        """W=1: each token sees only its own step's word_states."""
        T, num_words, W = 6, 3, 1
        mask = _build_sliding_window_cross_mask(T, num_words, W, torch.device("cpu"))

        for t in range(T):
            kv_start = t * num_words
            kv_end = (t + 1) * num_words
            assert mask[t].sum().item() == num_words
            assert mask[t, kv_start:kv_end].all()
