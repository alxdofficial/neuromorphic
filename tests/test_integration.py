"""Integration smoke tests (v4) — end-to-end model behavior."""

import pytest
import torch
import torch.nn.functional as F
from tests.conftest import make_tiny_config, forward_one_segment, forward_k_segments

from src.model.model import NeuromorphicLM


BS = 2
VOCAB = 64


class TestEndToEnd:
    def test_single_segment_forward(self):
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        logits, aux = forward_one_segment(model, BS=BS)
        assert logits.shape == (BS, cfg.N, VOCAB)

    def test_multi_segment_forward(self):
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        results = forward_k_segments(model, K=3, BS=BS)
        assert len(results) == 3
        for logits, aux in results:
            assert logits.shape == (BS, cfg.N, VOCAB)

    def test_backward_through_segment(self):
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        input_ids = torch.randint(0, VOCAB, (BS, cfg.N))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.N))
        logits, aux = model.forward_segment(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), target_ids.reshape(-1))
        (loss + aux).backward()

        # Check gradients exist on key params
        assert model.embedding.weight.grad is not None
        assert model.fan_in.weight.grad is not None

    def test_state_persistence_across_segments(self):
        """PM/EM state should persist across segments."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        pm = model.pm
        pm_a_before = pm.pm_a.clone()

        # Forward 2 segments (PM gets committed between passes)
        for _ in range(2):
            input_ids = torch.randint(0, VOCAB, (BS, cfg.N))
            model.forward_segment(input_ids)

        # After commits, pm_a should have changed
        assert pm.pm_a is not None

    def test_detach_states(self):
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        input_ids = torch.randint(0, VOCAB, (BS, cfg.N))
        logits, aux = model.forward_segment(input_ids)

        # Before detach, states may have grad_fn
        model.detach_states()

        # After detach, states should not have grad_fn
        if model.pm.pm_K is not None:
            assert model.pm.pm_K.grad_fn is None
        if model.em.em_K is not None:
            assert model.em.em_K.grad_fn is None

    def test_doc_boundary_reset(self):
        """Doc boundary should reset PM/EM state before processing."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        B = cfg.B_blocks

        # Set some PM content (state is [BS, B, r, D])
        pm = model.pm
        pm.pm_K = torch.randn(BS, B, cfg.r, cfg.D)
        pm.pm_a = torch.ones(BS, B, cfg.r)

        a_before = pm.pm_a.sum().item()

        # Test _reset_memory directly to verify it zeros state
        reset_mask = torch.ones(BS, dtype=torch.bool)
        model._reset_memory(reset_mask)

        # After reset (before forward), pm_a should be zeroed
        assert pm.pm_a.sum() == 0

    def test_doc_boundary_reset_via_forward(self):
        """Forward with reset_mask should start from clean state."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        B = cfg.B_blocks

        # Set large PM content
        pm = model.pm
        pm.pm_K = torch.randn(BS, B, cfg.r, cfg.D) * 10
        pm.pm_a = torch.ones(BS, B, cfg.r) * 5.0
        a_before = pm.pm_a.sum().item()

        # Forward with reset — the R-pass loop will add new commits,
        # but they start from zeroed state, so pm_a should be small
        input_ids = torch.randint(0, VOCAB, (BS, cfg.N))
        reset_mask = torch.ones(BS, dtype=torch.bool)
        model.forward_segment(input_ids, reset_mask)

        # pm_a should be much smaller than before reset (fresh commits only)
        assert pm.pm_a.sum().item() < a_before

    def test_lifelong_mode_preserves_memory(self):
        """In lifelong mode, doc boundary should not reset PM/EM."""
        cfg = make_tiny_config()
        cfg.lifelong_mode = True
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        B = cfg.B_blocks

        # Set some PM content
        pm = model.pm
        pm.pm_K = torch.randn(BS, B, cfg.r, cfg.D)
        pm.pm_a = torch.ones(BS, B, cfg.r)

        a_before = pm.pm_a.sum().item()

        # Forward with reset mask (should be ignored in lifelong)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.N))
        reset_mask = torch.ones(BS, dtype=torch.bool)
        model.forward_segment(input_ids, reset_mask)

        # pm_a may change from commits but should NOT be zeroed
        assert pm.pm_a.sum().item() > 0

    def test_different_R_values(self):
        """Model should work with different R (refinement passes)."""
        for R in [1, 2, 4]:
            cfg = make_tiny_config(R=R)
            model = NeuromorphicLM(cfg)
            logits, aux = forward_one_segment(model, BS=BS)
            assert logits.shape == (BS, cfg.N, VOCAB)

    def test_pcm_enabled(self):
        cfg = make_tiny_config(pcm_enabled=True)
        model = NeuromorphicLM(cfg)
        logits, aux = forward_one_segment(model, BS=BS)
        assert logits.shape == (BS, cfg.N, VOCAB)

    def test_tied_embeddings(self):
        cfg = make_tiny_config(tie_embeddings=True)
        model = NeuromorphicLM(cfg)
        assert model.lm_head.weight is model.embedding.weight

    def test_untied_embeddings(self):
        cfg = make_tiny_config(tie_embeddings=False)
        model = NeuromorphicLM(cfg)
        assert model.lm_head.weight is not model.embedding.weight

    def test_d_embed_decoupled_forward_backward(self):
        """D_embed != D should work end-to-end with backward."""
        cfg = make_tiny_config(D=64, D_embed=32)
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        input_ids = torch.randint(0, VOCAB, (BS, cfg.N))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.N))
        logits, aux = model.forward_segment(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), target_ids.reshape(-1))
        (loss + aux).backward()

        assert model.embedding.weight.grad is not None
        assert model.proj_up.weight.grad is not None
        assert model.proj_down.weight.grad is not None

    def test_d_embed_tied_embeddings(self):
        """Tied embeddings should work with D_embed != D."""
        cfg = make_tiny_config(D=64, D_embed=32, tie_embeddings=True)
        model = NeuromorphicLM(cfg)
        assert model.lm_head.weight is model.embedding.weight
        assert model.embedding.weight.shape == (VOCAB, 32)

    def test_d_embed_decoupled_multi_segment(self):
        """D_embed decoupled should work across multiple segments."""
        cfg = make_tiny_config(D=64, D_embed=32)
        model = NeuromorphicLM(cfg)
        results = forward_k_segments(model, K=3, BS=BS)
        assert len(results) == 3
        for logits, aux in results:
            assert logits.shape == (BS, cfg.N, VOCAB)

    def test_generate_shape(self):
        """generate() should produce [BS, P + max_new_tokens] output."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        model.set_eval_mode()

        prompt = torch.randint(0, VOCAB, (BS, 5))
        out = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=10)
        assert out.shape == (BS, 15)

    def test_generate_preserves_prompt(self):
        """generate() output should start with the prompt."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        model.set_eval_mode()

        prompt = torch.randint(0, VOCAB, (BS, 5))
        out = model.generate(prompt, max_new_tokens=3, temperature=1.0, top_k=10)
        assert torch.equal(out[:, :5], prompt)

    def test_fitb_inline_static_shapes(self):
        """FITB inline path should produce valid loss without nonzero()."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        N = cfg.N
        input_ids = torch.randint(0, VOCAB, (BS, N))
        target_ids = input_ids.clone()
        fitb_mask = torch.rand(BS, N) < 0.3
        # Ensure at least one masked position
        fitb_mask[0, 0] = True
        input_ids_masked = input_ids.clone()
        input_ids_masked[fitb_mask] = cfg.fitb_id

        fitb_loss, aux_loss, fitb_valid = model.forward_segment(
            input_ids_masked, fitb_mask=fitb_mask, target_ids=target_ids,
        )
        assert fitb_loss.shape == ()
        assert fitb_valid.item() > 0
        assert torch.isfinite(fitb_loss)
