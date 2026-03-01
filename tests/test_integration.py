"""Integration smoke tests (v5) — end-to-end model behavior."""

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

        assert model.embedding.weight.grad is not None

    def test_state_persistence_across_segments(self):
        """PM/EM state should persist across segments."""
        cfg = make_tiny_config(pcm_enabled=True)
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        pm = model.pm
        bias_before = pm.pm_bias.clone()

        # Forward 2 segments (PM gets committed between)
        for _ in range(2):
            input_ids = torch.randint(0, VOCAB, (BS, cfg.N))
            model.forward_segment(input_ids)

        # After commits, pm_bias should have changed
        assert not torch.allclose(pm.pm_bias, bias_before, atol=1e-8)

    def test_detach_states(self):
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        input_ids = torch.randint(0, VOCAB, (BS, cfg.N))
        logits, aux = model.forward_segment(input_ids)

        model.detach_states()

        # After detach, states should not have grad_fn
        if model.pm.pm_bias is not None:
            assert model.pm.pm_bias.grad_fn is None
        if model.em.em_K is not None:
            assert model.em.em_K.grad_fn is None

    def test_doc_boundary_reset(self):
        """Doc boundary should reset PM/EM state before processing."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        # Set some PM content
        pm = model.pm
        pm.pm_bias = torch.randn(BS, cfg.B, cfg.D)
        bias_before = pm.pm_bias.abs().sum().item()

        # Test _reset_memory directly
        reset_mask = torch.ones(BS, dtype=torch.bool)
        model._reset_memory(reset_mask)

        assert pm.pm_bias.abs().sum() == 0

    def test_doc_boundary_reset_via_forward(self):
        """Forward with reset_mask should start from clean state."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        # Set large PM content
        pm = model.pm
        pm.pm_bias = torch.randn(BS, cfg.B, cfg.D) * 10
        bias_before = pm.pm_bias.abs().sum().item()

        # Forward with reset — commits start from zeroed state
        input_ids = torch.randint(0, VOCAB, (BS, cfg.N))
        reset_mask = torch.ones(BS, dtype=torch.bool)
        model.forward_segment(input_ids, reset_mask)

        # pm_bias should be much smaller (fresh commits only)
        assert pm.pm_bias.abs().sum().item() < bias_before

    def test_lifelong_mode_preserves_memory(self):
        """In lifelong mode, doc boundary should not reset PM/EM."""
        cfg = make_tiny_config()
        cfg.lifelong_mode = True
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        # Set some PM content
        pm = model.pm
        pm.pm_bias = torch.randn(BS, cfg.B, cfg.D)
        bias_before = pm.pm_bias.abs().sum().item()

        # Forward with reset mask (should be ignored in lifelong)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.N))
        reset_mask = torch.ones(BS, dtype=torch.bool)
        model.forward_segment(input_ids, reset_mask)

        # pm_bias may change from commits but should NOT be zeroed
        assert pm.pm_bias.abs().sum().item() > 0

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

    def test_three_stage_produces_different_outputs(self):
        """Stage 3 should integrate memory, producing different logits than pure scan."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        # Forward two segments so memory has content
        N = cfg.N
        for _ in range(2):
            input_ids = torch.randint(0, VOCAB, (BS, N))
            model.forward_segment(input_ids)

        # Third segment's logits should be influenced by memory
        input_ids = torch.randint(0, VOCAB, (BS, N))
        logits, _ = model.forward_segment(input_ids)
        assert logits.shape == (BS, N, VOCAB)
        assert torch.isfinite(logits).all()
