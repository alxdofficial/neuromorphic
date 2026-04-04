"""Integration tests for v9-backprop model."""

import torch
import pytest
import sys
sys.path.insert(0, ".")

from src.v8.config import V8Config
from src.v8.model import V8Model

BS = 2
VOCAB = 64


def make_tiny(**overrides):
    cfg = V8Config.tier_tiny(**overrides)
    cfg.validate()
    return cfg


class TestV8ModelForward:
    def test_forward_chunk_shapes(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.initialize_states(BS)

        result = model.forward_chunk(input_ids, target_ids=target_ids)
        assert result["logits"].shape == (BS, cfg.T, VOCAB)
        assert torch.isfinite(result["logits"]).all()

    def test_forward_no_memory(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.initialize_states(BS)

        result = model.forward_chunk(input_ids, target_ids=target_ids,
                                     use_memory=False)
        assert result["logits"].shape == (BS, cfg.T, VOCAB)

    def test_forward_with_reset(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        reset_mask = torch.tensor([True, False])
        model.initialize_states(BS)

        result = model.forward_chunk(
            input_ids, target_ids=target_ids, reset_mask=reset_mask,
            has_reset=True)
        assert torch.isfinite(result["logits"]).all()

    def test_multiple_chunks(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS)

        for _ in range(3):
            input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            result = model.forward_chunk(input_ids, target_ids=target_ids)
            model.detach_states()
            assert torch.isfinite(result["logits"]).all()


class TestGradientFlow:
    def test_loss_backward_runs(self):
        cfg = make_tiny()
        model = V8Model(cfg).float()
        model.initialize_states(BS)

        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(input_ids, target_ids=target_ids)

        import torch.nn.functional as F
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1))
        loss.backward()

    def test_memory_params_get_grad(self):
        cfg = make_tiny()
        model = V8Model(cfg).float()
        model.initialize_states(BS)

        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(input_ids, target_ids=target_ids)

        import torch.nn.functional as F
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1))
        loss.backward()

        mg = model.memory
        # All memory params should get gradients
        for name, p in mg.named_parameters():
            assert p.grad is not None, f"{name} should have grad"
            assert p.grad.norm() > 0, f"{name} grad should be nonzero"

    def test_lm_params_get_grad(self):
        cfg = make_tiny()
        model = V8Model(cfg).float()
        model.initialize_states(BS)

        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(input_ids, target_ids=target_ids)

        import torch.nn.functional as F
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1))
        loss.backward()

        assert model.lm.mem_scale.grad is not None
        for i, layer in enumerate(model.lm.layers):
            has_grad = any(p.grad is not None and p.grad.norm() > 0
                          for p in layer.parameters())
            assert has_grad, f"Layer {i} should have grad"

    def test_split_mlp_gets_grad(self):
        cfg = make_tiny()
        model = V8Model(cfg).float()
        model.initialize_states(BS)

        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(input_ids, target_ids=target_ids)

        import torch.nn.functional as F
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1))
        loss.backward()

        if model.lm.split_mlp is not None:
            for p in model.lm.split_mlp.parameters():
                if p.requires_grad:
                    assert p.grad is not None, "split_mlp param should have grad"

    def test_mem_scale_gets_grad(self):
        """mem_scale should get gradients from memory path."""
        cfg = make_tiny()
        model = V8Model(cfg).float()
        model.initialize_states(BS)

        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(input_ids, target_ids=target_ids)

        import torch.nn.functional as F
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1))
        loss.backward()

        assert model.lm.mem_scale.grad is not None, "mem_scale should have grad"
        assert model.lm.mem_scale.grad.norm() > 0, "mem_scale grad should be nonzero"


class TestDocumentBoundaryReset:
    """Verify that EOS tokens within a chunk reset recurrent state."""

    def test_internal_eos_resets_scan_carry(self):
        """Logits after an internal EOS should be independent of tokens before it."""
        cfg = make_tiny()
        model = V8Model(cfg).float()
        # Put in inference mode to disable dropout for deterministic comparison
        model.train(False)
        model.initialize_states(BS)

        T = cfg.T
        eot = cfg.eot_id

        # Build two chunks that differ only BEFORE an internal EOS at position T//2.
        # Tokens after the EOS are identical.  If reset works, logits after
        # the EOS should be the same.
        mid = T // 2
        post_eos_tokens = torch.randint(3, VOCAB, (BS, T - mid - 1))

        ids_a = torch.randint(3, VOCAB, (BS, T))
        ids_a[:, mid] = eot
        ids_a[:, mid + 1:] = post_eos_tokens

        ids_b = torch.randint(3, VOCAB, (BS, T))
        ids_b[:, mid] = eot
        ids_b[:, mid + 1:] = post_eos_tokens

        # Ensure the pre-EOS tokens actually differ
        ids_a[:, :mid] = torch.randint(3, VOCAB, (BS, mid))
        ids_b[:, :mid] = torch.randint(3, VOCAB, (BS, mid))

        # Process chunk A (fresh model state)
        model.initialize_states(BS)
        model.lm.initialize_carries()
        with torch.no_grad():
            res_a = model.forward_chunk(ids_a, use_memory=False)

        # Process chunk B (fresh model state)
        model.initialize_states(BS)
        model.lm.initialize_carries()
        with torch.no_grad():
            res_b = model.forward_chunk(ids_b, use_memory=False)

        # Logits at position mid+1 (first token after EOS boundary) should
        # be identical because the scan state was reset at mid+1.
        logits_a_post = res_a["logits"][:, mid + 1]
        logits_b_post = res_b["logits"][:, mid + 1]
        diff = (logits_a_post - logits_b_post).abs().max().item()
        assert diff < 1e-2, (
            f"Logits diverge after internal EOS (max diff={diff:.6f}). "
            f"Scan state was not properly reset at document boundary.")

    def test_internal_eos_with_memory(self):
        """forward_chunk with memory should not crash on internal EOS."""
        cfg = make_tiny()
        model = V8Model(cfg).float()
        model.initialize_states(BS)

        T = cfg.T
        input_ids = torch.randint(0, VOCAB, (BS, T))
        input_ids[:, T // 2] = cfg.eot_id  # internal EOS
        target_ids = torch.randint(0, VOCAB, (BS, T))

        result = model.forward_chunk(input_ids, target_ids=target_ids,
                                     use_memory=True)
        assert result["logits"].shape == (BS, T, VOCAB)
        assert torch.isfinite(result["logits"]).all()

    def test_no_eos_no_reset(self):
        """When there are no EOS tokens, behavior is unchanged."""
        cfg = make_tiny()
        model = V8Model(cfg).float()
        model.initialize_states(BS)

        T = cfg.T
        # Ensure no EOS tokens in input
        input_ids = torch.randint(3, VOCAB, (BS, T))  # skip 0,1,2 (eot_id=2)

        with torch.no_grad():
            result = model.forward_chunk(input_ids, use_memory=False)
        assert torch.isfinite(result["logits"]).all()

    def test_backward_with_internal_eos(self):
        """Backward pass should work with internal EOS resets."""
        cfg = make_tiny()
        model = V8Model(cfg).float()
        model.initialize_states(BS)

        T = cfg.T
        input_ids = torch.randint(0, VOCAB, (BS, T))
        input_ids[:, T // 4] = cfg.eot_id
        input_ids[:, 3 * T // 4] = cfg.eot_id
        target_ids = torch.randint(0, VOCAB, (BS, T))

        result = model.forward_chunk(input_ids, target_ids=target_ids)

        import torch.nn.functional as F
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1))
        loss.backward()

        # Verify gradients still flow
        assert model.lm.embedding.weight.grad is not None


class TestParamCounts:
    def test_param_count(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        total = model.param_count()
        lm = model.lm_param_count()
        mem = model.memory_param_count()
        assert total > 0
        assert lm > 0
        assert mem > 0
        # Memory params should include modulator + state_mlp + msg_mlp + neuron_id + dendrites
        assert mem > cfg.N_neurons * cfg.neuromod_hidden
