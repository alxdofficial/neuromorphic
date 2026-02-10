"""Comprehensive equivalence tests for the parallel span forward pass.

Tests that forward_span() produces identical results to sequential step()
for all model components, across all training phases.
"""

import copy
import torch
import torch.nn.functional as F
import pytest

from tests.conftest import make_tiny_config
from src.model.scan import parallel_affine_scan
from src.model.layer import Layer
from src.model.procedural_memory import ProceduralMemory
from src.model.episodic_memory import EpisodicMemory
from src.model.working_memory import WorkingMemory
from src.model.block import Block
from src.model.model import NeuromorphicLM
from src.model.state import save_runtime_state, load_runtime_state


BS = 2
P = 4  # matches TINY_DEFAULTS


# ============================================================
# Helpers
# ============================================================

def _clone_state(model):
    """Deep-copy model runtime state for later comparison."""
    state = save_runtime_state(model)
    return copy.deepcopy({
        k: {n: v.clone() if isinstance(v, torch.Tensor) else v
            for n, v in sub.items()}
        for k, sub in state.items()
    })


def _assert_tensors_close(a, b, name, atol=1e-5):
    """Assert two tensors are close, with informative error."""
    if a is None and b is None:
        return
    assert a is not None and b is not None, f"{name}: one is None"
    assert a.shape == b.shape, f"{name}: shape mismatch {a.shape} vs {b.shape}"
    diff = (a - b).abs().max().item()
    assert diff < atol, f"{name}: max diff {diff} > {atol}"


# ============================================================
# 1. Parallel scan utility tests
# ============================================================

class TestParallelAffineScan:
    def test_matches_manual_loop(self):
        """Scan output matches a manual for-loop."""
        torch.manual_seed(42)
        a = torch.rand(BS, P, 16)
        b = torch.rand(BS, P, 16)
        h_init = torch.rand(BS, 16)

        # Manual loop
        h = h_init
        expected = []
        for t in range(P):
            h = a[:, t] * h + b[:, t]
            expected.append(h)
        expected = torch.stack(expected, dim=1)

        result = parallel_affine_scan(a, b, h_init)
        _assert_tensors_close(result, expected, "scan_output", atol=1e-6)

    def test_gradient_flow(self):
        """Backward through scan produces finite gradients."""
        torch.manual_seed(42)
        a = torch.rand(BS, P, 16, requires_grad=True)
        b = torch.rand(BS, P, 16, requires_grad=True)
        h_init = torch.rand(BS, 16, requires_grad=True)

        h_all = parallel_affine_scan(a, b, h_init)
        loss = h_all.sum()
        loss.backward()

        assert a.grad is not None and torch.isfinite(a.grad).all()
        assert b.grad is not None and torch.isfinite(b.grad).all()
        assert h_init.grad is not None and torch.isfinite(h_init.grad).all()

    def test_p_equals_1(self):
        """Edge case: single timestep."""
        a = torch.rand(BS, 1, 16)
        b = torch.rand(BS, 1, 16)
        h_init = torch.rand(BS, 16)

        result = parallel_affine_scan(a, b, h_init)
        expected = a[:, 0] * h_init + b[:, 0]
        _assert_tensors_close(result[:, 0], expected, "p1_output", atol=1e-6)

    def test_all_zero_a_is_reset(self):
        """When a=0, h_t = b_t (ignores history)."""
        a = torch.zeros(BS, P, 16)
        b = torch.rand(BS, P, 16)
        h_init = torch.rand(BS, 16) * 100  # large init, should be ignored

        result = parallel_affine_scan(a, b, h_init)
        for t in range(P):
            _assert_tensors_close(result[:, t], b[:, t], f"reset_t{t}", atol=1e-6)

    def test_all_one_a_is_accumulation(self):
        """When a=1, h_t = h_{t-1} + b_t (pure accumulation)."""
        a = torch.ones(BS, P, 16)
        b = torch.rand(BS, P, 16)
        h_init = torch.zeros(BS, 16)

        result = parallel_affine_scan(a, b, h_init)
        expected_last = b.sum(dim=1)  # cumulative sum
        _assert_tensors_close(result[:, -1], expected_last, "accum_last", atol=1e-5)

    def test_carry_integration(self):
        """a_eff = a * carry produces correct reset behavior."""
        torch.manual_seed(42)
        a = torch.rand(BS, P, 16)
        b = torch.rand(BS, P, 16)
        h_init = torch.rand(BS, 16)

        # carry=0 at position 2 for all batches
        carry = torch.ones(BS, P, 1)
        carry[:, 2, :] = 0.0
        a_eff = a * carry

        result = parallel_affine_scan(a_eff, b, h_init)

        # At position 2, a_eff=0 so h = b (reset)
        _assert_tensors_close(result[:, 2], b[:, 2], "carry_reset", atol=1e-6)


# ============================================================
# 2. Layer forward_span equivalence
# ============================================================

class TestLayerForwardSpan:
    @pytest.fixture
    def layer_and_inputs(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        layer = Layer(cfg, block_idx=0, layer_idx=0)
        D_h = cfg.D_h
        torch.manual_seed(42)
        x = torch.randn(BS, P, D_h)
        y_pm = torch.randn(BS, P, D_h)
        y_wm_proj = torch.randn(BS, P, D_h)
        y_em_proj = torch.randn(BS, P, D_h)
        surprise = torch.rand(BS, 1)
        carry = torch.ones(BS, P, 1)
        return layer, x, y_pm, y_wm_proj, y_em_proj, surprise, carry, cfg

    def test_outputs_match_sequential(self, layer_and_inputs):
        """forward_span outputs match P calls to step()."""
        layer, x, y_pm, y_wm_proj, y_em_proj, surprise, carry, cfg = layer_and_inputs

        # Sequential path
        layer_seq = copy.deepcopy(layer)
        layer_seq._lazy_init(BS, x.device)
        seq_outputs = []
        for t in range(P):
            out = layer_seq.step(x[:, t], y_pm[:, t], y_wm_proj[:, t],
                                 y_em_proj[:, t], surprise, carry[:, t])
            seq_outputs.append(out)
        seq_outputs = torch.stack(seq_outputs, dim=1)
        seq_h_final = layer_seq.h.clone()

        # Parallel path
        layer_par = copy.deepcopy(layer)
        layer_par._lazy_init(BS, x.device)
        par_outputs = layer_par.forward_span(x, y_pm, y_wm_proj, y_em_proj,
                                              surprise, carry)

        _assert_tensors_close(par_outputs, seq_outputs, "layer_output", atol=1e-5)
        _assert_tensors_close(layer_par.h, seq_h_final, "layer_h_final", atol=1e-5)

    def test_with_doc_boundary_reset(self, layer_and_inputs):
        """Carry=0 mid-span produces correct reset in both paths."""
        layer, x, y_pm, y_wm_proj, y_em_proj, surprise, _, cfg = layer_and_inputs
        carry = torch.ones(BS, P, 1)
        carry[:, 2, :] = 0.0  # reset at position 2

        layer_seq = copy.deepcopy(layer)
        layer_seq._lazy_init(BS, x.device)
        seq_outputs = []
        for t in range(P):
            out = layer_seq.step(x[:, t], y_pm[:, t], y_wm_proj[:, t],
                                 y_em_proj[:, t], surprise, carry[:, t])
            seq_outputs.append(out)
        seq_outputs = torch.stack(seq_outputs, dim=1)

        layer_par = copy.deepcopy(layer)
        layer_par._lazy_init(BS, x.device)
        par_outputs = layer_par.forward_span(x, y_pm, y_wm_proj, y_em_proj,
                                              surprise, carry)

        _assert_tensors_close(par_outputs, seq_outputs, "layer_reset_output", atol=1e-5)

    def test_gradients_match(self, layer_and_inputs):
        """Gradients through both paths match."""
        layer, x, y_pm, y_wm_proj, y_em_proj, surprise, carry, cfg = layer_and_inputs

        # Sequential gradient
        layer_seq = copy.deepcopy(layer)
        layer_seq._lazy_init(BS, x.device)
        seq_outs = []
        for t in range(P):
            out = layer_seq.step(x[:, t], y_pm[:, t], y_wm_proj[:, t],
                                 y_em_proj[:, t], surprise, carry[:, t])
            seq_outs.append(out)
        seq_loss = torch.stack(seq_outs, dim=1).sum()
        seq_loss.backward()
        seq_grads = {n: p.grad.clone() for n, p in layer_seq.named_parameters()
                     if p.grad is not None}

        # Parallel gradient
        layer_par = copy.deepcopy(layer)
        layer_par._lazy_init(BS, x.device)
        par_out = layer_par.forward_span(x, y_pm, y_wm_proj, y_em_proj,
                                          surprise, carry)
        par_loss = par_out.sum()
        par_loss.backward()

        for name, p in layer_par.named_parameters():
            if p.grad is not None and name in seq_grads:
                _assert_tensors_close(p.grad, seq_grads[name],
                                      f"layer_grad_{name}", atol=1e-4)


# ============================================================
# 3. PM apply_batch equivalence
# ============================================================

class TestPMApplyBatch:
    def test_matches_sequential(self):
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg)
        torch.manual_seed(42)
        D_h = cfg.D_h

        # Initialize PM state
        pm._lazy_init(BS, torch.device("cpu"))
        pm.pm_K = torch.randn(BS, cfg.r, D_h)
        pm.pm_V = torch.randn(BS, cfg.r, D_h)
        pm.pm_a = torch.rand(BS, cfg.r)

        x = torch.randn(BS, P, D_h)

        # Sequential
        seq_out = []
        for t in range(P):
            seq_out.append(pm.apply(x[:, t]))
        seq_out = torch.stack(seq_out, dim=1)

        # Batch
        batch_out = pm.apply_batch(x)

        _assert_tensors_close(batch_out, seq_out, "pm_apply_batch", atol=1e-5)


# ============================================================
# 4. EM retrieve_batch equivalence
# ============================================================

class TestEMRetrieveBatch:
    def test_matches_sequential(self):
        cfg = make_tiny_config()
        cfg.set_phase("C")
        em = EpisodicMemory(cfg)
        torch.manual_seed(42)

        # Initialize with some active slots
        em._lazy_init(BS, torch.device("cpu"))
        em.em_S[:, :4] = 1.0  # 4 active slots

        x = torch.randn(BS, P, cfg.D)
        y_wm = torch.randn(BS, P, cfg.D)

        # Sequential
        seq_out = []
        for t in range(P):
            seq_out.append(em.retrieve(x[:, t], y_wm[:, t]))
        seq_out = torch.stack(seq_out, dim=1)

        # Batch
        batch_out = em.retrieve_batch(x, y_wm)

        _assert_tensors_close(batch_out, seq_out, "em_retrieve_batch", atol=1e-5)


# ============================================================
# 4b. EM propose_candidate_batch equivalence
# ============================================================

class TestEMProposeCandidateBatch:
    def test_matches_sequential(self):
        cfg = make_tiny_config()
        cfg.set_phase("C")
        em = EpisodicMemory(cfg)
        torch.manual_seed(42)

        em._lazy_init(BS, torch.device("cpu"))
        em.em_S[:, :4] = 1.0

        x = torch.randn(BS, P, cfg.D)
        y_wm = torch.randn(BS, P, cfg.D)
        h_final = torch.randn(BS, P, cfg.D_h)
        surprise = torch.rand(BS, P, 1)

        # Sequential
        seq_k, seq_v, seq_nov = [], [], []
        for t in range(P):
            k, v, n = em.propose_candidate(x[:, t], y_wm[:, t], h_final[:, t],
                                            surprise[:, t])
            seq_k.append(k)
            seq_v.append(v)
            seq_nov.append(n)
        seq_k = torch.stack(seq_k, dim=1)
        seq_v = torch.stack(seq_v, dim=1)
        seq_nov = torch.stack(seq_nov, dim=1)

        # Batch
        batch_k, batch_v, batch_nov = em.propose_candidate_batch(
            x, y_wm, h_final, surprise
        )

        _assert_tensors_close(batch_k, seq_k, "em_cand_k", atol=1e-5)
        _assert_tensors_close(batch_v, seq_v, "em_cand_v", atol=1e-5)
        _assert_tensors_close(batch_nov, seq_nov, "em_cand_nov", atol=1e-5)


# ============================================================
# 5. WM forward_span equivalence
# ============================================================

class TestWMForwardSpan:
    def test_matches_sequential(self):
        cfg = make_tiny_config()
        wm = WorkingMemory(cfg)
        torch.manual_seed(42)

        x = torch.randn(BS, P, cfg.D)
        reset_mask = torch.zeros(BS, P, dtype=torch.bool)

        # Sequential
        wm_seq = copy.deepcopy(wm)
        wm_seq._lazy_init(BS, torch.device("cpu"))
        seq_out = []
        for t in range(P):
            seq_out.append(wm_seq.step(x[:, t], reset_mask[:, t]))
        seq_out = torch.stack(seq_out, dim=1)

        # Parallel
        wm_par = copy.deepcopy(wm)
        wm_par._lazy_init(BS, torch.device("cpu"))
        par_out = wm_par.forward_span(x, reset_mask)

        _assert_tensors_close(par_out, seq_out, "wm_output", atol=1e-5)

        # State consistency
        _assert_tensors_close(wm_par.wm_K, wm_seq.wm_K, "wm_K", atol=1e-5)
        _assert_tensors_close(wm_par.wm_V, wm_seq.wm_V, "wm_V", atol=1e-5)
        assert (wm_par.wm_valid == wm_seq.wm_valid).all(), "wm_valid mismatch"
        assert (wm_par.wm_ptr == wm_seq.wm_ptr).all(), "wm_ptr mismatch"

    def test_with_midspan_reset(self):
        cfg = make_tiny_config()
        wm = WorkingMemory(cfg)
        torch.manual_seed(42)

        x = torch.randn(BS, P, cfg.D)
        reset_mask = torch.zeros(BS, P, dtype=torch.bool)
        reset_mask[0, 2] = True  # stream 0 resets at position 2

        wm_seq = copy.deepcopy(wm)
        wm_seq._lazy_init(BS, torch.device("cpu"))
        seq_out = []
        for t in range(P):
            seq_out.append(wm_seq.step(x[:, t], reset_mask[:, t]))
        seq_out = torch.stack(seq_out, dim=1)

        wm_par = copy.deepcopy(wm)
        wm_par._lazy_init(BS, torch.device("cpu"))
        par_out = wm_par.forward_span(x, reset_mask)

        _assert_tensors_close(par_out, seq_out, "wm_reset_output", atol=1e-5)
        assert (wm_par.wm_valid == wm_seq.wm_valid).all()
        assert (wm_par.wm_ptr == wm_seq.wm_ptr).all()


# ============================================================
# 6. Block forward_span equivalence
# ============================================================

class TestBlockForwardSpan:
    @pytest.mark.parametrize("phase", ["A", "B", "C"])
    def test_matches_sequential(self, phase):
        cfg = make_tiny_config()
        cfg.set_phase(phase)
        block = Block(cfg, block_idx=0)
        torch.manual_seed(42)

        D_h = cfg.D_h
        x_block = torch.randn(BS, P, D_h)
        y_wm = torch.randn(BS, P, cfg.D)
        x_emb = torch.randn(BS, P, cfg.D)
        surprise = torch.rand(BS, 1)
        carry = torch.ones(BS, P, 1)

        # Sequential
        block_seq = copy.deepcopy(block)
        for layer in block_seq.layers:
            layer._lazy_init(BS, torch.device("cpu"))
        if cfg.em_enabled:
            block_seq.em._lazy_init(BS, torch.device("cpu"))
        seq_out = []
        for t in range(P):
            out = block_seq.step(x_block[:, t], y_wm[:, t], x_emb[:, t],
                                 surprise, carry[:, t, :1])
            seq_out.append(out)
        seq_out = torch.stack(seq_out, dim=1)

        # Parallel
        block_par = copy.deepcopy(block)
        for layer in block_par.layers:
            layer._lazy_init(BS, torch.device("cpu"))
        if cfg.em_enabled:
            block_par.em._lazy_init(BS, torch.device("cpu"))
        par_out = block_par.forward_span(x_block, y_wm, x_emb, surprise, carry)

        _assert_tensors_close(par_out, seq_out, f"block_{phase}_output", atol=1e-4)


# ============================================================
# 7. Full model forward_span equivalence
# ============================================================

class TestModelForwardSpan:
    @pytest.mark.parametrize("phase", ["A", "B", "C"])
    def test_logits_match_sequential(self, phase):
        """forward_span logits match P calls to forward_one_token (frozen surprise)."""
        cfg = make_tiny_config()
        cfg.set_phase(phase)
        torch.manual_seed(42)
        model = NeuromorphicLM(cfg)

        input_ids = torch.randint(0, cfg.vocab_size, (BS, P))
        reset_first = torch.zeros(BS, dtype=torch.bool)

        # Sequential path (with frozen surprise)
        model_seq = copy.deepcopy(model)
        model_seq.surprise = torch.zeros(BS, 1)
        frozen_surprise = model_seq.surprise.clone()
        seq_logits = []
        for t in range(P):
            if t == 0:
                rm = reset_first
            else:
                rm = (input_ids[:, t - 1] == cfg.eot_id)
            if not cfg.reset_on_doc_boundary:
                rm = torch.zeros_like(rm)

            logits, _, _ = model_seq.forward_one_token(input_ids[:, t], rm)
            seq_logits.append(logits)
            # Keep surprise frozen (don't update between tokens)
            model_seq.surprise = frozen_surprise.clone()
        seq_logits = torch.stack(seq_logits, dim=1)

        # Parallel path
        model_par = copy.deepcopy(model)
        model_par.surprise = torch.zeros(BS, 1)
        par_logits, _, _ = model_par.forward_span(input_ids, reset_first)

        _assert_tensors_close(par_logits, seq_logits, f"model_{phase}_logits",
                              atol=1e-4)

    def test_with_doc_boundary(self):
        """forward_span handles doc boundary reset mid-span."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        torch.manual_seed(42)
        model = NeuromorphicLM(cfg)

        input_ids = torch.randint(0, cfg.vocab_size, (BS, P))
        input_ids[0, 1] = cfg.eot_id  # force reset at position 2 for stream 0
        reset_first = torch.zeros(BS, dtype=torch.bool)

        model_seq = copy.deepcopy(model)
        model_seq.surprise = torch.zeros(BS, 1)
        frozen_surprise = model_seq.surprise.clone()
        seq_logits = []
        for t in range(P):
            if t == 0:
                rm = reset_first
            else:
                rm = (input_ids[:, t - 1] == cfg.eot_id)
            logits, _, _ = model_seq.forward_one_token(input_ids[:, t], rm)
            seq_logits.append(logits)
            model_seq.surprise = frozen_surprise.clone()
        seq_logits = torch.stack(seq_logits, dim=1)

        model_par = copy.deepcopy(model)
        model_par.surprise = torch.zeros(BS, 1)
        par_logits, _, _ = model_par.forward_span(input_ids, reset_first)

        _assert_tensors_close(par_logits, seq_logits, "model_docbound_logits",
                              atol=1e-4)

    def test_gradient_flow(self):
        """Backward through forward_span produces finite gradients."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        torch.manual_seed(42)
        model = NeuromorphicLM(cfg)
        model.surprise = torch.zeros(BS, 1)

        input_ids = torch.randint(0, cfg.vocab_size, (BS, P))
        reset_first = torch.zeros(BS, dtype=torch.bool)

        logits, _, _ = model.forward_span(input_ids, reset_first)
        loss = logits.sum()
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad: {name}"

    def test_state_consistency(self):
        """After forward_span, Layer.h and WM state match sequential."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        torch.manual_seed(42)
        model = NeuromorphicLM(cfg)

        input_ids = torch.randint(0, cfg.vocab_size, (BS, P))
        reset_first = torch.zeros(BS, dtype=torch.bool)

        # Sequential
        model_seq = copy.deepcopy(model)
        model_seq.surprise = torch.zeros(BS, 1)
        frozen_surprise = model_seq.surprise.clone()
        for t in range(P):
            if t == 0:
                rm = reset_first
            else:
                rm = (input_ids[:, t - 1] == cfg.eot_id)
            if not cfg.reset_on_doc_boundary:
                rm = torch.zeros_like(rm)
            model_seq.forward_one_token(input_ids[:, t], rm)
            model_seq.surprise = frozen_surprise.clone()

        # Parallel
        model_par = copy.deepcopy(model)
        model_par.surprise = torch.zeros(BS, 1)
        model_par.forward_span(input_ids, reset_first)

        # Check Layer.h
        for b in range(cfg.B):
            for l in range(cfg.L):
                h_seq = model_seq.blocks[b].layers[l].h
                h_par = model_par.blocks[b].layers[l].h
                _assert_tensors_close(h_par, h_seq,
                                      f"h_b{b}_l{l}", atol=1e-5)

        # Check WM state
        _assert_tensors_close(model_par.wm.wm_K, model_seq.wm.wm_K,
                              "wm_K", atol=1e-5)
        _assert_tensors_close(model_par.wm.wm_V, model_seq.wm.wm_V,
                              "wm_V", atol=1e-5)


# ============================================================
# 8. Batched cross-entropy test
# ============================================================

class TestBatchedCrossEntropy:
    def test_matches_online(self):
        from src.training.loss import online_cross_entropy, batched_cross_entropy

        torch.manual_seed(42)
        logits = torch.randn(BS, P, 64)
        targets = torch.randint(0, 64, (BS, P))
        mask = torch.ones(BS, P, dtype=torch.bool)
        mask[0, 1] = False

        # Online sum
        total_loss = 0.0
        total_count = 0
        for t in range(P):
            l, c = online_cross_entropy(logits[:, t], targets[:, t], mask[:, t])
            total_loss += l.item()
            total_count += c

        # Batched
        b_loss, b_count = batched_cross_entropy(logits, targets, mask)

        assert b_count == total_count
        assert abs(b_loss.item() - total_loss) < 1e-4


# ============================================================
# 9. Edge cases
# ============================================================

class TestEdgeCases:
    def test_p_equals_1(self):
        """forward_span with P=1 matches forward_one_token."""
        cfg = make_tiny_config(P=1, T=4)
        cfg.set_phase("A")
        torch.manual_seed(42)
        model = NeuromorphicLM(cfg)
        model.surprise = torch.zeros(BS, 1)

        input_ids = torch.randint(0, cfg.vocab_size, (BS, 1))
        reset_first = torch.zeros(BS, dtype=torch.bool)

        # Sequential
        model_seq = copy.deepcopy(model)
        logits_seq, _, _ = model_seq.forward_one_token(input_ids[:, 0], reset_first)

        # Parallel
        model_par = copy.deepcopy(model)
        logits_par, _, _ = model_par.forward_span(input_ids, reset_first)

        _assert_tensors_close(logits_par[:, 0], logits_seq, "p1_logits", atol=1e-5)

    def test_all_reset_span(self):
        """Every token in the span triggers a reset (all eot_id inputs)."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        torch.manual_seed(42)
        model = NeuromorphicLM(cfg)
        model.surprise = torch.zeros(BS, 1)

        input_ids = torch.full((BS, P), cfg.eot_id, dtype=torch.long)
        reset_first = torch.ones(BS, dtype=torch.bool)

        # Should not crash
        logits, _, _ = model.forward_span(input_ids, reset_first)
        assert logits.shape == (BS, P, cfg.vocab_size)
        assert torch.isfinite(logits).all()

    def test_no_reset_span(self):
        """No resets within the span (no eot_id tokens)."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        torch.manual_seed(42)
        model = NeuromorphicLM(cfg)
        model.surprise = torch.zeros(BS, 1)

        # Avoid eot_id (default 0) — use tokens 1..63
        input_ids = torch.randint(1, cfg.vocab_size, (BS, P))
        reset_first = torch.zeros(BS, dtype=torch.bool)

        logits, _, _ = model.forward_span(input_ids, reset_first)
        assert logits.shape == (BS, P, cfg.vocab_size)
        assert torch.isfinite(logits).all()

    def test_compute_reset_masks(self):
        """_compute_reset_masks produces correct masks."""
        cfg = make_tiny_config()
        eot = cfg.eot_id  # default is 2
        model = NeuromorphicLM(cfg)

        input_ids = torch.tensor([[eot, 5, eot, 3],   # eot at pos 0 and 2
                                  [5, 5, 5, 5]])       # no eot
        reset_first = torch.tensor([True, False])

        masks = model._compute_reset_masks(input_ids, reset_first)

        # Stream 0: reset at t=0 (reset_first), t=1 (input[0]=eot), t=3 (input[2]=eot)
        assert masks[0, 0] == True
        assert masks[0, 1] == True   # input_ids[0, 0] == eot_id
        assert masks[0, 2] == False  # input_ids[0, 1] == 5, not eot
        assert masks[0, 3] == True   # input_ids[0, 2] == eot_id

        # Stream 1: no resets
        assert masks[1, 0] == False
        assert masks[1, 1] == False
        assert masks[1, 2] == False
        assert masks[1, 3] == False


class TestEligibilityBatch:
    """Test that update_eligibility_batch matches sequential loop."""

    def test_matches_sequential(self):
        """Batched eligibility produces same result as P sequential calls."""
        cfg = make_tiny_config()
        pm_seq = ProceduralMemory(cfg)
        pm_batch = ProceduralMemory(cfg)

        # Share parameters
        pm_batch.load_state_dict(pm_seq.state_dict())

        D_h = cfg.D_h
        x_all = torch.randn(BS, P, D_h)
        h_all = torch.randn(BS, P, D_h)
        surprise_all = torch.rand(BS, P, 1) * 5.0
        reset_mask = torch.zeros(BS, P, dtype=torch.bool)

        # Sequential path
        pm_seq._lazy_init(BS, x_all.device)
        for t in range(P):
            pm_seq.update_eligibility(x_all[:, t], h_all[:, t], surprise_all[:, t])

        # Batch path
        pm_batch._lazy_init(BS, x_all.device)
        pm_batch.update_eligibility_batch(x_all, h_all, surprise_all, reset_mask)

        torch.testing.assert_close(pm_batch.elig_K, pm_seq.elig_K, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(pm_batch.elig_V, pm_seq.elig_V, atol=1e-6, rtol=1e-5)

    def test_with_midspan_reset(self):
        """Batched eligibility handles mid-span resets correctly."""
        cfg = make_tiny_config()
        pm_seq = ProceduralMemory(cfg)
        pm_batch = ProceduralMemory(cfg)
        pm_batch.load_state_dict(pm_seq.state_dict())

        D_h = cfg.D_h
        x_all = torch.randn(BS, P, D_h)
        h_all = torch.randn(BS, P, D_h)
        surprise_all = torch.rand(BS, P, 1) * 5.0
        # Reset at position 2 for stream 0 only
        reset_mask = torch.zeros(BS, P, dtype=torch.bool)
        reset_mask[0, 2] = True

        # Sequential path (reset before update at that position)
        pm_seq._lazy_init(BS, x_all.device)
        for t in range(P):
            if reset_mask[:, t].any():
                pm_seq.reset_eligibility(reset_mask[:, t])
            pm_seq.update_eligibility(x_all[:, t], h_all[:, t], surprise_all[:, t])

        # Batch path
        pm_batch._lazy_init(BS, x_all.device)
        pm_batch.update_eligibility_batch(x_all, h_all, surprise_all, reset_mask)

        torch.testing.assert_close(pm_batch.elig_K, pm_seq.elig_K, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(pm_batch.elig_V, pm_seq.elig_V, atol=1e-6, rtol=1e-5)

    def test_with_nonzero_initial_elig(self):
        """Batched eligibility works when starting from non-zero traces."""
        cfg = make_tiny_config()
        pm_seq = ProceduralMemory(cfg)
        pm_batch = ProceduralMemory(cfg)
        pm_batch.load_state_dict(pm_seq.state_dict())

        D_h = cfg.D_h
        x_all = torch.randn(BS, P, D_h)
        h_all = torch.randn(BS, P, D_h)
        surprise_all = torch.rand(BS, P, 1) * 5.0
        reset_mask = torch.zeros(BS, P, dtype=torch.bool)

        # Pre-populate eligibility with non-zero values
        pm_seq._lazy_init(BS, x_all.device)
        pm_batch._lazy_init(BS, x_all.device)
        init_elig_K = torch.randn(BS, cfg.r, D_h)
        init_elig_V = torch.randn(BS, cfg.r, D_h)
        pm_seq.elig_K = init_elig_K.clone()
        pm_seq.elig_V = init_elig_V.clone()
        pm_batch.elig_K = init_elig_K.clone()
        pm_batch.elig_V = init_elig_V.clone()

        # Sequential
        for t in range(P):
            pm_seq.update_eligibility(x_all[:, t], h_all[:, t], surprise_all[:, t])

        # Batch
        pm_batch.update_eligibility_batch(x_all, h_all, surprise_all, reset_mask)

        torch.testing.assert_close(pm_batch.elig_K, pm_seq.elig_K, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(pm_batch.elig_V, pm_seq.elig_V, atol=1e-6, rtol=1e-5)

    def test_gradient_flow(self):
        """Gradients flow through batched eligibility projections."""
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg)
        D_h = cfg.D_h

        x_all = torch.randn(BS, P, D_h)
        h_all = torch.randn(BS, P, D_h)
        surprise_all = torch.rand(BS, P, 1) * 5.0
        reset_mask = torch.zeros(BS, P, dtype=torch.bool)

        pm._lazy_init(BS, x_all.device)
        pm.update_eligibility_batch(x_all, h_all, surprise_all, reset_mask)

        loss = pm.elig_K.sum() + pm.elig_V.sum()
        loss.backward()

        assert pm.W_k_pre.weight.grad is not None
        assert pm.W_v_post.weight.grad is not None
        assert torch.isfinite(pm.W_k_pre.weight.grad).all()
        assert torch.isfinite(pm.W_v_post.weight.grad).all()


# ============================================================
# 11. Spatial decoder in forward_span
# ============================================================

class TestSpatialDecoderSpan:
    """Tests that the spatial decoder works correctly in forward_span."""

    def test_decoder_params_get_gradients(self):
        """Decoder parameters receive non-None gradients through forward_span."""
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase("A")
        torch.manual_seed(42)
        model = NeuromorphicLM(cfg)
        model.surprise = torch.zeros(BS, 1)

        input_ids = torch.randint(0, cfg.vocab_size, (BS, P))
        reset_first = torch.zeros(BS, dtype=torch.bool)

        logits, _, _ = model.forward_span(input_ids, reset_first)
        loss = logits.sum()
        loss.backward()

        assert model.spatial_decoder is not None
        decoder_grads = {n: p.grad for n, p in model.spatial_decoder.named_parameters()
                         if p.grad is not None}
        assert len(decoder_grads) > 0, "No decoder params got gradients"
        for name, grad in decoder_grads.items():
            assert torch.isfinite(grad).all(), f"Non-finite grad: {name}"

    @pytest.mark.parametrize("phase", ["A", "B", "C"])
    def test_span_matches_sequential_with_decoder(self, phase):
        """forward_span with decoder matches sequential forward_one_token with decoder."""
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase(phase)
        torch.manual_seed(42)
        model = NeuromorphicLM(cfg)

        input_ids = torch.randint(0, cfg.vocab_size, (BS, P))
        reset_first = torch.zeros(BS, dtype=torch.bool)

        # Sequential path (with frozen surprise, decoder enabled)
        model_seq = copy.deepcopy(model)
        model_seq.surprise = torch.zeros(BS, 1)
        frozen_surprise = model_seq.surprise.clone()
        seq_logits = []
        for t in range(P):
            if t == 0:
                rm = reset_first
            else:
                rm = (input_ids[:, t - 1] == cfg.eot_id)
            if not cfg.reset_on_doc_boundary:
                rm = torch.zeros_like(rm)
            logits, _, _ = model_seq.forward_one_token(input_ids[:, t], rm)
            seq_logits.append(logits)
            model_seq.surprise = frozen_surprise.clone()
        seq_logits = torch.stack(seq_logits, dim=1)

        # Parallel path (decoder enabled)
        model_par = copy.deepcopy(model)
        model_par.surprise = torch.zeros(BS, 1)
        par_logits, _, _ = model_par.forward_span(input_ids, reset_first)

        _assert_tensors_close(par_logits, seq_logits,
                              f"decoder_{phase}_logits", atol=1e-4)

    def test_decoder_disabled_skips_decoder(self):
        """When snapshot_enabled=False, forward_span uses direct lm_head."""
        cfg = make_tiny_config(snapshot_enabled=False)
        cfg.set_phase("A")
        torch.manual_seed(42)
        model = NeuromorphicLM(cfg)
        model.surprise = torch.zeros(BS, 1)

        assert model.spatial_decoder is None

        input_ids = torch.randint(0, cfg.vocab_size, (BS, P))
        reset_first = torch.zeros(BS, dtype=torch.bool)

        logits, _, _ = model.forward_span(input_ids, reset_first)
        assert logits.shape == (BS, P, cfg.vocab_size)
        assert torch.isfinite(logits).all()

    def test_block_layer_stack_shape(self):
        """Block._last_layer_stack has correct shape after forward_span."""
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase("A")
        torch.manual_seed(42)
        model = NeuromorphicLM(cfg)
        model.surprise = torch.zeros(BS, 1)

        input_ids = torch.randint(0, cfg.vocab_size, (BS, P))
        reset_first = torch.zeros(BS, dtype=torch.bool)

        model.forward_span(input_ids, reset_first)

        for b, block in enumerate(model.blocks):
            assert hasattr(block, '_last_layer_stack')
            assert block._last_layer_stack.shape == (BS, P, cfg.L, cfg.D_h), \
                f"block {b}: expected {(BS, P, cfg.L, cfg.D_h)}, got {block._last_layer_stack.shape}"
