"""State management tests (v4) — NEVER change.

If these fail, there's a state corruption bug.
"""

import pytest
import torch

from src.model.model import NeuromorphicLM
from src.model.state import (
    _walk_state_mixins, save_runtime_state, load_runtime_state,
)
from tests.conftest import make_tiny_config, forward_one_segment

pytestmark = pytest.mark.invariant

BS = 2


def _init_model(phase="A", **kw):
    """Create and warm up a tiny v4 model."""
    cfg = make_tiny_config(**kw)
    cfg.set_phase(phase)
    model = NeuromorphicLM(cfg)
    model.initialize_states(BS, torch.device("cpu"))

    # Forward a segment to populate state
    input_ids = torch.randint(0, cfg.vocab_size, (BS, cfg.N))
    model.forward_segment(input_ids)
    return model, cfg


class TestDetach:
    def test_detach_detaches_all_tensors(self):
        model, _ = _init_model("A")
        model.detach_states()

        for path, mixin in _walk_state_mixins(model):
            for name in mixin._state_tensor_names:
                t = getattr(mixin, name)
                if t is not None and isinstance(t, torch.Tensor):
                    assert not t.requires_grad, \
                        f"{path}.{name} still requires grad after detach"

    def test_detach_preserves_values(self):
        model, _ = _init_model("A")

        snapshots = {}
        for path, mixin in _walk_state_mixins(model):
            for name in mixin._state_tensor_names:
                t = getattr(mixin, name)
                if t is not None and isinstance(t, torch.Tensor):
                    snapshots[(path, name)] = t.detach().clone()

        model.detach_states()

        for (path, name), before in snapshots.items():
            mixin = dict(_walk_state_mixins(model))[path]
            after = getattr(mixin, name)
            assert torch.equal(before, after), \
                f"{path}.{name} values changed after detach"


class TestReset:
    def test_reset_zeros_masked_pm(self):
        model, cfg = _init_model("A")

        # Give PM some content
        pm = model.blocks[0].pm
        pm.pm_K = torch.randn(BS, cfg.r, cfg.D_mem)
        pm.pm_a = torch.ones(BS, cfg.r)

        mask = torch.tensor([True, False])
        pm.reset_content(mask)

        assert (pm.pm_K[0] == 0).all(), "masked stream pm_K should be zero"
        assert (pm.pm_a[0] == 0).all(), "masked stream pm_a should be zero"
        assert pm.pm_K[1].abs().sum() > 0, "unmasked stream should be preserved"


class TestEMReset:
    def test_em_reset_only_zeros_strengths(self):
        """EM reset: em_S zeroed for masked. em_K/em_V UNCHANGED."""
        model, cfg = _init_model("A")

        em = model.blocks[0].em
        em.em_K = torch.randn(BS, cfg.M, cfg.D_mem)
        em.em_V = torch.randn(BS, cfg.M, cfg.D_mem)
        em.em_S = torch.ones(BS, cfg.M)
        em.em_age = torch.ones(BS, cfg.M) * 50

        em_K_before = em.em_K.clone()
        em_V_before = em.em_V.clone()

        mask = torch.tensor([True, False])
        em.reset_states(mask)

        assert torch.equal(em.em_K, em_K_before), "em_K should be unchanged"
        assert torch.equal(em.em_V, em_V_before), "em_V should be unchanged"
        assert (em.em_S[0] == 0).all(), "em_S should be zeroed for masked"
        assert (em.em_age[0] == 0).all(), "em_age should be zeroed for masked"
        assert (em.em_S[1] > 0).all(), "em_S should be preserved for unmasked"


class TestSaveLoad:
    def test_save_load_roundtrip_identity(self):
        model, cfg = _init_model("A")
        state = save_runtime_state(model)

        model2 = NeuromorphicLM(cfg)
        model2.initialize_states(BS, torch.device("cpu"))
        load_runtime_state(model2, state)

        for path, mixin1 in _walk_state_mixins(model):
            mixin2 = dict(_walk_state_mixins(model2))[path]
            for name in mixin1._state_tensor_names:
                t1 = getattr(mixin1, name)
                t2 = getattr(mixin2, name)
                if t1 is None:
                    assert t2 is None, f"{path}.{name}: expected None"
                else:
                    assert torch.equal(t1, t2), \
                        f"{path}.{name}: values differ after load"

    def test_load_skips_shape_mismatched_tensors(self):
        cfg1 = make_tiny_config(r=4)
        cfg1.set_phase("A")
        model1 = NeuromorphicLM(cfg1)
        model1.initialize_states(BS, torch.device("cpu"))
        input_ids = torch.randint(0, 64, (BS, cfg1.N))
        model1.forward_segment(input_ids)
        state = save_runtime_state(model1)

        cfg2 = make_tiny_config(r=8)
        cfg2.set_phase("A")
        model2 = NeuromorphicLM(cfg2)
        model2.initialize_states(BS, torch.device("cpu"))

        # Snapshot PM state before load
        pm_K_before = model2.blocks[0].pm.pm_K.clone()

        load_runtime_state(model2, state)

        # r=8 state should be unchanged (mismatched shapes skipped)
        assert model2.blocks[0].pm.pm_K.shape[1] == 8
        assert torch.equal(model2.blocks[0].pm.pm_K, pm_K_before)

    def test_save_uses_stable_path_keys(self):
        model, cfg = _init_model("A")
        state = save_runtime_state(model)

        for key in state:
            assert "mixin_" not in key, f"Legacy key found: {key}"


class TestWalkModelTree:
    def test_walk_finds_all_mixins(self):
        model, cfg = _init_model("A")
        mixins = list(_walk_state_mixins(model))

        # Should find PM and EM for each block
        pm_count = sum(1 for _, m in mixins if type(m).__name__ == "ProceduralMemory")
        em_count = sum(1 for _, m in mixins if type(m).__name__ == "EpisodicMemory")
        assert pm_count == cfg.B_blocks
        assert em_count == cfg.B_blocks

    def test_detach_all_walks_model_tree(self):
        model, _ = _init_model("A")
        model.detach_states()

        for path, mixin in _walk_state_mixins(model):
            for name in mixin._state_tensor_names:
                t = getattr(mixin, name)
                if t is not None and isinstance(t, torch.Tensor):
                    assert not t.requires_grad


class TestStateDictCompleteness:
    def test_state_dict_runtime_has_all_keys(self):
        model, _ = _init_model("A")

        for path, mixin in _walk_state_mixins(model):
            sd = mixin.state_dict_runtime()
            expected = set(mixin._state_tensor_names)
            assert sd.keys() == expected, \
                f"{path}: state_dict keys {sd.keys()} != expected {expected}"
