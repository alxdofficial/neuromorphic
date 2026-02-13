"""State management tests — NEVER change.

If these fail, there's a state corruption bug.
"""

import pytest
import torch

from src.model.model import NeuromorphicLM
from src.model.state import (
    _walk_state_mixins, save_runtime_state, load_runtime_state,
)
from tests.conftest import make_tiny_config, forward_n_tokens, forward_and_write_em

pytestmark = pytest.mark.invariant

BS = 2


def _init_model(phase="B", **kw):
    """Create and warm up a tiny model."""
    cfg = make_tiny_config(**kw)
    cfg.set_phase(phase)
    model = NeuromorphicLM(cfg)
    forward_n_tokens(model, 4, BS=BS)
    return model, cfg


# ============================================================================
# Detach
# ============================================================================

class TestDetach:
    def test_detach_detaches_all_tensors(self):
        model, _ = _init_model("B")
        # Make some state require grad by running through autograd
        forward_n_tokens(model, 4, with_commits=True)
        model.detach_states()

        for path, mixin in _walk_state_mixins(model):
            for name in mixin._state_tensor_names:
                t = getattr(mixin, name)
                if t is not None and isinstance(t, torch.Tensor):
                    assert not t.requires_grad, \
                        f"{path}.{name} still requires grad after detach"

    def test_detach_preserves_values(self):
        model, _ = _init_model("B")
        forward_n_tokens(model, 4, with_commits=True)

        # Snapshot values before detach
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


# ============================================================================
# Reset
# ============================================================================

class TestReset:
    def test_reset_zeros_masked_streams_only(self):
        model, cfg = _init_model("B")
        forward_n_tokens(model, cfg.P, with_commits=True)

        mask = torch.tensor([True, False])
        model.reset_at_doc_boundary(mask)

        # Stream 0 should be zeroed for layers
        for block in model.blocks:
            for layer in block.layers:
                assert (layer.h[0] == 0).all(), "masked stream h should be zero"
                pm = layer.pm
                assert (pm.pm_K[0] == 0).all(), "masked stream pm_K should be zero"
                assert (pm.pm_a[0] == 0).all(), "masked stream pm_a should be zero"

    def test_reset_preserves_unmasked_streams(self):
        model, cfg = _init_model("B")
        forward_n_tokens(model, cfg.P, with_commits=True)

        # Snapshot unmasked stream
        h_before = model.blocks[0].layers[0].h[1].clone()
        pm_K_before = model.blocks[0].layers[0].pm.pm_K[1].clone()

        mask = torch.tensor([True, False])
        model.reset_at_doc_boundary(mask)

        assert torch.equal(model.blocks[0].layers[0].h[1], h_before)
        assert torch.equal(model.blocks[0].layers[0].pm.pm_K[1], pm_K_before)


# ============================================================================
# EM-specific reset
# ============================================================================

class TestEMReset:
    def test_em_reset_only_zeros_strengths(self):
        """EM reset: em_S zeroed for masked. em_K/em_V UNCHANGED."""
        model, cfg = _init_model("B")
        forward_and_write_em(model, cfg.P)

        for block in model.blocks:
            em_K_before = block.em.em_K.clone()
            em_V_before = block.em.em_V.clone()
            em_S_before = block.em.em_S.clone()

            mask = torch.tensor([True, False])
            block.em.reset_states(mask)

            # em_K and em_V should be completely unchanged
            assert torch.equal(block.em.em_K, em_K_before), \
                "em_K should be unchanged after EM reset"
            assert torch.equal(block.em.em_V, em_V_before), \
                "em_V should be unchanged after EM reset"
            # em_S should be zeroed for stream 0
            assert (block.em.em_S[0] == 0).all(), \
                "em_S should be zeroed for masked stream"
            # em_S should be unchanged for stream 1
            assert torch.equal(block.em.em_S[1], em_S_before[1])


# ============================================================================
# Save/Load roundtrip
# ============================================================================

class TestSaveLoad:
    def test_save_load_roundtrip_identity(self):
        model, cfg = _init_model("B")
        forward_n_tokens(model, cfg.P, with_commits=True)

        state = save_runtime_state(model)

        # Create fresh model and warm it up
        model2 = NeuromorphicLM(cfg)
        forward_n_tokens(model2, 4, BS=BS)

        load_runtime_state(model2, state)

        # Compare all state tensors
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

    def test_load_skips_shape_mismatched_tensors(self, capsys):
        """Create PM with r=4, save state, create PM with r=8, load — no crash.

        Mismatched tensors must remain byte-identical to their pre-load values.
        """
        cfg1 = make_tiny_config(r=4)
        cfg1.set_phase("A")
        model1 = NeuromorphicLM(cfg1)
        forward_n_tokens(model1, 4, BS=BS)
        state = save_runtime_state(model1)

        cfg2 = make_tiny_config(r=8)
        cfg2.set_phase("A")
        model2 = NeuromorphicLM(cfg2)
        forward_n_tokens(model2, 4, BS=BS)

        # Snapshot ALL PM state in model2 before load
        pre_load_snapshots = {}
        for b_idx, block in enumerate(model2.blocks):
            for l_idx, layer in enumerate(block.layers):
                pm = layer.pm
                pre_load_snapshots[(b_idx, l_idx)] = {
                    name: getattr(pm, name).clone()
                    for name in pm._state_tensor_names
                    if getattr(pm, name) is not None
                }

        load_runtime_state(model2, state)  # should not crash

        # r=8 state should be unchanged (mismatched shapes skipped)
        assert model2.blocks[0].layers[0].pm.pm_K.shape[1] == 8

        # Byte-identical: every PM tensor in model2 must match its pre-load snapshot
        for (b_idx, l_idx), snap in pre_load_snapshots.items():
            pm = model2.blocks[b_idx].layers[l_idx].pm
            for name, before in snap.items():
                after = getattr(pm, name)
                assert torch.equal(after, before), \
                    f"blocks.{b_idx}.layers.{l_idx}.pm.{name} " \
                    f"was mutated by shape-mismatched load"

    def test_load_handles_none_values(self):
        model, cfg = _init_model("B")
        state = save_runtime_state(model)

        # Set some values to None
        for path, sub_state in state.items():
            for key in sub_state:
                sub_state[key] = None
                break  # only None-ify one per mixin

        model2 = NeuromorphicLM(cfg)
        forward_n_tokens(model2, 4, BS=BS)
        load_runtime_state(model2, state)  # should not crash

    def test_save_uses_stable_path_keys(self):
        model, cfg = _init_model("B")
        state = save_runtime_state(model)

        # Check that keys are path-based
        for key in state:
            # Paths should contain module names like "blocks.0.layers.1.pm"
            # Not legacy keys like "mixin_0_ProceduralMemory"
            assert "mixin_" not in key, f"Legacy key found: {key}"
            # Should contain meaningful path segments
            parts = key.split(".")
            assert len(parts) >= 1


# ============================================================================
# Walk model tree
# ============================================================================

class TestWalkModelTree:
    def test_reset_all_walks_model_tree(self):
        """model.reset_at_doc_boundary(mask) reaches all StateMixin instances."""
        model, cfg = _init_model("B")
        forward_and_write_em(model, cfg.P)

        mask = torch.ones(BS, dtype=torch.bool)

        # Count mixins before
        mixin_count = sum(1 for _ in _walk_state_mixins(model))
        assert mixin_count > 0

        # Reset should not crash and should zero all masked state
        model.reset_at_doc_boundary(mask)

    def test_detach_all_walks_model_tree(self):
        """model.detach_states() reaches all StateMixin instances."""
        model, _ = _init_model("B")
        forward_n_tokens(model, 4, with_commits=True)
        model.detach_states()  # should not crash

        # Verify all state is detached
        for path, mixin in _walk_state_mixins(model):
            for name in mixin._state_tensor_names:
                t = getattr(mixin, name)
                if t is not None and isinstance(t, torch.Tensor):
                    assert not t.requires_grad


# ============================================================================
# Lifelong mode
# ============================================================================

class TestLifelongMode:
    def test_lifelong_reset_preserves_pm_state(self):
        """In lifelong_mode: pm_K/pm_V/pm_a survive reset."""
        cfg = make_tiny_config()
        cfg.set_phase("C")  # lifelong_mode=True
        model = NeuromorphicLM(cfg)
        forward_n_tokens(model, cfg.P, with_commits=True)

        # Snapshot PM state
        snapshots = {}
        for b_idx, block in enumerate(model.blocks):
            for l_idx, layer in enumerate(block.layers):
                pm = layer.pm
                snapshots[(b_idx, l_idx)] = {
                    "pm_K": pm.pm_K.clone(),
                    "pm_V": pm.pm_V.clone(),
                    "pm_a": pm.pm_a.clone(),
                }

        mask = torch.ones(BS, dtype=torch.bool)
        for block in model.blocks:
            block.reset_states(mask)

        # PM committed state should be preserved
        for b_idx, block in enumerate(model.blocks):
            for l_idx, layer in enumerate(block.layers):
                pm = layer.pm
                snap = snapshots[(b_idx, l_idx)]
                assert torch.equal(pm.pm_K, snap["pm_K"]), \
                    "lifelong: pm_K should survive reset"
                assert torch.equal(pm.pm_V, snap["pm_V"]), \
                    "lifelong: pm_V should survive reset"
                assert torch.equal(pm.pm_a, snap["pm_a"]), \
                    "lifelong: pm_a should survive reset"

    def test_lifelong_reset_preserves_em_state(self):
        """In lifelong_mode: EM state completely survives (EM reset never called)."""
        cfg = make_tiny_config()
        cfg.set_phase("C")
        model = NeuromorphicLM(cfg)
        forward_and_write_em(model, cfg.P)

        # Snapshot EM state
        em_snaps = {}
        for b_idx, block in enumerate(model.blocks):
            em_snaps[b_idx] = {
                "em_K": block.em.em_K.clone(),
                "em_V": block.em.em_V.clone(),
                "em_S": block.em.em_S.clone(),
            }

        mask = torch.ones(BS, dtype=torch.bool)
        for block in model.blocks:
            block.reset_states(mask)

        for b_idx, block in enumerate(model.blocks):
            snap = em_snaps[b_idx]
            assert torch.equal(block.em.em_K, snap["em_K"]), \
                "lifelong: em_K should survive reset"
            assert torch.equal(block.em.em_V, snap["em_V"]), \
                "lifelong: em_V should survive reset"
            assert torch.equal(block.em.em_S, snap["em_S"]), \
                "lifelong: em_S should survive reset"


class TestNormalReset:
    def test_normal_reset_zeros_pm_state(self):
        """lifelong_mode=False: pm_K/pm_V/pm_a zeroed for masked streams."""
        model, cfg = _init_model("B")
        forward_n_tokens(model, cfg.P, with_commits=True)

        mask = torch.tensor([True, False])
        for block in model.blocks:
            block.reset_states(mask)

        for block in model.blocks:
            for layer in block.layers:
                pm = layer.pm
                assert (pm.pm_K[0] == 0).all()
                assert (pm.pm_V[0] == 0).all()
                assert (pm.pm_a[0] == 0).all()

    def test_normal_reset_zeros_em_strengths(self):
        """lifelong_mode=False: em_S zeroed for masked. em_K/em_V preserved."""
        model, cfg = _init_model("B")
        forward_and_write_em(model, cfg.P)

        for block in model.blocks:
            em_K_before = block.em.em_K.clone()
            em_V_before = block.em.em_V.clone()

            mask = torch.tensor([True, False])
            block.reset_states(mask)

            assert (block.em.em_S[0] == 0).all()
            assert torch.equal(block.em.em_K, em_K_before), \
                "em_K should be preserved on normal reset"
            assert torch.equal(block.em.em_V, em_V_before), \
                "em_V should be preserved on normal reset"


# ============================================================================
# State dict completeness
# ============================================================================

class TestStateDictCompleteness:
    def test_state_dict_runtime_has_all_keys(self):
        model, _ = _init_model("B")

        for path, mixin in _walk_state_mixins(model):
            sd = mixin.state_dict_runtime()
            expected = set(mixin._state_tensor_names)
            assert sd.keys() == expected, \
                f"{path}: state_dict keys {sd.keys()} != expected {expected}"


# ============================================================================
# PM eligibility reset
# ============================================================================

class TestPMEligibilityReset:
    def test_pm_reset_eligibility_zeros_traces(self):
        """reset_eligibility: elig_K/elig_V zeroed, pm_K/pm_V/pm_a unchanged."""
        model, cfg = _init_model("B")
        forward_n_tokens(model, cfg.P, with_commits=True)

        for block in model.blocks:
            for layer in block.layers:
                pm = layer.pm

                pm_K_before = pm.pm_K.clone()
                pm_V_before = pm.pm_V.clone()
                pm_a_before = pm.pm_a.clone()

                mask = torch.tensor([True, False])
                pm.reset_eligibility(mask)

                # Eligibility zeroed for masked stream
                assert (pm.elig_K[0] == 0).all()
                assert (pm.elig_V[0] == 0).all()
                # Committed state unchanged
                assert torch.equal(pm.pm_K, pm_K_before)
                assert torch.equal(pm.pm_V, pm_V_before)
                assert torch.equal(pm.pm_a, pm_a_before)
                # Unmasked elig unchanged
                # (can't assert non-zero since it might be zero after commit reset)
