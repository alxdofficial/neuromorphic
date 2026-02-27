"""Procedural Memory tests (v4)."""

import pytest
import torch
from tests.conftest import make_tiny_config

from src.model.procedural_memory import ProceduralMemory, PMNeuromodulator
from src.model.utils import unit_normalize


BS = 2


class TestProceduralMemory:
    def test_initialize(self):
        cfg = make_tiny_config()
        B = cfg.B_blocks
        pm = ProceduralMemory(cfg.D_mem, cfg.r, cfg)
        assert not pm.is_initialized()

        pm.initialize(BS, torch.device("cpu"), torch.float32)
        assert pm.is_initialized()
        assert pm.pm_K.shape == (BS, B, cfg.r, cfg.D_mem)
        assert pm.pm_V.shape == (BS, B, cfg.r, cfg.D_mem)
        assert pm.pm_a.shape == (BS, B, cfg.r)

    def test_holographic_read(self):
        """Holographic read: output depends on input (not just retrieval)."""
        cfg = make_tiny_config()
        B = cfg.B_blocks
        pm = ProceduralMemory(cfg.D_mem, cfg.r, cfg)
        pm.initialize(BS, torch.device("cpu"), torch.float32)

        # Populate with random content
        pm.pm_K = unit_normalize(torch.randn(BS, B, cfg.r, cfg.D_mem))
        pm.pm_V = torch.randn(BS, B, cfg.r, cfg.D_mem)
        pm.pm_a = torch.ones(BS, B, cfg.r)

        q1 = torch.randn(BS, 4, B, cfg.C, cfg.D_mem)
        q2 = torch.randn(BS, 4, B, cfg.C, cfg.D_mem)

        y1 = pm.read(q1)
        y2 = pm.read(q2)

        # Different inputs should produce different outputs
        assert not torch.allclose(y1, y2)

    def test_commit_increases_strength(self):
        cfg = make_tiny_config()
        B = cfg.B_blocks
        pm = ProceduralMemory(cfg.D_mem, cfg.r, cfg)
        pm.initialize(BS, torch.device("cpu"), torch.float32)

        a_before = pm.pm_a.sum().item()

        elig_K = torch.randn(BS, B, cfg.r, cfg.D_mem)
        elig_V = torch.randn(BS, B, cfg.r, cfg.D_mem)
        g = torch.full((BS, B), 0.5)
        slot_logits = torch.randn(BS, B, cfg.r)
        tau = torch.ones(BS, B)

        pm.commit(elig_K, elig_V, g, slot_logits, tau)
        a_after = pm.pm_a.sum().item()

        assert a_after > a_before

    def test_base_decay(self):
        cfg = make_tiny_config()
        B = cfg.B_blocks
        pm = ProceduralMemory(cfg.D_mem, cfg.r, cfg)
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        pm.pm_a = torch.ones(BS, B, cfg.r)

        a_before = pm.pm_a.sum().item()
        pm.base_decay()
        a_after = pm.pm_a.sum().item()

        assert a_after < a_before

    def test_budget_enforcement(self):
        cfg = make_tiny_config()
        B = cfg.B_blocks
        pm = ProceduralMemory(cfg.D_mem, cfg.r, cfg)
        pm.initialize(BS, torch.device("cpu"), torch.float32)

        # Force over budget
        pm.pm_a = torch.full((BS, B, cfg.r), cfg.a_max)

        # Commit with high write strength
        elig_K = torch.randn(BS, B, cfg.r, cfg.D_mem)
        elig_V = torch.randn(BS, B, cfg.r, cfg.D_mem)
        g = torch.ones(BS, B)
        slot_logits = torch.zeros(BS, B, cfg.r)
        tau = torch.ones(BS, B)

        pm.commit(elig_K, elig_V, g, slot_logits, tau)

        # Budget should be enforced (per stream per block)
        assert pm.pm_a.sum(dim=-1).max().item() <= cfg.budget_pm + 0.01

    def test_reset_content(self):
        cfg = make_tiny_config()
        B = cfg.B_blocks
        pm = ProceduralMemory(cfg.D_mem, cfg.r, cfg)
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        pm.pm_K = torch.randn(BS, B, cfg.r, cfg.D_mem)
        pm.pm_a = torch.ones(BS, B, cfg.r)

        # Reset first stream only
        mask = torch.tensor([True, False])
        pm.reset_content(mask)

        assert pm.pm_K[0].abs().sum() == 0
        assert pm.pm_K[1].abs().sum() > 0
        assert pm.pm_a[0].sum() == 0
        assert pm.pm_a[1].sum() > 0


class TestPMNeuromodulator:
    def test_output_shapes(self):
        cfg = make_tiny_config()
        neuromod = PMNeuromodulator(cfg)

        elig_summary = torch.rand(BS)
        pm_usage = torch.rand(BS)
        content = torch.randn(BS, cfg.D_mem)

        g, slot_logits, tau = neuromod(elig_summary, pm_usage, content)
        assert g.shape == (BS,)
        assert slot_logits.shape == (BS, cfg.r)
        assert tau.shape == (BS,)

    def test_g_bounded(self):
        cfg = make_tiny_config()
        neuromod = PMNeuromodulator(cfg)

        g, _, _ = neuromod(torch.rand(BS), torch.rand(BS))
        assert (g >= 0).all() and (g <= 1).all()

    def test_tau_bounded(self):
        cfg = make_tiny_config()
        neuromod = PMNeuromodulator(cfg)

        _, _, tau = neuromod(torch.rand(BS), torch.rand(BS))
        assert (tau >= cfg.tau_pm_floor).all()
        assert (tau <= cfg.tau_pm_ceil).all()
