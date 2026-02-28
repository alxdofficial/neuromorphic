"""Neuromodulator behavior tests (v4)."""

import pytest
import torch
from tests.conftest import make_tiny_config

from src.model.procedural_memory import PMNeuromodulator
from src.model.episodic_memory import EMNeuromodulator


BS = 2


class TestPMNeuromodulatorBehavior:
    def test_differentiable(self):
        cfg = make_tiny_config()
        neuromod = PMNeuromodulator(cfg.D, cfg)

        elig = torch.randn(BS, requires_grad=True)
        usage = torch.randn(BS, requires_grad=True)
        g, slot_logits, tau, ww = neuromod(elig, usage)

        loss = g.sum() + slot_logits.sum() + tau.sum() + ww.sum()
        loss.backward()

        assert elig.grad is not None
        assert usage.grad is not None


class TestEMNeuromodulatorBehavior:
    def test_differentiable(self):
        cfg = make_tiny_config()
        neuromod = EMNeuromodulator(cfg.D, cfg)

        novelty = torch.randn(BS, requires_grad=True)
        usage = torch.randn(BS, requires_grad=True)
        g_em, tau, decay, ww = neuromod(novelty, usage)

        loss = g_em.sum() + tau.sum() + decay.sum() + ww.sum()
        loss.backward()

        assert novelty.grad is not None
        assert usage.grad is not None
