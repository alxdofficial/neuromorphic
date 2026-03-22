"""Neuromodulator behavior tests (v5) — EMNeuromodulator only."""

import pytest
import torch

from src.model.episodic_memory import EMNeuromodulator


BS = 2
B = 2


class TestEMNeuromodulatorBehavior:
    def test_differentiable(self):
        neuromod = EMNeuromodulator(hidden=8)

        novelty = torch.randn(BS, B, requires_grad=True)
        usage = torch.randn(BS, B, requires_grad=True)
        g = neuromod(novelty, usage)

        loss = g.sum()
        loss.backward()

        assert novelty.grad is not None
        assert usage.grad is not None

    def test_g_bounded(self):
        neuromod = EMNeuromodulator(hidden=8)
        g = neuromod(torch.rand(BS, B), torch.rand(BS, B))
        assert (g >= 0.001).all()
        assert (g <= 0.95).all()

    def test_custom_bounds(self):
        neuromod = EMNeuromodulator(hidden=8, g_floor=0.01, g_ceil=0.5)
        g = neuromod(torch.rand(BS, B), torch.rand(BS, B))
        assert (g >= 0.01).all()
        assert (g <= 0.5).all()
