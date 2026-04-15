"""Phase 2 rollout + GRPO tests for the factored-categorical policy."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from src.model.config import Config
from src.model.memory import MemoryGraph


def _tiny_config():
    return Config.tier_tiny()


class _StubLM(nn.Module):
    """Minimal LM stub for phase-2 rollout (needs lm_head/proj_down/ln_final for
    the live-surprise signal in forward_segment_phase2)."""
    def __init__(self, config):
        super().__init__()
        self.lm_head = nn.Linear(config.D_embed, config.vocab_size, bias=False).to(torch.bfloat16)
        self.proj_down = (
            nn.Linear(config.D, config.D_embed).to(torch.bfloat16)
            if config.D != config.D_embed else None)
        self.ln_final = nn.LayerNorm(config.D_embed).to(torch.bfloat16)


def _make_setup(BS=2, T=16):
    config = _tiny_config()
    mg = MemoryGraph(config)
    mg.initialize_states(BS, torch.device("cpu"))
    lm = _StubLM(config)
    return config, mg, lm


class TestPhase2Rollout:
    def test_forward_segment_phase2_shapes(self):
        config, mg, lm = _make_setup(BS=2, T=16)
        H_mid = torch.randn(2, 16, config.D, dtype=torch.bfloat16)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        result = mg.forward_segment_phase2(H_mid, input_ids, lm, tau=1.0, sample=True)

        T = 16
        M = config.modulation_interval
        expected_calls = (T + M - 1) // M  # positions 0, M, 2M, ...
        assert result["readouts"].shape == (2, T, config.D)
        assert result["mod_inputs"].shape == (expected_calls, 2, config.N_cells, config.mod_in)
        # codes shape is [n_calls, BS, NC] long (no extra levels dim)
        assert result["codes"].shape == (expected_calls, 2, config.N_cells)
        assert result["codes"].dtype == torch.long
        assert result["call_positions"].tolist() == list(range(0, T, M))[:expected_calls]

    def test_sample_vs_argmax(self):
        config, mg, lm = _make_setup(BS=2, T=16)
        H_mid = torch.randn(2, 16, config.D, dtype=torch.bfloat16)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))

        torch.manual_seed(0)
        mg.initialize_states(2, torch.device("cpu"))
        r1 = mg.forward_segment_phase2(H_mid, input_ids, lm, sample=False)
        torch.manual_seed(0)
        mg.initialize_states(2, torch.device("cpu"))
        r2 = mg.forward_segment_phase2(H_mid, input_ids, lm, sample=False)
        assert (r1["codes"] == r2["codes"]).all(), "argmax must be deterministic"


class TestGRPOGradientFlow:
    def test_grpo_backprop_reaches_only_logit_head(self):
        """GRPO in the new architecture should only update the neuromod's
        logit head — not codebook, decoder, memory dynamics, or LM."""
        config, mg, lm = _make_setup(BS=2, T=8)
        dp = mg.discrete_policy

        # Freeze everything except logit head (matches phase2 trainer setup)
        for p in mg.parameters():
            p.requires_grad = False
        for p in (dp.logit_w1, dp.logit_b1, dp.logit_w2, dp.logit_b2):
            p.requires_grad = True

        BS, NC = 2, config.N_cells
        mod_input = torch.randn(BS, NC, config.mod_in, dtype=torch.float32)
        codes = torch.randint(0, dp.K, (BS, NC))

        # Forward logits, score, backward
        logits = dp.compute_logits(mod_input)
        log_probs = F.log_softmax(logits, dim=-1)
        log_pi = log_probs.gather(-1, codes.unsqueeze(-1)).squeeze(-1)
        loss = -log_pi.sum()
        loss.backward()

        # Gradient should reach logit head
        assert dp.logit_w1.grad is not None
        assert dp.logit_w1.grad.abs().sum() > 0
        assert dp.logit_w2.grad is not None
        assert dp.logit_w2.grad.abs().sum() > 0

        # Should NOT reach frozen parts (codebook, decoder, dynamics)
        assert dp.codebook.grad is None
        assert dp.dec_w1.grad is None
        assert mg.state_w1.grad is None

    def test_decode_gradient_reaches_codebook_and_decoder(self):
        """Phase 1 backprop (via Gumbel-softmax + decode) must deliver
        gradient to codebook and decoder too."""
        config, mg, _lm = _make_setup(BS=2, T=8)
        dp = mg.discrete_policy

        BS, NC = 2, config.N_cells
        mod_input = torch.randn(BS, NC, config.mod_in)
        out = dp.forward(mod_input, phase="phase1", tau=1.0, hard_gumbel=True)
        loss = out["action"].pow(2).mean()
        loss.backward()

        assert dp.logit_w1.grad is not None
        assert dp.codebook.grad is not None
        assert dp.dec_w1.grad is not None
        assert dp.codebook.grad.abs().sum() > 0
        assert dp.dec_w1.grad.abs().sum() > 0
