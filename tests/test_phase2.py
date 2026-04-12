"""End-to-end smoke test for phase 2 rollout and GRPO step.

Uses tier_tiny config and a stubbed LM / VQVAE where possible. Validates:
- forward_segment_phase2 runs and returns the expected record shapes
- compute_modulator_action → codebook fit → phase 2 rollout pipeline works
- GRPO gradient flows to modulator params only
"""

import torch
import torch.nn.functional as F
import pytest

from src.model.config import Config
from src.model.memory import MemoryGraph
from src.codebook import ActionVQVAE
from tests.test_memory import _StubLM


def _make_setup(BS=2, T=16):
    config = Config.tier_tiny(T=T, tbptt_block=4, modulation_interval=4)
    mg = MemoryGraph(config)
    mg.initialize_states(BS, torch.device("cpu"))
    lm = _StubLM(config)
    return config, mg, lm


def _make_vqvae(action_dim):
    vq = ActionVQVAE(
        action_dim=action_dim, latent_dim=8, hidden=32,
        num_levels=2, codes_per_level=8, beta=0.25)
    # Give it sensible normalization stats (zero mean, unit std)
    vq.set_normalization(
        torch.zeros(1, action_dim),
        torch.ones(1, action_dim),
    )
    return vq


class TestPhase2Rollout:
    def test_forward_segment_phase2_shapes(self):
        config, mg, lm = _make_setup(BS=2, T=16)
        vq = _make_vqvae(action_dim=config.mod_out)

        H_mid = torch.randn(2, 16, config.D, dtype=torch.bfloat16)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        result = mg.forward_segment_phase2(H_mid, input_ids, lm, vq, tau=1.0, sample=True)

        assert result["readouts"].shape == (2, 16, config.D)
        # With T=16, modulation_interval=4, expect 4 modulator calls at t=0,4,8,12
        assert result["mod_inputs"].shape == (4, 2, config.N_cells, config.mod_in)
        assert result["codes"].shape == (4, 2, config.N_cells, 2)  # num_levels=2
        assert result["call_positions"].tolist() == [0, 4, 8, 12]

    def test_sample_vs_argmax(self):
        config, mg, lm = _make_setup(BS=2, T=16)
        vq = _make_vqvae(action_dim=config.mod_out)
        H_mid = torch.randn(2, 16, config.D, dtype=torch.bfloat16)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))

        # Deterministic (argmax) run twice → same codes. Both runs must start
        # from identical memory state, so we seed before each initialize_states
        # (which samples random h and random sparse W permutations).
        torch.manual_seed(0)
        mg.initialize_states(2, torch.device("cpu"))
        r1 = mg.forward_segment_phase2(H_mid, input_ids, lm, vq, sample=False)
        torch.manual_seed(0)
        mg.initialize_states(2, torch.device("cpu"))
        r2 = mg.forward_segment_phase2(H_mid, input_ids, lm, vq, sample=False)
        assert (r1["codes"] == r2["codes"]).all(), "argmax should be deterministic"

    def test_phase1_phase2_parity(self):
        """forward_segment (phase 1) and forward_segment_phase2 (deterministic)
        should run on the same input and produce readouts/state in the same
        ballpark.

        Strict numerical equality is not achievable: phase 2 quantizes the
        continuous modulator output through the RVQ bottleneck, which
        introduces reconstruction error. But the two paths must produce
        OUTPUT SHAPES that match exactly, run without errors on the same
        inputs, and evolve the memory state in the same regime (readouts
        should have similar order-of-magnitude).

        This is the guard against silent drift between the duplicated
        _run_block (phase 1) and forward_segment_phase2 (phase 2) glue.
        If someone breaks one path, the shape or magnitude check catches it.

        See audit #5.
        """
        config, mg, lm = _make_setup(BS=2, T=16)
        vq = _make_vqvae(action_dim=config.mod_out)

        H_mid = torch.randn(2, 16, config.D, dtype=torch.bfloat16)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))

        # Run phase 1
        torch.manual_seed(123)
        mg.initialize_states(2, torch.device("cpu"))
        readouts_p1, _ = mg.forward_segment(H_mid.clone(), input_ids.clone(), lm)
        W_p1 = mg.W.clone()
        decay_p1 = mg.decay.clone()
        hebbian_p1 = mg.hebbian.clone()

        # Run phase 2 with the same seed, deterministic sampling
        torch.manual_seed(123)
        mg.initialize_states(2, torch.device("cpu"))
        r_p2 = mg.forward_segment_phase2(
            H_mid.clone(), input_ids.clone(), lm, vq, sample=False)
        readouts_p2 = r_p2["readouts"]
        W_p2 = mg.W.clone()
        decay_p2 = mg.decay.clone()
        hebbian_p2 = mg.hebbian.clone()

        # Shapes must match exactly.
        assert readouts_p1.shape == readouts_p2.shape, \
            f"readouts shape drift: {readouts_p1.shape} vs {readouts_p2.shape}"
        assert W_p1.shape == W_p2.shape
        assert decay_p1.shape == decay_p2.shape
        assert hebbian_p1.shape == hebbian_p2.shape

        # Both must produce non-trivial, finite outputs.
        assert readouts_p1.abs().sum() > 0
        assert readouts_p2.abs().sum() > 0
        assert torch.isfinite(readouts_p1).all()
        assert torch.isfinite(readouts_p2).all()
        assert torch.isfinite(W_p2).all()
        assert torch.isfinite(decay_p2).all()
        assert torch.isfinite(hebbian_p2).all()

        # Readouts should be in the same regime. VQ reconstruction error
        # with small test codebooks can be large, so we only check that
        # readout magnitudes haven't diverged by an order of magnitude.
        # This catches gross regressions (one path outputs zeros, NaNs,
        # or wildly inflated values) without requiring numerical equality.
        mag_p1 = readouts_p1.abs().mean().item()
        mag_p2 = readouts_p2.abs().mean().item()
        ratio = max(mag_p1, mag_p2) / max(min(mag_p1, mag_p2), 1e-8)
        assert ratio < 10.0, (
            f"readout magnitude drift too large: "
            f"|phase1|={mag_p1:.4f}, |phase2|={mag_p2:.4f}, ratio={ratio:.2f}")

        # decay must stay in [0, 1] for both (convex EMA invariant).
        assert (decay_p1 >= 0).all() and (decay_p1 <= 1).all()
        assert (decay_p2 >= 0).all() and (decay_p2 <= 1).all()


class TestGRPOGradientFlow:
    def test_log_prob_grad_reaches_z(self):
        """Verify gradient flows from log_prob back through z (the pg path)."""
        vq = _make_vqvae(action_dim=64)
        z = torch.randn(10, 8, requires_grad=True)
        codes = torch.randint(0, 8, (10, 2))
        log_pi = vq.rvq.log_prob(z, codes, tau=1.0)
        loss = -log_pi.mean()
        loss.backward()
        assert z.grad is not None
        assert z.grad.abs().sum() > 0

    def test_grpo_update_only_touches_modulator(self):
        """Simulate a phase 2 gradient pass: backprop through modulator+encoder,
        verify gradients reach only the modulator params."""
        config, mg, lm = _make_setup(BS=2, T=8)
        vq = _make_vqvae(action_dim=config.mod_out)

        # Freeze everything but modulator
        for p in mg.parameters():
            p.requires_grad = False
        for p in (mg.mod_w1, mg.mod_b1, mg.mod_w2, mg.mod_b2):
            p.requires_grad = True
        for p in vq.parameters():
            p.requires_grad = False

        # Fake mod_input and pretend codes
        BS, NC = 2, config.N_cells
        mod_input = torch.randn(BS, NC, config.mod_in, dtype=torch.bfloat16)
        # Run modulator
        raw_action = mg._modulator_forward(mod_input)
        action_flat = raw_action.reshape(BS * NC, -1).float()
        action_norm = vq.normalize(action_flat)
        z = vq.encoder(action_norm)
        codes = torch.randint(0, 8, (BS * NC, 2))
        log_pi = vq.rvq.log_prob(z, codes, tau=1.0)
        advantages = torch.randn(BS * NC)
        loss = -(advantages * log_pi).mean()
        loss.backward()

        # Modulator params should have grad
        assert mg.mod_w1.grad is not None
        assert mg.mod_w1.grad.abs().sum() > 0
        assert mg.mod_w2.grad is not None
        assert mg.mod_w2.grad.abs().sum() > 0

        # Other memory params should NOT have grad
        assert mg.state_w1.grad is None or mg.state_w1.grad.abs().sum() == 0
        assert mg.msg_w1.grad is None or mg.msg_w1.grad.abs().sum() == 0
        assert mg.inject_w.grad is None or mg.inject_w.grad.abs().sum() == 0

        # VQVAE params should NOT have grad
        for name, p in vq.named_parameters():
            assert p.grad is None or p.grad.abs().sum() == 0, f"{name} has grad"


class TestCollectModulatorAction:
    def test_fallback_snapshot_shape(self):
        """With _collecting_actions=False, returns a single end-of-chunk snapshot."""
        config, mg, lm = _make_setup(BS=2, T=16)
        H_mid = torch.randn(2, 16, config.D, dtype=torch.bfloat16)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        mg.forward_segment(H_mid, input_ids, lm)

        action = mg.collect_modulator_action()
        assert action is not None
        # Returns [1, BS, NC, mod_out] — single snapshot with leading events dim.
        assert action.shape == (1, 2, config.N_cells, config.mod_out)
        assert action.dtype == torch.float32

    def test_per_event_collection_shape(self):
        """With _collecting_actions=True, returns one entry per modulation event."""
        config, mg, lm = _make_setup(BS=2, T=16)
        H_mid = torch.randn(2, 16, config.D, dtype=torch.bfloat16)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))

        mg.start_action_collection()
        mg.forward_segment(H_mid, input_ids, lm)
        action = mg.collect_modulator_action()
        mg.stop_action_collection()

        assert action is not None
        expected_events = 16 // config.modulation_interval
        assert action.shape == (expected_events, 2, config.N_cells, config.mod_out)
        assert action.dtype == torch.float32
