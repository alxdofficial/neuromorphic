"""Tests for the Memory Graph — per-token neuron dynamics + sparse message passing.

Includes numerical reference tests that serve as ground truth for verifying
optimized implementations (Triton kernels, CUDA graphs, etc).
"""

import torch
import pytest
import sys
sys.path.insert(0, ".")

from src.v8.config import V8Config
from src.v8.memory_graph import MemoryGraph, _cpu_scan

BS = 2


def make_tiny(**overrides):
    cfg = V8Config.tier_tiny(**overrides)
    cfg.validate()
    return cfg


class TestDiagonalScan:
    def test_output_shape(self):
        B, T, D = 4, 16, 8
        decay_logit = torch.randn(B, D)
        b = torch.randn(B, T, D)
        out = _cpu_scan(decay_logit, b)
        assert out.shape == (B, T, D)

    def test_with_carry(self):
        B, T, D = 2, 8, 4
        decay_logit = torch.randn(B, D)
        b = torch.randn(B, T, D)
        h0 = torch.randn(B, D)
        out = _cpu_scan(decay_logit, b, h0)
        assert out.shape == (B, T, D)

        # Verify first step: h[0] = sigmoid(decay) * h0 + b[0]
        a = torch.sigmoid(decay_logit)
        expected_h0 = a * h0 + b[:, 0]
        torch.testing.assert_close(out[:, 0], expected_h0, atol=1e-5, rtol=1e-4)

    def test_sequential_equivalence(self):
        B, T, D = 2, 16, 4
        decay_logit = torch.randn(B, D)
        b = torch.randn(B, T, D)
        h0 = torch.randn(B, D)

        # Parallel scan
        out_par = _cpu_scan(decay_logit, b, h0)

        # Sequential reference
        a = torch.sigmoid(decay_logit)
        h = h0
        out_seq = []
        for t in range(T):
            h = a * h + b[:, t]
            out_seq.append(h)
        out_seq = torch.stack(out_seq, dim=1)

        torch.testing.assert_close(out_par, out_seq, atol=1e-5, rtol=1e-4)


class TestMemoryGraphInit:
    def test_initialize(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        assert mg.is_initialized()
        assert mg.primitives.shape == (BS, cfg.N_neurons, cfg.D_mem)
        assert mg.h.shape == (BS, cfg.N_neurons, cfg.D_mem)
        assert mg.prev_messages.shape == (BS, cfg.N_neurons, cfg.D_mem)
        assert mg.conn_weights.shape == (BS, cfg.N_neurons, cfg.K_connections)
        assert mg.co_activation_ema.shape == (cfg.N_neurons, cfg.N_neurons)
        assert mg.co_activation_ema.shape == (cfg.N_neurons, cfg.N_neurons)

    def test_conn_weights_l1_normalized(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        # Each neuron's conn_weights should sum to 1.0 in abs value
        l1 = mg.conn_weights.abs().sum(dim=-1)  # [BS, N]
        torch.testing.assert_close(l1, torch.ones_like(l1), atol=1e-2, rtol=1e-2)

    def test_connectivity(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        assert mg.conn_indices.shape == (cfg.N_neurons, cfg.K_connections)
        assert mg.conn_mask.shape == (cfg.N_neurons, cfg.K_connections)
        # No self-connections
        for j in range(cfg.N_neurons):
            active = mg.conn_indices[j][mg.conn_mask[j]]
            assert j not in active.tolist()


class TestMemoryGraphForward:
    def test_forward_segment_shape(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        out = mg.forward_segment(cc)
        assert out.shape == (BS, cfg.action_every, cfg.C, cfg.D_mem)

    def test_forward_segment_finite(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        out = mg.forward_segment(cc)
        assert torch.isfinite(out).all()

    def test_multiple_segments_stable(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        for _ in range(10):
            cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem) * 0.1
            out = mg.forward_segment(cc)
            assert torch.isfinite(out).all()
            assert out.abs().max() < 100  # no explosion

    def test_state_persistence(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)

        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        mg.forward_segment(cc)

        # Internal state and messages should be nonzero after processing
        assert mg.h.abs().sum() > 0
        assert mg.prev_messages.abs().sum() > 0


class TestMemoryGraphActions:
    def test_apply_actions(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        prim_before = mg.primitives.clone()

        d_prim = torch.randn(BS, cfg.N_neurons, cfg.D_mem) * 0.01
        d_conn = torch.randn(BS, cfg.N_neurons, cfg.K_connections) * 0.01
        d_decay = torch.randn(BS, cfg.N_neurons) * 0.01
        mg.apply_actions(d_prim, d_conn, d_decay)

        assert not torch.equal(mg.primitives, prim_before)

    def test_get_neuron_obs(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        obs = mg.get_neuron_obs()
        assert obs.shape == (BS, cfg.N_neurons, mg.obs_dim)
        assert torch.isfinite(obs).all()


class TestMemoryGraphPlasticity:
    def test_co_activation_updates(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        phi_before = mg.co_activation_ema.clone()

        # Strong CC signals to ensure neurons fire above threshold
        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem) * 5.0
        mg.forward_segment(cc, update_co_activation=True)

        # Co-activation matrix should have changed after a segment
        assert torch.isfinite(mg.co_activation_ema).all()
        assert not torch.equal(mg.co_activation_ema, phi_before)

    def test_structural_plasticity(self):
        """Co-activation-based plasticity: anti-correlated connections get pruned."""
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)

        N = cfg.N_neurons
        # Set up co-activation matrix with some anti-correlated pairs
        mg.co_activation_ema = torch.randn(N, N) * 0.1
        mg._co_activation_ready = True
        # Make neuron 0's connection to conn_indices[0, 0] strongly anti-correlated
        target = mg.conn_indices[0, 0].item()
        mg.co_activation_ema[0, target] = -0.5

        indices_before = mg.conn_indices[0].clone()
        mg.structural_plasticity()

        # The anti-correlated connection should have been pruned and replaced
        indices_after = mg.conn_indices[0]
        # The anti-correlated target should no longer be in neuron 0's connections
        assert target not in indices_after.tolist() or \
            not torch.equal(indices_before, indices_after), \
            "Anti-correlated connection should be pruned and rewired"

        # Weights should still be L1-normalized after plasticity
        l1 = mg.conn_weights.abs().sum(dim=-1)
        torch.testing.assert_close(l1, torch.ones_like(l1), atol=1e-2, rtol=1e-2)


class TestMemoryGraphReset:
    def test_reset_streams(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)

        # Step to build up state
        for _ in range(3):
            cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
            mg.forward_segment(cc)

        h_before = mg.h.clone()
        msg_before = mg.prev_messages.clone()
        prim_before = mg.primitives.clone()
        conn_before = mg.conn_weights.clone()

        # Reset stream 0 only
        mask = torch.tensor([True, False])
        mg.reset_streams(mask)

        # Dynamic state: stream 0 zeroed, stream 1 unchanged
        assert mg.h[0].abs().sum() == 0
        assert mg.prev_messages[0].abs().sum() == 0
        torch.testing.assert_close(mg.h[1], h_before[1])
        torch.testing.assert_close(mg.prev_messages[1], msg_before[1])

        # Structural state: preserved for ALL streams (including stream 0)
        torch.testing.assert_close(mg.primitives, prim_before)
        torch.testing.assert_close(mg.conn_weights, conn_before)


class TestNeuronDynamicsReference:
    """Numerical reference tests for per-token neuron dynamics.

    These serve as ground truth for verifying optimized implementations
    (Triton kernels, CUDA graphs, etc). Any optimized forward_segment
    must produce identical output to these reference computations.
    """

    def _reference_forward(self, mg, cc_signals, eot_mask=None):
        """Pure Python reference implementation of per-token neuron dynamics.

        Args:
            mg: initialized MemoryGraph
            cc_signals: [BS, T_seg, C, D]
            eot_mask: [BS, T_seg] bool or None

        Returns:
            output: [BS, T_seg, C, D]
            final_h: [BS, N, D]
            final_msg: [BS, N, D]
        """
        BS, T_seg, C, D = cc_signals.shape
        N = mg.config.N_neurons

        # CC signals enter raw (no normalization — RMSNorm on h absorbs scale)

        decay = torch.sigmoid(mg.decay_logit).unsqueeze(-1)  # [BS, N, 1]
        one_minus_decay = 1.0 - decay
        A = mg._build_adjacency()

        h = mg.h.clone()
        prev_msg = mg.prev_messages.clone()
        output = torch.empty(BS, T_seg, C, D)

        for t in range(T_seg):
            # 1. Receive
            received = torch.bmm(A, prev_msg)
            received[:, :C] = received[:, :C] + cc_signals[:, t]

            # 2. Integrate
            if eot_mask is not None and eot_mask[:, t].any():
                eot_t = eot_mask[:, t].view(BS, 1, 1).float()
                d_t = decay * (1.0 - eot_t)
                omd_t = 1.0 - d_t
                h = d_t * h + omd_t * received
            else:
                h = decay * h + one_minus_decay * received

            # 3. Message
            prev_msg = torch.tanh(h * mg.primitives)
            output[:, t] = prev_msg[:, :C]

        return output, h, prev_msg

    def test_single_token_reference(self):
        """Verify forward_segment matches reference for a single token."""
        cfg = make_tiny(action_every=1)
        cfg.validate()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize(BS)

        cc = torch.randn(BS, 1, cfg.C, cfg.D_mem)

        # Reference
        ref_out, ref_h, ref_msg = self._reference_forward(mg, cc)

        # Actual
        actual_out = mg.forward_segment(cc)

        torch.testing.assert_close(actual_out, ref_out, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(mg.h, ref_h, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(mg.prev_messages, ref_msg, atol=1e-5, rtol=1e-4)

    def test_multi_token_reference(self):
        """Verify forward_segment matches reference for full segment."""
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize(BS)

        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem) * 0.5

        ref_out, ref_h, ref_msg = self._reference_forward(mg, cc)
        actual_out = mg.forward_segment(cc)

        torch.testing.assert_close(actual_out, ref_out, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(mg.h, ref_h, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(mg.prev_messages, ref_msg, atol=1e-5, rtol=1e-4)

    def test_multi_segment_reference(self):
        """Verify state carries correctly across multiple segments."""
        cfg = make_tiny()
        mg_ref = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg_ref.initialize(BS)
        mg_actual = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg_actual.initialize(BS)

        # Copy identical init state
        mg_actual.h = mg_ref.h.clone()
        mg_actual.prev_messages = mg_ref.prev_messages.clone()
        mg_actual.primitives = mg_ref.primitives.clone()
        mg_actual.decay_logit = mg_ref.decay_logit.clone()
        mg_actual.conn_weights = mg_ref.conn_weights.clone()
        mg_actual.conn_indices = mg_ref.conn_indices.clone()
        mg_actual.conn_mask = mg_ref.conn_mask.clone()
        mg_actual._adjacency_dirty = True

        # Run 3 segments
        for _ in range(3):
            cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem) * 0.3

            ref_out, ref_h, ref_msg = self._reference_forward(mg_ref, cc)
            mg_ref.h = ref_h
            mg_ref.prev_messages = ref_msg

            actual_out = mg_actual.forward_segment(cc)

            torch.testing.assert_close(actual_out, ref_out, atol=1e-5, rtol=1e-4)

        torch.testing.assert_close(mg_actual.h, mg_ref.h, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(mg_actual.prev_messages, mg_ref.prev_messages,
                                   atol=1e-5, rtol=1e-4)

    def test_eot_kills_state(self):
        """Verify EOT mask zeros out hidden state at boundary positions."""
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize(BS)

        # Build up state
        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem) * 0.5
        mg.forward_segment(cc)
        assert mg.h.abs().sum() > 0

        # Now run with EOT at position 0 for batch element 0
        cc2 = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem) * 0.5
        eot_mask = torch.zeros(BS, cfg.action_every, dtype=torch.bool)
        eot_mask[0, 0] = True  # batch 0, token 0: previous was EOT

        ref_out, ref_h, ref_msg = self._reference_forward(mg, cc2, eot_mask)
        actual_out = mg.forward_segment(cc2, eot_mask=eot_mask)

        torch.testing.assert_close(actual_out, ref_out, atol=1e-5, rtol=1e-4)

    def test_port_neurons_receive_cc_signal(self):
        """Verify port neurons receive CC signal additively on top of graph messages."""
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize(BS)

        # Set prev_messages to known values so graph messages are nonzero
        mg.prev_messages = torch.randn(BS, cfg.N_neurons, cfg.D_mem) * 0.1

        cc = torch.randn(BS, 1, cfg.C, cfg.D_mem)
        A = mg._build_adjacency()
        decay = torch.sigmoid(mg.decay_logit).unsqueeze(-1)

        # Manual single step (CC enters raw, no normalization)
        received = torch.bmm(A, mg.prev_messages)
        graph_msg_at_port = received[:, :cfg.C].clone()
        received[:, :cfg.C] += cc[:, 0]

        h_expected = decay * mg.h + (1 - decay) * received
        msg_expected = torch.tanh(h_expected * mg.primitives)

        # Port output should reflect both graph messages AND CC signal
        out = mg.forward_segment(cc)
        expected_port = msg_expected[:, :cfg.C]
        torch.testing.assert_close(out[:, 0], expected_port, atol=1e-5, rtol=1e-4)

    def test_signal_propagation_through_graph(self):
        """Verify signals propagate from port neurons to non-port neurons over time."""
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize(BS)

        # All state starts at zero
        mg.h.zero_()
        mg.prev_messages.zero_()

        # Strong CC signal
        cc = torch.ones(BS, cfg.action_every, cfg.C, cfg.D_mem)
        mg.forward_segment(cc)

        # After a full segment, non-port neurons should have nonzero state
        # (received signal via graph connectivity from port neurons)
        non_port_h = mg.h[:, cfg.C:]
        assert non_port_h.abs().sum() > 0, \
            "Non-port neurons should receive signal via graph message passing"

    def test_primitives_modulate_output(self):
        """Verify primitives affect outgoing messages (message = tanh(h * prim))."""
        cfg = make_tiny()
        mg1 = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg1.initialize(BS)
        mg2 = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg2.initialize(BS)

        # Same state, different primitives
        mg2.h = mg1.h.clone()
        mg2.prev_messages = mg1.prev_messages.clone()
        mg2.decay_logit = mg1.decay_logit.clone()
        mg2.conn_weights = mg1.conn_weights.clone()
        mg2.conn_indices = mg1.conn_indices.clone()
        mg2.conn_mask = mg1.conn_mask.clone()

        mg2.primitives = mg1.primitives * 2.0  # different primitives

        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem) * 0.5
        out1 = mg1.forward_segment(cc)
        out2 = mg2.forward_segment(cc.clone())

        # Outputs should differ because primitives differ
        assert not torch.allclose(out1, out2, atol=1e-6), \
            "Different primitives should produce different outputs"

    def test_decay_controls_persistence(self):
        """Verify high decay retains state magnitude, low decay forgets.

        With natural h dynamics (no RMSNorm), high decay neurons retain
        more of the original signal magnitude after new input overwrites.
        """
        cfg = make_tiny()
        mg_high = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg_high.initialize(BS)
        mg_low = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg_low.initialize(BS)

        # Copy identical state
        mg_low.h = mg_high.h.clone()
        mg_low.prev_messages = mg_high.prev_messages.clone()
        mg_low.primitives = mg_high.primitives.clone()
        mg_low.conn_indices = mg_high.conn_indices.clone()
        mg_low.conn_mask = mg_high.conn_mask.clone()

        # Zero connectivity so only direct CC input matters
        mg_high.conn_weights = torch.zeros_like(mg_high.conn_weights)
        mg_low.conn_weights = torch.zeros_like(mg_low.conn_weights)
        mg_high._adjacency_dirty = True
        mg_low._adjacency_dirty = True

        # Set decay: high=0.99, low=0.01
        mg_high.decay_logit.fill_(4.6)
        mg_low.decay_logit.fill_(-4.6)

        # Inject strong signal into port neurons
        cc_signal = torch.ones(BS, 1, cfg.C, cfg.D_mem) * 2.0
        mg_high.forward_segment(cc_signal)
        mg_low.forward_segment(cc_signal.clone())

        # Record h after signal
        h_after_high = mg_high.h[:, :cfg.C].clone()

        # Run with zero input — decay matters
        cc_zero = torch.zeros(BS, cfg.action_every, cfg.C, cfg.D_mem)
        mg_high.forward_segment(cc_zero)
        mg_low.forward_segment(cc_zero.clone())

        # High decay should retain more of original signal direction
        def cos_sim(a, b):
            return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-8)

        high_sim = cos_sim(mg_high.h[:, :cfg.C], h_after_high).mean().item()
        low_sim = cos_sim(mg_low.h[:, :cfg.C], h_after_high).mean().item()
        assert high_sim > low_sim, \
            f"High decay should retain signal better: high={high_sim:.4f} vs low={low_sim:.4f}"
