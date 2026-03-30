"""Tests for v10 GNN memory graph."""

import torch
import pytest

from src.v10.config import V10Config
from src.v10.memory_graph import MemoryGraph, NeuronStep


@pytest.fixture
def tiny_config():
    return V10Config.tier_tiny()


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def graph(tiny_config, device):
    mg = MemoryGraph(tiny_config, device=device, dtype=torch.float32)
    mg.initialize_states(BS=2)
    return mg


class TestConfig:
    def test_derived_properties(self):
        cfg = V10Config.tier_a()
        assert cfg.C == 16
        assert cfg.D_cc == 128
        assert cfg.neurons_per_word == 64  # 2048 // 32
        assert cfg.num_words == 64         # 4096 // 64

    def test_tiny_derived_properties(self, tiny_config):
        # D_scan=64, D_neuron=8 -> neurons_per_word=8
        # N_neurons=64 -> num_words=8
        assert tiny_config.neurons_per_word == 8
        assert tiny_config.num_words == 8

    def test_validate_passes(self, tiny_config):
        tiny_config.validate()  # should not raise

    def test_validate_bad_divisibility(self):
        with pytest.raises(ValueError, match="divisible"):
            cfg = V10Config(D_scan=64, D_neuron=13)
            cfg.validate()


class TestForwardSegmentShapes:
    def test_forward_segment_shapes(self, graph, tiny_config):
        """word_states has correct shape [BS, T, num_words, D_scan]."""
        BS = 2
        T = tiny_config.T
        D_scan = tiny_config.D_scan
        num_words = tiny_config.num_words

        cc_signals = torch.randn(BS, T, D_scan)
        word_states = graph.forward_segment(cc_signals)

        assert word_states.shape == (BS, T, num_words, D_scan), (
            f"Expected {(BS, T, num_words, D_scan)}, got {word_states.shape}")

    def test_forward_segment_dtype(self, graph, tiny_config):
        """word_states inherits the runtime dtype."""
        BS = 2
        cc_signals = torch.randn(BS, tiny_config.T, tiny_config.D_scan)
        word_states = graph.forward_segment(cc_signals)
        assert word_states.dtype == torch.float32


class TestWordStatesShape:
    def test_word_states_shape(self, tiny_config, device):
        """Each word_state[t] = neurons grouped into num_words of D_scan dims."""
        BS = 3
        mg = MemoryGraph(tiny_config, device=device, dtype=torch.float32)
        mg.initialize_states(BS=BS)

        cc_signals = torch.randn(BS, tiny_config.T, tiny_config.D_scan)
        word_states = mg.forward_segment(cc_signals)

        # Check per-timestep shapes
        for t in range(tiny_config.T):
            ws_t = word_states[:, t]  # [BS, num_words, D_scan]
            assert ws_t.shape == (BS, tiny_config.num_words, tiny_config.D_scan)

    def test_word_states_vary_across_time(self, graph, tiny_config):
        """word_states should differ across timesteps (sequential simulation)."""
        cc_signals = torch.randn(2, tiny_config.T, tiny_config.D_scan)
        word_states = graph.forward_segment(cc_signals)

        # First and last timestep should differ
        diff = (word_states[:, 0] - word_states[:, -1]).abs().max()
        assert diff > 1e-6, "word_states should vary across timesteps"


class TestStructuralDecayBoundsH:
    def test_structural_decay_bounds_h(self, tiny_config, device):
        """h stays bounded due to structural decay (convex combo of h and tanh)."""
        BS = 2
        mg = MemoryGraph(tiny_config, device=device, dtype=torch.float32)
        mg.initialize_states(BS=BS)

        # Start with large h to test bounding
        mg.h = torch.ones_like(mg.h) * 5.0

        cc_signals = torch.randn(BS, tiny_config.T, tiny_config.D_scan)
        mg.forward_segment(cc_signals)

        # After a segment with structural decay, h should be bounded
        # Each step: h = decay * h + (1-decay) * tanh(update)
        # tanh output in [-1,1], decay in [0,1], so |h| shrinks toward [-1,1]
        h_max = mg.h.abs().max().item()
        # After T steps of decay, h should be significantly smaller than initial 5.0
        # (exact bound depends on decay rate, but should be well below 5)
        assert h_max < 5.0, (
            f"h should decrease from initial 5.0, got max |h| = {h_max}")

    def test_h_bounded_after_many_segments(self, tiny_config, device):
        """h stays bounded even after multiple segments."""
        BS = 2
        mg = MemoryGraph(tiny_config, device=device, dtype=torch.float32)
        mg.initialize_states(BS=BS)

        for _ in range(5):
            cc_signals = torch.randn(BS, tiny_config.T, tiny_config.D_scan)
            mg.forward_segment(cc_signals)

        h_max = mg.h.abs().max().item()
        # After 5 segments, h should be well-bounded
        assert h_max < 3.0, f"h should be bounded, got max |h| = {h_max}"


class TestPlasticityChangesConnections:
    def test_plasticity_changes_connections(self, device):
        """rewire_connections() modifies conn_indices."""
        cfg = V10Config.tier_tiny(structural_plasticity=True)
        mg = MemoryGraph(cfg, device=device, dtype=torch.float32)
        mg.initialize_states(BS=2)

        # Run a segment to produce activity (needed for phi)
        cc_signals = torch.randn(2, cfg.T, cfg.D_scan)
        mg.forward_segment(cc_signals)

        # Manually set co_activation_ema so rewiring has signal
        N = cfg.N_neurons
        mg.co_activation_ema = torch.randn(N, N, device=device)
        mg.co_activation_ema = (
            mg.co_activation_ema + mg.co_activation_ema.t()) / 2
        mg._co_activation_ready = True

        # Snapshot connections
        old_conn = mg.conn_indices.clone()

        mg.rewire_connections()

        # Some connections should have changed
        changed = (mg.conn_indices != old_conn).sum().item()
        assert changed > 0, "rewire_connections() should modify some connections"

    def test_plasticity_rebuilds_edge_index(self, device):
        """After rewiring, edge_index is consistent with conn_indices."""
        cfg = V10Config.tier_tiny(structural_plasticity=True)
        mg = MemoryGraph(cfg, device=device, dtype=torch.float32)
        mg.initialize_states(BS=2)

        # Set up co_activation and rewire
        N = cfg.N_neurons
        mg.co_activation_ema = torch.randn(N, N, device=device)
        mg._co_activation_ready = True

        mg.rewire_connections()

        # Verify edge_index matches conn_indices
        expected_edge_index = mg._build_edge_index()
        assert torch.equal(mg.edge_index, expected_edge_index), (
            "edge_index should be rebuilt after rewiring")


class TestNeuronStep:
    def test_neuron_step_shapes(self, tiny_config):
        """NeuronStep returns correct shapes."""
        D = tiny_config.D_neuron
        D_id = tiny_config.D_id
        K = tiny_config.K_connections
        N = tiny_config.N_neurons
        BS = 2

        step = NeuronStep(D=D, D_id=D_id, K=K,
                          H_state=32, H_msg=16, H_mod=16)

        h = torch.randn(BS, N, D)
        msgs = torch.randn(BS, N, D)
        inject = torch.randn(BS, N, D)
        identity = torch.randn(N, D_id)
        w_conn = torch.randn(BS, N, K)
        hebbian = torch.randn(BS, N, K)

        # Build edge_index
        conn_indices = torch.randint(0, N, (N, K))
        src = conn_indices.reshape(-1)
        tgt = torch.arange(N).unsqueeze(1).expand(-1, K).reshape(-1)
        edge_index = torch.stack([src, tgt])

        h_new, msgs_new, w_conn_new, decay, id_new = step(
            h, msgs, inject, identity, edge_index, w_conn, hebbian)

        assert h_new.shape == (BS, N, D)
        assert msgs_new.shape == (BS, N, D)
        assert w_conn_new.shape == (BS, N, K)
        assert decay.shape == (BS, N, 1)
        assert id_new.shape == (N, D_id)


class TestGradientFlow:
    def test_gradient_flows_to_shared_mlps(self, graph, tiny_config):
        """Loss on word_states produces gradients for shared MLP params."""
        cc_signals = torch.randn(2, tiny_config.T, tiny_config.D_scan)
        word_states = graph.forward_segment(cc_signals)

        loss = word_states.sum()
        loss.backward()

        # Check state_mlp gets gradient
        for name, p in graph.neuron_step.state_mlp.named_parameters():
            assert p.grad is not None, f"state_mlp.{name} has no gradient"
            assert p.grad.abs().max() > 0, f"state_mlp.{name} has zero gradient"

        # Check msg_mlp gets gradient
        for name, p in graph.neuron_step.msg_mlp.named_parameters():
            assert p.grad is not None, f"msg_mlp.{name} has no gradient"

    def test_gradient_flows_to_identity(self, graph, tiny_config):
        """Loss on word_states produces gradient for identity parameter."""
        cc_signals = torch.randn(2, tiny_config.T, tiny_config.D_scan)
        word_states = graph.forward_segment(cc_signals)

        loss = word_states.sum()
        loss.backward()

        assert graph.identity.grad is not None, "identity has no gradient"
        assert graph.identity.grad.abs().max() > 0, "identity has zero gradient"
