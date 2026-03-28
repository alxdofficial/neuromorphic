"""Tests for v9-backprop Triton kernels.

Tests the fused dendritic gather kernel (forward + backward) against
the Python reference implementation.
"""

import torch
import pytest
import sys
sys.path.insert(0, ".")

from src.v8.config import V8Config


def make_tiny(**overrides):
    cfg = V8Config.tier_tiny(**overrides)
    cfg.validate()
    return cfg


# Skip all tests if no CUDA or no Triton
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton requires CUDA"
)


def _try_import_triton():
    try:
        from src.v8.triton_kernels import fused_dendritic_gather
        return fused_dendritic_gather
    except ImportError:
        pytest.skip("Triton not available")


def _python_dendritic_gather(prev_msg, conn_indices, w_conn_sig,
                              branch_w, group_w, cfg):
    """Python reference: gather + weight + dendritic tree."""
    neighbor_msgs = prev_msg[:, conn_indices]  # [BS, N, K, D]
    weighted = w_conn_sig.unsqueeze(-1) * neighbor_msgs

    BS, N, K, D = weighted.shape
    bsz = cfg.dendrite_branch_size
    nb = K // bsz
    bpg = min(4, nb)
    ng = max(1, nb // bpg)
    bpg = nb // ng
    n_tree = ng * bpg * bsz

    tree_msgs = weighted[:, :, :n_tree].view(BS, N, nb, bsz, D)
    branch_out = torch.tanh(
        (tree_msgs * branch_w.unsqueeze(0)).sum(dim=3))
    branch_grouped = branch_out.view(BS, N, ng, bpg, D)
    group_out = torch.tanh(
        (branch_grouped * group_w.unsqueeze(0)).sum(dim=3))
    received = group_out.mean(dim=2)

    if n_tree < K:
        leftover = weighted[:, :, n_tree:].sum(dim=2)
        tree_frac = n_tree / K
        received = tree_frac * received + (1 - tree_frac) * leftover
    return received


def _make_test_tensors(cfg, BS=2, device='cuda'):
    """Create test tensors matching memory graph structure."""
    N = cfg.N_neurons
    K = cfg.K_connections
    D = cfg.D_neuron

    prev_msg = torch.randn(BS, N, D, device=device, dtype=torch.float32)
    w_conn_sig = torch.sigmoid(torch.randn(BS, N, K, device=device))

    # Random connectivity
    all_idx = torch.arange(N, device=device)
    scores = torch.rand(N, N, device=device)
    scores[all_idx, all_idx] = -float('inf')
    K_actual = min(K, N - 1)
    conn_indices = torch.zeros(N, K, dtype=torch.long, device=device)
    conn_indices[:, :K_actual] = scores.topk(K_actual, dim=1).indices
    conn_indices, _ = conn_indices.sort(dim=-1)

    # Dendritic tree weights
    bsz = cfg.dendrite_branch_size
    nb = K // bsz
    bpg = min(4, nb)
    ng = max(1, nb // bpg)
    bpg = nb // ng

    branch_w = torch.randn(N, nb, bsz, D, device=device) * 0.1
    group_w = torch.randn(N, ng, bpg, D, device=device) * 0.1

    return prev_msg, conn_indices, w_conn_sig, branch_w, group_w, bsz, bpg, ng


class TestFusedDendriticGatherForward:
    def test_output_shape(self):
        fused_gather = _try_import_triton()
        cfg = make_tiny()
        BS = 2
        tensors = _make_test_tensors(cfg, BS)
        prev_msg, conn_indices, w_conn_sig, branch_w, group_w, bsz, bpg, ng = tensors

        received = fused_gather(
            prev_msg, conn_indices, w_conn_sig,
            branch_w, group_w, bsz, bpg, ng,
            use_dendrite=True)

        assert received.shape == (BS, cfg.N_neurons, cfg.D_neuron)

    def test_matches_python_reference(self):
        fused_gather = _try_import_triton()
        cfg = make_tiny()
        BS = 2
        tensors = _make_test_tensors(cfg, BS)
        prev_msg, conn_indices, w_conn_sig, branch_w, group_w, bsz, bpg, ng = tensors

        triton_out = fused_gather(
            prev_msg, conn_indices, w_conn_sig,
            branch_w, group_w, bsz, bpg, ng,
            use_dendrite=True)

        python_out = _python_dendritic_gather(
            prev_msg, conn_indices, w_conn_sig,
            branch_w, group_w, cfg)

        torch.testing.assert_close(triton_out, python_out, atol=1e-4, rtol=1e-4)

    def test_no_dendrite_matches_reference(self):
        fused_gather = _try_import_triton()
        cfg = make_tiny()
        BS = 2
        tensors = _make_test_tensors(cfg, BS)
        prev_msg, conn_indices, w_conn_sig, branch_w, group_w, bsz, bpg, ng = tensors

        triton_out = fused_gather(
            prev_msg, conn_indices, w_conn_sig,
            None, None, 1, 1, 1,
            use_dendrite=False)

        # Python reference: simple weighted sum
        neighbor_msgs = prev_msg[:, conn_indices]
        python_out = (w_conn_sig.unsqueeze(-1) * neighbor_msgs).sum(dim=2)

        torch.testing.assert_close(triton_out, python_out, atol=1e-4, rtol=1e-4)

    def test_output_finite(self):
        fused_gather = _try_import_triton()
        cfg = make_tiny()
        BS = 2
        tensors = _make_test_tensors(cfg, BS)
        prev_msg, conn_indices, w_conn_sig, branch_w, group_w, bsz, bpg, ng = tensors

        received = fused_gather(
            prev_msg, conn_indices, w_conn_sig,
            branch_w, group_w, bsz, bpg, ng,
            use_dendrite=True)

        assert torch.isfinite(received).all()


class TestFusedDendriticGatherBackward:
    def test_backward_runs(self):
        fused_gather = _try_import_triton()
        cfg = make_tiny()
        BS = 2
        tensors = _make_test_tensors(cfg, BS)
        prev_msg, conn_indices, w_conn_sig, branch_w, group_w, bsz, bpg, ng = tensors

        prev_msg.requires_grad_(True)
        w_conn_sig.requires_grad_(True)
        branch_w.requires_grad_(True)
        group_w.requires_grad_(True)

        received = fused_gather(
            prev_msg, conn_indices, w_conn_sig,
            branch_w, group_w, bsz, bpg, ng,
            use_dendrite=True)

        loss = received.sum()
        loss.backward()

        assert prev_msg.grad is not None
        assert w_conn_sig.grad is not None
        assert branch_w.grad is not None
        assert group_w.grad is not None

    def test_grad_matches_python(self):
        """Triton backward gradients should match Python autograd."""
        fused_gather = _try_import_triton()
        cfg = make_tiny()
        BS = 2
        tensors = _make_test_tensors(cfg, BS)
        prev_msg, conn_indices, w_conn_sig, branch_w, group_w, bsz, bpg, ng = tensors

        # Python reference with autograd
        pm_py = prev_msg.clone().detach().requires_grad_(True)
        wc_py = w_conn_sig.clone().detach().requires_grad_(True)
        bw_py = branch_w.clone().detach().requires_grad_(True)
        gw_py = group_w.clone().detach().requires_grad_(True)

        py_out = _python_dendritic_gather(
            pm_py, conn_indices, wc_py, bw_py, gw_py, cfg)
        py_out.sum().backward()

        # Triton
        pm_tr = prev_msg.clone().detach().requires_grad_(True)
        wc_tr = w_conn_sig.clone().detach().requires_grad_(True)
        bw_tr = branch_w.clone().detach().requires_grad_(True)
        gw_tr = group_w.clone().detach().requires_grad_(True)

        tr_out = fused_gather(
            pm_tr, conn_indices, wc_tr,
            bw_tr, gw_tr, bsz, bpg, ng,
            use_dendrite=True)
        tr_out.sum().backward()

        # Compare gradients
        torch.testing.assert_close(
            wc_tr.grad, wc_py.grad, atol=1e-3, rtol=1e-3,
            msg="w_conn_sig grad mismatch")
        torch.testing.assert_close(
            bw_tr.grad, bw_py.grad, atol=1e-3, rtol=1e-3,
            msg="branch_w grad mismatch")
        torch.testing.assert_close(
            gw_tr.grad, gw_py.grad, atol=1e-3, rtol=1e-3,
            msg="group_w grad mismatch")
        torch.testing.assert_close(
            pm_tr.grad, pm_py.grad, atol=1e-3, rtol=1e-3,
            msg="prev_msg grad mismatch")

    def test_no_dendrite_grad_matches(self):
        """Backward without dendritic tree should match Python."""
        fused_gather = _try_import_triton()
        cfg = make_tiny()
        BS = 2
        tensors = _make_test_tensors(cfg, BS)
        prev_msg, conn_indices, w_conn_sig, branch_w, group_w, bsz, bpg, ng = tensors

        # Python reference
        pm_py = prev_msg.clone().detach().requires_grad_(True)
        wc_py = w_conn_sig.clone().detach().requires_grad_(True)
        neighbor_msgs = pm_py[:, conn_indices]
        py_out = (wc_py.unsqueeze(-1) * neighbor_msgs).sum(dim=2)
        py_out.sum().backward()

        # Triton
        pm_tr = prev_msg.clone().detach().requires_grad_(True)
        wc_tr = w_conn_sig.clone().detach().requires_grad_(True)
        tr_out = fused_gather(
            pm_tr, conn_indices, wc_tr,
            None, None, 1, 1, 1,
            use_dendrite=False)
        tr_out.sum().backward()

        torch.testing.assert_close(
            wc_tr.grad, wc_py.grad, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(
            pm_tr.grad, pm_py.grad, atol=1e-3, rtol=1e-3)

    def test_grad_matches_with_leftover(self):
        """K not divisible by branch_size → leftover connections.
        Backward must scale tree portion by tree_frac."""
        fused_gather = _try_import_triton()
        # K=10, branch_size=4 → 2 branches, 1 group, 2 leftover connections
        cfg = make_tiny(K_connections=10, dendrite_branch_size=4,
                        N_mem_neurons=16)
        BS = 2
        tensors = _make_test_tensors(cfg, BS)
        prev_msg, conn_indices, w_conn_sig, branch_w, group_w, bsz, bpg, ng = tensors

        # Python reference
        pm_py = prev_msg.clone().detach().requires_grad_(True)
        wc_py = w_conn_sig.clone().detach().requires_grad_(True)
        bw_py = branch_w.clone().detach().requires_grad_(True)
        gw_py = group_w.clone().detach().requires_grad_(True)
        py_out = _python_dendritic_gather(
            pm_py, conn_indices, wc_py, bw_py, gw_py, cfg)
        py_out.sum().backward()

        # Triton
        pm_tr = prev_msg.clone().detach().requires_grad_(True)
        wc_tr = w_conn_sig.clone().detach().requires_grad_(True)
        bw_tr = branch_w.clone().detach().requires_grad_(True)
        gw_tr = group_w.clone().detach().requires_grad_(True)
        tr_out = fused_gather(
            pm_tr, conn_indices, wc_tr,
            bw_tr, gw_tr, bsz, bpg, ng,
            use_dendrite=True)
        tr_out.sum().backward()

        torch.testing.assert_close(
            tr_out, py_out, atol=1e-4, rtol=1e-4,
            msg="forward mismatch with leftover")
        torch.testing.assert_close(
            wc_tr.grad, wc_py.grad, atol=1e-3, rtol=1e-3,
            msg="w_conn_sig grad mismatch with leftover")
        torch.testing.assert_close(
            bw_tr.grad, bw_py.grad, atol=1e-3, rtol=1e-3,
            msg="branch_w grad mismatch with leftover")
        torch.testing.assert_close(
            gw_tr.grad, gw_py.grad, atol=1e-3, rtol=1e-3,
            msg="group_w grad mismatch with leftover")
        torch.testing.assert_close(
            pm_tr.grad, pm_py.grad, atol=1e-3, rtol=1e-3,
            msg="prev_msg grad mismatch with leftover")


class TestFusedNeuronStep:
    """Test the fully fused step kernel against Python reference."""

    def _try_import(self):
        try:
            from src.v8.triton_kernels import fused_neuron_step
            return fused_neuron_step
        except ImportError:
            pytest.skip("Triton not available")

    def _python_step(self, h, prev_msg, inject, conn_indices, w_conn_sig,
                     decay, primitives, neuron_id,
                     branch_w, group_w, state_w1, state_b1, state_w2, state_b2,
                     msg_w1, msg_b1, msg_w2, msg_b2, hebbian, cfg):
        """Python reference for one step."""
        BS, N, D = h.shape
        K = conn_indices.shape[1]

        # Gather + weight + dendritic
        neighbor_msgs = prev_msg[:, conn_indices]
        weighted = w_conn_sig.unsqueeze(-1) * neighbor_msgs

        bsz = cfg.dendrite_branch_size
        nb = K // bsz
        bpg = min(4, nb)
        ng = max(1, nb // bpg)
        bpg = nb // ng
        n_tree = ng * bpg * bsz

        tree_msgs = weighted[:, :, :n_tree].view(BS, N, nb, bsz, D)
        branch_out = torch.tanh(
            (tree_msgs * branch_w.unsqueeze(0)).sum(dim=3))
        branch_grouped = branch_out.view(BS, N, ng, bpg, D)
        group_out = torch.tanh(
            (branch_grouped * group_w.unsqueeze(0)).sum(dim=3))
        received = group_out.mean(dim=2)
        if n_tree < K:
            leftover = weighted[:, :, n_tree:].sum(dim=2)
            tf = n_tree / K
            received = tf * received + (1 - tf) * leftover

        input_vec = received + inject

        # State MLP (w1 transposed: [N, H, I])
        x_s = torch.cat([input_vec, h, decay.unsqueeze(-1)], dim=-1)
        sh = torch.einsum('bni,nhi->bnh', x_s, state_w1) + state_b1
        sh = torch.tanh(sh)
        h_new = torch.einsum('bnh,nhd->bnd', sh, state_w2) + state_b2
        h_new = torch.tanh(h_new)

        # Message MLP (w1 transposed: [N, H, I])
        x_m = torch.cat([h_new, primitives], dim=-1)
        mh = torch.einsum('bni,nhi->bnh', x_m, msg_w1) + msg_b1
        mh = torch.tanh(mh)
        msg = torch.einsum('bnh,nhd->bnd', mh, msg_w2) + msg_b2
        msg = torch.tanh(msg)

        # Neuron ID
        msg = msg + neuron_id

        # Hebbian
        msg_mag = msg.norm(dim=-1, keepdim=True)
        hebbian_new = hebbian + msg_mag * w_conn_sig

        return h_new, msg, hebbian_new

    def test_fused_step_matches_python(self):
        fused_step = self._try_import()
        cfg = make_tiny()
        BS = 2
        N, D, K = cfg.N_neurons, cfg.D_neuron, cfg.K_connections
        H_S, H_M = cfg.state_mlp_hidden, cfg.msg_mlp_hidden
        device = 'cuda'

        # Create random tensors
        h = torch.randn(BS, N, D, device=device)
        prev_msg = torch.randn(BS, N, D, device=device)
        inject = torch.randn(BS, N, D, device=device)
        w_conn_sig = torch.sigmoid(torch.randn(BS, N, K, device=device))
        decay = torch.sigmoid(torch.randn(BS, N, device=device))
        primitives = torch.randn(BS, N, D, device=device)
        neuron_id = torch.randn(N, D, device=device) * 0.02

        # Connectivity
        all_idx = torch.arange(N, device=device)
        scores = torch.rand(N, N, device=device)
        scores[all_idx, all_idx] = -float('inf')
        K_actual = min(K, N - 1)
        conn_indices = torch.zeros(N, K, dtype=torch.long, device=device)
        conn_indices[:, :K_actual] = scores.topk(K_actual, dim=1).indices
        conn_indices, _ = conn_indices.sort(dim=-1)

        # Dendritic weights
        bsz = cfg.dendrite_branch_size
        nb = K // bsz
        bpg = min(4, nb)
        ng = max(1, nb // bpg)
        bpg = nb // ng
        branch_w = torch.randn(N, nb, bsz, D, device=device) * 0.1
        group_w = torch.randn(N, ng, bpg, D, device=device) * 0.1

        # MLP weights — w1 transposed: [N, H, I] for contiguous Triton access
        state_in = 2 * D + 1
        state_w1 = torch.randn(N, H_S, state_in, device=device) * 0.1
        state_b1 = torch.randn(N, H_S, device=device) * 0.01
        state_w2 = torch.randn(N, H_S, D, device=device) * 0.1
        state_b2 = torch.randn(N, D, device=device) * 0.01
        msg_w1 = torch.randn(N, H_M, 2*D, device=device) * 0.1
        msg_b1 = torch.randn(N, H_M, device=device) * 0.01
        msg_w2 = torch.randn(N, H_M, D, device=device) * 0.1
        msg_b2 = torch.randn(N, D, device=device) * 0.01

        # Python reference
        heb_py = torch.zeros(BS, N, K, device=device)
        h_py, msg_py, heb_py = self._python_step(
            h, prev_msg, inject, conn_indices, w_conn_sig,
            decay, primitives, neuron_id,
            branch_w, group_w,
            state_w1, state_b1, state_w2, state_b2,
            msg_w1, msg_b1, msg_w2, msg_b2, heb_py, cfg)

        # Triton
        heb_tr = torch.zeros(BS, N, K, device=device)
        h_tr, msg_tr = fused_step(
            h, prev_msg, inject, conn_indices, w_conn_sig,
            decay, primitives, neuron_id,
            branch_w, group_w,
            state_w1, state_b1, state_w2, state_b2,
            msg_w1, msg_b1, msg_w2, msg_b2,
            heb_tr, bsz, bpg, ng, True)

        torch.testing.assert_close(h_tr, h_py, atol=1e-3, rtol=1e-3,
                                   msg="h mismatch")
        torch.testing.assert_close(msg_tr, msg_py, atol=1e-3, rtol=1e-3,
                                   msg="msg mismatch")
        torch.testing.assert_close(heb_tr, heb_py, atol=1e-3, rtol=1e-3,
                                   msg="hebbian mismatch")
