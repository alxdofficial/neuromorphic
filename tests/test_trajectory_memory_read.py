"""Unit tests for src.trajectory_memory.read_module."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.manifold import Manifold
from src.trajectory_memory.read_module import (
    ReadTrajectoryGenerator,
    gumbel_top1_ste,
    make_pos_enc,
)


# ── helpers ─────────────────────────────────────────────────────────────


def _small_setup(BS: int = 2) -> tuple[TrajMemConfig, Manifold, torch.Tensor, torch.Tensor]:
    cfg = TrajMemConfig.small()
    m = Manifold(cfg)
    states = m.reset_states(batch_size=BS)
    prev_hid = torch.randn(BS, cfg.T_window, cfg.d_lm)
    return cfg, m, states, prev_hid


# ── pos_enc ──────────────────────────────────────────────────────────────


def test_pos_enc_shape_and_values():
    pe = make_pos_enc(K=4, D=8)
    assert pe.shape == (4, 8)
    # Position 0 should be sin(0)=0, cos(0)=1 alternating
    expected_p0 = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)
    assert torch.allclose(pe[0], expected_p0)


# ── gumbel_top1_ste ─────────────────────────────────────────────────────


def test_gumbel_top1_inference_argmax():
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    one_hot, idx = gumbel_top1_ste(logits, tau=1.0, hard=False)
    assert idx.tolist() == [1]
    assert one_hot[0, 1] == 1.0


def test_gumbel_top1_train_one_hot_forward():
    logits = torch.tensor([[1.0, 5.0, 2.0]], requires_grad=True)
    one_hot, idx = gumbel_top1_ste(logits, tau=1.0, hard=True)
    # Forward should be one-hot (sums to 1, exactly one slot at 1.0)
    assert torch.allclose(one_hot.sum(-1), torch.ones(1))
    assert ((one_hot == 0) | (one_hot == 1)).all()


def test_gumbel_top1_train_grad_flows():
    """Backprop through the STE one_hot must reach logits. Use a
    non-constant downstream (one_hot · target) — `one_hot.sum()` would
    be constant=BS (softmax rows sum to 1) and trivially zero-gradient."""
    torch.manual_seed(0)
    logits = torch.randn(4, 8, requires_grad=True)
    target = torch.randn(4, 8)
    one_hot, _ = gumbel_top1_ste(logits, tau=1.0, hard=True)
    loss = (one_hot * target).sum()
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_gumbel_top1_no_nan_in_training():
    """Regression for the clamp_min precedence bug. Pre-fix, training-mode
    Gumbel produced NaNs because `-torch.log(x).clamp_min(1e-20)` parses
    as `-(torch.log(x).clamp_min(1e-20))` — clamps the negative log to
    1e-20, then negates → log(-1e-20) = NaN."""
    torch.manual_seed(0)
    logits = torch.randn(16, 32, requires_grad=True)
    one_hot, idx = gumbel_top1_ste(logits, tau=1.0, hard=True)
    assert torch.isfinite(one_hot).all(), "Gumbel produced NaN one_hot"
    one_hot.sum().backward()
    assert torch.isfinite(logits.grad).all(), "Gumbel produced NaN gradient"


# ── ReadTrajectoryGenerator ─────────────────────────────────────────────


def test_read_module_construct():
    cfg = TrajMemConfig.small()
    rm = ReadTrajectoryGenerator(cfg)
    # entry_proj (Hopfield-tied; standalone copy in tests), MLPs, attn
    # modules all present.
    assert rm.entry_proj.head_query.shape == (cfg.J, cfg.D_concept)
    assert rm.entry_proj.entry_mlp is not None
    assert rm.history_attn is not None
    assert rm.cross_attn is not None
    assert rm.step_mlp is not None
    # pos_enc buffer (size K_read — indexed pos_enc[:K_read] in longest hop)
    assert rm.pos_enc.shape == (cfg.K_read, cfg.D_concept)


def test_read_module_forward_shapes_inference():
    cfg, m, states, prev_hid = _small_setup(BS=2)
    rm = ReadTrajectoryGenerator(cfg)
    visited, vids, _ = rm(prev_hid, states, m, hard=False)
    assert visited.shape == (2, cfg.J, cfg.K_read, cfg.D_concept)
    assert vids.shape == (2, cfg.J, cfg.K_read)
    assert vids.dtype == torch.int64
    # All visited ids should be valid concept indices
    assert (vids >= 0).all() and (vids < cfg.N).all()


def test_read_module_forward_shapes_train():
    cfg, m, states, prev_hid = _small_setup(BS=2)
    rm = ReadTrajectoryGenerator(cfg)
    visited, vids, _ = rm(prev_hid, states, m, hard=True)
    assert visited.shape == (2, cfg.J, cfg.K_read, cfg.D_concept)


def test_read_module_all_params_receive_gradient():
    """Every named parameter in read_module must receive non-None gradient.
    Routing differentiability via Gumbel-STE one_hot is the load-bearing
    mechanism — if one_hot is discarded (bug 1), step_mlp / entry_mlp /
    head_query / attn weights all dead-end at integer indices.
    """
    cfg, m, _, prev_hid = _small_setup(BS=2)
    rm = ReadTrajectoryGenerator(cfg)
    states = m.reset_states(batch_size=2)
    visited, _, _ = rm(prev_hid, states, m, hard=True)
    visited.sum().backward()

    missing = [n for n, p in rm.named_parameters() if p.grad is None]
    assert not missing, (
        f"these read_module params got no gradient (routing dead-end?): {missing}"
    )
    # state_init flows via the soft-gather of current_states.
    assert m.state_init.grad is not None
    # concept_ids flows via Q · concept_ids in entry-point + per-hop scoring.
    assert m.concept_ids.grad is not None, (
        "concept_ids got no gradient — Q·K_id path dead-ended?"
    )


def test_read_module_does_not_mutate_concept_states():
    """Read should be non-destructive: manifold's concept_states buffer
    should be unchanged after a forward pass."""
    cfg, m, _, prev_hid = _small_setup(BS=2)
    rm = ReadTrajectoryGenerator(cfg)
    before = m.concept_states.clone()
    states = m.reset_states(batch_size=2)
    _ = rm(prev_hid, states, m, hard=True)
    assert torch.allclose(m.concept_states, before)


def test_read_module_visited_states_match_states_at_visited_ids():
    """visited[b, j, t] should equal states[b, visited_ids[b, j, t]]."""
    cfg, m, states, prev_hid = _small_setup(BS=2)
    rm = ReadTrajectoryGenerator(cfg)
    visited, vids, _ = rm(prev_hid, states, m, hard=False)
    # In read path, visited list contains the RAW pre-mutation states.
    BS = 2
    for b in range(BS):
        for j in range(cfg.J):
            for t in range(cfg.K_read):
                expected = states[b, vids[b, j, t]]
                assert torch.allclose(visited[b, j, t], expected, atol=1e-5), (
                    f"visited[{b},{j},{t}] doesn't match states at id={vids[b, j, t]}"
                )


def test_read_module_j_trajectories_diverge():
    """Different head_queries should produce different trajectories.
    Tested in both inference (argmax) and training (Gumbel) modes.
    Earlier head_query std=0.02 collapsed all J to identical paths
    because pooled prev_hiddens dominated."""
    cfg = TrajMemConfig.small()
    cfg.J = 4
    cfg.validate()
    m = Manifold(cfg)
    states = m.reset_states(batch_size=1)
    prev_hid = torch.randn(1, cfg.T_window, cfg.d_lm)
    rm = ReadTrajectoryGenerator(cfg)
    for hard in (False, True):
        torch.manual_seed(42)
        _, vids, _ = rm(prev_hid, states, m, hard=hard)
        flattened = vids[0].tolist()                              # [J, K]
        distinct_paths = len({tuple(p) for p in flattened})
        assert distinct_paths >= 2, (
            f"hard={hard}: all J={cfg.J} trajectories produced the same path: "
            f"{flattened}"
        )


# ── routing aux losses (Switch + ST-MoE) ────────────────────────────────

def test_routing_aux_losses_uniform_vs_concentrated():
    """Switch-style load_balance is N at maximally concentrated routing
    and 1 at perfectly uniform. z_loss is 0 when logits are all-zero."""
    from src.trajectory_memory.read_module import routing_aux_losses
    N = 16
    # Uniform: all logits zero → P_i = 1/N for all i.
    # Hard selection split uniformly across N (one sample per concept).
    uniform_logits = torch.zeros(N, N)
    uniform_oh = torch.eye(N)
    a = routing_aux_losses(uniform_logits, uniform_oh)
    assert abs(a["load_balance"].item() - 1.0) < 1e-4, (
        f"uniform load_balance should be 1.0, got {a['load_balance'].item():.4f}"
    )
    # logsumexp of zero vec of length N = log(N); z_loss = log(N)².
    expected_z = math.log(N) ** 2
    assert abs(a["z_loss"].item() - expected_z) < 0.1, (
        f"z_loss at zero logits should be log(N)²≈{expected_z:.2f}, "
        f"got {a['z_loss'].item():.4f}"
    )
    # Concentrated: all selections pick concept 0, all softmax mass on 0.
    spike_logits = torch.zeros(N, N)
    spike_logits[:, 0] = 100.0
    spike_oh = torch.zeros(N, N); spike_oh[:, 0] = 1.0
    b = routing_aux_losses(spike_logits, spike_oh)
    # f_0 = 1, P_0 ≈ 1, so loss ≈ N · 1 · 1 = N
    assert abs(b["load_balance"].item() - N) < 1.0, (
        f"concentrated load_balance should be ~N={N}, got {b['load_balance'].item():.4f}"
    )
    # z_loss with logsumexp ≈ 100 → 10000
    assert b["z_loss"].item() > 9000, (
        f"z_loss should be ~10000 with spike logits, got {b['z_loss'].item():.4f}"
    )


def test_routing_aux_losses_gradient_flows():
    """Both aux losses must produce gradient back to logits (entry_proj
    weights, etc.) — otherwise they don't actually push routing."""
    from src.trajectory_memory.read_module import routing_aux_losses
    N = 8
    logits = torch.zeros(2, 3, N, requires_grad=True)
    # Make a one-hot — doesn't need to match logits for the loss math.
    oh = F.one_hot(torch.randint(0, N, (2, 3)), num_classes=N).float()
    a = routing_aux_losses(logits, oh)
    (a["load_balance"] + a["z_loss"]).backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0, "no gradient flowed to logits"


def test_gumbel_top1_ste_accepts_tensor_tau():
    """tau as a 0-dim tensor must work (used by trainer's per-step
    schedule to avoid Dynamo recompiles on changing Python floats)."""
    from src.trajectory_memory.read_module import gumbel_top1_ste
    logits = torch.randn(4, 8)
    tau_t = torch.tensor(0.7)
    oh, idx = gumbel_top1_ste(logits, tau_t, hard=True)
    assert oh.shape == (4, 8)
    assert idx.shape == (4,)


def test_read_module_accepts_tensor_tau():
    """read_module must accept tau as a tensor without crashing or
    breaking the routing math."""
    cfg = TrajMemConfig.small()
    rm = ReadTrajectoryGenerator(cfg)
    m = Manifold(cfg)
    states = m.reset_states(batch_size=2)
    prev_hid = torch.randn(2, cfg.T_window, cfg.d_lm)
    tau_t = torch.tensor(0.7)
    visited, vids, aux = rm(prev_hid, states, m, hard=True, tau=tau_t)
    assert visited.shape == (2, cfg.J, cfg.K_read, cfg.D_concept)
    assert "load_balance" in aux
