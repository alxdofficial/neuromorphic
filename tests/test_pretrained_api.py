"""Smoke tests for the pretrained-LM integration API on GraphWalkerMemory.

Covers:
- walker_step_from_h: runs on an external [B, D_s] vector, mutates state,
  produces a finite motor_state.
- walk_segment: [B, T, D_s] → [B, T, D_s] readouts. Gradient flows
  through all expected params including the pretrained-only
  the walker's projections.
- Block-vs-per-token parity for the compiled block path.
"""

from __future__ import annotations

import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.standalone import StandaloneLM


def _tiny_cfg(**overrides) -> GraphWalkerConfig:
    base = dict(
        grid_rows=8, grid_cols=8, radius=2,
        K=8, D_model=64, D_s=64, D_id=16,
        n_heads=2,
        D_q_per_head=16, n_score_heads=2,
        K_horizons=4,
        vocab_size=256,
        mod_period=4, tbptt_block=4, segment_T=8,
        gumbel_tau_start=1.0, gumbel_tau_end=1.0, gumbel_anneal_steps=1,
        epsilon_start=0.0, epsilon_end=0.0, epsilon_anneal_steps=1,
        lambda_balance=0.0,
        use_neuromod=True,
        neuromod_D_mod=32, neuromod_n_layers=1, neuromod_n_heads=2,
        neuromod_edge_hidden=16, neuromod_eta=1.0,
    )
    base.update(overrides)
    return GraphWalkerConfig(**base)


def test_step_core_from_h_runs_and_mutates_state():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    lm = StandaloneLM(cfg).cpu()
    m = lm.memory
    m.begin_segment(B=2, device=torch.device("cpu"))
    s_before = m.s.clone()
    walker_pos_before = m.walker_pos.clone()

    h = torch.randn(2, cfg.D_s)
    readout = m.walker_step_from_h(h)

    assert readout.motor_state.shape == (2, cfg.D_s)
    assert torch.isfinite(readout.motor_state).all()
    # Walker position must have moved to anchor (new window)
    assert not torch.equal(walker_pos_before, m.walker_pos)
    # Column state must have changed where walkers wrote
    assert not torch.equal(s_before, m.s)


def test_walk_segment_shapes_and_finite():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    lm = StandaloneLM(cfg).cpu()
    m = lm.memory
    B, T = 2, cfg.segment_T
    h_mem = torch.randn(B, T, cfg.D_s)

    m.begin_segment(B, torch.device("cpu"))
    readouts = m.walk_segment(h_mem)

    assert readouts.shape == (B, T, cfg.D_s)
    assert torch.isfinite(readouts).all()


def test_walk_segment_gradient_reaches_pretrained_only_params():
    """Backward from a readouts.pow(2).mean() loss (proxy for the gradient
    Llama would inject through W_out) should reach:
    - h_mem (so W_in would learn)
    - content_mlp, q_proj, k_all, nbr_id_to_s, walker_state_alpha
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    lm = StandaloneLM(cfg).cpu()
    m = lm.memory
    # Perturb zero-init gates so the gradient chain has signal.
    with torch.no_grad():
        m.decay_proj.weight.normal_(std=0.05)
        m.decay_proj.bias.normal_(std=0.05)
        m.readout.pred_head.proj.weight.normal_(std=0.05)
        m.neuromod.edge_mlp[-1].weight.normal_(std=0.05)
        m.neuromod.edge_mlp[-1].bias.normal_(std=0.05)
        m.neuromod.blend_logit.fill_(0.0)

    B, T = 2, cfg.segment_T
    h_mem = torch.randn(B, T, cfg.D_s, requires_grad=True)

    m.begin_segment(B, torch.device("cpu"))
    readouts = m.walk_segment(h_mem)
    loss = readouts.float().pow(2).mean()
    loss.backward()

    # h_mem must receive gradient (so W_in would learn)
    assert h_mem.grad is not None
    assert h_mem.grad.abs().sum() > 0, "h_mem (Llama hidden proxy) got no grad"

    # Sanity: shared walker params get gradient too. The walker is vocab-
    # agnostic now — `state_to_model` (only used by the dropped aux CE
    # path) deliberately gets NO gradient from walk_segment.
    for name in [
        "cols.q_proj.0.weight",
        "nbr_id_to_s.weight",
        "walker_state_alpha",
    ]:
        # named_parameters walks m (GraphWalkerMemory); cols is a submodule
        p = dict(m.named_parameters())[name]
        assert p.grad is not None and p.grad.abs().sum() > 0, (
            f"{name} got no gradient from walk_segment"
        )


def test_walk_segment_preserve_graph_skips_detach():
    """When preserve_graph=True, the walker's per-block detach is skipped,
    so gradient from a late-segment readout should reach h_mem[:, 0]."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    lm = StandaloneLM(cfg).cpu()
    m = lm.memory
    B, T = 2, cfg.segment_T
    h_mem = torch.randn(B, T, cfg.D_s, requires_grad=True)
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))

    m.begin_segment(B, torch.device("cpu"))
    readouts = m.walk_segment(h_mem, preserve_graph=True)
    # Loss from the LAST position only — gradient must reach h_mem[:, 0]
    # only if detach was skipped.
    loss = readouts[:, -1].pow(2).mean()
    loss.backward()

    assert h_mem.grad is not None
    # With preserve_graph=True, the first-position h_mem should receive
    # gradient because TBPTT detach didn't sever the recurrent chain.
    assert h_mem.grad[:, 0].abs().sum() > 0, (
        "preserve_graph=True failed — detach still cut the gradient chain"
    )


def _parity_setup(seed: int):
    """Two identical walkers + identical adapters seeded the same way, plus
    matched h_mem and input_ids tensors. Used to assert per-token vs
    block-path equivalence."""
    torch.manual_seed(seed)
    cfg = _tiny_cfg(use_neuromod=False)  # neuromod off for cleaner parity
    lm_a = StandaloneLM(cfg).cpu()
    torch.manual_seed(seed)
    lm_b = StandaloneLM(cfg).cpu()
    B, T = 2, cfg.segment_T
    torch.manual_seed(seed + 100)
    h_mem = torch.randn(B, T, cfg.D_s)
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    return cfg, lm_a, lm_b, h_mem, input_ids, B, T


def test_walk_segment_block_matches_per_token():
    """The block path (compile_walk_block_from_h installed) must produce numerically
    identical readouts to the per-token path (no compile) when configs and
    inputs are matched. This locks in the parity that
    `_compiled_walk_block_from_h` is allowed to exist as a drop-in optimization
    rather than a behavior change.
    """
    cfg, lm_a, lm_b, h_mem, input_ids, B, T = _parity_setup(seed=42)

    # Path A: per-token (no compile). Seed gumbel noise so both paths see
    # identical random draws.
    m_a = lm_a.memory
    m_a.begin_segment(B, torch.device("cpu"))
    h_a = h_mem.clone().requires_grad_(True)
    torch.manual_seed(2024)
    readouts_a = m_a.walk_segment(h_a)

    # Path B: block path. Skip torch.compile to keep this test cheap on CPU
    # — install a thin trampoline that delegates to walk_block_from_h.
    m_b = lm_b.memory
    m_b._compiled_walk_block_from_h = m_b.walk_block_from_h
    m_b.begin_segment(B, torch.device("cpu"))
    h_b = h_mem.clone().requires_grad_(True)
    torch.manual_seed(2024)
    readouts_b = m_b.walk_segment(h_b)

    torch.testing.assert_close(readouts_a, readouts_b, rtol=1e-5, atol=1e-5)

    # Backward parity — gradient on h_mem must match.
    readouts_a.float().pow(2).mean().backward()
    readouts_b.float().pow(2).mean().backward()
    assert h_a.grad is not None and h_b.grad is not None
    torch.testing.assert_close(h_a.grad, h_b.grad, rtol=1e-5, atol=1e-5)
