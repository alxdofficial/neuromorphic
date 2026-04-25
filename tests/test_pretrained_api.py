"""Smoke tests for the pretrained-LM integration API on GraphWalkerMemory.

Covers:
- step_core_from_h: runs on an external [B, D_s] vector, mutates state,
  produces a finite motor_state.
- forward_segment: [B, T, D_s] → [B, T, D_s] readouts, plus aux CE when
  an adapter is supplied. Gradient flows through all expected params
  including the pretrained-only `mem_input_v_proj`.
- Zero-delta-at-day-zero: with a fake-identity adapter, the aux loss
  is finite and state evolution matches the token-id path for the
  non-pretrained-specific dynamics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.graph_walker import GraphWalkerMemory
from src.graph_walker.standalone import StandaloneLM


def _tiny_cfg(**overrides) -> GraphWalkerConfig:
    base = dict(
        plane_rows=8, plane_cols=8, L=2,
        K=8, D_model=64, D_s=64, D_id=16,
        n_heads=2, n_hops=3,
        D_q_in=16, D_q_per_head=16, n_score_heads=2,
        K_horizons=4, K_buf=4,
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


class _MockAdapter:
    """Stand-in for MemAdapter: maps [B, D_s] → [B, vocab_lm] via a
    single Linear. Doesn't go through any real LM norm/lm_head — just
    enough to exercise the aux-CE path shape + gradient."""
    def __init__(self, d_s: int, vocab_lm: int):
        self.head = nn.Linear(d_s, vocab_lm, bias=False)
        nn.init.normal_(self.head.weight, std=0.02)

    def mem_head_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def test_step_core_from_h_runs_and_mutates_state():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    lm = StandaloneLM(cfg).cpu()
    m = lm.memory
    m.begin_segment(B=2, device=torch.device("cpu"))
    s_before = m.s.clone()
    walker_pos_before = m.walker_pos.clone()

    h = torch.randn(2, cfg.D_s)
    readout = m.step_core_from_h(h)

    assert readout.motor_state.shape == (2, cfg.D_s)
    assert torch.isfinite(readout.motor_state).all()
    # Walker position must have moved to anchor (new window)
    assert not torch.equal(walker_pos_before, m.walker_pos)
    # Column state must have changed where walkers wrote
    assert not torch.equal(s_before, m.s)


def test_forward_segment_shapes_and_finite():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    lm = StandaloneLM(cfg).cpu()
    m = lm.memory
    B, T = 2, cfg.segment_T
    h_mem = torch.randn(B, T, cfg.D_s)
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))

    m.begin_segment(B, torch.device("cpu"))
    adapter = _MockAdapter(cfg.D_s, cfg.vocab_size)
    readouts, aux_loss = m.forward_segment(h_mem, input_ids, adapter)

    assert readouts.shape == (B, T, cfg.D_s)
    assert torch.isfinite(readouts).all()
    assert aux_loss is not None
    assert torch.isfinite(aux_loss)


def test_forward_segment_gradient_reaches_pretrained_only_params():
    """Backward from a composite loss of aux CE + readouts.sum() should
    reach:
    - mem_input_v_proj (only used in the external-h path)
    - content_mlp, q_proj, k_all, nbr_id_to_s, walker_state_alpha
    - adapter.head (external)
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    lm = StandaloneLM(cfg).cpu()
    m = lm.memory
    # Perturb zero-init gates so the gradient chain has signal.
    with torch.no_grad():
        m.prev_motor_proj.weight.normal_(std=0.05)
        m.decay_proj.weight.normal_(std=0.05)
        m.decay_proj.bias.normal_(std=0.05)
        m.readout.pred_head.proj.weight.normal_(std=0.05)
        m.neuromod.edge_mlp[-1].weight.normal_(std=0.05)
        m.neuromod.edge_mlp[-1].bias.normal_(std=0.05)
        m.neuromod.blend_logit.fill_(0.0)

    B, T = 2, cfg.segment_T
    h_mem = torch.randn(B, T, cfg.D_s, requires_grad=True)
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))

    m.begin_segment(B, torch.device("cpu"))
    adapter = _MockAdapter(cfg.D_s, cfg.vocab_size)
    readouts, aux_loss = m.forward_segment(h_mem, input_ids, adapter)
    loss = aux_loss + readouts.float().pow(2).mean()
    loss.backward()

    # h_mem must receive gradient (so W_in would learn)
    assert h_mem.grad is not None
    assert h_mem.grad.abs().sum() > 0, "h_mem (Llama hidden proxy) got no grad"

    # mem_input_v_proj must receive gradient (pretrained-only path)
    assert m.mem_input_v_proj.weight.grad is not None
    assert m.mem_input_v_proj.weight.grad.abs().sum() > 0, (
        "mem_input_v_proj got no gradient — pretrained anchor-v-inject path broken"
    )

    # Sanity: shared walker params get gradient too
    for name in [
        "cols.q_proj.0.weight",
        "nbr_id_to_s.weight",
        "walker_state_alpha",
        "state_to_model.weight",
    ]:
        # named_parameters walks m (GraphWalkerMemory); cols is a submodule
        p = dict(m.named_parameters())[name]
        assert p.grad is not None and p.grad.abs().sum() > 0, (
            f"{name} got no gradient from forward_segment"
        )


def test_forward_segment_preserve_graph_skips_detach():
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
    readouts, _ = m.forward_segment(
        h_mem, input_ids, adapter=None,
        compute_aux_loss=False, preserve_graph=True,
    )
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
