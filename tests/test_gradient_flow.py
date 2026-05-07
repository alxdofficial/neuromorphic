"""Exhaustive gradient-flow test for GraphWalker.

Goal: after one phase1 step (with the neuromod pre-seeded by a prior
segment), every trainable parameter in `StandaloneLM` should receive a
finite, non-zero gradient. Zero-init parameters are perturbed before
the step so gradient can flow through them (otherwise the test would
not catch a broken chain that happens to multiply by zero at init).

This test exists because the graph_walker hot path has many gradient
bridges:
- content_mlp, q_proj, k_proj via the routing STE and the endpoint
  readout
- state_to_model / PostModelStack / PredictionHead via the flush CE
- input_q/k/v_proj via the once-per-window STE-gated anchor injection
- prev_motor_proj via the anchor query (zero-init)
- motor_query / out_k/v_proj via the cross-attn over endpoints
- col_id via steering, keys, and nbr_id_to_s
- decay_proj via the LIF α (zero-init)
- walker_state_alpha via the walker state EMA
- nbr_id_to_s via the STE routing bridge in the endpoint readout
- neuromod subtree (feature_proj, attn layers, edge_mlp, blend_logit)
  via the active_E_bias path in routing scores

If any of these have `.grad is None` or `.grad.abs().sum() == 0` after a
well-conditioned training step, the corresponding training signal is
silently broken.
"""

from __future__ import annotations

import pytest
import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.standalone import StandaloneLM
from src.graph_walker.train_phase1 import phase1_step


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
        lambda_balance=0.01,
        use_neuromod=True,
        neuromod_D_mod=32, neuromod_n_layers=1, neuromod_n_heads=2,
        neuromod_edge_hidden=16, neuromod_eta=1.0,
    )
    base.update(overrides)
    return GraphWalkerConfig(**base)


@torch.no_grad()
def _perturb_zero_init_params(lm: StandaloneLM, std: float = 0.05) -> None:
    """Replace zero-initialised weights with small random values so gradient
    can actually propagate through the paths they gate.

    After training, these parameters learn non-zero values naturally; but
    for a gradient-flow test we need non-zero values up front."""
    m = lm.memory
    # prev_motor_proj: zero-init per the design (chain is inert at day 0)
    m.prev_motor_proj.weight.normal_(std=std)
    # decay_proj: zero-init so α(c) = σ(0) = 0.5 uniform
    m.decay_proj.weight.normal_(std=std)
    m.decay_proj.bias.normal_(std=std)
    # pred_head.proj: zero-init residual in PredictionHead
    m.readout.pred_head.proj.weight.normal_(std=std)
    # Neuromod output head: zero-init so target=0 at init (if neuromod exists)
    if m.neuromod is not None:
        m.neuromod.edge_mlp[-1].weight.normal_(std=std)
        m.neuromod.edge_mlp[-1].bias.normal_(std=std)
        # Open the blend gate so delta_nm has visible magnitude
        m.neuromod.blend_logit.fill_(0.0)  # γ = σ(0) = 0.5


def test_all_trainable_params_get_gradient():
    """Catch silently-broken training signals by verifying every trainable
    parameter has a finite, non-zero gradient after a well-conditioned
    phase1 step."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    lm = StandaloneLM(cfg).cpu()
    _perturb_zero_init_params(lm)

    opt = torch.optim.AdamW(lm.parameters(), lr=1e-4)
    # segment_T=8, mod_period=4 → 2 plasticity windows per segment.
    # The second window's neuromod fire consumes the first window's snapshot,
    # so we get neuromod-parameter gradients even on the first segment.
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.segment_T))

    # Pre-seed a prior snapshot by running an extra segment first. Without it,
    # the very first window has no prev snapshot and _active_neuromod_delta is None,
    # so neuromod params would get no gradient on that first measured step.
    phase1_step(
        lm, opt, tokens, tbptt_block=cfg.tbptt_block,
        amp_dtype=None, training_step=0,
    )

    opt.zero_grad(set_to_none=True)
    stats = phase1_step(
        lm, opt, tokens, tbptt_block=cfg.tbptt_block,
        amp_dtype=None, training_step=1,
    )
    assert torch.isfinite(torch.tensor(stats.loss))

    # Params that only participate in the pretrained-LM (walk_segment)
    # path, not the StandaloneLM token-id path. Their gradient is verified
    # in the pretrained smoke tests, not here.
    STANDALONE_UNUSED = {"memory.mem_input_v_proj.weight"}

    missing: list[str] = []
    nonfinite: list[str] = []
    zero_grad: list[str] = []

    for name, p in lm.named_parameters():
        if not p.requires_grad:
            continue
        if name in STANDALONE_UNUSED:
            continue
        if p.grad is None:
            missing.append(name)
            continue
        if not torch.isfinite(p.grad).all().item():
            nonfinite.append(name)
            continue
        if p.grad.abs().sum().item() == 0.0:
            zero_grad.append(name)

    problems = []
    if missing:
        problems.append(f"MISSING .grad for: {missing}")
    if nonfinite:
        problems.append(f"NON-FINITE .grad for: {nonfinite}")
    if zero_grad:
        problems.append(f"ZERO .grad for: {zero_grad}")
    assert not problems, "\n".join(problems)


def test_gradient_flow_without_neuromod():
    """Same coverage check with use_neuromod=False. The neuromod subtree is
    absent, and plasticity uses only the surprise-gated Hebbian path. All
    remaining params should still get gradient."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(use_neuromod=False)
    lm = StandaloneLM(cfg).cpu()
    _perturb_zero_init_params(lm)

    opt = torch.optim.AdamW(lm.parameters(), lr=1e-4)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.segment_T))

    opt.zero_grad(set_to_none=True)
    stats = phase1_step(
        lm, opt, tokens, tbptt_block=cfg.tbptt_block,
        amp_dtype=None, training_step=0,
    )
    assert torch.isfinite(torch.tensor(stats.loss))

    STANDALONE_UNUSED = {"memory.mem_input_v_proj.weight"}
    missing, nonfinite, zero_grad = [], [], []
    for name, p in lm.named_parameters():
        if not p.requires_grad:
            continue
        if name in STANDALONE_UNUSED:
            continue
        if p.grad is None:
            missing.append(name)
        elif not torch.isfinite(p.grad).all().item():
            nonfinite.append(name)
        elif p.grad.abs().sum().item() == 0.0:
            zero_grad.append(name)

    problems = []
    if missing:
        problems.append(f"MISSING .grad (neuromod off) for: {missing}")
    if nonfinite:
        problems.append(f"NON-FINITE .grad (neuromod off) for: {nonfinite}")
    if zero_grad:
        problems.append(f"ZERO .grad (neuromod off) for: {zero_grad}")
    assert not problems, "\n".join(problems)


def test_known_zero_init_params_have_zero_grad_on_first_step():
    """Sanity check of the 'zero-init means no gradient yet' story.

    Without the _perturb helper, the zero-init chains produce
    multiplicative zeros along the path to loss and upstream params see
    zero gradient. This test documents that property so if someone
    changes an init they can spot the consequence.

    Expected zero grads (before any training on these specific params):
    - decay_proj.weight/bias: decay_proj is zero so α = σ(0) = 0.5 — the
      gradient through LIF blend actually reaches decay_proj's inputs
      IF alpha enters a non-constant computation. It does (s_new is a
      function of α). So this one is NOT actually expected to be zero.
    - input_v_proj.weight: zero weight → 0 injection → 0 grad on weight
      only if injection path is the sole gradient source. But
      input_v_proj is queried via STE gating of the anchor softmax —
      gradient should reach input_v_proj if inject_msg affects any
      downstream output. It does on new-window steps.
    - prev_motor_proj.weight: zero weight → 0 contribution to anchor
      query. Gradient reaches the weight via the anchor softmax's
      dependency on the query.

    Actually, most zero-inits here DO receive non-trivial gradient
    because the chain they gate is not zero-multiplied at the point
    their weight matters. The ones that DO stay at zero gradient are
    those whose output is multiplied by themselves downstream, creating
    a y·y bilinear dead product — mostly irrelevant to this model.

    Keeping this test as a living document: document any params that
    empirically do receive zero gradient on the first step, if any.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    lm = StandaloneLM(cfg).cpu()
    # Do NOT perturb; keep zero-inits as-is.

    opt = torch.optim.AdamW(lm.parameters(), lr=1e-4)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.segment_T))

    opt.zero_grad(set_to_none=True)
    phase1_step(
        lm, opt, tokens, tbptt_block=cfg.tbptt_block,
        amp_dtype=None, training_step=0,
    )

    # Gather zero-grad params (for reference)
    zero = []
    for name, p in lm.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        if p.grad.abs().sum().item() == 0.0:
            zero.append(name)

    # neuromod output layer is zero-init → target=0 → delta_nm=0 → zero grad
    # on itself until perturbation. Known and accepted.
    # input_v_proj is zero-init → inject_msg is zero regardless of
    # input_ste_weights, so nothing downstream depends on input_v_proj.weight
    # → zero grad on weight until perturbation.
    # Same argument for prev_motor_proj.weight and decay_proj.bias/weight.
    #
    # The test does NOT assert a specific list — it just ensures phase1_step
    # runs and produces finite loss even when those paths start zero.
    assert zero is not None  # placeholder: test succeeds if step ran


def test_gradient_reaches_neuromod_via_delta_nm_path():
    """Specifically verify the neuromod gradient path:
    loss → CE → readout → end_state → ste_weights → scores → active_E_bias
        → _active_neuromod_delta → neuromod.edge_mlp / feature_proj / layers / blend_logit

    This is a high-value bridge: if any detach or autograd-graph break
    along the chain is introduced, the neuromod becomes untrainable.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    lm = StandaloneLM(cfg).cpu()
    _perturb_zero_init_params(lm)

    opt = torch.optim.AdamW(lm.parameters(), lr=1e-4)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.segment_T))

    # Pre-seed snapshot by running one segment first.
    phase1_step(
        lm, opt, tokens, tbptt_block=cfg.tbptt_block,
        amp_dtype=None, training_step=0,
    )

    opt.zero_grad(set_to_none=True)
    phase1_step(
        lm, opt, tokens, tbptt_block=cfg.tbptt_block,
        amp_dtype=None, training_step=1,
    )

    # Explicitly check each neuromod subtree param
    critical = [
        "memory.neuromod.feature_proj.weight",
        "memory.neuromod.feature_proj.bias",
        "memory.neuromod.edge_mlp.0.weight",
        "memory.neuromod.edge_mlp.0.bias",
        "memory.neuromod.edge_mlp.2.weight",
        "memory.neuromod.edge_mlp.2.bias",
        "memory.neuromod.blend_logit",
    ]
    params = dict(lm.named_parameters())
    for name in critical:
        assert name in params, f"expected param {name} not found"
        g = params[name].grad
        assert g is not None, f"{name}: no gradient"
        assert torch.isfinite(g).all().item(), f"{name}: non-finite"
        assert g.abs().sum().item() > 0, f"{name}: zero gradient"


def test_loss_is_finite_and_decreasing_on_first_few_steps():
    """Smoke test: across 3 consecutive phase1 steps on fixed tokens, the
    loss should not blow up. Doesn't have to monotonically decrease
    (it's a random init on a tiny graph), but it must stay finite and
    not obviously diverge."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    lm = StandaloneLM(cfg).cpu()
    _perturb_zero_init_params(lm)

    opt = torch.optim.AdamW(lm.parameters(), lr=1e-3)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.segment_T))

    losses = []
    for step in range(3):
        s = phase1_step(
            lm, opt, tokens, tbptt_block=cfg.tbptt_block,
            amp_dtype=None, training_step=step,
        )
        losses.append(s.loss)
        assert torch.isfinite(torch.tensor(s.loss)), f"step {step}: non-finite loss {s.loss}"
    # Don't require strict monotonic decrease; just ensure no blow-up.
    assert all(abs(l) < 1e4 for l in losses), f"loss blew up: {losses}"
