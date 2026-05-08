"""Regression tests for the second bug-sweep fixes (post 32684cf):

1. autoregressive_rollout restores model.training and memory.phase
   on exit. Previously it left the model in eval mode, which silently
   skipped aux loss in the next phase-1 step.
2. walk_segment skips the dense vocab readout AND plasticity finalize
   when a block has zero valid horizons (typical for T=1 generation
   steps inside an AR rollout).
3. unfreeze_all() re-freezes the standalone-only walker params
   (token_to_state). Without this they re-enter the
   optimizer state every cycle as dead weights.
4. begin_segment(clear_neuromod_carryover=True) wipes the previous
   segment's neuromod snapshot. model.begin_segment defaults to
   True for the pretrained path (independent-document batches).
5. phase1_pretrained_step asserts target_ids.shape == input_ids.shape
   to catch the "already pre-shifted" footgun.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.pretrained.config import PretrainedGWConfig
from src.graph_walker.pretrained.integrated_lm import IntegratedLM
from src.graph_walker.pretrained.rollout import autoregressive_rollout
from src.graph_walker.pretrained.train_phase1 import (
    Phase1Batch,
    phase1_pretrained_step,
)


def _tiny_cfg(**overrides):
    base = dict(
        grid_rows=4, grid_cols=4, radius=2,
        K=4, D_model=32, D_s=32, D_id=8,
        n_heads=2,
        D_q_per_head=8, n_score_heads=2,
        K_horizons=4, vocab_size=64,
        mod_period=8, tbptt_block=8, segment_T=8,
        gumbel_tau_start=1.0, gumbel_tau_end=1.0, gumbel_anneal_steps=1,
        epsilon_start=0.0, epsilon_end=0.0, epsilon_anneal_steps=1,
        lambda_balance=0.0, use_neuromod=True,
        plasticity_mode="neuromod_only",
        neuromod_D_mod=16, neuromod_n_layers=1, neuromod_n_heads=2,
        neuromod_edge_hidden=16, neuromod_eta=1.0,
    )
    base.update(overrides)
    return GraphWalkerConfig(**base)


def _make_tiny_llama(d_lm=32, n_layers=4, vocab=256):
    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=d_lm, intermediate_size=d_lm * 2,
        num_hidden_layers=n_layers, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=64,
        pad_token_id=0, bos_token_id=0, eos_token_id=0,
    )
    return LlamaForCausalLM(cfg)


def _make_tiny_wrapper(d_lm=32, vocab=256, T=8):
    hf = _make_tiny_llama(d_lm=d_lm, vocab=vocab)
    walker_cfg = _tiny_cfg(D_s=d_lm, D_model=d_lm, vocab_size=vocab, segment_T=T)
    cfg = PretrainedGWConfig(
        model_name="random", inject_layer=2, d_mem=d_lm,
        memory=walker_cfg, T=T, bs=2, llama_dtype="fp32",
    )
    return IntegratedLM(cfg, hf_model=hf)


# -- 1. rollout train-mode + memory.phase restore --


def test_autoregressive_rollout_restores_training_and_memory_phase():
    w = _make_tiny_wrapper()
    w.train()                                      # start in train mode
    w.current_phase = "phase1"
    w.memory.phase = "phase1"

    prefix = torch.randint(0, 256, (2, 8))
    autoregressive_rollout(
        w, prefix, gen_length=4, phase="phase2",
        grad_during_prefix=False, grad_during_gen=False,
    )

    assert w.training is True, (
        "rollout left model in eval mode — next phase-1 cycle would skip aux loss"
    )
    assert w.current_phase == "phase1"
    assert w.memory.phase == "phase1", (
        "rollout left memory.phase = 'phase2' — direct memory calls would use "
        "hard Categorical instead of Gumbel-STE"
    )


# -- 2. walk_segment skips work on targetless blocks --


def test_walk_segment_does_not_alter_surprise_or_e_bias():
    """`walk_segment` is now a pure forward (vocab-agnostic walker):
    it must NOT touch `surprise_ema` and must NOT fire plasticity. Both
    are externally driven by the trainer via `update_plasticity` after
    backward."""
    from src.graph_walker.standalone import StandaloneLM

    torch.manual_seed(0)
    walker_cfg = _tiny_cfg(segment_T=8, mod_period=4, tbptt_block=4)
    lm = StandaloneLM(walker_cfg).cpu()
    m = lm.memory

    m.begin_segment(B=2, device=torch.device("cpu"))
    m.surprise_ema = torch.full_like(m.surprise_ema, 0.7)
    surprise_before = m.surprise_ema.clone()
    e_bias_before = m.E_bias_flat.clone()

    # Single T=1 forward via walk_segment.
    h_mem = torch.randn(2, 1, walker_cfg.D_s)
    readouts = m.walk_segment(h_mem)
    assert readouts.shape == (2, 1, walker_cfg.D_s)
    # surprise EMA must be untouched.
    assert torch.equal(m.surprise_ema, surprise_before), (
        "walk_segment touched surprise_ema — should be externally driven only"
    )
    # E_bias must be untouched (plasticity is post-backward only).
    assert torch.equal(m.E_bias_flat, e_bias_before)


# -- 3. unfreeze_all keeps standalone-only params frozen --


def test_unfreeze_all_keeps_standalone_only_params_frozen():
    w = _make_tiny_wrapper()
    # First sanity: init froze them.
    assert w.memory.token_to_state.weight.requires_grad is False

    # Toggle phase 2 freeze, then unfreeze_all — the dead params must NOT
    # come back as trainable.
    w.freeze_all_but_E_bias_and_neuromod()
    assert w.memory.token_to_state.weight.requires_grad is False
    w.unfreeze_all()
    assert w.memory.token_to_state.weight.requires_grad is False, (
        "unfreeze_all re-enabled token_to_state — dead weight in optimizer"
    )


# -- 5. begin_segment(clear_neuromod_carryover) --


def test_begin_segment_clears_neuromod_carryover_when_requested():
    from src.graph_walker.standalone import StandaloneLM

    walker_cfg = _tiny_cfg()
    lm = StandaloneLM(walker_cfg).cpu()
    m = lm.memory
    # Manually plant carryover state. D_feat = D_s + D_id + 1 (+1 surprise
    # if neuromod_only mode adds a per-col surprise broadcast feature).
    extra = 1 if walker_cfg.plasticity_mode == "neuromod_only" else 0
    D_feat = walker_cfg.D_s + walker_cfg.D_id + 1 + extra
    m._neuromod_input_ids = torch.tensor([1, 2], dtype=torch.int64)
    m._neuromod_input_feats = torch.randn(2, D_feat)

    # Default (False) preserves carryover.
    m.begin_segment(B=2, device=torch.device("cpu"))
    assert m._neuromod_input_ids is not None

    # Explicit True clears.
    m._neuromod_input_ids = torch.tensor([1, 2], dtype=torch.int64)
    m.begin_segment(B=2, device=torch.device("cpu"), clear_neuromod_carryover=True)
    assert m._neuromod_input_ids is None
    assert m._neuromod_input_feats is None
    assert m._active_neuromod_delta is None


def test_model_begin_segment_preserves_carryover_by_default():
    """Under the external-surprise design, the previous step's neuromod
    snapshot must be preserved into the next segment so `_active_neuromod_delta`
    is non-None during the next forward — otherwise neuromod params get
    no gradient. Callers that want to clear it (e.g., across truly
    independent runs) can pass `clear_neuromod_carryover=True`."""
    w = _make_tiny_wrapper()
    w.train()
    # Run one full forward+backward pass to populate `_neuromod_input_*`
    # via update_plasticity.
    opt = torch.optim.AdamW([p for _, p in w.trainable_parameters()], lr=1e-4)
    batch = Phase1Batch(
        input_ids=torch.randint(0, 256, (2, 8)),
        target_ids=torch.randint(0, 256, (2, 8)),
    )
    phase1_pretrained_step(w, opt, batch, amp_dtype=None)
    assert w.memory._neuromod_input_ids is not None, (
        "post-step snapshot expected to be populated by update_plasticity"
    )
    snapshot_ids_before = w.memory._neuromod_input_ids

    w.begin_segment(bs=2)
    assert w.memory._neuromod_input_ids is snapshot_ids_before, (
        "model.begin_segment must preserve neuromod carryover by default"
    )

    # Explicit clear still works.
    w.begin_segment(bs=2, clear_neuromod_carryover=True)
    assert w.memory._neuromod_input_ids is None
    assert w.memory._neuromod_input_feats is None


def test_update_plasticity_preserves_grad_on_active_neuromod_delta():
    """Regression for the @torch.no_grad() decorator bug on update_plasticity
    (Codex audit 2026-05-04). The function MUST NOT be wrapped in no_grad
    because the trailing `_begin_plastic_window()` rebuilds `_active_neuromod_delta`
    by running the neuromod forward — under no_grad, that delta has no
    grad_fn, and the next segment's loss can't train the neuromod via the
    REINFORCE / phase-1-AR routing path. Symptom is silent: the model
    "trains" (W_in/W_out/walker still update) but the neuromod is stuck
    at init, which is the most-important learnable component.
    """
    w = _make_tiny_wrapper()
    w.train()
    opt = torch.optim.AdamW([p for _, p in w.trainable_parameters()], lr=1e-4)
    batch = Phase1Batch(
        input_ids=torch.randint(0, 256, (2, 8)),
        target_ids=torch.randint(0, 256, (2, 8)),
    )
    # Run one full step — this exercises update_plasticity on a real
    # snapshot path (post-step the snapshot is populated, then
    # _begin_plastic_window builds the next delta).
    phase1_pretrained_step(w, opt, batch, amp_dtype=None)
    delta = w.memory._active_neuromod_delta
    assert delta is not None, (
        "_active_neuromod_delta should have been rebuilt by _begin_plastic_window "
        "at the end of update_plasticity"
    )
    # The non-zero edge entries must carry grad_fn back to neuromod params.
    # If torch.no_grad is wrapping update_plasticity, this assertion fails.
    assert delta.requires_grad, (
        "_active_neuromod_delta has no grad — neuromod will get NO learning signal "
        "next segment. Probable cause: @torch.no_grad() on update_plasticity."
    )
    assert delta.grad_fn is not None, (
        "_active_neuromod_delta has no grad_fn — same root cause as requires_grad=False"
    )


# -- 6. phase1 target_ids contract --


def test_phase1_step_rejects_pre_shifted_targets():
    w = _make_tiny_wrapper()
    w.train()
    opt = torch.optim.AdamW([p for _, p in w.trainable_parameters()], lr=1e-4)
    # Pre-shifted targets (length T-1 instead of T) — common dataloader bug.
    bad_batch = Phase1Batch(
        input_ids=torch.randint(0, 256, (2, 8)),
        target_ids=torch.randint(0, 256, (2, 7)),
    )
    with pytest.raises(ValueError, match="must match input_ids"):
        phase1_pretrained_step(w, opt, bad_batch, amp_dtype=None)
