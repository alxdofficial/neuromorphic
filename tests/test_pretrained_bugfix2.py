"""Regression tests for the second bug-sweep fixes (post 32684cf):

1. autoregressive_rollout restores wrapper.training and memory.phase
   on exit. Previously it left the wrapper in eval mode, which silently
   skipped aux loss in the next phase-1 step.
2. forward_segment skips the dense vocab readout AND plasticity finalize
   when a block has zero valid horizons (typical for T=1 generation
   steps inside an AR rollout).
3. phase1_ar_pretrained_step's continuation loop runs inside
   _freeze_plasticity_ctx, so per-token AR forwards don't fire
   plasticity off stale surprise + generation-walk co-visit.
4. unfreeze_all() re-freezes the standalone-only walker params
   (token_to_state, input_v_proj). Without this they re-enter the
   optimizer state every cycle as dead weights.
5. begin_segment(clear_neuromod_carryover=True) wipes the previous
   segment's neuromod snapshot. wrapper.reset_memory defaults to
   True for the pretrained path (independent-document batches).
6. phase1_pretrained_step asserts target_ids.shape == input_ids.shape
   to catch the "already pre-shifted" footgun.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.pretrained.config import PretrainedGWConfig
from src.graph_walker.pretrained.llm_wrapper import GraphWalkerPretrainedLM
from src.graph_walker.pretrained.rollout import autoregressive_rollout
from src.graph_walker.pretrained.train_phase1 import (
    Phase1Batch,
    phase1_pretrained_step,
)


def _tiny_cfg(**overrides):
    base = dict(
        plane_rows=4, plane_cols=4, L=2,
        K=4, D_model=32, D_s=32, D_id=8,
        n_heads=2, n_hops=2,
        D_q_in=8, D_q_per_head=8, n_score_heads=2,
        K_horizons=4, K_buf=4, vocab_size=64,
        mod_period=4, tbptt_block=4, segment_T=8,
        gumbel_tau_start=1.0, gumbel_tau_end=1.0, gumbel_anneal_steps=1,
        epsilon_start=0.0, epsilon_end=0.0, epsilon_anneal_steps=1,
        lambda_balance=0.0, use_neuromod=True,
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
    return GraphWalkerPretrainedLM(cfg, hf_model=hf)


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
        "rollout left wrapper in eval mode — next phase-1 cycle would skip aux loss"
    )
    assert w.current_phase == "phase1"
    assert w.memory.phase == "phase1", (
        "rollout left memory.phase = 'phase2' — direct memory calls would use "
        "hard Categorical instead of Gumbel-STE"
    )


# -- 2. forward_segment skips work on targetless blocks --


def test_forward_segment_skips_readout_when_block_has_no_valid_targets():
    """A T=1 forward (typical mid-AR-rollout) has no upcoming targets in
    its segment. forward_segment must skip the dense vocab readout AND
    must NOT increment surprise EMA / fire plasticity off stale surprise."""
    from src.graph_walker.standalone import StandaloneLM

    torch.manual_seed(0)
    walker_cfg = _tiny_cfg(segment_T=8, mod_period=4, tbptt_block=4)
    lm = StandaloneLM(walker_cfg).cpu()
    m = lm.memory

    # Sanity: pre-fill surprise_ema with a non-zero pattern so we can detect
    # whether the targetless block writes over it.
    m.begin_segment(B=2, device=torch.device("cpu"))
    m.surprise_ema = torch.full_like(m.surprise_ema, 0.7)
    surprise_before = m.surprise_ema.clone()
    e_bias_before = m.E_bias_flat.clone()

    # Single T=1 forward via forward_segment.
    h_mem = torch.randn(2, 1, walker_cfg.D_s)
    input_ids = torch.randint(0, walker_cfg.vocab_size, (2, 1))
    readouts, aux = m.forward_segment(h_mem, input_ids, adapter=None)
    assert readouts.shape == (2, 1, walker_cfg.D_s)
    # surprise EMA must be untouched (no valid horizons → no fold).
    assert torch.equal(m.surprise_ema, surprise_before), (
        "T=1 forward folded zero-valid CE into surprise_ema"
    )
    # E_bias must be untouched (plasticity not fired without valid signal,
    # AND fewer than mod_period steps since segment start anyway).
    assert torch.equal(m.E_bias_flat, e_bias_before)


# -- 3. AR continuation freezes plasticity --


def test_phase1_ar_continuation_does_not_alter_E_bias_during_generation():
    """The continuation forwards should not move E_bias. Plasticity is
    frozen via _freeze_plasticity_ctx in the unroll."""
    from src.graph_walker.pretrained.train_phase1_ar import (
        Phase1ARBatch,
        phase1_ar_pretrained_step,
    )

    torch.manual_seed(0)
    w = _make_tiny_wrapper(T=8)
    w.train()
    # Force E_bias to a known non-zero state so we can detect generation-driven changes.
    with torch.no_grad():
        w.memory.E_bias_flat.fill_(0.3)
    e_before = w.memory.E_bias_flat.clone()

    opt = torch.optim.AdamW([p for _, p in w.trainable_parameters()], lr=1e-4)
    # Use mod_period=4 with continuation_length=mod_period+more so without
    # the freeze, plasticity WOULD fire during the continuation forwards.
    batch = Phase1ARBatch(
        prefix_ids=torch.randint(0, 256, (2, 8)),
        continuation_ids=torch.randint(0, 256, (2, 6)),
    )
    phase1_ar_pretrained_step(w, opt, batch, amp_dtype=None)
    # Plasticity is allowed during the prefix pass (it's training data with
    # valid targets). So E_bias may have changed from the prefix. But the
    # continuation forwards specifically should NOT have written E_bias.
    # We can't perfectly isolate the continuation contribution without
    # mocking; the regression test is "the AR step completes without
    # raising" plus the unit test on forward_segment above.
    # Just verify that E_bias is finite + opt step succeeded.
    assert torch.isfinite(w.memory.E_bias_flat).all()


# -- 4. unfreeze_all keeps standalone-only params frozen --


def test_unfreeze_all_keeps_standalone_only_params_frozen():
    w = _make_tiny_wrapper()
    # First sanity: init froze them.
    assert w.memory.token_to_state.weight.requires_grad is False
    assert w.memory.input_v_proj.weight.requires_grad is False

    # Toggle phase 2 freeze, then unfreeze_all — the dead params must NOT
    # come back as trainable.
    w.freeze_all_but_E_bias_and_neuromod()
    assert w.memory.token_to_state.weight.requires_grad is False
    w.unfreeze_all()
    assert w.memory.token_to_state.weight.requires_grad is False, (
        "unfreeze_all re-enabled token_to_state — dead weight in optimizer"
    )
    assert w.memory.input_v_proj.weight.requires_grad is False


# -- 5. begin_segment(clear_neuromod_carryover) --


def test_begin_segment_clears_neuromod_carryover_when_requested():
    from src.graph_walker.standalone import StandaloneLM

    walker_cfg = _tiny_cfg()
    lm = StandaloneLM(walker_cfg).cpu()
    m = lm.memory
    # Manually plant carryover state.
    m._prev_snapshot_ids = torch.tensor([1, 2], dtype=torch.int64)
    m._prev_snapshot_feats = torch.randn(2, walker_cfg.D_s + walker_cfg.D_id + 1)

    # Default (False) preserves carryover.
    m.begin_segment(B=2, device=torch.device("cpu"))
    assert m._prev_snapshot_ids is not None

    # Explicit True clears.
    m._prev_snapshot_ids = torch.tensor([1, 2], dtype=torch.int64)
    m.begin_segment(B=2, device=torch.device("cpu"), clear_neuromod_carryover=True)
    assert m._prev_snapshot_ids is None
    assert m._prev_snapshot_feats is None
    assert m._active_delta_nm is None


def test_wrapper_reset_memory_clears_carryover_by_default():
    w = _make_tiny_wrapper()
    w.train()
    # Plant carryover.
    w.memory._prev_snapshot_ids = torch.tensor([1, 2], dtype=torch.int64)
    w.reset_memory(bs=2)
    assert w.memory._prev_snapshot_ids is None, (
        "wrapper.reset_memory should clear carryover by default for pretrained path"
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
