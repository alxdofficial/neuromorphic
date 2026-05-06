"""Regression tests for the bug-sweep fixes (post 108540a):

1. log_pi_step is returned by `_step_core_pure` (not mutated on self)
   and folded into `_log_pi_sum` by `_apply_step_state`. Pure-function
   contract preserved → safe under torch.compile / checkpoint.
2. `wrapper.current_phase` is propagated to `memory.phase` inside
   `wrapper.forward()`. Setting the wrapper-level phase alone is now
   enough; previously it was silently ignored (memory.phase stayed at
   "phase1").
"""

from __future__ import annotations

import pytest  # noqa: F401  (kept for any future regression that needs raises)
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.graph_walker import GraphWalkerMemory, WalkerCorePureOutput
from src.graph_walker.pretrained.config import PretrainedGWConfig
from src.graph_walker.pretrained.llm_wrapper import GraphWalkerPretrainedLM
from src.graph_walker.standalone import StandaloneLM


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


# Bug 1: pure-function contract for _step_core_pure


def test_log_pi_step_returned_in_pure_output_not_mutated_on_self():
    """`_step_core_pure` must NOT mutate `self._log_pi_sum`. Instead, it
    returns `log_pi_step` in `WalkerCorePureOutput`. The fold to
    `self._log_pi_sum` happens in `_apply_step_state`. This keeps the
    function safe under torch.compile and gradient checkpointing
    (recompute path would otherwise double-count log_pi)."""
    torch.manual_seed(0)
    lm = StandaloneLM(_tiny_cfg()).cpu()
    lm.train()
    lm.memory.phase = "phase2"
    lm.memory.begin_segment(B=2, device=torch.device("cpu"))
    log_pi_before = lm.memory._log_pi_sum

    # Call _step_core_pure DIRECTLY (mimicking the compile / checkpoint path).
    cfg = lm.cfg
    tau, eps = lm.memory._schedule_tensors(torch.zeros(1, dtype=torch.long))
    e_bias = lm.memory._active_e_bias()
    lm.memory._ensure_block_caches(lm.memory.tied_token_emb.weight)
    out = lm.memory._step_core_pure(
        lm.memory.s, lm.memory.walker_pos, lm.memory.walker_state,
        lm.memory.prev_motor, e_bias,
        torch.zeros(2, dtype=torch.int64), tau, eps, is_new_window=True,
    )
    # Pure function: must NOT have touched self._log_pi_sum.
    assert lm.memory._log_pi_sum is log_pi_before, (
        "_step_core_pure mutated self._log_pi_sum — pure-function contract broken"
    )
    # Returned dataclass carries the routing log_pi for this step.
    assert out.log_pi_step is not None
    assert out.log_pi_step.shape == (2,)
    assert out.log_pi_step.requires_grad


def test_apply_step_state_folds_log_pi_into_self_sum():
    torch.manual_seed(1)
    lm = StandaloneLM(_tiny_cfg()).cpu()
    lm.train()
    lm.memory.phase = "phase2"
    lm.memory.begin_segment(B=2, device=torch.device("cpu"))

    # Two consecutive step_core_from_h calls with phase=phase2 should
    # accumulate log_pi (anchor on first call, per-token routing on both).
    h = torch.randn(2, lm.cfg.D_s)
    lm.memory.step_core_from_h(h)
    sum_after_1 = lm.memory._log_pi_sum.detach().clone()
    lm.memory.step_core_from_h(h)
    sum_after_2 = lm.memory._log_pi_sum
    # Two more routing decisions on step 2 → sum strictly increased in
    # absolute magnitude (could be more or less negative; check non-equal).
    assert not torch.equal(sum_after_1, sum_after_2)


# Bug 2: wrapper.current_phase propagation


def test_wrapper_current_phase_propagates_to_memory_phase():
    """Setting `wrapper.current_phase = "phase2"` must actually flip the
    routing path. Pre-fix, this was silently ignored — routing still ran
    Gumbel-STE because routing reads `memory.phase`, not the wrapper's."""
    torch.manual_seed(0)
    hf = _make_tiny_llama()
    # Integration walker requires segment_T == mod_period == tbptt_block.
    walker_cfg = _tiny_cfg(
        D_s=32, D_model=32, vocab_size=256,
        segment_T=8, mod_period=8, tbptt_block=8,
        plasticity_mode="neuromod_only",
    )
    cfg = PretrainedGWConfig(
        model_name="random", inject_layer=2, d_mem=32,
        memory=walker_cfg, T=8, bs=2, llama_dtype="fp32",
    )
    w = GraphWalkerPretrainedLM(cfg, hf_model=hf)
    w.train()
    w.current_phase = "phase2"
    # Before forward: memory.phase still default "phase1".
    assert w.memory.phase == "phase1"
    # Forward propagates.
    input_ids = torch.randint(0, 256, (2, 8))
    w.reset_memory(bs=2)
    w(input_ids)
    assert w.memory.phase == "phase2", (
        "wrapper.current_phase did not propagate to memory.phase — "
        "phase-2 routing would silently fall back to Gumbel-STE."
    )
    # log_pi accumulator should now be populated (phase-2 fired).
    assert w.memory._log_pi_sum is not None


