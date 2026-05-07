"""Smoke tests for phase-2 GRPO and the AR rollout primitive.

Validates:
- `autoregressive_rollout` produces K-rollout divergence in token space
  (different log_pi paths).
- `grpo_step` runs end-to-end: rewards → advantages → REINFORCE loss →
  gradient reaches the routing policy parameters (q_proj, k_all, neuromod).
- Phase-2 routing's log_pi accumulator is graph-connected to scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.pretrained.config import PretrainedGWConfig
from src.graph_walker.pretrained.integrated_lm import IntegratedLM
from src.graph_walker.pretrained.rollout import autoregressive_rollout
from src.graph_walker.pretrained.train_phase2 import grpo_step


def _make_tiny_llama(d_lm=32, n_layers=4, vocab=256):
    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=d_lm, intermediate_size=d_lm * 2,
        num_hidden_layers=n_layers, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=64,
        pad_token_id=0, bos_token_id=0, eos_token_id=0,
    )
    return LlamaForCausalLM(cfg)


def _tiny_walker_cfg(D_s, vocab, T=8):
    return GraphWalkerConfig(
        plane_rows=4, plane_cols=4, L=2,
        K=4, D_model=D_s, D_s=D_s, D_id=8,
        n_heads=2, n_hops=2,
        D_q_in=8, D_q_per_head=8, n_score_heads=2,
        K_horizons=4, K_buf=4, vocab_size=vocab,
        # Single-knob clock under external-surprise plasticity.
        mod_period=T, tbptt_block=T, segment_T=T,
        gumbel_tau_start=1.0, gumbel_tau_end=1.0, gumbel_anneal_steps=1,
        epsilon_start=0.0, epsilon_end=0.0, epsilon_anneal_steps=1,
        lambda_balance=0.0, use_neuromod=True,
        plasticity_mode="neuromod_only",
        neuromod_D_mod=16, neuromod_n_layers=1, neuromod_n_heads=2,
        neuromod_edge_hidden=16, neuromod_eta=1.0,
    )


def _make_tiny_wrapper(d_lm=32, vocab=256, T=8):
    hf = _make_tiny_llama(d_lm=d_lm, vocab=vocab)
    walker_cfg = _tiny_walker_cfg(D_s=d_lm, vocab=vocab, T=T)
    cfg = PretrainedGWConfig(
        model_name="random", inject_layer=2, d_mem=d_lm,
        memory=walker_cfg, T=T, bs=2, llama_dtype="fp32",
    )
    return IntegratedLM(cfg, hf_model=hf)


def _perturb_walker_weights(w):
    m = w.memory
    with torch.no_grad():
        m.prev_motor_proj.weight.normal_(std=0.05)
        m.decay_proj.weight.normal_(std=0.05)
        m.decay_proj.bias.normal_(std=0.05)
        m.readout.pred_head.proj.weight.normal_(std=0.05)
        if m.neuromod is not None:
            m.neuromod.edge_mlp[-1].weight.normal_(std=0.3)
            m.neuromod.edge_mlp[-1].bias.normal_(std=0.05)
            m.neuromod.blend_logit.fill_(0.0)


# ----- AR rollout primitive -----


def test_autoregressive_rollout_K_rollouts_diverge():
    torch.manual_seed(0)
    w = _make_tiny_wrapper()
    _perturb_walker_weights(w)

    K = 4
    prefix = torch.randint(0, 256, (1, 8))
    prefix_K = prefix.expand(K, -1).contiguous()

    out = autoregressive_rollout(
        w, prefix_K,
        gen_length=8, temperature=1.0, top_p=1.0,
        phase="phase2", grad_during_prefix=True, grad_during_gen=False,
    )

    # log_pi_mean should be K-shaped, finite, gradient-connected.
    assert out.log_pi_mean is not None
    assert out.log_pi_mean.shape == (K,)
    assert torch.isfinite(out.log_pi_mean).all()
    assert out.log_pi_mean.requires_grad, (
        "log_pi must remain graph-connected to walker params for REINFORCE"
    )
    # Per-step normalization keeps |log_pi_mean| at ~|log p| (a few nats),
    # not the unnormalized sum which would be ~thousands of nats.
    assert out.log_pi_mean.detach().abs().max().item() < 100.0, (
        "log_pi_mean magnitude unexpectedly large — normalization regression?"
    )
    # K rollouts should diverge in their generated tail (different samples).
    tail = out.new_tokens
    n_unique = tail.unique(dim=0).shape[0]
    assert n_unique >= 2, (
        f"K=4 rollouts collapsed to {n_unique} unique tails — phase-2 sampling broken"
    )


# ----- Phase 2 GRPO -----


def test_grpo_step_runs_and_grad_reaches_neuromod():
    torch.manual_seed(0)
    w = _make_tiny_wrapper()
    w.train()
    _perturb_walker_weights(w)
    opt = torch.optim.AdamW([p for _, p in w.trainable_parameters()], lr=1e-4)

    # Seed neuromod carryover by running one phase-1 step. Without this,
    # `_neuromod_input_*` is None and `_begin_plastic_window` produces a
    # None `_active_neuromod_delta` — routing falls back to E_bias_flat alone
    # and neuromod params receive no gradient. Real training never hits
    # this state because GRPO follows phase-1 in the cycle loop.
    from src.graph_walker.pretrained.train_phase1 import (
        Phase1Batch, phase1_pretrained_step,
    )
    seed_batch = Phase1Batch(
        input_ids=torch.randint(0, 256, (2, 8)),
        target_ids=torch.randint(0, 256, (2, 8)),
    )
    phase1_pretrained_step(w, opt, seed_batch, amp_dtype=None)

    prefix = torch.randint(0, 256, (1, 8))
    reference = torch.randint(0, 256, (8,))

    # Reward fn that returns *some* variance so GRPO has signal.
    def reward(generated, ref):
        # Random rewards — we just want the loss to be finite + grad to flow.
        return torch.randn(generated.shape[0])

    stats = grpo_step(
        w, opt,
        prefix_ids=prefix, reference_cont=reference,
        reward_fn=reward, num_rollouts=4, gen_length=4,
        temperature=1.0, top_p=1.0,
    )
    assert torch.isfinite(torch.tensor(stats.loss))

    # Neuromod is the explicit phase-2 policy head — its gradient must
    # flow through log_pi · advantage.
    grad_norms = {
        n: p.grad.abs().sum().item() if p.grad is not None else 0.0
        for n, p in w.memory.named_parameters()
        if "neuromod" in n
    }
    has_some_neuromod_grad = any(v > 0 for v in grad_norms.values())
    assert has_some_neuromod_grad, (
        "GRPO step left every neuromod param with zero/None gradient — "
        f"REINFORCE chain broken. Grad norms: {grad_norms}"
    )


def test_phase2_log_pi_resets_at_begin_segment():
    """Per-segment reset of the log_pi accumulator. Without it, GRPO
    would credit the policy for decisions made in earlier segments."""
    w = _make_tiny_wrapper()
    w.memory.phase = "phase2"
    w.train()
    w.begin_segment(bs=2)
    h = torch.randn(2, w.config.d_mem)
    w.memory.step_core_from_h(h)
    assert w.memory._log_pi_sum is not None
    assert w.memory._log_pi_sum.shape == (2,)
    # New segment wipes the accumulator.
    w.begin_segment(bs=2)
    assert w.memory._log_pi_sum is None
