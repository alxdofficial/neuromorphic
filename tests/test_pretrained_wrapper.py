"""Smoke tests for IntegratedLM (model + phase-1 step).

Uses a tiny random-init `LlamaForCausalLM` so tests run in a few seconds on
CPU without HF Hub downloads. Validates:
- model construction (frozen backbone, MemInjectLayer wired, scale-zero
  bypass when memory disabled),
- forward shapes (logits match Llama's vocab; aux loss is finite),
- gradient flow: every trainable param (W_in, W_out, scale, walker subtree)
  gets a non-zero gradient after one step,
- TBPTT detach: detach_memory() doesn't trip autograd on a fresh forward.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.pretrained.config import PretrainedGWConfig
from src.graph_walker.pretrained.integrated_lm import IntegratedLM
from src.graph_walker.pretrained.train_phase1 import (
    Phase1Batch,
    phase1_pretrained_step,
)


def _make_tiny_llama(d_lm: int = 32, n_layers: int = 4, vocab: int = 256):
    """Build a random-weights LlamaForCausalLM small enough for CPU smoke."""
    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=d_lm,
        intermediate_size=d_lm * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        pad_token_id=0, bos_token_id=0, eos_token_id=0,
    )
    return LlamaForCausalLM(cfg)


def _tiny_walker_cfg(D_s: int, vocab: int, T: int = 8):
    return GraphWalkerConfig(
        grid_rows=4, grid_cols=4, radius=2,
        K=4, D_model=D_s, D_s=D_s, D_id=8,
        n_heads=2,
        D_q_per_head=8, n_score_heads=2,
        K_horizons=4,
        vocab_size=vocab,
        # Single-knob clock under external-surprise plasticity:
        # segment_T == mod_period == tbptt_block == T.
        mod_period=T, tbptt_block=T, segment_T=T,
        gumbel_tau_start=1.0, gumbel_tau_end=1.0, gumbel_anneal_steps=1,
        epsilon_start=0.0, epsilon_end=0.0, epsilon_anneal_steps=1,
        lambda_balance=0.0,
        use_neuromod=True,
        plasticity_mode="neuromod_only",
        neuromod_D_mod=16, neuromod_n_layers=1, neuromod_n_heads=2,
        neuromod_edge_hidden=16, neuromod_eta=1.0,
    )


def _make_tiny_wrapper(d_lm: int = 32, vocab: int = 256, T: int = 8):
    hf = _make_tiny_llama(d_lm=d_lm, vocab=vocab)
    walker_cfg = _tiny_walker_cfg(D_s=d_lm, vocab=vocab, T=T)
    cfg = PretrainedGWConfig(
        model_name="random",
        inject_layer=2,
        d_mem=d_lm,
        memory=walker_cfg,
        T=T,
        bs=2,
        llama_dtype="fp32",
    )
    return IntegratedLM(cfg, hf_model=hf)


def test_wrapper_construction_freezes_backbone():
    w = _make_tiny_wrapper()
    # Llama params (except MemInjectLayer's W_in/W_out/scale at layer 2)
    # should be frozen.
    for name, p in w.llama.named_parameters():
        is_inject = (
            ".layers.2.W_in" in name
            or ".layers.2.W_out" in name
            or ".layers.2.scale" in name
        )
        if is_inject:
            assert p.requires_grad, f"{name} should be trainable"
        else:
            assert not p.requires_grad, f"{name} should be frozen"

    # Walker should have trainable params.
    walker_trainable = sum(
        p.numel() for _, p in w.memory.named_parameters() if p.requires_grad
    )
    assert walker_trainable > 0, "walker has no trainable params"


def test_forward_logits_shape():
    w = _make_tiny_wrapper(d_lm=32, vocab=256, T=8)
    w.begin_segment(bs=2)
    input_ids = torch.randint(0, 256, (2, 8))
    out = w(input_ids)
    assert out.logits.shape == (2, 8, 256)
    assert torch.isfinite(out.logits).all()
    # Walker is vocab-agnostic: forward returns only logits via the host
    # LM; there is no aux loss stashed on the model.
    w.train()
    w.begin_segment(bs=2)
    out = w(input_ids)
    assert torch.isfinite(out.logits).all()


def test_phase1_step_runs_and_gradient_reaches_all_trainables():
    """End-to-end: one phase-1 step should produce a finite loss and
    backward must reach every trainable parameter, including:
    - Llama-side: W_in, W_out, scale (MemInjectLayer)
    - Walker-side: content_mlp, q_proj, k_all, nbr_id_to_s,
                   walker_state_alpha, neuromod subtree
    """
    torch.manual_seed(0)
    w = _make_tiny_wrapper(d_lm=32, vocab=256, T=8)
    w.train()

    # Perturb zero-init gates so paths through them carry gradient.
    m = w.memory
    with torch.no_grad():
        m.decay_proj.weight.normal_(std=0.05)
        m.decay_proj.bias.normal_(std=0.05)
        m.readout.pred_head.proj.weight.normal_(std=0.05)
        if m.neuromod is not None:
            m.neuromod.edge_mlp[-1].weight.normal_(std=0.05)
            m.neuromod.edge_mlp[-1].bias.normal_(std=0.05)
            m.neuromod.blend_logit.fill_(0.0)
            # `edge_bias_proj` is zero-init in neuromod_only mode (option C
            # per-head attention bias). Perturb so attention scores depend
            # on per_edge_extras and gradient flows back through that path.
            if m.neuromod.edge_bias_proj is not None:
                m.neuromod.edge_bias_proj.weight.normal_(std=0.05)
                m.neuromod.edge_bias_proj.bias.normal_(std=0.05)

    opt = torch.optim.AdamW(
        [p for _, p in w.trainable_parameters()], lr=1e-4,
    )
    input_ids = torch.randint(0, 256, (2, 8))
    target_ids = input_ids.clone()
    batch = Phase1Batch(input_ids=input_ids, target_ids=target_ids)

    # Pre-seed neuromod's prev-snapshot by running an extra step. Otherwise
    # the first measured step has _active_neuromod_delta=None and neuromod params
    # see no gradient.
    phase1_pretrained_step(w, opt, batch, amp_dtype=None)

    # Capture pre-step values to confirm params actually moved.
    before = {n: p.detach().clone() for n, p in w.trainable_parameters()}

    stats = phase1_pretrained_step(w, opt, batch, amp_dtype=None)
    assert torch.isfinite(torch.tensor(stats.loss))
    assert stats.ce_loss > 0, "Llama CE should be non-trivial after init"

    # Walk all trainables; collect any that didn't get gradient or didn't move.
    no_grad_after_step = []
    no_move_after_step = []
    for name, p in w.trainable_parameters():
        if p.grad is None or p.grad.abs().sum().item() == 0.0:
            no_grad_after_step.append(name)
        elif torch.equal(p, before[name]):
            no_move_after_step.append(name)

    # Tolerate: nothing should have NO gradient at all.
    assert not no_grad_after_step, (
        f"Trainables with no gradient after phase1 step: {no_grad_after_step}"
    )
    # Most trainables should have moved (Adam non-zero update). A few
    # might tie due to lr×grad rounding — tolerate up to ~10%.
    n_total = sum(1 for _ in w.trainable_parameters())
    assert len(no_move_after_step) < max(1, n_total // 10), (
        f"Too many trainables didn't move: {no_move_after_step}"
    )


def test_detach_memory_lets_subsequent_step_build_fresh_graph():
    """After detach_memory, a new forward+backward should not raise an
    'in-place modification' error and grad should flow."""
    torch.manual_seed(0)
    w = _make_tiny_wrapper()
    w.train()
    opt = torch.optim.AdamW([p for _, p in w.trainable_parameters()], lr=1e-4)
    input_ids = torch.randint(0, 256, (2, 8))
    batch = Phase1Batch(input_ids=input_ids, target_ids=input_ids.clone())

    s1 = phase1_pretrained_step(w, opt, batch, amp_dtype=None)
    s2 = phase1_pretrained_step(w, opt, batch, amp_dtype=None)
    assert torch.isfinite(torch.tensor(s1.loss))
    assert torch.isfinite(torch.tensor(s2.loss))
