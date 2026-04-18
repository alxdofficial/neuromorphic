"""Smoke tests for the pretrained-LM + memory wrapper.

These tests require meta-llama/Llama-3.2-1B in the HF cache. Skip if absent
so CI / devs without access aren't blocked.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM


def _hf_cache_has_1b():
    import huggingface_hub
    try:
        huggingface_hub.snapshot_download(
            "meta-llama/Llama-3.2-1B", local_files_only=True,
            allow_patterns=["config.json"])
        return True
    except Exception:
        return False


requires_llama = pytest.mark.skipif(
    not _hf_cache_has_1b(),
    reason="Llama-3.2-1B not present in HF cache")


@requires_llama
def test_wrapper_loads_and_freezes_backbone():
    """Constructor loads the model and freezes backbone — only the injection
    projections, scale, and the memory graph should have requires_grad."""
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory

    cfg = PretrainedConfig.llama_1b()
    model = PretrainedLMWithMemory(cfg)

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    L = cfg.inject_layer

    # W_in / W_out / scale live on the injected layer; memory.* is its own
    # nn.Module subtree and everything there should also be trainable.
    inject_required = {
        f"llama.model.layers.{L}.W_in.weight",
        f"llama.model.layers.{L}.W_out.weight",
        f"llama.model.layers.{L}.scale",
    }
    assert inject_required.issubset(trainable), (
        f"missing inject params in trainable set; got {trainable - set()}")
    assert all(t.startswith(f"llama.model.layers.{L}.") or t.startswith("memory.")
               for t in trainable), (
        "unexpected trainable params outside mem_inject/memory: "
        f"{[t for t in trainable if not t.startswith(f'llama.model.layers.{L}.') and not t.startswith('memory.')]}")

    # Frozen backbone: embed / layers[0] / final norm / lm_head must have
    # requires_grad=False.
    assert not model.llama.model.embed_tokens.weight.requires_grad
    assert not model.llama.model.layers[0].self_attn.q_proj.weight.requires_grad
    assert not model.llama.model.norm.weight.requires_grad
    assert not model.llama.lm_head.weight.requires_grad


@requires_llama
def test_scale_zero_reproduces_vanilla_llama():
    """With scale pinned to 0, wrapper output == vanilla Llama output exactly."""
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory

    torch.manual_seed(0)

    cfg = PretrainedConfig.llama_1b()
    wrapper = PretrainedLMWithMemory(cfg)
    wrapper.train(False)

    vanilla = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.float32)
    vanilla.train(False)

    with torch.no_grad():
        wrapper.mem_inject.scale.zero_()

    input_ids = torch.randint(0, 128256, (2, 16))
    with torch.no_grad():
        out_wrapper = wrapper(input_ids).logits
        out_vanilla = vanilla(input_ids=input_ids).logits

    assert out_wrapper.shape == out_vanilla.shape
    # Bit-exact: scale=0 means the forward path is literally
    # `orig_layer(hidden_states)` — identical floating point arithmetic.
    assert torch.equal(out_wrapper, out_vanilla), (
        f"wrapper diverges from vanilla. "
        f"max abs diff = {(out_wrapper - out_vanilla).abs().max().item()}")


@requires_llama
def test_scale_nonzero_with_no_memory_is_still_transparent():
    """memory_fn=None + nonzero scale still bypasses (defensive path)."""
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory

    cfg = PretrainedConfig.llama_1b()
    wrapper = PretrainedLMWithMemory(cfg, attach_memory=False)
    wrapper.train(False)

    vanilla = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.float32)
    vanilla.train(False)

    # Scale stays at init (2.0). memory is not attached — the wrapper's
    # forward() goes through the no-memory path, which is bit-for-bit vanilla.
    input_ids = torch.randint(0, 128256, (1, 8))
    with torch.no_grad():
        out_wrapper = wrapper(input_ids).logits
        out_vanilla = vanilla(input_ids=input_ids).logits

    assert torch.equal(out_wrapper, out_vanilla)


@requires_llama
def test_memory_wired_forward_runs_and_differs_from_vanilla():
    """With memory attached and scale>0, forward runs end-to-end and produces
    logits distinct from vanilla Llama (because the injection residual is
    nonzero). Shape + finiteness are the minimum bar."""
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory

    torch.manual_seed(0)
    cfg = PretrainedConfig.llama_1b()
    # T_forward must be >= modulation_interval so the modulator fires at
    # least once (otherwise the write path has zero gradient, fine for a
    # forward-only test but pointless).
    T = cfg.memory.modulation_interval                  # 16
    wrapper = PretrainedLMWithMemory(cfg)
    wrapper.train(False)
    wrapper.reset_memory(bs=1)

    vanilla = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.float32)
    vanilla.train(False)

    input_ids = torch.randint(0, cfg.vocab_size_lm, (1, T))
    with torch.no_grad():
        out_wrapper = wrapper(input_ids).logits
        out_vanilla = vanilla(input_ids=input_ids).logits

    assert out_wrapper.shape == out_vanilla.shape
    assert torch.isfinite(out_wrapper).all()
    # Must diverge from vanilla — the inject path is active.
    assert not torch.equal(out_wrapper, out_vanilla)
    # Aux memory-pred loss populated and finite.
    assert wrapper._last_mem_pred_loss is not None
    assert torch.isfinite(wrapper._last_mem_pred_loss)


@requires_llama
def test_memory_wired_backward_reaches_trainable_params():
    """Loss.backward() produces nonzero grads on W_in / W_out / scale and on
    the memory graph's modulator / codebook / decoder. Frozen backbone has
    no grads."""
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory

    torch.manual_seed(0)
    # Override tbptt_block so the segment runs with no intra-segment detach.
    # Default tbptt_block=16 matches mod_interval=16, so the newly-written
    # W gets detached at t=16 and the modulator's update never reaches
    # a readout that feeds into the loss.
    from src.model.config import Config as MemoryConfig
    mem_cfg = MemoryConfig.tier_a(D=2048, tbptt_block=64)
    cfg = PretrainedConfig.llama_1b(memory=mem_cfg)
    # T must be > modulation_interval so readouts AFTER the modulator fire
    # get into the loss.
    T = 2 * cfg.memory.modulation_interval            # 32
    wrapper = PretrainedLMWithMemory(cfg)
    wrapper.train()
    wrapper.reset_memory(bs=1)

    input_ids = torch.randint(0, cfg.vocab_size_lm, (1, T))
    # Minimal CE target: shift-by-one on input_ids.
    target_ids = torch.cat([input_ids[:, 1:], input_ids[:, :1]], dim=1)

    out = wrapper(input_ids)
    ce = torch.nn.functional.cross_entropy(
        out.logits.reshape(-1, out.logits.shape[-1]),
        target_ids.reshape(-1))
    # Add memory aux loss to exercise the mem_pred path gradient too.
    loss = ce + 0.1 * wrapper._last_mem_pred_loss
    loss.backward()

    L = cfg.inject_layer
    mem_inj = wrapper.mem_inject

    # Injection projections + scale must all have gradient.
    assert mem_inj.W_in.weight.grad is not None
    assert mem_inj.W_out.weight.grad is not None
    assert mem_inj.scale.grad is not None
    assert torch.isfinite(mem_inj.W_in.weight.grad).all()
    assert torch.isfinite(mem_inj.W_out.weight.grad).all()
    assert torch.isfinite(mem_inj.scale.grad).all()
    # Each must be nonzero (modulo zero-init in decoder's last layer —
    # not applicable here).
    assert mem_inj.W_in.weight.grad.abs().sum() > 0
    assert mem_inj.W_out.weight.grad.abs().sum() > 0
    assert mem_inj.scale.grad.abs().sum() > 0

    # Memory graph: modulator first-layer + codebook + inject_w should all
    # receive gradient.
    assert wrapper.memory.modulator.tok_proj[0].weight.grad is not None
    assert wrapper.memory.discrete_policy.codebook.grad is not None
    assert wrapper.memory.inject_w.grad is not None

    # Frozen backbone: no grads on Llama's embed / layers[0] / norm.
    assert wrapper.llama.model.embed_tokens.weight.grad is None
    assert wrapper.llama.model.layers[0].self_attn.q_proj.weight.grad is None
    assert wrapper.llama.model.norm.weight.grad is None
    assert wrapper.llama.lm_head.weight.grad is None


@requires_llama
def test_full_optimizer_step_runs():
    """Forward → CE → backward → optimizer.step for one iteration. The
    trainable params must change after step; frozen params must not."""
    from src.model.config import Config as MemoryConfig
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory

    torch.manual_seed(0)
    mem_cfg = MemoryConfig.tier_a(D=2048, tbptt_block=64)
    cfg = PretrainedConfig.llama_1b(memory=mem_cfg)
    wrapper = PretrainedLMWithMemory(cfg)
    wrapper.train()
    wrapper.reset_memory(bs=1)

    trainable = [p for _, p in wrapper.trainable_parameters()]
    opt = torch.optim.AdamW(trainable, lr=1e-3)

    # Snapshot a couple of params to check they move.
    W_in_before = wrapper.mem_inject.W_in.weight.detach().clone()
    scale_before = wrapper.mem_inject.scale.detach().clone()
    # And one frozen param to check it doesn't move.
    embed_before = wrapper.llama.model.embed_tokens.weight.detach().clone()

    T = 2 * cfg.memory.modulation_interval
    input_ids = torch.randint(0, cfg.vocab_size_lm, (1, T))
    target_ids = torch.cat([input_ids[:, 1:], input_ids[:, :1]], dim=1)

    out = wrapper(input_ids)
    ce = torch.nn.functional.cross_entropy(
        out.logits.reshape(-1, out.logits.shape[-1]),
        target_ids.reshape(-1))
    loss = ce + 0.1 * wrapper._last_mem_pred_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    assert not torch.equal(wrapper.mem_inject.W_in.weight, W_in_before), (
        "W_in did not update after optimizer.step")
    assert not torch.equal(wrapper.mem_inject.scale, scale_before), (
        "scale did not update after optimizer.step")
    assert torch.equal(wrapper.llama.model.embed_tokens.weight, embed_before), (
        "frozen embed_tokens weight changed — backbone not frozen properly")


@requires_llama
def test_multi_segment_tbptt_carries_memory_state():
    """Two consecutive forward calls: memory state must persist across them,
    and detach_memory() must break the gradient graph between segments."""
    from src.model.config import Config as MemoryConfig
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory

    torch.manual_seed(0)
    mem_cfg = MemoryConfig.tier_a(D=2048, tbptt_block=64)
    cfg = PretrainedConfig.llama_1b(memory=mem_cfg)
    wrapper = PretrainedLMWithMemory(cfg)
    wrapper.train()
    wrapper.reset_memory(bs=1)

    T = 2 * cfg.memory.modulation_interval
    input_ids_1 = torch.randint(0, cfg.vocab_size_lm, (1, T))
    input_ids_2 = torch.randint(0, cfg.vocab_size_lm, (1, T))

    # First forward: memory starts at zero state, produces a snapshot.
    _ = wrapper(input_ids_1)
    W_after_seg1 = wrapper.memory.W.detach().clone()
    assert wrapper.memory.is_initialized
    assert W_after_seg1.abs().sum() > 0, (
        "W is all-zero after first segment; modulator never wrote")

    # Detach between segments (simulates TBPTT boundary).
    wrapper.detach_memory()

    # Second forward: state should continue evolving from W_after_seg1,
    # not reset to zeros.
    _ = wrapper(input_ids_2)
    W_after_seg2 = wrapper.memory.W.detach()
    # W must have evolved further — different from the post-seg1 snapshot.
    assert not torch.equal(W_after_seg2, W_after_seg1), (
        "memory state did not change across the second segment")
    assert torch.isfinite(W_after_seg2).all()
