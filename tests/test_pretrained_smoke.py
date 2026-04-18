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


def test_pretrained_factory_defaults_align_memory_clocks():
    """Default pretrained driver knobs should match the default-created
    memory config so docs and runtime clocks stay aligned."""
    from src.pretrained.config import PretrainedConfig

    cfg = PretrainedConfig.llama_1b()
    assert cfg.memory.T == cfg.T
    assert cfg.memory.tbptt_block == cfg.tbptt_block


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

    # Bit-exact vanilla comparison requires fp32 on both sides — the default
    # bf16 path is numerically equivalent in production but not bit-identical
    # across the layer-wrap boundary. Explicitly opt in to fp32 here.
    cfg = PretrainedConfig.llama_1b(llama_dtype="fp32")
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

    # Same fp32 opt-in as the bit-exact test — this one also compares to
    # an fp32 vanilla Llama.
    cfg = PretrainedConfig.llama_1b(llama_dtype="fp32")
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
    # T_forward must be > msg_interval so `msg = MLP(h)` fires INSIDE the
    # segment and `received = W @ msg` can populate output-port neurons.
    # Otherwise h at output ports stays zero, readout is zero, and the
    # wrapper output equals vanilla regardless of memory being attached.
    # 2 * modulation_interval is the simplest bound that also gives the
    # modulator at least one fire whose ΔW gets consumed by downstream
    # readouts within the segment.
    T = 2 * cfg.memory.modulation_interval
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


@requires_llama
def test_llama_tokenizer_round_trips_and_feeds_wrapper():
    """The Llama-3.2 tokenizer (128K BPE) encodes/decodes cleanly and the
    encoded ids pass through PretrainedLMWithMemory without a shape or
    vocab mismatch."""
    from src.data.tokenizer import get_tokenizer
    from src.model.config import Config as MemoryConfig
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory

    tok = get_tokenizer("llama-3.2-1b")
    # Basic tokenizer sanity. 128256 vocab (Llama 3.2 shipping size).
    assert len(tok) == 128256
    assert tok.bos_token_id is not None

    text = "The quick brown fox jumps over the lazy dog. Memory is a test."
    ids = tok.encode(text, return_tensors="pt")
    assert ids.ndim == 2 and ids.shape[0] == 1
    round_tripped = tok.decode(ids[0].tolist(), skip_special_tokens=True)
    assert "quick brown fox" in round_tripped

    # Pad/truncate to exactly T=32 so the memory modulator fires.
    mem_cfg = MemoryConfig.tier_a(D=2048, tbptt_block=64)
    cfg = PretrainedConfig.llama_1b(memory=mem_cfg)
    T = 2 * cfg.memory.modulation_interval
    if ids.shape[1] < T:
        pad = torch.full((1, T - ids.shape[1]), tok.eos_token_id, dtype=ids.dtype)
        ids = torch.cat([ids, pad], dim=1)
    ids = ids[:, :T]

    wrapper = PretrainedLMWithMemory(cfg)
    wrapper.train(False)
    wrapper.reset_memory(bs=1)
    with torch.no_grad():
        out = wrapper(ids)
    assert out.logits.shape == (1, T, cfg.vocab_size_lm)
    assert torch.isfinite(out.logits).all()


@requires_llama
def test_multi_step_training_loop_runs_without_crashing():
    """Five consecutive train steps on synthetic data. Loss stays finite,
    trainable params receive updates each step, and memory state carries
    across TBPTT boundaries. No assertion on loss DECREASING (data is
    random noise) — just that the loop runs and numerics hold."""
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
    opt = torch.optim.AdamW(trainable, lr=1e-4)

    T = 2 * cfg.memory.modulation_interval
    losses = []
    W_in_snapshot = wrapper.mem_inject.W_in.weight.detach().clone()

    for step in range(5):
        input_ids = torch.randint(0, cfg.vocab_size_lm, (1, T))
        target_ids = torch.cat([input_ids[:, 1:], input_ids[:, :1]], dim=1)

        out = wrapper(input_ids)
        ce = torch.nn.functional.cross_entropy(
            out.logits.reshape(-1, out.logits.shape[-1]),
            target_ids.reshape(-1))
        loss = ce + 0.1 * wrapper._last_mem_pred_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        opt.step()
        wrapper.detach_memory()

        losses.append(loss.item())

    # All losses finite.
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    # W_in moved from its initial value over the 5 steps.
    W_in_after = wrapper.mem_inject.W_in.weight.detach()
    assert not torch.equal(W_in_after, W_in_snapshot)
    # Memory state is still initialized and finite after 5 detach cycles.
    assert wrapper.memory.is_initialized
    assert torch.isfinite(wrapper.memory.W).all()
    assert torch.isfinite(wrapper.memory.h).all()


@requires_llama
def test_phase1_training_function_runs_and_anneals_tau():
    """`run_phase1` executes N steps on synthetic data, accumulates per-step
    logs with finite losses, and anneals Gumbel tau from start→end."""
    from src.model.config import Config as MemoryConfig
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory
    from src.pretrained.train_phase1 import Phase1Batch, run_phase1

    torch.manual_seed(0)
    mem_cfg = MemoryConfig.tier_a(D=2048, tbptt_block=64)
    cfg = PretrainedConfig.llama_1b(memory=mem_cfg)
    wrapper = PretrainedLMWithMemory(cfg)
    wrapper.reset_memory(bs=1)

    trainable = [p for _, p in wrapper.trainable_parameters()]
    opt = torch.optim.AdamW(trainable, lr=1e-4)

    T = 2 * cfg.memory.modulation_interval

    def data_iter():
        while True:
            input_ids = torch.randint(0, cfg.vocab_size_lm, (1, T))
            target_ids = torch.cat([input_ids[:, 1:], input_ids[:, :1]], dim=1)
            yield Phase1Batch(input_ids=input_ids, target_ids=target_ids,
                              prev_token=None)

    logs = []
    run_phase1(
        wrapper, opt, data_iter(),
        steps=4,
        mem_pred_weight=0.1,
        gumbel_tau_start=1.0,
        gumbel_tau_end=0.3,
        anneal_across_steps=3,   # hit floor before final step
        on_step=logs.append,
    )

    assert len(logs) == 4
    # tau anneals linearly across 3 steps, then pins at 0.3.
    assert abs(logs[0].gumbel_tau - 1.0) < 1e-6
    assert logs[1].gumbel_tau < logs[0].gumbel_tau
    assert abs(logs[3].gumbel_tau - 0.3) < 1e-6
    # All losses + grad norms finite.
    for lg in logs:
        assert torch.isfinite(torch.tensor(lg.loss))
        assert torch.isfinite(torch.tensor(lg.ce))
        assert torch.isfinite(torch.tensor(lg.grad_norm))


def _wikitext_cached():
    try:
        from datasets import load_dataset
        load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                     streaming=True)
        return True
    except Exception:
        return False


requires_wikitext = pytest.mark.skipif(
    not _wikitext_cached(),
    reason="wikitext-2 not reachable / cached")


@requires_llama
@requires_wikitext
def test_phase1_on_real_wikitext_stream():
    """Run phase-1 for 3 steps on real tokenized wikitext. Confirms the
    end-to-end path — real text → Llama tokenizer → wrapper forward →
    memory write → backward → optimizer step — works without crashing
    and produces finite, sensible losses (start near vanilla Llama CE on
    wikitext, ~5-8 nats for a cold memory head)."""
    from datasets import load_dataset

    from src.data.tokenizer import get_tokenizer
    from src.model.config import Config as MemoryConfig
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory
    from src.pretrained.train_phase1 import Phase1Batch, run_phase1

    torch.manual_seed(0)
    mem_cfg = MemoryConfig.tier_a(D=2048, tbptt_block=64)
    cfg = PretrainedConfig.llama_1b(memory=mem_cfg)
    T = 2 * cfg.memory.modulation_interval    # 32 so modulator fires

    tok = get_tokenizer("llama-3.2-1b")

    # Harvest a buffer of Llama-3.2 token ids from wikitext-2 until we
    # have enough to build a few BS=1, T=32 batches.
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                      streaming=True)
    buf: list[int] = []
    needed = (T + 1) * 5      # 5 batches' worth of shift-by-one pairs
    for rec in ds:
        text = rec.get("text", "").strip()
        if not text:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        buf.extend(ids)
        if len(buf) >= needed:
            break
    assert len(buf) >= needed, f"wikitext too small after tokenization: {len(buf)}"

    def data_iter():
        i = 0
        while True:
            chunk = buf[i:i + T + 1]
            if len(chunk) < T + 1:
                i = 0
                chunk = buf[:T + 1]
            input_ids = torch.tensor(chunk[:T], dtype=torch.long).unsqueeze(0)
            target_ids = torch.tensor(chunk[1:T + 1], dtype=torch.long).unsqueeze(0)
            yield Phase1Batch(input_ids=input_ids, target_ids=target_ids,
                              prev_token=None)
            i += T

    wrapper = PretrainedLMWithMemory(cfg)
    wrapper.reset_memory(bs=1)
    trainable = [p for _, p in wrapper.trainable_parameters()]
    opt = torch.optim.AdamW(trainable, lr=1e-4)

    logs: list = []
    run_phase1(wrapper, opt, data_iter(), steps=3,
               mem_pred_weight=0.1,
               gumbel_tau_start=1.0, gumbel_tau_end=0.3,
               anneal_across_steps=3,
               on_step=logs.append)

    assert len(logs) == 3
    for lg in logs:
        assert torch.isfinite(torch.tensor(lg.loss))
        assert torch.isfinite(torch.tensor(lg.grad_norm))
        # Real-text CE on cold memory head: expect ballpark 5-20 nats.
        # Wide bound so the test isn't brittle to tokenizer/initialization.
        assert 0.5 < lg.loss < 50.0, f"loss {lg.loss} outside sanity range"


@requires_llama
def test_autoregressive_rollout_runs_and_diverges_across_k():
    """K=3 rollouts from the same prefix produce at least one pair of
    differing token streams. Divergence comes from stochastic memory code
    sampling across the batch dimension during prefix processing."""
    from src.model.config import Config as MemoryConfig
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory
    from src.pretrained.rollout import autoregressive_rollout

    torch.manual_seed(0)
    mem_cfg = MemoryConfig.tier_a(D=2048, tbptt_block=64)
    cfg = PretrainedConfig.llama_1b(memory=mem_cfg)
    wrapper = PretrainedLMWithMemory(cfg)

    T_prefix = 2 * cfg.memory.modulation_interval   # 32: modulator fires
    gen_length = 4
    K = 3
    prefix_ids = torch.randint(0, cfg.vocab_size_lm, (1, T_prefix))

    result = autoregressive_rollout(
        wrapper, prefix_ids, gen_length=gen_length, num_rollouts=K,
        temperature=1.0, seed=42)

    assert result.generated_ids.shape == (K, gen_length)
    assert result.prefix_ids.shape == (K, T_prefix)
    assert result.final_logits.shape == (K, cfg.vocab_size_lm)
    # All token ids are valid vocabulary indices.
    assert (result.generated_ids >= 0).all()
    assert (result.generated_ids < cfg.vocab_size_lm).all()
    # At least one pair of rollouts must differ — otherwise the memory code
    # sampling isn't differentiating across K, and phase-2 GRPO has no
    # statistical power. Token sampling noise also contributes, but that
    # alone is not the point: memory-driven divergence is.
    any_diff = False
    for i in range(K):
        for j in range(i + 1, K):
            if not torch.equal(result.generated_ids[i], result.generated_ids[j]):
                any_diff = True
                break
        if any_diff:
            break
    assert any_diff, "K rollouts all produced identical token streams"


@requires_llama
def test_autoregressive_rollout_restores_wrapper_mode():
    """The rollout helper should restore the caller's train/eval state and
    phase after it returns."""
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory
    from src.pretrained.rollout import autoregressive_rollout

    cfg = PretrainedConfig.llama_1b()
    wrapper = PretrainedLMWithMemory(cfg)
    wrapper.train(True)
    wrapper.current_phase = "phase1"

    prefix_ids = torch.randint(0, cfg.vocab_size_lm, (1, 4))
    _ = autoregressive_rollout(
        wrapper, prefix_ids, gen_length=2, num_rollouts=2, seed=0)

    assert wrapper.training is True
    assert wrapper.current_phase == "phase1"


@requires_llama
def test_phase2_grpo_step_runs_and_flows_gradient_to_modulator():
    """One GRPO step: K=4 rollouts from a shared prefix, per-rollout reward
    from token-match to a reference continuation, advantage normalization,
    policy-gradient loss backward. Must produce finite loss, a non-zero
    log π sum, and gradient on the modulator + codebook (the REINFORCE
    signal's actual targets)."""
    from src.model.config import Config as MemoryConfig
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory
    from src.pretrained.train_phase2 import grpo_step

    torch.manual_seed(0)
    mem_cfg = MemoryConfig.tier_a(D=2048, tbptt_block=64)
    cfg = PretrainedConfig.llama_1b(memory=mem_cfg)
    wrapper = PretrainedLMWithMemory(cfg)

    T_prefix = 2 * cfg.memory.modulation_interval     # 32
    gen_length = 4
    K = 4
    prefix_ids = torch.randint(0, cfg.vocab_size_lm, (1, T_prefix))
    reference = torch.randint(0, cfg.vocab_size_lm, (gen_length,))

    trainable = [p for _, p in wrapper.trainable_parameters()]
    opt = torch.optim.AdamW(trainable, lr=1e-4)

    # Capture a snapshot of codebook + modulator logit_head for post-step
    # diff check (GRPO signal must actually reach these).
    codebook_before = wrapper.memory.discrete_policy.codebook.detach().clone()
    logit_head_before = wrapper.memory.modulator.logit_head.weight.detach().clone()

    # Token-match reward on a random reference over a 128K vocab is ~0 for
    # every rollout (K=4 × length=4 matches vs 128K-way choice). That's
    # a real-world problem for GRPO smokes — no signal means no gradient.
    # For the smoke we want to verify PLUMBING: rollouts diverge → a
    # variance-bearing reward produces non-zero advantages → log_pi
    # backward reaches modulator. Use a reward tied to the first
    # generated token id (divergent rollouts → divergent first tokens →
    # divergent rewards).
    def stub_reward_fn(generated, reference):
        return generated[:, 0].float() / 10000.0   # [K], finite variance

    log = grpo_step(
        wrapper, opt,
        prefix_ids=prefix_ids, reference_cont=reference,
        num_rollouts=K, gen_length=gen_length, temperature=1.0,
        reward_fn=stub_reward_fn,
        seed=42,
    )

    assert torch.isfinite(torch.tensor(log.loss))
    assert torch.isfinite(torch.tensor(log.log_pi_mean))
    # Log π sum over a real segment must be strictly negative on average
    # (log-probs of sampled categoricals over K=2048 classes).
    assert log.log_pi_mean < 0
    # Stub reward has variance by construction (different first tokens
    # across K rollouts) so advantage magnitude must be > 0.
    assert log.advantage_max_abs > 0

    # Modulator + codebook must have moved (the policy gradient's
    # actual targets).
    assert not torch.equal(
        wrapper.memory.discrete_policy.codebook, codebook_before)
    assert not torch.equal(
        wrapper.memory.modulator.logit_head.weight, logit_head_before)


@requires_llama
def test_phase2_advantage_std_floor_clamps_noise_level_rewards():
    """When rewards are near-uniform (std ≪ floor), advantages must stay
    bounded. Without the floor (eps=1e-8), a std of ~1e-5 produces
    advantages of ~1e4 which explode gradients before clipping."""
    from src.model.config import Config as MemoryConfig
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory
    from src.pretrained.train_phase2 import grpo_step

    torch.manual_seed(0)
    mem_cfg = MemoryConfig.tier_a(D=2048, tbptt_block=64)
    cfg = PretrainedConfig.llama_1b(memory=mem_cfg)
    wrapper = PretrainedLMWithMemory(cfg)

    T_prefix = 2 * cfg.memory.modulation_interval
    gen_length = 4
    K = 4
    prefix_ids = torch.randint(0, cfg.vocab_size_lm, (1, T_prefix))
    reference = torch.randint(0, cfg.vocab_size_lm, (gen_length,))

    opt = torch.optim.AdamW(
        [p for _, p in wrapper.trainable_parameters()], lr=1e-4)

    # Construct a reward function whose outputs have *tiny* std (~1e-5)
    # but non-zero variance — the canonical noise-floor-exceeds-signal
    # early-training case.
    def noise_level_reward(generated, reference):
        base = torch.ones(generated.shape[0], device=generated.device) * 0.5
        noise = 1e-5 * torch.randn(generated.shape[0], device=generated.device)
        return base + noise

    log = grpo_step(
        wrapper, opt,
        prefix_ids=prefix_ids, reference_cont=reference,
        num_rollouts=K, gen_length=gen_length,
        reward_fn=noise_level_reward,
        adv_std_floor=1e-3,
        seed=42,
    )

    # With rewards std ≈ 1e-5 and floor = 1e-3, advantage magnitudes must
    # be bounded above by ~(max_reward - mean_reward) / 1e-3 ≈ 1e-2. The
    # unclamped version would produce ~1e3.
    assert log.advantage_max_abs < 1.0, (
        f"advantage_max_abs={log.advantage_max_abs} — std_floor not clamping")
    assert torch.isfinite(torch.tensor(log.loss))


@requires_llama
def test_last_log_pi_sum_cleared_after_phase1_forward():
    """Phase-1 forward calls must not leave a stale log_pi_sum tensor on
    memory — readers that missed the phase-2 capture window should get
    None, not zeros that look like real graph-connected data."""
    from src.model.config import Config as MemoryConfig
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory

    torch.manual_seed(0)
    mem_cfg = MemoryConfig.tier_a(D=2048, tbptt_block=64)
    cfg = PretrainedConfig.llama_1b(memory=mem_cfg)
    wrapper = PretrainedLMWithMemory(cfg)
    wrapper.reset_memory(bs=1)

    T = 2 * cfg.memory.modulation_interval
    ids = torch.randint(0, cfg.vocab_size_lm, (1, T))

    # Phase-1 train call — current_phase defaults to "phase1".
    wrapper.train()
    _ = wrapper(ids)
    assert wrapper.memory._last_log_pi_sum is None, (
        "phase-1 call should not leave a meaningful log_pi_sum")

    # Eval call — no sampling, also no log_pi.
    wrapper.train(False)
    _ = wrapper(ids)
    assert wrapper.memory._last_log_pi_sum is None

    # Phase-2 train call — sampling happens, log_pi_sum should be present
    # and graph-connected.
    wrapper.train()
    wrapper.current_phase = "phase2"
    _ = wrapper(ids)
    lp = wrapper.memory._last_log_pi_sum
    assert lp is not None, "phase-2 train call should set log_pi_sum"
    assert lp.requires_grad, "log_pi_sum must be graph-connected for REINFORCE"


@requires_llama
def test_phase1_autoregressive_unroll_backprop_reaches_prefix_fires():
    """Autoregressive Gumbel phase-1: gradient from a continuation-token CE
    must reach the modulator weights that fired during the prefix pass —
    otherwise the unroll is only training memory's fast state and the whole
    point of the rewrite is defeated."""
    from src.model.config import Config as MemoryConfig
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory
    from src.pretrained.train_phase1_ar import Phase1ARBatch, run_phase1_ar

    torch.manual_seed(0)
    # Need tbptt_block >= 2*mod_interval AND >= T_pre so the whole prefix
    # stays in one graph. Pick tbptt_block=64, T_pre=32 (two mod fires).
    mem_cfg = MemoryConfig.tier_a(D=2048, tbptt_block=64)
    cfg = PretrainedConfig.llama_1b(memory=mem_cfg)
    wrapper = PretrainedLMWithMemory(cfg)
    T_pre = 2 * cfg.memory.modulation_interval    # 32: two fires
    T_cont = 4

    # Snapshot the modulator's tok_proj and cell_emb before the step. These
    # are set during the PREFIX fires, not at continuation time (continuation
    # steps run with T=1 and the modulator never fires). If the unroll
    # doesn't backprop through the prefix, these stay unchanged.
    tok_proj_before = wrapper.memory.modulator.tok_proj[0].weight.detach().clone()
    cell_emb_before = wrapper.memory.modulator.cell_emb.detach().clone()

    opt = torch.optim.AdamW(
        [p for _, p in wrapper.trainable_parameters()], lr=1e-3)

    BS = 1
    prefix = torch.randint(0, cfg.vocab_size_lm, (BS, T_pre))
    cont = torch.randint(0, cfg.vocab_size_lm, (BS, T_cont))

    def data_iter():
        while True:
            yield Phase1ARBatch(prefix_ids=prefix, continuation_ids=cont)

    logs = []
    run_phase1_ar(wrapper, opt, data_iter(),
                  steps=1, gumbel_tau_start=1.0, gumbel_tau_end=0.3,
                  anneal_across_steps=1, on_step=logs.append)
    assert len(logs) == 1
    assert torch.isfinite(torch.tensor(logs[0].loss))

    # Both prefix-fire targets must have moved — that proves the gradient
    # from the continuation CE crossed the prefix boundary into the
    # modulator.
    assert not torch.equal(
        wrapper.memory.modulator.tok_proj[0].weight, tok_proj_before), (
        "tok_proj did not update — AR unroll grad did not reach prefix fires")
    assert not torch.equal(
        wrapper.memory.modulator.cell_emb, cell_emb_before), (
        "cell_emb did not update — AR unroll grad did not reach prefix fires")


def test_freeze_codebook_decoder_leaves_modulator_trainable():
    """freeze_codebook_decoder() should pin codebook + DirectDecoder but
    keep the modulator + W_in/W_out/scale + dynamics trainable."""
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory

    if not _hf_cache_has_1b():
        pytest.skip("Llama-3.2-1B not present in HF cache")

    cfg = PretrainedConfig.llama_1b()
    wrapper = PretrainedLMWithMemory(cfg)
    wrapper.freeze_codebook_decoder()

    assert not wrapper.memory.discrete_policy.codebook.requires_grad
    for p in wrapper.memory.decoder.parameters():
        assert not p.requires_grad
    # Modulator + W_in/W_out/scale still trainable.
    assert wrapper.memory.modulator.logit_head.weight.requires_grad
    assert wrapper.memory.modulator.tok_proj[0].weight.requires_grad
    assert wrapper.mem_inject.W_in.weight.requires_grad
    assert wrapper.mem_inject.W_out.weight.requires_grad
    assert wrapper.mem_inject.scale.requires_grad
    # Memory dynamics (inject_w, msg MLPs, gamma logits) still trainable.
    assert wrapper.memory.inject_w.requires_grad
    assert wrapper.memory.msg_w1.requires_grad


def test_freeze_all_but_logit_head_leaves_only_logit_head():
    """freeze_all_but_logit_head() should pin everything trainable by default
    except modulator.logit_head — the phase-2 GRPO surface from old main."""
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory

    if not _hf_cache_has_1b():
        pytest.skip("Llama-3.2-1B not present in HF cache")

    cfg = PretrainedConfig.llama_1b()
    wrapper = PretrainedLMWithMemory(cfg)
    wrapper.freeze_all_but_logit_head()

    trainable = {n for n, p in wrapper.trainable_parameters()}
    assert trainable == {
        "memory.modulator.logit_head.weight",
        "memory.modulator.logit_head.bias",
    }, f"unexpected trainable set: {trainable}"


def test_unfreeze_all_restores_full_training_surface():
    """unfreeze_all() should restore the full training surface —
    memory + projections + scale all trainable, Llama backbone frozen."""
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory

    if not _hf_cache_has_1b():
        pytest.skip("Llama-3.2-1B not present in HF cache")

    cfg = PretrainedConfig.llama_1b()
    wrapper = PretrainedLMWithMemory(cfg)
    wrapper.freeze_all_but_logit_head()
    wrapper.unfreeze_all()

    # All memory params trainable.
    for p in wrapper.memory.parameters():
        assert p.requires_grad
    # Inject params trainable.
    assert wrapper.mem_inject.W_in.weight.requires_grad
    assert wrapper.mem_inject.W_out.weight.requires_grad
    assert wrapper.mem_inject.scale.requires_grad
    # Backbone still frozen.
    assert not wrapper.llama.model.embed_tokens.weight.requires_grad
    assert not wrapper.llama.lm_head.weight.requires_grad


@requires_llama
def test_cycle_loop_runs_end_to_end():
    """Minimal cycle: bootstrap (2 steps) → 1 cycle with phase-1 AR (2 steps)
    → phase-2 GRPO (2 steps). Confirms all three sub-loops run under a single
    orchestrator without the freeze transitions breaking the optimizer or
    memory reset plumbing."""
    import tempfile
    from src.model.config import Config as MemoryConfig
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory
    from src.pretrained.train_loop import CycleConfig, run_cycle_loop
    from src.pretrained.train_phase1 import Phase1Batch
    from src.pretrained.train_phase1_ar import Phase1ARBatch

    torch.manual_seed(0)
    mem_cfg = MemoryConfig.tier_a(D=2048, tbptt_block=64)
    cfg = PretrainedConfig.llama_1b(memory=mem_cfg)
    wrapper = PretrainedLMWithMemory(cfg)

    T_boot = 2 * cfg.memory.modulation_interval
    T_pre = 2 * cfg.memory.modulation_interval
    T_cont = 4
    BS = 1

    def bootstrap_iter():
        while True:
            ids = torch.randint(0, cfg.vocab_size_lm, (BS, T_boot))
            tgt = torch.cat([ids[:, 1:], ids[:, :1]], dim=1)
            yield Phase1Batch(input_ids=ids, target_ids=tgt, prev_token=None)

    def cycle_p1_iter():
        while True:
            pre = torch.randint(0, cfg.vocab_size_lm, (BS, T_pre))
            cont = torch.randint(0, cfg.vocab_size_lm, (BS, T_cont))
            yield Phase1ARBatch(prefix_ids=pre, continuation_ids=cont)

    def cycle_p2_iter():
        while True:
            pre = torch.randint(0, cfg.vocab_size_lm, (1, T_pre))
            ref = torch.randint(0, cfg.vocab_size_lm, (T_cont,))
            yield (pre, ref)

    def stub_reward(generated, reference):
        # Variance-bearing reward so advantage normalization isn't zero.
        return generated[:, 0].float() / 10000.0

    with tempfile.TemporaryDirectory() as tmpdir:
        loop_cfg = CycleConfig(
            work_dir=tmpdir,
            bootstrap_steps=2, cycles=1,
            cycle_phase1_steps=2, cycle_phase2_steps=2,
            bs=BS, T_pre=T_pre, T_cont=T_cont,
            grpo_K=4, grpo_rollout_len=T_cont,
        )
        logs = []
        run_cycle_loop(
            wrapper,
            bootstrap_iter(), cycle_p1_iter(), cycle_p2_iter(),
            stub_reward, loop_cfg,
            log=logs.append,
        )

    # Orchestrator printed CYCLE LOOP COMPLETE; memory/optimizer survived.
    assert any("CYCLE LOOP COMPLETE" in str(lg) for lg in logs)
    # Backbone still frozen after all the freeze/unfreeze cycling.
    assert not wrapper.llama.model.embed_tokens.weight.requires_grad
    assert not wrapper.llama.lm_head.weight.requires_grad
    # After the cycle completes, the full training surface is restored
    # (unfreeze_all was the last freeze call before phase-2, so strictly
    # speaking we expect the phase-2 freeze to still be in effect — but
    # memory state must at least be initialized + finite).
    assert wrapper.memory.is_initialized
    assert torch.isfinite(wrapper.memory.W).all()


# ======================================================================
# Gradient flow smoke tests
#
# Every trainable module should receive non-zero gradient from its
# corresponding training loss. A silent zero-gradient on any module is
# a training bug (the module can't learn) — these smokes assert the
# gradient flow contract explicitly so we catch regressions like the
# TBPTT-detach / log_pi_sum bugs before they reach a training run.
# ======================================================================


def _trainable_param_groups(wrapper) -> dict:
    """Enumerate every trainable parameter path in the wrapper, grouped
    by what they test. Any dead group is a potential training bug."""
    L = wrapper.config.inject_layer
    groups: dict[str, list[tuple[str, torch.Tensor]]] = {
        "inject_projections": [
            (f"mem_inject.W_in", wrapper.mem_inject.W_in.weight),
            (f"mem_inject.W_out", wrapper.mem_inject.W_out.weight),
            (f"mem_inject.scale", wrapper.mem_inject.scale),
        ],
        "modulator_encoder": [
            ("tok_proj[0]", wrapper.memory.modulator.tok_proj[0].weight),
            ("tok_proj[-1]", wrapper.memory.modulator.tok_proj[-1].weight),
            ("h_proj", wrapper.memory.modulator.h_proj.weight),
            ("msg_emit_proj", wrapper.memory.modulator.msg_emit_proj.weight),
            ("msg_recv_proj", wrapper.memory.modulator.msg_recv_proj.weight),
            ("role_emb", wrapper.memory.modulator.role_emb.weight),
            ("cell_emb", wrapper.memory.modulator.cell_emb),
            ("edge_bias_mlp[0]", wrapper.memory.modulator.edge_bias_mlp[0].weight),
        ],
        "modulator_attention": [
            (f"layers[{i}].qkv", wrapper.memory.modulator.layers[i].qkv.weight)
            for i in range(len(wrapper.memory.modulator.layers))
        ] + [
            (f"layers[{i}].out_proj", wrapper.memory.modulator.layers[i].out_proj.weight)
            for i in range(len(wrapper.memory.modulator.layers))
        ],
        "modulator_output": [
            ("pool_norm", wrapper.memory.modulator.pool_norm.weight),
            ("logit_head", wrapper.memory.modulator.logit_head.weight),
        ],
        "codebook": [("codebook", wrapper.memory.discrete_policy.codebook)],
        "decoder": [
            ("decoder.mlp[0]", wrapper.memory.decoder.mlp[0].weight),
            ("decoder.mlp[-1]", wrapper.memory.decoder.mlp[-1].weight),
        ],
        "memory_dynamics": [
            ("inject_w", wrapper.memory.inject_w),
            ("msg_w1", wrapper.memory.msg_w1),
            ("msg_w2", wrapper.memory.msg_w2),
            ("W_decay_logit", wrapper.memory.W_decay_logit),
            ("decay_gamma_logit", wrapper.memory.decay_gamma_logit),
            ("hebbian_decay_logit", wrapper.memory.hebbian_decay_logit),
            ("neuron_id", wrapper.memory.neuron_id),
        ],
    }
    return groups


def _assert_all_have_nonzero_grad(groups, required_groups: list[str],
                                   context: str):
    missing = []
    zero = []
    for g_name in required_groups:
        for p_name, p in groups[g_name]:
            if p.grad is None:
                missing.append(f"{g_name}/{p_name}")
            elif p.grad.abs().sum().item() == 0.0:
                zero.append(f"{g_name}/{p_name}")
    assert not missing, (
        f"[{context}] params with .grad=None (graph not connected): {missing}")
    assert not zero, (
        f"[{context}] params with all-zero grad (graph connected but "
        f"gradient vanished): {zero}")


def _warm_up_decoder(wrapper, opt, ids, tgt):
    """One CE step to move `decoder.mlp[-1].weight` off its zero-init.
    Without this warm-up, `mlp[-1] = 0` blocks the chain-rule gradient
    through the decoder — grad can't reach decoder.mlp[0], codebook, or
    the modulator body even though the graph IS connected. This is by
    design for the memory no-op start (test_decoder_starts_at_no_op
    enforces it); the gradient-flow smokes just need to see post-init
    behavior."""
    wrapper.train()
    out = wrapper(ids)
    ce = torch.nn.functional.cross_entropy(
        out.logits.reshape(-1, out.logits.shape[-1]), tgt.reshape(-1))
    loss = ce + 0.1 * wrapper._last_mem_pred_loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    wrapper.detach_memory()


@requires_llama
def test_gradient_flow_phase1_parallel_covers_all_trainables():
    """Parallel phase-1 CE + mem_pred_loss must reach every trainable
    module — inject projections, modulator (encoder/attention/output),
    codebook, decoder, and memory dynamics. Runs one warm-up step first
    so decoder.mlp[-1]'s zero-init doesn't block the chain-rule
    gradient to mlp[0] / codebook / modulator body."""
    from src.model.config import Config as MemoryConfig
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory

    torch.manual_seed(0)
    mem_cfg = MemoryConfig.tier_a(D=2048, tbptt_block=64)
    cfg = PretrainedConfig.llama_1b(memory=mem_cfg, llama_dtype="fp32")
    wrapper = PretrainedLMWithMemory(cfg)
    wrapper.train()
    wrapper.reset_memory(bs=1)

    T = 2 * cfg.memory.modulation_interval
    ids = torch.randint(0, cfg.vocab_size_lm, (1, T))
    tgt = torch.cat([ids[:, 1:], ids[:, :1]], dim=1)

    # Warm up so decoder.mlp[-1] is non-zero, enabling grad flow through
    # the decoder to everything upstream (codebook, modulator).
    opt = torch.optim.AdamW(
        [p for _, p in wrapper.trainable_parameters()], lr=1e-3)
    _warm_up_decoder(wrapper, opt, ids, tgt)

    # Fresh forward + backward. Now check gradient coverage.
    wrapper.reset_memory(bs=1)
    out = wrapper(ids)
    ce = torch.nn.functional.cross_entropy(
        out.logits.reshape(-1, out.logits.shape[-1]),
        tgt.reshape(-1))
    loss = ce + 0.1 * wrapper._last_mem_pred_loss
    opt.zero_grad(set_to_none=True)
    loss.backward()

    groups = _trainable_param_groups(wrapper)
    _assert_all_have_nonzero_grad(
        groups,
        ["inject_projections", "modulator_encoder", "modulator_attention",
         "modulator_output", "codebook", "decoder", "memory_dynamics"],
        "phase1_parallel")

    # Frozen backbone must stay at grad=None.
    assert wrapper.llama.model.embed_tokens.weight.grad is None
    assert wrapper.llama.lm_head.weight.grad is None
    assert wrapper.llama.model.norm.weight.grad is None
    assert wrapper.llama.model.layers[0].self_attn.q_proj.weight.grad is None


@requires_llama
def test_gradient_flow_phase1_ar_covers_all_trainables():
    """Autoregressive Gumbel unroll: continuation-CE gradient must reach
    the SAME full set of trainables through the preserve_memory_graph
    unroll chain (prefix fires → carried state → per-token unroll).
    Warms up one parallel step first so decoder.mlp[-1] is non-zero."""
    from src.model.config import Config as MemoryConfig
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory

    torch.manual_seed(0)
    mem_cfg = MemoryConfig.tier_a(D=2048, tbptt_block=64)
    cfg = PretrainedConfig.llama_1b(memory=mem_cfg, llama_dtype="fp32")
    wrapper = PretrainedLMWithMemory(cfg)
    wrapper.train()
    wrapper.reset_memory(bs=1)

    T_pre = 2 * cfg.memory.modulation_interval
    T_cont = 4
    prefix = torch.randint(0, cfg.vocab_size_lm, (1, T_pre))
    cont = torch.randint(0, cfg.vocab_size_lm, (1, T_cont))

    # Warm up decoder so mlp[-1] != 0 (otherwise chain-rule blocks
    # gradient through the decoder).
    opt = torch.optim.AdamW(
        [p for _, p in wrapper.trainable_parameters()], lr=1e-3)
    warm_ids = prefix
    warm_tgt = torch.cat([warm_ids[:, 1:], warm_ids[:, :1]], dim=1)
    _warm_up_decoder(wrapper, opt, warm_ids, warm_tgt)

    wrapper.reset_memory(bs=1)
    with wrapper.preserve_memory_graph():
        out = wrapper(prefix, use_cache=True)
        past = out.past_key_values
        last = out.logits[:, -1]
        ce_terms = [torch.nn.functional.cross_entropy(last.float(), cont[:, 0])]
        for i in range(T_cont - 1):
            out_i = wrapper(cont[:, i:i + 1], past_key_values=past, use_cache=True)
            past = out_i.past_key_values
            ce_terms.append(torch.nn.functional.cross_entropy(
                out_i.logits[:, -1].float(), cont[:, i + 1]))
    loss = torch.stack(ce_terms).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()

    groups = _trainable_param_groups(wrapper)
    _assert_all_have_nonzero_grad(
        groups,
        ["inject_projections", "modulator_encoder", "modulator_attention",
         "modulator_output", "codebook", "decoder", "memory_dynamics"],
        "phase1_ar")


@requires_llama
def test_gradient_flow_phase2_grpo_reaches_modulator_and_recurrent():
    """Phase-2 GRPO REINFORCE loss = -(log_pi_sum * advantage).mean().
    The gradient path is: loss → log_pi_sum → sampled-code log-probs →
    modulator logits → modulator weights. Cells communicate across
    fires through W updates, so codebook / decoder / γ logits / msg /
    inject_w also receive RECURRENT gradient (fire k's W-write affects
    fire k+1's modulator inputs).

    What should NOT receive gradient: W_out and scale (they only affect
    LM logits, which are NOT in the phase-2 loss). This is the structural
    signature of REINFORCE-through-memory."""
    from src.model.config import Config as MemoryConfig
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory
    from src.pretrained.train_phase2 import grpo_step

    torch.manual_seed(0)
    # T_pre must be >= 2*mod_interval to have at least two fires (one
    # that writes, one that reads the previous write via recurrent W).
    mem_cfg = MemoryConfig.tier_a(D=2048, tbptt_block=64)
    cfg = PretrainedConfig.llama_1b(memory=mem_cfg, llama_dtype="fp32")
    wrapper = PretrainedLMWithMemory(cfg)

    T_prefix = 3 * cfg.memory.modulation_interval    # at least 3 fires
    K = 4
    gen_length = 4
    prefix = torch.randint(0, cfg.vocab_size_lm, (1, T_prefix))
    ref = torch.randint(0, cfg.vocab_size_lm, (gen_length,))

    opt = torch.optim.AdamW(
        [p for _, p in wrapper.trainable_parameters()], lr=1e-4)

    # Variance-bearing reward to guarantee non-zero advantages.
    def stub_reward(gen, ref):
        return gen[:, 0].float() / 10000.0

    _ = grpo_step(
        wrapper, opt, prefix_ids=prefix, reference_cont=ref,
        num_rollouts=K, gen_length=gen_length,
        reward_fn=stub_reward, seed=42,
        collect_heavy_telemetry=False,
    )

    # grpo_step calls opt.step() which zeros grads via set_to_none=True
    # AFTER the step. Re-run just the backward part so grads are visible.
    # (The groups test needs to see .grad populated; opt.step() cleared
    # them.)
    wrapper.current_phase = "phase2"
    wrapper.train(True)
    wrapper.reset_memory(bs=K)
    prefix_rep = prefix.expand(K, -1).contiguous()
    out = wrapper(prefix_rep, use_cache=True)
    log_pi_sum = wrapper.memory._last_log_pi_sum
    loss = -(log_pi_sum * torch.ones_like(log_pi_sum)).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()

    groups = _trainable_param_groups(wrapper)

    # Direct gradient from log_pi_sum: modulator (encoder + attention +
    # output). Codebook + decoder + memory_dynamics receive gradient via
    # the recurrent chain (prior fires' W/decay writes affect later
    # fires' modulator inputs).
    _assert_all_have_nonzero_grad(
        groups,
        ["modulator_encoder", "modulator_attention", "modulator_output",
         "codebook", "decoder", "memory_dynamics"],
        "phase2_grpo_recurrent")

    # W_in receives recurrent gradient too (H_mid → h_mem → h → modulator).
    assert wrapper.mem_inject.W_in.weight.grad is not None, (
        "W_in should receive recurrent gradient in phase 2")
    assert wrapper.mem_inject.W_in.weight.grad.abs().sum() > 0

    # W_out and scale should NOT have gradient — they only feed LM logits,
    # and the phase-2 loss doesn't depend on LM logits.
    W_out_g = wrapper.mem_inject.W_out.weight.grad
    scale_g = wrapper.mem_inject.scale.grad
    W_out_zero = (W_out_g is None) or (W_out_g.abs().sum().item() == 0.0)
    scale_zero = (scale_g is None) or (scale_g.abs().sum().item() == 0.0)
    assert W_out_zero, (
        "W_out got gradient in phase 2 but shouldn't — phase-2 loss must "
        "not depend on LM logits.")
    assert scale_zero, (
        "scale got gradient in phase 2 but shouldn't.")
