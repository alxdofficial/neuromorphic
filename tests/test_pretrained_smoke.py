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
    """Constructor loads the model and freezes everything except W_in/W_out/scale."""
    from src.pretrained.config import PretrainedConfig
    from src.pretrained.llm_wrapper import PretrainedLMWithMemory

    cfg = PretrainedConfig.llama_1b()
    model = PretrainedLMWithMemory(cfg)

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    # Only MemInjectLayer's W_in, W_out, scale should be trainable. They sit
    # at the injected-layer path (e.g., llama.model.layers.8.*).
    L = cfg.inject_layer
    expected = {
        f"llama.model.layers.{L}.W_in.weight",
        f"llama.model.layers.{L}.W_out.weight",
        f"llama.model.layers.{L}.scale",
    }
    assert trainable == expected, (
        f"trainable mismatch: got {trainable}, expected {expected}")


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
    wrapper = PretrainedLMWithMemory(cfg)
    wrapper.train(False)

    vanilla = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.float32)
    vanilla.train(False)

    # Scale stays at init (2.0). memory_fn is None. Expect bypass.
    input_ids = torch.randint(0, 128256, (1, 8))
    with torch.no_grad():
        out_wrapper = wrapper(input_ids).logits
        out_vanilla = vanilla(input_ids=input_ids).logits

    assert torch.equal(out_wrapper, out_vanilla)
