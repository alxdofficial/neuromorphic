"""PretrainedLMWithMemory — loads Llama-3.2, freezes backbone, wraps one layer.

Public API:
    wrapper = PretrainedLMWithMemory(config)
    logits = wrapper(input_ids)              # standard LM forward
    wrapper.reset_memory(bs)                 # zero memory state for batch
    wrapper.detach_memory()                  # TBPTT boundary

The memory graph is NOT wired in this initial cut — `memory_fn` stays None
on the MemInjectLayer so forward reproduces vanilla Llama bit-for-bit.
Memory wiring comes in the next commit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from src.pretrained.config import PretrainedConfig
from src.pretrained.mem_inject_layer import MemInjectLayer


class PretrainedLMWithMemory(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config

        # 1. Inspect HF config to populate derived fields.
        hf_cfg = AutoConfig.from_pretrained(config.model_name)
        config.d_lm = hf_cfg.hidden_size
        config.n_lm_layers = hf_cfg.num_hidden_layers
        config.vocab_size_lm = hf_cfg.vocab_size
        config.validate_after_load()

        # 2. Load the LM. Keep it in its native dtype so the eq-to-vanilla
        # smoke test is bit-exact. Training will cast via autocast.
        self.llama = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.float32)

        # 3. Freeze the backbone. W_in / W_out / scale stay trainable
        # (they live on the MemInjectLayer and are added below).
        if config.freeze_backbone:
            for p in self.llama.parameters():
                p.requires_grad = False

        # 4. Swap the chosen layer with a MemInjectLayer wrapping the original.
        L = config.inject_layer
        orig_layer: LlamaDecoderLayer = self.llama.model.layers[L]
        self.llama.model.layers[L] = MemInjectLayer(
            orig_layer=orig_layer,
            d_lm=config.d_lm,
            d_mem=config.d_mem,
            scale_init=config.scale_init,
            memory_fn=None,  # wired later
        )

    @property
    def mem_inject(self) -> MemInjectLayer:
        """Reference to the inserted MemInjectLayer, accessed via the llama path.

        Stored only under `llama.model.layers[L]` to keep a single nn.Module
        parent path — otherwise param names would double-register under two
        prefixes and trip `named_parameters` consumers.
        """
        return self.llama.model.layers[self.config.inject_layer]

    def set_memory_fn(self, memory_fn):
        self.mem_inject.set_memory_fn(memory_fn)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        """Standard LM forward. Returns HF output (has `.logits`)."""
        return self.llama(input_ids=input_ids, **kwargs)

    def trainable_parameters(self):
        """Yield only the params that require grad — memory + projections + scale.

        Use this for the optimizer so the frozen backbone stays cheap (no
        momentum / second-moment state on 1B frozen weights).
        """
        for name, p in self.named_parameters():
            if p.requires_grad:
                yield name, p
