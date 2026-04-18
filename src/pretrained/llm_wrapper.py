"""PretrainedLMWithMemory — loads Llama-3.2, freezes backbone, wraps one layer.

Public API:
    wrapper = PretrainedLMWithMemory(config)
    out = wrapper(input_ids)                # returns (logits, mem_pred_loss)
    wrapper.reset_memory(bs)                # zero memory state for batch
    wrapper.detach_memory()                  # TBPTT boundary

Memory is wired during forward via a closure that captures input_ids; the
MemInjectLayer calls memory_fn(h_mem_in) and receives the per-token
readouts back in d_mem space. Memory's auxiliary NTP loss is collected
from self._last_mem_pred_loss after each forward call.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from src.model.memory import MemoryGraph
from src.pretrained.config import PretrainedConfig
from src.pretrained.llama_mem_adapter import LlamaMemAdapter
from src.pretrained.mem_inject_layer import MemInjectLayer


class PretrainedLMWithMemory(nn.Module):
    def __init__(self, config: PretrainedConfig, attach_memory: bool = True):
        super().__init__()
        self.config = config

        # 1. Inspect HF config to populate derived fields.
        hf_cfg = AutoConfig.from_pretrained(config.model_name)
        config.d_lm = hf_cfg.hidden_size
        config.n_lm_layers = hf_cfg.num_hidden_layers
        config.vocab_size_lm = hf_cfg.vocab_size
        config.validate_after_load()
        self._rms_eps = getattr(hf_cfg, "rms_norm_eps", 1e-5)

        # 2. Load the LM in its native dtype. Training casts via autocast.
        self.llama = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.float32)

        # 3. Freeze backbone. Trainable pieces (W_in/W_out/scale + memory) are
        # added after this.
        if config.freeze_backbone:
            for p in self.llama.parameters():
                p.requires_grad = False

        # 4. Swap the chosen layer with a MemInjectLayer. memory_fn=None so the
        # layer is a transparent pass-through until memory is wired per-forward.
        L = config.inject_layer
        orig_layer: LlamaDecoderLayer = self.llama.model.layers[L]
        self.llama.model.layers[L] = MemInjectLayer(
            orig_layer=orig_layer,
            d_lm=config.d_lm,
            d_mem=config.d_mem,
            scale_init=config.scale_init,
            memory_fn=None,
        )

        # 5. Memory graph: config.memory has D set to d_mem. vocab_size on the
        # memory config doesn't need to match the LM vocab — the aux-loss head
        # goes through the adapter, which uses Llama's lm_head directly.
        config.memory.vocab_size = config.vocab_size_lm
        config.memory.validate()
        self.memory = MemoryGraph(config.memory) if attach_memory else None
        self._adapter = (LlamaMemAdapter(self.llama, self.mem_inject.W_out,
                                         rms_eps=self._rms_eps)
                         if attach_memory else None)

        # Per-call scratch: auxiliary memory CE loss produced by the last
        # forward. None when memory is not wired.
        self._last_mem_pred_loss: torch.Tensor | None = None
        # "phase1" = Gumbel-softmax backprop (bootstrap). "phase2" = hard
        # Categorical with log_pi tracking for REINFORCE/GRPO rollouts.
        # Flip via `wrapper.current_phase = "phase2"` before phase-2 runs.
        self.current_phase: str = "phase1"

    @property
    def mem_inject(self) -> MemInjectLayer:
        """Reference to the inserted MemInjectLayer via the llama path."""
        return self.llama.model.layers[self.config.inject_layer]

    def reset_memory(self, bs: int):
        if self.memory is not None:
            device = next(self.llama.parameters()).device
            self.memory.initialize_states(bs, device)

    def detach_memory(self):
        if self.memory is not None:
            self.memory.detach_states()

    def forward(self, input_ids: torch.Tensor, **kwargs):
        """Run a full LM forward with memory read/write at the injection layer.

        Returns the HuggingFace ModelOutput (has `.logits`). The auxiliary
        memory-CE loss is stashed on `self._last_mem_pred_loss` (None if
        memory not attached).
        """
        if self.memory is None:
            return self.llama(input_ids=input_ids, **kwargs)

        # Closure captures input_ids + adapter for the per-segment memory call.
        # set_memory_fn / unset dance keeps the wrapper safe for calls without
        # memory (e.g., direct `self.llama(input_ids)` from helpers).
        prev_token = kwargs.pop("prev_token", None)
        use_rms = True

        def memory_fn(h_mem: torch.Tensor) -> torch.Tensor:
            readouts, mem_loss = self.memory.forward_segment(
                h_mem, input_ids, self._adapter,
                prev_token=prev_token,
                use_rmsnorm=use_rms,
                rms_eps=self._rms_eps,
                phase=self.current_phase)
            self._last_mem_pred_loss = mem_loss
            return readouts

        self.mem_inject.set_memory_fn(memory_fn)
        try:
            out = self.llama(input_ids=input_ids, **kwargs)
        finally:
            self.mem_inject.set_memory_fn(None)
        return out

    def trainable_parameters(self):
        """Yield only params that require grad — memory + projections + scale."""
        for name, p in self.named_parameters():
            if p.requires_grad:
                yield name, p
