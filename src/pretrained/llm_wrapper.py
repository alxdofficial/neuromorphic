"""PretrainedLMWithMemory — loads Llama-3.2, freezes backbone, wraps one layer.

Public API:
    wrapper = PretrainedLMWithMemory(config)
    out = wrapper(input_ids)                # returns HF ModelOutput with `.logits`
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

        # 2. Load the LM in the configured dtype. Default bf16 matches the
        # production GPU run path and skips the fp32→bf16 weight cast that
        # autocast otherwise triggers on every matmul. Memory state dtype
        # follows the LM dtype (see reset_memory) so both sides agree.
        _dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16,
                       "fp16": torch.float16}
        llama_dtype = _dtype_map[config.llama_dtype]
        self.llama = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=llama_dtype)

        # 3. Freeze backbone. Trainable pieces (W_in/W_out/scale + memory) are
        # added after this.
        if config.freeze_backbone:
            for p in self.llama.parameters():
                p.requires_grad = False

        # 4. Swap the chosen layer with a MemInjectLayer. memory_fn=None so
        # the layer is a transparent pass-through until memory is wired
        # per-forward. W_in / W_out / scale stay at their fp32 nn.Parameter
        # default — bf16 param updates round small Adam steps to zero,
        # which silently stalls training on small smoke runs. The forward
        # handles the cross-dtype add explicitly.
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
        # Memory PARAMS stay in fp32. Bf16 optimizer updates round small
        # gradients to zero (param = 2.0, update = 1e-5 → 2.0 in bf16).
        # Memory STATE tensors (h, W, hebbian, ...) follow Llama's dtype
        # via reset_memory so the forward-path matmuls don't mix dtypes.
        # Buffers (out_port_mask, role_id) also stay fp32 / int64; the
        # fused step casts out_port_mask to state dtype per call.
        self._adapter = (LlamaMemAdapter(self.llama, self.mem_inject.W_out,
                                         rms_eps=self._rms_eps)
                         if attach_memory else None)
        # Without memory, pin scale to zero so the inject residual stays zero
        # and MemInjectLayer can pass through transparently. The loud assert
        # in MemInjectLayer would otherwise fire on every forward.
        if not attach_memory:
            with torch.no_grad():
                self.mem_inject.scale.zero_()

        # Per-call scratch: auxiliary memory CE loss produced by the last
        # forward. None when memory is not wired.
        self._last_mem_pred_loss: torch.Tensor | None = None
        # "phase1" = Gumbel-softmax backprop (bootstrap). "phase2" = hard
        # Categorical with log_pi tracking for REINFORCE/GRPO rollouts.
        # Flip via `wrapper.current_phase = "phase2"` before phase-2 runs.
        self.current_phase: str = "phase1"
        # When True, memory.forward_segment keeps state graph-connected
        # across calls — no detach at end, no intra-call tbptt detach.
        # Used by autoregressive phase-1 unroll so gradient from each
        # continuation token reaches the prefix's modulator fires.
        # Caller is responsible for calling `detach_memory()` after
        # backward completes.
        self._preserve_memory_graph: bool = False

    @property
    def mem_inject(self) -> MemInjectLayer:
        """Reference to the inserted MemInjectLayer via the llama path."""
        return self.llama.model.layers[self.config.inject_layer]

    def reset_memory(self, bs: int):
        if self.memory is not None:
            p = next(self.llama.parameters())
            # Match memory state dtype to Llama's dtype so h_mem crossings
            # don't fight mixed dtypes under no-autocast code paths (CPU
            # tests and eager inference). Under bf16 autocast on CUDA the
            # training loop would handle casts anyway, but forcing dtype
            # parity at init is one less source of surprise dtype bugs.
            self.memory.initialize_states(bs, p.device, dtype=p.dtype)

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

        # Aux loss only matters during training. In eval / rollout we skip
        # the 128K-vocab mem_head matmul entirely — it's the biggest single
        # cost inside forward_segment at T=1 autoregressive gen.
        compute_aux = self.training

        def memory_fn(h_mem: torch.Tensor) -> torch.Tensor:
            readouts, mem_loss = self.memory.forward_segment(
                h_mem, input_ids, self._adapter,
                prev_token=prev_token,
                use_rmsnorm=use_rms,
                rms_eps=self._rms_eps,
                phase=self.current_phase,
                preserve_graph=self._preserve_memory_graph,
                compute_aux_loss=compute_aux)
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

    # ------------------------------------------------------------------
    # Freeze controls — mirror the old main's cycle semantics. Bootstrap
    # trains everything; cycle phase-1 freezes codebook + decoder; cycle
    # phase-2 GRPO freezes everything except the modulator's logit head.
    # ------------------------------------------------------------------

    def freeze_codebook_decoder(self):
        """Freeze codebook + DirectDecoder. The rest of memory + W_in/W_out/
        scale keep training. Used for cycle phase-1."""
        if self.memory is None:
            return
        for p in self.memory.discrete_policy.parameters():
            p.requires_grad = False
        for p in self.memory.decoder.parameters():
            p.requires_grad = False

    def freeze_all_but_logit_head(self):
        """Freeze everything except the modulator's logit_head. Matches the
        old main's `phase 2 = GRPO on logit head only` semantics — most
        restrictive update surface for stable policy-gradient training."""
        if self.memory is None:
            return
        for _, p in self.trainable_parameters():
            p.requires_grad = False
        for p in self.memory.modulator.logit_head.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        """Re-enable gradient on everything that's trainable by default.
        Use to restore the bootstrap training surface."""
        if self.config.freeze_backbone:
            for p in self.llama.parameters():
                p.requires_grad = False
        else:
            for p in self.llama.parameters():
                p.requires_grad = True
        if self.memory is not None:
            for p in self.memory.parameters():
                p.requires_grad = True
        self.mem_inject.W_in.weight.requires_grad = True
        self.mem_inject.W_out.weight.requires_grad = True
        self.mem_inject.scale.requires_grad = True

    def preserve_memory_graph(self):
        """Context manager: within the block, memory.forward_segment keeps
        state graph-connected across calls. Needed for autoregressive
        phase-1 unroll where continuation-token gradient must reach the
        prefix-pass modulator fires."""
        return _PreserveMemoryGraph(self)


class _PreserveMemoryGraph:
    def __init__(self, wrapper: "PretrainedLMWithMemory"):
        self.wrapper = wrapper
        self._prior = wrapper._preserve_memory_graph

    def __enter__(self):
        self.wrapper._preserve_memory_graph = True
        return self.wrapper

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.wrapper._preserve_memory_graph = self._prior
        return False
