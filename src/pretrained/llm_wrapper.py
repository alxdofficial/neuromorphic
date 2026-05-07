"""PretrainedLMWithMemory — loads a HF causal LM, freezes backbone, wraps one layer.

Public API:
    wrapper = PretrainedLMWithMemory(config)
    out = wrapper(input_ids)                # returns HF ModelOutput with `.logits`
    wrapper.begin_segment(bs)                # zero memory state for batch
    wrapper.detach_memory()                  # TBPTT boundary

The wrapper is host-agnostic via `src.pretrained.hosts.HostAdapter`. For
Llama / TinyLlama / SmolLM2 / Mistral / Qwen2 the single `LlamaHost`
adapter covers all attribute paths; other families (GPT-NeoX, GPT-2) get
their own adapter when needed.

Memory is wired during forward via a closure that captures input_ids; the
MemInjectLayer calls memory_fn(h_mem_in) and receives the per-token
readouts back in d_mem space. Memory's auxiliary NTP loss is collected
from self._last_mem_pred_loss after each forward call.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from src.model.memory import MemoryGraph
from src.pretrained.config import PretrainedConfig
from src.pretrained.hosts import HostAdapter, build_host
from src.pretrained.mem_adapter import MemAdapter
from src.pretrained.mem_inject_layer import MemInjectLayer


class PretrainedLMWithMemory(nn.Module):
    def __init__(self, config: PretrainedConfig, attach_memory: bool = True):
        super().__init__()
        self.config = config

        # 1. Inspect HF config to populate derived fields. HostAdapter reads
        # the same values back off the model once it's loaded; this pre-load
        # peek is only so PretrainedConfig.validate_after_load() passes
        # BEFORE we commit to the heavy AutoModelForCausalLM download.
        hf_cfg = AutoConfig.from_pretrained(config.model_name)
        config.d_lm = hf_cfg.hidden_size
        config.n_lm_layers = hf_cfg.num_hidden_layers
        config.vocab_size_lm = hf_cfg.vocab_size
        config.validate_after_load()

        # 2. Load the LM in the configured dtype. Default bf16 matches the
        # production GPU run path and skips the fp32→bf16 weight cast that
        # autocast otherwise triggers on every matmul. Memory state dtype
        # follows the LM dtype (see begin_segment) so both sides agree.
        _dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16,
                       "fp16": torch.float16}
        llama_dtype = _dtype_map[config.llama_dtype]
        self.llama = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=llama_dtype)

        # 3. Build HostAdapter — dispatches on config.model_type. All
        # model-specific attribute paths go through this from here on.
        self.host: HostAdapter = build_host(self.llama)
        # Read norm eps back off the live adapter (source of truth post-load).
        self._rms_eps = self.host.norm_eps()

        # 4. Freeze backbone. Trainable pieces (W_in/W_out/scale + memory) are
        # added after this.
        if config.freeze_backbone:
            self.host.freeze_backbone()

        # 5. Swap the chosen layer with a MemInjectLayer. memory_fn=None so
        # the layer is a transparent pass-through until memory is wired
        # per-forward. W_in / W_out / scale stay at their fp32 nn.Parameter
        # default — bf16 param updates round small Adam steps to zero,
        # which silently stalls training on small smoke runs. The forward
        # handles the cross-dtype add explicitly.
        L = config.inject_layer
        orig_layer = self.host.layer_list()[L]
        self.host.replace_layer(L, MemInjectLayer(
            orig_layer=orig_layer,
            d_lm=config.d_lm,
            d_mem=config.d_mem,
            scale_init=config.scale_init,
            memory_fn=None,
        ))

        # 6. Memory graph: config.memory has D set to d_mem. vocab_size on the
        # memory config doesn't need to match the LM vocab — the aux-loss head
        # goes through the adapter, which uses the host's lm_head directly.
        config.memory.vocab_size = config.vocab_size_lm
        config.memory.validate()
        self.memory = MemoryGraph(config.memory) if attach_memory else None
        # Memory PARAMS stay in fp32. Bf16 optimizer updates round small
        # gradients to zero (param = 2.0, update = 1e-5 → 2.0 in bf16).
        # Memory STATE tensors (h, W, hebbian, ...) follow the host's dtype
        # via begin_segment so the forward-path matmuls don't mix dtypes.
        # Buffers (out_port_mask, role_id) also stay fp32 / int64; the
        # fused step casts out_port_mask to state dtype per call.
        self._adapter = (MemAdapter(self.host, self.mem_inject.W_out)
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
        # When True, memory.walk_segment keeps state graph-connected
        # across calls — no detach at end, no intra-call tbptt detach.
        # Used by autoregressive phase-1 unroll so gradient from each
        # continuation token reaches the prefix's modulator fires.
        # Caller is responsible for calling `detach_memory()` after
        # backward completes.
        self._preserve_autograd_graph: bool = False
        # `None` = follow `self.training`. `True`/`False` forces aux-loss
        # computation on or off regardless of training mode — useful for
        # AR continuation unroll that wants aux loss on the prefix pass
        # but not per-token during the unroll.
        self._compute_aux_loss_override: bool | None = None

    @property
    def mem_inject(self) -> MemInjectLayer:
        """Reference to the inserted MemInjectLayer via the host layer list."""
        return self.host.layer_list()[self.config.inject_layer]

    def begin_segment(self, bs: int):
        if self.memory is not None:
            p = next(self.llama.parameters())
            # Match memory state dtype to the host's dtype so h_mem crossings
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
        use_rms = self.host.use_rmsnorm()

        # Aux loss only matters during training. In eval / rollout we skip
        # the 128K-vocab mem_head matmul entirely — biggest single cost
        # at T=1 AR gen. `_compute_aux_loss_override` lets callers force
        # aux loss off even in training mode (AR unroll uses this to
        # skip aux loss during per-token continuation steps while keeping
        # modulator-fire gradient intact).
        override = getattr(self, "_compute_aux_loss_override", None)
        compute_aux = self.training if override is None else bool(override)

        def memory_fn(h_mem: torch.Tensor) -> torch.Tensor:
            readouts, mem_loss = self.memory.walk_segment(
                h_mem, input_ids, self._adapter,
                prev_token=prev_token,
                use_rmsnorm=use_rms,
                rms_eps=self._rms_eps,
                phase=self.current_phase,
                preserve_graph=self._preserve_autograd_graph,
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
            self.host.freeze_backbone()
        else:
            self.host.unfreeze_backbone()
        if self.memory is not None:
            for p in self.memory.parameters():
                p.requires_grad = True
        self.mem_inject.W_in.weight.requires_grad = True
        self.mem_inject.W_out.weight.requires_grad = True
        self.mem_inject.scale.requires_grad = True

    def preserve_autograd_graph(self):
        """Context manager: within the block, memory.walk_segment keeps
        state graph-connected across calls. Needed for autoregressive
        phase-1 unroll where continuation-token gradient must reach the
        prefix-pass modulator fires."""
        return _PreserveMemoryGraph(self)

    def rollout_mode(self):
        """Context manager for phase-2 rollouts.

        Inside the block:
          - `wrapper.train(False)` — modulator dropout is OFF (without
             this, two seeded rollouts diverge on the dropout noise that
             the Generator doesn't control, and log_pi_sum doesn't
             represent the actual action probabilities).
          - `memory._force_phase2_sampling = True` — `_modulate` still
             takes the hard-Categorical-with-log_pi branch even though
             we're in eval mode.
          - `memory.current_phase = "phase2"` for consistency (also
             threads to walk_segment so `_last_log_pi_sum` lands).
          - Autocast bf16 on CUDA.

        The `log_pi_sum` backward path remains graph-connected to the
        modulator logits — REINFORCE gradient is unchanged. Only dropout
        and the self.training flag have flipped."""
        return _RolloutMode(self)

    def compute_aux_loss_override(self, compute: bool | None):
        """Temporarily override whether `walk_segment` runs the aux
        `mem_pred_loss` head. `None` → use `self.training`. `False` → skip
        the 128K-vocab matmul (used during AR continuation unroll where
        only the prefix pass's aux signal is part of the loss)."""
        return _ComputeAuxLossOverride(self, compute)


class _PreserveMemoryGraph:
    def __init__(self, wrapper: "PretrainedLMWithMemory"):
        self.wrapper = wrapper
        self._prior = wrapper._preserve_autograd_graph

    def __enter__(self):
        self.wrapper._preserve_autograd_graph = True
        return self.wrapper

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.wrapper._preserve_autograd_graph = self._prior
        return False


class _RolloutMode:
    """Phase-2 rollout context: eval mode (no dropout) + hard Categorical
    sampling in memory + current_phase=phase2 + bf16 autocast on CUDA.
    Restores prior state on exit."""

    def __init__(self, wrapper: "PretrainedLMWithMemory"):
        self.wrapper = wrapper
        self._prior_training = wrapper.training
        self._prior_phase = wrapper.current_phase
        self._prior_force = (wrapper.memory._force_phase2_sampling
                             if wrapper.memory is not None else False)
        self._amp = None

    def __enter__(self):
        self.wrapper.train(False)
        self.wrapper.current_phase = "phase2"
        if self.wrapper.memory is not None:
            self.wrapper.memory._force_phase2_sampling = True
        # bf16 autocast on CUDA keeps dtype math consistent with the
        # training loops. CPU path uses a null context.
        dev = next(self.wrapper.parameters()).device.type
        if dev == "cuda":
            self._amp = torch.autocast(device_type=dev, dtype=torch.bfloat16)
            self._amp.__enter__()
        return self.wrapper

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._amp is not None:
            self._amp.__exit__(exc_type, exc_val, exc_tb)
            self._amp = None
        if self.wrapper.memory is not None:
            self.wrapper.memory._force_phase2_sampling = self._prior_force
        self.wrapper.current_phase = self._prior_phase
        self.wrapper.train(self._prior_training)
        return False


class _ComputeAuxLossOverride:
    def __init__(self, wrapper: "PretrainedLMWithMemory",
                  compute: bool | None):
        self.wrapper = wrapper
        self.new = compute
        self._prior = getattr(wrapper, "_compute_aux_loss_override", None)

    def __enter__(self):
        self.wrapper._compute_aux_loss_override = self.new
        return self.wrapper

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.wrapper._compute_aux_loss_override = self._prior
        return False
