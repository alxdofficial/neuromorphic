"""GraphWalkerPretrainedLM — wraps a frozen HF causal LM with a graph_walker
memory side-channel at a chosen decoder layer.

Structural layout (unchanged from v2 — the scaffolding is memory-agnostic):

    input_ids
       ▼
    embed_tokens
       ▼
    layers 0..L-1              (frozen)
       ▼
    MemInjectLayer:            (W_in / W_out / scale trainable)
        h_mem = W_in(h)        [BS, T, d_mem=D_s]
        readout = memory_fn(h_mem, input_ids)
        h' = h + scale * W_out(readout)
        orig_layer(h')         (frozen layer L body)
       ▼
    layers L+1..N-1            (frozen)
       ▼
    norm → lm_head → logits

Public API:
    wrapper = GraphWalkerPretrainedLM(config)
    wrapper.reset_memory(bs)
    out = wrapper(input_ids)              # returns HF ModelOutput; `.logits`
    wrapper.detach_memory()
    wrapper._last_mem_loss                # aux loss from the last forward (None in eval)
    wrapper.trainable_parameters()        # yields (name, param) for requires_grad params
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from src.graph_walker.graph_walker import GraphWalkerMemory
from src.graph_walker.pretrained.config import PretrainedGWConfig
from src.pretrained.hosts import build_host
from src.pretrained.hosts.base import HostAdapter
from src.pretrained.mem_adapter import MemAdapter
from src.pretrained.mem_inject_layer import MemInjectLayer


class GraphWalkerPretrainedLM(nn.Module):
    def __init__(
        self,
        config: PretrainedGWConfig,
        attach_memory: bool = True,
        hf_model: nn.Module | None = None,
    ):
        """Construct a wrapper.

        `hf_model` lets callers pass a pre-instantiated HF CausalLM — useful
        for tests that want to avoid the HF Hub download (e.g. build a
        tiny random-weights LlamaForCausalLM via `AutoModelForCausalLM.from_config`).
        When `None`, loads `config.model_name` via the Hub in the standard
        bf16/fp32 flow.
        """
        super().__init__()
        self.config = config

        # Phase indicator. "phase1" = Gumbel-soft STE in routing; "phase2" =
        # hard Categorical + log_pi accumulation (flip before grpo_step).
        # The wrapper-level setting is propagated to `self.memory.phase`
        # at the start of every `forward()` call (which is what routing
        # actually reads), so wrapper.current_phase is the canonical
        # control surface — callers should never set memory.phase
        # directly when going through the wrapper.
        self.current_phase: str = "phase1"
        # When True, `forward_segment` keeps memory state graph-connected
        # across forwards (no intra-segment detach, no end-of-forward
        # detach). Set by `preserve_memory_graph()` for AR unroll.
        self._preserve_memory_graph: bool = False
        # Override for aux-loss computation. None = follow self.training.
        self._compute_aux_loss_override: bool | None = None
        # Scratch: aux loss from the last forward (None if no memory or eval).
        self._last_mem_loss: torch.Tensor | None = None

        # validate() catches direct PretrainedGWConfig(...) construction
        # where d_mem and memory.D_s drift apart (default d_mem=512 but
        # default GraphWalkerConfig.D_s=256 would crash later in
        # forward_segment). Factories already produce matching configs;
        # this guards the "user instantiated by hand" path.
        config.validate()

        if hf_model is None:
            hf_cfg = AutoConfig.from_pretrained(config.model_name)
            config.d_lm = hf_cfg.hidden_size
            config.n_lm_layers = hf_cfg.num_hidden_layers
            config.vocab_size_lm = hf_cfg.vocab_size
            config.validate_after_load()

            dtype_map = {
                "fp32": torch.float32, "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }
            llama_dtype = dtype_map[config.llama_dtype]
            self.llama = AutoModelForCausalLM.from_pretrained(
                config.model_name, torch_dtype=llama_dtype,
            )
        else:
            hf_cfg = hf_model.config
            config.d_lm = hf_cfg.hidden_size
            config.n_lm_layers = hf_cfg.num_hidden_layers
            config.vocab_size_lm = hf_cfg.vocab_size
            config.validate_after_load()
            self.llama = hf_model

        self.host: HostAdapter = build_host(self.llama)
        self._rms_eps = self.host.norm_eps()

        if config.freeze_backbone:
            self.host.freeze_backbone()

        # Replace the chosen layer with MemInjectLayer (transparent until
        # memory_fn is installed per-call in self.forward).
        L = config.inject_layer
        orig_layer = self.host.layer_list()[L]
        self.host.replace_layer(L, MemInjectLayer(
            orig_layer=orig_layer,
            d_lm=config.d_lm,
            d_mem=config.d_mem,
            scale_init=config.scale_init,
            memory_fn=None,
        ))

        # Populate walker's tied_token_emb vocab to match Llama's. Walker
        # uses its own internal embedding for its multi-horizon aux CE
        # (surprise fold + walker-side aux loss) — same vocab as Llama so
        # `input_ids` work as targets directly.
        config.memory.vocab_size = config.vocab_size_lm
        self.memory = GraphWalkerMemory(
            config.memory,
            tied_token_emb=nn.Embedding(config.vocab_size_lm, config.memory.D_model),
        ) if attach_memory else None

        # Small-init the walker's tied embed so the aux-CE readout isn't
        # catastrophic at step 0.
        if self.memory is not None:
            nn.init.normal_(self.memory.tied_token_emb.weight, std=0.02)
            # Freeze walker params that ONLY participate in the standalone
            # (token-id-driven) hot path — they get no gradient through
            # forward_segment so leaving them trainable is dead weight in
            # the optimizer state.
            for p_name in ("token_to_state", "input_v_proj"):
                p = getattr(self.memory, p_name).weight
                p.requires_grad = False

        # Adapter: supplies mem_head_logits(x: [B, d_mem] → [B, vocab]) to
        # the walker's forward_segment.
        self._adapter = (
            MemAdapter(self.host, self.mem_inject.W_out)
            if attach_memory else None
        )

        # Without memory, pin scale to zero so MemInjectLayer passes through
        # transparently (its runtime check enforces scale==0 then).
        if not attach_memory:
            with torch.no_grad():
                self.mem_inject.scale.zero_()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mem_inject(self) -> MemInjectLayer:
        """Reference to the inserted MemInjectLayer via the host layer list."""
        return self.host.layer_list()[self.config.inject_layer]

    # ------------------------------------------------------------------
    # State management (per-segment reset + TBPTT detach)
    # ------------------------------------------------------------------

    def reset_memory(
        self, bs: int, *, clear_neuromod_carryover: bool = True,
    ) -> None:
        """Re-init walker working state for a new batch of `bs` segments.
        E_bias persists across calls (it's the long-term plastic state).

        `clear_neuromod_carryover` defaults to True for the pretrained
        path because batches are typically shuffled independent documents,
        and carrying the previous batch's last-window snapshot into the
        new batch's first neuromod target would inject cross-document
        noise into credit assignment. Set to False explicitly when
        batches are contiguous chunks of the same document.
        """
        if self.memory is None:
            return
        device = next(self.llama.parameters()).device
        self.memory.begin_segment(
            bs, device, clear_neuromod_carryover=clear_neuromod_carryover,
        )

    def detach_memory(self) -> None:
        if self.memory is not None:
            self.memory.detach_state()

    def compile_walker_block(
        self, mode: str = "default", fullgraph: bool = True,
    ) -> None:
        """Compile the walker's whole-block forward path.

        ``forward_segment`` then routes each ``tbptt_block``-token window
        through one compiled call instead of T_block per-token
        ``step_core_from_h`` calls. Inductor fuses across step boundaries,
        replacing the per-token launch overhead that bottlenecked the
        eager Llama+walker path. ``mode="default"`` mirrors the standalone
        speedup configuration (~3.7× over per-token eager); the
        ``"reduce-overhead"`` cudagraph variant is incompatible with
        Llama's dynamic activation addresses.

        Idempotent: calling twice replaces the compiled function.
        """
        if self.memory is not None:
            self.memory.compile_block_from_h(mode=mode, fullgraph=fullgraph)

    # ------------------------------------------------------------------
    # AR-unroll support
    # ------------------------------------------------------------------

    class _PreserveGraphCtx:
        def __init__(self, wrapper: "GraphWalkerPretrainedLM"):
            self.wrapper = wrapper
        def __enter__(self):
            self.wrapper._preserve_memory_graph = True
            return self.wrapper
        def __exit__(self, *args):
            self.wrapper._preserve_memory_graph = False

    def preserve_memory_graph(self) -> "GraphWalkerPretrainedLM._PreserveGraphCtx":
        """Context manager: memory state stays graph-connected across calls
        made inside. Caller is responsible for calling `detach_memory()`
        after backward completes."""
        return GraphWalkerPretrainedLM._PreserveGraphCtx(self)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, **kwargs):
        """Full LM forward with memory read/write at the injection layer.

        Returns the HF ModelOutput (has `.logits`). Aux loss from the walker
        is stashed on `self._last_mem_loss` (None if memory not attached or
        aux disabled).
        """
        if self.memory is None:
            return self.llama(input_ids=input_ids, **kwargs)

        # Propagate the wrapper-level phase indicator into the memory module
        # — routing inside `_step_core_pure` reads `memory.phase` only, so
        # without this propagation `wrapper.current_phase` would be dead
        # state and a caller setting `wrapper.current_phase = "phase2"`
        # would silently still get phase-1 Gumbel-STE routing.
        self.memory.phase = self.current_phase

        override = self._compute_aux_loss_override
        compute_aux = self.training if override is None else bool(override)

        # Closure: MemInjectLayer calls this with h_mem = W_in(hidden_states).
        # We feed h_mem into the walker along with input_ids for targets.
        walker_aux_weight = self.config.walker_aux_weight

        def memory_fn(h_mem: torch.Tensor) -> torch.Tensor:
            readouts, mem_loss = self.memory.forward_segment(
                h_mem, input_ids, self._adapter,
                compute_aux_loss=compute_aux,
                preserve_graph=self._preserve_memory_graph,
                walker_aux_weight=walker_aux_weight,
            )
            self._last_mem_loss = mem_loss
            return readouts

        self.mem_inject.set_memory_fn(memory_fn)
        try:
            out = self.llama(input_ids=input_ids, **kwargs)
        finally:
            self.mem_inject.set_memory_fn(None)
        return out

    # ------------------------------------------------------------------
    # Parameter accessors
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """Yield only params that require grad — walker + projections + scale."""
        for name, p in self.named_parameters():
            if p.requires_grad:
                yield name, p

    def memory_parameters(self):
        """Yield walker-only trainables (excludes W_in/W_out/scale)."""
        if self.memory is None:
            return
        prefix = "memory."
        for name, p in self.named_parameters():
            if p.requires_grad and name.startswith(prefix):
                yield name, p

    def inject_parameters(self):
        """Yield W_in/W_out/scale (the pretrained-specific trainables)."""
        prefix = f"host.hf_model.model.layers.{self.config.inject_layer}."
        # MemInjectLayer is registered at layer_list()[L], which for Llama
        # lives at host.hf_model.model.layers[L]. Its sub-params:
        # W_in.weight, W_out.weight, scale.
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if ".W_in.weight" in name or ".W_out.weight" in name or ".scale" in name:
                yield name, p

    # ------------------------------------------------------------------
    # Freeze helpers — one per stage in the cycle loop.
    # ------------------------------------------------------------------

    def unfreeze_all(self) -> None:
        """Restore the bootstrap trainable surface (memory + inject, LM frozen)."""
        for name, p in self.named_parameters():
            # Everything EXCEPT the backbone.
            if name.startswith("llama."):
                # But: MemInjectLayer is a replacement layer nested inside llama.
                is_mem_inject = (
                    f".layers.{self.config.inject_layer}.W_in" in name
                    or f".layers.{self.config.inject_layer}.W_out" in name
                    or f".layers.{self.config.inject_layer}.scale" in name
                )
                p.requires_grad = is_mem_inject
            else:
                p.requires_grad = True
        # Re-freeze Llama backbone (except MemInjectLayer's W_in/W_out/scale).
        if self.config.freeze_backbone:
            for name, p in self.llama.named_parameters():
                is_mem_inject = (
                    f".layers.{self.config.inject_layer}.W_in" in name
                    or f".layers.{self.config.inject_layer}.W_out" in name
                    or f".layers.{self.config.inject_layer}.scale" in name
                )
                p.requires_grad = is_mem_inject
        # Re-freeze walker params that are standalone-only and never see
        # gradient through `forward_segment` — without this, every cycle's
        # `unfreeze_all()` puts dead weights back into the optimizer.
        if self.memory is not None:
            for p_name in ("token_to_state", "input_v_proj"):
                getattr(self.memory, p_name).weight.requires_grad = False

    def freeze_all_but_E_bias_and_neuromod(self) -> None:
        """Phase-2 minimal policy surface: only neuromod + E_bias evolve.
        Everything else frozen. E_bias is a buffer (not a Parameter) so it
        evolves via the plasticity pathway rather than gradient — but freezing
        other walker params means only neuromod gradients drive the policy."""
        for name, p in self.named_parameters():
            p.requires_grad = name.startswith("memory.neuromod.")
