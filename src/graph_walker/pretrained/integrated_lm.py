"""IntegratedLM — wraps a frozen HF causal LM with a graph_walker
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
    model = IntegratedLM(config)
    model.begin_segment(bs)
    out = model(input_ids)              # returns HF ModelOutput; `.logits`
    model.detach_memory()
    model.trainable_parameters()        # yields (name, param) for requires_grad params

The walker is vocab-agnostic: it has no LM head, no aux loss, no surprise
of its own. The trainer drives plasticity externally — after `loss.backward()`,
compute Llama's per-token CE and call ``model.memory.update_plasticity(per_token_ce)``.
See `src/graph_walker/pretrained/train_phase1.py` for the canonical pattern.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from src.graph_walker.graph_walker import GraphWalkerMemory
from src.graph_walker.pretrained.config import PretrainedGWConfig
from src.pretrained.hosts import build_host
from src.pretrained.hosts.base import HostAdapter
from src.pretrained.mem_inject_layer import MemInjectLayer


class IntegratedLM(nn.Module):
    def __init__(
        self,
        config: PretrainedGWConfig,
        attach_memory: bool = True,
        hf_model: nn.Module | None = None,
    ):
        """Construct a model.

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
        # The model-level setting is propagated to `self.memory.phase`
        # at the start of every `forward()` call (which is what routing
        # actually reads), so model.current_phase is the canonical
        # control surface — callers should never set memory.phase
        # directly when going through the model.
        self.current_phase: str = "phase1"
        # When True, `walk_segment` keeps memory state graph-connected
        # across forwards (no intra-segment detach, no end-of-forward
        # detach). Set by `preserve_autograd_graph()` for AR unroll.
        self._preserve_autograd_graph: bool = False

        # validate() catches direct PretrainedGWConfig(...) construction
        # where d_mem and memory.D_s drift apart (default d_mem=512 but
        # default GraphWalkerConfig.D_s=256 would crash later in
        # walk_segment). Factories already produce matching configs;
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

        # The walker is vocab-agnostic in the integration path (no LM head,
        # no aux CE), but `GraphWalkerMemory` still requires a `tied_token_emb`
        # at construction because the standalone walker uses it. Stub it with
        # a small embedding sized to Llama's vocab so any back-compat code
        # path that touches it doesn't blow up; the integration path never
        # reads it. Mark its weight non-trainable so the optimizer doesn't
        # carry dead state for it.
        config.memory.vocab_size = config.vocab_size_lm
        self.memory = GraphWalkerMemory(
            config.memory,
            tied_token_emb=nn.Embedding(config.vocab_size_lm, config.memory.D_model),
        ) if attach_memory else None

        if self.memory is not None:
            nn.init.normal_(self.memory.tied_token_emb.weight, std=0.02)
            # Freeze params that participate only in the standalone walker's
            # token-id-driven hot path. In the integration we feed h_mem
            # not token ids, and the walker has no LM head of its own, so
            # these get no gradient and are dead weight in the optimizer
            # state.
            for p_name in ("token_to_state", "input_v_proj", "tied_token_emb"):
                p = getattr(self.memory, p_name).weight
                p.requires_grad = False
            # `state_to_model` and the entire `readout` submodule (multi-
            # horizon walker LM head) are exercised only by the dropped
            # aux-CE path. Freeze them too.
            self.memory.state_to_model.weight.requires_grad = False
            for p in self.memory.readout.parameters():
                p.requires_grad = False
            # `prev_motor_proj` only contributes when is_new_window fires
            # mid-segment. Under T = mod_period (the integration's clock
            # invariant) it fires exactly once per segment at tick 0,
            # where `prev_motor` is always zeros from begin_segment — so
            # the Linear's weight gradient is identically zero. Freeze.
            self.memory.prev_motor_proj.weight.requires_grad = False

            # Disable activation checkpointing on the whole-block forward in
            # the integration training path. This is the OPPOSITE of the
            # GraphWalkerMemory default (which is True via getattr fallback)
            # because the integration's BS-scaling profile inverts the
            # tradeoff:
            #
            # When True (`_checkpoint_block=True`), backward re-runs the
            # `compile_block_from_h`-compiled walker block to regenerate
            # intermediates. This second invocation triggers a SEPARATE
            # inductor compilation of the backward gradient kernels. At
            # BS≥16 that backward compile hits a cuBLAS autotuner failure
            # (`select_algorithm.py: "Constructing input/output tensor
            # meta failed for Extern Choice"`) and falls back to a
            # significantly slower kernel path. Empirically:
            #   BS=12 + ckpt=True:  4.5k tok/s, 11.5 GB peak  (works)
            #   BS=16 + ckpt=True:  2.2k tok/s, 14.0 GB peak  (cliff)
            #   BS=16 + ckpt=False: 8.8k tok/s, 14.6 GB peak  (4× faster
            #                       than ckpt=True, +0.6 GB cost)
            #
            # The memory savings ckpt=True buys (~3 GB at BS=16) is not
            # worth the 4× wall-clock cost when the GPU has 9+ GB of
            # headroom at BS=16. At BS=4 ckpt=True is still preferable
            # (small block forward → cheap recompute → memory matters
            # more), but the integration's production-target BS is well
            # above that threshold.
            #
            # Standalone walker training (`src/graph_walker/train_phase1.py`)
            # uses `step_core` per-token, not the compile-block path, so
            # `_checkpoint_block` is irrelevant there and the fallback
            # True default is harmless.
            #
            # See `docs/bench_results.md` (2026-05-03) for the full
            # diagnostic table.
            self.memory._checkpoint_block = False

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

    def begin_segment(
        self, bs: int, *, clear_neuromod_carryover: bool = False,
    ) -> None:
        """Re-init walker working state for a new batch of `bs` segments.
        E_bias persists across calls (it's the long-term plastic state).

        `clear_neuromod_carryover` defaults to **False** under the external-
        surprise design: plasticity fires once per training step (post-
        backward), which means the only way ``_active_neuromod_delta`` is non-
        None during the next segment's forward is to keep the previous
        step's `_neuromod_input_*`. Clearing the snapshot starves neuromod
        of gradient — its parameters get no signal because routing reads
        a None delta and falls back to E_bias_flat only.

        For shuffled independent batches the previous batch's snapshot is
        from different documents, but the gradient signal is still
        "given THIS column-feature snapshot → produce a delta that helps
        the LM here", which is a sensible generalizing target.
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
        self,
        mode: str = "default",
        fullgraph: bool = True,
        *,
        regional: bool = False,
        dynamic: bool | None = False,
    ) -> None:
        """Compile the walker's forward path. Two regimes:

        **Whole-block (default, `regional=False`):** compile
        ``block_forward_from_h`` — one big graph that unrolls T=mod_period
        walker steps + the inject path. Inductor fuses across step
        boundaries → ~3.7× over eager. **First-compile cost: 10-15 min**
        (T=256 unrolling means thousands of ops to lower).

        **Regional (`regional=True`):** compile ``step_core_from_h`` only
        — the per-step graph that gets called T times in the Python loop.
        Inductor fuses within each step (~2× over eager) but does NOT fuse
        across step boundaries. **First-compile cost: 1-2 min** because
        the per-step graph is ~T× smaller. This is the canonical PyTorch
        recommendation for "repeated-block" patterns
        (https://docs.pytorch.org/tutorials/recipes/regional_compilation.html).

        Trade-off: regional gives 5-15% lower per-iter throughput in
        exchange for an order-of-magnitude faster compile. Use it for
        dev iteration / BS sweeps; flip to whole-block for final
        production training where compile cost is amortized over a
        ~30-hour run.

        ``mode="default"`` mirrors the standalone speedup configuration;
        the ``"reduce-overhead"`` cudagraph variant is incompatible with
        Llama's dynamic activation addresses.

        ``dynamic=None`` enables PyTorch's auto-detect dynamic shapes
        (after the second call with a different shape, inductor recompiles
        a shape-polymorphic kernel). Useful for cross-shape compile reuse
        in BS sweeps. ``dynamic=False`` (default) static-specializes —
        best per-iter throughput but recompiles on every shape change.
        ``dynamic=True`` is NOT supported (PyTorch warns it crashes / is
        slow on big graphs like ours).

        Idempotent: calling twice replaces the compiled function.
        """
        if self.memory is None:
            return
        if dynamic is True:
            raise ValueError(
                "dynamic=True is unsafe for the walker's whole-block compile "
                "(per PyTorch docs: crashes / runs slow on big graphs). "
                "Use dynamic=None (auto-detect) or dynamic=False (default)."
            )
        if regional:
            self.memory.compile_step(mode=mode, dynamic=dynamic)
        else:
            self.memory.compile_block_from_h(
                mode=mode, fullgraph=fullgraph, dynamic=dynamic,
            )

    # ------------------------------------------------------------------
    # AR-unroll support
    # ------------------------------------------------------------------

    class _PreserveAutogradGraphCtx:
        def __init__(self, model: "IntegratedLM"):
            self.model = model
        def __enter__(self):
            self.model._preserve_autograd_graph = True
            return self.model
        def __exit__(self, *args):
            self.model._preserve_autograd_graph = False

    def preserve_autograd_graph(self) -> "IntegratedLM._PreserveAutogradGraphCtx":
        """Context manager: memory state stays graph-connected across calls
        made inside. Caller is responsible for calling `detach_memory()`
        after backward completes."""
        return IntegratedLM._PreserveAutogradGraphCtx(self)

    # ------------------------------------------------------------------
    # Per-batch state snapshot / restore (multi-turn GRPO support)
    # ------------------------------------------------------------------

    def snapshot_memory_state(self) -> dict | None:
        """Snapshot the walker's PER-BATCH working state for later
        restoration. Used by multi-turn GRPO's aligned-trajectory
        protocol: snapshot at turn boundary, K rollouts diverge, restore
        before forwarding the ground-truth turn to advance the canonical
        state.

        What's saved:
        - `s` (LIF state per column)
        - `prev_motor`, `walker_pos`, `walker_state`
        - `surprise_ema`, `surprise_prev`
        - `_log_pi_sum`, `_log_pi_count` (phase-2 routing log-π accumulator)
        - `tick_counter`, `window_len` (segment-level counters)
        - `co_visit_flat`, `visit_count` (per-segment counters)

        What's NOT saved (keeps evolving across the whole session):
        - `E_bias_flat` (long-term plastic state)
        - `_neuromod_input_*` (neuromod cross-window snapshots)
        - `_active_neuromod_delta` (current window's neuromod delta — this
          IS captured indirectly via _neuromod_input's lifecycle, but
          would need an explicit save if cross-rollout consistency is
          ever needed)

        Returns None if memory is detached. Returned dict's tensors are
        clones (independent of the walker's live tensors). The caller
        owns the dict.
        """
        if self.memory is None:
            return None
        m = self.memory
        if not getattr(m, "_state_initialized", False):
            return None

        def _clone_or_none(x):
            return x.clone() if isinstance(x, torch.Tensor) else None

        return {
            "s": _clone_or_none(m.s),
            "prev_motor": _clone_or_none(m.prev_motor),
            "walker_pos": _clone_or_none(m.walker_pos),
            "walker_state": _clone_or_none(m.walker_state),
            "surprise_ema": _clone_or_none(m.surprise_ema),
            "surprise_prev": _clone_or_none(m.surprise_prev),
            "_log_pi_sum": _clone_or_none(m._log_pi_sum),
            "_log_pi_count": int(m._log_pi_count),
            "tick_counter": int(m.tick_counter),
            "window_len": int(m.window_len),
            "co_visit_flat": _clone_or_none(m.co_visit_flat),
            "visit_count": _clone_or_none(m.visit_count),
        }

    def restore_memory_state(self, state: dict | None) -> None:
        """Restore a snapshot from `snapshot_memory_state`. The walker's
        per-batch working state is overwritten from clones of the
        snapshot's tensors (so a subsequent forward doesn't mutate the
        snapshot itself).

        Long-term plastic state (E_bias_flat, _neuromod_input_*) is NOT
        touched — it keeps evolving on its own update cadence.
        """
        if state is None or self.memory is None:
            return
        m = self.memory

        def _restore(name):
            v = state.get(name)
            if isinstance(v, torch.Tensor):
                setattr(m, name, v.clone())

        _restore("s")
        _restore("prev_motor")
        _restore("walker_pos")
        _restore("walker_state")
        _restore("surprise_ema")
        _restore("surprise_prev")
        _restore("co_visit_flat")
        _restore("visit_count")
        # log_pi sum: tensor or None
        log_pi_sum = state.get("_log_pi_sum")
        m._log_pi_sum = (
            log_pi_sum.clone() if isinstance(log_pi_sum, torch.Tensor) else None
        )
        m._log_pi_count = int(state.get("_log_pi_count", 0))
        m.tick_counter = int(state.get("tick_counter", 0))
        m.window_len = int(state.get("window_len", 0))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, **kwargs):
        """Full LM forward with memory read/write at the injection layer.

        Returns the HF ModelOutput (has `.logits`). The walker is vocab-
        agnostic — no aux loss is computed here. Plasticity is driven by
        the trainer via `self.memory.update_plasticity(per_token_ce)` after
        `loss.backward()` (see `train_phase1.phase1_pretrained_step`).
        """
        if self.memory is None:
            return self.llama(input_ids=input_ids, **kwargs)

        # Propagate the model-level phase indicator into the memory module
        # — routing inside `_step_core_pure` reads `memory.phase` only, so
        # without this propagation `model.current_phase` would be dead
        # state and a caller setting `model.current_phase = "phase2"`
        # would silently still get phase-1 Gumbel-STE routing.
        self.memory.phase = self.current_phase

        # Closure: MemInjectLayer calls this with h_mem = W_in(hidden_states).
        # The walker turns h_mem into per-token readouts; W_out then projects
        # readouts back to LM hidden-state space and adds them as a residual.
        def memory_fn(h_mem: torch.Tensor) -> torch.Tensor:
            return self.memory.walk_segment(
                h_mem, preserve_graph=self._preserve_autograd_graph,
            )

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
        # gradient through `walk_segment` — without this, every cycle's
        # `unfreeze_all()` puts dead weights back into the optimizer.
        # MUST mirror the freeze list applied at __init__ (lines 147-161).
        # Earlier this only re-froze 2 of 6 dead params, so cycle-phase-1
        # rebuilds optimizer state for ~10M params that get no gradient.
        if self.memory is not None:
            for p_name in ("token_to_state", "input_v_proj", "tied_token_emb"):
                getattr(self.memory, p_name).weight.requires_grad = False
            self.memory.state_to_model.weight.requires_grad = False
            for p in self.memory.readout.parameters():
                p.requires_grad = False
            self.memory.prev_motor_proj.weight.requires_grad = False

    def freeze_all_but_E_bias_and_neuromod(self) -> None:
        """Phase-2 minimal policy surface: only neuromod + E_bias evolve.
        Everything else frozen. E_bias is a buffer (not a Parameter) so it
        evolves via the plasticity pathway rather than gradient — but freezing
        other walker params means only neuromod gradients drive the policy."""
        for name, p in self.named_parameters():
            p.requires_grad = name.startswith("memory.neuromod.")
