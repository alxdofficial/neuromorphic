"""PretrainedGWConfig — wraps a GraphWalkerConfig with HF-LM + integration knobs.

Layer 0..L-1 and L+1..N-1 of the host LM are frozen; layer L is wrapped
with a `MemInjectLayer` that runs the graph_walker and adds
`scale * W_out(readout)` to the residual before delegating to the original
layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.graph_walker.config import GraphWalkerConfig


@dataclass
class PretrainedGWConfig:
    # === Host LM ===
    model_name: str = "meta-llama/Llama-3.2-1B"
    inject_layer: int = 8
    freeze_backbone: bool = True
    # "bf16" matches the production GPU path; "fp32" is only for the
    # bit-exact vanilla-Llama parity smoke test.
    llama_dtype: str = "bf16"

    # LM fields populated from HF config at load time.
    d_lm: int = -1
    n_lm_layers: int = -1
    vocab_size_lm: int = -1

    # === Graph-walker memory ===
    # d_mem is the LM-facing dim of the inject layer; walker's internal state
    # dim is GraphWalkerConfig.D_s. If d_mem == D_s they match directly (no
    # extra projection); else W_in/W_out Xavier-init bridges them.
    d_mem: int = 512
    memory: GraphWalkerConfig = field(default_factory=GraphWalkerConfig)

    # === Inject gate ===
    # Per-dim scale applied to memory readout before adding into the LM
    # residual. Init = sqrt(alpha) = 2.0 by default; pinned to 0 when
    # memory is detached so the layer is bit-exact with vanilla Llama.
    scale_init: float = 2.0

    # === Phase 1 driver defaults ===
    T: int = 256           # segment length passed to `wrapper(...)`
    bs: int = 8            # phase-1 batch size
    grad_clip: float = 1.0
    walker_aux_weight: float = 1.0    # weight on walker's own multi-horizon aux CE
    lm_aux_weight: float = 0.1        # weight on adapter-side horizon-1 CE
    ce_weight: float = 1.0            # weight on primary Llama CE

    # === Phase 2 (GRPO) ===
    grpo_K: int = 8
    grpo_rollout_len: int = 128
    grpo_adv_std_floor: float = 1e-3

    def validate(self) -> None:
        assert self.inject_layer >= 1, "inject_layer must be >= 1"
        # The graph_walker's D_s governs the internal graph state. MemInjectLayer's
        # d_mem is what W_in/W_out use. Keep them aligned by default — if the
        # caller wants a narrower internal graph state they can tune both.
        assert self.memory.D_s == self.d_mem, (
            f"memory.D_s ({self.memory.D_s}) must equal d_mem ({self.d_mem}). "
            "Walker's internal state dim and the MemInjectLayer d_mem must match "
            "— forward_segment passes h_mem directly into the walker as h_input."
        )
        # Runtime clocks must satisfy graph_walker's invariants.
        if self.T % self.memory.mod_period != 0:
            raise ValueError(
                f"T ({self.T}) must be a multiple of memory.mod_period "
                f"({self.memory.mod_period})."
            )

    def validate_after_load(self) -> None:
        assert self.d_lm > 0, "d_lm populated from HF config"
        assert self.n_lm_layers > 0, "n_lm_layers populated from HF config"
        assert 1 <= self.inject_layer < self.n_lm_layers, (
            f"inject_layer ({self.inject_layer}) must be in [1, {self.n_lm_layers})"
        )

    # ----------------------------------------------------------------
    # Factories
    # ----------------------------------------------------------------

    @classmethod
    def _make_memory(cls, d_mem: int, T: int, **mem_kw) -> GraphWalkerConfig:
        """Build a graph_walker memory config sized for pretrained runs.

        Picks topology / clocks compatible with the driver defaults:
          - `D_s = d_mem` so W_in/W_out match the walker's internal dim.
          - `segment_T = T` so the walker runs exactly one segment per forward.
          - Default topology (16x16, L=4) gives N=1024 columns — good balance
            of capacity vs per-token wall time. Override `plane_rows` etc via
            `mem_kw` for smaller smoke runs.
        """
        base = dict(
            D_s=d_mem,
            D_model=d_mem,            # walker's own internal "model width"; tied to state dim for simplicity
            segment_T=T,
            mod_period=min(128, T),
            tbptt_block=min(128, T),  # must equal mod_period
            vocab_size=32_000,        # placeholder; overwritten from HF
        )
        base.update(mem_kw)
        return GraphWalkerConfig(**base)

    @classmethod
    def llama_1b(cls, **kw) -> "PretrainedGWConfig":
        defaults = dict(
            model_name="meta-llama/Llama-3.2-1B",
            inject_layer=8,
            d_mem=512,
            T=256,
        )
        defaults.update(kw)
        mem_kw = defaults.pop("memory_kw", {})
        mem = defaults.pop("memory", None) or cls._make_memory(
            d_mem=defaults["d_mem"], T=defaults["T"], **mem_kw,
        )
        c = cls(memory=mem, **defaults)
        c.validate()
        return c

    @classmethod
    def llama_3b(cls, **kw) -> "PretrainedGWConfig":
        defaults = dict(
            model_name="meta-llama/Llama-3.2-3B",
            inject_layer=14,
            d_mem=512,
            T=256,
        )
        defaults.update(kw)
        mem_kw = defaults.pop("memory_kw", {})
        mem = defaults.pop("memory", None) or cls._make_memory(
            d_mem=defaults["d_mem"], T=defaults["T"], **mem_kw,
        )
        c = cls(memory=mem, **defaults)
        c.validate()
        return c

    @classmethod
    def smollm2_135m(cls, **kw) -> "PretrainedGWConfig":
        """Smallest dev host — 135M params, 30 layers, Llama arch.

        `d_lm=576` on this checkpoint; we leave `d_mem=576` so W_in/W_out
        are identity-initable.
        """
        defaults = dict(
            model_name="HuggingFaceTB/SmolLM2-135M",
            inject_layer=15,
            d_mem=576,
            T=256,
        )
        defaults.update(kw)
        mem_kw = defaults.pop("memory_kw", {})
        mem = defaults.pop("memory", None) or cls._make_memory(
            d_mem=defaults["d_mem"], T=defaults["T"], **mem_kw,
        )
        c = cls(memory=mem, **defaults)
        c.validate()
        return c

    @classmethod
    def tinyllama_1b1(cls, **kw) -> "PretrainedGWConfig":
        """TinyLlama 1.1B — 22 layers, LlamaForCausalLM. d_lm=2048."""
        defaults = dict(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            inject_layer=11,
            d_mem=512,
            T=256,
        )
        defaults.update(kw)
        mem_kw = defaults.pop("memory_kw", {})
        mem = defaults.pop("memory", None) or cls._make_memory(
            d_mem=defaults["d_mem"], T=defaults["T"], **mem_kw,
        )
        c = cls(memory=mem, **defaults)
        c.validate()
        return c

    @classmethod
    def tiny_test(cls, **kw) -> "PretrainedGWConfig":
        """Miniature config for CPU smoke tests — uses SmolLM2-135M + a
        tiny graph walker (8x8, L=2). Runs in a few seconds."""
        tiny_mem = GraphWalkerConfig(
            plane_rows=8, plane_cols=8, L=2,
            K=8, D_model=64, D_s=64, D_id=16,
            n_heads=2, n_hops=3,
            D_q_in=16, D_q_per_head=16, n_score_heads=2,
            K_horizons=4, K_buf=4,
            vocab_size=256,
            mod_period=4, tbptt_block=4, segment_T=8,
            gumbel_tau_start=1.0, gumbel_tau_end=1.0, gumbel_anneal_steps=1,
            epsilon_start=0.0, epsilon_end=0.0, epsilon_anneal_steps=1,
            lambda_balance=0.0,
            use_neuromod=True,
            neuromod_D_mod=32, neuromod_n_layers=1, neuromod_n_heads=2,
            neuromod_edge_hidden=16, neuromod_eta=1.0,
        )
        defaults = dict(
            model_name="HuggingFaceTB/SmolLM2-135M",
            inject_layer=15,
            d_mem=64,
            T=8,
            bs=2,
            llama_dtype="fp32",
            memory=tiny_mem,
        )
        defaults.update(kw)
        c = cls(**defaults)
        c.validate()
        return c
