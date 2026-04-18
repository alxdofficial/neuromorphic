"""Config for pretrained-LM + memory integration.

Holds both the HF model identifiers and the memory-graph hyperparameters.
Layers 0..L-1 and L+1..N-1 of the host LM are frozen; layer L is wrapped
with a `MemInjectLayer` that runs the memory graph and adds scale·readout
to the residual before delegating to the original layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.model.config import Config as MemoryConfig


@dataclass
class PretrainedConfig:
    # === Host LM ===
    model_name: str = "meta-llama/Llama-3.2-1B"
    # Layer to wrap with MemInjectLayer. Defaults are doc-recommended mid-stack
    # choices: L=8 for 1B (16 layers), L=14 for 3B (28 layers).
    inject_layer: int = 8
    # Backbone freeze. The smoke test flips this off briefly; training stays on.
    freeze_backbone: bool = True
    # Llama load dtype. "fp32" preserves bit-exact vanilla-Llama parity for
    # unit tests. "bf16" matches CUDA autocast behavior and skips the
    # on-the-fly fp32→bf16 weight cast that autocast triggers on every
    # matmul (profile showed this was 75% of CUDA time at T=1 AR rollout).
    # Callers on GPU production runs should pick "bf16".
    llama_dtype: str = "fp32"
    # LM hidden dim — populated from HF config at load time.
    d_lm: int = -1
    # LM num layers — populated from HF config at load time.
    n_lm_layers: int = -1
    # LM vocab size — populated from HF config at load time.
    vocab_size_lm: int = -1

    # === Memory side ===
    # D_mem is the hidden dim of the memory graph (the custom Config.D).
    # Projection path: W_in: d_lm → d_mem, W_out: d_mem → d_lm, both trainable.
    # If d_lm == d_mem, projections are still Linear (trainable) but identity-init.
    d_mem: int = 2048
    # Full memory config. `validate()` enforces D_mem divisibility etc.
    # When callers use llama_1b()/llama_3b() without an explicit `memory=...`,
    # the factory mirrors the top-level driver defaults (e.g. T, tbptt_block)
    # into this config so docs and runtime clocks stay aligned.
    memory: MemoryConfig = field(default_factory=lambda: MemoryConfig())

    # === Inject gate ===
    # Per-dim scale applied to memory readout before adding into the LM
    # residual. Init = sqrt(alpha) per the pivot doc (empirically well-matched
    # to readout magnitude under 1/sqrt(alpha) pool).
    scale_init: float = 2.0

    # === Phase 1 (bootstrap) ===
    T: int = 512              # default segment length for phase-1 drivers
    bs: int = 16              # default batch size for phase-1 drivers
    tbptt_block: int = 32     # default detach boundary for default-created memory
    gumbel_tau_start: float = 1.0
    gumbel_tau_end: float = 0.3
    mem_pred_weight: float = 0.1

    # === Phase 2 (GRPO rollouts) ===
    grpo_K: int = 8           # rollouts per group
    grpo_rollout_len: int = 256  # tokens generated per rollout
    grpo_adv_std_floor: float = 1e-3  # min denominator for advantage normalization
    # KL-to-reference-policy and entropy-bonus penalties are not yet wired
    # into grpo_step. Adding them requires a reference modulator snapshot
    # (KL) and per-fire logits plumbed out of the memory graph (entropy).
    # Kept as future work; no config knobs here to avoid silent no-ops.

    def validate(self):
        assert self.inject_layer >= 1, "inject_layer must be >= 1 (no point at L=0)"
        # Align memory.D with d_mem so the memory code's divisibility assumptions hold.
        self.memory.D = self.d_mem
        self.memory.D_embed = self.d_mem
        # `memory.validate()` runs at MemoryConfig.tier_a() time; rerun here
        # because we may have mutated D.
        self.memory.validate()
        # d_lm/n_lm_layers/vocab_size_lm are populated from HF; validate after load.

    def validate_after_load(self):
        """Run once HF config is inspected and d_lm/n_lm_layers are known."""
        assert self.d_lm > 0, "d_lm must be set from HF config before validate_after_load"
        assert self.n_lm_layers > 0, "n_lm_layers must be set from HF config"
        assert 1 <= self.inject_layer < self.n_lm_layers, (
            f"inject_layer ({self.inject_layer}) must be in "
            f"[1, {self.n_lm_layers}); L=0 injects before any LM computation, "
            f"L=n_layers injects after all layers (pointless).")

    @classmethod
    def llama_1b(cls, **kw) -> "PretrainedConfig":
        """Dev config — 1B host, 16 layers, inject at L=8."""
        defaults = dict(
            model_name="meta-llama/Llama-3.2-1B",
            inject_layer=8,
            d_mem=2048,
        )
        defaults.update(kw)
        mem = defaults.pop("memory", None) or MemoryConfig.tier_a(
            D=defaults["d_mem"],
            T=defaults.get("T", 512),
            tbptt_block=defaults.get("tbptt_block", 32),
        )
        c = cls(memory=mem, **defaults)
        c.validate()
        return c

    @classmethod
    def llama_3b(cls, **kw) -> "PretrainedConfig":
        """Target config — 3B host, 28 layers, inject at L=14."""
        defaults = dict(
            model_name="meta-llama/Llama-3.2-3B",
            inject_layer=14,
            d_mem=2048,
        )
        defaults.update(kw)
        mem = defaults.pop("memory", None) or MemoryConfig.tier_a(
            D=defaults["d_mem"],
            T=defaults.get("T", 512),
            tbptt_block=defaults.get("tbptt_block", 32),
        )
        c = cls(memory=mem, **defaults)
        c.validate()
        return c
