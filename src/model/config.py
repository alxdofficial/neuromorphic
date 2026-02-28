"""
Model configuration for the Neuromorphic LM (v4: iterative refinement).

Single dataclass holding all hyperparameters. Phase toggles control which
memory systems are active. Tier presets provide size configurations.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Architecture — v4 iterative refinement
    D: int = 512              # model width (internal)
    D_embed: int = -1         # embedding dim (defaults to D in validate())
    R: int = 4                # iterative refinement passes
    B_blocks: int = 4         # memory blocks
    C: int = 4                # columns per block
    D_col: int = -1            # derived: D // C (set in validate())
    D_pcm: int = 64           # PCM encoding dim
    position_attn_dim: int = -1  # derived as D_col // 4 in validate(), 0 to disable
    N: int = 512              # segment length
    N_C: int = -1             # derived: N // C (tokens per column after interleaved partitioning)
    K_segments: int = 2       # TBPTT chunk = K segments
    lambda_mix: float = 0.5   # damped pass mixing init
    ffn_expansion: int = 2    # column FFN width multiplier
    ffn_depth: int = 1        # L = FFN layers per stack; total FFN layers per R pass = 2L
    vocab_size: int = 32000   # set from tokenizer at runtime
    eot_id: int = 2           # set from tokenizer at runtime

    # Procedural Memory (single instance, batched as BS*B)
    r: int = 32               # PM slots
    a_max: float = 3.0        # max strength per slot
    budget_pm: float = 16.0   # sum(pm_a) budget per stream
    decay_pm: float = 0.999   # per-pass strength decay
    tau_pm: float = 1.0       # softmax temperature (default for PM slot selection)
    tau_pm_floor: float = 0.05
    tau_pm_ceil: float = 5.0
    ww_pm_default: float = 0.5
    ww_pm_floor: float = 0.0
    ww_pm_ceil: float = 2.0
    tau_route_pm: float = 1.0
    surprise_scale: float = 5.0
    g_pm_default: float = 0.5

    # Episodic Memory (single instance, batched as BS*B)
    M: int = 256              # EM capacity per bank
    k_ret: int = 16           # retrieval count
    C_em: int = 32            # candidates per pass (top-C_em across N positions)
    tau_em: float = 1.0
    ww_em_default: float = 0.5
    S_max: float = 3.0
    budget_em: float = 32.0
    decay_em: float = 0.999
    g_em_default: float = 0.3
    g_em_floor: float = 0.001
    g_em_ceil: float = 0.95
    tau_em_floor: float = 0.05
    tau_em_ceil: float = 5.0
    ww_em_floor: float = 0.0
    ww_em_ceil: float = 2.0
    decay_em_floor: float = 0.99
    decay_em_ceil: float = 0.9999

    # Neuromodulator architecture
    neuromod_hidden: int = 32
    content_proj_dim: int = 8

    # Predictive Coding Module (per column group, via grouped ops)
    pcm_enabled: bool = True
    pcm_pred_weight: float = 0.01

    # Regularization
    dropout: float = 0.1
    tie_embeddings: bool = True

    # FITB (Fill-In-The-Blank) pretraining
    fitb_id: int = -1              # <FITB> token ID, set from tokenizer at runtime
    null_id: int = -1              # <NULL> token ID, set from tokenizer at runtime
    mask_rate: float = 0.3         # fraction of tokens to mask for FITB pretraining
    span_mask_prob: float = 0.5    # probability of span masking vs random
    span_mask_mean_len: int = 3    # mean span length (geometric distribution)

    # Training
    use_compile: bool = True
    gradient_checkpointing: bool = False
    reset_on_doc_boundary: bool = True
    lifelong_mode: bool = False

    # Phase toggles
    pm_enabled: bool = True
    em_enabled: bool = True

    @property
    def T(self) -> int:
        """TBPTT chunk length = K_segments * N."""
        return self.K_segments * self.N

    def validate(self):
        """Check config validity. Call before model construction."""
        if self.D <= 0:
            raise ValueError(f"D ({self.D}) must be positive.")
        # Derive D_embed (defaults to D when not explicitly set)
        if self.D_embed == -1:
            self.D_embed = self.D
        if self.D_embed <= 0:
            raise ValueError(f"D_embed ({self.D_embed}) must be positive.")
        if self.R < 1:
            raise ValueError(f"R ({self.R}) must be >= 1.")
        if self.B_blocks < 1:
            raise ValueError(f"B_blocks ({self.B_blocks}) must be >= 1.")
        if self.C < 1:
            raise ValueError(f"C ({self.C}) must be >= 1.")
        if self.D % self.C != 0:
            raise ValueError(f"D ({self.D}) must be divisible by C ({self.C}).")
        # Derive D_col from D and C (MHA-style split)
        self.D_col = self.D // self.C
        # Derive position_attn_dim from D_col (0 to disable)
        if self.position_attn_dim == -1:
            self.position_attn_dim = self.D_col // 4
        if self.position_attn_dim < 0:
            raise ValueError(
                f"position_attn_dim ({self.position_attn_dim}) must be >= 0 "
                f"(0 to disable, -1 to auto-derive from D_col // 4)."
            )
        if self.D_pcm <= 0 and self.pcm_enabled:
            raise ValueError(f"D_pcm ({self.D_pcm}) must be positive when pcm_enabled.")
        if self.N < 1:
            raise ValueError(f"N ({self.N}) must be >= 1.")
        if self.N % self.C != 0:
            raise ValueError(
                f"N ({self.N}) must be divisible by C ({self.C}) "
                f"for interleaved token partitioning."
            )
        self.N_C = self.N // self.C
        if self.K_segments < 1:
            raise ValueError(f"K_segments ({self.K_segments}) must be >= 1.")
        if self.tau_route_pm <= 0:
            raise ValueError(
                f"tau_route_pm ({self.tau_route_pm}) must be positive."
            )
        if self.r < 1:
            raise ValueError(f"r ({self.r}) must be >= 1 (PM slots).")
        if self.M < 1:
            raise ValueError(f"M ({self.M}) must be >= 1 (EM capacity).")
        if self.k_ret > self.M:
            raise ValueError(
                f"k_ret ({self.k_ret}) must be <= M ({self.M})."
            )
        if not 0.0 <= self.mask_rate <= 1.0:
            raise ValueError(
                f"mask_rate ({self.mask_rate}) must be in [0, 1]."
            )
        if not 0.0 <= self.span_mask_prob <= 1.0:
            raise ValueError(
                f"span_mask_prob ({self.span_mask_prob}) must be in [0, 1]."
            )

    def set_phase(self, phase: str):
        """Set component toggles for training phase.

        A: PM + EM (all memory systems active)
        B: A + lifelong mode (PM/EM persist across doc boundaries)
        """
        phase = phase.upper()
        if phase == "A":
            self.pm_enabled = True
            self.em_enabled = True
        elif phase == "B":
            self.pm_enabled = True
            self.em_enabled = True
            self.lifelong_mode = True
        else:
            raise ValueError(f"Unknown phase: {phase}. Expected A/B.")

        if phase != "B":
            self.lifelong_mode = False

    @classmethod
    def tier_a(cls, **overrides) -> "ModelConfig":
        """Dev tier (~101M). Matches Pythia-70M / Mamba-130M scale."""
        defaults = dict(
            D=2048, D_embed=384, B_blocks=6, C=16,
            D_pcm=64, R=4, N=512, r=32, M=256,
            ffn_depth=3, ffn_expansion=4,
            k_ret=16, C_em=32, budget_pm=16, budget_em=32,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_b(cls, **overrides) -> "ModelConfig":
        """Research tier (~406M). Matches Pythia-410M / Mamba-370M."""
        defaults = dict(
            D=3072, D_embed=512, B_blocks=12, C=16,
            D_pcm=96, R=6, N=512, r=64, M=512,
            ffn_depth=3, ffn_expansion=4,
            k_ret=32, C_em=64, budget_pm=32, budget_em=64,
            neuromod_hidden=64, content_proj_dim=16,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_c(cls, **overrides) -> "ModelConfig":
        """1B-class tier (~944M). Matches Pythia-1B / TinyLlama-1.1B / Mamba-790M."""
        defaults = dict(
            D=4096, D_embed=768, B_blocks=16, C=16,
            D_pcm=128, R=8, N=512, r=128, M=1024,
            ffn_depth=3, ffn_expansion=4,
            k_ret=64, C_em=128, budget_pm=64, budget_em=128,
            neuromod_hidden=64, content_proj_dim=16,
        )
        defaults.update(overrides)
        return cls(**defaults)
