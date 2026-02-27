"""
Model configuration for the Neuromorphic LM (v4: iterative refinement).

Single dataclass holding all hyperparameters. Phase toggles control which
memory systems are active. Tier presets provide size configurations.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Architecture — v4 iterative refinement
    D: int = 512              # model width (embedding dim)
    R: int = 4                # iterative refinement passes
    B_blocks: int = 4         # memory blocks
    C: int = 4                # columns per block
    D_col: int = 128          # column width
    D_mem: int = 256          # PM/EM dimension (decoupled from D_col)
    D_pcm: int = 64           # PCM encoding dim
    N: int = 128              # segment length
    K_segments: int = 2       # TBPTT chunk = K segments
    lambda_mix: float = 0.5   # damped pass mixing init
    ffn_expansion: int = 4    # column FFN expansion
    vocab_size: int = 32000   # set from tokenizer at runtime
    eot_id: int = 2           # set from tokenizer at runtime

    # Procedural Memory (single instance, batched as BS*B)
    r: int = 8                # PM slots
    rho: float = 0.95         # eligibility decay
    a_max: float = 3.0        # max strength per slot
    budget_pm: float = 4.0    # sum(pm_a) budget per stream
    decay_pm: float = 0.999   # per-pass strength decay
    tau_pm: float = 1.0       # softmax temperature (default for PM slot selection)
    tau_pm_floor: float = 0.05
    tau_pm_ceil: float = 5.0
    weakness_weight_pm: float = 0.5
    tau_route_pm: float = 1.0
    surprise_scale: float = 5.0
    g_pm_default: float = 0.5

    # Episodic Memory (single instance, batched as BS*B)
    M: int = 64               # EM capacity per bank
    k_ret: int = 4            # retrieval count
    C_em: int = 8             # candidates per pass (top-C_em across N*C)
    tau_em: float = 1.0
    weakness_weight_em: float = 0.5
    S_max: float = 3.0
    budget_em: float = 8.0
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

    # Training
    use_compile: bool = False
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
        if self.R < 1:
            raise ValueError(f"R ({self.R}) must be >= 1.")
        if self.B_blocks < 1:
            raise ValueError(f"B_blocks ({self.B_blocks}) must be >= 1.")
        if self.C < 1:
            raise ValueError(f"C ({self.C}) must be >= 1.")
        if self.D_col <= 0:
            raise ValueError(f"D_col ({self.D_col}) must be positive.")
        if self.D_mem <= 0:
            raise ValueError(f"D_mem ({self.D_mem}) must be positive.")
        if self.D_pcm <= 0 and self.pcm_enabled:
            raise ValueError(f"D_pcm ({self.D_pcm}) must be positive when pcm_enabled.")
        if self.N < 1:
            raise ValueError(f"N ({self.N}) must be >= 1.")
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
        """Dev tier (~100M). Matches Pythia-70M / Mamba-130M scale."""
        defaults = dict(
            D=768, B_blocks=6, C=4, D_col=384, D_mem=384,
            D_pcm=64, R=4, N=128, r=8, M=64,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_b(cls, **overrides) -> "ModelConfig":
        """Research tier (~400M). Matches Pythia-410M / Mamba-370M."""
        defaults = dict(
            D=1536, B_blocks=8, C=8, D_col=448, D_mem=640,
            D_pcm=96, R=4, N=128, r=16, M=128,
            k_ret=8, C_em=16,
            neuromod_hidden=64, content_proj_dim=16,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_c(cls, **overrides) -> "ModelConfig":
        """1B-class tier (~1.05B). Matches Pythia-1B / TinyLlama-1.1B / Mamba-790M."""
        defaults = dict(
            D=2048, B_blocks=12, C=8, D_col=640, D_mem=768,
            D_pcm=128, R=6, N=128, r=16, M=256,
            k_ret=8, C_em=16,
            neuromod_hidden=64, content_proj_dim=16,
        )
        defaults.update(overrides)
        return cls(**defaults)
