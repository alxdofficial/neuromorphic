"""
Model configuration for the Neuromorphic LM (v5: scan-memory-scan).

Single dataclass holding all hyperparameters. Phase toggles control which
memory systems are active. Tier presets provide size configurations.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Architecture — v5 scan-memory-scan
    D: int = 512              # model width (internal)
    D_embed: int = -1         # embedding dim (defaults to D in validate())
    B: int = 4                # memory banks
    C: int = 4                # cortical columns
    D_col: int = -1           # derived: D // C (set in validate())
    N: int = 512              # segment length
    K_segments: int = 2       # TBPTT chunk = K segments
    L_scan: int = 12          # scan layers per stage (24 total)
    scan_expansion: int = 4   # E = scan_expansion * D_col per layer
    vocab_size: int = 32000   # set from tokenizer at runtime
    eot_id: int = 2           # set from tokenizer at runtime

    # Procedural Memory (bias vector per bank)
    budget_pm: float = 16.0   # PM bias norm budget per stream
    decay_pm: float = 0.999   # per-segment bias decay

    # Episodic Memory (primitive dictionary per bank)
    M: int = 256              # EM capacity (primitives) per bank
    n_trail_steps: int = 2    # trail iteration count
    S_max: float = 3.0        # max primitive strength
    budget_em: float = 32.0   # sum(em_S) budget per stream
    decay_em: float = 0.999   # per-segment strength decay

    # Neuromodulator architecture
    neuromod_hidden: int = 32

    # Predictive Coding Module (within-scan, per column)
    pcm_enabled: bool = True
    pcm_pred_weight: float = 0.01

    # Regularization
    dropout: float = 0.1
    tie_embeddings: bool = True

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
        if self.B < 1:
            raise ValueError(f"B ({self.B}) must be >= 1.")
        if self.C < 1:
            raise ValueError(f"C ({self.C}) must be >= 1.")
        if self.D % self.C != 0:
            raise ValueError(f"D ({self.D}) must be divisible by C ({self.C}).")
        # Derive D_col from D and C
        self.D_col = self.D // self.C
        if self.N < 1:
            raise ValueError(f"N ({self.N}) must be >= 1.")
        if self.K_segments < 1:
            raise ValueError(f"K_segments ({self.K_segments}) must be >= 1.")
        if self.L_scan < 1:
            raise ValueError(f"L_scan ({self.L_scan}) must be >= 1.")
        if self.scan_expansion < 1:
            raise ValueError(f"scan_expansion ({self.scan_expansion}) must be >= 1.")
        if self.M < 1:
            raise ValueError(f"M ({self.M}) must be >= 1 (EM capacity).")
        if self.n_trail_steps < 1:
            raise ValueError(f"n_trail_steps ({self.n_trail_steps}) must be >= 1.")

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
    def tier_tiny(cls, **overrides) -> "ModelConfig":
        """Test tier (~tiny). For unit tests only."""
        defaults = dict(
            D=64, D_embed=64, B=2, C=2,
            vocab_size=64, N=16, K_segments=2,
            M=8, L_scan=2, scan_expansion=2, n_trail_steps=2,
            budget_pm=4.0, budget_em=8.0,
            neuromod_hidden=8,
            pcm_enabled=True, pm_enabled=True, em_enabled=True,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_a(cls, **overrides) -> "ModelConfig":
        """Dev tier (~130M). Matches Mamba-130M scale."""
        defaults = dict(
            D=2048, D_embed=384, B=4, C=16,
            N=512, L_scan=6, scan_expansion=8,
            M=384, n_trail_steps=1,
            budget_pm=16, budget_em=32,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_b(cls, **overrides) -> "ModelConfig":
        """Research tier (~400M). Matches Mamba-370M scale."""
        defaults = dict(
            D=3072, D_embed=512, B=12, C=16,
            N=512, L_scan=16, scan_expansion=4,
            M=512, n_trail_steps=2,
            budget_pm=32, budget_em=64,
            neuromod_hidden=64,
        )
        defaults.update(overrides)
        return cls(**defaults)
