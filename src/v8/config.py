"""V8 configuration — Neural Memory Graph + Cortical Columns."""

from dataclasses import dataclass


@dataclass
class V8Config:
    # Scan Stack (Language Model)
    D: int = 2048
    D_embed: int = 768
    C: int = 16                  # cortical columns
    D_cc: int = -1               # derived: D // C = neuron dim
    L_total: int = 10            # total scan layers (single pass, no split)
    d_inner: int = 1024
    glu_output: bool = True
    vocab_size: int = 32000
    eot_id: int = 2
    tie_embeddings: bool = True
    dropout: float = 0.1

    # PCM (per-CC, independent weights)
    pcm_enabled: bool = True
    pcm_pred_weight: float = 0.1
    pcm_hidden: int = 256        # hidden dim for per-CC PCM

    # Memory Graph — diagonal scan + sparse graph message passing
    # D_mem = D_cc always (neurons match CC width)
    N_mem_neurons: int = 1024    # total neurons
    K_connections: int = 96      # sparse presynaptic connections per neuron

    # Plasticity
    plasticity_ema_decay: float = 0.99
    co_activation_ema_decay: float = 0.995  # slow EMA for co-activation matrix
    structural_plasticity_every: int = 4    # segments between prune-regrow (twice per chunk)
    plasticity_exploration_frac: float = 0.2  # fraction of regrowth that's random

    # Neuromodulator
    neuromod_hidden: int = 1024
    neuromod_layers: int = 3
    action_every: int = 256      # act every N tokens (8 segments per T=2048 chunk)
    max_action_magnitude: float = 1.0  # generous — L1 normalization bounds the effect

    # Neuromodulator RL
    neuromod_lr: float = 3e-4    # learning rate for neuromod optimizer

    # Training
    T: int = 2048                # full chunk length
    gradient_checkpointing: bool = False
    use_compile: bool = True
    lifelong_mode: bool = False

    # Regularization
    reg_weight: float = 0.1

    @property
    def D_mem(self) -> int:
        """Neuron primitive dim = CC dim. Always equal."""
        return self.D_cc if self.D_cc > 0 else self.D // self.C

    @property
    def N_neurons(self) -> int:
        """Total neurons."""
        return self.N_mem_neurons

    @property
    def actions_per_chunk(self) -> int:
        return self.T // self.action_every

    def validate(self):
        if self.D <= 0:
            raise ValueError(f"D ({self.D}) must be positive.")
        if self.D_embed == -1:
            self.D_embed = self.D
        if self.D_embed <= 0:
            raise ValueError(f"D_embed ({self.D_embed}) must be positive.")
        if self.C < 1:
            raise ValueError(f"C ({self.C}) must be >= 1.")
        if self.D % self.C != 0:
            raise ValueError(f"D ({self.D}) must be divisible by C ({self.C}).")
        self.D_cc = self.D // self.C
        if self.L_total < 1:
            raise ValueError(f"L_total ({self.L_total}) must be >= 1.")
        if self.d_inner < 1:
            raise ValueError(f"d_inner ({self.d_inner}) must be >= 1.")
        if self.N_mem_neurons < 1:
            raise ValueError(f"N_mem_neurons must be >= 1.")
        if self.K_connections < 1:
            raise ValueError(f"K_connections must be >= 1.")
        if self.K_connections > self.N_mem_neurons:
            raise ValueError(
                f"K_connections ({self.K_connections}) must be <= "
                f"N_mem_neurons ({self.N_mem_neurons})."
            )
        if self.T < 1:
            raise ValueError(f"T ({self.T}) must be >= 1.")
        if self.action_every < 1:
            raise ValueError(f"action_every ({self.action_every}) must be >= 1.")
        if self.T % self.action_every != 0:
            raise ValueError(
                f"T ({self.T}) must be divisible by action_every ({self.action_every})."
            )

    @classmethod
    def tier_a(cls, **overrides) -> "V8Config":
        defaults = dict(
            D=2048, D_embed=768, C=16, L_total=7,
            d_inner=1024, glu_output=True, T=2048,
            # Memory: 1024 neurons, 96 presynaptic connections
            N_mem_neurons=1024, K_connections=96,
            pcm_hidden=256,
            neuromod_hidden=2048, neuromod_layers=3,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_tiny(cls, **overrides) -> "V8Config":
        """Tiny config for unit tests."""
        defaults = dict(
            D=64, D_embed=64, C=4, L_total=4,
            d_inner=64, glu_output=False, vocab_size=64, T=32,
            N_mem_neurons=16, K_connections=6,
            pcm_hidden=32,
            neuromod_hidden=32, neuromod_layers=2,
            action_every=8,
        )
        defaults.update(overrides)
        return cls(**defaults)
