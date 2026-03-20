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

    # Memory Graph
    # D_mem = D_cc always (neurons match CC width)
    N_blocks: int = 8            # memory blocks (independent from C)
    M_per_block: int = 1024      # neurons per block
    K_intra: int = 128           # random connections within block
    K_inter: int = 32            # random connections to other blocks
    mem_temperature: float = 1.0 # global default routing temperature
    mem_sparsity: float = 0.5    # fraction of connections zeroed in routing

    # Neuromodulator
    neuromod_hidden: int = 1024
    neuromod_layers: int = 3
    action_every: int = 8        # act every N tokens
    max_action_magnitude: float = 0.1

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
        """Total neurons across all blocks."""
        return self.N_blocks * self.M_per_block

    @property
    def max_connections(self) -> int:
        """Max connections per neuron (intra-block + inter-block)."""
        return self.K_intra + self.K_inter

    @property
    def CCs_per_block(self) -> int:
        """Number of cortical columns attached to each block."""
        return self.C // self.N_blocks

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
        if self.N_blocks < 1:
            raise ValueError(f"N_blocks ({self.N_blocks}) must be >= 1.")
        if self.C % self.N_blocks != 0:
            raise ValueError(
                f"C ({self.C}) must be divisible by N_blocks ({self.N_blocks})."
            )
        if self.M_per_block < 1:
            raise ValueError(f"M_per_block ({self.M_per_block}) must be >= 1.")
        if self.K_intra < 1:
            raise ValueError(f"K_intra ({self.K_intra}) must be >= 1.")
        if self.K_intra > self.M_per_block:
            raise ValueError(
                f"K_intra ({self.K_intra}) must be <= M_per_block ({self.M_per_block})."
            )
        if self.T < 1:
            raise ValueError(f"T ({self.T}) must be >= 1.")
        if self.action_every < 1:
            raise ValueError(f"action_every ({self.action_every}) must be >= 1.")

    @classmethod
    def tier_a(cls, **overrides) -> "V8Config":
        defaults = dict(
            D=2048, D_embed=768, C=16, L_total=7,
            d_inner=1024, glu_output=True, T=2048,
            # Memory graph: 8 blocks × 1024 neurons, D_mem=D_cc=128
            N_blocks=8, M_per_block=1024,
            K_intra=128, K_inter=32,
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
            N_blocks=2, M_per_block=8,
            K_intra=4, K_inter=2,
            pcm_hidden=32,
            neuromod_hidden=32, neuromod_layers=2,
            action_every=4,
        )
        defaults.update(overrides)
        return cls(**defaults)
