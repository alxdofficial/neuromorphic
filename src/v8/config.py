"""v10 configuration — Scalar Neuron Memory Graph."""

from dataclasses import dataclass


@dataclass
class V8Config:
    # Scan Stack (Language Model)
    D: int = 2048
    D_embed: int = 768
    C: int = 16                  # cortical columns
    D_cc: int = -1               # derived: D // C
    L_total: int = 4             # total scan layers
    scan_split_at: int = 2       # layers 0..split-1 = lower, split..L-1 = upper
    d_inner: int = 512
    glu_output: bool = True
    vocab_size: int = 32000
    eot_id: int = 2
    tie_embeddings: bool = True
    dropout: float = 0.1

    # PCM (per-CC, independent weights)
    pcm_enabled: bool = True
    pcm_pred_weight: float = 0.1
    pcm_hidden: int = 256

    # Memory Graph — scalar neurons
    N_mem_neurons: int = 524288   # total neurons (512 groups × 1024)
    K_connections: int = 128      # sparse connections per neuron
    n_groups: int = 512           # number of neuromodulator groups
    group_size: int = 1024        # neurons per group
    min_intra_connections: int = 4  # min connections to groupmates

    # Neuromodulator (per-group MLP, runs once per segment)
    neuromod_hidden: int = 32

    # Structural plasticity
    structural_plasticity: bool = True
    plasticity_n_swap: int = 4

    # Segment / training
    action_every: int = 128      # segment length
    memory_update_stride: int = 1
    mem_lr_scale: float = 0.3

    # Training
    T: int = 128
    gradient_checkpointing: bool = False
    use_compile: bool = True

    @property
    def D_neuron(self) -> int:
        return 1

    @property
    def N_neurons(self) -> int:
        return self.N_mem_neurons

    @property
    def replicas_per_dim(self) -> int:
        """How many neurons map to each LM hidden dimension."""
        return self.N_neurons // self.D

    @property
    def segments_per_chunk(self) -> int:
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
        if self.N_mem_neurons % self.D != 0:
            raise ValueError(
                f"N_mem_neurons ({self.N_mem_neurons}) must be divisible by "
                f"D ({self.D}) for inject/readout replication.")
        if self.N_mem_neurons != self.n_groups * self.group_size:
            raise ValueError(
                f"N_mem_neurons ({self.N_mem_neurons}) must equal "
                f"n_groups ({self.n_groups}) × group_size ({self.group_size}).")
        if self.L_total < 1:
            raise ValueError(f"L_total ({self.L_total}) must be >= 1.")
        if self.d_inner < 1:
            raise ValueError(f"d_inner ({self.d_inner}) must be >= 1.")
        if self.K_connections < 1:
            raise ValueError(f"K_connections must be >= 1.")
        if self.K_connections >= self.N_mem_neurons:
            raise ValueError(
                f"K_connections ({self.K_connections}) must be < "
                f"N_mem_neurons ({self.N_mem_neurons}).")
        if self.scan_split_at < 1 or self.scan_split_at >= self.L_total:
            raise ValueError(
                f"scan_split_at ({self.scan_split_at}) must be in "
                f"[1, L_total-1={self.L_total-1}].")
        if self.T < 1:
            raise ValueError(f"T ({self.T}) must be >= 1.")
        if self.action_every < 1:
            raise ValueError(f"action_every ({self.action_every}) must be >= 1.")
        if self.T % self.action_every != 0:
            raise ValueError(
                f"T ({self.T}) must be divisible by action_every ({self.action_every}).")
        if self.min_intra_connections > self.K_connections:
            raise ValueError(
                f"min_intra_connections ({self.min_intra_connections}) must be <= "
                f"K_connections ({self.K_connections}).")

    @classmethod
    def tier_a(cls, **overrides) -> "V8Config":
        defaults = dict(
            D=2048, D_embed=768, C=16, L_total=4, scan_split_at=2,
            d_inner=512, glu_output=True, T=128,
            N_mem_neurons=524288, K_connections=128,
            n_groups=512, group_size=1024,
            min_intra_connections=4,
            pcm_hidden=256,
            neuromod_hidden=32,
            action_every=128,
            memory_update_stride=1,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_tiny(cls, **overrides) -> "V8Config":
        """Tiny config for unit tests."""
        defaults = dict(
            D=64, D_embed=64, C=4, L_total=4, scan_split_at=2,
            d_inner=64, glu_output=False, vocab_size=64, T=8,
            N_mem_neurons=256, K_connections=8,
            n_groups=4, group_size=64,
            min_intra_connections=2,
            pcm_hidden=32,
            neuromod_hidden=8,
            action_every=8,
            memory_update_stride=1,
            structural_plasticity=False,
        )
        defaults.update(overrides)
        return cls(**defaults)
