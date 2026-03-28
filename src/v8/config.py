"""v9-backprop configuration — Differentiable Memory Graph."""

from dataclasses import dataclass


@dataclass
class V8Config:
    # Scan Stack (Language Model)
    D: int = 2048
    D_embed: int = 768
    C: int = 16                  # cortical columns
    D_cc: int = -1               # derived: D // C
    L_total: int = 5             # total scan layers
    scan_split_at: int = 3       # layers 0..split-1 = lower, split..L-1 = upper
    d_inner: int = 1024
    glu_output: bool = True
    vocab_size: int = 32000
    eot_id: int = 2
    tie_embeddings: bool = True
    dropout: float = 0.1

    # PCM (per-CC, independent weights)
    pcm_enabled: bool = True
    pcm_pred_weight: float = 0.1
    pcm_hidden: int = 256

    # Memory Graph — differentiable, trained by backprop
    N_mem_neurons: int = 4096    # total neurons
    D_neuron: int = 32           # per-neuron state dimension
    K_connections: int = 128     # sparse presynaptic connections per neuron
    dendrite_branch_size: int = 16

    # Per-neuron MLPs
    neuromod_hidden: int = 16    # hidden dim for segment-boundary modulator
    state_mlp_hidden: int = 24   # hidden dim for per-step state update MLP
    msg_mlp_hidden: int = 24     # hidden dim for per-step message MLP

    # Structural plasticity
    structural_plasticity: bool = True
    plasticity_n_swap: int = 8   # connections swapped per neuron per rewire

    # Segment / training
    action_every: int = 128      # segment length
    memory_update_stride: int = 1 # neuron dynamics every token (1:1 with sequence)
    mem_lr_scale: float = 0.3    # memory param LR = base_LR * mem_lr_scale

    # Training
    T: int = 2048
    gradient_checkpointing: bool = False
    use_compile: bool = True

    @property
    def D_mem(self) -> int:
        return self.D_neuron

    @property
    def N_neurons(self) -> int:
        return self.N_mem_neurons

    @property
    def C_mem(self) -> int:
        """Number of CC slices = D // D_neuron."""
        return self.D // self.D_neuron

    @property
    def N_per_slice(self) -> int:
        """Neurons per CC slice for inject/readout."""
        return self.N_neurons // self.C_mem

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
        if self.D % self.D_neuron != 0:
            raise ValueError(
                f"D ({self.D}) must be divisible by D_neuron ({self.D_neuron}).")
        if self.N_mem_neurons % (self.D // self.D_neuron) != 0:
            raise ValueError(
                f"N_mem_neurons ({self.N_mem_neurons}) must be divisible by "
                f"D//D_neuron ({self.D // self.D_neuron}).")
        if self.L_total < 1:
            raise ValueError(f"L_total ({self.L_total}) must be >= 1.")
        if self.d_inner < 1:
            raise ValueError(f"d_inner ({self.d_inner}) must be >= 1.")
        if self.N_mem_neurons < 1:
            raise ValueError(f"N_mem_neurons must be >= 1.")
        if self.K_connections < 1:
            raise ValueError(f"K_connections must be >= 1.")
        if self.K_connections >= self.N_mem_neurons:
            raise ValueError(
                f"K_connections ({self.K_connections}) must be < "
                f"N_mem_neurons ({self.N_mem_neurons}) "
                f"(each neuron needs K distinct non-self neighbors).")
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
        if self.memory_update_stride < 1:
            raise ValueError(f"memory_update_stride must be >= 1.")
        if self.action_every % self.memory_update_stride != 0:
            raise ValueError(
                f"action_every ({self.action_every}) must be divisible by "
                f"memory_update_stride ({self.memory_update_stride}).")
        if self.dendrite_branch_size > self.K_connections:
            raise ValueError(
                f"dendrite_branch_size ({self.dendrite_branch_size}) must be <= "
                f"K_connections ({self.K_connections}).")

    @classmethod
    def tier_a(cls, **overrides) -> "V8Config":
        defaults = dict(
            D=2048, D_embed=768, C=16, L_total=4, scan_split_at=2,
            d_inner=580, glu_output=True, T=128,
            N_mem_neurons=512, D_neuron=256, K_connections=32,
            dendrite_branch_size=16,
            pcm_hidden=256,
            neuromod_hidden=80, state_mlp_hidden=24, msg_mlp_hidden=24,
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
            d_inner=64, glu_output=False, vocab_size=64, T=32,
            N_mem_neurons=32, D_neuron=16, K_connections=8,
            dendrite_branch_size=4,
            pcm_hidden=32,
            neuromod_hidden=16, state_mlp_hidden=16, msg_mlp_hidden=16,
            action_every=8,
            memory_update_stride=1,
            structural_plasticity=False,
        )
        defaults.update(overrides)
        return cls(**defaults)
