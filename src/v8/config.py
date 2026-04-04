"""Configuration — Sequential differentiable memory graph + split-scan LM."""

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
    d_inner: int = 580
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
    N_mem_neurons: int = 512     # total neurons
    D_neuron: int = 256          # per-neuron state dimension
    K_connections: int = 32      # sparse presynaptic connections per neuron

    # Memory hidden sizes
    neuromod_hidden: int = 112   # per-neuron modulator hidden dim
    state_mlp_hidden: int = 24   # shared state MLP hidden dim
    msg_mlp_hidden: int = 24     # shared message MLP hidden dim

    # Structural plasticity
    structural_plasticity: bool = True
    plasticity_pct: float = 0.02  # fraction of total connections to prune/grow per chunk
    plasticity_exploration_frac: float = 0.2  # fraction of regrowth that's random
    co_activation_ema_decay: float = 0.995    # slow EMA for phi matrix

    # Segment / training
    T: int = 128                 # tokens per chunk = segment length
    mem_lr_scale: float = 0.3    # memory param LR = base_LR * mem_lr_scale

    @property
    def action_every(self) -> int:
        """Segment length = T (one segment per chunk)."""
        return self.T

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
        return 1

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

    @classmethod
    def tier_a(cls, **overrides) -> "V8Config":
        defaults = dict(
            D=2048, D_embed=768, C=16, L_total=4, scan_split_at=2,
            d_inner=580, glu_output=True, T=128,
            N_mem_neurons=512, D_neuron=256, K_connections=32,
            pcm_hidden=256,
            neuromod_hidden=112, state_mlp_hidden=24, msg_mlp_hidden=24,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_tiny(cls, **overrides) -> "V8Config":
        """Tiny config for unit tests."""
        defaults = dict(
            D=64, D_embed=64, C=4, L_total=4, scan_split_at=2,
            d_inner=64, glu_output=False, vocab_size=64, T=8,
            N_mem_neurons=32, D_neuron=16, K_connections=8,
            pcm_hidden=32,
            neuromod_hidden=16, state_mlp_hidden=16, msg_mlp_hidden=16,
            structural_plasticity=False,
        )
        defaults.update(overrides)
        return cls(**defaults)
