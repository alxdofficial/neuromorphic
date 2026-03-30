"""v10-gnn configuration -- Shared-Weight GNN Memory Graph."""

from dataclasses import dataclass


@dataclass
class V10Config:
    # Scan Stack (Lower Sensory Cortex)
    D_scan: int = 2048               # LM hidden dim, also word dim
    D_embed: int = 768               # embedding dim
    D_dec: int = 1024                # decoder hidden dim
    L_scan: int = 2                  # lower scan layers
    L_dec: int = 3                   # decoder layers
    d_inner_scan: int = 512          # scan layer inner dim
    d_ff_dec: int = 4096             # decoder FFN inner dim
    n_heads_dec: int = 8             # decoder attention heads
    W_sliding: int = 16              # sliding window size for cross-attention

    # Memory Graph
    D_neuron: int = 32               # per-neuron state dimension
    D_id: int = 32                   # identity embedding dim
    N_neurons: int = 4096            # total neurons
    K_connections: int = 32          # sparse presynaptic connections per neuron
    H_state: int = 4096             # shared state MLP hidden
    H_msg: int = 2048               # shared message MLP hidden
    H_mod: int = 2048               # shared modulator MLP hidden

    # Segment / training
    T: int = 128                     # tokens per chunk = segment length
    vocab_size: int = 32000
    eot_id: int = 2
    tie_embeddings: bool = True
    dropout: float = 0.1

    # PCM
    pcm_enabled: bool = True
    pcm_hidden: int = 256
    pcm_pred_weight: float = 0.1

    # Structural plasticity
    structural_plasticity: bool = True
    plasticity_pct: float = 0.02
    plasticity_exploration_frac: float = 0.2
    co_activation_ema_decay: float = 0.995

    # Memory learning rate
    mem_lr_scale: float = 0.3

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def C(self) -> int:
        """Cortical columns (number of D_neuron-sized slices in D_scan)."""
        return self.D_scan // (self.D_scan // 16)

    @property
    def D_cc(self) -> int:
        """Per-column dimension."""
        return self.D_scan // self.C

    @property
    def neurons_per_word(self) -> int:
        """Neurons grouped into one word = D_scan // D_neuron."""
        return self.D_scan // self.D_neuron

    @property
    def num_words(self) -> int:
        """Number of words = N_neurons // neurons_per_word."""
        return self.N_neurons // self.neurons_per_word

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self):
        if self.D_scan <= 0:
            raise ValueError(f"D_scan ({self.D_scan}) must be positive.")
        if self.D_embed <= 0:
            raise ValueError(f"D_embed ({self.D_embed}) must be positive.")
        if self.D_dec <= 0:
            raise ValueError(f"D_dec ({self.D_dec}) must be positive.")
        if self.D_neuron <= 0:
            raise ValueError(f"D_neuron ({self.D_neuron}) must be positive.")
        if self.D_id <= 0:
            raise ValueError(f"D_id ({self.D_id}) must be positive.")
        if self.D_scan % self.D_neuron != 0:
            raise ValueError(
                f"D_scan ({self.D_scan}) must be divisible by "
                f"D_neuron ({self.D_neuron}).")
        if self.N_neurons % self.neurons_per_word != 0:
            raise ValueError(
                f"N_neurons ({self.N_neurons}) must be divisible by "
                f"neurons_per_word ({self.neurons_per_word}).")
        if self.N_neurons < 1:
            raise ValueError(f"N_neurons ({self.N_neurons}) must be >= 1.")
        if self.K_connections < 1:
            raise ValueError(
                f"K_connections ({self.K_connections}) must be >= 1.")
        if self.K_connections >= self.N_neurons:
            raise ValueError(
                f"K_connections ({self.K_connections}) must be < "
                f"N_neurons ({self.N_neurons}).")
        if self.L_scan < 1:
            raise ValueError(f"L_scan ({self.L_scan}) must be >= 1.")
        if self.L_dec < 1:
            raise ValueError(f"L_dec ({self.L_dec}) must be >= 1.")
        if self.T < 1:
            raise ValueError(f"T ({self.T}) must be >= 1.")
        if self.d_inner_scan < 1:
            raise ValueError(
                f"d_inner_scan ({self.d_inner_scan}) must be >= 1.")
        if self.d_ff_dec < 1:
            raise ValueError(f"d_ff_dec ({self.d_ff_dec}) must be >= 1.")
        if self.n_heads_dec < 1:
            raise ValueError(
                f"n_heads_dec ({self.n_heads_dec}) must be >= 1.")
        if self.D_dec % self.n_heads_dec != 0:
            raise ValueError(
                f"D_dec ({self.D_dec}) must be divisible by "
                f"n_heads_dec ({self.n_heads_dec}).")
        if self.W_sliding < 1:
            raise ValueError(
                f"W_sliding ({self.W_sliding}) must be >= 1.")

    # ------------------------------------------------------------------
    # Preset tiers
    # ------------------------------------------------------------------

    @classmethod
    def tier_a(cls, **overrides) -> "V10Config":
        """Full-size config (~110M params)."""
        defaults = dict(
            D_scan=2048, D_embed=768, D_dec=1024,
            D_neuron=32, D_id=32, N_neurons=4096, K_connections=32,
            L_scan=2, L_dec=3, d_inner_scan=512,
            H_state=4096, H_msg=2048, H_mod=2048,
            d_ff_dec=4096, n_heads_dec=8, W_sliding=16,
            T=128, pcm_hidden=256,
        )
        defaults.update(overrides)
        cfg = cls(**defaults)
        cfg.validate()
        return cfg

    @classmethod
    def tier_tiny(cls, **overrides) -> "V10Config":
        """Tiny config for unit tests."""
        defaults = dict(
            D_scan=64, D_embed=64, D_dec=64,
            D_neuron=8, D_id=8, N_neurons=64, K_connections=8,
            L_scan=1, L_dec=1, d_inner_scan=32,
            H_state=32, H_msg=16, H_mod=16,
            d_ff_dec=64, n_heads_dec=4, W_sliding=4,
            T=4, vocab_size=64, pcm_hidden=16,
            structural_plasticity=False, dropout=0.0,
        )
        defaults.update(overrides)
        cfg = cls(**defaults)
        cfg.validate()
        return cfg
