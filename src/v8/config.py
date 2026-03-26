"""V8 configuration — Neural Memory Graph + Cortical Columns."""

from dataclasses import dataclass


@dataclass
class V8Config:
    # Scan Stack (Language Model)
    D: int = 2048
    D_embed: int = 768
    C: int = 16                  # cortical columns
    D_cc: int = -1               # derived: D // C = neuron dim
    L_total: int = 10            # total scan layers
    scan_split_at: int = 4       # layers 0..split-1 = lower, split..L-1 = upper
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
    dendrite_branch_size: int = 12  # connections per dendritic branch (0 = flat, no tree)

    # Plasticity
    plasticity_ema_decay: float = 0.99
    co_activation_ema_decay: float = 0.995  # slow EMA for co-activation matrix
    structural_plasticity_every: int = 4    # segments between prune-regrow (twice per chunk)
    plasticity_exploration_frac: float = 0.2  # fraction of regrowth that's random

    # Neuromodulator — gates Hebbian eligibility traces + controls decay
    neuromod_hidden: int = 512
    neuromod_layers: int = 2
    action_every: int = 128      # act every N tokens (16 segments per T=2048 chunk)
    memory_update_stride: int = 1 # neuron dynamics step every N tokens (>1 for larger N)
    hebbian_lr: float = 0.01     # local learning rate for gated Hebbian updates
    trace_decay: float = 0.95    # EMA decay for eligibility traces (~14 segment half-life)

    # Neuromodulator RL
    neuromod_lr: float = 3e-4    # learning rate for neuromod optimizer
    neuromod_logstd_init: float = -0.5  # initial log_std for policy (exp(-0.5)≈0.6)
    rl_collect_chunks: int = 4   # chunks to collect before RL update (longer horizon)
    rl_entropy_coef: float = 0.01  # entropy bonus coefficient
    rl_counterfactual_k: int = 96   # neurons to evaluate per counterfactual trajectory
    rl_counterfactual_n: int = 8    # number of sampled trajectories for GRPO scoring

    # Training
    T: int = 2048                # full chunk length
    gradient_checkpointing: bool = False
    use_compile: bool = True

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
        if self.scan_split_at < 1 or self.scan_split_at >= self.L_total:
            raise ValueError(
                f"scan_split_at ({self.scan_split_at}) must be in [1, L_total-1={self.L_total-1}]."
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
            neuromod_hidden=512, neuromod_layers=2,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_b(cls, **overrides) -> "V8Config":
        """~300M params. Wider + deeper LM, 2K neurons.
        C=24 so D_mem=128 (power of 2 for Triton kernel)."""
        defaults = dict(
            D=3072, D_embed=1024, C=24, L_total=12, scan_split_at=6,
            d_inner=1536, glu_output=True, T=2048,
            N_mem_neurons=2048, K_connections=96,
            pcm_hidden=256,
            neuromod_hidden=512, neuromod_layers=2,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_c(cls, **overrides) -> "V8Config":
        """~1B params. Large LM, 4K neurons.
        C=16 so D_mem=256 (power of 2 for Triton kernel)."""
        defaults = dict(
            D=4096, D_embed=1024, C=16, L_total=28, scan_split_at=14,
            d_inner=2048, glu_output=True, T=2048,
            N_mem_neurons=4096, K_connections=128,
            pcm_hidden=512,
            neuromod_hidden=512, neuromod_layers=2,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_tiny(cls, **overrides) -> "V8Config":
        """Tiny config for unit tests."""
        defaults = dict(
            D=64, D_embed=64, C=4, L_total=4, scan_split_at=2,
            d_inner=64, glu_output=False, vocab_size=64, T=32,
            N_mem_neurons=16, K_connections=6,
            dendrite_branch_size=3,
            pcm_hidden=32,
            neuromod_hidden=32, neuromod_layers=2,
            action_every=8,
            memory_update_stride=2,
        )
        defaults.update(overrides)
        return cls(**defaults)
