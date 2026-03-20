"""V8 configuration — Neural Memory Graph + Cortical Columns."""

from dataclasses import dataclass


@dataclass
class V8Config:
    # Scan Stack (Language Model)
    D: int = 2048
    D_embed: int = 768
    C: int = 16                  # cortical columns (= memory blocks)
    D_cc: int = -1               # derived: D // C = neuron dim
    L_total: int = 10            # total scan layers
    L_mem: int = 5               # memory injection point
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
    # D_mem = D_cc always (neurons match CC width, no projections needed)
    N_neurons: int = 4096        # total neurons (C * M_per_block)
    M_per_block: int = 256       # neurons per block
    inter_block_k: int = 32      # sparse connections per neuron to other blocks
    mem_temperature: float = 1.0 # routing softmax temperature
    mem_sparsity: float = 0.5    # fraction of connections zeroed in routing
    mem_mod_hidden: int = 512    # W_mod MLP hidden dim (inside memory graph)

    # Neuromodulator
    neuromod_hidden: int = 1024
    neuromod_layers: int = 3
    action_every: int = 8        # act every N tokens
    max_action_magnitude: float = 0.1

    # PPO
    ppo_gamma: float = 0.99
    ppo_lambda: float = 0.95
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    ppo_minibatch: int = 512
    ppo_lr: float = 3e-4
    ppo_ent_coef: float = 0.003
    ppo_vf_coef: float = 0.5

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
    def max_connections(self) -> int:
        """Max connections per neuron (intra-block + inter-block)."""
        return self.M_per_block + self.inter_block_k

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
        if self.L_mem < 0 or self.L_mem >= self.L_total:
            raise ValueError(f"L_mem ({self.L_mem}) must be in [0, L_total).")
        if self.d_inner < 1:
            raise ValueError(f"d_inner ({self.d_inner}) must be >= 1.")
        if self.N_neurons != self.C * self.M_per_block:
            raise ValueError(
                f"N_neurons ({self.N_neurons}) must equal C * M_per_block "
                f"({self.C} * {self.M_per_block} = {self.C * self.M_per_block})."
            )
        if self.T < 1:
            raise ValueError(f"T ({self.T}) must be >= 1.")
        if self.action_every < 1:
            raise ValueError(f"action_every ({self.action_every}) must be >= 1.")

    @classmethod
    def tier_a(cls, **overrides) -> "V8Config":
        defaults = dict(
            D=2048, D_embed=768, C=16, L_total=8, L_mem=4,
            d_inner=1024, glu_output=True, T=2048,
            # Memory graph: 4096 neurons, D_mem=D_cc=128 (derived)
            N_neurons=4096, M_per_block=256,
            inter_block_k=32, mem_mod_hidden=512,
            pcm_hidden=256,
            neuromod_hidden=1024, neuromod_layers=3,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_tiny(cls, **overrides) -> "V8Config":
        """Tiny config for unit tests."""
        defaults = dict(
            D=64, D_embed=64, C=4, L_total=4, L_mem=2,
            d_inner=64, glu_output=False, vocab_size=64, T=32,
            N_neurons=32, M_per_block=8,
            inter_block_k=4, mem_mod_hidden=32,
            pcm_hidden=32,
            neuromod_hidden=32, neuromod_layers=2,
            action_every=4, ppo_minibatch=16,
        )
        defaults.update(overrides)
        return cls(**defaults)
