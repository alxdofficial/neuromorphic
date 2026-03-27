"""V8/v9 configuration — Neural Memory Graph + Cortical Columns.

v9.1: Lightweight hippocampal neurons (D_mem=16, N=8192) trained by backprop.
Per-neuron small MLP (2.5K params) with message + modulator heads.
Hebbian plasticity with learned modulation for inference-time adaptation.
"""

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
    pcm_hidden: int = 256        # hidden dim for per-CC PCM

    # Memory Graph — lightweight hippocampal neurons
    D_mem: int = 16              # neuron state/message dim (compact memory code)
    N_mem_neurons: int = 8192    # total neurons (many simple > few complex)
    K_connections: int = 32      # sparse presynaptic connections per neuron
    neuron_hidden: int = 48      # hidden dim for per-neuron MLP
    hebbian_lr_init: float = 0.01  # initial Hebbian learning rate

    # Segment / training
    action_every: int = 128      # segment length (tokens per memory graph forward)
    memory_update_stride: int = 1 # neuron dynamics step every N tokens
    mem_lr_scale: float = 0.3    # memory param LR = this × LM LR

    # Training
    T: int = 2048                # full chunk length
    gradient_checkpointing: bool = False
    use_compile: bool = True

    @property
    def N_neurons(self) -> int:
        return self.N_mem_neurons

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
        if self.D_mem < 1:
            raise ValueError(f"D_mem ({self.D_mem}) must be >= 1.")

    @classmethod
    def tier_a(cls, **overrides) -> "V8Config":
        defaults = dict(
            D=2048, D_embed=768, C=16, L_total=5, scan_split_at=3,
            d_inner=768, glu_output=True, T=2048,
            # Memory: 8192 lightweight neurons, D_mem=16
            D_mem=16, N_mem_neurons=8192, K_connections=32,
            neuron_hidden=48,
            pcm_hidden=256,
            action_every=128,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_b(cls, **overrides) -> "V8Config":
        """~300M params. Wider + deeper LM, more neurons."""
        defaults = dict(
            D=3072, D_embed=1024, C=24, L_total=10, scan_split_at=5,
            d_inner=1536, glu_output=True, T=2048,
            D_mem=24, N_mem_neurons=16384, K_connections=32,
            neuron_hidden=64,
            pcm_hidden=256,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_c(cls, **overrides) -> "V8Config":
        """~1B params. Large LM, large graph."""
        defaults = dict(
            D=4096, D_embed=1024, C=16, L_total=24, scan_split_at=12,
            d_inner=2048, glu_output=True, T=2048,
            D_mem=32, N_mem_neurons=32768, K_connections=48,
            neuron_hidden=96,
            pcm_hidden=512,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_tiny(cls, **overrides) -> "V8Config":
        """Tiny config for unit tests."""
        defaults = dict(
            D=64, D_embed=64, C=4, L_total=4, scan_split_at=2,
            d_inner=64, glu_output=False, vocab_size=64, T=32,
            D_mem=8, N_mem_neurons=32, K_connections=6,
            neuron_hidden=16,
            pcm_hidden=32,
            action_every=8,
            memory_update_stride=2,
        )
        defaults.update(overrides)
        return cls(**defaults)
