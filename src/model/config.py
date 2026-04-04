"""Unified configuration for LM + Memory Graph."""

from dataclasses import dataclass


@dataclass
class Config:
    # === LM ===
    D: int = 2048
    D_embed: int = 768
    L_total: int = 4
    scan_split_at: int = 2
    d_inner: int = 580
    glu_output: bool = True
    vocab_size: int = 32000
    eot_id: int = 2
    tie_embeddings: bool = True
    dropout: float = 0.1

    # === PCM ===
    pcm_enabled: bool = True
    pcm_pred_weight: float = 0.1
    pcm_hidden: int = 256
    C: int = 16  # cortical columns

    # === Memory Graph ===
    N: int = 8096  # total neurons
    D_n: int = 32  # neuron hidden dim
    K: int = 64  # connections per neuron
    alpha: int = 4  # port multiplier
    neuromod_hidden: int = 32  # modulator MLP hidden (= D_n)
    state_mlp_hidden: int = 128
    msg_mlp_hidden: int = 128

    # === Structural Plasticity ===
    structural_plasticity: bool = True
    plasticity_pct: float = 0.02
    plasticity_exploration_frac: float = 0.2
    plasticity_interval: int = 1024  # tokens between rewiring
    hebbian_ema_decay: float = 0.995

    # === Training ===
    T: int = 128  # tokens per segment
    mem_lr_scale: float = 0.3

    # === Derived (set by validate()) ===
    D_cc: int = -1
    C_mem: int = -1
    N_port: int = -1
    N_internal: int = -1

    def validate(self):
        assert self.D > 0
        assert self.D % self.C == 0, f"D ({self.D}) must be divisible by C ({self.C})"
        self.D_cc = self.D // self.C
        assert self.D % self.D_n == 0, f"D ({self.D}) must be divisible by D_n ({self.D_n})"
        self.C_mem = self.D // self.D_n
        self.N_port = self.C_mem * self.alpha
        assert 2 * self.N_port <= self.N, (
            f"Need 2*N_port={2 * self.N_port} <= N={self.N}")
        self.N_internal = self.N - 2 * self.N_port
        assert self.K < self.N, f"K ({self.K}) must be < N ({self.N})"
        assert self.K >= 1
        assert self.scan_split_at >= 1
        assert self.scan_split_at < self.L_total
        assert self.T >= 1
        if self.D_embed == -1:
            self.D_embed = self.D

    @classmethod
    def tier_a(cls, **kw) -> "Config":
        c = cls(**kw)
        c.validate()
        return c

    @classmethod
    def tier_tiny(cls, **kw) -> "Config":
        """Small config for unit tests."""
        defaults = dict(
            D=64, D_embed=64, C=4, L_total=4, scan_split_at=2,
            d_inner=64, glu_output=False, vocab_size=256, T=8,
            N=128, D_n=8, K=16, alpha=2,
            neuromod_hidden=8, state_mlp_hidden=32, msg_mlp_hidden=32,
            pcm_hidden=32, structural_plasticity=False,
        )
        defaults.update(kw)
        c = cls(**defaults)
        c.validate()
        return c

    @property
    def mod_in(self) -> int:
        """Neuromodulator input dim: identity + hebbian + w_conn + decay."""
        return self.D_n + self.K + self.K + 1

    @property
    def mod_out(self) -> int:
        """Neuromodulator output dim: dw_conn + ddecay + didentity."""
        return self.K + 1 + self.D_n

    @property
    def state_in(self) -> int:
        """State MLP input dim: received + h + identity + decay."""
        return 3 * self.D_n + 1

    @property
    def msg_in(self) -> int:
        """Message MLP input dim: h + identity."""
        return 2 * self.D_n
