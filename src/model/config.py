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

    # === Memory Graph (dense-W) ===
    D_n: int = 128  # neuron hidden dim
    alpha: int = 4  # input/output ports per cell
    grid_h: int = 4
    grid_w: int = 4
    neurons_per_cell: int = 128
    K: int = 8  # initial sparse connections per neuron (for W init only)
    border_per_cell: int = 4
    mlp_groups: int = 8
    cell_mod_hidden: int = 128
    state_mlp_hidden: int = 256
    msg_mlp_hidden: int = 256
    mod_rank: int = 16  # rank of low-rank W updates
    modulation_interval: int = 4
    w_decay_rate: float = 1e-3  # soft sparsity: W *= (1 - rate) each step

    # === Training ===
    T: int = 128  # tokens per segment
    mem_lr_scale: float = 0.3
    tbptt_block: int = 8
    checkpoint_every: int = 8

    # === Derived (set by validate()) ===
    D_cc: int = -1
    C_mem: int = -1
    N_cells: int = -1
    N: int = -1
    N_port: int = -1
    N_internal: int = -1
    N_internal_per_cell: int = -1

    def validate(self):
        assert self.D > 0
        assert self.D % self.C == 0, f"D ({self.D}) must be divisible by C ({self.C})"
        self.D_cc = self.D // self.C
        assert self.D % self.D_n == 0, f"D ({self.D}) must be divisible by D_n ({self.D_n})"
        self.C_mem = self.D // self.D_n
        self.N_cells = self.C_mem
        assert self.grid_h * self.grid_w == self.N_cells, (
            f"grid_h*grid_w ({self.grid_h * self.grid_w}) must equal "
            f"N_cells ({self.N_cells})")
        min_neurons = 2 * self.alpha + self.border_per_cell + 1
        assert self.neurons_per_cell >= min_neurons, (
            f"neurons_per_cell ({self.neurons_per_cell}) must be >= {min_neurons}")
        self.N = self.N_cells * self.neurons_per_cell
        self.N_port = self.N_cells * self.alpha
        self.N_internal_per_cell = (
            self.neurons_per_cell - 2 * self.alpha - self.border_per_cell)
        self.N_internal = self.N_cells * self.N_internal_per_cell
        assert self.K >= 1
        assert self.K <= self.neurons_per_cell
        assert self.border_per_cell == 4
        assert self.mlp_groups >= 1
        assert self.N_cells % self.mlp_groups == 0
        assert self.scan_split_at >= 1
        assert self.scan_split_at < self.L_total
        assert self.T >= 1
        assert self.modulation_interval >= 1
        assert self.mod_rank >= 1
        assert self.tbptt_block >= 1
        assert self.checkpoint_every >= 1
        assert self.checkpoint_every >= self.tbptt_block
        assert self.checkpoint_every % self.tbptt_block == 0
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
            D_n=8, alpha=2, grid_h=2, grid_w=4, neurons_per_cell=16, K=4,
            border_per_cell=4, mlp_groups=4, cell_mod_hidden=16,
            modulation_interval=2, tbptt_block=4, checkpoint_every=4,
            state_mlp_hidden=32, msg_mlp_hidden=32, mod_rank=4,
            pcm_hidden=32, w_decay_rate=1e-3,
        )
        defaults.update(kw)
        c = cls(**defaults)
        c.validate()
        return c

    @property
    def mod_in(self) -> int:
        """Per-cell modulator input: h_mean + msg_mean + ctx + W_stats + decay_mean."""
        return 3 * self.D_n + 1 + 1

    @property
    def mod_out(self) -> int:
        """Per-cell modulator output: u[N*r] + v[N*r] + ddecay[N] + dctx[D_n] + dborder[B]."""
        N, r = self.neurons_per_cell, self.mod_rank
        return 2 * N * r + N + self.D_n + self.border_per_cell

    @property
    def state_in(self) -> int:
        """State MLP input dim: received + h."""
        return 2 * self.D_n

    @property
    def msg_in(self) -> int:
        """Message MLP input dim: just h."""
        return self.D_n
