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
    D_n: int = 32  # neuron hidden dim
    alpha: int = 4  # input/output ports per cell
    grid_h: int = 8
    grid_w: int = 8
    neurons_per_cell: int = 128
    K: int = 32  # local presynaptic neighbors per neuron
    border_per_cell: int = 4
    mlp_groups: int = 8
    cell_mod_hidden: int = 64
    state_mlp_hidden: int = 128
    msg_mlp_hidden: int = 128
    modulation_interval: int = 4

    # === Structural Plasticity ===
    structural_plasticity: bool = True
    plasticity_pct: float = 0.02
    plasticity_exploration_frac: float = 0.2
    plasticity_interval: int = 1024  # tokens between rewiring
    hebbian_ema_decay: float = 0.995

    # === Training ===
    T: int = 128  # tokens per segment
    mem_lr_scale: float = 0.3
    compile_step: bool = True  # torch.compile the per-step function
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
        assert self.K < self.neurons_per_cell, (
            f"K ({self.K}) must be < neurons_per_cell ({self.neurons_per_cell})")
        assert self.K >= 1
        assert self.border_per_cell == 4, "Current implementation expects 4 border neurons per cell"
        assert self.mlp_groups >= 1
        assert self.N_cells % self.mlp_groups == 0, (
            f"N_cells ({self.N_cells}) must be divisible by mlp_groups ({self.mlp_groups})")
        assert self.scan_split_at >= 1
        assert self.scan_split_at < self.L_total
        assert self.T >= 1
        assert self.modulation_interval >= 1
        assert self.tbptt_block >= 1
        assert self.checkpoint_every >= 1
        assert self.checkpoint_every >= self.tbptt_block, (
            f"checkpoint_every ({self.checkpoint_every}) must be >= "
            f"tbptt_block ({self.tbptt_block})")
        assert self.checkpoint_every % self.tbptt_block == 0, (
            f"checkpoint_every ({self.checkpoint_every}) must be a multiple of "
            f"tbptt_block ({self.tbptt_block})")
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
            state_mlp_hidden=32, msg_mlp_hidden=32,
            pcm_hidden=32, structural_plasticity=False, compile_step=False,
        )
        defaults.update(kw)
        c = cls(**defaults)
        c.validate()
        return c

    @property
    def mod_in(self) -> int:
        """Per-cell modulator input: h_mean + msg_mean + ctx + hebb_mean + decay_mean."""
        return 3 * self.D_n + self.K + 1

    @property
    def mod_out(self) -> int:
        """Per-cell modulator output: dw_conn + ddecay + dctx + dborder."""
        return (
            self.neurons_per_cell * self.K
            + self.neurons_per_cell
            + self.D_n
            + self.border_per_cell
        )

    @property
    def state_in(self) -> int:
        """State MLP input dim: received + h (identity/ctx/decay handled elsewhere)."""
        return 2 * self.D_n

    @property
    def msg_in(self) -> int:
        """Message MLP input dim: just h (identity/ctx handled by modulator)."""
        return self.D_n

    @property
    def neuromod_hidden(self) -> int:
        """Compatibility alias for older logging / diagnostics."""
        return self.cell_mod_hidden
