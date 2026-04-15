"""Unified configuration for LM + Memory Graph."""

from dataclasses import dataclass


@dataclass
class Config:
    # === LM ===
    D: int = 2048
    D_embed: int = 768
    L_total: int = 4
    scan_split_at: int = 2
    d_inner: int = 1200
    glu_output: bool = True
    vocab_size: int = 32000
    eot_id: int = 2
    tie_embeddings: bool = True
    dropout: float = 0.1

    # === Memory Graph (dense-W) ===
    D_n: int = 256             # neuron hidden dim
    alpha: int = 4             # input/output ports per cell
    neurons_per_cell: int = 48 # 16-aligned for bf16 tensor cores; ~114M total
    K: int = 8                 # initial sparse connections per neuron (W init only)
    cell_mod_hidden: int = 2048
    state_mlp_hidden: int = 256
    msg_mlp_hidden: int = 256
    modulation_interval: int = 4

    # === Tuning knobs (validated empirically — change with care) ===
    mem_pred_weight: float = 0.1   # weight of mem_pred_loss in total loss
    mem_lr_scale: float = 1.0      # memory LR ratio vs LM LR (was 0.3 in
                                   # earlier versions to damp bf16-rounded
                                   # memory grads; no longer needed after the
                                   # W-bounding fix)
    gain_ema_fast: float = 0.3     # ~3-token horizon EMA on memory-head surprise
                                   # (no_grad path, not learnable — would have
                                   # no gradient signal)

    # === Training ===
    T: int = 128               # tokens per segment
    tbptt_block: int = 8       # detach (and unroll) memory loop every N tokens
    checkpoint_memory: bool = False  # activation checkpointing on memory block

    # === Derived (set by validate()) ===
    C_mem: int = -1
    N_cells: int = -1
    N: int = -1
    N_port: int = -1
    N_internal: int = -1
    N_internal_per_cell: int = -1

    def validate(self):
        assert self.D > 0
        assert self.D % self.D_n == 0, f"D ({self.D}) must be divisible by D_n ({self.D_n})"
        self.C_mem = self.D // self.D_n
        self.N_cells = self.C_mem
        min_neurons = 2 * self.alpha + 1
        assert self.neurons_per_cell >= min_neurons, (
            f"neurons_per_cell ({self.neurons_per_cell}) must be >= {min_neurons}")
        self.N = self.N_cells * self.neurons_per_cell
        self.N_port = self.N_cells * self.alpha
        self.N_internal_per_cell = self.neurons_per_cell - 2 * self.alpha
        self.N_internal = self.N_cells * self.N_internal_per_cell
        assert self.K >= 1
        assert self.K <= self.neurons_per_cell
        assert self.scan_split_at >= 1
        assert self.scan_split_at < self.L_total
        assert self.T >= 1
        assert self.modulation_interval >= 1
        assert self.tbptt_block >= 1
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
            D=64, D_embed=64, L_total=4, scan_split_at=2,
            d_inner=64, glu_output=False, vocab_size=256, T=8,
            D_n=8, alpha=2, neurons_per_cell=16, K=4,
            cell_mod_hidden=16,
            modulation_interval=2, tbptt_block=4,
            state_mlp_hidden=32, msg_mlp_hidden=32,
        )
        defaults.update(kw)
        c = cls(**defaults)
        c.validate()
        return c

    @property
    def mod_in(self) -> int:
        """Per-cell modulator input.

        Biologically principled: rates + correlations + global modulators only.
        No feature-space "content" peek (h_mean / msg_mean were dropped).

        h_norms + msg_norms       : 2*N        (per-neuron firing rates)
        decay_mean                : 1          (per-cell average leakiness)
        readout_drift             : 1          (per-cell volatility)
        s_mem_live + s_mem_ema    : 2          (global surprise, broadcast)
        hebbian_flat              : N*N        (per-pair coactivation history)
        """
        N = self.neurons_per_cell
        return 2 * N + 1 + 1 + 2 + N * N

    @property
    def mod_out(self) -> int:
        """Per-cell modulator output: delta_W[N*N] + ddecay[N]."""
        N = self.neurons_per_cell
        return N * N + N

    @property
    def state_in(self) -> int:
        """State MLP input dim: received + h."""
        return 2 * self.D_n

    @property
    def msg_in(self) -> int:
        """Message MLP input dim: just h."""
        return self.D_n
