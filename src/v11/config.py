"""v11 configuration — Cell-Based Neuromorphic Memory Graph.

Flat world of cells: N_cells cells of C_neurons thin neurons each.
All connections cell-local. Multiple message-passing rounds per token.
Dedicated inject/readout port neurons (alpha redundancy factor).
"""

from dataclasses import dataclass


@dataclass
class V11Config:
    # Scan Stack (Language Model — unchanged from v9)
    D: int = 2048
    D_embed: int = 768
    C: int = 16                  # cortical columns (LM)
    D_cc: int = -1               # derived: D // C
    L_total: int = 4
    scan_split_at: int = 2
    d_inner: int = 580
    glu_output: bool = True
    vocab_size: int = 32000
    eot_id: int = 2
    tie_embeddings: bool = True
    dropout: float = 0.1

    # PCM (unchanged)
    pcm_enabled: bool = True
    pcm_pred_weight: float = 0.1
    pcm_hidden: int = 256

    # Cell-Based Memory Graph
    N_cells: int = 256           # number of cells (= D // D_neuron)
    C_neurons: int = 256         # neurons per cell
    D_neuron: int = 8            # per-neuron state dimension
    K_connections: int = 16      # cell-local connections per neuron
    R_rounds: int = 4            # message-passing rounds per token step
    alpha: int = 4               # inject/readout redundancy factor

    # Border neurons (inter-cell connectivity)
    N_border_per_cell: int = 4   # border neurons per cell
    K_border: int = 4            # inter-cell connections per border neuron

    # Shared MLP hidden sizes (H = D for balanced design at small D)
    state_mlp_hidden: int = 16   # shared state core hidden dim
    msg_mlp_hidden: int = 16     # shared message core hidden dim

    # Cell modulator
    cell_mod_hidden: int = 32    # per-cell modulator hidden dim

    # Structural plasticity (within-cell)
    structural_plasticity: bool = True
    plasticity_pct: float = 0.02
    plasticity_exploration_frac: float = 0.2
    co_activation_ema_decay: float = 0.995

    # Segment / training
    T: int = 128                 # tokens per chunk = segment length
    mem_lr_scale: float = 0.3

    # ================================================================
    # Derived properties
    # ================================================================

    @property
    def N_total(self) -> int:
        """Total neurons across all cells."""
        return self.N_cells * self.C_neurons

    @property
    def N_neurons(self) -> int:
        """Alias for compatibility with v9 interfaces."""
        return self.N_total

    @property
    def N_border_total(self) -> int:
        """Total border neurons across all cells."""
        return self.N_cells * self.N_border_per_cell

    @property
    def N_inject_per_cell(self) -> int:
        """Inject port neurons per cell."""
        return self.alpha

    @property
    def N_readout_per_cell(self) -> int:
        """Readout port neurons per cell."""
        return self.alpha

    @property
    def N_inject_total(self) -> int:
        """Total inject neurons: alpha * D_lm / D_neuron."""
        return self.alpha * self.N_cells

    @property
    def N_readout_total(self) -> int:
        """Total readout neurons."""
        return self.alpha * self.N_cells

    @property
    def D_mem(self) -> int:
        return self.D_neuron

    @property
    def C_mem(self) -> int:
        """Number of LM dim slices = N_cells."""
        return self.N_cells

    @property
    def action_every(self) -> int:
        return self.T

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
        expected_cells = self.D // self.D_neuron
        if self.N_cells != expected_cells:
            raise ValueError(
                f"N_cells ({self.N_cells}) must equal D // D_neuron "
                f"({expected_cells}).")
        total_ports = self.alpha * 2 + self.N_border_per_cell
        if self.C_neurons < total_ports + 1:
            raise ValueError(
                f"C_neurons ({self.C_neurons}) must be > "
                f"2*alpha + N_border ({total_ports}) to have interneurons.")
        if self.K_border > self.N_border_total - self.N_border_per_cell:
            raise ValueError(
                f"K_border ({self.K_border}) must be <= total border neurons "
                f"in other cells ({self.N_border_total - self.N_border_per_cell}).")
        if self.K_connections >= self.C_neurons:
            raise ValueError(
                f"K_connections ({self.K_connections}) must be < "
                f"C_neurons ({self.C_neurons}).")
        if self.R_rounds < 1:
            raise ValueError(f"R_rounds ({self.R_rounds}) must be >= 1.")
        if self.alpha < 1:
            raise ValueError(f"alpha ({self.alpha}) must be >= 1.")
        if self.T < 1:
            raise ValueError(f"T ({self.T}) must be >= 1.")
        if self.scan_split_at < 1 or self.scan_split_at >= self.L_total:
            raise ValueError(
                f"scan_split_at ({self.scan_split_at}) must be in "
                f"[1, L_total-1={self.L_total-1}].")

    @classmethod
    def tier_a(cls, **overrides) -> "V11Config":
        """Default config: 65K neurons, D=8, 256 cells of 256."""
        defaults = dict(
            D=2048, D_embed=768, C=16, L_total=4, scan_split_at=2,
            d_inner=580, glu_output=True, T=128,
            N_cells=256, C_neurons=256, D_neuron=8,
            K_connections=16, R_rounds=4, alpha=4,
            N_border_per_cell=4, K_border=4,
            pcm_hidden=256,
            cell_mod_hidden=32, state_mlp_hidden=16, msg_mlp_hidden=16,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_tiny(cls, **overrides) -> "V11Config":
        """Tiny config for unit tests."""
        defaults = dict(
            D=64, D_embed=64, C=4, L_total=4, scan_split_at=2,
            d_inner=64, glu_output=False, vocab_size=64, T=8,
            N_cells=8, C_neurons=16, D_neuron=8,
            K_connections=4, R_rounds=2, alpha=2,
            N_border_per_cell=2, K_border=2,
            pcm_hidden=32,
            cell_mod_hidden=8, state_mlp_hidden=8, msg_mlp_hidden=8,
            structural_plasticity=False,
        )
        defaults.update(overrides)
        return cls(**defaults)
