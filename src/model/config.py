"""Unified configuration for LM + Memory Graph (conv-grid modulator design).

See `docs/design_conv_modulator.md` for architectural rationale.
"""

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

    # === Memory Graph: NC cells × neurons_per_cell neurons each ===
    # W is block-diagonal: [NC, Nc, Nc] — cells can't talk directly via W.
    # Cross-cell mixing happens in the modulator (it attends across cells)
    # and implicitly via LM readout/inject. NC=1 = single pool (fully connected).
    N_cells: int = 8
    neurons_per_cell: int = 32
    D_n: int = 256                # per-neuron hidden dim
    alpha: int = 4                # input/output ports per cell
    modulation_interval: int = 4
    state_mlp_hidden: int = 256
    msg_mlp_hidden: int = 256

    # === Attention Modulator (encoder) ===
    d_proj: int = 16              # node feature compression (h_proj, msg_proj)
    role_dim: int = 4             # role embedding: input port / output port / internal
    attn_token_dim: int = 64      # F — per-neuron token width
    attn_n_heads: int = 4         # multi-head attention heads (F must divide by heads)
    attn_n_layers: int = 2        # number of attention blocks
    attn_ffn_mult: int = 4        # FFN expansion factor
    attn_dropout: float = 0.1

    # === Discrete Action Policy ===
    num_codes: int = 2048         # K — vocabulary of plasticity templates
    code_dim: int = 128           # D_code — per-code intent vector width

    # === Decoder (direct emission) ===
    # Decoder: code_emb -> MLP -> [N²+N] raw scalars -> reshape to ΔW, Δdecay.
    # No rank approximation; every entry of ΔW is an independent output of
    # the decoder (same approach main takes). decoder_hidden = 512 is the
    # MLP's hidden width and IS the output-space rank constraint (rank
    # <= min(code_dim, decoder_hidden) for the code→output map, but the
    # emitted ΔW *matrix* itself is unconstrained in rank).
    decoder_hidden: int = 512

    # === Plasticity rate clamp (bf16 safety) ===
    gamma_max: float = 0.97       # γ = gamma_max · sigmoid(logit); keeps (1-γ) ≥ 0.03

    # === Tuning knobs ===
    mem_pred_weight: float = 0.1   # weight of mem_pred_loss in total loss
    mem_lr_scale: float = 1.0      # memory LR ratio vs LM LR
    gain_ema_fast: float = 0.3     # surprise EMA (no_grad, not learnable)

    # === Training ===
    T: int = 128                   # tokens per segment
    tbptt_block: int = 8           # detach memory loop every N tokens
    # BS=48 fits comfortably without checkpointing (18 GB peak) and is 20%
    # faster than BS=72 with checkpointing. If you need bigger BS, flip this
    # to True and accept the ~1.5x slowdown from forward-replay in backward.
    checkpoint_memory: bool = False

    # === Derived (set by validate()) ===
    N_total: int = -1              # N_cells × neurons_per_cell
    NC_pools: int = -1             # virtual I/O pools = D / D_n (for LM interface)
    N_port: int = -1               # total port neurons = N_cells · 2 · alpha
    N_internal: int = -1

    def validate(self):
        assert self.D > 0
        assert self.D % self.D_n == 0, (
            f"D ({self.D}) must be divisible by D_n ({self.D_n})")
        self.NC_pools = self.D // self.D_n
        self.N_total = self.N_cells * self.neurons_per_cell
        # Each cell needs alpha input ports + alpha output ports + >=1 internal.
        min_per_cell = 2 * self.alpha + 1
        assert self.neurons_per_cell >= min_per_cell, (
            f"neurons_per_cell ({self.neurons_per_cell}) must be >= {min_per_cell} "
            f"(need 2·alpha ports + 1 internal per cell)")
        # LM interface: NC_pools = D / D_n virtual I/O pools. For simplicity
        # we require N_cells be a multiple of NC_pools so each pool maps to
        # whole cells. (Most practical: N_cells == NC_pools.)
        if self.N_cells % self.NC_pools != 0 and self.NC_pools % self.N_cells != 0:
            raise ValueError(
                f"For clean LM interface, N_cells ({self.N_cells}) and "
                f"NC_pools ({self.NC_pools}) should be multiples of each other.")
        self.N_port = self.N_cells * 2 * self.alpha
        self.N_internal = self.N_total - self.N_port
        assert self.scan_split_at >= 1
        assert self.scan_split_at < self.L_total
        assert self.T >= 1
        assert self.modulation_interval >= 1
        assert self.tbptt_block >= 1
        assert self.attn_n_layers >= 1
        assert self.attn_token_dim % self.attn_n_heads == 0, (
            f"attn_token_dim ({self.attn_token_dim}) must be divisible by "
            f"attn_n_heads ({self.attn_n_heads})")
        assert 0 < self.gamma_max < 1
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
        # NC_pools = D/D_n = 64/8 = 8. Use N_cells=8, neurons_per_cell=6 so
        # N_total = 48 (>= 2*alpha*N_cells = 32 port neurons).
        defaults = dict(
            D=64, D_embed=64, L_total=4, scan_split_at=2,
            d_inner=64, glu_output=False, vocab_size=256, T=8,
            D_n=8, alpha=2,
            N_cells=8, neurons_per_cell=6,
            modulation_interval=2, tbptt_block=4,
            state_mlp_hidden=16, msg_mlp_hidden=16,
            d_proj=4, role_dim=2,
            attn_token_dim=16, attn_n_heads=2, attn_n_layers=2,
            attn_ffn_mult=2, attn_dropout=0.0,
            num_codes=8, code_dim=16,
            decoder_hidden=32,
        )
        defaults.update(kw)
        c = cls(**defaults)
        c.validate()
        return c
