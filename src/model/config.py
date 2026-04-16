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

    # === Memory Graph: single connectivity pool (NC=1) of N_total neurons ===
    N_total: int = 256            # total neurons in the cell; W is N_total × N_total
    D_n: int = 256                # per-neuron hidden dim
    alpha: int = 4                # input/output ports per virtual I/O pool
    modulation_interval: int = 4
    state_mlp_hidden: int = 256
    msg_mlp_hidden: int = 256

    # === Conv-Grid Modulator ===
    d_proj: int = 16              # node feature compression for modulator input
    role_dim: int = 4             # role embedding (input port / output port / internal)
    conv_channels: int = 192      # encoder conv hidden width (C_h)
    conv_layers: int = 6
    conv_kernel: int = 7
    conv_groups: int = 32         # GroupNorm groups; shared encoder + decoder
    conv_dropout: float = 0.1

    # === Discrete Action Policy ===
    num_codes: int = 4096         # K — vocabulary of plasticity templates
    code_dim: int = 384           # D_code — per-code intent vector width

    # === Conv-Transpose Decoder ===
    decoder_seed_spatial: int = 4      # initial spatial dim before 6 upsample stages
    decoder_seed_channels: int = 256   # initial channel count (decoupled from D_code)
    # Upsample stages go [seed_channels, 128, 96, 64, 48, 32, 32] — last three layers
    # use 32 feature channels. The final 1×1 dW_head projects 32 → 1.

    # === Plasticity rate clamp (bf16 safety) ===
    gamma_max: float = 0.97       # γ = gamma_max · sigmoid(logit); keeps (1-γ) ≥ 0.03

    # === Tuning knobs ===
    mem_pred_weight: float = 0.1   # weight of mem_pred_loss in total loss
    mem_lr_scale: float = 1.0      # memory LR ratio vs LM LR
    gain_ema_fast: float = 0.3     # surprise EMA (no_grad, not learnable)

    # === Training ===
    T: int = 128                   # tokens per segment
    tbptt_block: int = 8           # detach memory loop every N tokens
    # At N_total=256, the conv encoder's activation stack (~11 GB per
    # modulation event × 32 events) dwarfs VRAM. Checkpointing the memory
    # block is non-negotiable at production scale. Keep on unless you're
    # specifically debugging gradient paths.
    checkpoint_memory: bool = True
    checkpoint_decoder: bool = True

    # === Derived (set by validate()) ===
    NC_pools: int = -1             # virtual I/O pools = D / D_n (for LM interface)
    N_port: int = -1               # total port neurons = NC_pools · 2 · alpha
    N_internal: int = -1

    def validate(self):
        assert self.D > 0
        assert self.D % self.D_n == 0, (
            f"D ({self.D}) must be divisible by D_n ({self.D_n})")
        self.NC_pools = self.D // self.D_n
        min_port_neurons = 2 * self.alpha * self.NC_pools
        assert self.N_total >= min_port_neurons + 1, (
            f"N_total ({self.N_total}) must be >= {min_port_neurons + 1} "
            f"(need at least 1 internal neuron)")
        self.N_port = min_port_neurons
        self.N_internal = self.N_total - self.N_port
        assert self.scan_split_at >= 1
        assert self.scan_split_at < self.L_total
        assert self.T >= 1
        assert self.modulation_interval >= 1
        assert self.tbptt_block >= 1
        assert self.conv_layers >= 1
        assert self.conv_kernel % 2 == 1, "kernel size must be odd"
        assert self.decoder_seed_spatial >= 1
        # Decoder must upsample from seed_spatial to N_total via stride-2 stages.
        # 6 stages doubles the spatial dim 6 times: seed_spatial * 2^6 = N_total.
        expected = self.decoder_seed_spatial * (2 ** 6)
        assert expected == self.N_total, (
            f"decoder_seed_spatial ({self.decoder_seed_spatial}) * 2^6 "
            f"= {expected} must equal N_total ({self.N_total})")
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
        """Small config for unit tests. N_total must = seed_spatial · 2^6."""
        defaults = dict(
            D=64, D_embed=64, L_total=4, scan_split_at=2,
            d_inner=64, glu_output=False, vocab_size=256, T=8,
            D_n=8, alpha=2,
            N_total=64,                   # seed_spatial=1 · 2^6 = 64
            decoder_seed_spatial=1,
            decoder_seed_channels=16,
            modulation_interval=2, tbptt_block=4,
            state_mlp_hidden=16, msg_mlp_hidden=16,
            d_proj=4, role_dim=2,
            conv_channels=16, conv_layers=3, conv_kernel=3, conv_groups=4,
            num_codes=8, code_dim=16,
        )
        defaults.update(kw)
        c = cls(**defaults)
        c.validate()
        return c

    @property
    def mod_in_channels(self) -> int:
        """Per-position channel count of the modulator observation tensor.

        3 raw edge (W, hebbian, asymmetry)
        + 2·d_proj node features (h_proj for receiver and sender)
        + 2·d_proj msg features (emit and recv, receiver-indexed)
        + 2·role_dim (receiver and sender roles)
        + 1 decay (receiver)
        + 2 global surprise
        """
        return (3 + 4 * self.d_proj + 2 * self.role_dim + 1 + 2)
