"""Unified configuration for LM + Memory Graph.

See `docs/design.md` for the current architecture (multi-timescale
LIF + per-cell attention modulator + Triton-fused per-token step).
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
    # W is block-diagonal: [NC, N, N] — cells communicate only via their
    # input/output ports through the LM.
    #
    # D_n=256 is the per-neuron state width (kept large for rich per-neuron
    # capacity). NC=8 matches NC_pools = D/D_n = 2048/256 = 8 for clean
    # 1-to-1 mapping between cells and LM interface pools. Throughput at
    # this config: 37K tok/s on RTX 4090 at BS=64.
    N_cells: int = 8
    neurons_per_cell: int = 32
    D_n: int = 256
    alpha: int = 4                # input/output ports per cell
    # Multi-timescale clocks (biological): per-token membrane integration
    # (h update + W@msg message passing), event-driven spike emission (msg)
    # and Hebbian update at msg_interval, slow neuromodulation at
    # modulation_interval.
    msg_interval: int = 4         # msg MLP + hebbian fire every N tokens
    modulation_interval: int = 4  # modulator + plasticity fire every N tokens
                                  # (4 = matches msg_interval so every fire
                                  #  sees a fresh msg. Trades ~9% throughput
                                  #  for ~7× more gradient-bearing fires per
                                  #  bootstrap step vs 16 — fires that land
                                  #  right before a tbptt boundary waste
                                  #  their gradient either way.)
    # State update is a LIF-style leaky integrator — no MLP, no hidden dim.
    # Only msg emission keeps a 2-layer MLP.
    msg_mlp_hidden: int = 256

    # === Attention Modulator (encoder) ===
    # Shared trunk + per-cell conditioning via cell_emb [NC, d_cell].
    d_proj: int = 24              # node feature compression (h_proj, msg_proj)
    role_dim: int = 4             # role embedding: input port / output port / internal
    d_cell: int = 16              # per-cell identity embedding dim (feeds modulator + decoder)
    attn_token_dim: int = 128     # F — per-neuron token width
    attn_n_heads: int = 8         # multi-head attention heads (F must divide by heads)
    attn_n_layers: int = 3        # per-cell attention blocks
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
    tbptt_block: int = 32          # detach memory loop every N tokens
                                    # Must be >= 2*modulation_interval so at least
                                    # one modulator fire per block produces writes
                                    # that feed subsequent readouts with gradient.
                                    # tbptt_block == mod_interval silently zeroes
                                    # the modulator's phase-1 gradient.
    aux_loss_chunk: int = 128      # chunk size for mem_head CE (no-grad state;
                                   # larger = faster. Default = T (one pass).)
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
        # LM interface: NC_pools = D / D_n virtual I/O pools. The runtime
        # (memory.py _inject / _readout) assumes one cell per pool — we used
        # to allow multiple-of relationships in validate() but the code
        # doesn't implement that branch. Keep it strict to avoid
        # validate()-passes-but-runtime-crashes surprises.
        if self.N_cells != self.NC_pools:
            raise ValueError(
                f"N_cells ({self.N_cells}) must equal NC_pools "
                f"({self.NC_pools}) = D/D_n = {self.D}/{self.D_n}. "
                f"Set D_n such that D/D_n == N_cells.")
        self.N_port = self.N_cells * 2 * self.alpha
        self.N_internal = self.N_total - self.N_port
        assert self.scan_split_at >= 1
        assert self.scan_split_at < self.L_total
        assert self.T >= 1
        assert self.modulation_interval >= 1
        assert self.msg_interval >= 1
        assert self.modulation_interval % self.msg_interval == 0, (
            f"modulation_interval ({self.modulation_interval}) must be a "
            f"multiple of msg_interval ({self.msg_interval}) so modulator "
            f"always sees fresh msg/hebbian at its fire times")
        # Event clocks are segment-local: `t = start_t + offset` resets to 0
        # every forward_segment call. If T < modulation_interval, the modulator
        # never fires and receives no gradient; debug configs with small T used
        # to silently disable learning in the modulator/decoder/codebook path.
        assert self.T % self.modulation_interval == 0, (
            f"T ({self.T}) must be a multiple of modulation_interval "
            f"({self.modulation_interval}) so each segment contains a whole "
            f"number of modulator fire events (first fire at t=mod_interval-1)")
        assert self.tbptt_block >= 1
        # Modulator writes at t=k·mod_interval-1 get used by readouts at
        # t=k·mod_interval ... next_detach-1. If the next detach lands on the
        # token immediately after the fire (tbptt_block == mod_interval), the
        # write never feeds a readout with gradient → modulator gets zero
        # phase-1 gradient. Require at least one full mod interval of
        # graph-connected readouts per block.
        assert self.tbptt_block >= 2 * self.modulation_interval, (
            f"tbptt_block ({self.tbptt_block}) must be >= 2 * "
            f"modulation_interval ({self.modulation_interval}) so modulator "
            f"writes have at least one full interval of downstream readouts "
            f"still in the gradient graph before the next detach.")
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
            msg_interval=2, modulation_interval=2, tbptt_block=8,
            aux_loss_chunk=8,
            msg_mlp_hidden=16,
            d_proj=4, role_dim=2, d_cell=4,
            attn_token_dim=16, attn_n_heads=2, attn_n_layers=2,
            attn_ffn_mult=2, attn_dropout=0.0,
            num_codes=8, code_dim=16,
            decoder_hidden=32,
        )
        defaults.update(kw)
        c = cls(**defaults)
        c.validate()
        return c
