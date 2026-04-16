"""Conv-grid modulator: encoder (observation → code logits) + conv-transpose decoder (code embedding → ΔW, Δdecay).

Architecture spec: `docs/design_conv_modulator.md`.

The encoder is a pyramid of depthwise-separable conv stages over the N×N edge
feature map — spatial dim halves at each stage, channels grow to a target
width. This avoids running the deep (wide-channel) layers on the full grid,
which was catastrophic for throughput at N=256.

The decoder mirrors this: 6 resize+DW-sep upsample stages lift a small seed
from [seed_s, seed_s] to [N, N] with channel count decreasing as spatial
dim grows.

Both use depthwise-separable convs throughout (except 1×1 heads) — these
cut compute by ~40× per layer vs dense convs at the same kernel size.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import Config


def _make_group_norm(groups: int, channels: int) -> nn.GroupNorm:
    """GroupNorm with groups adjusted so channels is divisible."""
    g = math.gcd(groups, channels)
    return nn.GroupNorm(g, channels)


class DWSepBlock(nn.Module):
    """Depthwise-separable conv block.

    depthwise kxk (groups=c_in, preserves channels) → pointwise 1×1 (c_in→c_out)
    → GroupNorm → GELU → (residual if shapes match).

    Total compute per pixel: c_in · k² + c_in · c_out, vs dense c_in · c_out · k².
    For k=5, c_in=c_out=128: dense = 410K, DW+PW = 19.6K → 21× cheaper.
    """

    def __init__(self, c_in: int, c_out: int, kernel: int = 5,
                 stride: int = 1, groups: int = 32):
        super().__init__()
        pad = kernel // 2
        self.dw = nn.Conv2d(c_in, c_in, kernel_size=kernel,
                             stride=stride, padding=pad, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.norm = _make_group_norm(groups, c_out)
        self.use_residual = (c_in == c_out and stride == 1)

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.use_residual else None
        y = self.dw(x)
        y = self.pw(y)
        y = self.norm(y)
        y = F.gelu(y)
        if identity is not None:
            y = y + identity
        return y


class DWSepUpsampleBlock(nn.Module):
    """Upsample 2× then DW-separable conv."""

    def __init__(self, c_in: int, c_out: int, kernel: int = 3,
                 groups: int = 32):
        super().__init__()
        self.block = DWSepBlock(c_in, c_out, kernel=kernel, stride=1,
                                 groups=groups)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=False)
        return self.block(x)


# ======================================================================
# Encoder: observation → code logits
# ======================================================================


class ConvGridModulator(nn.Module):
    """Conv-over-edge-grid encoder.

    Input: per-neuron state (h, msg, received) + per-edge features (W, hebbian)
    + global surprise + role embeddings.
    Output: code_logits [BS, K].

    The conv stack uses pre-norm residual blocks (GroupNorm → Conv → GELU →
    Dropout → add residual) for stability at conv_layers=6 depth.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.N = config.N_total
        self.D_n = config.D_n
        self.d_proj = config.d_proj
        self.K = config.num_codes
        self.role_dim = config.role_dim

        C_h = config.conv_channels
        k = config.conv_kernel

        # Three per-neuron feature projections — compress D_n → d_proj before
        # broadcasting into the grid. Otherwise the full D_n in the grid
        # channel dim blows VRAM.
        self.h_proj = nn.Linear(self.D_n, self.d_proj, bias=False)
        self.msg_emit_proj = nn.Linear(self.D_n, self.d_proj, bias=False)
        self.msg_recv_proj = nn.Linear(self.D_n, self.d_proj, bias=False)

        # Role embedding: 0=input-port, 1=output-port, 2=internal
        self.role_emb = nn.Embedding(3, self.role_dim)

        # Input channel count from config (matches mod_in_channels property).
        C_in = config.mod_in_channels

        # Pyramid channel ladder. Starts narrow, widens to C_h. After stem,
        # each stage downsamples spatially 2× and keeps channels at C_h.
        # Deep layers never run at full N×N — the N² cost is paid only at the
        # stem, which uses narrow channels to keep it cheap.
        ladder = [max(C_h // 4, 16),  # stem channels
                  max(C_h // 2, 32),
                  C_h, C_h, C_h]

        # Stem: dense 1× conv to land in pyramid space. Narrow channels, small
        # kernel to keep the only full-resolution layer cheap.
        stem_k = min(k, 5)
        self.stem = nn.Conv2d(C_in, ladder[0], kernel_size=stem_k,
                               padding=stem_k // 2)
        self.stem_norm = _make_group_norm(config.conv_groups, ladder[0])

        # Pyramid stages: each halves spatial dim, may widen channels.
        self.stages = nn.ModuleList()
        for i in range(1, len(ladder)):
            self.stages.append(DWSepBlock(
                ladder[i - 1], ladder[i],
                kernel=k, stride=2, groups=config.conv_groups))

        self.dropout = nn.Dropout2d(config.conv_dropout)
        self.final_channels = ladder[-1]

        # Pool → code logits.
        self.logit_head = nn.Linear(self.final_channels, self.K)

        # Initialization: Kaiming for GELU-following convs.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------
    # Observation-tensor construction
    # ------------------------------------------------------------------

    def build_input(
        self,
        h: Tensor,                  # [BS, N, D_n]
        msg: Tensor,                # [BS, N, D_n]
        received: Tensor,           # [BS, N, D_n]  = W @ msg
        W: Tensor,                  # [BS, N, N]
        hebbian: Tensor,            # [BS, N, N]
        decay: Tensor,              # [BS, N]
        s_live: Tensor,             # [BS]
        s_ema: Tensor,              # [BS]
        role_id: Tensor,            # [N] long
    ) -> Tensor:
        """Returns [BS, C_in, N, N] ready for Conv2d."""
        BS, N, _ = h.shape
        dt = h.dtype

        # Align state input dtype with projection weight dtype so this module
        # works without autocast (CPU / tier_tiny tests / smoke-test scripts).
        # Under CUDA autocast, the weight appears as bf16 and this is a no-op.
        w_dt = self.h_proj.weight.dtype
        if h.dtype != w_dt:
            h = h.to(w_dt)
            msg = msg.to(w_dt)
            received = received.to(w_dt)

        # Per-neuron projections.
        h_p = self.h_proj(h).to(dt)
        me_p = self.msg_emit_proj(msg).to(dt)
        mr_p = self.msg_recv_proj(received).to(dt)
        role_e = self.role_emb(role_id).to(dt)     # [N, role_dim]

        # Edge features [BS, N, N, 1].
        W_ij = W.unsqueeze(-1)
        heb_ij = hebbian.unsqueeze(-1)
        asym = (W - W.transpose(-1, -2)).unsqueeze(-1)

        # Broadcast helpers — use expand() which doesn't allocate.
        def bcast_row(x):  # [BS, N, F] -> [BS, N, N, F] broadcasting over j axis
            return x.unsqueeze(2).expand(BS, N, N, x.shape[-1])

        def bcast_col(x):  # [BS, N, F] -> [BS, N, N, F] broadcasting over i axis
            return x.unsqueeze(1).expand(BS, N, N, x.shape[-1])

        role_e_b = role_e.unsqueeze(0).expand(BS, N, role_e.shape[-1])  # [BS, N, R]

        # Global scalars broadcast to [BS, N, N, 1].
        s_live_b = s_live.view(BS, 1, 1, 1).expand(BS, N, N, 1).to(dt)
        s_ema_b = s_ema.view(BS, 1, 1, 1).expand(BS, N, N, 1).to(dt)

        # Decay per-neuron (receiver, broadcast over j).
        decay_b = bcast_row(decay.unsqueeze(-1).to(dt))  # [BS, N, N, 1]

        channels = torch.cat([
            W_ij,                      # 1
            heb_ij,                    # 1
            asym,                      # 1
            bcast_row(h_p),            # d_proj (receiver state)
            bcast_col(h_p),            # d_proj (sender state)
            bcast_row(me_p),           # d_proj (receiver's outgoing msg)
            bcast_row(mr_p),           # d_proj (receiver's incoming msg)
            bcast_row(role_e_b),       # role_dim (receiver role)
            bcast_col(role_e_b),       # role_dim (sender role)
            decay_b,                   # 1
            s_live_b, s_ema_b,         # 2
        ], dim=-1)                      # [BS, N, N, C_in]

        # Permute to NCHW for Conv2d.
        return channels.permute(0, 3, 1, 2).contiguous()

    # ------------------------------------------------------------------
    # Encoder forward
    # ------------------------------------------------------------------

    def encoder_forward(self, x: Tensor) -> Tensor:
        """x: [BS, C_in, N, N] → pooled [BS, final_channels].

        Pyramid: stem keeps [N, N] with narrow channels, then each stage
        halves spatial dim. By the final stage we're at [N/16, N/16, C_h]
        so global pool is cheap and deep layers don't see the full grid.
        """
        w_dt = self.stem.weight.dtype
        if x.dtype != w_dt:
            x = x.to(w_dt)
        x = F.gelu(self.stem_norm(self.stem(x)))
        x = self.dropout(x)
        for stage in self.stages:
            x = stage(x)
        return x.mean(dim=(2, 3))          # global avg pool

    def forward(
        self,
        h: Tensor, msg: Tensor, received: Tensor,
        W: Tensor, hebbian: Tensor, decay: Tensor,
        s_live: Tensor, s_ema: Tensor,
        role_id: Tensor,
    ) -> Tensor:
        """Full forward → [BS, K] code logits."""
        E = self.build_input(h, msg, received, W, hebbian, decay,
                             s_live, s_ema, role_id)
        pooled = self.encoder_forward(E)
        return self.logit_head(pooled)


# ======================================================================
# Decoder: code embedding → (ΔW, Δdecay)
# ======================================================================


class ConvTransposeDecoder(nn.Module):
    """Conv-transpose generator that upsamples a code embedding to N×N ΔW.

    Uses resize+conv (not native ConvTranspose2d) to avoid checkerboard
    artifacts. 6 upsample stages from [seed_spatial, seed_spatial] → [N, N].
    Pre-norm residual blocks for stability.

    Final 1×1 dW_head is zero-init so at training start, ΔW ≈ 0. This keeps
    W_new ≈ W_old via the EMA blend, so the modulator can't destabilize the
    memory state before it's learned anything.
    """

    # Fixed channel ladder: each entry is the output channel count of the
    # upsample stage. Input to stage 0 is `seed_channels` (from config).
    # 6 stages: seed_channels → 128 → 96 → 64 → 48 → 32 → 32.
    CHANNEL_LADDER = [128, 96, 64, 48, 32, 32]

    def __init__(self, config: Config):
        super().__init__()
        self.N = config.N_total
        self.D_code = config.code_dim
        self.seed_spatial = config.decoder_seed_spatial
        self.seed_channels = config.decoder_seed_channels

        # Initial projection: code_emb → spatial seed.
        seed_numel = self.seed_channels * self.seed_spatial * self.seed_spatial
        self.init_proj = nn.Linear(self.D_code, seed_numel)

        # Upsample stages. Each stage: 2× bilinear upsample then a
        # depthwise-separable conv block. The deep stages run at bigger
        # spatial dims but with progressively fewer channels; DW-sep keeps
        # the compute manageable.
        channel_sequence = [self.seed_channels] + list(self.CHANNEL_LADDER)
        self.stages = nn.ModuleList()
        for c_in, c_out in zip(channel_sequence[:-1], channel_sequence[1:]):
            self.stages.append(DWSepUpsampleBlock(
                c_in, c_out, kernel=3, groups=config.conv_groups))

        # Final 1×1 head → ΔW_raw.
        final_channels = channel_sequence[-1]
        self.dW_head = nn.Conv2d(final_channels, 1, kernel_size=1)

        # Δdecay head: row-pool [N, final_channels] → per-neuron MLP.
        self.decay_head = nn.Sequential(
            nn.Linear(final_channels, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Diagonal mask (cached buffer).
        self.register_buffer('diag_mask', torch.eye(self.N).unsqueeze(0))

        # Default init for all linear/conv layers first (Kaiming / Xavier).
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # THEN zero-init the output heads so the decoder starts at no-op.
        # dW_head's output → rms_norm → EMA blend with γ → safe no-op on W.
        # decay_head's final layer → sigmoid → 0.5 = init decay → safe no-op.
        # Both must happen AFTER the default init loop above.
        nn.init.zeros_(self.dW_head.weight)
        nn.init.zeros_(self.dW_head.bias)
        nn.init.zeros_(self.decay_head[-1].weight)
        nn.init.zeros_(self.decay_head[-1].bias)

    def forward(self, emb: Tensor) -> tuple[Tensor, Tensor]:
        """emb: [BS, D_code] → (ΔW_normed [BS, N, N], Δdecay_raw [BS, N])."""
        BS = emb.shape[0]
        S = self.seed_spatial
        # Cast-for-non-autocast guard.
        w_dt = self.init_proj.weight.dtype
        orig_dt = emb.dtype
        if emb.dtype != w_dt:
            emb = emb.to(w_dt)
        x = self.init_proj(emb).reshape(BS, self.seed_channels, S, S)

        for stage in self.stages:
            x = stage(x)

        # x: [BS, final_channels, N, N]
        dW_raw = self.dW_head(x).squeeze(1)             # [BS, N, N]
        dW_raw = dW_raw * (1.0 - self.diag_mask.to(dW_raw.dtype))
        dW_normed = F.rms_norm(dW_raw, normalized_shape=(self.N,))

        # Row-pool feature map to per-neuron representation.
        row_feat = x.mean(dim=-1).transpose(1, 2)        # [BS, N, final_channels]
        dDecay_raw = self.decay_head(row_feat).squeeze(-1)  # [BS, N]

        return dW_normed.to(orig_dt), dDecay_raw.to(orig_dt)


# ======================================================================
# Helpers
# ======================================================================


def port_layout(config: Config) -> dict:
    """Compute port and internal neuron indices for a single-pool cell.

    Layout across N_total neurons:
      [0, 2*alpha*NC_pools)  —  port neurons, organized as
          pool 0: input ports (alpha), output ports (alpha),
          pool 1: input ports (alpha), output ports (alpha), ...
      [2*alpha*NC_pools, N_total) — internal neurons.

    Returns dict with:
      input_port_idx:  [NC_pools, alpha] long — per-pool input neuron indices
      output_port_idx: [NC_pools, alpha] long — per-pool output neuron indices
      role_id:         [N_total] long — 0=input, 1=output, 2=internal
    """
    NC = config.NC_pools
    alpha = config.alpha
    N = config.N_total

    input_idx = torch.zeros(NC, alpha, dtype=torch.long)
    output_idx = torch.zeros(NC, alpha, dtype=torch.long)
    stride = 2 * alpha
    for p in range(NC):
        base = p * stride
        for a in range(alpha):
            input_idx[p, a] = base + a
            output_idx[p, a] = base + alpha + a

    role_id = torch.full((N,), 2, dtype=torch.long)   # default: internal
    # Each pool's block of `stride` consecutive slots: first alpha are input, next alpha are output.
    for p in range(NC):
        base = p * stride
        role_id[base : base + alpha] = 0               # input port
        role_id[base + alpha : base + stride] = 1      # output port

    return {
        "input_port_idx": input_idx,
        "output_port_idx": output_idx,
        "role_id": role_id,
    }
