"""Conv-grid modulator: encoder (observation → code logits) + conv-transpose decoder (code embedding → ΔW, Δdecay).

Architecture spec: `docs/design_conv_modulator.md`.

The encoder operates on an [N, N] edge feature map built by broadcasting
per-neuron projections into the grid. The decoder upsamples a seed from the
code embedding to the full N×N ΔW. Both use pre-norm residual blocks for
training stability at depth.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import Config


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
        pad = k // 2

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

        # Stem: project observation channels to C_h.
        self.stem = nn.Conv2d(C_in, C_h, kernel_size=k, padding=pad)
        self.stem_norm = nn.GroupNorm(config.conv_groups, C_h)

        # Residual pre-norm conv blocks. Layers 2..L_conv, all C_h → C_h.
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.GroupNorm(config.conv_groups, C_h),
                'conv': nn.Conv2d(C_h, C_h, kernel_size=k, padding=pad),
            }) for _ in range(config.conv_layers - 1)
        ])
        self.dropout = nn.Dropout2d(config.conv_dropout)

        # Pool → code logits.
        self.logit_head = nn.Linear(C_h, self.K)

        # Initialization: Kaiming for GELU-following convs.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Xavier for linear layers (unit-variance init through MLPs).
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

        # Per-neuron projections (bf16).
        h_p = self.h_proj(h)                       # [BS, N, d_proj]
        me_p = self.msg_emit_proj(msg)             # [BS, N, d_proj]
        mr_p = self.msg_recv_proj(received)        # [BS, N, d_proj]
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
        """x: [BS, C_in, N, N] → pooled [BS, C_h]."""
        x = F.gelu(self.stem_norm(self.stem(x)))
        for block in self.blocks:
            h = block['norm'](x)
            h = F.gelu(block['conv'](h))
            h = self.dropout(h)
            x = x + h                     # residual (both C_h)
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

        # Upsample stages. GroupNorm groups picked per-layer via gcd so that
        # each c_in divides evenly (channel ladder includes 96, 48, etc. that
        # aren't always divisible by config.conv_groups=32).
        import math as _math
        channel_sequence = [self.seed_channels] + list(self.CHANNEL_LADDER)
        self.stages = nn.ModuleList()
        for c_in, c_out in zip(channel_sequence[:-1], channel_sequence[1:]):
            g = _math.gcd(config.conv_groups, c_in)
            self.stages.append(nn.ModuleDict({
                'norm': nn.GroupNorm(g, c_in),
                'conv': nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                'proj': (nn.Conv2d(c_in, c_out, kernel_size=1)
                         if c_in != c_out else nn.Identity()),
            }))

        # Final 1×1 head → ΔW_raw. ZERO-INIT: decoder starts at no-op.
        final_channels = channel_sequence[-1]
        self.dW_head = nn.Conv2d(final_channels, 1, kernel_size=1)
        nn.init.zeros_(self.dW_head.weight)
        nn.init.zeros_(self.dW_head.bias)

        # Δdecay head: row-pool [N, final_channels] → per-neuron MLP.
        self.decay_head = nn.Sequential(
            nn.Linear(final_channels, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Diagonal mask (cached buffer).
        self.register_buffer('diag_mask', torch.eye(self.N).unsqueeze(0))

        # Kaiming init for conv layers (GELU ~ ReLU for init).
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m is not self.dW_head:
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, emb: Tensor) -> tuple[Tensor, Tensor]:
        """emb: [BS, D_code] → (ΔW_normed [BS, N, N], Δdecay_raw [BS, N])."""
        BS = emb.shape[0]
        S = self.seed_spatial
        x = self.init_proj(emb).reshape(BS, self.seed_channels, S, S)

        for stage in self.stages:
            y = F.interpolate(x, scale_factor=2, mode='bilinear',
                              align_corners=False)
            h = F.gelu(stage['conv'](stage['norm'](y)))
            x = h + stage['proj'](y)        # residual over upsampled input

        # x: [BS, final_channels, N, N]
        dW_raw = self.dW_head(x).squeeze(1)             # [BS, N, N]
        dW_raw = dW_raw * (1.0 - self.diag_mask.to(dW_raw.dtype))
        dW_normed = F.rms_norm(dW_raw, normalized_shape=(self.N,))

        # Row-pool feature map to per-neuron representation.
        row_feat = x.mean(dim=-1).transpose(1, 2)        # [BS, N, final_channels]
        dDecay_raw = self.decay_head(row_feat).squeeze(-1)  # [BS, N]

        return dW_normed, dDecay_raw


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
