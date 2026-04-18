"""Attention-based neuromodulator — shared trunk + per-cell conditioning.

Weights are SHARED across the NC=8 cells (the expensive ones: attention,
FFN, logit head, decoder MLP). Cell specialization comes from a small
learned cell embedding that is (1) injected as an extra per-token feature
for the modulator, and (2) concatenated to the decoder input. This gives
each cell an identity signal at both the observation and action stages
without duplicating trunk weights.

Sharing keeps compiled matmuls fat (better GPU utilization) and parameter
count modest. Per-cell behavior still differs because (a) each cell sees
its own h/msg/W state, (b) each cell carries its own identity vector
through the shared trunk.

State shape: [BS, NC, N, D_n] for h/msg; [BS, NC, N, N] for W/hebbian.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import Config


# ======================================================================
# Modulator encoder (shared trunk, cell-embedding-conditioned)
# ======================================================================


class AttentionModulator(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.NC = config.N_cells
        self.N = config.neurons_per_cell
        self.D_n = config.D_n
        self.d_proj = config.d_proj
        self.role_dim = config.role_dim
        self.d_cell = config.d_cell
        self.K = config.num_codes

        F_tok = config.attn_token_dim
        H = config.attn_n_heads
        assert F_tok % H == 0
        self.F = F_tok
        self.H = H
        self.head_dim = F_tok // H

        # Shared per-neuron feature projections.
        self.h_proj = nn.Linear(self.D_n, self.d_proj, bias=False)
        self.msg_emit_proj = nn.Linear(self.D_n, self.d_proj, bias=False)
        self.msg_recv_proj = nn.Linear(self.D_n, self.d_proj, bias=False)
        self.role_emb = nn.Embedding(3, self.role_dim)

        # Per-cell identity embedding — the only cell-specific parameter in
        # the modulator. Broadcast across (BS, N) as a per-token feature.
        self.cell_emb = nn.Parameter(
            torch.randn(self.NC, self.d_cell) * (self.d_cell ** -0.5))

        tok_in_dim = (1 + 1 + 1 + 3 * self.d_proj + self.role_dim
                      + self.d_cell + 2)
        self.tok_in_dim = tok_in_dim
        self.tok_proj = nn.Sequential(
            nn.Linear(tok_in_dim, F_tok),
            nn.GELU(),
            nn.Linear(F_tok, F_tok),
        )

        # Shared edge-bias MLP: (W, hebbian, asym) → H heads.
        self.edge_bias_mlp = nn.Sequential(
            nn.Linear(3, max(H, 8)),
            nn.GELU(),
            nn.Linear(max(H, 8), H),
        )

        # Shared attention blocks (cells are batched over NC).
        self.layers = nn.ModuleList([
            AttnBlock(F_tok, H, ffn_mult=config.attn_ffn_mult,
                      dropout=config.attn_dropout)
            for _ in range(config.attn_n_layers)
        ])

        # Shared pool + logit head.
        self.pool_norm = nn.LayerNorm(F_tok)
        self.logit_head = nn.Linear(F_tok, self.K)

        # Init Linear / Embedding.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def build_tokens(
        self,
        h: Tensor,          # [BS, NC, N, D_n]
        msg: Tensor,        # [BS, NC, N, D_n]
        received: Tensor,   # [BS, NC, N, D_n]
        decay: Tensor,      # [BS, NC, N]
        s_live: Tensor,     # [BS]
        s_ema: Tensor,      # [BS]
        role_id: Tensor,    # [NC, N] long
    ) -> Tensor:
        """Returns [BS, NC, N, F] tokens."""
        BS, NC, N, _ = h.shape
        dt = h.dtype

        w_dt = self.h_proj.weight.dtype
        if h.dtype != w_dt:
            h = h.to(w_dt); msg = msg.to(w_dt); received = received.to(w_dt)

        h_p = self.h_proj(h)                             # [BS, NC, N, d_proj]
        me_p = self.msg_emit_proj(msg)
        mr_p = self.msg_recv_proj(received)

        h_norm = h.norm(dim=-1, keepdim=True)
        msg_norm = msg.norm(dim=-1, keepdim=True)
        dec = decay.unsqueeze(-1).to(w_dt)

        role_e = self.role_emb(role_id).to(w_dt)          # [NC, N, role_dim]
        role_e = role_e.unsqueeze(0).expand(BS, NC, N, self.role_dim)

        # Cell identity — broadcast [NC, d_cell] across (BS, N).
        cell_e = self.cell_emb.to(w_dt)                   # [NC, d_cell]
        cell_e = cell_e.view(1, NC, 1, self.d_cell).expand(BS, NC, N, self.d_cell)

        s_live_b = s_live.view(BS, 1, 1, 1).expand(BS, NC, N, 1).to(w_dt)
        s_ema_b = s_ema.view(BS, 1, 1, 1).expand(BS, NC, N, 1).to(w_dt)

        tok_in = torch.cat([
            h_norm, msg_norm, dec,
            h_p, me_p, mr_p,
            role_e, cell_e,
            s_live_b, s_ema_b,
        ], dim=-1)

        tokens = self.tok_proj(tok_in)                    # [BS, NC, N, F]
        # Keep tokens in the modulator's weight dtype (not the state dtype)
        # because downstream AttnBlock / pool_norm / logit_head all have
        # weights at w_dt and can't cleanly run on mixed-dtype inputs.
        # The state-dtype cast happens at the modulator's OUTPUT boundary
        # (codebook lookup and decoder emit .to(W.dtype) explicitly).
        return tokens

    def build_edge_bias(self, W: Tensor, hebbian: Tensor) -> Tensor:
        """Per-cell edge bias: [BS, NC, H, N, N]."""
        w_dt = self.edge_bias_mlp[0].weight.dtype
        if W.dtype != w_dt:
            W = W.to(w_dt); hebbian = hebbian.to(w_dt)
        asym = W - W.transpose(-1, -2)
        edge_feat = torch.stack([W, hebbian, asym], dim=-1)  # [BS, NC, N, N, 3]
        bias = self.edge_bias_mlp(edge_feat)
        # Keep in w_dt so SDPA inside the attention blocks sees a matching
        # dtype for q/k/v (also in w_dt after build_tokens). The modulator's
        # state-dtype cast happens at the OUTPUT boundary in `_modulate`.
        return bias.permute(0, 1, 4, 2, 3).contiguous()

    def forward(
        self,
        h: Tensor, msg: Tensor, received: Tensor,
        W: Tensor, hebbian: Tensor, decay: Tensor,
        s_live: Tensor, s_ema: Tensor,
        role_id: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Returns (logits [BS, NC, K], tokens [BS, NC, N, F])."""
        tokens = self.build_tokens(h, msg, received, decay, s_live, s_ema, role_id)
        BS, NC, N, F_tok = tokens.shape

        edge_bias = self.build_edge_bias(W, hebbian)
        tokens_flat = tokens.reshape(BS * NC, N, F_tok)
        edge_bias_flat = edge_bias.reshape(BS * NC, self.H, N, N)
        for layer in self.layers:
            tokens_flat = layer(tokens_flat, edge_bias_flat)
        tokens = tokens_flat.reshape(BS, NC, N, F_tok)

        pooled = self.pool_norm(tokens).mean(dim=2)       # [BS, NC, F]
        return self.logit_head(pooled), tokens


class AttnBlock(nn.Module):
    """Shared-weights attention block. NC as batch dim."""

    def __init__(self, F_tok: int, H: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.F = F_tok
        self.H = H
        self.head_dim = F_tok // H
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(F_tok)
        self.qkv = nn.Linear(F_tok, 3 * F_tok, bias=False)
        self.out_proj = nn.Linear(F_tok, F_tok, bias=False)

        self.norm2 = nn.LayerNorm(F_tok)
        self.ffn = nn.Sequential(
            nn.Linear(F_tok, ffn_mult * F_tok),
            nn.GELU(),
            nn.Linear(ffn_mult * F_tok, F_tok),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, edge_bias: Tensor | None = None) -> Tensor:
        """x: [B, N, F], edge_bias: [B, H, N, N] or None → [B, N, F]."""
        B, N, F_tok = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.H, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=edge_bias, scale=self.scale)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, F_tok)
        x = x + self.dropout(self.out_proj(attn_out))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ======================================================================
# Decoder (shared trunk, cell-embedding-conditioned input)
# ======================================================================


class DirectDecoder(nn.Module):
    """Shared decoder MLP conditioned on a per-cell identity vector.

    Cells differ only by their learned cell_emb (shared with the modulator's
    `cell_emb`). The MLP trunk is a single set of weights; it reads
    [code_emb | cell_emb] and emits cell-appropriate ΔW, Δdecay.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.N = config.neurons_per_cell
        self.NC = config.N_cells
        self.D_code = config.code_dim
        self.d_cell = config.d_cell

        in_dim = self.D_code + self.d_cell
        out_dim = self.N * self.N + self.N
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, config.decoder_hidden),
            nn.GELU(),
            nn.Linear(config.decoder_hidden, out_dim),
        )
        # Zero-init final layer: memory graph starts as an exact no-op so W and
        # decay stay at their initial values until the decoder learns to write.
        # This costs one step of upstream gradient dead-zone (reviewer flagged
        # B5-scale: not fatal but wastes step 0 on decoder trunk, codebook,
        # modulator qkv) — accepted trade-off because the RMS-norm on dW means
        # small-magnitude inits get amplified at init and undermine the no-op.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.register_buffer("diag_mask", torch.eye(self.N).unsqueeze(0))

    def forward(self, emb: Tensor, cell_emb: Tensor) -> tuple[Tensor, Tensor]:
        """emb: [BS, NC, D_code]. cell_emb: [NC, d_cell] (passed from modulator).

        Returns (ΔW [BS, NC, N, N], Δdecay [BS, NC, N]).
        """
        BS, NC, _ = emb.shape
        w_dt = self.mlp[0].weight.dtype
        orig_dt = emb.dtype
        if emb.dtype != w_dt:
            emb = emb.to(w_dt)
        cell_expanded = cell_emb.to(w_dt).unsqueeze(0).expand(BS, NC, self.d_cell)
        combined = torch.cat([emb, cell_expanded], dim=-1)   # [BS, NC, D_code + d_cell]

        raw = self.mlp(combined)
        dW_raw = raw[..., : self.N * self.N].reshape(BS, NC, self.N, self.N)
        dDecay_raw = raw[..., self.N * self.N:]
        dW_raw = dW_raw * (1.0 - self.diag_mask.to(dW_raw.dtype).unsqueeze(0))
        dW_normed = F.rms_norm(dW_raw, normalized_shape=(self.N,))

        if orig_dt != w_dt:
            dW_normed = dW_normed.to(orig_dt)
            dDecay_raw = dDecay_raw.to(orig_dt)
        return dW_normed, dDecay_raw


# ======================================================================
# Port layout (unchanged)
# ======================================================================


def port_layout(config: Config) -> dict:
    """Per-cell port/role layout.

    Returns:
      input_port_idx:  [NC, alpha] — local indices within each cell
      output_port_idx: [NC, alpha] — local indices within each cell
      role_id:         [NC * N]   — flat per-neuron role
                       (0=input, 1=output, 2=internal)
    """
    NC = config.N_cells
    N = config.neurons_per_cell
    alpha = config.alpha
    N_total = NC * N

    input_local = torch.arange(alpha, dtype=torch.long).unsqueeze(0).expand(NC, alpha)
    output_local = torch.arange(alpha, 2 * alpha, dtype=torch.long).unsqueeze(0).expand(NC, alpha)

    role_id = torch.full((N_total,), 2, dtype=torch.long)
    for c in range(NC):
        base = c * N
        role_id[base : base + alpha] = 0
        role_id[base + alpha : base + 2 * alpha] = 1

    return {
        "input_port_idx": input_local.contiguous(),
        "output_port_idx": output_local.contiguous(),
        "role_id": role_id,
    }
