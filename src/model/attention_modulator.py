"""Attention-based neuromodulator for multi-cell memory graph.

State shape: [BS, NC, Nc, D_n] for h/msg; [BS, NC, Nc, Nc] for W/hebbian.

Encoder: per-cell attention. Each cell's Nc neurons become Nc tokens;
attention runs within each cell independently (NC cells in parallel as
batch dim). Edge biases from W/hebbian per cell. Cells do not share
attention context — they communicate only via their input/output ports
through the LM.

Decoder: per-cell code embedding → MLP → (ΔW [Nc×Nc], Δdecay [Nc]) per cell.

Both use bmm primitives — tensor-core friendly.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import Config


# ======================================================================
# Per-cell attention encoder
# ======================================================================


class AttentionModulator(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.NC = config.N_cells
        self.Nc = config.neurons_per_cell
        self.D_n = config.D_n
        self.d_proj = config.d_proj
        self.role_dim = config.role_dim
        self.K = config.num_codes

        F_tok = config.attn_token_dim
        H = config.attn_n_heads
        assert F_tok % H == 0
        self.F = F_tok
        self.H = H
        self.head_dim = F_tok // H

        # Per-neuron feature projections (shared across cells).
        self.h_proj = nn.Linear(self.D_n, self.d_proj, bias=False)
        self.msg_emit_proj = nn.Linear(self.D_n, self.d_proj, bias=False)
        self.msg_recv_proj = nn.Linear(self.D_n, self.d_proj, bias=False)
        self.role_emb = nn.Embedding(3, self.role_dim)

        tok_in_dim = (1 + 1 + 1 + 3 * self.d_proj + self.role_dim + 2)
        self.tok_in_dim = tok_in_dim
        self.tok_proj = nn.Sequential(
            nn.Linear(tok_in_dim, F_tok),
            nn.GELU(),
            nn.Linear(F_tok, F_tok),
        )

        # Edge bias: (W, hebbian, asym) → H heads of scalar bias per edge.
        self.edge_bias_mlp = nn.Sequential(
            nn.Linear(3, max(H, 8)),
            nn.GELU(),
            nn.Linear(max(H, 8), H),
        )

        # Per-cell attention blocks (shared weights across cells).
        self.layers = nn.ModuleList([
            AttnBlock(F_tok, H, ffn_mult=config.attn_ffn_mult,
                      dropout=config.attn_dropout)
            for _ in range(config.attn_n_layers)
        ])

        # Pool + logit head (shared across cells — same projection applied to
        # each cell's pooled feature, producing [BS, NC, K] logits).
        self.pool_norm = nn.LayerNorm(F_tok)
        self.logit_head = nn.Linear(F_tok, self.K)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def build_tokens(
        self,
        h: Tensor,          # [BS, NC, Nc, D_n]
        msg: Tensor,        # [BS, NC, Nc, D_n]
        received: Tensor,   # [BS, NC, Nc, D_n]
        decay: Tensor,      # [BS, NC, Nc]
        s_live: Tensor,     # [BS]
        s_ema: Tensor,      # [BS]
        role_id: Tensor,    # [NC, Nc] long
    ) -> Tensor:
        """Returns [BS, NC, Nc, F] tokens."""
        BS, NC, Nc, _ = h.shape
        dt = h.dtype

        w_dt = self.h_proj.weight.dtype
        if h.dtype != w_dt:
            h = h.to(w_dt); msg = msg.to(w_dt); received = received.to(w_dt)

        h_p = self.h_proj(h)                             # [BS, NC, Nc, d_proj]
        me_p = self.msg_emit_proj(msg)
        mr_p = self.msg_recv_proj(received)

        h_norm = h.norm(dim=-1, keepdim=True)             # [BS, NC, Nc, 1]
        msg_norm = msg.norm(dim=-1, keepdim=True)
        dec = decay.unsqueeze(-1).to(w_dt)                # [BS, NC, Nc, 1]

        role_e = self.role_emb(role_id).to(w_dt)          # [NC, Nc, role_dim]
        role_e = role_e.unsqueeze(0).expand(BS, NC, Nc, self.role_dim)

        s_live_b = s_live.view(BS, 1, 1, 1).expand(BS, NC, Nc, 1).to(w_dt)
        s_ema_b = s_ema.view(BS, 1, 1, 1).expand(BS, NC, Nc, 1).to(w_dt)

        tok_in = torch.cat([
            h_norm, msg_norm, dec,
            h_p, me_p, mr_p,
            role_e,
            s_live_b, s_ema_b,
        ], dim=-1)

        tokens = self.tok_proj(tok_in)                    # [BS, NC, Nc, F]
        return tokens.to(dt) if dt != w_dt else tokens

    def build_edge_bias(self, W: Tensor, hebbian: Tensor) -> Tensor:
        """Per-cell edge bias: [BS, NC, H, Nc, Nc]."""
        w_dt = self.edge_bias_mlp[0].weight.dtype
        if W.dtype != w_dt:
            W = W.to(w_dt); hebbian = hebbian.to(w_dt)
        asym = W - W.transpose(-1, -2)
        edge_feat = torch.stack([W, hebbian, asym], dim=-1)  # [BS, NC, Nc, Nc, 3]
        bias = self.edge_bias_mlp(edge_feat)                 # [BS, NC, Nc, Nc, H]
        # Permute to [BS, NC, H, Nc, Nc] for attention.
        return bias.permute(0, 1, 4, 2, 3).contiguous()

    def forward(
        self,
        h: Tensor, msg: Tensor, received: Tensor,
        W: Tensor, hebbian: Tensor, decay: Tensor,
        s_live: Tensor, s_ema: Tensor,
        role_id: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Returns (logits [BS, NC, K], tokens [BS, NC, Nc, F])."""
        tokens = self.build_tokens(h, msg, received, decay, s_live, s_ema, role_id)
        BS, NC, Nc, F_tok = tokens.shape

        # Per-cell attention with edge bias from (W, hebbian, asymmetry).
        # NC runs as batch dim; no cross-cell attention — cells communicate
        # only via their inject/readout ports at the LM interface.
        edge_bias = self.build_edge_bias(W, hebbian)
        tokens_flat = tokens.reshape(BS * NC, Nc, F_tok)
        edge_bias_flat = edge_bias.reshape(BS * NC, self.H, Nc, Nc)
        for layer in self.layers:
            tokens_flat = layer(tokens_flat, edge_bias_flat)
        tokens = tokens_flat.reshape(BS, NC, Nc, F_tok)

        # Pool per cell → per-cell logits.
        pooled = self.pool_norm(tokens).mean(dim=2)       # [BS, NC, F]
        return self.logit_head(pooled), tokens


class AttnBlock(nn.Module):

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
# Per-cell decoder: code_emb → MLP → ΔW [Nc×Nc] + Δdecay [Nc]
# ======================================================================


class DirectDecoder(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.Nc = config.neurons_per_cell
        self.D_code = config.code_dim

        out_dim = self.Nc * self.Nc + self.Nc
        self.mlp = nn.Sequential(
            nn.Linear(self.D_code, config.decoder_hidden),
            nn.GELU(),
            nn.Linear(config.decoder_hidden, out_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.register_buffer('diag_mask', torch.eye(self.Nc).unsqueeze(0))

    def forward(self, emb: Tensor) -> tuple[Tensor, Tensor]:
        """emb: [B*, D_code] → (ΔW [B*, Nc, Nc], Δdecay [B*, Nc]).

        Caller reshapes B* as needed (e.g. [BS*NC] → [BS, NC, Nc, Nc]).
        """
        B = emb.shape[0]
        w_dt = self.mlp[0].weight.dtype
        orig_dt = emb.dtype
        if emb.dtype != w_dt:
            emb = emb.to(w_dt)

        raw = self.mlp(emb)
        dW_raw = raw[..., : self.Nc * self.Nc].reshape(B, self.Nc, self.Nc)
        dDecay_raw = raw[..., self.Nc * self.Nc:]
        dW_raw = dW_raw * (1.0 - self.diag_mask.to(dW_raw.dtype))
        dW_normed = F.rms_norm(dW_raw, normalized_shape=(self.Nc,))

        if orig_dt != w_dt:
            dW_normed = dW_normed.to(orig_dt)
            dDecay_raw = dDecay_raw.to(orig_dt)
        return dW_normed, dDecay_raw


# ======================================================================
# Port layout (per-cell local indices)
# ======================================================================


def port_layout(config: Config) -> dict:
    """Per-cell port/role layout.

    Returns:
      input_port_idx:  [NC, alpha] — local indices within each cell
      output_port_idx: [NC, alpha] — local indices within each cell
      role_id:         [NC * Nc]   — flat per-neuron role
                       (0=input, 1=output, 2=internal)
    """
    NC = config.N_cells
    Nc = config.neurons_per_cell
    alpha = config.alpha
    N = NC * Nc

    # Per-cell local indices: first alpha = input, next alpha = output,
    # rest = internal.
    input_local = torch.arange(alpha, dtype=torch.long).unsqueeze(0).expand(NC, alpha)
    output_local = torch.arange(alpha, 2 * alpha, dtype=torch.long).unsqueeze(0).expand(NC, alpha)

    # Flat role_id: per cell, [0:alpha)=input, [alpha:2alpha)=output, rest=internal.
    role_id = torch.full((N,), 2, dtype=torch.long)
    for c in range(NC):
        base = c * Nc
        role_id[base : base + alpha] = 0
        role_id[base + alpha : base + 2 * alpha] = 1

    return {
        "input_port_idx": input_local.contiguous(),
        "output_port_idx": output_local.contiguous(),
        "role_id": role_id,
    }
