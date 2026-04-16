"""Attention-based neuromodulator: tokens-as-neurons transformer.

Architecture spec: `docs/plan_attention_neuromod.md`.

Replaces the slow conv encoder + conv-transpose decoder with:
  - Encoder: per-neuron tokens → attention (with edge-bias from W/hebbian)
    → pool → logit head over K codes
  - Decoder: code embedding → MLP → low-rank factored (U, V) + Δdecay,
    with ΔW = U @ Vᵀ

Both built from `bmm`/`linear` primitives that saturate tensor cores well,
unlike the conv approach which ran at ~8% of peak compute.

Preserves conv-grid's principled observation + full-rank action
philosophy (U and V are [N, r], so ΔW is rank-r but spans the full
N×N edge space).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import Config


# ======================================================================
# Encoder: per-neuron tokens + attention + pool → code logits
# ======================================================================


class AttentionModulator(nn.Module):
    """Observation → code logits via multi-head self-attention over neurons.

    Neurons are treated as unordered tokens. Per-node features go in as
    token embeddings; per-edge features (W, hebbian) enter as per-head
    attention biases.

    Preserves:
      - per-node content (h, msg, received — projected to d_proj)
      - per-edge observation (W, hebbian, asymmetry — as attention biases)
      - role markers (input port / output port / internal)
      - global surprise signals
    Permutation-equivariant over internal neurons by construction.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.N = config.N_total
        self.D_n = config.D_n
        self.d_proj = config.d_proj
        self.role_dim = config.role_dim
        self.K = config.num_codes

        F_tok = config.attn_token_dim
        H = config.attn_n_heads
        assert F_tok % H == 0, "attn_token_dim must be divisible by n_heads"
        self.F = F_tok
        self.H = H
        self.head_dim = F_tok // H

        # Per-neuron feature projections (reused semantics from conv-grid).
        self.h_proj = nn.Linear(self.D_n, self.d_proj, bias=False)
        self.msg_emit_proj = nn.Linear(self.D_n, self.d_proj, bias=False)
        self.msg_recv_proj = nn.Linear(self.D_n, self.d_proj, bias=False)
        self.role_emb = nn.Embedding(3, self.role_dim)

        # Token-in MLP: concat per-neuron features → F-dim token.
        # Input layout:
        #   h_norm[i]:      1
        #   msg_norm[i]:    1
        #   decay[i]:       1
        #   h_proj[i]:      d_proj
        #   msg_emit[i]:    d_proj
        #   msg_recv[i]:    d_proj
        #   role_emb[i]:    role_dim
        #   s_mem_live:     1      (broadcast global)
        #   s_mem_ema_fast: 1      (broadcast global)
        tok_in_dim = (1 + 1 + 1 + 3 * self.d_proj + self.role_dim + 2)
        self.tok_in_dim = tok_in_dim
        self.tok_proj = nn.Sequential(
            nn.Linear(tok_in_dim, F_tok),
            nn.GELU(),
            nn.Linear(F_tok, F_tok),
        )

        # Edge bias MLP: (W, hebbian, asymmetry) → per-head bias scalar.
        #   Input per (i,j): 3 scalars.
        #   Output: H scalars (one per head).
        self.edge_bias_mlp = nn.Sequential(
            nn.Linear(3, max(H, 8)),
            nn.GELU(),
            nn.Linear(max(H, 8), H),
        )

        # Attention layers.
        self.layers = nn.ModuleList([
            AttnBlock(F_tok, H, ffn_mult=config.attn_ffn_mult,
                      dropout=config.attn_dropout)
            for _ in range(config.attn_n_layers)
        ])

        # Pool → code logits head.
        self.pool_norm = nn.LayerNorm(F_tok)
        self.logit_head = nn.Linear(F_tok, self.K)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------
    # Token construction
    # ------------------------------------------------------------------

    def build_tokens(
        self,
        h: Tensor,                  # [BS, N, D_n]
        msg: Tensor,                # [BS, N, D_n]
        received: Tensor,           # [BS, N, D_n]
        decay: Tensor,              # [BS, N]
        s_live: Tensor,             # [BS]
        s_ema: Tensor,              # [BS]
        role_id: Tensor,            # [N] long
    ) -> Tensor:
        """Returns [BS, N, F] token features."""
        BS, N, _ = h.shape
        dt = h.dtype

        # Align dtype with projection weights (no-op under autocast).
        w_dt = self.h_proj.weight.dtype
        if h.dtype != w_dt:
            h = h.to(w_dt)
            msg = msg.to(w_dt)
            received = received.to(w_dt)

        # Per-neuron content projections.
        h_p = self.h_proj(h)                      # [BS, N, d_proj]
        me_p = self.msg_emit_proj(msg)            # [BS, N, d_proj]
        mr_p = self.msg_recv_proj(received)       # [BS, N, d_proj]

        # Scalar per-neuron features.
        h_norm = h.norm(dim=-1, keepdim=True)                        # [BS, N, 1]
        msg_norm = msg.norm(dim=-1, keepdim=True)                    # [BS, N, 1]
        dec = decay.unsqueeze(-1).to(w_dt)                           # [BS, N, 1]

        # Role embedding.
        role_e = self.role_emb(role_id).to(w_dt)                     # [N, role_dim]
        role_e = role_e.unsqueeze(0).expand(BS, N, self.role_dim)    # [BS, N, role_dim]

        # Global surprise broadcast per neuron.
        s_live_b = s_live.view(BS, 1, 1).expand(BS, N, 1).to(w_dt)
        s_ema_b = s_ema.view(BS, 1, 1).expand(BS, N, 1).to(w_dt)

        tok_in = torch.cat([
            h_norm, msg_norm, dec,
            h_p, me_p, mr_p,
            role_e,
            s_live_b, s_ema_b,
        ], dim=-1)                                                    # [BS, N, tok_in_dim]

        tokens = self.tok_proj(tok_in)                                # [BS, N, F]
        return tokens.to(dt) if dt != w_dt else tokens

    def build_edge_bias(self, W: Tensor, hebbian: Tensor) -> Tensor:
        """Returns [BS, H, N, N] per-head attention bias from edge features."""
        BS, N, _ = W.shape
        w_dt = self.edge_bias_mlp[0].weight.dtype
        if W.dtype != w_dt:
            W = W.to(w_dt)
            hebbian = hebbian.to(w_dt)
        asym = W - W.transpose(-1, -2)
        edge_feat = torch.stack([W, hebbian, asym], dim=-1)           # [BS, N, N, 3]
        bias = self.edge_bias_mlp(edge_feat)                          # [BS, N, N, H]
        # Reorder to [BS, H, N, N] for attention.
        return bias.permute(0, 3, 1, 2).contiguous()

    # ------------------------------------------------------------------
    # Full forward
    # ------------------------------------------------------------------

    def forward(
        self,
        h: Tensor, msg: Tensor, received: Tensor,
        W: Tensor, hebbian: Tensor, decay: Tensor,
        s_live: Tensor, s_ema: Tensor,
        role_id: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Returns (code_logits [BS, K], tokens [BS, N, F]).

        Tokens are returned so the decoder could optionally condition on
        per-neuron representations. For now the decoder only uses the code
        embedding, but keeping the hook open.
        """
        tokens = self.build_tokens(h, msg, received, decay,
                                    s_live, s_ema, role_id)
        edge_bias = self.build_edge_bias(W, hebbian)

        for layer in self.layers:
            tokens = layer(tokens, edge_bias)

        # Pool → code logits.
        pooled = self.pool_norm(tokens).mean(dim=1)                   # [BS, F]
        return self.logit_head(pooled), tokens


class AttnBlock(nn.Module):
    """Pre-norm multi-head self-attention block with edge bias + FFN."""

    def __init__(self, F_tok: int, H: int, ffn_mult: int = 4,
                 dropout: float = 0.0):
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

    def forward(self, x: Tensor, edge_bias: Tensor) -> Tensor:
        """x: [BS, N, F], edge_bias: [BS, H, N, N] → [BS, N, F]."""
        BS, N, F_tok = x.shape

        # Attention (pre-norm, residual).
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(BS, N, 3, self.H, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                              # [3, BS, H, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # scaled_dot_product_attention supports attn_mask as bias.
        # Wrap all broadcasting through attn_mask [BS, H, N, N] added to
        # the scaled QK^T.
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=edge_bias, scale=self.scale)
        attn_out = attn_out.transpose(1, 2).reshape(BS, N, F_tok)     # [BS, N, F]
        x = x + self.dropout(self.out_proj(attn_out))

        # FFN (pre-norm, residual).
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ======================================================================
# Decoder: code embedding → low-rank factored (U, V, Δdecay)
# ======================================================================


class FactoredDecoder(nn.Module):
    """Cheap MLP decoder that emits ΔW as rank-r factored product.

    Avoids the slow conv-transpose pipeline entirely. All compute goes
    through linear layers, which saturate tensor cores well.

    Output: U [N, r], V [N, r] such that ΔW = U @ Vᵀ (rank-r), plus
    Δdecay_raw [N]. Full ΔW is the rank-r matrix multiplied out, which
    still lets any (i,j) edge have its own value but constrained to a
    rank-r surface in the 1024-dim action space.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.N = config.N_total
        self.r = config.action_rank
        self.D_code = config.code_dim

        out_dim = 2 * self.N * self.r + self.N
        self.mlp = nn.Sequential(
            nn.Linear(self.D_code, config.decoder_hidden),
            nn.GELU(),
            nn.Linear(config.decoder_hidden, out_dim),
        )

        # Zero-init the final layer so ΔW ≈ 0 and Δdecay ≈ 0 at start:
        # U, V start at 0 → ΔW = 0; Δdecay = 0 → sigmoid(0) = 0.5 = init
        # decay. Safe no-op start. Matches conv-grid's dW_head zero-init.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Diagonal mask (cached buffer) for ΔW self-connection clearing.
        self.register_buffer('diag_mask', torch.eye(self.N).unsqueeze(0))

    def forward(self, emb: Tensor) -> tuple[Tensor, Tensor]:
        """emb: [BS, D_code] → (ΔW_normed [BS, N, N], Δdecay_raw [BS, N])."""
        BS = emb.shape[0]
        w_dt = self.mlp[0].weight.dtype
        orig_dt = emb.dtype
        if emb.dtype != w_dt:
            emb = emb.to(w_dt)

        raw = self.mlp(emb)                                          # [BS, 2Nr + N]
        U = raw[..., : self.N * self.r].reshape(BS, self.N, self.r)
        V = raw[..., self.N * self.r : 2 * self.N * self.r].reshape(BS, self.N, self.r)
        dDecay_raw = raw[..., 2 * self.N * self.r:]                   # [BS, N]

        # ΔW = U @ Vᵀ, then zero diagonal + row RMSNorm.
        dW_raw = torch.matmul(U, V.transpose(-1, -2))                 # [BS, N, N]
        dW_raw = dW_raw * (1.0 - self.diag_mask.to(dW_raw.dtype))
        dW_normed = F.rms_norm(dW_raw, normalized_shape=(self.N,))

        if orig_dt != w_dt:
            dW_normed = dW_normed.to(orig_dt)
            dDecay_raw = dDecay_raw.to(orig_dt)
        return dW_normed, dDecay_raw


# ======================================================================
# Helpers (port layout, unchanged from conv-grid)
# ======================================================================


def port_layout(config: Config) -> dict:
    """Same as conv-grid's port_layout — returns per-pool input/output indices
    and a per-neuron role_id tensor."""
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

    role_id = torch.full((N,), 2, dtype=torch.long)
    for p in range(NC):
        base = p * stride
        role_id[base : base + alpha] = 0
        role_id[base + alpha : base + stride] = 1

    return {
        "input_port_idx": input_idx,
        "output_port_idx": output_idx,
        "role_id": role_id,
    }
