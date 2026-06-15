"""soft_pointer_graph: soft-pointer graph memory with a no-op-free, per-token read.

Design: docs/graph_v6.md.

Governing principle — **no-op-free**: every persistent quantity (N, q_src, q_dst,
state) is consumed by the read on the critical path, so the optimizer can't zero it.

Persistent state (chunk-fresh init, carried across windows by slot index):
  N      [B, K_node, d_node]   — shared node bank (soft basis)
  q_src  [B, K_edge, d_node]   — per-edge source soft-pointer query
  q_dst  [B, K_edge, d_node]   — per-edge destination soft-pointer query
  state  [B, K_edge, d_state]  — per-edge relation

WRITE (per window): a unified typed-token transformer (TokenGT-style) over
  [K_node node tokens, 3*K_edge edge sub-tokens (src/dst/state)] + pins as cross-attn KV,
  with learnable type embeddings (x5) and a learnable per-edge instance tag binding an
  edge's three sub-tokens. Per-token FFN readout produces the new field targets; an
  anchor-biased gate blends old->target. No proposal pool, no competitive write head
  (anti-collapse handled by per-slot id embeddings (std=1.0) + cross-position whitening
  + the gate; node-only slot competition is a fallback if probes show collapse).

READ:
  Stage A (finalize_memory): build per-edge FACT-TOKENS = directional, FiLM-by-state of
    the materialized endpoints. fact = (1+gamma(state)) ⊙ combine(W_src·src, W_dst·dst)
    + beta(state). `state` MODULATES (multiplier), so it is structurally un-ignorable;
    W_src != W_dst gives directionality.
  Stage B (inject, per decode token): project each position's hidden state -> query,
    soft (full-support softmax) retrieval over the fact-tokens, query-conditioned readout,
    ReZero-scaled residual fuse. Multi-hop emerges from autoregression, not internal
    message passing.

Self-contained: AttnBlock (+ QKNormAttention), SoftPointer, and _rmsnorm —
formerly imported from the retired graph_substrate / graph_substrate_v5 — are
inlined below.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# token type ids for the write transformer
T_NODE, T_SRC, T_DST, T_STATE, T_PIN = 0, 1, 2, 3, 4
N_TYPES = 5


# ─────────────────────────────────────────────────────────────────────────
# Inlined attention primitives (formerly graph_substrate.py)
# ─────────────────────────────────────────────────────────────────────────
class QKNormAttention(nn.Module):
    """Multi-head attention with QK-Norm. Q, K projections are L-normalized
    (per-head LayerNorm) BEFORE QK^T. Output goes through standard out_proj.

    Forward: Q_in [B, Nq, D], KV_in [B, Nkv, D], optional kv_pad_mask [B, Nkv]
    where True = padded (will be masked out of attention).
    """
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        assert d % n_heads == 0, f"d={d} not divisible by n_heads={n_heads}"
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        # Per-head LN on Q and K (QK-Norm). LN over the head dim.
        self.q_norm = nn.LayerNorm(self.d_head)
        self.k_norm = nn.LayerNorm(self.d_head)

    def forward(self, q_in, kv_in, kv_pad_mask=None):
        B, Nq, D = q_in.shape
        Nkv = kv_in.shape[1]
        H, Dh = self.n_heads, self.d_head
        q = self.q_proj(q_in).view(B, Nq, H, Dh).transpose(1, 2)    # [B, H, Nq, Dh]
        k = self.k_proj(kv_in).view(B, Nkv, H, Dh).transpose(1, 2)  # [B, H, Nkv, Dh]
        v = self.v_proj(kv_in).view(B, Nkv, H, Dh).transpose(1, 2)  # [B, H, Nkv, Dh]
        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        # Build attn mask from kv_pad_mask. SDPA convention: bool mask where
        # True = include in attention (NOT masked).
        attn_mask = None
        if kv_pad_mask is not None:
            # kv_pad_mask: [B, Nkv] True=padded → invert and broadcast to [B, 1, 1, Nkv]
            attn_mask = (~kv_pad_mask).view(B, 1, 1, Nkv).expand(B, H, Nq, Nkv)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, Nq, D)
        return self.o_proj(out)


class AttnBlock(nn.Module):
    """Pre-Q-LN + KV-LN + QK-Norm attention + post-attn-LN + residual.

    Structure (per NormFormer + QK-Norm best practices):
      q_norm    = LN(Q_in)
      kv_norm   = LN(KV_in)
      attn_out  = QKNormAttention(q_norm, kv_norm, kv_pad_mask)
      attn_out  = post_norm(attn_out)            # ← caps magnitude
      return     Q_in + attn_out                  # ← residual
    For self-attention, pass kv=q.
    """
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.q_in_norm = nn.LayerNorm(d)
        self.kv_in_norm = nn.LayerNorm(d)
        self.attn = QKNormAttention(d, n_heads)
        self.post_norm = nn.LayerNorm(d)

    def forward(self, q, kv, kv_pad_mask=None, residual=True):
        q_n = self.q_in_norm(q)
        kv_n = self.kv_in_norm(kv)
        out = self.attn(q_n, kv_n, kv_pad_mask=kv_pad_mask)
        out = self.post_norm(out)
        # residual=False (reader): return the PURE attention output so the
        # caller injects a fact-only signal with no q (hidden-state) leak-through.
        return (q + out) if residual else out


def _rmsnorm(x: Tensor, eps: float = 1e-6) -> Tensor:
    d = x.shape[-1]
    return x * (d ** 0.5) / (x.pow(2).sum(-1, keepdim=True).sqrt() + eps)


# ─────────────────────────────────────────────────────────────────────────
# SoftPointer (formerly graph_substrate_v5.py)
# ─────────────────────────────────────────────────────────────────────────
class SoftPointer(nn.Module):
    """Trained soft-pointer attention from edge queries to the node bank.

    Two additions over the stateless function:
    1) Key/value separation — N is projected through separate W_k (for
       scoring) and W_v (for aggregation). Csordás et al. 2019 identified
       that without K/V separation the address distribution from content-
       based lookup is "noisy and flat, since the value influences the score
       calculation, although only the key should." This is the standard DNC
       pathology. W_k and W_v init to identity so the module starts equal
       to the un-projected function.
    2) Learnable temperature — log_tau is a trained scalar. The model
       self-tunes sharpness (CLIP / Focal Attention style). Init from a
       passed temperature; clamped to avoid pathological extremes.
    """

    def __init__(
        self,
        d_node: int,
        init_temperature: float = 1.0,
        log_tau_floor: float = -3.0,    # τ_min = exp(-3) ≈ 0.05 → very sharp
        log_tau_ceiling: float = 3.0,   # τ_max = exp(3) ≈ 20 → very flat
        kv_split: bool = True,
    ):
        super().__init__()
        self.kv_split = kv_split
        if kv_split:
            self.W_k = nn.Linear(d_node, d_node, bias=False)
            self.W_v = nn.Linear(d_node, d_node, bias=False)
            nn.init.eye_(self.W_k.weight)
            nn.init.eye_(self.W_v.weight)
        init_log_tau = math.log(max(float(init_temperature), 1e-3))
        self.log_tau = nn.Parameter(torch.tensor(float(init_log_tau)))
        self.log_tau_floor = float(log_tau_floor)
        self.log_tau_ceiling = float(log_tau_ceiling)

    @property
    def temperature(self) -> Tensor:
        """Current τ as a tensor (for telemetry)."""
        return self.log_tau.clamp(self.log_tau_floor, self.log_tau_ceiling).exp()

    def project_kv(self, N: Tensor) -> tuple[Tensor, Tensor]:
        """Precompute K and V projections of the node bank (cache once per N)."""
        if self.kv_split:
            k = self.W_k(N)
            v = self.W_v(N)
        else:
            k = N
            v = N
        return k, v

    def attend(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Compute soft-pointer attention given precomputed (k, v)."""
        d_node = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-1, -2)) * (d_node ** -0.5)
        tau = self.log_tau.clamp(self.log_tau_floor, self.log_tau_ceiling).exp()
        attn = (scores / tau.to(scores.dtype)).softmax(dim=-1)
        endpoint = torch.matmul(attn, v)
        return endpoint, attn

    def forward(self, q: Tensor, N: Tensor) -> tuple[Tensor, Tensor]:
        """
        q : [B, K_edge, d_node] — edge queries (src or dst)
        N : [B, K_node, d_node] — shared node bank
        Returns:
          endpoint : [B, K_edge, d_node]  — α @ V
          attn     : [B, K_edge, K_node]  — soft pointer weights
        """
        k, v = self.project_kv(N)
        return self.attend(q, k, v)


# ─────────────────────────────────────────────────────────────────────────
# Chunk-fresh state init (Slot-Attention-style stochastic init; no per-slot params)
# ─────────────────────────────────────────────────────────────────────────
def init_soft_pointer_graph_state(
    B: int, K_node: int, K_edge: int, d_node: int, d_state: int,
    mu_node: Tensor, log_sigma_node: Tensor,
    mu_state: Tensor, log_sigma_state: Tensor,
    mu_q: Tensor, log_sigma_q: Tensor,
    device, dtype, generator: Optional[torch.Generator] = None,
) -> dict:
    """N/q_src/q_dst/state ~ N(mu, sigma) drawn fresh per pass: per-pass noise breaks
    symmetry, learned (mu, log_sigma) set scale/center, q_src and q_dst share
    (mu_q, log_sigma_q) but draw independent noise."""
    def _sample(mu, log_sigma, K, d):
        eps = torch.randn(B, K, d, device=device, dtype=dtype, generator=generator)
        return mu.view(1, 1, -1) + log_sigma.exp().view(1, 1, -1) * eps
    return {
        "N":     _sample(mu_node, log_sigma_node, K_node, d_node),
        "q_src": _sample(mu_q, log_sigma_q, K_edge, d_node),
        "q_dst": _sample(mu_q, log_sigma_q, K_edge, d_node),
        "state": _sample(mu_state, log_sigma_state, K_edge, d_state),
    }


def _sinusoidal_pe(seq_len: int, d: int, offset: int, device, dtype) -> Tensor:
    """Standard sinusoidal positional encoding for pins, offset by chunk start so
    token position is global across windows. Returns [seq_len, d]."""
    pos = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32).unsqueeze(0)
    ang = pos / (10000.0 ** (i / d))
    pe = torch.zeros(seq_len, d, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang[:, : pe[:, 1::2].shape[1]])
    return pe.to(dtype)


# ─────────────────────────────────────────────────────────────────────────
# WRITE: unified typed-token transformer + per-token FFN readout
# ─────────────────────────────────────────────────────────────────────────
class SoftPointerGraphUpdater(nn.Module):
    """Tokenize [nodes, edge-src, edge-dst, edge-state] with type + instance/id
    embeddings; L layers of cross-attn(pins) + self-attn(graph tokens) + FFN; cross-
    position whitening; per-token FFN heads -> new field TARGETS (gated downstream)."""

    def __init__(self, d_node: int, d_state: int, d: int, K_node: int, K_edge: int,
                 d_pin: int, n_layers: int = 5, n_heads: int = 16, ffn_mult: int = 4):
        super().__init__()
        self.d_node, self.d_state, self.d = d_node, d_state, d
        self.K_node, self.K_edge = K_node, K_edge

        # input projections per role
        self.node_in = nn.Linear(d_node, d)
        self.src_in = nn.Linear(d_node, d)
        self.dst_in = nn.Linear(d_node, d)
        self.state_in = nn.Linear(d_state, d)
        self.pin_in = nn.Linear(d_pin, d) if d_pin != d else nn.Identity()

        # type embeddings (role) — small init; learnable
        self.type_emb = nn.Parameter(torch.randn(N_TYPES, d) * 0.02)
        # per-slot id embeddings (symmetry-break for the write transformer ONLY; the
        # READ addresses N by content, never sees these). std=1.0 is load-bearing:
        # without it identically-sampled slots collapse to a shared direction (v5 finding).
        self.node_id = nn.Parameter(torch.randn(K_node, d) * 1.0)
        # instance tag per edge — SAME on the edge's 3 sub-tokens so they bind, and
        # distinct across edges so edge slots break symmetry. std=1.0 for the same reason.
        self.edge_id = nn.Parameter(torch.randn(K_edge, d) * 1.0)

        self.cross_blocks = nn.ModuleList(AttnBlock(d, n_heads) for _ in range(n_layers))
        self.self_blocks = nn.ModuleList(AttnBlock(d, n_heads) for _ in range(n_layers))
        self.ffns = nn.ModuleList(
            nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))
            for _ in range(n_layers)
        )
        self.ffn_norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layers))

        # per-token FFN readout heads -> new field target values
        def _head(out_dim):
            return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, out_dim))
        self.node_head = _head(d_node)
        self.src_head = _head(d_node)
        self.dst_head = _head(d_node)
        self.state_head = _head(d_state)

    def forward(self, pins: Tensor, pins_pad_mask: Optional[Tensor],
                N: Tensor, q_src: Tensor, q_dst: Tensor, state: Tensor) -> dict:
        # pins: [B, T_w, d_pin]; field tensors as documented above. Returns target dict.
        te = self.type_emb
        node_tok = self.node_in(N) + te[T_NODE] + self.node_id.unsqueeze(0)
        src_tok = self.src_in(q_src) + te[T_SRC] + self.edge_id.unsqueeze(0)
        dst_tok = self.dst_in(q_dst) + te[T_DST] + self.edge_id.unsqueeze(0)
        st_tok = self.state_in(state) + te[T_STATE] + self.edge_id.unsqueeze(0)
        tokens = torch.cat([node_tok, src_tok, dst_tok, st_tok], dim=1)  # [B, K_node+3K_edge, d]
        pin_tok = self.pin_in(pins) + te[T_PIN]                          # cross-attn KV

        for L in range(len(self.cross_blocks)):
            tokens = self.cross_blocks[L](tokens, pin_tok, kv_pad_mask=pins_pad_mask)
            tokens = self.self_blocks[L](tokens, tokens, kv_pad_mask=None)
            tokens = tokens + self.ffns[L](self.ffn_norms[L](tokens))

        # cross-position whitening (anti rank-collapse; Dong 2021 / cheap σReparam)
        tokens = tokens - tokens.mean(dim=1, keepdim=True)

        Kn, Ke = self.K_node, self.K_edge
        node_o = tokens[:, :Kn]
        src_o = tokens[:, Kn:Kn + Ke]
        dst_o = tokens[:, Kn + Ke:Kn + 2 * Ke]
        st_o = tokens[:, Kn + 2 * Ke:Kn + 3 * Ke]
        return {
            "N": self.node_head(node_o),
            "q_src": self.src_head(src_o),
            "q_dst": self.dst_head(dst_o),
            "state": self.state_head(st_o),
        }


class SoftPointerGraphGate(nn.Module):
    """Per-slot anchor-biased gate g in [0,1] blending old -> target.
    field_new = old + g * (target - old). init_bias makes g low by default (stability)."""

    def __init__(self, in_dim: int, hidden: int = 64, init_bias: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.normal_(self.net[2].weight, std=0.01)
        nn.init.constant_(self.net[2].bias, -float(init_bias))

    def forward(self, old: Tensor, target: Tensor) -> Tensor:
        return torch.sigmoid(self.net(torch.cat([old, target], dim=-1)).squeeze(-1))  # [B, K]


# ─────────────────────────────────────────────────────────────────────────
# READ Stage A: directional FiLM-by-state fact-token builder
# ─────────────────────────────────────────────────────────────────────────
class SoftPointerGraphFactBuilder(nn.Module):
    """fact = post-MLP( LN( (W_src·src) ⊙ (W_dst·dst) ⊙ (1 + W_rel·state) ) ).

    BIND (multiplicative), replacing the old ADDITIVE FiLM pool
    `h = gelu(W_src·src + W_dst·dst)` then FiLM-by-state. The elementwise
    (Hadamard) product of role-specific projections is an order-aware
    multiplicative bind — W_src ≠ W_dst encode direction, and ⊙ binds
    which-src-with-which-dst-with-which-relation — so each fact token distinctly
    encodes its (subject, relation, object) instead of smearing them into a SUM
    (the membership→binding fix: a multiply preserves which-with-which where the
    add averages it away). `(1 + W_rel·state)`: W_rel is 1/√d_state-init (B4) so
    it starts near-transparent (fact ≈ src⊙dst) but the relation pathway is
    gradient-LIVE from step 0 — zero-init left k_rel≡0 with a zero local
    derivative, so state got no gradient until perturbed elsewhere. It then
    MODULATES multiplicatively — un-ignorable, can't null a fact (same role the
    old (1+gamma) FiLM played). Returns [B, K_edge, d_read].
    film_hidden kept only for call-site compatibility (unused)."""

    def __init__(self, d_node: int, d_state: int, d_read: int,
                 film_hidden: int = 512, mlp_hidden: int = 768):
        super().__init__()
        self.W_src = nn.Linear(d_node, d_read)
        self.W_dst = nn.Linear(d_node, d_read)
        self.W_rel = nn.Linear(d_state, d_read)
        # B4: small-NONZERO weight (std = 1/√d_state) so the edge-state→loss
        # gradient is live from step 0. W_rel was zero-init, which made the
        # relation pathway start gradient-dead — k_rel≡0 ⇒ the multiplicative
        # (1 + k_rel) term is exactly 1 with a zero local derivative, so state
        # received no gradient until something else perturbed it. A 1/√fan_in
        # init keeps the relation near-transparent (small) but live.
        nn.init.normal_(self.W_rel.weight, std=1.0 / math.sqrt(d_state))
        nn.init.zeros_(self.W_rel.bias)
        self.fact_norm = nn.LayerNorm(d_read)
        self.mlp_norm = nn.LayerNorm(d_read)
        self.mlp = nn.Sequential(nn.Linear(d_read, mlp_hidden), nn.GELU(),
                                 nn.Linear(mlp_hidden, d_read))
        self.d_read = d_read

    def forward(self, src_ep: Tensor, dst_ep: Tensor, state: Tensor,
                zero_state: bool = False) -> Tensor:
        # src_ep/dst_ep: [B, K_edge, d_node]; state: [B, K_edge, d_state]
        st = torch.zeros_like(state) if zero_state else state    # state-ablation probe hook
        k_src = self.W_src(src_ep)                               # [B, K_edge, d_read]
        k_dst = self.W_dst(dst_ep)
        k_rel = self.W_rel(st)
        bind = k_src * k_dst * (1.0 + k_rel)                     # multiplicative BIND
        fact = self.fact_norm(bind)                             # LN tames the product magnitude
        fact = fact + self.mlp(self.mlp_norm(fact))            # per-token residual refine
        return fact

