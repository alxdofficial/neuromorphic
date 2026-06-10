"""Graph substrate v6: soft-pointer graph memory with a no-op-free, per-token read.

Design: docs/graph_v6.md. Supersedes graph_substrate_v5.

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

Reuses graph_substrate(.v5): AttnBlock, SoftPointer, _rmsnorm.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .graph_substrate import AttnBlock          # normalized cross/self attention block
from .graph_substrate_v5 import SoftPointer, _rmsnorm

# token type ids for the write transformer
T_NODE, T_SRC, T_DST, T_STATE, T_PIN = 0, 1, 2, 3, 4
N_TYPES = 5


# ─────────────────────────────────────────────────────────────────────────
# Chunk-fresh state init (Slot-Attention-style stochastic init; no per-slot params)
# ─────────────────────────────────────────────────────────────────────────
def init_graph_v6_state(
    B: int, K_node: int, K_edge: int, d_node: int, d_state: int,
    mu_node: Tensor, log_sigma_node: Tensor,
    mu_state: Tensor, log_sigma_state: Tensor,
    mu_q: Tensor, log_sigma_q: Tensor,
    device, dtype, generator: Optional[torch.Generator] = None,
) -> dict:
    """N/q_src/q_dst/state ~ N(mu, sigma) drawn fresh per pass. Same pattern as
    init_graph_v5_state: per-pass noise breaks symmetry, learned (mu, log_sigma) set
    scale/center, q_src and q_dst share (mu_q, log_sigma_q) but draw independent noise."""
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
class GraphV6Updater(nn.Module):
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


class GraphV6Gate(nn.Module):
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
class GraphV6FactBuilder(nn.Module):
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


# ─────────────────────────────────────────────────────────────────────────
# READ Stage B: per-decode-token multi-head soft retrieval + ReZero residual fuse
# ─────────────────────────────────────────────────────────────────────────
class GraphV6FactReader(nn.Module):
    """At every position: project hidden -> query, MULTI-HEAD cross-attention over the
    fact-tokens (soft, full-support — no hard top-k, no dead gradients), FFN, then a
    ReZero-scaled residual into the hidden state."""

    def __init__(self, d_llama: int, d_read: int, n_heads: int = 8,
                 ffn_mult: int = 4, scale_max: float = 1.0):
        super().__init__()
        self.q_proj = nn.Linear(d_llama, d_read)
        self.attn = AttnBlock(d_read, n_heads)        # query attends to fact-tokens (cross-attn)
        self.ffn_norm = nn.LayerNorm(d_read)
        self.ffn = nn.Sequential(nn.Linear(d_read, ffn_mult * d_read), nn.GELU(),
                                 nn.Linear(ffn_mult * d_read, d_read))
        self.W_out = nn.Linear(d_read, d_llama)
        nn.init.normal_(self.W_out.weight, std=0.01)   # v6.1: gentler init (was 0.02)
        nn.init.zeros_(self.W_out.bias)
        # v6.1 F1: per-position relevance GATE. A residual inject is otherwise always-on
        # and the decoder can't down-weight it where memory is irrelevant (unlike a
        # prepend token, which Llama's own attention can zero) — that ungated noise is
        # what pushed graph_v6 BELOW the no-memory floor. gate=σ(g(hidden))∈[0,1] makes
        # the read a true no-op per position. Init weight 0, bias +2 → gate≈0.88 (read
        # available, W_out gets gradient) and learns to gate DOWN where the read hurts.
        self.gate_proj = nn.Linear(d_llama, 1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0)
        self._last_gate_mean = None
        self.scale_max = scale_max
        self.scale_raw = nn.Parameter(torch.full((d_llama,), 0.05))  # eff ≈ 0.05 at init (gentle)

    def forward(self, hidden_states: Tensor, fact_value: Tensor,
                fact_pad_mask: Optional[Tensor] = None) -> Tensor:
        # hidden_states: [B, T, d_llama]; fact_value [B, K_edge, d_read]
        h_dtype = hidden_states.dtype
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            q = self.q_proj(hidden_states)                          # [B, T, d_read]
            # v6.1 F2: residual=False — the read is a PURE function of the retrieved
            # facts (AttnBlock's q-residual re-injected a fact-INDEPENDENT map of the
            # hidden state = the corruption term that flipped correct decodes).
            attended = self.attn(q, fact_value, kv_pad_mask=fact_pad_mask, residual=False)
            r = attended + self.ffn(self.ffn_norm(attended))
            eff = self.scale_max * torch.tanh(self.scale_raw)
            gate = torch.sigmoid(self.gate_proj(hidden_states))     # [B, T, 1] per-position off-switch
            inj = gate * eff * self.W_out(r)                        # [B, T, d_llama]
        self._last_gate_mean = gate.detach().float().mean()
        return hidden_states + (inj.to(h_dtype) if inj.dtype != h_dtype else inj)
