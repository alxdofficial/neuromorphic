"""graph model substrate — relational-parser graph memory over a learnable node bank.

Design (2026-06-16 dialogue, supersedes the VQ-codebook version): the model is a
learned RELATIONAL PARSER, not synaptic plasticity. A fixed **learnable node bank**
is the vocabulary (replaces the VQ-VAE — no encode-snap/EMA/commitment collapse).
The **write** is a TokenGT-style parser: E edge-query slots self-attend + cross-attend
the observation, and each slot SELECTS its src/dst by *pointing* into the bank (sharp
softmax — never regresses an endpoint) and regresses an edge state. The **read** binds
each edge `op(src,dst,edge)` into one vector and cross-attends those into the frozen LLM.

Why this is a graph and not a transformer-in-disguise: endpoints are constrained to
the discrete vocabulary (pointer-select, reuse via shared nodes); per-edge instance
tags make slots specialize (DETR-style anti-collapse); the read binds before pooling.
See docs/graph_model.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


def _rmsnorm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


@dataclass
class GraphConfig:
    d_llama: int = 576
    d_graph: int = 256               # graph/vocabulary space (decoupled from d_llama)
    n_nodes: int = 1024              # N — node bank size (the learnable vocabulary)
    n_edges: int = 16                # E — edge budget
    write_layers: int = 2            # parser depth (self-attend edges + cross-attend obs)
    read_layers: int = 2             # reader depth (cross-attend edges + causal self)
    heads: int = 4
    ffn_mult: int = 2


# ── attention with QK-RMSNorm + learnable temp (the read/select cold-start fix) ──
class _Attn(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.log_temp = nn.Parameter(torch.zeros(heads))           # learnable sharpness

    def forward(self, xq: Tensor, xkv: Tensor, kv_mask: Tensor = None,
                causal: bool = False) -> Tensor:
        B, Tq, _ = xq.shape; Tk = xkv.shape[1]
        q = self.q(xq).view(B, Tq, self.h, self.dh).transpose(1, 2)
        k = self.k(xkv).view(B, Tk, self.h, self.dh).transpose(1, 2)
        v = self.v(xkv).view(B, Tk, self.h, self.dh).transpose(1, 2)
        q, k = _rmsnorm(q), _rmsnorm(k)                            # QK-RMSNorm
        temp = self.log_temp.clamp(-3.0, 3.0).exp().view(1, self.h, 1, 1)
        scores = (q @ k.transpose(-1, -2)) * (self.dh ** -0.5) / temp
        if causal:
            cm = torch.triu(torch.ones(Tq, Tk, dtype=torch.bool, device=xq.device), 1)
            scores = scores.masked_fill(cm, float("-inf"))
        if kv_mask is not None:                                    # [B,Tk] True=valid
            scores = scores.masked_fill(~kv_mask[:, None, None, :], float("-inf"))
            # guard: a query whose kv is ALL masked (e.g. an all-padding window for
            # one example in the persistent carry) → all -inf → NaN softmax. Make it
            # uniform instead (the caller treats such a window as a no-op update).
            allmask = ~kv_mask.any(-1)                            # [B] True = no valid kv
            if allmask.any():
                scores = scores.masked_fill(allmask[:, None, None, None], 0.0)
        a = scores.softmax(-1)
        out = (a @ v).transpose(1, 2).reshape(B, Tq, self.h * self.dh)
        return self.o(out)


class _Block(nn.Module):
    """cross-attend(kv) → self-attend(x, optional causal) → FFN, pre-LN residual."""
    def __init__(self, d: int, heads: int, ffn_mult: int):
        super().__init__()
        self.cn = nn.LayerNorm(d); self.cross = _Attn(d, heads)
        self.sn = nn.LayerNorm(d); self.slf = _Attn(d, heads)
        self.fn = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, x: Tensor, kv: Tensor, kv_mask: Tensor = None,
                self_causal: bool = False) -> Tensor:
        x = x + self.cross(self.cn(x), kv, kv_mask=kv_mask)
        xn = self.sn(x)
        x = x + self.slf(xn, xn, causal=self_causal)
        x = x + self.ff(self.fn(x))
        return x


# ── WRITE: learned relational parser over a learnable node bank ──────────────
class GraphParser(nn.Module):
    def __init__(self, cfg: GraphConfig):
        super().__init__()
        self.cfg = cfg
        N, E, d = cfg.n_nodes, cfg.n_edges, cfg.d_graph
        # the vocabulary: N learnable node vectors. Gradient-trained (not VQ-EMA),
        # static within a forward → selection keys cache for free.
        self.node_bank = nn.Parameter(torch.randn(N, d) / math.sqrt(d))
        self.bank_key = nn.Linear(d, d, bias=False)                # match-key projection of the bank
        self.obs_proj = nn.Linear(cfg.d_llama, d)                  # LLM obs → d_graph (cross-attn KV)
        # E edge-query slots, 3 tokens each (src/dst/edge) + role + per-edge instance
        # tag (the "signature" → DETR-style specialization = anti-collapse).
        self.init_tok = nn.Parameter(torch.randn(3, E, d) / math.sqrt(d))
        self.role = nn.Parameter(torch.randn(3, d) / math.sqrt(d))
        self.tag = nn.Parameter(torch.randn(E, d) / math.sqrt(d))
        self.blocks = nn.ModuleList(_Block(d, cfg.heads, cfg.ffn_mult) for _ in range(cfg.write_layers))
        self.q_src = nn.Linear(d, d, bias=False)                   # src/dst pointer queries
        self.q_dst = nn.Linear(d, d, bias=False)
        self.log_temp = nn.Parameter(torch.zeros(2))              # learnable src/dst pointer sharpness
        self.edge_head = nn.Linear(d, d)                          # regress the relation state

    def _point(self, q: Tensor, which: int) -> tuple[Tensor, Tensor]:
        """Select a node by pointing: QK-RMSNorm + learnable-temp softmax over the
        bank → gather the bank value. Never regresses an endpoint (sharpens to a
        near-hard pick; gathered value stays on the vocabulary). Returns (value, ptr)."""
        d = q.shape[-1]
        qn = _rmsnorm(q)                                           # [B,E,d]
        kn = _rmsnorm(self.bank_key(self.node_bank))              # [N,d]
        temp = self.log_temp[which].clamp(-3.0, 3.0).exp()
        scores = (qn @ kn.t()) * (d ** -0.5) / temp               # [B,E,N]
        ptr = scores.softmax(-1)
        val = ptr @ self.node_bank                                # [B,E,d] gather (sharp → near-exact)
        return val, ptr

    def forward(self, obs: Tensor, obs_mask: Tensor, state: dict = None) -> dict:
        """Parse/UPDATE the graph from one observation window. `state` = the current
        graph (carried across windows) or None (first window → fresh init_tok slots).
        Ingesting the prior state lets the parser REFINE — re-point endpoints, re-state
        edges relative to what's there — instead of regenerating from scratch and
        scrambling slot identities. The per-edge instance tag is the persistent slot id."""
        B, E, d = obs.shape[0], self.cfg.n_edges, self.cfg.d_graph
        kv = self.obs_proj(obs.float())                           # [B,T,d]
        if state is None:                                         # window 1: fresh slots
            base = self.init_tok + self.role[:, None, :] + self.tag[None, :, :]   # [3,E,d]
            x = base.reshape(3 * E, d).unsqueeze(0).expand(B, 3 * E, d).contiguous()
        else:                                                     # window t+1: ingest carried state
            src_in = state["src_value"] + self.role[0] + self.tag    # [B,E,d]
            dst_in = state["dst_value"] + self.role[1] + self.tag
            edge_in = state["edge_state"] + self.role[2] + self.tag
            x = torch.stack([src_in, dst_in, edge_in], dim=1).reshape(B, 3 * E, d)
        for blk in self.blocks:
            x = blk(x, kv, kv_mask=obs_mask.bool())               # self-attend edges + cross-attend obs
        src_t, dst_t, edge_t = x.view(B, 3, E, d).unbind(1)
        src_v, src_ptr = self._point(self.q_src(src_t), 0)
        dst_v, dst_ptr = self._point(self.q_dst(dst_t), 1)
        edge_state = self.edge_head(edge_t)
        return {"src_value": src_v, "dst_value": dst_v, "edge_state": edge_state,
                "src_ptr": src_ptr, "dst_ptr": dst_ptr}


# ── READ: per-edge bound vector → cross-attention inject ─────────────────────
class GraphReader(nn.Module):
    def __init__(self, cfg: GraphConfig):
        super().__init__()
        self.cfg = cfg
        d, dl = cfg.d_graph, cfg.d_llama
        # per-edge bind op: bind the two endpoints, modulate by the relation (FiLM).
        # binding installed BEFORE attention (the side-car lesson) — not pooling raw tokens.
        self.w_sd = nn.Linear(2 * d, d)
        self.w_gamma = nn.Linear(d, d); self.w_beta = nn.Linear(d, d)
        self.q_in = nn.Linear(dl, d)
        self.blocks = nn.ModuleList(_Block(d, cfg.heads, cfg.ffn_mult) for _ in range(cfg.read_layers))
        self.out = nn.Linear(d, dl)
        # learnable gate, dim-scaled init (1/√d_llama) — small-but-NONZERO so the reader
        # gets gradient from step 0; tanh-bounded so the inject can't exceed stream RMS.
        self.gate = nn.Parameter(torch.tensor([dl ** -0.5]))

    def edge_tokens(self, graph: dict) -> Tensor:
        sd = self.w_sd(torch.cat([graph["src_value"], graph["dst_value"]], dim=-1))  # bind endpoints
        g = self.w_gamma(graph["edge_state"]); b = self.w_beta(graph["edge_state"])
        return g * sd + b                                          # [B,E,d] — relation FiLM-modulates the pair

    def forward(self, dec_hidden: Tensor, graph: dict) -> Tensor:
        mem = self.edge_tokens(graph)                             # [B,E,d]
        x = self.q_in(dec_hidden.float())
        for blk in self.blocks:
            x = blk(x, mem, self_causal=True)                    # cross-attend edges + causal self
        inj = self.out(x)
        stream_rms = dec_hidden.float().pow(2).mean(-1, keepdim=True).sqrt()
        inj = _rmsnorm(inj) * stream_rms                          # RMS-match to the stream
        return torch.tanh(self.gate) * inj.to(dec_hidden.dtype)
