"""Graph substrate v7: stable vocabulary-atom bank + co-activation-masked mobile edges.

Design: docs/graph_v7_doctrine.md. Supersedes the graph_v6 WRITE path; reuses the v6
⊙ bind readout (GraphV6FactBuilder) and the prepend read.

Spine:
  - STABLE atom bank (learned, NOT data-modified; the v6 node_gate is gone).
  - per scope: route tokens->atoms, accumulate co-activation C (WITHIN-scope outer
    products, decayed) + per-atom content (pooled token content).
  - per window: a TokenGT over [atom tokens (carry content; un-lit masked) + edge
    tokens] residually updates the FREE edge queries q_src/q_dst + edge_state.
  - finalize: materialize endpoints by masked softmax (src over active atoms; dst over
    active + co-active-with-src via log(p_src @ C)), ⊙ bind -> prepend memory.

Gradient discipline (doc §10): eps-floored log masks, soft softmaxes (no top-k/Gumbel),
residual (differentiable-carry) edge updates, C used as a DETACHED mask (v1).
Competition over edge slots is deferred to phase 2b (edge_id breaks symmetry for now).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .graph_substrate import AttnBlock
from .graph_substrate_v5 import _rmsnorm
from .graph_substrate_v6 import GraphV6FactBuilder

T_NODE, T_SRC, T_DST, T_STATE = 0, 1, 2, 3
N_TYPES = 4


class _NormMatch(nn.Module):
    """Rescale projected memory tokens to ~Llama token magnitude (local copy of the
    encoder helper, kept here to avoid a circular import)."""

    def __init__(self, d: int, target: float = 0.9):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.scale = nn.Parameter(torch.tensor(float(target)))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.ln(x), dim=-1) * self.scale


def init_graph_v7_state(B, K_node, K_edge, d_node, d_state, d_val,
                        mu_q, log_sigma_q, mu_state, log_sigma_state,
                        device, dtype, generator: Optional[torch.Generator] = None) -> dict:
    """Edges drawn fresh per pass from learned (mu, log_sigma); accumulators start at 0.
    Atoms are NOT here — they are a stable module Parameter."""
    def _sample(mu, log_sigma, K, d):
        eps = torch.randn(B, K, d, device=device, dtype=dtype, generator=generator)
        return mu.view(1, 1, -1) + log_sigma.exp().view(1, 1, -1) * eps
    z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)
    return {
        "C": z(B, K_node, K_node),          # co-activation (within-scope, decayed)
        "content": z(B, K_node, d_val),     # per-atom pooled token content
        "a_accum": z(B, K_node),            # per-atom total activation (the active mask)
        "q_src": _sample(mu_q, log_sigma_q, K_edge, d_node),
        "q_dst": _sample(mu_q, log_sigma_q, K_edge, d_node),
        "state": _sample(mu_state, log_sigma_state, K_edge, d_state),
        "n_windows": 0,
    }


class GraphV7Substrate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.K_node = cfg.graph_v6_K_node
        self.K_edge = cfg.graph_v6_K_edge
        self.d_node = cfg.graph_v6_d_node
        self.d_state = cfg.graph_v6_d_state
        self.d_val = self.d_state
        d = cfg.graph_v6_d_updater
        d_read = cfg.graph_v6_d_read
        d_llama = cfg.d_llama
        self.scope_size = int(getattr(cfg, "graph_v7_scope_size", 16))
        self.decay = float(getattr(cfg, "graph_v7_decay", 0.98))

        # STABLE atom bank (the vocabulary; phase-5 k-means init hook lives in the encoder)
        self.atoms = nn.Parameter(torch.randn(self.K_node, self.d_node) / math.sqrt(self.d_node))
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.3)))   # routing/endpoint sharpness

        # token -> atom-query (routing) and token -> content value
        self.route_enc = nn.Sequential(nn.Linear(d_llama, d), nn.GELU(), nn.Linear(d, self.d_node))
        self.content_enc = nn.Sequential(nn.Linear(d_llama, d), nn.GELU(), nn.Linear(d, self.d_val))

        # learned edge init
        self.mu_q = nn.Parameter(torch.zeros(self.d_node))
        self.log_sigma_q = nn.Parameter(torch.zeros(self.d_node))
        self.mu_state = nn.Parameter(torch.zeros(self.d_state))
        self.log_sigma_state = nn.Parameter(torch.zeros(self.d_state))

        # ── per-window TokenGT edge updater (self-attn over [atoms(+content), edges]) ──
        self.atom_in = nn.Linear(self.d_node, d)
        self.content_in = nn.Linear(self.d_val, d)
        self.src_in = nn.Linear(self.d_node, d)
        self.dst_in = nn.Linear(self.d_node, d)
        self.state_in = nn.Linear(self.d_state, d)
        self.type_emb = nn.Parameter(torch.randn(N_TYPES, d) * 0.02)
        self.atom_id = nn.Parameter(torch.randn(self.K_node, d) * 1.0)  # symmetry-break (write only)
        self.edge_id = nn.Parameter(torch.randn(self.K_edge, d) * 1.0)
        nL, nH = cfg.graph_v6_updater_layers, cfg.graph_v6_updater_heads
        self.self_blocks = nn.ModuleList(AttnBlock(d, nH) for _ in range(nL))
        self.ffns = nn.ModuleList(
            nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(nL))
        self.ffn_norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(nL))

        def _head(out_dim):
            h = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, out_dim))
            nn.init.normal_(h[-1].weight, std=0.02)   # small (near-carry) but NONZERO -> grad flows
            nn.init.zeros_(h[-1].bias)
            return h
        self.src_head = _head(self.d_node)
        self.dst_head = _head(self.d_node)
        self.state_head = _head(self.d_state)

        # ── readout: reuse the v6 ⊙ bind ──
        self.fact_builder = GraphV6FactBuilder(
            d_node=self.d_node, d_state=self.d_state, d_read=d_read,
            film_hidden=cfg.graph_v6_film_hidden, mlp_hidden=cfg.graph_v6_builder_mlp_hidden)
        self.W_out = nn.Linear(d_read, d_llama)
        self.prepend_norm = _NormMatch(d_llama)
        # competition (edges claim distinct pairs) + atom decorrelation (keep vocab spread)
        self.comp_coef = float(getattr(cfg, "graph_v7_competition_coef", 0.01))
        self.decorr_coef = float(getattr(cfg, "graph_v7_decorr_coef", 0.01))

    # ── state ────────────────────────────────────────────────────────────────
    def init_state(self, B, device, dtype, generator=None):
        return init_graph_v7_state(
            B, self.K_node, self.K_edge, self.d_node, self.d_state, self.d_val,
            self.mu_q, self.log_sigma_q, self.mu_state, self.log_sigma_state,
            device, dtype, generator)

    def _tau(self):
        return self.log_tau.clamp(-3.0, 3.0).exp()

    # ── per-scope cheap accumulation ───────────────────────────────────────────
    def route(self, token_embeds, mask):
        tq = F.normalize(self.route_enc(token_embeds), dim=-1)         # [B,T,d_node]
        ak = F.normalize(self.atoms, dim=-1)                          # [Kn,d_node]
        logits = torch.einsum('btd,kd->btk', tq, ak) / self._tau()
        a = torch.softmax(logits, dim=-1)                             # [B,T,Kn]
        if mask is not None:
            a = a * mask.unsqueeze(-1).to(a.dtype)
        return a

    def accumulate(self, state, token_embeds, mask):
        a = self.route(token_embeds, mask)                            # [B,T,Kn]
        B, T, Kn = a.shape
        ss = self.scope_size
        pad = (ss - T % ss) % ss
        a_pad = F.pad(a, (0, 0, 0, pad)) if pad else a
        a_sc = a_pad.view(B, -1, ss, Kn).sum(dim=2)                   # [B,n_scope,Kn] within-scope sum
        C_delta = torch.einsum('bsi,bsj->bij', a_sc, a_sc)           # Σ_scope a⊗a (NOT (Σa)⊗(Σa))
        content_tok = self.content_enc(token_embeds)                 # [B,T,d_val]
        content_delta = torch.einsum('btk,btd->bkd', a, content_tok)
        a_delta = a.sum(dim=1)                                        # [B,Kn]
        d = self.decay
        state = dict(state)
        state["C"] = d * state["C"] + C_delta
        state["content"] = d * state["content"] + content_delta
        state["a_accum"] = d * state["a_accum"] + a_delta
        return state

    # ── per-window edge update ─────────────────────────────────────────────────
    def update_edges(self, state):
        B = state["q_src"].shape[0]
        atoms = self.atoms.unsqueeze(0).expand(B, -1, -1)
        te = self.type_emb
        atom_tok = (self.atom_in(atoms) + self.content_in(state["content"])
                    + te[T_NODE] + self.atom_id.unsqueeze(0))
        src_tok = self.src_in(state["q_src"]) + te[T_SRC] + self.edge_id.unsqueeze(0)
        dst_tok = self.dst_in(state["q_dst"]) + te[T_DST] + self.edge_id.unsqueeze(0)
        st_tok = self.state_in(state["state"]) + te[T_STATE] + self.edge_id.unsqueeze(0)
        tokens = torch.cat([atom_tok, src_tok, dst_tok, st_tok], dim=1)
        Kn, Ke = self.K_node, self.K_edge
        active = state["a_accum"] > 1e-6                              # [B,Kn]
        kv_mask = torch.zeros(B, tokens.shape[1], dtype=torch.bool, device=tokens.device)
        kv_mask[:, :Kn] = ~active                                    # un-lit atoms not attendable
        for L in range(len(self.self_blocks)):
            tokens = self.self_blocks[L](tokens, tokens, kv_pad_mask=kv_mask)
            tokens = tokens + self.ffns[L](self.ffn_norms[L](tokens))
        src_o = tokens[:, Kn:Kn + Ke]
        dst_o = tokens[:, Kn + Ke:Kn + 2 * Ke]
        st_o = tokens[:, Kn + 2 * Ke:Kn + 3 * Ke]
        state = dict(state)
        state["q_src"] = state["q_src"] + self.src_head(src_o)        # residual = carry + migrate
        state["q_dst"] = state["q_dst"] + self.dst_head(dst_o)
        state["state"] = _rmsnorm(state["state"] + self.state_head(st_o))
        return state

    def streaming_write(self, state, token_embeds, mask):
        state = self.accumulate(state, token_embeds, mask)
        state = self.update_edges(state)
        state = dict(state)
        state["n_windows"] = state["n_windows"] + 1
        return state

    # ── endpoint materialization + readout ─────────────────────────────────────
    def materialize(self, state):
        atoms = self.atoms
        ak = F.normalize(atoms, dim=-1)
        qs = F.normalize(state["q_src"], dim=-1)
        qd = F.normalize(state["q_dst"], dim=-1)
        active = (state["a_accum"] > 1e-6).unsqueeze(1)              # [B,1,Kn]
        neg = torch.finfo(torch.float32).min
        s_logits = (torch.einsum('bed,kd->bek', qs, ak) / self._tau()).float()
        s_logits = s_logits.masked_fill(~active, neg)
        p_src = torch.softmax(s_logits, dim=-1).to(atoms.dtype)      # [B,Ke,Kn]
        # dst restricted to src's co-activation partners (p_src @ C), diagonal excluded
        C = state["C"].detach()                                      # C is a MASK for v1
        Cz = C - torch.diag_embed(torch.diagonal(C, dim1=1, dim2=2))
        co = torch.einsum('bek,bkj->bej', p_src, Cz)                 # [B,Ke,Kn]
        d_logits = (torch.einsum('bed,kd->bek', qd, ak) / self._tau()).float()
        d_logits = d_logits + torch.log(co.float().clamp_min(1e-6))
        d_logits = d_logits.masked_fill(~active, neg)
        p_dst = torch.softmax(d_logits, dim=-1).to(atoms.dtype)
        src_ep = torch.einsum('bek,kd->bed', p_src, atoms)
        dst_ep = torch.einsum('bek,kd->bed', p_dst, atoms)
        return src_ep, dst_ep, p_src, p_dst

    def finalize(self, state):
        src_ep, dst_ep, p_src, p_dst = self.materialize(state)
        fact = self.fact_builder(src_ep, dst_ep, state["state"])     # [B,Ke,d_read]
        memory = self.prepend_norm(self.W_out(fact).float())         # [B,Ke,d_llama]
        # ── competition + atom decorrelation (single load_balance_loss the trainer weights;
        # aux-loss-as-fallback for collapse, consistent with the routing load-balance/z-loss) ──
        eye_e = torch.eye(self.K_edge, device=memory.device, dtype=torch.bool)
        joint = (torch.einsum('bek,bfk->bef', p_src, p_src)
                 * torch.einsum('bek,bfk->bef', p_dst, p_dst)).float()   # both-endpoints-shared
        comp_loss = joint.masked_fill(eye_e, 0.0).mean()             # penalize duplicate (src,dst)
        af = F.normalize(self.atoms, dim=-1)
        G = (af @ af.t()).float()
        eye_n = torch.eye(self.K_node, device=G.device, dtype=torch.bool)
        decorr_loss = (G.masked_fill(eye_n, 0.0) ** 2).mean()        # keep the vocabulary spread
        aux = {"load_balance_loss": self.comp_coef * comp_loss + self.decorr_coef * decorr_loss}
        with torch.no_grad():
            def _ent(p):
                p = p.float().clamp_min(1e-9)
                return (-(p * p.log()).sum(-1)).mean().to(torch.float32)
            aux["graph_v7_src_entropy"] = _ent(p_src)
            aux["graph_v7_dst_entropy"] = _ent(p_dst)
            aux["graph_v7_active_frac"] = (state["a_accum"] > 1e-6).float().mean().to(torch.float32)
            aux["graph_v7_competition_loss"] = comp_loss.to(torch.float32)
            aux["graph_v7_decorr_loss"] = decorr_loss.to(torch.float32)
            aux["graph_v7_atom_collapse_cos"] = G.masked_fill(eye_n, 0.0).abs().mean().to(torch.float32)
        return memory, aux
