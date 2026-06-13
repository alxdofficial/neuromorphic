"""graph_substrate_v9.py — Compression-by-Vocabulary (graph_v9, 2026-06-13).

Full design: docs/compression_model_design.md. This REPLACES the operator/
Householder pyramid (retired — operators bought state-tracking, which compression
doesn't need).

v1 (THIS FILE) — nodes-only soft-clustering compressor:
  A learned NODE vocabulary across a few layers. Each input token is scored
  against the layer's node keys (routing), RE-DESCRIBED by the nodes it activates
  (residual + top-k + norm) and passed up; the sequence length stays L (layers
  re-describe, they don't shorten). The code = the M most-PRESENT node-clusters
  across ALL layers (multi-resolution), each carrying its assigned tokens' content
  (the activation-weighted centroid). Selection is NPMI-anti-hub (a node earns a
  slot by firing ABOVE its corpus baseline, not by being a hub).

v2 (LATER) — adds directed STDP edges (within + inter-layer), stateless
  instance-tag node-tokens, and a dedicated graph reader. The layer/routing/
  perturbation machinery here is the shared base v2 extends.

INTERFACE: forward(hiddens [B,L,d_model], mask [B,L]) -> memory [B, m_max, d_llama]
ranked by presence (best first); the harness slices to k = ceil(L/ratio).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _unit(x: Tensor, dim: int = -1) -> Tensor:
    return F.normalize(x, dim=dim, eps=1e-6)


def _unit_rms(x: Tensor) -> Tensor:
    return x * x.pow(2).mean(-1, keepdim=True).add(1e-12).rsqrt()


@dataclass
class GraphV9Config:
    d_model: int = 2048             # frozen-LLM hidden size (input)
    d_llama: int = 2048             # decoder hidden size (emitted token width)
    d_code: int = 256               # shared code space (vocabulary lives here)
    nodes: tuple = (512, 256, 128)  # nodes per layer (low->high; multi-resolution)
    top_k: int = 4                  # perturbation sparsity (active nodes per token)
    m_max: int = 16                 # max emitted tokens (>= max k in the data)
    effective_k: float = 8.0        # target #active nodes at init -> route temp DERIVED
    ema_decay: float = 0.99         # corpus node-activation marginal (anti-hub NPMI)

    def __post_init__(self):
        if self.d_code % 32 != 0:
            raise ValueError(f"d_code={self.d_code} must be a multiple of 32")


class GraphV9Substrate(nn.Module):
    def __init__(self, config: GraphV9Config):
        super().__init__()
        self.config = config
        cfg = config
        L = len(cfg.nodes)
        self.depth = L

        # ── slow vocabulary ────────────────────────────────────────────────────
        self.node_keys = nn.ParameterList(
            nn.Parameter(torch.randn(n, cfg.d_code)) for n in cfg.nodes)   # unit-on-use
        self.node_values = nn.ParameterList(
            nn.Parameter(torch.randn(n, cfg.d_code) / math.sqrt(cfg.d_code)) for n in cfg.nodes)
        # layer-0 reads d_model hiddens; higher layers read d_code codes
        self.route_projs = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.d_code, bias=False)]
            + [nn.Linear(cfg.d_code, cfg.d_code, bias=False) for _ in range(L - 1)])
        self.seed_proj = nn.Linear(cfg.d_model, cfg.d_code, bias=False)
        for lin in (*self.route_projs, self.seed_proj):
            nn.init.normal_(lin.weight, std=1.0 / math.sqrt(lin.in_features))
        # per-layer emit projection (cluster centroid in d_code -> d_llama token)
        self.emit_projs = nn.ModuleList(
            nn.Linear(cfg.d_code, cfg.d_llama, bias=False) for _ in range(L))
        for lin in self.emit_projs:
            nn.init.normal_(lin.weight, std=1.0 / math.sqrt(cfg.d_code))
        # per-layer presence gate: sigmoid(a*npmi + b), trains the selection
        self.presence_a = nn.Parameter(torch.ones(L))
        self.presence_b = nn.Parameter(torch.zeros(L))

        # ── learnable-but-bounded routing temperature (derived at init) ─────────
        self.log_route_temp = nn.Parameter(torch.zeros(L))
        self._calibrate_route_temps()
        # corpus node-activation marginal (EMA buffer) — the NPMI denominator
        for l, n in enumerate(cfg.nodes):
            self.register_buffer(f"act_marginal_L{l}", torch.ones(n))

    def _route_temp(self, l: int) -> Tensor:
        return self.log_route_temp[l].clamp(math.log(0.05), math.log(20.0)).exp()

    @torch.no_grad()
    def _calibrate_route_temps(self, n_query: int = 4096, iters: int = 30):
        for l, keys in enumerate(self.node_keys):
            q = _unit(torch.randn(n_query, self.config.d_code))
            logits = (q @ _unit(keys.float()).t()) * math.sqrt(self.config.d_code)
            lo, hi = math.log(0.05), math.log(20.0)
            target = math.log(min(max(self.config.effective_k, 1.0), keys.shape[0]))
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                p = torch.softmax(logits / math.exp(mid), dim=-1)
                ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean().item()
                lo, hi = (mid, hi) if ent < target else (lo, mid)
            self.log_route_temp.data[l] = 0.5 * (lo + hi)

    def route(self, l: int, x: Tensor) -> Tensor:
        """x [B,L,d_in] -> routing scores [B,L,N_l] (softmax over nodes)."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = _unit(self.route_projs[l](x.float()))
            k = _unit(self.node_keys[l].float())
            logits = torch.einsum("bld,nd->bln", q, k) * math.sqrt(self.config.d_code)
            return torch.softmax(logits / self._route_temp(l), dim=-1)

    def _perturb(self, l: int, x: Tensor, scores: Tensor) -> Tensor:
        """Re-describe each token by its top-k activated node values: residual +
        sparse + norm (no smear). x [B,L,d_code], scores [B,L,N_l]."""
        k = min(self.config.top_k, scores.shape[-1])
        topv, topi = scores.topk(k, dim=-1)                         # [B,L,k]
        vals = self.node_values[l].to(x.dtype)                      # [N,d_code]
        gathered = vals[topi]                                       # [B,L,k,d_code]
        add = (topv.unsqueeze(-1) * gathered).sum(dim=2)            # [B,L,d_code]
        return _unit_rms(x + add)

    def forward(self, hiddens: Tensor, mask: Tensor) -> tuple[Tensor, dict]:
        """hiddens [B,L,d_model] (frozen-LM contextualized span), mask [B,L].
        Returns (memory [B, m_max, d_llama] ranked by presence, aux)."""
        cfg = self.config
        B, L = hiddens.shape[:2]
        m = mask.float().unsqueeze(-1)                              # [B,L,1]
        x = _unit_rms(self.seed_proj(hiddens.float())) * m         # layer-0 input codes

        cand_tokens, cand_presence, aux = [], [], {}
        for l in range(self.depth):
            route_in = hiddens.float() if l == 0 else x
            scores = self.route(l, route_in) * m                   # [B,L,N_l]
            # cluster centroid per node = activation-weighted mean of token codes
            denom = scores.sum(dim=1).clamp_min(1e-6)              # [B,N_l]
            centroid = torch.einsum("bln,bld->bnd", scores, x) / denom.unsqueeze(-1)
            # presence (NPMI-anti-hub): activation mass above the corpus marginal
            act_mass = scores.sum(dim=1)                            # [B,N_l]
            marg = getattr(self, f"act_marginal_L{l}").clamp_min(1e-6)
            if self.training:
                with torch.no_grad():
                    marg.mul_(cfg.ema_decay).add_(act_mass.mean(0), alpha=1 - cfg.ema_decay)
            npmi = act_mass / marg                                  # >1 = above baseline
            presence = torch.sigmoid(self.presence_a[l] * npmi + self.presence_b[l])
            # token carries node IDENTITY (value) + its assigned CONTENT (centroid)
            token = self.emit_projs[l](centroid + self.node_values[l].unsqueeze(0))
            # re-describe the stream for the next layer (skip on the last layer —
            # its output would be unused)
            if l < self.depth - 1:
                x = self._perturb(l, x, scores) * m
            cand_tokens.append(token)
            cand_presence.append(presence)
            with torch.no_grad():
                aux[f"graph_v9_npmi_max_L{l}"] = npmi.max()
                aux[f"graph_v9_active_frac_L{l}"] = (act_mass > marg).float().mean()

        tokens = torch.cat(cand_tokens, dim=1)                     # [B, sumN, d_llama]
        pres = torch.cat(cand_presence, dim=1)                     # [B, sumN]
        # select the m_max most-present nodes across ALL layers (multi-resolution),
        # ranked best-first; gate each token by its presence so selection trains.
        topp, topi = pres.topk(min(cfg.m_max, pres.shape[1]), dim=1)   # [B,m_max]
        sel = tokens.gather(1, topi.unsqueeze(-1).expand(-1, -1, cfg.d_llama))
        memory = sel * topp.unsqueeze(-1)                          # presence-gated
        with torch.no_grad():
            aux["graph_v9_presence_top_mean"] = topp.mean()
            aux["graph_v9_presence_spread"] = topp.std()
        return memory, aux
