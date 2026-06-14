"""graph_substrate_v9.py — Compression-by-Vocabulary (graph_v9).

Build spec: docs/compression_model_design.md. Compress a passage into its OWN
learned vocabulary — a directed graph of concept-nodes — and reconstruct via MAE.

STAGING (config.use_graph):
  v2 (use_graph=True, DEFAULT) — the FULL graph:
    Phase 1 (signal processing): tokens -> frozen-LM hiddens -> per-layer routing
      activations A_l [L x N_l]; each token re-described (residual+top-k+norm) and
      passed up; sequence stays length L (multi-resolution, low->high).
    Phase 2 (graph-code, read off the activation traces):
      unified STDP  C[i->j] = Σ_t Σ_{Δ>0} exp(−Δ/τ)·A[t,i]·A[t+Δ,j]  over the token
      axis — WITHIN-layer = relational edges, INTER-layer = compositional (low->high
      prior). Shaped by a learned plasticity-bias grammar. Score edges by NPMI
      (above-chance co-activation) × a reconstruction-trained gate; select the top
      directed edges within budget. Emit STATELESS node-tokens [role, value,
      instance-tag]: an edge = a (src,dst) pair sharing a tag, in firing order.
    Phase 3 (decode): a dedicated FULLY-trainable graph-reader (groups by tag,
      orients by role) -> M soft memory tokens in Llama space -> prepend -> MAE CE.
  v1 (use_graph=False) — nodes-only ablation: no edges/reader; slots = anti-hub
    activation-weighted node centroids, prepended directly (baseline decode path).

Three timescales: backprop (vocabulary/reader/LoRA = WHAT each concept is) ·
plasticity bias (corpus co-activation grammar = WHICH relate) · per-input STDP
(the instantaneous directed graph = the CODE).

INTERFACE: forward(hiddens [B,L,d_model], mask [B,L]) -> (memory [B, m_max, d_llama]
ranked best-first, aux); the harness slices the decoder-read to k = ceil(L/ratio).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _unit(x: Tensor, dim: int = -1) -> Tensor:
    return F.normalize(x, dim=dim, eps=1e-6)


def _unit_rms(x: Tensor) -> Tensor:
    return x * x.pow(2).mean(-1, keepdim=True).add(1e-12).rsqrt()


class _NormMatch(nn.Module):
    """Put memory tokens in the decoder's token-embedding magnitude region
    (LayerNorm + L2-normalize + learnable scale). Mirrors encoder._NormMatch; the
    target scale is set to the backbone embed norm by the encoder after build."""

    def __init__(self, d: int, target: float = 0.9):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.scale = nn.Parameter(torch.tensor(float(target)))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.ln(x), dim=-1) * self.scale


@dataclass
class GraphV9Config:
    d_model: int = 2048             # frozen-LLM hidden size (input)
    d_llama: int = 2048             # decoder hidden size (emitted token width)
    d_code: int = 256               # shared code space (vocabulary lives here)
    nodes: tuple = (512, 256, 128)  # nodes per layer (low->high; multi-resolution)
    top_k: int = 4                  # perturbation sparsity (active nodes per token)
    m_max: int = 16                 # max emitted tokens (>= max k in the data)
    effective_k: float = 8.0        # target #active nodes at init -> route temp DERIVED
    ema_decay: float = 0.99         # corpus node-activation marginal (anti-hub)
    perturb_gate_init: float = 0.1  # ReZero gate on the stream re-description
    # ── v2 graph ────────────────────────────────────────────────────────────
    use_graph: bool = True          # v2 (edges+reader) vs v1 (nodes-only ablation)
    edge_topP: int = 32             # node prefilter: edges only among top-P active nodes/layer
    stdp_tau_init: tuple = (1.0, 2.0, 4.0)   # per-layer decay ladder (tokens; small->large)
    # learnable, context-aware EDGE SELECTOR (transformer over candidate edges —
    # "whole-list view": each candidate is scored with attention to all the others,
    # so it can learn non-redundant coverage instead of independent per-edge scores).
    edge_cand: int = 48             # candidate edges after the cheap STDP-lift prefilter
    d_sel: int = 192                # selector transformer width
    sel_layers: int = 2
    sel_heads: int = 4
    # dedicated graph reader (over the SELECTED stateless node-tokens)
    d_read: int = 192               # graph-reader width
    reader_layers: int = 2
    reader_heads: int = 4

    def __post_init__(self):
        if self.d_code % 32 != 0:
            raise ValueError(f"d_code={self.d_code} must be a multiple of 32")
        if self.use_graph:   # fail-fast on v2 shape assumptions [merge #9]
            if self.m_max < 2:
                raise ValueError(f"use_graph needs m_max>=2 (edges are pairs); got {self.m_max}")
            if self.d_read % self.reader_heads != 0:
                raise ValueError(f"d_read={self.d_read} must be divisible by reader_heads={self.reader_heads}")
            if len(self.stdp_tau_init) < len(self.nodes):
                raise ValueError(f"stdp_tau_init needs >= {len(self.nodes)} entries (one τ per layer)")
            if self.d_sel % self.sel_heads != 0:
                raise ValueError(f"d_sel={self.d_sel} must be divisible by sel_heads={self.sel_heads}")
            if self.edge_cand < self.m_max // 2:
                raise ValueError(f"edge_cand={self.edge_cand} must be >= n_edges={self.m_max // 2}")


class GraphV9Substrate(nn.Module):
    def __init__(self, config: GraphV9Config):
        super().__init__()
        self.config = config
        cfg = config
        L = len(cfg.nodes)
        self.depth = L

        # ── slow vocabulary (small-init keys: cosine-unit direction must move) ──
        self.node_keys = nn.ParameterList(
            nn.Parameter(torch.randn(n, cfg.d_code) / math.sqrt(cfg.d_code)) for n in cfg.nodes)
        self.node_values = nn.ParameterList(
            nn.Parameter(torch.randn(n, cfg.d_code) / math.sqrt(cfg.d_code)) for n in cfg.nodes)
        self.route_projs = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.d_code, bias=False)]
            + [nn.Linear(cfg.d_code, cfg.d_code, bias=False) for _ in range(L - 1)])
        self.seed_proj = nn.Linear(cfg.d_model, cfg.d_code, bias=False)
        for lin in (*self.route_projs, self.seed_proj):
            nn.init.normal_(lin.weight, std=1.0 / math.sqrt(lin.in_features))
        self.emit_projs = nn.ModuleList(
            nn.Linear(cfg.d_code, cfg.d_llama, bias=False) for _ in range(L))
        for lin in self.emit_projs:
            nn.init.normal_(lin.weight, std=1.0 / math.sqrt(cfg.d_code))
        self.token_norm = _NormMatch(cfg.d_llama)
        self.presence_a = nn.Parameter(torch.ones(L))      # v1 selection gate
        self.presence_b = nn.Parameter(torch.zeros(L))
        self.perturb_gate = nn.Parameter(torch.full((L,), float(cfg.perturb_gate_init)))

        # routing temperature (calibrated at init; pure cosine, no sqrt(d))
        self.log_route_temp = nn.Parameter(torch.zeros(L))
        self._calibrate_route_temps()
        # bias-corrected corpus node marginals (anti-hub denominator)
        self.register_buffer("marg_count", torch.zeros(()))
        for l, n in enumerate(cfg.nodes):
            self.register_buffer(f"act_marginal_L{l}", torch.zeros(n))

        # ── v2 graph machinery ─────────────────────────────────────────────────
        if cfg.use_graph:
            # learnable STDP decay ladder (within-layer τ_l, inter-layer τ_{l->l+1})
            self.log_tau_within = nn.Parameter(torch.log(torch.tensor(
                [cfg.stdp_tau_init[l] for l in range(L)])))
            self.log_tau_inter = nn.Parameter(torch.log(torch.tensor(
                [cfg.stdp_tau_init[l] for l in range(L - 1)])))
            # context-aware EDGE SELECTOR: a transformer over the candidate edges.
            # Each candidate edge token = [src value, dst value, STDP scalar feats];
            # self-attention gives every candidate a view of all the others, so the
            # keep-logit is scored IN CONTEXT (learns the grammar + non-redundancy,
            # replacing the hard-coded log(lift)+log(gate) and the bilinear prior).
            self._n_edge_scalar = 3                              # log_lift, log_C, is_inter
            self.edge_in = nn.Linear(2 * cfg.d_llama + self._n_edge_scalar, cfg.d_sel)
            self.selector = nn.ModuleList(
                nn.TransformerEncoderLayer(
                    cfg.d_sel, cfg.sel_heads, dim_feedforward=4 * cfg.d_sel,
                    dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
                for _ in range(cfg.sel_layers))
            self.sel_head = nn.Linear(cfg.d_sel, 1)
            # the selector's CONTEXTUALIZED edge rep feeds token content (not just the
            # keep-logit) — else the selector only gates magnitude, which token_norm
            # washes out, starving the selector + STDP grammar of gradient.
            self.sel_to_tok = nn.Linear(cfg.d_sel, cfg.d_llama)
            self.edge_score_temp = nn.Parameter(torch.tensor(0.0))   # selection softmax temp (recon-trained)
            # stateless-token embeddings: role (src/dst/standalone) + instance-tags.
            # init at norm ~1 (1/sqrt d) to MATCH the unit value-token scale — else
            # the structural tags are ~2% of the signal and the reader can't see them.
            self.role_emb = nn.Parameter(torch.randn(3, cfg.d_llama) / math.sqrt(cfg.d_llama))
            n_edges = cfg.m_max // 2
            self.n_edges = n_edges
            self.tag_emb = nn.Parameter(torch.randn(n_edges, cfg.d_llama) / math.sqrt(cfg.d_llama))
            # directed (src,dst) layer pairs: within-layer + adjacent inter-layer
            self.edge_sources = ([(l, l) for l in range(L)]
                                 + [(l, l + 1) for l in range(L - 1)])
            # dedicated FULLY-trainable graph reader (small width)
            self.read_in = nn.Linear(cfg.d_llama, cfg.d_read)
            self.reader = nn.ModuleList(
                nn.TransformerEncoderLayer(
                    cfg.d_read, cfg.reader_heads, dim_feedforward=4 * cfg.d_read,
                    dropout=0.0,   # memory module: no stochastic noise (MAE mask regularizes)
                    batch_first=True, norm_first=True, activation="gelu")
                for _ in range(cfg.reader_layers))
            self.read_out = nn.Linear(cfg.d_read, cfg.d_llama)

    # ── routing ────────────────────────────────────────────────────────────────
    def _route_temp(self, l: int) -> Tensor:
        return self.log_route_temp[l].clamp(math.log(0.02), math.log(20.0)).exp()

    @torch.no_grad()
    def _calibrate_route_temps(self, n_query: int = 4096, iters: int = 30):
        for l, keys in enumerate(self.node_keys):
            q = _unit(torch.randn(n_query, self.config.d_code))
            logits = q @ _unit(keys.float()).t()
            lo, hi = math.log(0.02), math.log(20.0)
            target = math.log(min(max(self.config.effective_k, 1.0), keys.shape[0]))
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                p = torch.softmax(logits / math.exp(mid), dim=-1)
                ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean().item()
                lo, hi = (mid, hi) if ent < target else (lo, mid)
            self.log_route_temp.data[l] = 0.5 * (lo + hi)

    def route(self, l: int, x: Tensor) -> Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = _unit(self.route_projs[l](x.float()))
            k = _unit(self.node_keys[l].float())
            logits = torch.einsum("bld,nd->bln", q, k)
            return torch.softmax(logits / self._route_temp(l), dim=-1)

    def _perturb(self, l: int, x: Tensor, scores: Tensor) -> Tensor:
        k = min(self.config.top_k, scores.shape[-1])
        topv, topi = scores.topk(k, dim=-1)
        vals = self.node_values[l].to(x.dtype)
        add = (topv.unsqueeze(-1) * vals[topi]).sum(dim=2)
        return _unit_rms(x + self.perturb_gate[l].to(x.dtype) * add)

    # ── phase 1: signal processing (shared by v1/v2) ─────────────────────────────
    def _phase1(self, hiddens: Tensor, mask: Tensor):
        """Returns A_list[l] = routing scores [B,L,N_l] (mask-zeroed), per-node
        bias-corrected marginal margs[l] [N_l], and the per-layer code stream
        (unused downstream but kept for telemetry)."""
        cfg = self.config
        m = mask.float().unsqueeze(-1)
        n_tok = mask.float().sum(1, keepdim=True).clamp_min(1.0)
        x = _unit_rms(self.seed_proj(hiddens.float())) * m
        if self.training:
            self.marg_count += 1
        bias_corr = (1.0 - cfg.ema_decay ** self.marg_count.clamp_min(1.0)).clamp_min(1e-6)

        A_list, margs, aux = [], [], {}
        for l in range(self.depth):
            scores = self.route(l, hiddens.float() if l == 0 else x) * m   # [B,L,N]
            mean_act = scores.sum(dim=1) / n_tok
            marg_buf = getattr(self, f"act_marginal_L{l}")
            if self.training:
                with torch.no_grad():
                    marg_buf.mul_(cfg.ema_decay).add_(mean_act.mean(0), alpha=1 - cfg.ema_decay)
            marg = (marg_buf / bias_corr).clamp_min(1e-9)
            A_list.append(scores)
            margs.append(marg)
            if l < self.depth - 1:
                x = self._perturb(l, x, scores) * m
            with torch.no_grad():
                p = scores.clamp_min(1e-12)
                aux[f"graph_v9_route_entropy_L{l}"] = ((-(p * p.log()).sum(-1)) * mask).sum() / mask.sum().clamp_min(1.0)
                ms = mean_act
                aux[f"graph_v9_hub_share_L{l}"] = (ms / ms.sum(-1, keepdim=True).clamp_min(1e-9)).max(-1).values.mean()
                aux[f"graph_v9_coverage_L{l}"] = torch.tensor(
                    scores.argmax(-1)[mask.bool()].unique().numel() / scores.shape[-1], device=hiddens.device)
        return A_list, margs, aux

    @staticmethod
    def _decay_kernel(L: int, tau: Tensor, device) -> Tensor:
        """Causal STDP kernel W[t,u] = exp(−(u−t)/τ) for u>t else 0. [L,L]."""
        t = torch.arange(L, device=device)
        delta = (t.view(1, L) - t.view(L, 1)).float()          # u−t
        return torch.where(delta > 0, torch.exp(-delta / tau.clamp_min(0.1)), torch.zeros_like(delta))

    # ── v2 forward ───────────────────────────────────────────────────────────────
    def _forward_v2(self, hiddens: Tensor, mask: Tensor) -> tuple[Tensor, dict]:
        cfg = self.config
        B, L = hiddens.shape[:2]
        device = hiddens.device
        d_llama = cfg.d_llama
        if L > 1024:   # STDP is O(L^2) — guard until a streaming recurrence exists [merge #2]
            raise ValueError(
                f"graph_v9 v2 STDP builds an [L,L] kernel (L={L}); guarded at 1024. "
                "Use sentence_mae (L<=128); long-context needs the streaming-recurrence STDP.")
        A_list, margs, aux = self._phase1(hiddens, mask)
        P = min(cfg.edge_topP, *cfg.nodes)
        E = self.n_edges

        Cand = min(cfg.edge_cand, len(self.edge_sources) * P * P)
        # Phase 2 (STDP + edge selection) in float32 — selection must be stable.
        with torch.autocast(device_type=device.type, enabled=False):
            # prefilter: top-P ACTIVE nodes per layer (edges form among active concepts)
            AP, valtokP, margP = [], [], []
            for l in range(self.depth):
                act = A_list[l].float().sum(dim=1)             # [B,N] total activation mass
                ti = act.topk(P, dim=1).indices                # [B,P]
                AP.append(torch.gather(A_list[l].float(), 2, ti.unsqueeze(1).expand(B, L, P)))
                valtokP.append(self.emit_projs[l](self.node_values[l][ti].float()))   # [B,P,d_llama]
                margP.append(margs[l].float()[ti])                                   # [B,P]
            valstack = torch.stack(valtokP, dim=0)             # [depth,B,P,d_llama]
            src_layers = torch.tensor([s for s, _ in self.edge_sources], device=device)
            dst_layers = torch.tensor([d for _, d in self.edge_sources], device=device)

            # unified STDP co-activation per directed source -> lift (the cheap prefilter
            # signal AND a feature for the learned selector). τ learnable per source.
            eye = torch.eye(P, device=device, dtype=torch.bool)
            C_list, lift_list = [], []
            for (ls, ld) in self.edge_sources:
                tau = self.log_tau_within[ls].exp() if ls == ld else self.log_tau_inter[ls].exp()
                W = self._decay_kernel(L, tau, device)         # [L,L]
                C = torch.einsum("bti,tu,buj->bij", AP[ls], W, AP[ld])    # [B,P,P]
                lift = (C / (margP[ls].unsqueeze(2) * margP[ld].unsqueeze(1)).clamp_min(1e-9)).clamp_min(1e-9).log()
                if ls == ld:                                   # mask self-loops i->i [merge #3]
                    lift = lift.masked_fill(eye, -1e30)
                C_list.append(C); lift_list.append(lift)
            all_C = torch.stack(C_list, dim=1).reshape(B, -1)          # [B, S*P*P]
            all_lift = torch.stack(lift_list, dim=1).reshape(B, -1)    # [B, S*P*P] (prefilter score)
            PP = P * P

            # cheap prefilter: top-Cand candidate edges by STDP lift (recall step)
            cand_lift, cand = all_lift.topk(Cand, dim=1)       # [B,Cand]
            cs = cand // PP; cij = cand % PP; ci = cij // P; cj = cij % P
            srcLc = src_layers[cs]; dstLc = dst_layers[cs]     # [B,Cand]
            bC = torch.arange(B, device=device).unsqueeze(1).expand(B, Cand)
            src_val = valstack[srcLc, bC, ci]                  # [B,Cand,d_llama]
            dst_val = valstack[dstLc, bC, cj]
            log_C = torch.gather(all_C.clamp_min(1e-9).log(), 1, cand)
            feats = torch.stack([cand_lift, log_C, (srcLc != dstLc).float()], -1)   # [B,Cand,3]

            # context-aware SELECTOR: transformer over the candidate edges. Each edge
            # is scored WITH attention to all the others (whole-list view), so it can
            # learn the grammar + non-redundant coverage instead of independent scores.
            e = self.edge_in(torch.cat([_unit(src_val), _unit(dst_val), feats], dim=-1))
            for layer in self.selector:
                e = layer(e)                                   # bidirectional (sees all candidates)
            keep_logit = self.sel_head(e).squeeze(-1)          # [B,Cand]

            # STE soft-top-k: hard top-E forward (discrete graph), softmax-over-Cand
            # backward so reconstruction can push a near-miss candidate over a chosen one.
            temp = self.edge_score_temp.exp().clamp(0.1, 10.0)
            keep_soft = torch.softmax(keep_logit / temp, dim=1)               # [B,Cand]
            _, top = keep_logit.topk(E, dim=1)                 # [B,E] best-first
            p_sel = torch.gather(keep_soft, 1, top)            # [B,E]
            edge_w = 1.0 + p_sel - p_sel.detach()              # STE: fwd≈1, grad=∂(softmax over Cand)

            # gather the selected edges' endpoints (back to global layer/local index)
            bE = torch.arange(B, device=device).unsqueeze(1).expand(B, E)
            srcL = torch.gather(srcLc, 1, top); dstL = torch.gather(dstLc, 1, top)
            i_idx = torch.gather(ci, 1, top); j_idx = torch.gather(cj, 1, top)
            s_val = valstack[srcL, bE, i_idx]                  # [B,E,d_llama]
            d_val = valstack[dstL, bE, j_idx]
            # contextualized edge summary from the selector (puts it in the content
            # path -> strong gradient to selector + STDP grammar). Shared by the pair.
            e_sel = torch.gather(e, 1, top.unsqueeze(-1).expand(-1, -1, cfg.d_sel))   # [B,E,d_sel]
            edge_ctx = self.sel_to_tok(e_sel)                  # [B,E,d_llama]

            # node-tokens: [role, value, tag, edge-context], emitted src-then-dst.
            tags = torch.arange(E, device=device)              # one tag per edge
            src_tok = (_unit(s_val) + self.role_emb[0] + self.tag_emb[tags] + edge_ctx) * edge_w.unsqueeze(-1)
            dst_tok = (_unit(d_val) + self.role_emb[1] + self.tag_emb[tags] + edge_ctx) * edge_w.unsqueeze(-1)
            tokens = torch.stack([src_tok, dst_tok], dim=2).reshape(B, 2 * E, d_llama)

            # dedicated graph reader — CAUSAL mask so the first-k memory tokens form a
            # valid PREFIX code (the harness slices to k AFTER). Bidirectional mixing
            # would let token 0 see all m_max=16 graph tokens, so the k-token budget
            # would secretly encode the full graph = capacity LEAK vs the causal-LM
            # baselines (whose appended slots are prefix-stable too). [merge #1 CRITICAL]
            Mtok = tokens.shape[1]
            causal = torch.triu(torch.full((Mtok, Mtok), float("-inf"), device=device), 1)
            h = self.read_in(tokens)
            for layer in self.reader:
                h = layer(h, src_mask=causal)
            memory = self.token_norm(self.read_out(h))         # [B, 2E=m_max, d_llama]

        with torch.no_grad():
            # node identity = (layer, local index) [merge #8]
            uniq = torch.tensor(
                [torch.cat([srcL[b] * 1000 + i_idx[b], dstL[b] * 1000 + j_idx[b]]).unique().numel()
                 for b in range(B)], device=device, dtype=torch.float).mean()
            aux["graph_v9_edge_w_mean"] = edge_w.mean()
            aux["graph_v9_edge_w_spread"] = edge_w.std()
            aux["graph_v9_edge_inter_frac"] = (srcL != dstL).float().mean()
            aux["graph_v9_self_loop_frac"] = ((srcL == dstL) & (i_idx == j_idx)).float().mean()  # [merge #3 telem]
            aux["graph_v9_edge_uniq_nodes"] = uniq
            aux["graph_v9_keep_logit_spread"] = keep_logit.std()   # selector discrimination
            aux["graph_v9_tau_within"] = self.log_tau_within.exp().mean()
            aux["graph_v9_memory_norm"] = memory.float().norm(dim=-1).mean()
            # multi-element diagnostics (dropped by the JSONL logger; read by _v9v2_diag)
            aux["_sel_i"] = i_idx; aux["_sel_j"] = j_idx
            aux["_sel_srcL"] = srcL; aux["_sel_dstL"] = dstL; aux["_edge_w"] = edge_w
        return memory, aux

    # ── v1 forward (nodes-only ablation) ────────────────────────────────────────
    def _forward_v1(self, hiddens: Tensor, mask: Tensor) -> tuple[Tensor, dict]:
        cfg = self.config
        B, L = hiddens.shape[:2]
        m = mask.float().unsqueeze(-1)
        n_tok = mask.float().sum(1, keepdim=True).clamp_min(1.0)
        x = _unit_rms(self.seed_proj(hiddens.float())) * m
        if self.training:
            self.marg_count += 1
        bias_corr = (1.0 - cfg.ema_decay ** self.marg_count.clamp_min(1.0)).clamp_min(1e-6)

        cand_tokens, cand_presence, aux = [], [], {}
        for l in range(self.depth):
            scores = self.route(l, hiddens.float() if l == 0 else x) * m
            denom = scores.sum(dim=1).clamp_min(1e-6)
            centroid = torch.einsum("bln,bld->bnd", scores, x) / denom.unsqueeze(-1)
            mean_act = scores.sum(dim=1) / n_tok
            marg_buf = getattr(self, f"act_marginal_L{l}")
            if self.training:
                with torch.no_grad():
                    marg_buf.mul_(cfg.ema_decay).add_(mean_act.mean(0), alpha=1 - cfg.ema_decay)
            marg = (marg_buf / bias_corr).clamp_min(1e-9)
            lift = mean_act / marg
            presence = torch.sigmoid(self.presence_a[l] * lift + self.presence_b[l])
            token = self.emit_projs[l](centroid + self.node_values[l].unsqueeze(0))
            if l < self.depth - 1:
                x = self._perturb(l, x, scores) * m
            cand_tokens.append(token); cand_presence.append(presence)
        tokens = torch.cat(cand_tokens, dim=1)
        pres = torch.cat(cand_presence, dim=1)
        topp, topi = pres.topk(min(cfg.m_max, pres.shape[1]), dim=1)
        sel = tokens.gather(1, topi.unsqueeze(-1).expand(-1, -1, cfg.d_llama))
        memory = self.token_norm(sel) * topp.unsqueeze(-1)
        with torch.no_grad():
            aux["graph_v9_presence_spread"] = pres.std()
            aux["graph_v9_memory_norm"] = memory.float().norm(dim=-1).mean()
        return memory, aux

    def forward(self, hiddens: Tensor, mask: Tensor) -> tuple[Tensor, dict]:
        if self.config.use_graph:
            return self._forward_v2(hiddens, mask)
        return self._forward_v1(hiddens, mask)
