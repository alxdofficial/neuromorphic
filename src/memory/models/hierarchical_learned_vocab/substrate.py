"""hierarchical_learned_vocab.py — Compression-by-Vocabulary (hlvocab).

Build spec: docs/compression_model_design.md (+ design-evolution notes there).
Compress a passage into its OWN learned vocabulary — a directed graph of
concept-nodes — and reconstruct via MAE.

v2 (use_graph=True, DEFAULT) — the full graph:
  Phase 1 (signal processing): tokens → frozen-LM hiddens → per-layer routing
    activations A_l [L×N_l]. Each token is RE-EXPRESSED purely as the blend of the
    nodes it activates — the input token is fully destroyed; only the node
    description survives to the next layer. This forces each layer into a genuinely
    more-abstract space and makes the slow node params data-load-bearing.
  Phase 2 (graph code): MULTI-SCALE STDP over the token axis, K learnable time
    scales τ_k, per directed node pair:
        C_k[i→j] = Σ_t Σ_{Δ>0} exp(−Δ/τ_k)·A[t,i]·A[t+Δ,j]
    within-layer = relational, inter-layer = compositional (low→high). The K lifts
    (above-chance co-activation) are features of each candidate edge.
  Phase 3 (selection + render + read): a transformer over candidate edges (whole-
    list view); E = m_max//2 edge-query slots each take a SHARP softmax over the
    candidates (soft + differentiable, near-one-hot ⇒ ≈ one edge — no hard top-k).
    Each selected edge renders as TWO stateless node-tokens [role(src/dst), value,
    instance-tag, edge-context]; a causal graph reader → M memory tokens (causal ⇒
    the first-k form a valid prefix code, no capacity leak). Norm-matched, prepended,
    MAE-CE → backprop updates everything.

v1 (use_graph=False) — nodes-only ablation: anti-hub activation-weighted node
  centroids, prepended directly (baseline decode path).

INTERFACE: forward(hiddens [B,L,d_model], mask [B,L]) → (memory [B, m_max, d_llama]
ranked best-first, aux); the harness slices the decoder-read to k = ceil(L/ratio).
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


class _NormMatch(nn.Module):
    """Put memory tokens in the decoder's token-embedding magnitude region
    (LayerNorm + L2-normalize + learnable scale; target set to embed norm later)."""

    def __init__(self, d: int, target: float = 0.9):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.scale = nn.Parameter(torch.tensor(float(target)))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.ln(x), dim=-1) * self.scale


@dataclass
class HLVocabConfig:
    d_model: int = 2048
    d_llama: int = 2048
    d_code: int = 256
    nodes: tuple = (512, 256, 128)
    top_k: int = 4
    m_max: int = 16
    effective_k: float = 8.0
    ema_decay: float = 0.99
    # ── v2 graph ────────────────────────────────────────────────────────────
    use_graph: bool = True
    edge_topP: int = 32             # node prefilter: edges among top-P active nodes/layer
    stdp_tau_init: tuple = (1.0, 2.0, 4.0)   # per-layer base decay (tokens; small->large)
    stdp_n_scales: int = 3          # K learnable τ-heads (multi-scale time)
    edge_cand: int = 48             # candidate edges after the cheap STDP-lift prefilter
    d_sel: int = 192                # candidate-edge selector width
    sel_layers: int = 2
    sel_heads: int = 4
    d_read: int = 192               # causal graph-reader width
    reader_layers: int = 2
    reader_heads: int = 4
    # emit read-out: "edge_query" = E independent sharp-softmax slots (collapse-prone);
    # "slotattn" = Slot-Attention competitive slots (softmax OVER slots → partition).
    emit: str = "edge_query"
    slot_iters: int = 3             # Slot-Attention refinement iterations (emit="slotattn")

    def __post_init__(self):
        if self.d_code % 32 != 0:
            raise ValueError(f"d_code={self.d_code} must be a multiple of 32")
        if self.use_graph:
            if self.m_max < 2:
                raise ValueError(f"use_graph needs m_max>=2 (edges are pairs); got {self.m_max}")
            if self.m_max % 2 != 0:
                # v2 emits 2 tokens per edge (src+dst) → 2*(m_max//2); an odd m_max
                # would silently emit m_max-1 tokens. Require even to avoid the surprise.
                raise ValueError(f"use_graph needs an EVEN m_max (emits 2 per edge); got {self.m_max}")
            if self.d_sel % self.sel_heads != 0:
                raise ValueError(f"d_sel={self.d_sel} % sel_heads={self.sel_heads} != 0")
            if self.d_read % self.reader_heads != 0:
                raise ValueError(f"d_read={self.d_read} % reader_heads={self.reader_heads} != 0")
            if len(self.stdp_tau_init) < len(self.nodes):
                raise ValueError(f"stdp_tau_init needs >= {len(self.nodes)} entries")
            if self.edge_cand < self.m_max // 2:
                raise ValueError(f"edge_cand={self.edge_cand} must be >= n_edges={self.m_max // 2}")


class _SlotAttentionEmit(nn.Module):
    """Slot-Attention competitive read-out (Locatello 2020). K slots compete for the
    candidate inputs via softmax-OVER-SLOTS, so they PARTITION the candidates instead
    of all collapsing onto the same one (the edge-query failure). Stochastic shared-
    Gaussian slot init (symmetry breaking) + GRU refinement. Returns the refined slot
    vectors and the per-slot assignment weights over inputs (rows ~sum-to-1 over N)."""

    def __init__(self, n_slots: int, d: int, iters: int = 3):
        super().__init__()
        self.n_slots, self.d, self.iters = n_slots, d, iters
        self.mu = nn.Parameter(torch.zeros(1, 1, d))
        self.log_sigma = nn.Parameter(torch.zeros(1, 1, d))     # σ=1 at init
        self.norm_in = nn.LayerNorm(d)
        self.norm_slots = nn.LayerNorm(d)
        self.norm_mlp = nn.LayerNorm(d)
        self.to_q = nn.Linear(d, d, bias=False)
        self.to_k = nn.Linear(d, d, bias=False)
        self.gru = nn.GRUCell(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.scale = d ** -0.5

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        B, N, _ = inputs.shape
        k = self.to_k(self.norm_in(inputs))                                  # [B,N,d]
        slots = self.mu + self.log_sigma.exp() * torch.randn(
            B, self.n_slots, self.d, device=inputs.device, dtype=inputs.dtype)
        attn = None
        for _ in range(self.iters):
            q = self.to_q(self.norm_slots(slots)) * self.scale               # [B,K,d]
            logits = torch.einsum("bkd,bnd->bkn", q, k)                      # [B,K,N]
            attn = logits.softmax(dim=1) + 1e-8                              # over SLOTS → competition
            w = attn / attn.sum(dim=-1, keepdim=True)                        # weighted-mean norm over N
            updates = torch.einsum("bkn,bnd->bkd", w, inputs)               # [B,K,d]
            slots = self.gru(updates.reshape(-1, self.d),
                             slots.reshape(-1, self.d)).reshape(B, self.n_slots, self.d)
            slots = slots + self.mlp(self.norm_mlp(slots))
        w = attn / attn.sum(dim=-1, keepdim=True)
        return slots, w


class HLVocabSubstrate(nn.Module):
    def __init__(self, config: HLVocabConfig):
        super().__init__()
        self.config = config
        cfg = config
        L = len(cfg.nodes)
        self.depth = L

        # ── slow vocabulary (small-init keys so cosine-unit direction moves) ────
        self.node_keys = nn.ParameterList(
            nn.Parameter(torch.randn(n, cfg.d_code) / math.sqrt(cfg.d_code)) for n in cfg.nodes)
        self.node_values = nn.ParameterList(
            nn.Parameter(torch.randn(n, cfg.d_code) / math.sqrt(cfg.d_code)) for n in cfg.nodes)
        self.route_projs = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.d_code, bias=False)]
            + [nn.Linear(cfg.d_code, cfg.d_code, bias=False) for _ in range(L - 1)])
        for lin in self.route_projs:
            nn.init.normal_(lin.weight, std=1.0 / math.sqrt(lin.in_features))
        if not cfg.use_graph:    # v1-only: the no-residual v2 stream comes from nodes, not a seed
            self.seed_proj = nn.Linear(cfg.d_model, cfg.d_code, bias=False)
            nn.init.normal_(self.seed_proj.weight, std=1.0 / math.sqrt(cfg.d_model))
        self.emit_projs = nn.ModuleList(
            nn.Linear(cfg.d_code, cfg.d_llama, bias=False) for _ in range(L))
        for lin in self.emit_projs:
            nn.init.normal_(lin.weight, std=1.0 / math.sqrt(cfg.d_code))
        self.token_norm = _NormMatch(cfg.d_llama)
        self.presence_a = nn.Parameter(torch.ones(L))      # v1-only
        self.presence_b = nn.Parameter(torch.zeros(L))

        self.log_route_temp = nn.Parameter(torch.zeros(L))
        self._calibrate_route_temps()
        self.register_buffer("marg_count", torch.zeros(()))
        for l, n in enumerate(cfg.nodes):
            self.register_buffer(f"act_marginal_L{l}", torch.zeros(n))

        # ── v2 graph machinery ─────────────────────────────────────────────────
        if cfg.use_graph:
            K = cfg.stdp_n_scales
            self.n_scales = K
            self.n_edges = cfg.m_max // 2
            base_w = torch.tensor([cfg.stdp_tau_init[l] for l in range(L)]).log().unsqueeze(1)
            base_i = torch.tensor([cfg.stdp_tau_init[l] for l in range(L - 1)]).log().unsqueeze(1)
            mult = torch.linspace(-1.0, 1.0, K).unsqueeze(0)
            self.log_tau_within = nn.Parameter(base_w + mult)        # [L, K]
            self.log_tau_inter = nn.Parameter(base_i + mult)        # [L-1, K]
            self.edge_sources = ([(l, l) for l in range(L)]
                                 + [(l, l + 1) for l in range(L - 1)])
            # candidate edge token: [unit(src_val), unit(dst_val), K lifts, is_inter]
            self.edge_in = nn.Linear(2 * cfg.d_llama + K + 1, cfg.d_sel)
            self.selector = nn.ModuleList(
                nn.TransformerEncoderLayer(
                    cfg.d_sel, cfg.sel_heads, dim_feedforward=4 * cfg.d_sel,
                    dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
                for _ in range(cfg.sel_layers))
            # emit read-out: edge_query (independent sharp-softmax, collapse-prone) OR
            # slotattn (Slot-Attention competition: slots PARTITION candidates).
            if cfg.emit == "slotattn":
                self.slot_emit = _SlotAttentionEmit(self.n_edges, cfg.d_sel, iters=cfg.slot_iters)
            else:
                self.edge_query = nn.Parameter(torch.randn(self.n_edges, cfg.d_sel) / math.sqrt(cfg.d_sel))
                self.sel_log_temp = nn.Parameter(torch.tensor(0.0))   # selection sharpness (learnable)
            # stateless TokenGT pieces: role(src/dst/standalone), instance-tags, edge-ctx
            self.role_emb = nn.Parameter(torch.randn(3, cfg.d_llama) / math.sqrt(cfg.d_llama))
            self.tag_emb = nn.Parameter(torch.randn(self.n_edges, cfg.d_llama) / math.sqrt(cfg.d_llama))
            self.sel_to_tok = nn.Linear(cfg.d_sel, cfg.d_llama)     # edge-context → token
            # causal graph reader (prefix-valid)
            self.read_in = nn.Linear(cfg.d_llama, cfg.d_read)
            self.reader = nn.ModuleList(
                nn.TransformerEncoderLayer(
                    cfg.d_read, cfg.reader_heads, dim_feedforward=4 * cfg.d_read,
                    dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
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
        """Re-express each token PURELY as its activated nodes' value-blend — the
        input token is fully destroyed; only the node description survives. Forces
        the next layer into a more-abstract space + data-dependence. [no residual]"""
        del x
        k = min(self.config.top_k, scores.shape[-1])
        topv, topi = scores.topk(k, dim=-1)
        add = (topv.unsqueeze(-1) * self.node_values[l].to(scores.dtype)[topi]).sum(dim=2)
        return _unit_rms(add)

    # ── phase 1 (shared) ─────────────────────────────────────────────────────────
    def _phase1(self, hiddens: Tensor, mask: Tensor):
        cfg = self.config
        m = mask.float().unsqueeze(-1)
        n_tok = mask.float().sum(1, keepdim=True).clamp_min(1.0)
        x = None   # v2 stream is regenerated each layer from node values (no seed/residual)
        if self.training:
            self.marg_count += 1
        bias_corr = (1.0 - cfg.ema_decay ** self.marg_count.clamp_min(1.0)).clamp_min(1e-6)

        A_list, margs, aux = [], [], {}
        for l in range(self.depth):
            scores = self.route(l, hiddens.float() if l == 0 else x) * m
            mean_act = scores.sum(dim=1) / n_tok
            marg_buf = getattr(self, f"act_marginal_L{l}")
            if self.training:
                with torch.no_grad():
                    marg_buf.mul_(cfg.ema_decay).add_(mean_act.mean(0), alpha=1 - cfg.ema_decay)
            A_list.append(scores)
            margs.append((marg_buf / bias_corr).clamp_min(1e-9))
            if l < self.depth - 1:
                x = self._perturb(l, x, scores) * m
            with torch.no_grad():
                p = scores.clamp_min(1e-12)
                aux[f"hlvocab_route_entropy_L{l}"] = ((-(p * p.log()).sum(-1)) * mask).sum() / mask.sum().clamp_min(1.0)
                ms = mean_act
                aux[f"hlvocab_hub_share_L{l}"] = (ms / ms.sum(-1, keepdim=True).clamp_min(1e-9)).max(-1).values.mean()
                aux[f"hlvocab_coverage_L{l}"] = torch.tensor(
                    scores.argmax(-1)[mask.bool()].unique().numel() / scores.shape[-1], device=hiddens.device)
        return A_list, margs, aux

    @staticmethod
    def _decay_kernel(L: int, tau: Tensor, device) -> Tensor:
        """Causal STDP kernel W[t,u] = exp(−(u−t)/τ) for u>t else 0. NaN-safe: clamp
        the exp argument ≤ 0 (else autograd evaluates exp(+Δ/τ)=inf on the masked
        non-causal half → inf·0 = NaN grad to τ via torch.where)."""
        t = torch.arange(L, device=device)
        delta = (t.view(1, L) - t.view(L, 1)).float()          # u−t at [t,u]
        W = torch.exp(-delta.clamp_min(0.0) / tau.clamp_min(0.1))   # arg ≤ 0 ⇒ finite
        return W * (delta > 0).float()                         # zero non-causal + diagonal

    # ── v2 forward ───────────────────────────────────────────────────────────────
    def _forward_v2(self, hiddens: Tensor, mask: Tensor) -> tuple[Tensor, dict]:
        cfg = self.config
        B, L = hiddens.shape[:2]
        device = hiddens.device
        d_llama = cfg.d_llama
        if L > 1024:
            raise ValueError(
                f"hlvocab v2 STDP builds an [L,L] kernel (L={L}); guarded at 1024. "
                "Use masked_reconstruction (L<=128); long-context needs a streaming-recurrence STDP.")
        A_list, margs, aux = self._phase1(hiddens, mask)
        P = min(cfg.edge_topP, *cfg.nodes)
        K = self.n_scales
        E = self.n_edges
        Cand = min(cfg.edge_cand, len(self.edge_sources) * P * P)

        with torch.autocast(device_type=device.type, enabled=False):
            # prefilter: top-P ACTIVE nodes per layer
            AP, valtokP, margP = [], [], []
            for l in range(self.depth):
                ti = A_list[l].float().sum(dim=1).topk(P, dim=1).indices
                AP.append(torch.gather(A_list[l].float(), 2, ti.unsqueeze(1).expand(B, L, P)))
                valtokP.append(self.emit_projs[l](self.node_values[l][ti].float()))
                margP.append(margs[l].float()[ti])
            valstack = torch.stack(valtokP, dim=0)             # [depth,B,P,d_llama]
            src_layers = torch.tensor([s for s, _ in self.edge_sources], device=device)
            dst_layers = torch.tensor([d for _, d in self.edge_sources], device=device)
            eye = torch.eye(P, device=device, dtype=torch.bool)

            # MULTI-SCALE STDP → K lifts per source
            src_lifts = []
            for (ls, ld) in self.edge_sources:
                lifts_k = []
                for kk in range(K):
                    tau = (self.log_tau_within[ls, kk] if ls == ld else self.log_tau_inter[ls, kk]).exp()
                    W = self._decay_kernel(L, tau, device)
                    C = torch.einsum("bti,tu,buj->bij", AP[ls], W, AP[ld])
                    lift = (C / (margP[ls].unsqueeze(2) * margP[ld].unsqueeze(1)).clamp_min(1e-9)).clamp_min(1e-9).log()
                    if ls == ld:
                        lift = lift.masked_fill(eye, -1e30)
                    lifts_k.append(lift)
                src_lifts.append(torch.stack(lifts_k, dim=-1))     # [B,P,P,K]
            all_lift = torch.stack(src_lifts, dim=1).reshape(B, -1, K)   # [B,S*P*P,K]
            PP = P * P

            # cheap prefilter: top-Cand candidates by max-over-scale lift
            prefilt = all_lift.max(dim=-1).values
            _, cand = prefilt.topk(Cand, dim=1)
            cand_lifts = torch.gather(all_lift, 1, cand.unsqueeze(-1).expand(B, Cand, K))
            cs = cand // PP; cij = cand % PP; ci = cij // P; cj = cij % P
            srcLc = src_layers[cs]; dstLc = dst_layers[cs]
            bC = torch.arange(B, device=device).unsqueeze(1).expand(B, Cand)
            src_val = valstack[srcLc, bC, ci]                  # [B,Cand,d_llama]
            dst_val = valstack[dstLc, bC, cj]
            feats = torch.cat([cand_lifts, (srcLc != dstLc).float().unsqueeze(-1)], dim=-1)

            # context-aware selector over candidates (whole-list view)
            R = self.edge_in(torch.cat([_unit(src_val), _unit(dst_val), feats], dim=-1))
            for layer in self.selector:
                R = layer(R)                                   # [B,Cand,d_sel]

            # emit read-out → per-slot assignment w [B,E,Cand] + edge context ctx_e
            if cfg.emit == "slotattn":
                slots, w = self.slot_emit(R)                   # competition: softmax OVER slots
                # slots are permutation-symmetric → mass-order so the prefix k-slice
                # takes the highest-claim edges first (prefix-valid causal read).
                order = w.sum(-1).argsort(dim=-1, descending=True)            # [B,E]
                w = torch.gather(w, 1, order.unsqueeze(-1).expand(-1, -1, w.shape[-1]))
                slots = torch.gather(slots, 1, order.unsqueeze(-1).expand(-1, -1, slots.shape[-1]))
                ctx_e = self.sel_to_tok(slots)
                temp = torch.tensor(0.0, device=device)        # n/a (telemetry placeholder)
            else:
                temp = self.sel_log_temp.exp().clamp(0.05, 5.0)
                logits = torch.einsum("ed,bcd->bec", self.edge_query, R) / math.sqrt(cfg.d_sel)
                w = torch.softmax(logits / temp, dim=-1)       # [B,E,Cand] sharp (per-slot, independent)
                ctx_e = self.sel_to_tok(torch.einsum("bec,bcd->bed", w, R))
            src_e = torch.einsum("bec,bcd->bed", w, src_val)   # blended edge endpoints
            dst_e = torch.einsum("bec,bcd->bed", w, dst_val)

            # stateless TokenGT node-tokens [role, value, tag, edge-ctx], src-then-dst
            tags = torch.arange(E, device=device)
            src_tok = _unit(src_e) + self.role_emb[0] + self.tag_emb[tags] + ctx_e
            dst_tok = _unit(dst_e) + self.role_emb[1] + self.tag_emb[tags] + ctx_e
            tokens = torch.stack([src_tok, dst_tok], dim=2).reshape(B, 2 * E, d_llama)

            # causal graph reader → prefix-valid memory
            Mtok = tokens.shape[1]
            causal = torch.triu(torch.full((Mtok, Mtok), float("-inf"), device=device), 1)
            h = self.read_in(tokens)
            for layer in self.reader:
                h = layer(h, src_mask=causal)
            memory = self.token_norm(self.read_out(h))         # [B,M,d_llama]

        with torch.no_grad():
            picked = w.argmax(-1)                              # [B,E] which candidate each slot favors
            ent = -(w.clamp_min(1e-9) * w.clamp_min(1e-9).log()).sum(-1).mean()
            inter_of_pick = (srcLc.gather(1, picked) != dstLc.gather(1, picked)).float().mean()
            aux["hlvocab_memory_norm"] = memory.float().norm(dim=-1).mean()
            aux["hlvocab_sel_temp"] = temp
            aux["hlvocab_sel_attn_entropy"] = ent             # low = sharp (≈ discrete pick)
            aux["hlvocab_sel_attn_max"] = w.max(-1).values.mean()
            aux["hlvocab_slot_uniq_edges"] = torch.tensor(
                [picked[b].unique().numel() for b in range(B)], device=device, dtype=torch.float).mean()
            aux["hlvocab_edge_inter_frac"] = inter_of_pick
            aux["hlvocab_tau_within"] = self.log_tau_within.exp().mean()
        return memory, aux

    # ── v1 forward (nodes-only ablation) ────────────────────────────────────────
    def _forward_v1(self, hiddens: Tensor, mask: Tensor) -> tuple[Tensor, dict]:
        cfg = self.config
        # v1 nodes-only path: seed_proj only exists when use_graph=False (built
        # conditionally in __init__). Guard so a mis-dispatch fails loudly here
        # rather than with an opaque AttributeError on self.seed_proj.
        assert not cfg.use_graph, "_forward_v1 is the v1 nodes-only path; use_graph=True must dispatch to _forward_v2"
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
            lift = mean_act / (marg_buf / bias_corr).clamp_min(1e-9)
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
            aux["hlvocab_presence_spread"] = pres.std()
            aux["hlvocab_memory_norm"] = memory.float().norm(dim=-1).mean()
        return memory, aux

    def forward(self, hiddens: Tensor, mask: Tensor) -> tuple[Tensor, dict]:
        if self.config.use_graph:
            return self._forward_v2(hiddens, mask)
        return self._forward_v1(hiddens, mask)
