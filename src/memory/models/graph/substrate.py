"""graph model substrate — relational-parser graph memory over a learnable node bank.

Design: docs/graph_model.md (SOURCE OF TRUTH). The model is a learned RELATIONAL
PARSER over a fixed learnable node bank (the vocabulary; replaces the VQ-VAE).

WRITE (GraphParser) — the working set is TWO copies of the E edges:
  • Part 1 = the CURRENT graph WITH values: per edge `[src_val, edge_state, dst_val]`
    + role + instance tag (+ a "current" part-marker). Window 1 → a learnable initial
    graph; window t+1 → the carried previous graph (persistence).
  • Part 2 = PREDICTION slots, NO values: per edge `[role_src, role_edge, role_dst]`
    + instance tag (+ a "prediction" part-marker).
Per layer (×write_layers, ≥3): the working set self-attends → cross-attends the
AVAILABLE NODES (all N, role="available") → cross-attends the OBSERVATION → FFN.
We then read the NEW graph off Part 2 only: a head per slot → src/dst SNAP to a bank
node (pointer-select, never regressed), edge_state from the edge slot itself. Reading
off fresh value-less slots (not in-place on Part 1) teaches active re-arrangement, not
copy-the-current-graph.

READ (GraphReader) — per edge bind op(src,dst,edge)→one vector, cross-attn inject
(RMS-matched, gated) into the frozen LLM at a mid-late layer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


def _rmsnorm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


class _EntmaxBisect(torch.autograd.Function):
    """α-entmax (Peters/Niculae/Martins 2019) via bisection on the threshold τ, with the
    EXACT entmax Jacobian in backward (no straight-through bias). α=1 → softmax (dense),
    α=2 → sparsemax, 1<α<2 → sparse-but-soft. p_i = [(α−1)z_i − τ]_+^{1/(α−1)}, Σp=1, so
    sub-threshold entries are EXACTLY zero — a genuine selection, not a blend. The model
    chooses how many survive (data-dependent support). α is a fixed hyperparameter here."""

    @staticmethod
    def forward(ctx, X, alpha, dim, n_iter):
        ctx.alpha = alpha; ctx.dim = dim
        d = X.shape[dim]
        Xs = X * (alpha - 1.0)                                   # work in (α−1)·z
        max_val, _ = Xs.max(dim=dim, keepdim=True)
        tau_lo = max_val - 1.0                                   # τ bounds (sum decreases as τ↑)
        tau_hi = max_val - ((1.0 / d) ** (alpha - 1.0))
        dm = tau_hi - tau_lo
        p = torch.clamp(Xs - tau_lo, min=0) ** (1.0 / (alpha - 1.0))
        for _ in range(n_iter):                                  # bisection → τ with Σp=1
            dm = dm / 2.0
            tau_m = tau_lo + dm
            p = torch.clamp(Xs - tau_m, min=0) ** (1.0 / (alpha - 1.0))
            f_m = p.sum(dim=dim, keepdim=True) - 1.0
            tau_lo = torch.where(f_m >= 0, tau_m, tau_lo)
        p = torch.clamp(Xs - tau_lo, min=0) ** (1.0 / (alpha - 1.0))
        p = p / p.sum(dim=dim, keepdim=True).clamp_min(1e-12)
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, dY):
        (p,) = ctx.saved_tensors
        gppr = torch.where(p > 0, p ** (2.0 - ctx.alpha), torch.zeros_like(p))  # 0 off-support
        dX = dY * gppr
        q = dX.sum(dim=ctx.dim, keepdim=True) / gppr.sum(dim=ctx.dim, keepdim=True).clamp_min(1e-12)
        dX = dX - q * gppr
        return dX, None, None, None


def entmax(X: Tensor, alpha: float = 1.5, dim: int = -1, n_iter: int = 30) -> Tensor:
    return _EntmaxBisect.apply(X, alpha, dim, n_iter)


@dataclass
class GraphConfig:
    d_llama: int = 576
    d_graph: int = 256               # graph/vocabulary space (decoupled from d_llama)
    n_nodes: int = 1024              # N — node bank size (the learnable vocabulary)
    n_edges: int = 16                # E — edge budget
    write_layers: int = 3            # parser depth (self → cross-nodes → cross-obs); ≥3
    read_layers: int = 2             # reader depth (cross-attend edges + causal self)
    heads: int = 4
    ffn_mult: int = 2
    ptr_logit_temp_init: float = 0.0  # pointer log-temp init (0 ⇒ temp=1; negative ⇒ sharper)
    entmax_alpha: float = 1.0         # selection sparsity: 1.0 ⇒ softmax (dense), 1.5/2.0 ⇒ sparse


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
            allmask = ~kv_mask.any(-1)                            # [B] True = no valid kv
            if allmask.any():                                    # avoid NaN softmax (all-padding window)
                scores = scores.masked_fill(allmask[:, None, None, None], 0.0)
        a = scores.softmax(-1)
        out = (a @ v).transpose(1, 2).reshape(B, Tq, self.h * self.dh)
        return self.o(out)


class _ParserBlock(nn.Module):
    """self-attend(working set) → cross-attend(available nodes) → cross-attend(obs) → FFN.
    nodes=None (the free-endpoint graph, no bank) skips the node cross-attention."""
    def __init__(self, d: int, heads: int, ffn_mult: int, use_nodes: bool = True):
        super().__init__()
        self.sn = nn.LayerNorm(d); self.slf = _Attn(d, heads)
        self.use_nodes = use_nodes
        if use_nodes:
            self.nn_norm = nn.LayerNorm(d); self.cross_nodes = _Attn(d, heads)
        self.on = nn.LayerNorm(d); self.cross_obs = _Attn(d, heads)
        self.fn = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, x: Tensor, nodes: Tensor, obs: Tensor, obs_mask: Tensor) -> Tensor:
        xn = self.sn(x); x = x + self.slf(xn, xn)                 # self-attend the working set
        if self.use_nodes and nodes is not None:
            x = x + self.cross_nodes(self.nn_norm(x), nodes)     # cross-attend available nodes (all valid)
        x = x + self.cross_obs(self.on(x), obs, kv_mask=obs_mask.bool())   # cross-attend observation
        x = x + self.ff(self.fn(x))
        return x


class _SelfBlock(nn.Module):
    """self-attend → FFN, pre-LN residual. The prepend memory-former: the E edge tokens
    contextualize each other (a set, no causal mask), then project to d_llama. No decoder
    cross-attn — the frozen LM reads the prepended tokens via its own attention."""
    def __init__(self, d: int, heads: int, ffn_mult: int):
        super().__init__()
        self.sn = nn.LayerNorm(d); self.slf = _Attn(d, heads)
        self.fn = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, x: Tensor) -> Tensor:
        xn = self.sn(x); x = x + self.slf(xn, xn)
        x = x + self.ff(self.fn(x))
        return x


# ── WRITE: learned relational parser over a learnable node bank ──────────────
class GraphParser(nn.Module):
    def __init__(self, cfg: GraphConfig):
        super().__init__()
        self.cfg = cfg
        N, E, d = cfg.n_nodes, cfg.n_edges, cfg.d_graph
        # the vocabulary: N learnable node vectors. Gradient-trained (not VQ-EMA).
        self.node_bank = nn.Parameter(torch.randn(N, d) / math.sqrt(d))
        self.bank_key = nn.Linear(d, d, bias=False)                # pointer match-keys (raw bank)
        self.node_role_avail = nn.Parameter(torch.randn(d) / math.sqrt(d))   # "available" role (cross-attn)
        self.obs_proj = nn.Linear(cfg.d_llama, d)                  # obs → d_graph (cross-attn)
        self.role = nn.Parameter(torch.randn(3, d) / math.sqrt(d))    # src / edge / dst role
        self.tag = nn.Parameter(torch.randn(E, d) / math.sqrt(d))     # per-edge instance tag
        self.part = nn.Parameter(torch.randn(2, d) / math.sqrt(d))    # current-graph vs prediction-slot
        self.init_graph = nn.Parameter(torch.randn(3, E, d) / math.sqrt(d))  # window-1 "initial graph" values
        self.blocks = nn.ModuleList(_ParserBlock(d, cfg.heads, cfg.ffn_mult) for _ in range(cfg.write_layers))
        self.q_src = nn.Linear(d, d, bias=False)                   # head: src/dst slot → pointer query
        self.q_dst = nn.Linear(d, d, bias=False)
        # learnable src/dst pointer sharpness; init from cfg (0 ⇒ temp=1; negative ⇒ sharper).
        self.log_temp = nn.Parameter(torch.full((2,), float(cfg.ptr_logit_temp_init)))
        self.edge_head = nn.Linear(d, d)                          # head: edge slot → edge_state

    def _point(self, q: Tensor, which: int) -> tuple[Tensor, Tensor]:
        """Snap: QK-RMSNorm + learnable-temp softmax over the bank → gather the RAW bank
        value (stable identity). Never regresses an endpoint. Returns (value, ptr)."""
        d = q.shape[-1]
        qn = _rmsnorm(q)                                           # [B,E,d]
        kn = _rmsnorm(self.bank_key(self.node_bank))             # [N,d]
        temp = self.log_temp[which].clamp(-3.0, 3.0).exp()
        scores = (qn @ kn.t()) * (d ** -0.5) / temp               # [B,E,N]
        if self.cfg.entmax_alpha > 1.0:                           # sparse selection (commits to few)
            ptr = entmax(scores, alpha=self.cfg.entmax_alpha, dim=-1)
        else:                                                     # α=1 ⇒ dense softmax (default)
            ptr = scores.softmax(-1)
        return ptr @ self.node_bank, ptr                          # gather raw bank (sharp → near-exact)

    def forward(self, obs: Tensor, obs_mask: Tensor, state: dict = None) -> dict:
        """Parse/UPDATE the graph. Working set = [Part 1: current graph WITH values ;
        Part 2: prediction slots, no values]; self-attend, cross-attend available nodes,
        cross-attend obs (×write_layers); predict the new graph off Part 2 (snap src/dst,
        regress edge_state). `state` = carried graph (None on window 1 → init_graph)."""
        B, E, d = obs.shape[0], self.cfg.n_edges, self.cfg.d_graph
        obs_kv = self.obs_proj(obs.float())                       # [B,T,d]
        nodes_kv = (self.node_bank + self.node_role_avail).unsqueeze(0).expand(B, -1, -1)  # [B,N,d]
        rt = self.role[:, None, :] + self.tag[None, :, :]        # [3,E,d] role+tag
        # Part 1 — current graph WITH values (window 1: learnable init; else carried)
        if state is None:
            p1 = (self.init_graph + rt + self.part[0]).reshape(3 * E, d).unsqueeze(0).expand(B, 3 * E, d).contiguous()
        else:
            vals = torch.stack([state["src_value"], state["edge_state"], state["dst_value"]], dim=1)  # [B,3,E,d]
            p1 = (vals + rt[None] + self.part[0]).reshape(B, 3 * E, d)
        # Part 2 — prediction slots, NO values
        p2 = (rt + self.part[1]).reshape(3 * E, d).unsqueeze(0).expand(B, 3 * E, d).contiguous()
        x = torch.cat([p1, p2], dim=1)                           # [B,6E,d]
        for blk in self.blocks:
            x = blk(x, nodes_kv, obs_kv, obs_mask)
        # predict the new graph off Part 2 (the value-less prediction slots)
        src_t, edge_t, dst_t = x[:, 3 * E:].view(B, 3, E, d).unbind(1)
        src_v, src_ptr = self._point(self.q_src(src_t), 0)
        dst_v, dst_ptr = self._point(self.q_dst(dst_t), 1)
        edge_state = self.edge_head(edge_t)
        return {"src_value": src_v, "dst_value": dst_v, "edge_state": edge_state,
                "src_ptr": src_ptr, "dst_ptr": dst_ptr}


# ── READ: per-edge FiLM-bound token → PREPEND memory (read by the decoder natively) ──
class GraphReader(nn.Module):
    """Forms the graph's MEMORY TOKENS for a PREPEND read — NOT a custom inject.

    Same FiLM token-creation as before: bind op(src,dst,edge)→one vector per edge
    (binding installed in the WRITE, the side-car lesson). The E edge tokens then
    self-attend to contextualize each other and project to d_llama; the frozen decoder
    reads them as M=E prepended slots via its OWN attention. This replaces the old
    cross-attention inject reader, whose injected signal collapsed to ≈rank-1 (the same
    additive nudge at every position) — see scripts/diagnostics/why_graph_collapses_mae.
    """
    def __init__(self, cfg: GraphConfig):
        super().__init__()
        self.cfg = cfg
        d, dl = cfg.d_graph, cfg.d_llama
        # per-edge bind op: bind the two endpoints, modulate by the relation (FiLM).
        # binding installed BEFORE the memory is read (the side-car lesson) — not pooling.
        self.w_sd = nn.Linear(2 * d, d)
        self.w_gamma = nn.Linear(d, d); self.w_beta = nn.Linear(d, d)
        # FiLM near-identity init: γ.bias=1, β.bias=0 anchors the bind (edge_vec≈sd at
        # start, so a random relation can't drown the endpoint binding — R1-L5), while
        # the (default-init, nonzero) weights keep edge_state effective AND gradient-
        # flowing. NB: zeroing the γ/β *weights* would give edge_state zero path to the
        # output → starve edge_head (the relation channel) of gradient — don't.
        nn.init.ones_(self.w_gamma.bias); nn.init.zeros_(self.w_beta.bias)
        # memory-former: the E edge tokens self-attend (reuses the read-layer budget so
        # the graph stays capacity-matched), then a LN + projection to d_llama.
        self.blocks = nn.ModuleList(_SelfBlock(d, cfg.heads, cfg.ffn_mult) for _ in range(cfg.read_layers))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, dl)

    def edge_tokens(self, graph: dict) -> Tensor:
        sd = self.w_sd(torch.cat([graph["src_value"], graph["dst_value"]], dim=-1))  # bind endpoints
        g = self.w_gamma(graph["edge_state"]); b = self.w_beta(graph["edge_state"])
        return g * sd + b                                          # [B,E,d] — relation FiLM-modulates the pair

    def forward(self, graph: dict) -> Tensor:
        """graph dict → [B, E, d_llama] memory tokens to PREPEND (decoder reads natively)."""
        x = self.edge_tokens(graph)                               # [B,E,d]
        for blk in self.blocks:
            x = blk(x)                                            # edges contextualize each other
        return self.out(self.norm(x))                            # [B,E,d_llama]
