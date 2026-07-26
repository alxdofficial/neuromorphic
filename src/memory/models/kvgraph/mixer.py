"""Stage 6 — TokenGT mixer with relation-indexed operators.

Nodes AND edges are tokens (Kim et al. 2022, *Pure Transformers are Powerful Graph Learners*):

    node token  v      = [ node_vec_v , P_v , P_v ] + type_NODE
    edge token (u,v)   = [ edge_vec_uv, P_u , P_v ] + type_EDGE

Structure travels by INCIDENCE, not by a global positional encoding. An edge token literally contains its
endpoints' identifiers, so the topology is reconstructible from the token set alone — evict half the graph
and every surviving token's identifiers are unchanged. A Laplacian embedding would shift its whole spectrum
on any edit, which is disqualifying for a memory that is constantly being contracted.

Identifiers are random orthonormal and **resampled every forward**. That is what makes this different from
the retired slotgraph's "id-tags are free" failure: an identifier carries no information by itself, and
resampling stops the model memorising particular ones. The information is in the PATTERN OF REUSE — `P_u`
appearing in one node token and three edge tokens *is* the statement that those edges are incident to it.

Each layer is a Graph-Networks block: an edge update (the attention, which sees both endpoints) followed by
a node update whose message is a **relation-indexed operator**

    m_{j->i} = R_{rel(ij)} h_j ,   R_r = diag(d_r) + U_r diag(g(e_ij)) V_r^T

`U_r, V_r` are **per-relation**, not shared. This matters more than it looks: with a shared basis the only
thing distinguishing two relations is a DIAGONAL, which can rescale dimensions but cannot mix them — so
`agent` and `theme` could never route information along different subspaces, and the edge-typing ablation
would come back flat for a reason about this file rather than about the hypothesis.

`g` is a per-edge gate bounding how far an individual edge deviates from its relation's generic behaviour.
Zero-initialised `U_r` makes the correction exactly zero at step 0 — the operator *is* its relation's
diagonal — while still admitting gradient (`dL/dU_r` depends on `V_r` and `g`, both non-zero). Initialising
the GATE to zero instead would look equivalent and is not: it kills the gradient to `U_r` and `V_r` too, and
the low-rank branch never wakes up.

The ablation is TWO-SIDED and both switches live here. `ablate_relations` **averages the per-relation
tensors across relations** rather than dropping them, so every parameter stays in the graph and stays
trained — the control is capacity-matched, and a win cannot be explained by "the full arm simply had more
parameters". `ablate_edge_vec` zeroes the low-rank correction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import Relation

N_RELATIONS = len(Relation)


class TokenGTLayer(nn.Module):
    """Pre-norm self-attention + FFN over the combined node/edge token set."""

    def __init__(self, d: int, n_heads: int, ffn_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        return x + self.ffn(self.norm2(x))


class RelationMP(nn.Module):
    """Relation-indexed, multiplicative message passing with a bounded per-edge correction.

    Sum aggregation (injective, per GIN) rather than mean: mean-aggregation over-smooths hardest, and
    over-smoothing is the graph-native restatement of the representation collapse this project has already
    produced twice.
    """

    def __init__(self, d: int, rank: int = 8):
        super().__init__()
        self.rank = rank
        # Diagonal per relation, initialised to IDENTITY so an untrained mixer is a no-op, not noise.
        self.rel_diag = nn.Parameter(torch.ones(N_RELATIONS, d))
        # Per-relation low-rank bases. U zero-init => the correction is exactly 0 at step 0 while dL/dU is
        # still non-zero (it depends on V and the gate), so the branch bootstraps in one step. See module
        # docstring for why zero-initialising the gate instead would be a permanent kill.
        self.rel_U = nn.Parameter(torch.zeros(N_RELATIONS, d, rank))
        self.rel_V = nn.Parameter(torch.randn(N_RELATIONS, d, rank) / d ** 0.5)   # 1/sqrt(fan_in)
        self.gate = nn.Linear(d, rank)
        self.norm = nn.LayerNorm(d)
        #: Last forward's |low-rank term| / |diag term|. Not a buffer (never checkpointed) — it is the live
        #: readout of the two-sided ablation: climbing toward 1 means the continuous edge vector is carrying
        #: everything and the discrete relation has gone decorative.
        self.last_lowrank_frac: float = 0.0

    def forward(self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                rel: torch.Tensor, edge_vec: torch.Tensor | None,
                *, ablate_relations: bool = False, ablate_edge_vec: bool = False) -> torch.Tensor:
        if src.numel() == 0:
            return h
        if ablate_relations:
            # Capacity-matched control: average over the relation axis so every parameter is still present
            # and still trained, but the operator is forced to be relation-INVARIANT.
            diag = self.rel_diag.mean(0, keepdim=True).expand(rel.shape[0], -1)
            U = self.rel_U.mean(0, keepdim=True).expand(rel.shape[0], -1, -1)
            V = self.rel_V.mean(0, keepdim=True).expand(rel.shape[0], -1, -1)
        else:
            diag, U, V = self.rel_diag[rel], self.rel_U[rel], self.rel_V[rel]

        hs = h[src]                                                              # [E, d]
        base = diag * hs                                                         # typed, multiplicative
        msg = base
        if edge_vec is not None and not ablate_edge_vec:
            g = self.gate(edge_vec)                                              # [E, rank]
            proj = torch.einsum("ed,edr->er", hs, V) * g                         # bounded low-rank drift
            drift = torch.einsum("edr,er->ed", U, proj)
            msg = msg + drift
            with torch.no_grad():
                self.last_lowrank_frac = float(drift.norm() / base.norm().clamp_min(1e-6))
        else:
            self.last_lowrank_frac = 0.0
        agg = torch.zeros_like(h).index_add_(0, dst, msg)                        # SUM aggregation
        return h + self.norm(agg)                                                # residual: signal survives


class GraphMixer(nn.Module):
    """The main trainable module. Frozen LM in, mixed node vectors out.

    Works in a NARROWER space than the backbone (`d_mix < d_llama`). The mixer routes relations; it does not
    re-represent the LM's semantics. Both the quadratic token attention and the per-relation operator bank
    scale with this width, so it is the single highest-leverage sizing choice in the module.
    """

    def __init__(self, d_llama: int, d_mix: int = 384, n_layers: int = 4, n_heads: int = 6,
                 rank: int = 8, d_id: int = 64, ffn_mult: int = 2, center_inputs: bool = True):
        super().__init__()
        self.d_mix, self.d_id, self.center_inputs = d_mix, d_id, center_inputs
        self.node_in = nn.Linear(d_llama, d_mix)
        self.edge_in = nn.Linear(d_llama, d_mix)
        self.id_proj = nn.Linear(2 * d_id, d_mix, bias=False)
        self.type_emb = nn.Embedding(2, d_mix)              # 0 = node token, 1 = edge token
        self.rel_emb = nn.Embedding(N_RELATIONS, d_mix)
        self.layers = nn.ModuleList([TokenGTLayer(d_mix, n_heads, ffn_mult) for _ in range(n_layers)])
        self.mp = nn.ModuleList([RelationMP(d_mix, rank) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_mix)

    def _identifiers(self, n: int, device, dtype) -> torch.Tensor:
        """Random near-orthonormal node identifiers, resampled per forward.

        In `d_id` dimensions independent normalised Gaussians are near-orthogonal for n << exp(d_id), which
        holds comfortably at d_id=64 and a ~100-node budget. Cheaper than a QR and just as usable: all that
        matters is that attention can match a node token against the edge tokens sharing its identifier.
        """
        p = torch.randn(n, self.d_id, device=device, dtype=dtype)
        return F.normalize(p, dim=-1)

    def forward(self, node_vec: torch.Tensor, edge_vec: torch.Tensor | None,
                src: torch.Tensor, dst: torch.Tensor, rel: torch.Tensor,
                *, ablate_relations: bool = False, ablate_edge_vec: bool = False,
                ablate_graph: bool = False) -> torch.Tensor:
        """node_vec `[N, d_llama]`, edge_vec `[E, d_llama]` or None, src/dst/rel `[E]` -> `[N, d_mix]`.

        `ablate_graph` drops every edge — nodes still get projected and self-attend as a flat bank. That is
        the "does structure help at all?" control, and per the literature sweep it is the FIRST experiment,
        not the last.
        """
        n = node_vec.shape[0]
        if self.center_inputs and n > 1:
            # Subtract the mean across the read. LM hidden states are strongly ANISOTROPIC — a large shared
            # component dominates every vector (Ethayarajh 1909.00512 measures ~0.6 mean cosine between
            # uniformly random mid-layer tokens) — and the smoke run confirmed it here: raw node summaries
            # came out at effective rank ~1.0, i.e. the mixer was being handed almost no signal to route.
            # This is "All-but-the-Top" (Mu & Viswanath, ICLR'18), and it is the SAME centring fix the
            # bundling analysis independently arrived at. Per-vector LayerNorm would NOT do this: it
            # rescales each vector but leaves the shared direction intact.
            node_vec = node_vec - node_vec.mean(0, keepdim=True)
            if edge_vec is not None and edge_vec.shape[0] > 1:
                edge_vec = edge_vec - edge_vec.mean(0, keepdim=True)
        h = self.node_in(node_vec)
        if ablate_graph or src.numel() == 0:
            src = dst = torch.zeros(0, dtype=torch.long, device=h.device)
            rel = torch.zeros(0, dtype=torch.long, device=h.device)
            e_h = h.new_zeros(0, self.d_mix)
            e_raw = None
        else:
            e_raw = edge_vec
            e_h = self.edge_in(edge_vec) if edge_vec is not None else self.rel_emb(rel)
            e_h = e_h + self.rel_emb(rel)

        pid = self._identifiers(n, h.device, h.dtype)
        node_ids = self.id_proj(torch.cat([pid, pid], dim=-1))
        h = h + node_ids + self.type_emb.weight[0]
        if e_h.shape[0]:
            e_h = e_h + self.id_proj(torch.cat([pid[src], pid[dst]], dim=-1)) + self.type_emb.weight[1]

        e_mix = self.edge_in(e_raw) if (e_raw is not None and e_h.shape[0]) else None
        for layer, mp in zip(self.layers, self.mp):
            x = torch.cat([h, e_h], dim=0)[None]            # one "sequence" of node+edge tokens
            x = layer(x)[0]
            h, e_h = x[:n], x[n:]
            h = mp(h, src, dst, rel, e_mix,
                   ablate_relations=ablate_relations, ablate_edge_vec=ablate_edge_vec)
        return self.out_norm(h)
