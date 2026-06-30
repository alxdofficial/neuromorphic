"""slotgraph v3.1 — graph memory with load-bearing binding. See docs/slotgraph_v3_design.md.

The memory is a graph-utterance in a learned VOCABULARY: N concept-nodes (the words) wired by E
edges (the sentences). Memorizing a passage = (a) tweak the contextual MEANING of every word
(soft, all nodes), then (b) draw edges committing to which words relate and how (hard selection).
A FROZEN SmolLM2 reads the result. Hardness is an EDGE thing; nodes are continuous vocabulary.

Each NODE splits into a fixed address + a writable meaning:
  • key_n   — fixed, learnable, input-INDEPENDENT. The address edges select against.
  • id_n    — fixed orthogonal tag (buffer). Identity for read-coreference (same word across edges).
  • value_n — learnable BASE meaning + bounded contextual delta. The word's current meaning.
Each EDGE = a writable state_e (the relation) + a hard-selected (src,dst) pair of nodes. No key of
its own; its identity is its endpoints.

WRITE (encoder, enc_layers, parallel within a passage). Per layer, two sub-steps. Both aggregate
  their source by COMPETITION (slot attention: softmax over the SLOT axis, not the source axis) — so
  slots compete for the content and each specializes to a DIFFERENT aspect. (Ordinary attention lets
  every slot independently pool the same content → redundant slots → hub-collapse, the failure the
  operator-read run hit: 144 edges all selecting ~6 nodes, rank-1 read = pooling.)
  Step A (nodes, SOFT): node tokens (value+id+role) COMPETE over the input passage → a head proposes
    a target meaning → DELTA-write toward it: value += β·(target−value). All nodes, no selection.
    β learnable + small (bounded drift = continuity).
  Step B (edges, HARD): edge tokens (state+role+endpoint-ids) COMPETE over input ∪ updated-nodes → a
    head delta-writes state, and q_src/q_dst (L2-normed, learnable temp) match the FIXED node keys
    → Gumbel straight-through (train) / hard argmax (eval) pick the two endpoints (dst≠src).
  Fixed key/id/role are re-injected every layer (stable addresses); only value/state are written.
  (Within-layer flow is input→nodes→edges; no node↔edge back-messaging — competition does the
  coordinating, and edges see the updated nodes in their own competition source.)

READ (decoder): edges only. Each edge → a STRUCTURED bound triple (concat into disjoint blocks,
NOT a sum, so A·rel·B ≠ B·rel·A): edge_vec = W_e·concat[state, value_src+id_src, value_dst+id_dst],
with endpoint values gathered FRESH from the final-layer node values via the hard onehots. Injected
into the frozen LM by per-layer GATED CROSS-ATTENTION (`GatedGraphXAttn`, installed by model.py).

Ablations: use_structure=False ⇒ read node values directly as a flat set (the membership control);
use_id=False ⇒ drop the orthogonal id tags.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...config import ReprConfig


def _participation_ratio(x: Tensor) -> float:
    x = x.detach().float()
    if x.shape[0] < 2:
        return 0.0
    xc = x - x.mean(0, keepdim=True); C = xc.t() @ xc
    return float((torch.diagonal(C).sum() ** 2 / (C * C).sum().clamp_min(1e-12)).item())


def _graph_components(src_row, dst_row, N):
    """Connected components of the undirected graph with edges (src_row[i], dst_row[i]) over N nodes,
    counting only nodes that appear as an endpoint. Returns (n_components, n_nodes_used). ~1 component
    = a connected graph (clauses link into one structure); ~E components = a bag of disconnected dyads
    (the 'separate clauses' failure)."""
    parent = list(range(N))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    used = set()
    for a, c in zip(src_row, dst_row):
        used.add(a); used.add(c)
        ra, rc = find(a), find(c)
        if ra != rc:
            parent[ra] = rc
    return len({find(u) for u in used}), len(used)


def _sinkhorn(log_alpha: Tensor, n_iter: int) -> Tensor:
    """Balanced assignment via Sinkhorn-Knopp in log-space. log_alpha [B,E,N] = (optionally
    Gumbel-perturbed) scores. Alternate COL(node)- then ROW(edge)-normalization: the col step caps how
    much total selection mass any one node absorbs (the anti-hub force the plain per-edge softmax lacks
    — it normalizes rows only, so a node can be picked by all E edges); ending on the row step leaves
    each edge a valid distribution over nodes for the argmax. E=N here ⇒ uniform marginals both ways.
    FEW iters = a gentle de-hub nudge (a node may still take several edges), not a rigid 1-1 matching."""
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)   # cols (nodes): cap usage
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)   # rows (edges): distribution
    return log_alpha.exp()


class GatedGraphXAttn(nn.Module):
    """Bottleneck gated cross-attention: a decoder hidden [B,S,d] reads the graph tokens G [B,U,dn].
    Attends IN the small graph space (down-project query d→dn, attend over G, up-project dn→d). The
    tanh gate is small +ve init → weak read from step 1 (the encoder is random, not pretrained, so a
    zero gate would deadlock the bootstrap). Returns the gated delta to ADD."""

    def __init__(self, d: int, dn: int, n_heads: int):
        super().__init__()
        assert dn % n_heads == 0, f"d_node {dn} must be divisible by xattn_heads {n_heads}"
        self.dn, self.h, self.hd = dn, n_heads, dn // n_heads
        self.q = nn.Linear(d, dn, bias=False)     # decoder hidden → graph space (down)
        self.k = nn.Linear(dn, dn, bias=False)
        self.v = nn.Linear(dn, dn, bias=False)
        self.o = nn.Linear(dn, d, bias=False)     # back up to d_llama
        self.gate = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, h: Tensor, G: Tensor, kv_keep: Tensor | None = None) -> Tensor:
        # h [B,S,d], G [B,U,dn], kv_keep [B,U] bool (True = attendable; None = all). Runs in the decoder's
        # autocast dtype (bf16) — no fp32 cast (the sensitive graph math is the encoder's fp32 job; a bare
        # .float() here is dead under the outer autocast). Mask is built in q.dtype so SDPA keeps its fast
        # (flash/mem-efficient) kernels instead of falling back to the math backend on a dtype mismatch.
        B, S, _ = h.shape; U = G.shape[1]
        q = self.q(h).view(B, S, self.h, self.hd).transpose(1, 2)         # [B,h,S,hd]
        k = self.k(G).view(B, U, self.h, self.hd).transpose(1, 2)         # [B,h,U,hd]
        v = self.v(G).view(B, U, self.h, self.hd).transpose(1, 2)
        attn_mask = None
        if kv_keep is not None:
            attn_mask = torch.zeros(B, 1, 1, U, device=h.device, dtype=q.dtype).masked_fill(
                ~kv_keep.view(B, 1, 1, U), float("-inf"))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)   # [B,h,S,hd]
        o = o.transpose(1, 2).reshape(B, S, self.dn)
        return (torch.tanh(self.gate) * self.o(o)).to(h.dtype)            # gated delta in d_llama


class _GTLayer(nn.Module):
    """One write layer: Step A (soft node meaning-tweak) → Step B (edge delta + hard endpoint
    selection). Each sub-step aggregates its source by COMPETITION (slot attention: softmax over the
    slot axis → slots specialize to distinct aspects) + pre-norm residual + FFN (skip+FFN mandatory —
    pure attention collapses to rank-1, Dong 2021). Deltas are bounded (learnable sigmoid β, small
    init)."""

    def __init__(self, dn: int, N: int, E: int, sinkhorn_iters: int = 3):
        super().__init__()
        self.N, self.E, self.dn, self.dk = N, E, dn, dn
        self.sinkhorn_iters = int(sinkhorn_iters)   # balanced-selection iters (gentle de-hub; see config)
        # Step A — node meaning update
        self.nqA = nn.LayerNorm(dn); self.nkA = nn.LayerNorm(dn)
        self.qA = nn.Linear(dn, dn); self.kA = nn.Linear(dn, dn)
        self.vA = nn.Linear(dn, dn); self.oA = nn.Linear(dn, dn)
        self.nffA = nn.LayerNorm(dn)
        self.ffA = nn.Sequential(nn.Linear(dn, 4 * dn), nn.GELU(), nn.Linear(4 * dn, dn))
        self.headA = nn.Linear(dn, dn)                    # proposes target node meaning
        # Step B — edge state update + endpoint selection
        self.nqB = nn.LayerNorm(dn); self.nkB = nn.LayerNorm(dn)
        self.qB = nn.Linear(dn, dn); self.kB = nn.Linear(dn, dn)
        self.vB = nn.Linear(dn, dn); self.oB = nn.Linear(dn, dn)
        self.nffB = nn.LayerNorm(dn)
        self.ffB = nn.Sequential(nn.Linear(dn, 4 * dn), nn.GELU(), nn.Linear(4 * dn, dn))
        self.headB = nn.Linear(dn, dn)                    # proposes Δstate target
        self.sel_norm = nn.LayerNorm(dn)                  # spread the selection queries (PKM)
        # bias-free: L2-norm makes magnitude irrelevant (only direction selects), and a shared bias
        # would tilt every edge's query toward the same keys. Default init: cosine is scale-invariant,
        # so init std only sets the backward 1/‖q‖ gradient scale — default keeps it O(1).
        self.q_src = nn.Linear(dn, self.dk, bias=False); self.q_dst = nn.Linear(dn, self.dk, bias=False)
        # learnable selection temperature (cosine→logit). Init √dk so the init logit spread
        # (sel_scale · cos-std ≈ √dk · 1/√dk = 1) ≈ the Gumbel(0,1) noise scale → signal and
        # exploration balanced. Plain scalar: the objective drives it up; saturating softmax caps it.
        self.sel_scale = nn.Parameter(torch.tensor(math.sqrt(self.dk)))
        # bounded delta rates (continuity): sigmoid(-1.2)≈0.23 — small tweak, not overwrite. learnable.
        self.beta_node = nn.Parameter(torch.tensor(-1.2))
        self.beta_edge = nn.Parameter(torch.tensor(-1.2))
        # post-delta LayerNorm on the PERSISTENT state (design §3.4 stability): re-standardize node_val /
        # edge_state after each delta so magnitude/distribution don't drift across the 4 layers (eff-rank
        # decays ~45→23 without it). Per-row norm preserves direction (the meaning); only scale is fixed.
        self.vnorm = nn.LayerNorm(dn); self.snorm = nn.LayerNorm(dn)

    def _compete(self, slot_q, src, qp, kp, vp, op, src_keep, eps=1e-6):
        """Competitive (slot-attention) write: the K slots COMPETE for the M source elements — softmax
        over the SLOT axis (dim=1), not the source axis — so each source element is assigned across the
        slots and each slot specializes to a distinct aspect. (Ordinary cross-attn softmaxes over the
        source, letting every slot independently grab the same content → redundancy.) Single-head for a
        coherent assignment. Scaled dot-product (1/√dn), as in slot attention: SHARPNESS is learned via
        the q/k projection magnitudes — NOT a learnable scalar temperature, which can't sharpen (its
        gradient is tiny and Adam moves a lone scalar only ~lr·steps; the prior cosine+temp selection
        confirmed this, sel_scale 8.00→8.02 over 4k steps ⇒ stuck diffuse). slot_q [B,K,dn] & src
        [B,M,dn] are pre-normed; src_keep [B,M] bool (True=valid). Returns the update."""
        q = qp(slot_q)                                                    # [B,K,dn]
        k = kp(src); v = vp(src)                                          # [B,M,dn]
        logits = torch.einsum("bkd,bmd->bkm", q, k) / math.sqrt(q.shape[-1])   # [B,K,M] scaled dot-prod
        attn = logits.softmax(dim=1)                                      # COMPETITION over the K slots
        if src_keep is not None:
            attn = attn * src_keep.unsqueeze(1).to(attn.dtype)           # drop padded sources [B,1,M]
        weights = attn / attn.sum(dim=-1, keepdim=True).clamp_min(eps)    # per-slot weighted mean over M
        upd = torch.einsum("bkm,bmd->bkd", weights, v)                   # [B,K,dn]
        return op(upd)

    def _balanced_pick(self, scores, tau):
        """One endpoint role (src or dst). Gumbel-Sinkhorn (Mena 2018): (train) add Gumbel noise →
        Sinkhorn-balance → soft assignment P; (eval) Sinkhorn, no noise. HARD one-hot per edge in the
        forward (argmax over nodes — binding-capable, no blend), straight-through the soft balanced P in
        the backward (smooth gradient + the col-cap anti-hub pressure). Returns hard [B,E,N], soft P."""
        log_alpha = scores / tau
        if self.training:
            u = torch.rand_like(log_alpha).clamp_(1e-9, 1.0)
            log_alpha = log_alpha - torch.log(-torch.log(u))          # + Gumbel(0,1) noise
        P = _sinkhorn(log_alpha, self.sinkhorn_iters)                 # [B,E,N] balanced soft
        idx = P.argmax(dim=-1, keepdim=True)
        hard = torch.zeros_like(P).scatter_(-1, idx, 1.0)
        return hard + (P - P.detach()), P                            # straight-through; soft for canaries

    def _select(self, e, node_key, tau):
        """Edge endpoint selection over the FIXED node keys: L2-normed cosine · temp → BALANCED
        Gumbel-Sinkhorn assignment (caps node over-subscription → breaks the hub-collapse), hard forward
        per edge. dst masked off src (no self-loops). fp32 (caller runs autocast-disabled). Returns hard
        src,dst [B,E,N] + the balanced soft assignments (for the routing/entropy canaries)."""
        qe = self.sel_norm(e)
        qs = F.normalize(self.q_src(qe), dim=-1, eps=1e-6)              # [B,E,dk]
        qd = F.normalize(self.q_dst(qe), dim=-1, eps=1e-6)
        kk = F.normalize(node_key, dim=-1, eps=1e-6).unsqueeze(0)      # [1,N,dk]
        sc_src = self.sel_scale * (qs @ kk.transpose(-1, -2))          # [B,E,N] cosine·temp (sel_scale = fixed Sinkhorn temp)
        src, soft_s = self._balanced_pick(sc_src, tau)
        sc_dst = self.sel_scale * (qd @ kk.transpose(-1, -2)) - 1e4 * src.detach()   # mask dst off src
        dst, soft_d = self._balanced_pick(sc_dst, tau)
        return src, dst, soft_s, soft_d

    def forward(self, node_val, edge_state, prev_src, prev_dst, ctx, ctx_keep,
                node_key, node_id, role, use_structure, tau):
        # role: [3,dn] → node / edge / input. id contributions from the PREVIOUS layer's edges.
        r_node, r_edge, r_in = role[0], role[1], role[2]
        if prev_src is not None:
            id_src = torch.einsum("ben,nd->bed", prev_src, node_id)
            id_dst = torch.einsum("ben,nd->bed", prev_dst, node_id)
        else:
            id_src = id_dst = torch.zeros_like(edge_state)
        input_tok = ctx + r_in                                          # [B,T,dn]
        B, N, E = node_val.shape[0], self.N, self.E

        # ── Step A: node meaning update — the N node-slots COMPETE over the input passage ──
        node_tok = node_val + node_id + r_node                          # [B,N,dn]
        a = node_tok + self._compete(self.nqA(node_tok), self.nkA(input_tok),
                                     self.qA, self.kA, self.vA, self.oA, ctx_keep)
        a = a + self.ffA(self.nffA(a))
        target_n = self.headA(a)
        node_val = self.vnorm(node_val + torch.sigmoid(self.beta_node) * (target_n - node_val))

        src = dst = soft_s = soft_d = None
        if not use_structure:
            return node_val, edge_state, src, dst, soft_s, soft_d

        # ── Step B: edge update + selection — the E edge-slots COMPETE over input ∪ UPDATED-nodes ──
        node_tok2 = node_val + node_id + r_node                         # [B,N,dn] (post Step-A)
        edge_tok2 = edge_state + r_edge + id_src + id_dst              # [B,E,dn]
        srcB = torch.cat([input_tok, node_tok2], dim=1)                 # [B,T+N,dn]
        keepB = torch.cat([ctx_keep, torch.ones(B, N, dtype=torch.bool, device=node_val.device)], dim=1)
        e = edge_tok2 + self._compete(self.nqB(edge_tok2), self.nkB(srcB),
                                      self.qB, self.kB, self.vB, self.oB, keepB)
        e = e + self.ffB(self.nffB(e))
        target_e = self.headB(e)
        edge_state = self.snorm(edge_state + torch.sigmoid(self.beta_edge) * (target_e - edge_state))
        src, dst, soft_s, soft_d = self._select(e, node_key, tau)
        return node_val, edge_state, src, dst, soft_s, soft_d


class SlotGraphEncoder(nn.Module):
    is_conditioned_read = False
    wants_surprise = False
    wants_prepend_refresh = False
    reinforce_prepend_each_layer = False
    wants_graph_xattn = True             # model.py installs per-layer gated cross-attn read hooks

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        from ...decoder import load_frozen_llama, apply_lora_to_llama
        base, _ = load_frozen_llama(cfg.llama_model)
        for p in base.parameters():
            p.requires_grad_(False)
        n_wrapped = apply_lora_to_llama(base, rank=cfg.slotgraph_lora_rank, alpha=cfg.slotgraph_lora_alpha,
                                        target_names=tuple(cfg.llama_lora_target_names))
        self.base = base
        d = cfg.d_llama
        dn = int(cfg.slotgraph_d_node)
        self.d, self.dn = d, dn
        self.N = int(cfg.slotgraph_n_nodes); self.E = int(cfg.slotgraph_n_edges)
        self.use_structure = bool(cfg.slotgraph_use_structure)
        self.use_id = bool(cfg.slotgraph_use_id)
        self.xattn_every = int(cfg.slotgraph_xattn_every)
        assert self.xattn_every >= 1, f"slotgraph_xattn_every must be >=1, got {self.xattn_every}"
        self.gumbel_tau = float(cfg.slotgraph_gumbel_tau_start)   # FIXED (not annealed); see config
        self.read_ablate = None   # diagnostic: None | "zero_values" (endpoint test) | "mean_state" (relation-type test)
        self.collect_layer_metrics = False  # diagnostic: per-write-layer depth-trace (val-only; adds a materialize/layer)

        # ── vocabulary substrate ── all init to norm≈1 (1/√dn) so value / id / role / key are balanced
        # in the node token. Learnable, so init only sets a balanced, dimension-scaled starting point.
        self.node_key = nn.Parameter(torch.randn(self.N, dn) / math.sqrt(dn))       # selection address
        self.node_val_base = nn.Parameter(torch.randn(self.N, dn) / math.sqrt(dn))  # base meaning
        self.edge_state_base = nn.Parameter(torch.randn(self.E, dn) / math.sqrt(dn))
        self.role = nn.Parameter(torch.randn(3, dn) / math.sqrt(dn))                # node / edge / input role
        # fixed orthogonal id tag (spread, unit-norm): identity for read-coreference. NOT trained
        # (the validated "free id-tag" win — 0 trainable params). N>dn ⇒ as-orthogonal-as-possible.
        nid = torch.empty(self.N, dn); nn.init.orthogonal_(nid)
        self.register_buffer("node_id", F.normalize(nid, dim=-1), persistent=True)

        # passage perception → graph space
        self.ctx_proj = nn.Linear(d, dn)
        self.in_norm = nn.LayerNorm(dn)
        self.gt_layers = nn.ModuleList([_GTLayer(dn, self.N, self.E, int(cfg.slotgraph_sinkhorn_iters))
                                        for _ in range(int(cfg.slotgraph_enc_layers))])
        # read materialization (relation-operator bind, "src rel dst"): rel acts as an OPERATOR on the
        # endpoints — out = W_o[(W_s·src) ⊙ (W_r·rel) ⊙ (W_d·dst)] — so zeroing the endpoints zeros the
        # output (no edge_state free-channel/bypass; bias-free ⇒ the multiplicative collapse is exact).
        # Directional via distinct W_s/W_d. Diagonal operator (TP-Transformer contraction); the low-rank
        # full-matrix W(rel) is the staged upgrade if relations must ROUTE, not just gate, meaning.
        self.bind_s = nn.Linear(dn, dn, bias=False)
        self.bind_r = nn.Linear(dn, dn, bias=False)
        self.bind_d = nn.Linear(dn, dn, bias=False)
        self.bind_o = nn.Linear(dn, dn, bias=False)
        self.out_norm = nn.LayerNorm(dn)

        # decoder read: one gated cross-attn per hooked decoder layer (owned here; applied via hooks)
        n_dec_layers = base.config.num_hidden_layers
        self._hook_layers = list(range(0, n_dec_layers, self.xattn_every))
        self.read_xattn = nn.ModuleList([GatedGraphXAttn(d, dn, cfg.slotgraph_xattn_heads)
                                         for _ in self._hook_layers])
        print(f"[slotgraph v3.1] {self.N} nodes (vocab) + {self.E} edges @ d_node={dn}; "
              f"{int(cfg.slotgraph_enc_layers)}-layer write (compete node-tweak + balanced Sinkhorn edge-select, "
              f"{int(cfg.slotgraph_sinkhorn_iters)} iters), "
              f"edges-only relation-operator read over {n_dec_layers} dec layers (every {self.xattn_every}), "
              f"encoder-LoRA r{cfg.slotgraph_lora_rank} ({n_wrapped} layers), "
              f"use_structure={self.use_structure}, use_id={self.use_id}")

    def train(self, mode: bool = True):
        super().train(mode); self.base.train(False); return self

    # ── streaming: accumulate passage embeds (icae pattern) ──
    def init_streaming_state(self, batch_size, device, dtype):
        return {"emb": torch.zeros(batch_size, 0, self.d, device=device, dtype=dtype),
                "mask": torch.zeros(batch_size, 0, device=device, dtype=torch.bool)}

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0, **extra):
        del chunk_offset, extra
        if attention_mask is None:
            attention_mask = torch.ones(token_embeds.shape[:2], device=token_embeds.device, dtype=torch.bool)
        return {"emb": torch.cat([state["emb"], token_embeds], dim=1),
                "mask": torch.cat([state["mask"], attention_mask.bool()], dim=1)}, {}

    def finalize_memory(self, state):
        emb, mask = state["emb"], state["mask"]                               # [B,T,d], [B,T]
        B = emb.shape[0]
        if emb.shape[1] == 0:
            raise ValueError("slotgraph.finalize_memory: empty context (T=0) — nothing to encode")
        attn = mask.long()
        H = self.base.model(inputs_embeds=emb, attention_mask=attn, use_cache=False).last_hidden_state
        idz = self.node_id if self.use_id else torch.zeros_like(self.node_id)
        with torch.autocast("cuda", enabled=False):                          # graph in fp32 (small + sensitive)
            ctx = self.in_norm(self.ctx_proj(H.float()))                     # [B,T,dn]
            node_val = self.node_val_base.float().unsqueeze(0).expand(B, -1, -1).contiguous()
            edge_state = self.edge_state_base.float().unsqueeze(0).expand(B, -1, -1).contiguous()
            node_key = self.node_key.float(); idn = idz.float(); role = self.role.float()
            src = dst = soft_s = soft_d = None
            layer_metrics = [] if self.collect_layer_metrics else None
            for layer in self.gt_layers:
                node_val, edge_state, src, dst, soft_s, soft_d = layer(
                    node_val, edge_state, src, dst, ctx, mask.bool(),
                    node_key, idn, role, self.use_structure, self.gumbel_tau)
                if layer_metrics is not None:
                    layer_metrics.append(self._layer_metrics(node_val, edge_state, src, dst, soft_s, idn))
            G = self._materialize(node_val, edge_state, src, dst, idn)
        memory = emb.new_zeros(B, 0, self.d)                                  # NO prepend; read = cross-attn
        aux = self._canaries(G, node_val, edge_state, src, dst, soft_s, soft_d, emb.device)
        if layer_metrics is not None:                                          # depth-trace: per-write-layer
            for li, lm in enumerate(layer_metrics):
                for k, v in lm.items():
                    aux[f"slotgraph_L{li}_{k}"] = torch.tensor(float(v), device=emb.device)
        aux["graph_G"] = G                                                    # consumed by the read hooks
        return memory, aux

    def _materialize(self, node_val, edge_state, src, dst, idn):
        """Build the read tokens. STRUCTURE: edges-only bound triples (concat disjoint blocks, ordered
        A·rel·B), endpoint values gathered FRESH from final node_val. FLAT control (no structure):
        read the node meanings directly as a set."""
        if not self.use_structure or src is None:
            return self.out_norm(node_val + idn.unsqueeze(0))                # [B,N,dn] flat set
        val_src = torch.einsum("ben,bnd->bed", src, node_val)               # [B,E,dn] meaning of A
        val_dst = torch.einsum("ben,bnd->bed", dst, node_val)               # meaning of B
        if self.read_ablate == "zero_values":                              # endpoint-binding test: zero the
            val_src = torch.zeros_like(val_src); val_dst = torch.zeros_like(val_dst)   # endpoints ⇒ bound=0
        elif self.read_ablate == "mean_state":                             # relation-TYPE test: replace the
            edge_state = edge_state.mean(dim=1, keepdim=True).expand_as(edge_state)    # specific rel w/ the mean
            # (under a multiplicative bind, ZEROING edge_state would just give bound=0 == zero_values, so to
            #  probe whether the relation TYPE matters we swap in a generic/mean relation instead.)
        # relation-operator bind: rel modulates the endpoints (src rel dst). Zero endpoints ⇒ zero ⇒ the
        # relation can carry nothing on its own (no bypass). Directional via distinct W_s/W_d.
        bound = self.bind_o(self.bind_s(val_src) * self.bind_r(edge_state) * self.bind_d(val_dst))
        # additive fixed id-tags = identity (NOT content), the connective tissue letting the LM link
        # clauses that share a node into a graph (vs a bag of separate triples).
        id_src = torch.einsum("ben,nd->bed", src, idn)
        id_dst = torch.einsum("ben,nd->bed", dst, idn)
        return self.out_norm(bound + id_src + id_dst)                       # [B,E,dn]

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        st = self.init_streaming_state(token_embeds.shape[0], token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)

    def node_keep_mask(self, B, device, training: bool):
        """KV-keep over the read tokens. Read is edges-only (structure) or node-set (flat); all kept
        (the v2 node-dropout anti-bypass is moot — the read no longer exposes raw node content)."""
        U = self.E if (self.use_structure) else self.N
        return torch.ones(B, U, dtype=torch.bool, device=device)

    @torch.no_grad()
    def _layer_metrics(self, node_val, edge_state, src, dst, soft_s, idn):
        """Per-write-layer depth-trace snapshot. Crucially `mem_effrank` = rank of the read tokens IF
        materialized from THIS layer's state → pinpoints the DEPTH at which the read collapses (rank-1 =
        pooling). nodes_used/avg_degree/n_components/src_entropy/routing_diversity = where selection
        collapses to a hub. Returns plain floats (caller tensors+prefixes them)."""
        m = {"node_effrank": _participation_ratio(node_val.reshape(-1, self.dn)),
             "edge_effrank": _participation_ratio(edge_state.reshape(-1, self.dn))}
        G = self._materialize(node_val, edge_state, src, dst, idn)
        m["mem_effrank"] = _participation_ratio(G.reshape(-1, G.shape[-1]))
        if src is not None:
            sp = src.argmax(-1); dp = dst.argmax(-1)
            m["src_entropy"] = float((-(soft_s.clamp_min(1e-9).log() * soft_s).sum(-1)).mean())
            if sp.shape[0] > 1:
                oh = F.one_hot(sp, self.N).float().mean(0)
                m["routing_diversity"] = float((-(oh.clamp_min(1e-9).log() * oh).sum(-1)).mean() / math.log(self.N))
            sp_l = sp.cpu().tolist(); dp_l = dp.cpu().tolist()
            comps, nused = [], []
            for b in range(len(sp_l)):
                c, u = _graph_components(sp_l[b], dp_l[b], self.N); comps.append(c); nused.append(u)
            mu = sum(nused) / len(nused)
            m["nodes_used"] = mu
            m["avg_degree"] = 2.0 * self.E / max(1.0, mu)
            m["n_components"] = sum(comps) / len(comps)
        return m

    @torch.no_grad()
    def _canaries(self, G, node_val, edge_state, src, dst, soft_s, soft_d, device):
        aux = {}
        aux["slotgraph_mem_effrank"] = torch.tensor(
            _participation_ratio(G.reshape(-1, G.shape[-1])), device=device)
        # the vocabulary-spread canary: did the node meanings stay distinct (v2 collapsed to ~2)?
        aux["slotgraph_node_effrank"] = torch.tensor(
            _participation_ratio(node_val.reshape(-1, self.dn)), device=device)
        aux["slotgraph_edge_effrank"] = torch.tensor(
            _participation_ratio(edge_state.reshape(-1, self.dn)), device=device)
        aux["slotgraph_key_effrank"] = torch.tensor(_participation_ratio(self.node_key), device=device)
        aux["slotgraph_node_norm"] = node_val.float().norm(dim=-1).mean()    # post-delta-norm ⇒ ≈√dn, stable
        aux["slotgraph_edge_norm"] = edge_state.float().norm(dim=-1).mean()
        aux["slotgraph_edge_frac"] = torch.tensor(float(self.E) / (self.N + self.E), device=device)
        aux["slotgraph_read_gate"] = torch.stack([x.gate for x in self.read_xattn]).detach().tanh().mean()
        aux["slotgraph_gumbel_tau"] = torch.tensor(float(self.gumbel_tau), device=device)
        aux["slotgraph_beta_node"] = torch.sigmoid(self.gt_layers[0].beta_node).detach()
        aux["slotgraph_beta_edge"] = torch.sigmoid(self.gt_layers[0].beta_edge).detach()
        if src is not None:
            N = self.N
            sp = src.argmax(-1); dp = dst.argmax(-1)                          # [B,E]
            aux["slotgraph_src"] = sp; aux["slotgraph_dst"] = dp
            aux["slotgraph_selfloop_frac"] = (sp == dp).float().mean()
            # within-batch cross-input routing diversity (THE topology signal): ↑ ⇒ graph responds to input
            if sp.shape[0] > 1:
                oh = F.one_hot(sp, N).float().mean(0)                         # [E,N] src-pick freq over batch
                aux["slotgraph_routing_diversity"] = (
                    -(oh.clamp_min(1e-9).log() * oh).sum(-1)).mean() / math.log(N)
            def _ent(p):
                return (-(p.clamp_min(1e-9).log() * p).sum(-1)).mean()
            aux["slotgraph_src_entropy"] = _ent(soft_s); aux["slotgraph_dst_entropy"] = _ent(soft_d)
            aux["slotgraph_endpoint_entropy_max"] = torch.tensor(math.log(N), device=device)
            use = torch.bincount(torch.cat([sp.reshape(-1), dp.reshape(-1)]), minlength=N).float()
            pu = use / use.sum().clamp_min(1e-9)
            aux["slotgraph_node_entropy"] = -(pu.clamp_min(1e-9).log() * pu).sum()  # node-usage (collapse canary)
            aux["slotgraph_sel_scale"] = self.gt_layers[-1].sel_scale.detach()
            # graph CONNECTEDNESS (the "clauses connect, not separate" criterion): node reuse + components
            sp_l = sp.cpu().tolist(); dp_l = dp.cpu().tolist()
            comps, nused = [], []
            for b in range(len(sp_l)):
                c, u = _graph_components(sp_l[b], dp_l[b], N)
                comps.append(c); nused.append(u)
            mean_used = sum(nused) / len(nused)
            aux["slotgraph_n_components"] = torch.tensor(sum(comps) / len(comps), device=device)  # ↓ = connected
            aux["slotgraph_nodes_used"] = torch.tensor(mean_used, device=device)
            aux["slotgraph_avg_degree"] = torch.tensor(2.0 * self.E / max(1.0, mean_used), device=device)  # ↑ = reuse
        return aux
