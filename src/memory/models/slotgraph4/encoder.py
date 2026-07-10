"""slotgraph4 — fixed-topology sparse edge-state slot graph (d_node = d_llama = 576, d_edge small).

The free-invent graph memory done right (docs/slotgraph4_design.md): the routing head is DELETED. The graph
is N node slots + a FIXED k-regular small-world edge-state tensor `E:[N,k,d_e]` with a frozen neighbor table
`nbr:[N,k]` (Watts-Strogatz, degree-preserved). Every edge slot's endpoints are fixed by position, so there
is nothing to *choose* — the model learns only the edge STATES + the node ASSIGNMENT (which entity lands on
which slot, via competitive attention). "Which entities relate" is realized by the competitive write placing
related entities onto connected slots.

  INPUT   : the last-layer hiddens of a FROZEN LM over each 256-token window (no grad, no LoRA — the LM has
            already computed the in-context graph; memory's job is to persist + compress it).
  WRITE   : a few (recurrent) _SG4Block layers — graph tokens query [frozen text hiddens ; graph tokens]
            (self+cross fused in ONE SDPA; text is keys/values only, never re-written). ReZero zero-init
            residuals (identity-at-init) + competitive assignment (softmax over the SLOT axis).
            PROPOSE→COMMIT: the stack is pure scratch; persistence changes at ONE point per window —
              Δ      = o(stack output)                     # zero-init proj → proposal is a CHANGE (no-op@init)
              α, β   = decoupled retain/write gates([state‖Δ])
              state  = post_norm( α ⊙ state + β ⊙ Δ )       # bounded (EntNet norm), erase≠write (GDN-2)
            NO delta rule: addressing is positional (each edge is its own slot → cross-edge interference is
            structurally zero) and writes are read-modify-write (the merge happens in activation space), so
            the delta rule would degenerate to gated-interp here. Same commit rule for nodes AND edges.
  READ    : PREPEND (Set-LLM bidirectional) — node-centric tokens (each node ← attn-pool of [X_i ‖ its k
            edge states]) + top-k explicit edge tokens (learned salience) carrying endpoint-id pointers.

Fixes vs slotgraph3: routing collapse (no routing head), routing-absorption (graph is a separate channel),
dead-gradient materialization (none), N² blowup (k-regular sparse). Does NOT fix loss-neutrality — that is
the OBJECTIVE's job (behavioral-KL + exclusive channel + provenance-InfoNCE + bypass-gap; docs/OBJECTIVES.md).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt
from torch import Tensor

from ...common import _NormMatch
from ...config import ReprConfig


def _participation_ratio(x: Tensor) -> float:
    x = x.detach().float()
    if x.shape[0] < 2:
        return 0.0
    xc = x - x.mean(0, keepdim=True); C = xc.t() @ xc
    return float((torch.diagonal(C).sum() ** 2 / (C * C).sum().clamp_min(1e-12)).item())


def _watts_strogatz_directed(N: int, k: int, beta: float, seed: int) -> Tensor:
    """Degree-preserved (out-degree = k) directed small-world graph → neighbor table nbr:[N,k].

    Ring lattice: node i → {i+1, ..., i+k} (mod N). Then each out-edge is rewired with probability beta to a
    uniformly-random target that is neither i nor an existing neighbor of i (degree stays exactly k, no
    self-loops, no duplicate edges). Deterministic (seeded torch.Generator) — the topology is a FIXED buffer
    baked at construction; only the edge STATES are learned. beta=0 → pure ring lattice; beta=1 → random
    k-regular; 0<beta<1 → small-world (short paths ~log N)."""
    if not (1 <= k <= N - 1):
        raise ValueError(f"slotgraph4 needs 1 ≤ edges_per_node ≤ N-1 (N={N}, k={k})")
    g = torch.Generator().manual_seed(int(seed))
    nbr = torch.stack([(torch.arange(k) + i + 1) % N for i in range(N)], dim=0)  # [N,k] ring lattice
    for i in range(N):
        neigh = set(nbr[i].tolist())
        for m in range(k):
            if float(torch.rand((), generator=g)) >= beta:
                continue
            pool = [j for j in range(N) if j != i and j not in neigh]
            if not pool:
                continue
            neigh.discard(int(nbr[i, m]))
            new = pool[int(torch.randint(len(pool), (), generator=g))]
            nbr[i, m] = new
            neigh.add(new)
    return nbr.long()


class _SG4Block(nn.Module):
    """Pre-norm graph-mixer block (inherited from slotgraph3's `_SG3Block`). GRAPH tokens are the queries;
    keys/values = [frozen-LM window hiddens ; graph tokens] — self-attention among the graph + cross-attention
    to the contextualized text, fused into ONE SDPA. Text is static (keys/values only, never re-written).
    NO positional encoding: word order lives inside the LM hiddens' CONTENT; graph tokens are a SET identified
    by id/role embeddings. Residual output projections are ZERO-init (ReZero/Fixup) → exact identity at init;
    each block earns influence only as it trains, and the state highway is undisturbed until then."""
    def __init__(self, d: int, n_heads: int, d_ff: int):
        super().__init__()
        assert d % n_heads == 0, f"d={d} not divisible by n_heads={n_heads}"
        self.nh = n_heads; self.hd = d // n_heads
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)          # identity-at-init residual branches
        nn.init.zeros_(self.ff[2].weight); nn.init.zeros_(self.ff[2].bias)

    def forward(self, g: Tensor, ctx: Tensor, keep: Tensor) -> Tensor:
        # g:[B,G,d] graph tokens (queries)  ctx:[B,T,d] frozen window hiddens (keys/values only)
        # keep:[B,T+G] bool key-validity — graph slots are always valid, so no query row is ever fully masked
        B, G, d = g.shape
        h = self.ln1(torch.cat([ctx, g], dim=1))                           # shared pre-LN over keys/values
        q = self.q(h[:, ctx.shape[1]:]).view(B, G, self.nh, self.hd).transpose(1, 2)
        k = self.k(h).view(B, -1, self.nh, self.hd).transpose(1, 2)
        v = self.v(h).view(B, -1, self.nh, self.hd).transpose(1, 2)
        m = keep[:, None, None, :]                                         # bool mask, True = attend
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)           # SDPA does the 1/√hd scaling
        a = a.transpose(1, 2).reshape(B, G, d)
        g = g + self.o(a)
        g = g + self.ff(self.ln2(g))
        return g


class SlotGraph4Encoder(nn.Module):
    is_conditioned_read = False              # READ = PREPEND the node-centric + top-k edge tokens
    reads_per_layer_kv = False               # relational read needs intra-memory attention → prepend, not KV
    wants_surprise = False

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        from ...decoder import load_frozen_llama
        d = cfg.d_llama; self.d = d
        self.N = int(cfg.slotgraph4_n_nodes)
        self.k = int(cfg.slotgraph4_edges_per_node)
        self.de = int(cfg.slotgraph4_d_edge)
        self.window = int(cfg.slotgraph4_window)
        self.dk = int(cfg.slotgraph4_d_key)
        self.recurrent = bool(cfg.slotgraph4_recurrent)
        self.n_layers = int(cfg.slotgraph4_write_layers)
        self.read_topk = int(cfg.slotgraph4_read_topk)
        self.boundary_tokens = bool(cfg.slotgraph4_boundary_tokens)
        N, k, de = self.N, self.k, self.de
        h = d // 2                                                          # half-dim id block (TokenGT)
        if N > h:
            raise ValueError(f"slotgraph4 needs N ≤ d/2 orthonormal ids (N={N}, d/2={h})")

        # ── frozen LM window featurizer (no grad, no LoRA — a fixed encoder; the write stack is the learner) ──
        base, _ = load_frozen_llama(cfg.llama_model)
        for p in base.parameters():
            p.requires_grad_(False)
        embed = base.get_input_embeddings()
        with torch.no_grad():
            mean_vec = embed.weight.float().mean(0); emb_std = embed.weight.float().std().item()
            emb_norm = embed.weight.float().norm(dim=-1).mean().item()
        self.base = base

        # ── FIXED topology (Watts-Strogatz k-regular small-world) — buffers, never learned ──
        nbr = _watts_strogatz_directed(N, k, float(cfg.slotgraph4_ws_beta), int(cfg.slotgraph4_seed))
        self.register_buffer("nbr", nbr, persistent=True)                  # [N,k] dst index per edge slot
        src_idx = torch.arange(N).view(N, 1).expand(N, k).reshape(-1)      # [N*k] source node of each edge
        self.register_buffer("edge_src_idx", src_idx.long(), persistent=True)
        self.register_buffer("edge_dst_idx", nbr.reshape(-1).long(), persistent=True)

        # ── frozen orthonormal ids + type embeddings (the anti-collapse basis / EntNet reusable keys) ──
        idh = torch.empty(N, h); nn.init.orthogonal_(idh)
        self.register_buffer("id_half", F.normalize(idh, dim=-1), persistent=True)
        self.type_embed = nn.Parameter(torch.empty(3, h))                  # [node, edge, boundary]
        nn.init.orthogonal_(self.type_embed)
        with torch.no_grad():
            self.type_embed.copy_(F.normalize(self.type_embed, dim=-1))
        # ONE shared concat-then-project token former (TokenGT w_in): the id subspace lands in the SAME output
        # directions for node AND edge tokens, so attention matches an edge's endpoint ids against node tokens
        # by construction. Input = [content(d) ‖ id_a(h) ‖ id_b(h) ‖ type(h)] → d.
        self.tok_proj = nn.Linear(d + 3 * h, d)
        self.edge_up = nn.Linear(de, d)                                    # lift compact edge state → d for tokens

        # ── initial state (free latents; per-forward noise breaks slot symmetry — Slot Attention) ──
        self.node_init = nn.Parameter(mean_vec.view(1, d).repeat(N, 1) + emb_std * torch.randn(N, d))
        self.edge_init = nn.Parameter(torch.randn(N, k, de) / math.sqrt(de))
        self.init_noise = bool(cfg.slotgraph4_init_noise)
        if self.init_noise:
            self.node_logsig = nn.Parameter(torch.full((d,), math.log(max(emb_std, 1e-4))))
            self.edge_logsig = nn.Parameter(torch.full((de,), math.log(1.0 / math.sqrt(de))))
        if self.boundary_tokens:
            self.boundary = nn.Parameter(mean_vec.view(1, d).repeat(2, 1) + emb_std * torch.randn(2, d))

        # ── write stack (recurrent shared block, applied n_layers times; or n_layers distinct blocks) ──
        n_heads = int(cfg.slotgraph4_heads); d_ff = int(cfg.slotgraph4_d_ff)
        n_blocks = 1 if self.recurrent else self.n_layers
        self.blocks = nn.ModuleList([_SG4Block(d, n_heads, d_ff) for _ in range(n_blocks)])

        # ── competitive assignment (Slot Attention anti-duplication): slots COMPETE (softmax over the SLOT
        # axis) to claim each window token; weighted-mean over tokens. comp_proj ZERO-init → step-0 uses the
        # write-stack slot hidden only (ReZero-safe; the zero-init Linear still receives gradient). ──
        self.comp_q = nn.Linear(d, self.dk)
        self.comp_k = nn.Linear(d, self.dk)
        self.comp_v = nn.Linear(d, d)
        self.comp_proj = nn.Linear(d, d)
        nn.init.zeros_(self.comp_proj.weight); nn.init.zeros_(self.comp_proj.bias)
        self.n_head = nn.LayerNorm(d)

        # ── propose→commit: delta-parameterized proposal (zero-init o_*) + decoupled retain/write gates ──
        # α (retain) bias +2 → sigmoid≈0.88 (retention-biased: keep old state); β (write) bias +1.5 = write-
        # OPEN (gradient flows from step 1). Both weights zero-init so the gate starts uniform and LEARNS
        # content-dependence. Δ is a zero-init projection → step-0 is a no-op REGARDLESS of the gate, so the
        # write-open gate and identity-at-init do not fight (the three tricks compose into one story).
        self.o_node = nn.Linear(d, d)                                      # node proposal Δ (in d)
        self.o_edge = nn.Linear(d, de)                                     # edge proposal Δ (in d_e)
        nn.init.zeros_(self.o_node.weight); nn.init.zeros_(self.o_node.bias)
        nn.init.zeros_(self.o_edge.weight); nn.init.zeros_(self.o_edge.bias)
        self.alpha_node = nn.Linear(d, d); self.beta_node = nn.Linear(d, d)          # channel-wise (GDN-2)
        self.alpha_edge = nn.Linear(d, de); self.beta_edge = nn.Linear(d, de)
        # Gate BIASES set the open/retain mean (α retain ≈0.88, β write ≈0.82); gate WEIGHTS keep default
        # (small) init and are NOT zeroed. This is load-bearing: with Δ zero-init (dX=0 at init) AND zero-init
        # gate weights, the write-stack output `gh` would be behind a double-zero-init wall — ∂state/∂gh = 0 on
        # every path → the write stack gets NO gradient at step 0. Non-zero gate weights keep `gh` gradient-
        # connected to the loss through the retention path (α(gh)⊙state) while dX=0 preserves content
        # identity-at-init. (The write MAGNITUDE still warms up from 0 as o_node/o_edge train — the desired
        # ReZero-style stability, without disconnecting the learner.)
        for _g, _b in ((self.alpha_node, 2.0), (self.beta_node, 1.5),
                       (self.alpha_edge, 2.0), (self.beta_edge, 1.5)):
            with torch.no_grad():
                _g.bias.fill_(_b)
        # EntNet-style post-write normalization (bounds magnitude; the erase/overwrite semantics streaming
        # needs — a new "Mary" mention UPDATES the persistent Mary node, never accumulates unboundedly).
        self.node_norm = _NormMatch(d)
        self.edge_norm = _NormMatch(de)
        with torch.no_grad():
            self.node_norm.scale.data.fill_(emb_norm)
            self.edge_norm.scale.data.fill_(1.0)

        # ── read heads: node-centric attention pool + learned edge salience (soft-gated top-k, never sparsemax) ──
        self.read_q = nn.Linear(d, self.dk)
        self.read_k = nn.Linear(d, self.dk)
        self.edge_sal = nn.Linear(d, 1)
        self.norm = _NormMatch(d)                                          # prepend-token norm-match
        with torch.no_grad():
            self.norm.scale.data.fill_(emb_norm)

        # M = node-centric tokens (N) + explicit edge tokens (read_topk) + boundary
        self.M = N + self.read_topk + (2 if self.boundary_tokens else 0)
        self.force_no_edges = False                                        # eval control: zero edge states (canary)

        top = f"top-{self.read_topk} salience edges" if self.read_topk > 0 else "no explicit edges"
        wr = (f"{self.n_layers}× shared _SG4Block" if self.recurrent
              else f"{self.n_layers} distinct _SG4Block")
        print(f"[slotgraph4] {N} nodes + WS k={k} edges (β={cfg.slotgraph4_ws_beta}) @ d={d}, d_e={de}; "
              f"frozen-LM-hiddens input; {wr} (heads={n_heads}, d_ff={d_ff}); competitive assignment; "
              f"propose→commit (Δ zero-init, decoupled α/β gates, EntNet post-norm; NO delta rule); "
              f"PREPEND read ({N} node-centric + {top} + {'2 boundary' if self.boundary_tokens else 'no boundary'} "
              f"= {self.M} tokens)")

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.train(False)                                             # base is a frozen featurizer, always eval
        return self

    # ── streaming interface (mirrors slotgraph3: accumulate raw embeds; window internally in finalize) ──
    def init_streaming_state(self, batch_size, device, dtype):
        return {"emb": torch.zeros(batch_size, 0, self.d, device=device, dtype=dtype),
                "mask": torch.zeros(batch_size, 0, device=device, dtype=torch.bool)}

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0, **extra):
        del chunk_offset, extra
        if attention_mask is None:
            attention_mask = torch.ones(token_embeds.shape[:2], device=token_embeds.device, dtype=torch.bool)
        return {"emb": torch.cat([state["emb"], token_embeds], dim=1),
                "mask": torch.cat([state["mask"], attention_mask.bool()], dim=1)}, {}

    def _tok(self, content, id_a, id_b, type_idx):
        """Concat-then-project graph token (TokenGT w_in): tok_proj([content ‖ id_a ‖ id_b ‖ type]) → d.
        Shared by node tokens ([X ‖ id ‖ id ‖ node-type]) and edge tokens ([E↑ ‖ id_src ‖ id_dst ‖ edge-type])
        so the id subspace lands in the SAME output directions."""
        t = self.type_embed[type_idx].expand(*content.shape[:-1], -1)
        return self.tok_proj(torch.cat([content, id_a, id_b, t], dim=-1))

    def _tokenize(self, X, E, B):
        """(node_tok:[B,N,d], edge_tok:[B,N*k,d]) — the graph tokens for the write stack / read."""
        N, k, h = self.N, self.k, self.d // 2
        idh = self.id_half.unsqueeze(0).expand(B, -1, -1)                  # [B,N,h]
        node_tok = self._tok(X, idh, idh, 0)                              # node = [X ‖ id ‖ id ‖ node-type]
        E_lift = self.edge_up(E.reshape(B, N * k, self.de))               # [B,N*k,d]
        src_id = self.id_half[self.edge_src_idx].unsqueeze(0).expand(B, -1, -1)   # [B,N*k,h]
        dst_id = self.id_half[self.edge_dst_idx].unsqueeze(0).expand(B, -1, -1)
        edge_tok = self._tok(E_lift, src_id, dst_id, 1)                   # edge = [E↑ ‖ id_src ‖ id_dst ‖ edge-type]
        return node_tok, edge_tok

    def _encode_window(self, we, wm, active):
        """Frozen-LM last-layer hiddens over the window (NO grad, no LoRA). Idle rows keep ≥1 valid key
        (position 0) so the all-masked-row → NaN pathology can't occur. Cast the window to the base's dtype
        so the featurizer is robust to caller dtype (bf16 train path vs fp32 eval)."""
        with torch.no_grad():
            wm_enc = wm.clone(); wm_enc[:, 0] |= ~active
            we = we.to(next(self.base.parameters()).dtype)
            return self.base.model(inputs_embeds=we, attention_mask=wm_enc.long(),
                                   use_cache=False).last_hidden_state

    def _write_stack(self, graph_in, th, keep):
        g = graph_in
        blocks = [self.blocks[0]] * self.n_layers if self.recurrent else list(self.blocks)
        for blk in blocks:
            if self.training and torch.is_grad_enabled():
                g = _ckpt.checkpoint(blk, g, th, keep, use_reentrant=False)
            else:
                g = blk(g, th, keep)
        return g

    def _competitive_read(self, slots, th, wm):
        """Slot Attention: slots query the window text, softmax over the SLOT axis (slots COMPETE to claim
        each token — anti-duplication), weighted-mean over valid tokens → per-slot claimed content."""
        qs = self.comp_q(slots)                                           # [B,S,dk]
        ks = self.comp_k(th); vs = self.comp_v(th)                        # [B,Tw,·]
        attn = torch.softmax(torch.einsum("bsd,btd->bst", qs, ks) / math.sqrt(self.dk), dim=1)
        attn = attn * wm.unsqueeze(1).float()                            # kill pad tokens
        wgt = attn / attn.sum(-1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("bst,btd->bsd", wgt, vs)                      # [B,S,d]

    def finalize_memory(self, state):
        emb, mask = state["emb"], state["mask"]
        B, T, d = emb.shape
        if T == 0:
            raise ValueError("slotgraph4.finalize_memory: empty context (T=0)")
        N, k, de = self.N, self.k, self.de
        X = self.node_init.float().unsqueeze(0).expand(B, -1, -1).contiguous()          # [B,N,d]
        E = self.edge_init.float().unsqueeze(0).expand(B, -1, -1, -1).contiguous()       # [B,N,k,d_e]
        if self.init_noise and self.training:
            X = X + self.node_logsig.float().exp() * torch.randn_like(X)
            E = E + self.edge_logsig.float().exp() * torch.randn_like(E)

        for w in range(0, T, self.window):
            wm = mask[:, w:w + self.window].bool()
            active = wm.any(dim=1)
            if not bool(active.any()):
                continue
            we = emb[:, w:w + self.window]
            X0, E0 = X, E
            with torch.autocast("cuda", enabled=False):                   # tokenization in fp32 (delicate ids)
                node_tok, edge_tok = self._tokenize(X, E, B)
                graph_in = torch.cat([node_tok, edge_tok], dim=1)         # [B, N + N*k, d]
                keep_g = torch.ones(B, graph_in.shape[1], device=we.device, dtype=torch.bool)
                if self.boundary_tokens:                                   # mark the graph span start in the write
                    bs = self.boundary[0].float().view(1, 1, -1).expand(B, 1, -1)
                    graph_in = torch.cat([bs, graph_in], dim=1)
                    keep_g = torch.cat([torch.ones(B, 1, device=we.device, dtype=torch.bool), keep_g], dim=1)
            th = self._encode_window(we, wm, active)                      # frozen hiddens (no grad)
            keep = torch.cat([wm, keep_g], dim=1)                         # text validity ; graph slots (all valid)
            G = self._write_stack(graph_in.to(th.dtype), th, keep)        # scratch (grad)
            with torch.autocast("cuda", enabled=False):
                off = 1 if self.boundary_tokens else 0
                node_slots = G[:, off:off + N].float()                    # [B,N,d]
                edge_slots = G[:, off + N:off + N + N * k].float()        # [B,N*k,d]
                slots = torch.cat([node_slots, edge_slots], dim=1)
                comp = self._competitive_read(slots, th.float(), wm)      # [B, N + N*k, d]
                gh = self.n_head(slots + self.comp_proj(comp))
                gh_n, gh_e = gh[:, :N], gh[:, N:N + N * k]
                # ── the ONE persistent write: propose (zero-init Δ) → commit (decoupled α/β, post-norm) ──
                dX = self.o_node(gh_n)                                     # [B,N,d]  (Δ=0 at init)
                aX = torch.sigmoid(self.alpha_node(gh_n)); bX = torch.sigmoid(self.beta_node(gh_n))
                X = self.node_norm(aX * X + bX * dX)
                dE = self.o_edge(gh_e).reshape(B, N, k, de)               # [B,N,k,d_e] (Δ=0 at init)
                aE = torch.sigmoid(self.alpha_edge(gh_e)).reshape(B, N, k, de)
                bE = torch.sigmoid(self.beta_edge(gh_e)).reshape(B, N, k, de)
                E = self.edge_norm(aE * E + bE * dE)
                if not bool(active.all()):                                # freeze rows idle this window
                    X = torch.where(active[:, None, None], X, X0)
                    E = torch.where(active[:, None, None, None], E, E0)

        with torch.autocast("cuda", enabled=False):
            memory_full, keep_read = self._build_read(X, E, B)
        aux = self._canaries(memory_full, X, E, emb.device)
        aux["memory_mask"] = keep_read
        aux["latents"] = (X, E)
        return memory_full, aux

    def _build_read(self, X, E, B):
        """Compress the (N node + N·k edge) state into M prepend tokens:
          · node-centric: each node ← attention-pool of [X_i ‖ its k edge states] (lifted), then tokenized
            with the node id (a node is potential src AND dst → [id ‖ id]).
          · top-k explicit edge tokens: the read_topk highest-SALIENCE edges (learned score, soft-gated),
            as pointer tokens carrying [id_src ‖ id_dst] so the sharpest relations survive intact.
        Present via PREPEND + bidirectional memory attention (cfg.bidir_mem_attn / uniform_mem_pos)."""
        N, k, de = self.N, self.k, self.de
        if self.force_no_edges:                                           # eval canary: are edges load-bearing?
            E = torch.zeros_like(E)
        E_lift = self.edge_up(E.reshape(B, N * k, de)).reshape(B, N, k, self.d)   # [B,N,k,d]
        # node-centric attention pool: query = node latent; keys/values = [X_i ; its k edge states]
        items = torch.cat([X.unsqueeze(2), E_lift], dim=2)                # [B,N,1+k,d]
        q = self.read_q(X)                                                # [B,N,dk]
        kk = self.read_k(items)                                           # [B,N,1+k,dk]
        pa = torch.softmax(torch.einsum("bnd,bnid->bni", q, kk) / math.sqrt(self.dk), dim=-1)
        pooled = torch.einsum("bni,bnid->bnd", pa, items)                 # [B,N,d]
        idh = self.id_half.unsqueeze(0).expand(B, -1, -1)
        node_read = self._tok(pooled, idh, idh, 0)                        # [B,N,d]

        toks = [node_read]
        keep_parts = [torch.ones(B, N, dtype=torch.bool, device=X.device)]
        if self.read_topk > 0:
            sal = self.edge_sal(E_lift.reshape(B, N * k, self.d)).squeeze(-1)   # [B,N*k] learned salience
            topv, topi = sal.topk(min(self.read_topk, N * k), dim=-1)     # soft-gated top-k (never sparsemax)
            gate = torch.sigmoid(topv).unsqueeze(-1)                      # [B,tk,1] soft gate
            sel_lift = torch.gather(E_lift.reshape(B, N * k, self.d), 1,
                                    topi.unsqueeze(-1).expand(-1, -1, self.d))
            src_id = self.id_half[self.edge_src_idx][topi]               # [B,tk,h]
            dst_id = self.id_half[self.edge_dst_idx][topi]
            edge_read = gate * self._tok(sel_lift, src_id, dst_id, 1)
            toks.append(edge_read)
            keep_parts.append(torch.ones(B, topi.shape[1], dtype=torch.bool, device=X.device))
        memory = self.norm(torch.cat(toks, dim=1))                        # [B, N+tk, d]
        keep_read = torch.cat(keep_parts, dim=1)
        if self.boundary_tokens:
            bnd = self.norm(self.boundary.float()).unsqueeze(0).expand(B, -1, -1)
            memory = torch.cat([bnd[:, :1], memory, bnd[:, 1:]], dim=1)
            ones = torch.ones(B, 1, dtype=torch.bool, device=X.device)
            keep_read = torch.cat([ones, keep_read, ones], dim=1)
        return memory, keep_read

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        st = self.init_streaming_state(token_embeds.shape[0], token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)

    @torch.no_grad()
    def _canaries(self, memory, X, E, device):
        def _within_cos(x):
            S = x.shape[1]
            if S < 2:
                return 0.0
            xn = F.normalize(x, dim=-1); cos = xn @ xn.transpose(-1, -2)
            off = cos.sum((-1, -2)) - cos.diagonal(dim1=-2, dim2=-1).sum(-1)
            return float((off / (S * (S - 1))).mean())
        B = memory.shape[0]
        E_flat = E.reshape(B, -1, self.de)
        aux = {
            "slotgraph4_mem_effrank": torch.tensor(_participation_ratio(memory.reshape(-1, self.d)), device=device),
            "slotgraph4_node_effrank": torch.tensor(_participation_ratio(X.reshape(-1, self.d)), device=device),
            "slotgraph4_edge_effrank": torch.tensor(_participation_ratio(E_flat.reshape(-1, self.de)), device=device),
            "slotgraph4_node_cos": torch.tensor(_within_cos(X), device=device),
            "slotgraph4_edge_cos": torch.tensor(_within_cos(E_flat), device=device),
            # per-example vs cross-example mem rank (blur vs shared-frame; see slotgraph3 canaries)
            "slotgraph4_mem_effrank_perex": torch.tensor(
                sum(_participation_ratio(memory[i]) for i in range(B)) / max(1, B), device=device),
            "slotgraph4_mem_effrank_cross": torch.tensor(
                _participation_ratio(memory.mean(dim=1)) if B > 1 else 0.0, device=device),
        }
        return aux
