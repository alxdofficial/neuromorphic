"""slotgraph3 — compressed-implicit graph memory; LM-attention write; expanded edge read (d = d_llama = 576).

The graph STATE is compressed: per node we carry (node_latent, edge_latent) + a fixed learnable id, so the
stored footprint is 2·K vectors. The graph is EXPANDED to explicit edge tokens (top-k per node) both during
the write (so the frozen LM's attention sees the current structure) and before the read (so the LM reads
relations, not free slots).

  route:   A = sparsemax_j( q_route(edge_lat_i) · k_proj(edge_lat_j) / √dk ),  diag = −∞
           CONTENT-addressed (query = source intent, key = DESTINATION's edge state — NOT the fixed id, or the
           wiring would be positional/generic). sparsemax → exact zeros; top-k picks the strongest edges/node
           (sparsemax makes the dropped ones zero-gradient, so the top-k truncation is exact & differentiable).
  expand:  edge_token(i,j) = A[i,j] · φ(node_lat_i, node_lat_j, edge_lat_i)   # A-gated content (load-bearing)
                           + id_scale · (node_id_i + node_id_j)               # endpoint labels (coord frame → chaining)
                           + role_edge
  WRITE:   per streaming window, tokenization + expansion IDENTICAL across two swappable mixers:
           · "lm"     — ONE frozen-LM (+encoder-LoRA) forward over
                        [ window-tokens ; expanded-edges ; node-slot-tokens ; edge-slot-tokens ]
                        (causal; the PRETRAINED attention is the graph mixer, re-expanded each window).
           · "custom" — frozen LM contextualizes the window (no grad, no LoRA) → last hiddens; then N
                        from-scratch _SG3Blocks where GRAPH tokens query [hiddens ; graph] (self-attn among
                        graph + cross-attn to text, position-free, zero-init residuals).
           Either way: read the slot hiddens → ADDITIVE-residual update of node/edge latents.
  READ:    expand top-k, PREPEND those K·topk edge tokens (norm-matched). NO raw node/edge latent is prepended.

Control: force_identity_A sets A := I (each node reads itself) — EM must collapse if the edges are load-bearing
(a free diagnostic; NOT an atomicity mechanism).
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


class _Sparsemax(torch.autograd.Function):
    """sparsemax over the last dim (Martins & Astudillo 2016) = α=2 entmax: projection onto the simplex →
    SPARSE (exact zeros), differentiable. ADAPTIVE support (dense when logits are flat, sparse when peaked)."""
    @staticmethod
    def forward(ctx, z):
        z = z - z.max(dim=-1, keepdim=True).values
        zsort = torch.sort(z, dim=-1, descending=True).values
        rng = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
        cssv = zsort.cumsum(dim=-1) - 1.0
        cond = (zsort - cssv / rng) > 0
        k = cond.sum(dim=-1, keepdim=True).clamp(min=1)
        tau = cssv.gather(-1, (k - 1).long()) / k.to(z.dtype)
        p = torch.clamp(z - tau, min=0.0)
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, g):
        (p,) = ctx.saved_tensors
        supp = (p > 0).to(g.dtype)
        v = (g * supp).sum(dim=-1, keepdim=True) / supp.sum(dim=-1, keepdim=True).clamp(min=1)
        return supp * (g - v)


def sparsemax(z: Tensor) -> Tensor:
    return _Sparsemax.apply(z)


class _SG3Block(nn.Module):
    """From-scratch pre-norm graph-mixer block (the CUSTOM write alternative to the frozen LM's attention).
    GRAPH tokens are the queries; keys/values = [frozen-LM window hiddens ; graph tokens] — i.e. self-attention
    among the graph tokens + cross-attention to the contextualized text, fused into one SDPA. Text is static
    (never re-written by these blocks). NO positional encoding: word order already lives in the LM hiddens'
    CONTENT, and graph tokens are a SET identified by id/role embeddings. Residual output projections are
    ZERO-init (ReZero/Fixup-style; same precedent as the jun24 slotgraph zero-init MP-read) so each block is
    exact identity at init and the latent highway is undisturbed until the mixer earns its influence."""
    def __init__(self, d: int, n_heads: int, d_ff: int):
        super().__init__()
        assert d % n_heads == 0, f"d={d} not divisible by n_heads={n_heads}"
        self.nh = n_heads; self.hd = d // n_heads
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)        # identity-at-init residual branches
        nn.init.zeros_(self.ff[2].weight); nn.init.zeros_(self.ff[2].bias)

    def forward(self, g: Tensor, ctx: Tensor, keep: Tensor) -> Tensor:
        # g:[B,G,d] graph tokens (queries)  ctx:[B,T,d] frozen window hiddens (keys/values only)
        # keep:[B,T+G] bool key-validity — slots are always valid, so no query row is ever fully masked
        B, G, d = g.shape
        h = self.ln1(torch.cat([ctx, g], dim=1))                          # shared pre-LN over keys/values
        q = self.q(h[:, ctx.shape[1]:]).view(B, G, self.nh, self.hd).transpose(1, 2)
        k = self.k(h).view(B, -1, self.nh, self.hd).transpose(1, 2)
        v = self.v(h).view(B, -1, self.nh, self.hd).transpose(1, 2)
        m = keep[:, None, None, :]                                        # bool mask, True = attend
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)          # SDPA does the 1/√hd scaling
        a = a.transpose(1, 2).reshape(B, G, d)
        g = g + self.o(a)
        g = g + self.ff(self.ln2(g))
        return g


class SlotGraph3Encoder(nn.Module):
    is_conditioned_read = False              # READ = PREPEND the expanded edge tokens

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        from ...decoder import load_frozen_llama, apply_lora_to_llama
        d = cfg.d_llama; self.d = d
        self.write_mode = str(getattr(cfg, "slotgraph3_write", "lm"))     # "lm" (frozen prior) | "custom" (from scratch)
        base, _ = load_frozen_llama(cfg.llama_model)
        for p in base.parameters():
            p.requires_grad_(False)
        embed = base.get_input_embeddings()                              # embed stats (needed for BOTH modes)
        with torch.no_grad():
            mean_vec = embed.weight.float().mean(0); emb_std = embed.weight.float().std().item()
            emb_norm = embed.weight.float().norm(dim=-1).mean().item()
        if self.write_mode == "lm":
            n_wrapped = apply_lora_to_llama(base, rank=cfg.slotgraph3_lora_rank, alpha=cfg.slotgraph3_lora_alpha,
                                            target_names=tuple(cfg.llama_lora_target_names))
            base.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.base = base; self.blocks = None; n_wrapped = int(n_wrapped)
        else:                                                            # custom: frozen-LM ENCODES each window
            n_layers = int(cfg.slotgraph3_custom_layers); d_ff = int(cfg.slotgraph3_custom_dff)   # (no grad, no
            n_heads = int(cfg.slotgraph3_custom_heads)                                            # LoRA); blocks
            self.blocks = nn.ModuleList([_SG3Block(d, n_heads, d_ff) for _ in range(n_layers)])   # mix the graph
            self.base = base; n_wrapped = 0                              # kept frozen — window contextualizer only
        self.K = int(cfg.slotgraph3_n_nodes); self.window = int(cfg.slotgraph3_window)
        self.dk = int(cfg.slotgraph3_d_key); self.read_topk = int(cfg.slotgraph3_read_topk)
        self.write_expand = bool(getattr(cfg, "slotgraph3_write_expand", True))
        self.gate_ids = bool(getattr(cfg, "slotgraph3_gate_ids", False))
        self.st_leak = bool(getattr(cfg, "slotgraph3_st_leak", False))
        self.edge_budget = int(getattr(cfg, "slotgraph3_edge_budget", 0))
        self.route_key = str(getattr(cfg, "slotgraph3_route_key", "edge"))
        self.edge_state = str(getattr(cfg, "slotgraph3_edge_state", "flat"))
        self.write_layers = int(getattr(cfg, "slotgraph3_write_layers", 0))   # 0 = full ride; N = last-N splice
        if self.edge_state == "matrix":
            r = int(round(d ** 0.5))
            if r * r != d:
                raise ValueError(f"matrix edge state needs square d (got d={d})")
            self.r = r
            self.rel_key = nn.Linear(d, r)                     # dst content → retrieval key
            self.rel_up = nn.Linear(r, d)                      # per-pair relation code → φ input space
        if self.edge_budget > 0 and not self.st_leak:
            raise ValueError("slotgraph3_edge_budget (global top-E) requires slotgraph3_st_leak "
                             "(the backward for non-materialized edges is the global ST leak)")
        if self.edge_budget == 0 and not (1 <= self.read_topk <= self.K - 1):
            raise ValueError(f"slotgraph3_read_topk must be in [1, K-1] (K={self.K}, got "
                             f"{self.read_topk}; self-loops are masked so only K-1 dests exist)")
        self.read_mode = str(getattr(cfg, "slotgraph3_read", "raw"))
        self.boundary_tokens = bool(getattr(cfg, "slotgraph3_boundary_tokens", True))
        n_edge_tok = self.edge_budget if self.edge_budget > 0 else self.K * self.read_topk
        self.M = (n_edge_tok + (self.K if self.read_mode == "raw" else 0)   # raw: + K node-content tokens
                  + (2 if self.boundary_tokens else 0))                     # + <mem_start>/<mem_end>
        self.force_identity_A = False                          # eval-only control: A := I (edges decorative)
        if self.read_mode == "raw":
            # fixed orthonormal HALF-dim ids: edge pointer = concat[id_src; id_dst] (TokenGT — ordered,
            # direction-preserving); node token carries [id; id] (a node is potential src AND dst).
            h = d // 2
            if self.K > h:
                raise ValueError(f"raw read needs K ≤ d/2 orthonormal half-ids (K={self.K}, d/2={h})")
            idh = torch.empty(self.K, h); nn.init.orthogonal_(idh)
            self.register_buffer("id_half", F.normalize(idh, dim=-1), persistent=True)
            # concat-then-PROJECT token formation (TokenGT w_in; 2026-07-02 geometry batch):
            # ONE shared linear over [content(d) ‖ id_a(h) ‖ id_b(h) ‖ type(h)] → d for node AND edge
            # tokens — the id subspace lands in the SAME output directions for both, so attention can
            # match an edge's endpoint ids against the node token by construction. Replaces the additive
            # content+id·scale+role superposition (√3 norm-inflation → attention-sink; a FROZEN LM has
            # no trained direction to unmix a sum — TokenGT/BERT train theirs in). The projection also
            # rotates the block-structured concat onto the manifold the frozen attention reads.
            self.type_embed = nn.Parameter(torch.empty(3, h))   # [node, edge-slot, edge-pointer]
            nn.init.orthogonal_(self.type_embed)
            with torch.no_grad():
                self.type_embed.copy_(F.normalize(self.type_embed, dim=-1))
            self.tok_proj = nn.Linear(d + 3 * h, d)

        if self.read_mode == "edges":
            # LEGACY edges-mode identity system (raw mode uses id_half/type_embed/tok_proj instead —
            # creating these unconditionally would be ~9K dead params polluting grad diagnostics).
            nid = torch.empty(self.K, d); nn.init.orthogonal_(nid)
            self.node_id = nn.Parameter(F.normalize(nid, dim=-1))
            self.role = nn.Parameter(torch.randn(3, d) / math.sqrt(d))   # [node-slot, edge-slot, edge-token]
            # id_scale ~ the LM EMBEDDING row-norm: the additive id tag must sit at the embedding scale
            # (√d·unit-id ≈ 7-10× the embed norm → coordinate labels drown the content).
            self.id_scale = nn.Parameter(torch.tensor(emb_norm))
        self.register_buffer("diag_mask", torch.eye(self.K) * -1e9, persistent=False)
        self.node_lat_init = nn.Parameter(mean_vec.view(1, d).repeat(self.K, 1) + emb_std * torch.randn(self.K, d))
        self.edge_lat_init = nn.Parameter(mean_vec.view(1, d).repeat(self.K, 1) + emb_std * torch.randn(self.K, d))
        if self.boundary_tokens:
            # learned <mem_start>/<mem_end> span markers, init ON the embedding distribution
            # (mean + std·randn, same recipe as the latents) — every working frozen-LM injection
            # marks its span with explicit tokens (Qwen vision_start/end, ICAE [AE]/[LM]).
            self.boundary = nn.Parameter(mean_vec.view(1, d).repeat(2, 1) + emb_std * torch.randn(2, d))
        # routing = 2-layer MLPs (the wiring decision is the thesis crux → give it real capacity, not a bare bilinear)
        self.q_route = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, self.dk))   # source outgoing intent
        self.k_proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, self.dk))    # dest key (input-dependent)
        if self.read_mode == "edges":                          # raw mode has NO content synthesizer (the
            self.phi = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(),               # φ/Q-Former lesson)
                                     nn.Linear(d, d))          # (src,dst,edge) → edge content
        # ── write-audit repairs (2026-07-03; research_graph_write_audit) ─────────────────
        self.route_act = str(getattr(cfg, "slotgraph3_route_act", "softmax"))
        self.init_noise = bool(getattr(cfg, "slotgraph3_init_noise", True))
        self.write_norm_match = bool(getattr(cfg, "slotgraph3_write_norm_match", True))
        self.write_bb = bool(getattr(cfg, "slotgraph3_write_boundary_bidir", True))
        self.write_update = str(getattr(cfg, "slotgraph3_write_update", "delta"))
        if self.init_noise:
            # per-forward Gaussian sampling around the learned init (Slot Attention): breaks slot
            # symmetry — identical deterministic slots + shared heads receive identical gradients
            # and stay identical forever. σ learned per-dim, init = the measured embedding std.
            self.node_lat_logsig = nn.Parameter(torch.full((d,), math.log(max(emb_std, 1e-4))))
            self.edge_lat_logsig = nn.Parameter(torch.full((d,), math.log(max(emb_std, 1e-4))))
        if self.write_norm_match:
            # graph tokens are scaled to the TEXT stream's RMS at the insertion depth (learned
            # scalar × dynamic match). Measured 400× mismatch at the last-4 splice (graph ~1.6 vs
            # text ~641): attention/MLP outputs are calibrated to the local stream scale, so a
            # quiet token's residual is swamped after one layer — slot identity erased in the write.
            self.write_scale = nn.Parameter(torch.tensor(1.0))
        # T3 edge write (2026-07-03): keyed delta-rule associative write into the persistent map
        self.edge_write = str(getattr(cfg, "slotgraph3_edge_write", "assoc"))
        if self.edge_write == "assoc":
            if self.edge_state != "matrix":
                raise ValueError("slotgraph3_edge_write='assoc' needs edge_state='matrix' "
                                 "(the write files into the r×r map the read looks up)")
            if self.write_update != "delta":
                raise ValueError("slotgraph3_edge_write='assoc' needs write_update='delta' "
                                 "(the filed value/gate come from the delta update path)")
        # write heads: slot hiddens → candidate latent (delta mode: gated INTERPOLATION toward the
        # candidate; additive mode: LEGACY scalar-gated residual delta). head_edge is skipped in
        # assoc mode — the edge update is the keyed delta write, not a candidate interpolation.
        self.n_head = nn.LayerNorm(d)
        self.head_node = nn.Linear(d, d)
        if self.edge_write != "assoc":
            self.head_edge = nn.Linear(d, d)
        if self.write_update == "additive":
            # LEGACY scalar gates (additive mode only — dead params otherwise). Init +1.5 = write-
            # OPEN: gates are biased toward the behavior that should be free (Jozefowicz +1 forget-
            # init); the old −1.2 (gate 0.23) throttled the write 77% and NEVER moved over 4k steps.
            self.beta_node = nn.Parameter(torch.tensor(1.5))
            self.beta_edge = nn.Parameter(torch.tensor(1.5))
        if self.write_update == "delta":
            # competitive read (Slot Attention): queries = slot hiddens, keys/values = window text
            # hiddens, softmax over the SLOT axis — slots COMPETE to claim each token (the anti-
            # duplication mechanism); weighted-mean over tokens. comp_proj ZERO-init: step-0
            # behavior = slot-hidden-only (zero-init Linear still receives gradient — ReZero-safe).
            self.comp_q = nn.Linear(d, self.dk)
            self.comp_k = nn.Linear(d, self.dk)
            self.comp_v = nn.Linear(d, d)
            self.comp_proj = nn.Linear(d, d)
            nn.init.zeros_(self.comp_proj.weight); nn.init.zeros_(self.comp_proj.bias)
            # per-slot CONTENT-dependent write gate (EntNet-style), bias +1.5 = write-open; weight
            # zero-init so the gate starts uniform-open and LEARNS to differentiate by content.
            self.gate_head = nn.Linear(d, 1)
            nn.init.zeros_(self.gate_head.weight)
            with torch.no_grad():
                self.gate_head.bias.fill_(1.5)
        if self.edge_write == "assoc":
            # filed VALUE: edge-slot hidden → r-dim relation code (what the read's rel_up lifts back).
            self.rel_val = nn.Linear(d, self.r)
            # retention-biased per-slot DECAY: α init = 1/(2·n_windows) → half-life ≈ the full
            # context (forget-gate wisdom inverted for a decay: bias toward RETAIN). n_windows is
            # derived from the ACTUAL training context (cfg.ctx_len, set by the trainer) — a
            # hardcoded 1024 would silently mis-scale the half-life under --mixed-ctx changes.
            # Weight zero-init: starts uniform, learns content-dependence.
            n_win = max(1, int(round(int(getattr(cfg, "ctx_len", 1024)) / max(1, self.window))))
            a0 = 1.0 / (2.0 * n_win)
            self.decay_head = nn.Linear(d, 1)
            nn.init.zeros_(self.decay_head.weight)
            with torch.no_grad():
                self.decay_head.bias.fill_(math.log(a0 / (1.0 - a0)))

        self._trace = None                                     # set to [] to record per-window (node_lat, edge_lat) for grad-credit
        self.norm = _NormMatch(d)
        with torch.no_grad():
            self.norm.scale.data.fill_(emb_norm)
        if self.write_mode == "lm":
            depth = (f"LAST-{self.write_layers} layers (text: frozen no-grad prefix)" if self.write_layers > 0
                     else "full ride, all layers")
            writer = f"LM-attention write ({depth}; frozen SmolLM2 + enc-LoRA r{cfg.slotgraph3_lora_rank})"
        else:
            writer = (f"CUSTOM write (frozen-LM window encode → {len(self.blocks)} from-scratch graph-mixer "
                      f"_SG3Block × d_ff={cfg.slotgraph3_custom_dff}, {cfg.slotgraph3_custom_heads} heads, "
                      f"position-free, zero-init residuals)")
        budget = (f"GLOBAL top-{self.edge_budget} edges" if self.edge_budget > 0
                  else f"top-{self.read_topk}/node = {self.M} tokens")
        read_desc = (f"RAW read ({self.K} node-content tokens + pointer edges, concat-PROJECT "
                     f"tok_proj([content‖id_src‖id_dst‖type])→d, no φ)"
                     if self.read_mode == "raw" else "φ-edge read (LEGACY additive id-sum)")
        if self.boundary_tokens:
            read_desc += " + <mem_start>/<mem_end> boundary"
        upd_desc = ("DELTA write (per-slot content gate b+1.5 · competitive slot-read · gated interpolation)"
                    if self.write_update == "delta" else "ADDITIVE write (LEGACY scalar gate)")
        if self.edge_write == "assoc":
            upd_desc += f"; ASSOC edge write (keyed Δ-rule (v−M·k̂)⊗k̂ into persistent {self.r}×{self.r}, α₀={1.0/(2*max(1,round(1024/max(1,self.window)))):.3f})"
        fixes = "+".join(s for s, on in (("initnoise", self.init_noise), ("splicescale", self.write_norm_match),
                                         ("writeBB", self.write_bb)) if on)
        print(f"[slotgraph3] {self.K} nodes (2×{self.K} latents) @ d={d}; {writer} "
              f"({'+edges in context' if self.write_expand else 'slots-only, edges at READ only'}); "
              f"route-by-{self.route_key} ({self.route_act}); expand→edges ({budget}"
              f"{', ST leak' if self.st_leak else ''}); {upd_desc}; [{fixes}]; "
              f"{read_desc}; PREPEND ({self.M} tokens)")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.base is not None:
            self.base.train(False)
        return self

    def init_streaming_state(self, batch_size, device, dtype):
        return {"emb": torch.zeros(batch_size, 0, self.d, device=device, dtype=dtype),
                "mask": torch.zeros(batch_size, 0, device=device, dtype=torch.bool)}

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0, **extra):
        del chunk_offset, extra
        if attention_mask is None:
            attention_mask = torch.ones(token_embeds.shape[:2], device=token_embeds.device, dtype=torch.bool)
        return {"emb": torch.cat([state["emb"], token_embeds], dim=1),
                "mask": torch.cat([state["mask"], attention_mask.bool()], dim=1)}, {}

    def _rel_input(self, er_src, dst_content):
        """φ's relation input for aligned [..., d] src-edge-state / dst-content tensors.
        flat: the shared per-source edge_lat (v1). matrix: per-PAIR retrieval — view the same d floats as
        an r×r associative map and read it with the destination's UNIT key: rel(i→j) = M_i · k̂(n_j).
        The key is L2-normalized so the read address matches the assoc WRITE address exactly (T3:
        the write files (v − M·k̂)⊗k̂ — with unit keys repeated writes drive M·k̂ → g/(g+α)·v, exact
        overwrite in the g=1, α=0 limit; an unnormalized read key would rescale retrieval by ‖k‖
        and break the write↔read pairing)."""
        if self.edge_state != "matrix":
            return er_src
        r = self.r
        Mi = er_src.reshape(*er_src.shape[:-1], r, r)
        key = F.normalize(self.rel_key(dst_content), dim=-1)             # [..., r] unit address
        rel = torch.einsum("...ij,...j->...i", Mi, key)                  # [..., r] pair-specific code
        return self.rel_up(rel)

    def _tok(self, content, id_a, id_b, type_idx, weight=None):
        """Concat-then-project graph token (raw mode; TokenGT w_in): tok_proj([content ‖ id_a ‖ id_b ‖
        type]) → d. ONE projection shared by node tokens ([lat ‖ id ‖ id ‖ node-type]) and edge tokens
        ([rel ‖ id_src ‖ id_dst ‖ edge-type]) so the id subspace lands in the SAME output directions —
        attention matches an edge's endpoint ids against node tokens by construction. `weight` (routing
        strength) gates content+ids INSIDE the projection; the type block stays un-gated. Linearity ⇒
        tok = w·proj([c‖ids‖0]) + proj([0‖0‖0‖type]) — the un-gated type term anchors the token so
        NormMatch cannot erase the weight signal (same pattern as the old gate_ids anchor)."""
        t = self.type_embed[type_idx].expand(*content.shape[:-1], -1)
        if weight is not None:
            w = weight.unsqueeze(-1)
            content = w * content; id_a = w * id_a; id_b = w * id_b
        return self.tok_proj(torch.cat([content, id_a, id_b, t], dim=-1))

    # ── expansion: compressed latents → content-addressed sparse wiring A → top-k explicit edge tokens ──
    def _route(self, key_lat):
        """key_lat = node_lat (route by CONTENT; K/V split — edge_lat freed for pure relation semantics)
        or edge_lat (v1: one vector serves routing AND relation)."""
        q = self.q_route(key_lat)                                        # [B,K,dk] source outgoing intent
        k = self.k_proj(key_lat)                                         # [B,K,dk] dest key (input-dependent)
        sc = torch.einsum("bik,bjk->bij", q, k) / math.sqrt(self.dk)     # [B,K,K]
        sc = sc + self.diag_mask.unsqueeze(0)                            # forbid self-loops
        if self.route_act == "sparsemax":
            A = sparsemax(sc)                                            # LEGACY: exact zeros — out-of-support
        else:                                                            # dests get EXACTLY 0 gradient → the
            if self.training:                                            # support only shrinks (dead ratchet).
                # train-time Gumbel noise: parameter-free exploration ~ the logit scale at init;
                # the learned q/k signal outgrows it (softmax is shift-invariant, mean offset moot)
                u = torch.rand_like(sc).clamp_(1e-9, 1.0 - 1e-9)
                sc = sc - torch.log(-torch.log(u))                       # + Gumbel(0,1)
            A = torch.softmax(sc, dim=-1)
        if self.force_identity_A:
            A = torch.eye(self.K, device=A.device, dtype=A.dtype).unsqueeze(0).expand_as(A)
        return A

    def _expand_topk(self, node_lat, edge_lat, nid, id_scale, role):
        """Top-k edge tokens per node: [B, K·topk, d] + keep mask + (A, topk-weights) for canaries."""
        B, K, d = node_lat.shape; k = self.read_topk
        A = self._route(node_lat if self.route_key == "node" else edge_lat)   # [B,K,K]
        if self.edge_budget > 0:
            return self._expand_global(node_lat, edge_lat, nid, id_scale, role, A)
        if self.st_leak:
            return self._expand_st(node_lat, edge_lat, nid, id_scale, role, A)
        topv, topi = A.topk(k, dim=-1)                                   # [B,K,k] weights + dst indices
        src = node_lat.unsqueeze(2).expand(B, K, k, d)                   # source content (node i)
        dst = torch.gather(node_lat.unsqueeze(1).expand(B, K, K, d), 2,
                           topi.unsqueeze(-1).expand(B, K, k, d))        # dst content (node topi)
        er = self._rel_input(edge_lat.unsqueeze(2).expand(B, K, k, d), dst)   # source edge state (per-pair if matrix)
        if self.read_mode == "raw":
            # POINTER token (concat-project): [rel ‖ id_src ‖ id_dst ‖ edge-type] → d; ordered id slots
            # (direction survives), routing weight gates content+ids inside the projection, no φ
            h = d // 2
            E = self._tok(er, self.id_half.view(1, K, 1, h).expand(B, K, k, h),
                          self.id_half[topi], 2, weight=topv)
            return E.reshape(B, K * k, d), (topv > 0).reshape(B, K * k), A, topv
        phi = self.phi(torch.cat([src, dst, er], dim=-1))               # [B,K,k,d]
        id_src = nid.unsqueeze(1).expand(K, k, d).unsqueeze(0)          # [1,K,k,d]
        id_dst = nid[topi]                                              # [B,K,k,d]
        lab = id_scale * (id_src + id_dst)
        if self.gate_ids:
            # soft-id: the endpoint label rides INSIDE the routing weight (Switch gate-multiplication
            # pattern) → the router gets gradient through the dominant id channel of every selected
            # edge, and a weak edge no longer emits a full-loudness (wrong) label. The un-gated role
            # term anchors the token so NormMatch does not erase the topv information.
            E = topv.unsqueeze(-1) * (phi + lab) + role[2]
        else:
            E = topv.unsqueeze(-1) * phi + lab + role[2]
        return E.reshape(B, K * k, d), (topv > 0).reshape(B, K * k), A, topv

    def _expand_st(self, node_lat, edge_lat, nid, id_scale, role, A):
        """STRAIGHT-THROUGH expansion — dense gradient at sparse token cost. All K×K candidate edge
        tokens are computed in FLOPs (φ is a cheap MLP; K² never becomes tokens); the FORWARD is the
        exact hard top-k gather (the decoder reads 100% pure edge tokens, context stays K·topk), the
        BACKWARD flows through a soft mixture M = (1−ε)·hard + ε·rest, where the leak ε = the router's
        own out-of-top-k sparsemax mass and rest = the mass-renormalized mixture of unselected edges.
        SELF-ANNEALING: routing flat → fat leak → dense exploration gradient; routing sharp → ε→0 →
        the estimator becomes exact (95/5 → 100/0 with no schedule, paced by router confidence)."""
        B, K, d = node_lat.shape; k = self.read_topk
        src_f = node_lat.unsqueeze(2).expand(B, K, K, d)
        dst_f = node_lat.unsqueeze(1).expand(B, K, K, d)
        er_f = self._rel_input(edge_lat.unsqueeze(2).expand(B, K, K, d), dst_f)
        if self.read_mode == "raw":
            h = d // 2
            Tf = self._tok(er_f, self.id_half.view(1, K, 1, h).expand(B, K, K, h),
                           self.id_half.view(1, 1, K, h).expand(B, K, K, h), 2, weight=A)
        else:
            phi_f = self.phi(torch.cat([src_f, dst_f, er_f], dim=-1))   # [B,K,K,d] — all candidate edges
            lab_f = id_scale * (nid.view(1, K, 1, d) + nid.view(1, 1, K, d))
            if self.gate_ids:
                Tf = A.unsqueeze(-1) * (phi_f + lab_f) + role[2]
            else:
                Tf = A.unsqueeze(-1) * phi_f + lab_f + role[2]
        topv, topi = A.topk(k, dim=-1)                                   # [B,K,k]
        H = torch.gather(Tf, 2, topi.unsqueeze(-1).expand(B, K, k, d))   # hard tokens (the forward)
        sel = torch.zeros_like(A, dtype=torch.bool).scatter_(2, topi, True)
        rest_w = A * (~sel)                                              # unselected sparsemax mass
        rest_sum = rest_w.sum(-1, keepdim=True)                          # [B,K,1]
        eps = (rest_sum / A.sum(-1, keepdim=True).clamp_min(1e-9)).unsqueeze(-1)   # leak fraction [B,K,1,1]
        R = torch.einsum("bij,bijd->bid", rest_w / rest_sum.clamp_min(1e-9), Tf)   # rest mixture [B,K,d]
        M = (1.0 - eps) * H + eps * R.unsqueeze(2)                       # soft slot mixture [B,K,k,d]
        E = (H - M).detach() + M                                         # ST: forward = H, backward = ∂M
        return E.reshape(B, K * k, d), (topv > 0).reshape(B, K * k), A, topv

    def _token_block(self, node_lat, edge_lat, nid, id_scale, role, A_blk, src_sel, dst_idx):
        """Edge tokens for an arbitrary (src,dst) index set. src_sel/dst_idx: [B,E] long."""
        B, K, d = node_lat.shape; E = src_sel.shape[1]
        src = torch.gather(node_lat, 1, src_sel.unsqueeze(-1).expand(B, E, d))
        dst = torch.gather(node_lat, 1, dst_idx.unsqueeze(-1).expand(B, E, d))
        er = self._rel_input(torch.gather(edge_lat, 1, src_sel.unsqueeze(-1).expand(B, E, d)), dst)
        if self.read_mode == "raw":
            return self._tok(er, self.id_half[src_sel], self.id_half[dst_idx], 2, weight=A_blk)
        phi = self.phi(torch.cat([src, dst, er], dim=-1))
        lab = id_scale * (nid[src_sel] + nid[dst_idx])
        if self.gate_ids:
            return A_blk.unsqueeze(-1) * (phi + lab) + role[2]
        return A_blk.unsqueeze(-1) * phi + lab + role[2]

    def _expand_global(self, node_lat, edge_lat, nid, id_scale, role, A):
        """GLOBAL edge budget + per-source FLOOR + ST leak — the read stays at E tokens for ANY node count.
        Selection = every source's top-1 edge (degree floor: no node's forward starves, similarity cliques
        can't monopolize the budget) + the globally strongest remaining edges (hubs still allowed where mass
        genuinely concentrates). Backward leaks through a single global rest-mixture weighted by the
        unselected sparsemax mass (chunked over sources so the K² φ tensor never materializes).
        Self-annealing as in _expand_st: routing sharp → rest mass → 0 → estimator exact."""
        B, K, d = node_lat.shape; E = self.edge_budget
        flat = A.reshape(B, K * K)
        if E >= K:                                                       # floor: top-1 per source, then global rest
            f1 = A.argmax(-1)                                            # [B,K] each source's strongest dst
            floor_i = torch.arange(K, device=A.device).unsqueeze(0) * K + f1
            rest_pool = flat.scatter(1, floor_i, -1.0)                   # exclude floor edges from the rest pick
            rest_i = rest_pool.topk(E - K, dim=-1).indices
            flat_i = torch.cat([floor_i, rest_i], dim=1)                 # [B,E]
            topv = flat.gather(1, flat_i)
        else:
            topv, flat_i = flat.topk(E, dim=-1)                          # pure global (small budgets)
        src_sel, dst_idx = flat_i // K, flat_i % K
        H = self._token_block(node_lat, edge_lat, nid, id_scale, role, topv, src_sel, dst_idx)
        sel = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, flat_i, True).reshape(B, K, K)
        rest_w = A * (~sel)                                              # unselected mass [B,K,K]
        tot_rest = rest_w.sum((-1, -2))                                  # [B]
        eps = (tot_rest / A.sum((-1, -2)).clamp_min(1e-9)).view(B, 1, 1)
        # global rest mixture, chunked over source nodes (φ on [B,chunk,K,3d]) — each chunk is
        # gradient-CHECKPOINTED: the K² activations are recomputed in backward instead of retained
        # (~2 GB/call otherwise; the expansion is cheap FLOPs, expensive activations)
        R = torch.zeros(B, d, device=A.device, dtype=H.dtype)
        w_norm = rest_w / tot_rest.view(B, 1, 1).clamp_min(1e-9)
        for c0 in range(0, K, 32):
            c1 = min(c0 + 32, K); C = c1 - c0

            def chunk_fn(nl, el, Af, wf, c0=c0, c1=c1, C=C):
                src_f = nl[:, c0:c1].unsqueeze(2).expand(B, C, K, d)
                dst_f = nl.unsqueeze(1).expand(B, C, K, d)
                er_f = self._rel_input(el[:, c0:c1].unsqueeze(2).expand(B, C, K, d), dst_f)
                Ab = Af[:, c0:c1]
                if self.read_mode == "raw":
                    h = d // 2
                    Tf = self._tok(er_f, self.id_half[c0:c1].view(1, C, 1, h).expand(B, C, K, h),
                                   self.id_half.view(1, 1, K, h).expand(B, C, K, h), 2, weight=Ab)
                else:
                    phi_f = self.phi(torch.cat([src_f, dst_f, er_f], dim=-1))
                    lab_f = id_scale * (nid[c0:c1].view(1, C, 1, d) + nid.view(1, 1, K, d))
                    Tf = Ab.unsqueeze(-1) * (phi_f + lab_f) + role[2] if self.gate_ids \
                        else Ab.unsqueeze(-1) * phi_f + lab_f + role[2]
                return torch.einsum("bck,bckd->bd", wf[:, c0:c1], Tf)

            if self.training and torch.is_grad_enabled():
                R = R + _ckpt.checkpoint(chunk_fn, node_lat, edge_lat, A, w_norm, use_reentrant=False)
            else:
                R = R + chunk_fn(node_lat, edge_lat, A, w_norm)
        Msoft = (1.0 - eps) * H + eps * R.unsqueeze(1)                   # [B,E,d]
        out = (H - Msoft).detach() + Msoft                               # ST: forward = H, backward = ∂M
        return out, (topv > 0), A, topv

    def _lm(self, seq, keep):
        def _run(s, m):
            return self.base.model(inputs_embeds=s, attention_mask=m, use_cache=False).last_hidden_state
        if self.training and torch.is_grad_enabled():
            return _ckpt.checkpoint(_run, seq, keep, use_reentrant=False)
        return _run(seq, keep)

    def _lm_suffix(self, we, wm, graph_in, keep_g, active):
        """LAST-N-LAYERS LM write. Text runs the frozen prefix layers ALONE (no-grad — prefix LoRA
        receives no gradient, so lora_B stays 0 = exact pretrained identity); the graph tokens splice
        in for the last N layers, whose LoRA is the trainable mixer. Depth-matched to the custom arm
        at N=4 and ~2× cheaper than the full ride; RoPE positions match the full ride exactly
        (text 0..T-1, graph tokens T..T+G-1)."""
        m = self.base.model
        B, T, _ = we.shape; G = graph_in.shape[1]
        split = len(m.layers) - self.write_layers
        neg = torch.finfo(we.dtype).min
        # ── prefix: text only, no grad; idle rows keep position 0 valid (all-masked rows → NaN) ──
        wm_enc = wm.clone(); wm_enc[:, 0] |= ~active
        causal_t = torch.tril(torch.ones(T, T, device=we.device, dtype=torch.bool))
        allow_t = causal_t.unsqueeze(0) & wm_enc.bool().unsqueeze(1)
        mask_t = torch.zeros(B, 1, T, T, device=we.device, dtype=we.dtype)
        mask_t.masked_fill_(~allow_t.unsqueeze(1), neg)
        pos_t = torch.arange(T, device=we.device).unsqueeze(0)
        with torch.no_grad():
            h = we
            cs_t = m.rotary_emb(h, pos_t)
            for lyr in m.layers[:split]:
                h = lyr(h, attention_mask=mask_t, position_embeddings=cs_t, use_cache=False)
        # ── suffix: [text hiddens ; graph tokens], grad + LoRA. Graph tokens are scale-matched to
        # the text stream (measured 400× quieter raw — identity swamped after one layer), and with
        # write_bb the graph block attends to itself BIDIRECTIONALLY (matches the read geometry —
        # under plain causal the expanded edges could never see the node slots at all). ──
        if self.write_norm_match:
            tnorm = ((h.float().norm(dim=-1) * wm).sum(1) / wm.sum(1).clamp_min(1)).detach()  # [B]
            g_dir = graph_in / graph_in.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            graph_in = g_dir * (tnorm.view(-1, 1, 1) * self.write_scale)
        h = torch.cat([h.detach(), graph_in.to(h.dtype)], dim=1)
        S = T + G
        keep = torch.cat([wm.bool(), keep_g], dim=1)
        causal = torch.tril(torch.ones(S, S, device=we.device, dtype=torch.bool))
        if self.write_bb:
            causal[T:, T:] = True                                        # graph block: full bidirectional
        allow = causal.unsqueeze(0) & keep.unsqueeze(1)
        allow = allow | torch.eye(S, device=we.device, dtype=torch.bool).unsqueeze(0)
        mask4 = torch.zeros(B, 1, S, S, device=we.device, dtype=we.dtype)
        mask4.masked_fill_(~allow.unsqueeze(1), neg)
        pos = torch.arange(S, device=we.device).unsqueeze(0)
        cs = m.rotary_emb(h, pos)

        def run_layer(x, lyr):
            return lyr(x, attention_mask=mask4, position_embeddings=cs, use_cache=False)

        for lyr in m.layers[split:]:
            if self.training and torch.is_grad_enabled():
                h = _ckpt.checkpoint(run_layer, h, lyr, use_reentrant=False)
            else:
                h = run_layer(h, lyr)
        h = m.norm(h)
        return h, h[:, :T]                                               # (full sequence, text hiddens)

    def _custom(self, graph, we, wm, keep_g, active):
        """Custom write mixer: (1) frozen LM contextualizes the window (NO grad, no LoRA) → last-layer hiddens
        (word order lives in their content — the blocks need no positional encoding); (2) the graph tokens
        self-attend + cross-attend over [hiddens ; graph] through the from-scratch blocks. Returns graph rows."""
        with torch.no_grad():
            wm_enc = wm.clone()
            wm_enc[:, 0] |= ~active                                      # idle rows: ≥1 valid key for the frozen
            ctx = self.base.model(inputs_embeds=we, attention_mask=wm_enc.long(),   # forward (all-masked → NaN)
                                  use_cache=False).last_hidden_state
        keep = torch.cat([wm, keep_g], dim=1)                            # ORIGINAL wm: idle rows see graph keys only
        g = graph
        for blk in self.blocks:
            if self.training and torch.is_grad_enabled():
                g = _ckpt.checkpoint(blk, g, ctx, keep, use_reentrant=False)
            else:
                g = blk(g, ctx, keep)
        return g, ctx                                                    # (graph rows, frozen text hiddens)

    def finalize_memory(self, state):
        emb, mask = state["emb"], state["mask"]
        B, T, d = emb.shape
        if T == 0:
            raise ValueError("slotgraph3.finalize_memory: empty context (T=0)")
        node_lat = self.node_lat_init.float().unsqueeze(0).expand(B, -1, -1).contiguous()
        edge_lat = self.edge_lat_init.float().unsqueeze(0).expand(B, -1, -1).contiguous()
        if self.init_noise and self.training:
            # per-forward sampling around the learned init (Slot Attention symmetry breaking):
            # deterministic identical slots + shared heads receive identical gradients forever.
            node_lat = node_lat + self.node_lat_logsig.float().exp() * torch.randn_like(node_lat)
            edge_lat = edge_lat + self.edge_lat_logsig.float().exp() * torch.randn_like(edge_lat)
        if self.read_mode == "edges":
            nid = F.normalize(self.node_id.float(), dim=-1)
            role = self.role.float(); id_scale = self.id_scale.float()
        else:                                                  # raw: identity lives in id_half/type_embed
            nid = role = id_scale = None
        K = self.K
        _assoc_alpha = _assoc_keycos = None                    # assoc-write telemetry (set per window)
        for w in range(0, T, self.window):
            wm = mask[:, w:w + self.window].bool()
            active = wm.any(dim=1)
            if not bool(active.any()):
                continue
            we = emb[:, w:w + self.window]
            with torch.autocast("cuda", enabled=False):                 # graph tokenization in fp32
                # raw mode: node/edge SLOTS use the SAME concat-projection + id basis as the pointer
                # edge tokens and the read node tokens (TokenGT node = [X ‖ P ‖ P ‖ type]) so the write
                # mixer shares ONE identity system across all graph tokens. edges mode (legacy): the
                # additive full-d node_id + role superposition.
                if self.read_mode == "raw":
                    idh = self.id_half.unsqueeze(0).expand(B, -1, -1)
                    node_tok = self._tok(node_lat, idh, idh, 0)
                    edge_tok = self._tok(edge_lat, idh, idh, 1)
                else:
                    node_tok = node_lat + id_scale * nid.unsqueeze(0) + role[0]
                    edge_tok = edge_lat + id_scale * nid.unsqueeze(0) + role[1]
                if self.write_expand:
                    E, keep_e, A_w, _ = self._expand_topk(node_lat, edge_lat, nid, id_scale, role)
                    graph_in = torch.cat([E, node_tok, edge_tok], dim=1) # [B, Kk+2K, d] — slots LAST (see edges)
                    keep_g = torch.cat([keep_e, torch.ones(B, 2 * K, device=we.device, dtype=torch.bool)], dim=1)
                else:                                                    # write over [window; slots] only;
                    A_w = None                                           # (assoc write recomputes the wiring)
                    graph_in = torch.cat([node_tok, edge_tok], dim=1)    # graph expanded for the READ only
                    keep_g = torch.ones(B, 2 * K, device=we.device, dtype=torch.bool)
                if self.write_bb and self.boundary_tokens:
                    # dialect harmonization: mark the text|graph span start in the WRITE too
                    bs = self.boundary[0].float().view(1, 1, -1).expand(B, 1, -1)
                    graph_in = torch.cat([bs, graph_in], dim=1)          # slots stay LAST (update read-out)
                    keep_g = torch.cat([torch.ones(B, 1, device=we.device, dtype=torch.bool), keep_g], dim=1)
            if self.write_mode == "lm" and self.write_layers > 0:        # last-N-layers suffix splice
                H, th = self._lm_suffix(we, wm, graph_in, keep_g, active)
            elif self.write_mode == "lm":                                # joint [window; graph] full-ride
                if self.write_norm_match:                                # here graph joins at the EMBED scale
                    tn = ((we.float().norm(dim=-1) * wm).sum(1) / wm.sum(1).clamp_min(1)).detach()
                    graph_in = (graph_in / graph_in.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                                ) * (tn.view(-1, 1, 1) * self.write_scale)
                seq = torch.cat([we, graph_in.to(we.dtype)], dim=1)
                keep = torch.cat([wm, keep_g], dim=1)
                H = self._lm(seq, keep.long())                           # frozen LM (+LoRA) does the mixing
                th = H[:, :we.shape[1]]
            else:                                                        # frozen encode → custom graph mixer
                H, th = self._custom(graph_in.to(we.dtype), we, wm, keep_g, active)
            with torch.autocast("cuda", enabled=False):
                slots = H[:, -2 * K:].float()                            # node-slot + edge-slot hiddens
                nl0, el0 = node_lat, edge_lat
                if self.write_update == "delta":
                    # competitive read: slots COMPETE (softmax over the SLOT axis) to claim each window
                    # token — Slot Attention's anti-duplication mechanism; weighted-mean over tokens.
                    th_f = th.float()
                    qs = self.comp_q(slots)                              # [B,2K,dk]
                    ks = self.comp_k(th_f); vs = self.comp_v(th_f)       # [B,Tw,·]
                    attn_sl = torch.softmax(torch.einsum("bsd,btd->bst", qs, ks) / math.sqrt(self.dk), dim=1)
                    attn_sl = attn_sl * wm.unsqueeze(1).float()          # kill pad tokens
                    wgt = attn_sl / attn_sl.sum(-1, keepdim=True).clamp_min(1e-6)
                    comp = torch.einsum("bst,btd->bsd", wgt, vs)         # [B,2K,d] per-slot claimed content
                    gh = self.n_head(slots + self.comp_proj(comp))       # comp_proj zero-init: step-0 = slots-only
                    g_wr = torch.sigmoid(self.gate_head(gh))             # [B,2K,1] per-slot CONTENT gate (b=+1.5)
                    cand_n = self.head_node(gh[:, :K])
                    node_lat = node_lat + g_wr[:, :K] * (cand_n - node_lat)   # gated INTERPOLATION: repeated
                    if self.edge_write == "assoc":                       # same-content writes CONVERGE
                        # T3 keyed delta-rule write: file this window's relation UNDER THE PARTNER
                        # ADDRESS the routing actually selected — the same unit key the read uses.
                        #   M_i ← (1−α)·M_i + g_i·(v − M_i·k̂)⊗k̂
                        # outer product = write into k̂'s slice only (no smear across the 576 dims);
                        # (v − retrieved) = overwrite what that address held (update, don't average);
                        # M_i persists across windows = state NOT derivable from node content.
                        if A_w is None:                                  # write_expand=False: wiring not built
                            A_w = self._route(nl0 if self.route_key == "node" else el0)
                        kdst = F.normalize(self.rel_key(nl0), dim=-1)    # [B,K,r] partner unit keys (= read's)
                        v_rel = self.rel_val(gh[:, K:])                  # [B,K,r] the filed relation code
                        Mi = el0.reshape(B, K, self.r, self.r)
                        # PER-DESTINATION keyed delta writes: file under EVERY partner address the read
                        # will query, weighted by the window's wiring (Σ_n A=1 bounds total write mass).
                        # A single mixture address would mismatch the read: reads query each k̂_n
                        # individually, so diffuse routing would file facts nobody looks up. [review fix]
                        #   corr_i = Σ_n A[i,n]·(v_i − M_i·k̂_n) ⊗ k̂_n
                        # Under repeated writes at a fixed address, retrieval converges to g/(g+α)·v
                        # (≈0.87·v at init; EXACT overwrite in the g=1, α=0 limit) — the delta form's
                        # fixed point is v-bounded where an additive write grows without bound.
                        retr = torch.einsum("birs,bns->binr", Mi, kdst)  # [B,K(i),K(n),r] filed@every addr
                        alpha = torch.sigmoid(self.decay_head(gh[:, K:]))          # [B,K,1] retention-biased
                        err = v_rel.unsqueeze(2) - retr                  # [B,K,K,r] correction per address
                        corr = torch.einsum("bin,binr,bns->birs", A_w, err, kdst)
                        Mi = Mi * (1.0 - alpha).unsqueeze(-1) + g_wr[:, K:].unsqueeze(-1) * corr
                        edge_lat = Mi.reshape(B, K, d)
                        _assoc_alpha = float(alpha.mean())               # telemetry (last window's values)
                        kk = kdst @ kdst.transpose(1, 2)
                        _koff = ~torch.eye(K, dtype=torch.bool, device=kk.device)
                        _assoc_keycos = float(kk[:, _koff].mean())       # address distinctness (↓ = distinct)
                    else:                                                # LEGACY slot interpolation
                        cand_e = self.head_edge(gh[:, K:])
                        edge_lat = edge_lat + g_wr[:, K:] * (cand_e - edge_lat)
                else:                                                    # LEGACY additive highway
                    gh = self.n_head(slots); gn, ge = gh[:, :K], gh[:, K:]
                    node_lat = node_lat + torch.sigmoid(self.beta_node) * self.head_node(gn)
                    edge_lat = edge_lat + torch.sigmoid(self.beta_edge) * self.head_edge(ge)
                if not bool(active.all()):                               # freeze rows idle this window
                    a = active[:, None, None]
                    node_lat = torch.where(a, node_lat, nl0)
                    edge_lat = torch.where(a, edge_lat, el0)
                if self._trace is not None:                              # opt-in per-window grad-credit tracer
                    node_lat.retain_grad(); edge_lat.retain_grad()
                    self._trace.append((w // self.window, node_lat, edge_lat))
        with torch.autocast("cuda", enabled=False):
            E, keep_read, A, topv = self._expand_topk(node_lat, edge_lat, nid, id_scale, role)
            if self.read_mode == "raw":
                # RAW read: node latents themselves are the content tokens (pretrained-space reps, direct
                # decoder→latent gradient, no φ bottleneck); E are pointer tokens. Same [id ‖ id] slots as
                # the pointer edges' [id_src ‖ id_dst] so the decoder matches them by inner product.
                idh = self.id_half.unsqueeze(0).expand(B, -1, -1)
                node_read = self._tok(node_lat, idh, idh, 0)
                E = torch.cat([node_read, E], dim=1)
                keep_read = torch.cat([torch.ones(B, self.K, dtype=torch.bool, device=E.device),
                                       keep_read], dim=1)
            memory = self.norm(E)                                        # [B, M_content, d] prepend tokens
            if self.boundary_tokens:
                # learned on-manifold <mem_start>/<mem_end> wrap the span; canaries stay on content rows
                bnd = self.norm(self.boundary.float()).unsqueeze(0).expand(B, -1, -1)
                memory_full = torch.cat([bnd[:, :1], memory, bnd[:, 1:]], dim=1)
            else:
                memory_full = memory
        aux = self._canaries(memory, node_lat, edge_lat, A, topv, emb.device)
        # mask out topv==0 edge tokens (arise once sparsemax support < read_topk) — else they'd be prepended
        # as full-norm, content-free, arbitrary-destination distractors. Matches the WRITE keep-mask.
        if self.boundary_tokens:
            ones = torch.ones(B, 1, dtype=torch.bool, device=memory.device)
            keep_read = torch.cat([ones, keep_read, ones], dim=1)
        aux["memory_mask"] = keep_read                                   # [B, M] bool; consumed by model.py prepend/SHUF
        aux["latents"] = (node_lat, edge_lat)                            # trajectory-mode rollout sampling reads these
        if self.edge_write == "assoc" and _assoc_alpha is not None:
            aux["slotgraph3_edge_alpha"] = _assoc_alpha                  # mean decay (retention health)
            aux["slotgraph3_edge_keycos"] = _assoc_keycos                # write-address diversity (↓ distinct)
        return memory_full, aux

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        st = self.init_streaming_state(token_embeds.shape[0], token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)

    def sample_read_expansion(self, node_lat, edge_lat, n_samples):
        """TRAJECTORY-mode rollouts: sample G read expansions from the routing distribution.

        The discrete read choice (which k dests each source wires to) has no honest gradient —
        sparsemax+top-k is a biased relaxation and routing_diversity stays pinned (the argmax-starved
        router). REINFORCE needs SAMPLES: perturb the routing logits with Gumbel noise and take top-k
        per source (Gumbel-top-k = sampling k without replacement from the Plackett-Luce induced by
        softmax(sc)). Sampled-edge weights use softmax probs (always >0; sparsemax would zero
        off-support samples into content-free tokens) — internally consistent across the group, which
        is all the group-relative advantage needs.

        Returns ([(memory, keep_mask, logprob[B])] × G, router_entropy). logprob = the EXACT
        Plackett-Luce log-probability of the without-replacement sample: Σ_sources Σ_picks
        [sc(chosen_r) − logsumexp(sc over the REMAINING set at pick r)]. (2026-07-03 audit fix:
        the independent-softmax surrogate is NOT the sample's log-prob — the group baseline
        cancels the common −k·logZ but NOT the sample-dependent remaining-set renormalization,
        which at k/K≈½ is large and biases the gradient TOWARD hub concentration, the exact
        pathology being fought. Gumbel-top-k = a partial PL ranking and topk returns picks in
        PL order — Kool et al. 2019.) router_entropy = mean per-source softmax entropy, grad-
        capable — the caller's entropy bonus fights collapse (Gumbel noise alone stops exploring
        once logits sharpen). GRADIENT: logprob/entropy flow to the ROUTER (q_route/k_proj)
        ONLY — pass latents DETACHED; memories are built under no_grad (scored, not backpropped).
        """
        if self.read_mode != "raw":
            raise ValueError("trajectory sampling is implemented for the raw read")
        if self.edge_budget != 0:
            raise ValueError("trajectory sampling uses per-node top-k (set edge_budget=0)")
        B, K, d = node_lat.shape
        k = self.read_topk; h = d // 2
        node_lat = node_lat.float(); edge_lat = edge_lat.float()
        key_lat = node_lat if self.route_key == "node" else edge_lat
        q = self.q_route(key_lat); kk = self.k_proj(key_lat)
        sc = torch.einsum("bik,bjk->bij", q, kk) / math.sqrt(self.dk) + self.diag_mask.unsqueeze(0)
        p_route = torch.softmax(sc, dim=-1)                              # [B,K,K]
        entropy = -(p_route.clamp_min(1e-9).log() * p_route).sum(-1).mean()   # grad → router
        out = []
        for _ in range(int(n_samples)):
            u = torch.rand_like(sc).clamp_(1e-9, 1.0 - 1e-9)
            gumbel = -torch.log(-torch.log(u))
            topi = (sc.detach() + gumbel).topk(k, dim=-1).indices        # [B,K,k] picks in PL order
            # exact Plackett-Luce log-prob: per pick, normalize over the REMAINING candidates
            remain = torch.zeros_like(sc)                                # additive -inf mask of used picks
            logp = torch.zeros(sc.shape[0], device=sc.device)
            for rpick in range(k):
                j_r = topi[..., rpick:rpick + 1]                         # [B,K,1]
                lse = torch.logsumexp(sc + remain, dim=-1)               # [B,K] remaining-set normalizer
                logp = logp + (sc.gather(-1, j_r).squeeze(-1) - lse).sum(dim=-1)
                remain = remain.scatter(-1, j_r, float("-inf"))          # remove the pick
            with torch.no_grad():
                w = torch.softmax(sc, dim=-1).gather(-1, topi)           # [B,K,k] sampled-edge weights
                dst = torch.gather(node_lat.unsqueeze(1).expand(B, K, K, d), 2,
                                   topi.unsqueeze(-1).expand(B, K, k, d))
                er = self._rel_input(edge_lat.unsqueeze(2).expand(B, K, k, d), dst)
                E = self._tok(er, self.id_half.view(1, K, 1, h).expand(B, K, k, h),
                              self.id_half[topi], 2, weight=w).reshape(B, K * k, d)
                idh = self.id_half.unsqueeze(0).expand(B, -1, -1)
                node_read = self._tok(node_lat, idh, idh, 0)
                mem = self.norm(torch.cat([node_read, E], dim=1))
                if self.boundary_tokens:
                    bnd = self.norm(self.boundary.float()).unsqueeze(0).expand(B, -1, -1)
                    mem = torch.cat([bnd[:, :1], mem, bnd[:, 1:]], dim=1)
                keep = torch.ones(B, mem.shape[1], dtype=torch.bool, device=mem.device)
            out.append((mem, keep, logp))
        return out, entropy

    @torch.no_grad()
    def _canaries(self, memory, node_lat, edge_lat, A, topv, device):
        def _within_cos(x):
            S = x.shape[1]
            if S < 2:
                return 0.0
            xn = F.normalize(x, dim=-1); cos = xn @ xn.transpose(-1, -2)
            off = cos.sum((-1, -2)) - cos.diagonal(dim1=-2, dim2=-1).sum(-1)
            return float((off / (S * (S - 1))).mean())
        aux = {"slotgraph3_mem_effrank": torch.tensor(_participation_ratio(memory.reshape(-1, self.d)), device=device),
               "slotgraph3_node_effrank": torch.tensor(_participation_ratio(node_lat.reshape(-1, self.d)), device=device),
               "slotgraph3_edge_effrank": torch.tensor(_participation_ratio(edge_lat.reshape(-1, self.d)), device=device),
               "slotgraph3_node_cos": torch.tensor(_within_cos(node_lat), device=device),
               "slotgraph3_edge_cos": torch.tensor(_within_cos(edge_lat), device=device)}
        # support = mass above uniform (softmax A is dense — (A>0) would read K always; this
        # threshold reduces to the old sparsemax support when A is sparse)
        supp = (A > 1.0 / self.K).float()
        aux["slotgraph3_edges_per_node"] = supp.sum(-1).mean()          # effective support (↓ = sharper wiring)
        if topv.dim() == 2:                                             # global edge budget: [B,E] → captured FRACTION
            aux["slotgraph3_topk_mass"] = (topv.sum(-1) / A.sum((-1, -2)).clamp_min(1e-9)).mean()
        else:
            aux["slotgraph3_topk_mass"] = topv.sum(-1).mean()           # A-mass captured by the prepended top-k (↑ good)
        dp = A.argmax(-1)
        use = torch.bincount(dp.reshape(-1), minlength=self.K).float()
        pu = use / use.sum().clamp_min(1e-9)
        aux["slotgraph3_node_entropy"] = -(pu.clamp_min(1e-9).log() * pu).sum()
        aux["slotgraph3_nodes_used"] = torch.tensor(float(dp.reshape(-1).unique().numel()), device=device)
        if dp.shape[0] > 1:                                             # KEY — input-dependence of the wiring
            oh = F.one_hot(dp, self.K).float().mean(0)
            aux["slotgraph3_routing_diversity"] = (-(oh.clamp_min(1e-9).log() * oh).sum(-1)).mean() / math.log(self.K)
        return aux
