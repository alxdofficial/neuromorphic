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
  WRITE:   per streaming window, run ONE frozen-LM (+encoder-LoRA) forward over
             [ window-tokens ; expanded-edges ; node-slot-tokens ; edge-slot-tokens ]
           (causal, so the passage stays pure and the graph tokens attend back over the passage + structure);
           read the slot hiddens → ADDITIVE-residual update of node/edge latents. The PRETRAINED attention is
           the graph mixer (the mechanism that empirically binds); re-expanded each window.
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


class SlotGraph3Encoder(nn.Module):
    is_conditioned_read = False              # READ = PREPEND the expanded edge tokens

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        from ...decoder import load_frozen_llama, apply_lora_to_llama
        base, _ = load_frozen_llama(cfg.llama_model)
        for p in base.parameters():
            p.requires_grad_(False)
        n_wrapped = apply_lora_to_llama(base, rank=cfg.slotgraph3_lora_rank, alpha=cfg.slotgraph3_lora_alpha,
                                        target_names=tuple(cfg.llama_lora_target_names))
        base.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.base = base
        d = cfg.d_llama; self.d = d
        self.K = int(cfg.slotgraph3_n_nodes); self.window = int(cfg.slotgraph3_window)
        self.dk = int(cfg.slotgraph3_d_key); self.read_topk = int(cfg.slotgraph3_read_topk)
        self.write_expand = bool(getattr(cfg, "slotgraph3_write_expand", True))
        self.M = self.K * self.read_topk                       # prepend budget = edges kept per node × nodes
        self.force_identity_A = False                          # eval-only control: A := I (edges decorative)

        nid = torch.empty(self.K, d); nn.init.orthogonal_(nid)
        self.node_id = nn.Parameter(F.normalize(nid, dim=-1))
        self.role = nn.Parameter(torch.randn(3, d) / math.sqrt(d))   # [node-slot, edge-slot, edge-token]
        self.register_buffer("diag_mask", torch.eye(self.K) * -1e9, persistent=False)

        embed = base.get_input_embeddings()
        with torch.no_grad():
            mean_vec = embed.weight.float().mean(0); emb_std = embed.weight.float().std().item()
            emb_norm = embed.weight.float().norm(dim=-1).mean().item()
        # id_scale ~ the LM EMBEDDING row-norm: graph tokens ride the frozen LM as inputs_embeds, so the id tag
        # must sit at the embedding scale. (NOT √d — that was slotgraph2's HIDDEN-scale constant; here latents
        # are embed-scale, and √d·unit-id ≈ 7-10× the embed norm → the LM reads coordinate labels with content drowned.)
        self.id_scale = nn.Parameter(torch.tensor(emb_norm))
        self.node_lat_init = nn.Parameter(mean_vec.view(1, d).repeat(self.K, 1) + emb_std * torch.randn(self.K, d))
        self.edge_lat_init = nn.Parameter(mean_vec.view(1, d).repeat(self.K, 1) + emb_std * torch.randn(self.K, d))
        # routing = 2-layer MLPs (the wiring decision is the thesis crux → give it real capacity, not a bare bilinear)
        self.q_route = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, self.dk))   # source outgoing intent
        self.k_proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, self.dk))    # dest key (input-dependent)
        self.phi = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))   # (src,dst,edge) → edge content
        # write heads: frozen-LM slot hiddens → ADDITIVE-residual latent delta (identity gradient highway)
        self.n_head = nn.LayerNorm(d)
        self.head_node = nn.Linear(d, d)
        self.head_edge = nn.Linear(d, d)
        self.beta_node = nn.Parameter(torch.tensor(-1.2))
        self.beta_edge = nn.Parameter(torch.tensor(-1.2))

        self._trace = None                                     # set to [] to record per-window (node_lat, edge_lat) for grad-credit
        self.norm = _NormMatch(d)
        with torch.no_grad():
            self.norm.scale.data.fill_(emb_norm)
        print(f"[slotgraph3] {self.K} nodes (2×{self.K} latents) @ d={d}; LM-attention write "
              f"({'+edges in context' if self.write_expand else 'slots-only, edges at READ only'}); "
              f"expand→edges (sparsemax + top-{self.read_topk}/node = {self.M} tokens); "
              f"PREPEND read; encoder-LoRA r{cfg.slotgraph3_lora_rank} ({n_wrapped} layers)")

    def train(self, mode: bool = True):
        super().train(mode); self.base.train(False); return self

    def init_streaming_state(self, batch_size, device, dtype):
        return {"emb": torch.zeros(batch_size, 0, self.d, device=device, dtype=dtype),
                "mask": torch.zeros(batch_size, 0, device=device, dtype=torch.bool)}

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0, **extra):
        del chunk_offset, extra
        if attention_mask is None:
            attention_mask = torch.ones(token_embeds.shape[:2], device=token_embeds.device, dtype=torch.bool)
        return {"emb": torch.cat([state["emb"], token_embeds], dim=1),
                "mask": torch.cat([state["mask"], attention_mask.bool()], dim=1)}, {}

    # ── expansion: compressed latents → content-addressed sparse wiring A → top-k explicit edge tokens ──
    def _route(self, edge_lat):
        q = self.q_route(edge_lat)                                       # [B,K,dk] source outgoing intent
        k = self.k_proj(edge_lat)                                        # [B,K,dk] dest key (input-dependent)
        sc = torch.einsum("bik,bjk->bij", q, k) / math.sqrt(self.dk)     # [B,K,K]
        sc = sc + self.diag_mask.unsqueeze(0)                            # forbid self-loops
        A = sparsemax(sc)                                                # sparse, per-source over dests
        if self.force_identity_A:
            A = torch.eye(self.K, device=A.device, dtype=A.dtype).unsqueeze(0).expand_as(A)
        return A

    def _expand_topk(self, node_lat, edge_lat, nid, id_scale, role):
        """Top-k edge tokens per node: [B, K·topk, d] + keep mask + (A, topk-weights) for canaries."""
        B, K, d = node_lat.shape; k = self.read_topk
        A = self._route(edge_lat)                                        # [B,K,K]
        topv, topi = A.topk(k, dim=-1)                                   # [B,K,k] weights + dst indices
        src = node_lat.unsqueeze(2).expand(B, K, k, d)                   # source content (node i)
        dst = torch.gather(node_lat.unsqueeze(1).expand(B, K, K, d), 2,
                           topi.unsqueeze(-1).expand(B, K, k, d))        # dst content (node topi)
        er = edge_lat.unsqueeze(2).expand(B, K, k, d)                    # source edge role
        phi = self.phi(torch.cat([src, dst, er], dim=-1))               # [B,K,k,d]
        id_src = nid.unsqueeze(1).expand(K, k, d).unsqueeze(0)          # [1,K,k,d]
        id_dst = nid[topi]                                              # [B,K,k,d]
        E = topv.unsqueeze(-1) * phi + id_scale * (id_src + id_dst) + role[2]
        return E.reshape(B, K * k, d), (topv > 0).reshape(B, K * k), A, topv

    def _lm(self, seq, keep):
        def _run(s, m):
            return self.base.model(inputs_embeds=s, attention_mask=m, use_cache=False).last_hidden_state
        if self.training and torch.is_grad_enabled():
            return _ckpt.checkpoint(_run, seq, keep, use_reentrant=False)
        return _run(seq, keep)

    def finalize_memory(self, state):
        emb, mask = state["emb"], state["mask"]
        B, T, d = emb.shape
        if T == 0:
            raise ValueError("slotgraph3.finalize_memory: empty context (T=0)")
        node_lat = self.node_lat_init.float().unsqueeze(0).expand(B, -1, -1).contiguous()
        edge_lat = self.edge_lat_init.float().unsqueeze(0).expand(B, -1, -1).contiguous()
        nid = F.normalize(self.node_id.float(), dim=-1)
        role = self.role.float(); id_scale = self.id_scale.float()
        K = self.K
        for w in range(0, T, self.window):
            wm = mask[:, w:w + self.window].bool()
            active = wm.any(dim=1)
            if not bool(active.any()):
                continue
            we = emb[:, w:w + self.window]
            with torch.autocast("cuda", enabled=False):                 # graph tokenization in fp32
                node_tok = node_lat + id_scale * nid.unsqueeze(0) + role[0]
                edge_tok = edge_lat + id_scale * nid.unsqueeze(0) + role[1]
                if self.write_expand:
                    E, keep_e, _, _ = self._expand_topk(node_lat, edge_lat, nid, id_scale, role)
                    graph_in = torch.cat([E, node_tok, edge_tok], dim=1) # [B, Kk+2K, d] — slots LAST (see edges)
                    keep_g = torch.cat([keep_e, torch.ones(B, 2 * K, device=we.device, dtype=torch.bool)], dim=1)
                else:                                                    # write over [window; slots] only;
                    graph_in = torch.cat([node_tok, edge_tok], dim=1)    # graph expanded for the READ only
                    keep_g = torch.ones(B, 2 * K, device=we.device, dtype=torch.bool)
            seq = torch.cat([we, graph_in.to(we.dtype)], dim=1)          # bf16 LM sequence
            keep = torch.cat([wm, keep_g], dim=1)
            H = self._lm(seq, keep.long())                               # frozen LM (+LoRA) does the mixing
            with torch.autocast("cuda", enabled=False):
                slots = H[:, -2 * K:].float()                            # node-slot + edge-slot hiddens
                nl0, el0 = node_lat, edge_lat
                gh = self.n_head(slots); gn, ge = gh[:, :K], gh[:, K:]
                node_lat = node_lat + torch.sigmoid(self.beta_node) * self.head_node(gn)   # additive highway
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
            memory = self.norm(E)                                        # [B, K·topk, d] prepend tokens
        aux = self._canaries(memory, node_lat, edge_lat, A, topv, emb.device)
        # mask out topv==0 edge tokens (arise once sparsemax support < read_topk) — else they'd be prepended
        # as full-norm, content-free, arbitrary-destination distractors. Matches the WRITE keep-mask.
        aux["memory_mask"] = keep_read                                   # [B, K·topk] bool; consumed by model.py prepend/SHUF
        return memory, aux

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        st = self.init_streaming_state(token_embeds.shape[0], token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)

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
        supp = (A > 0).float()
        aux["slotgraph3_edges_per_node"] = supp.sum(-1).mean()          # sparsemax support (↓ = sharper wiring)
        aux["slotgraph3_topk_mass"] = topv.sum(-1).mean()               # A-mass captured by the prepended top-k (↑ good)
        dp = A.argmax(-1)
        use = torch.bincount(dp.reshape(-1), minlength=self.K).float()
        pu = use / use.sum().clamp_min(1e-9)
        aux["slotgraph3_node_entropy"] = -(pu.clamp_min(1e-9).log() * pu).sum()
        aux["slotgraph3_nodes_used"] = torch.tensor(float(dp.reshape(-1).unique().numel()), device=device)
        if dp.shape[0] > 1:                                             # KEY — input-dependence of the wiring
            oh = F.one_hot(dp, self.K).float().mean(0)
            aux["slotgraph3_routing_diversity"] = (-(oh.clamp_min(1e-9).log() * oh).sum(-1)).mean() / math.log(self.K)
        return aux
