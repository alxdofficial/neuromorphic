"""slotgraph v2 — graph-as-language. See docs/slotgraph_redesign.md.

The memory is a graph-structured *utterance* in an invented vocabulary: many small (d_node) node/edge
latents wired into a graph, that re-describes the passage.

WRITE (encoder):
  • the frozen LM (encoder-LoRA) PERCEIVES the passage → contextual features H_ctx [B,T,d_llama];
  • a small d_node graph-transformer (enc_layers) holds N node + E edge latents, and per layer:
    cross-attends to H_ctx (gather observation) → self-attends (mix) → predicts each edge's src/dst
    endpoints (content-addressed: edge queries · node keys → log-Sinkhorn competition →
    straight-through argmax, no self-loops) → re-injects the endpoint-derived edge id
    (combine(id_src,id_dst)) so the next layer sees the current graph (the structure-feedback loop).
  • output G = [B, N+E, d_node] (the graph) + the final endpoints + canaries. NO prepend memory.

READ (decoder): per-layer bottleneck GATED CROSS-ATTENTION over the FROZEN G (`GatedGraphXAttn`),
installed as forward-pre-hooks on the decoder layers by model.py. Node-dropout (anti-bypass) is a KV
mask on the node entries (edges kept), applied by model.py. No prepend, no message-passing read.

Ablations: use_structure=False ⇒ flat set (no endpoint prediction / edge-id re-injection);
use_id=False ⇒ drop the learned id embeddings.
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


def _st_onehot(logits: Tensor, temp: float) -> Tensor:
    """Straight-through: one-hot(argmax) forward, softmax(logits/temp) gradient backward."""
    soft = (logits / temp).softmax(-1)
    idx = soft.argmax(-1, keepdim=True)
    hard = torch.zeros_like(soft).scatter_(-1, idx, 1.0)
    return hard + (soft - soft.detach())


def _sinkhorn1(scores: Tensor) -> Tensor:
    """One log-space Sinkhorn step: row-normalize over nodes (each edge commits) then column-normalize
    over edges (competition — a node demanded by many edges is pushed down). scores [B,E,N]."""
    logA = scores - torch.logsumexp(scores, dim=-1, keepdim=True)
    logA = logA - torch.logsumexp(logA, dim=-2, keepdim=True)
    return logA


class GatedGraphXAttn(nn.Module):
    """Bottleneck gated cross-attention: a decoder hidden [B,S,d] reads the graph G [B,U,dn].
    Attends IN the small graph space (down-project query d→dn, attend over G, up-project dn→d). The
    tanh gate is init 0 → cold-start no-op (read starts as pure LM). Returns the gated delta to ADD."""

    def __init__(self, d: int, dn: int, n_heads: int):
        super().__init__()
        assert dn % n_heads == 0, f"d_node {dn} must be divisible by xattn_heads {n_heads}"
        self.dn, self.h, self.hd = dn, n_heads, dn // n_heads
        self.q = nn.Linear(d, dn, bias=False)     # decoder hidden → graph space (down)
        self.k = nn.Linear(dn, dn, bias=False)
        self.v = nn.Linear(dn, dn, bias=False)
        self.o = nn.Linear(dn, d, bias=False)     # back up to d_llama
        # Small POSITIVE gate init (not 0): a pure-0 Flamingo gate deadlocks here because our graph G is
        # random at init (not a pretrained encoder), so the gate gradient never opens it and the encoder
        # stays gradient-starved. tanh(0.1)≈0.1 gives the encoder real gradient from step 1 (weak read,
        # minimal LM perturbation) → encoder improves → read becomes useful → gate can grow. Breaks the
        # chicken-and-egg. OFF (zero_memory) at eval still measures the no-read baseline.
        self.gate = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, h: Tensor, G: Tensor, kv_keep: Tensor | None = None) -> Tensor:
        # h [B,S,d], G [B,U,dn], kv_keep [B,U] bool (True = attendable; None = all)
        B, S, _ = h.shape; U = G.shape[1]
        hf = h.float()
        q = self.q(hf).view(B, S, self.h, self.hd).transpose(1, 2)        # [B,h,S,hd]
        k = self.k(G).view(B, U, self.h, self.hd).transpose(1, 2)         # [B,h,U,hd]
        v = self.v(G).view(B, U, self.h, self.hd).transpose(1, 2)
        attn_mask = None
        if kv_keep is not None:
            attn_mask = torch.zeros(B, 1, 1, U, device=h.device, dtype=torch.float32)
            attn_mask = attn_mask.masked_fill(~kv_keep.view(B, 1, 1, U), float("-inf"))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)   # [B,h,S,hd]
        o = o.transpose(1, 2).reshape(B, S, self.dn)
        return (torch.tanh(self.gate) * self.o(o)).to(h.dtype)            # gated delta in d_llama


class _GTLayer(nn.Module):
    """One graph-transformer layer (in d_node): cross-attn to passage → self-attn → (edges) predict
    endpoints + re-inject endpoint-derived id → FFN. Pre-norm residuals."""

    def __init__(self, cfg: ReprConfig, dn: int, N: int, E: int):
        super().__init__()
        self.N, self.E, self.dn = N, E, dn
        self.d_key = cfg.slotgraph_d_key
        self.use_structure = bool(getattr(cfg, "slotgraph_use_structure", True))
        self.use_id = bool(getattr(cfg, "slotgraph_use_id", True))
        h = 4
        self.n1 = nn.LayerNorm(dn); self.cq = nn.Linear(dn, dn); self.ck = nn.Linear(dn, dn)
        self.cv = nn.Linear(dn, dn); self.co = nn.Linear(dn, dn)
        self.n2 = nn.LayerNorm(dn); self.sq = nn.Linear(dn, dn); self.sk = nn.Linear(dn, dn)
        self.sv = nn.Linear(dn, dn); self.so = nn.Linear(dn, dn)
        self.heads = h
        self.n3 = nn.LayerNorm(dn); self.ff = nn.Sequential(nn.Linear(dn, 4 * dn), nn.GELU(), nn.Linear(4 * dn, dn))
        # structure heads (edges → src/dst queries; nodes → keys)
        self.ns = nn.LayerNorm(dn)
        self.q_src = nn.Linear(dn, self.d_key); self.q_dst = nn.Linear(dn, self.d_key)
        self.k_node = nn.Linear(dn, self.d_key)
        self.log_temp = nn.Parameter(torch.tensor(math.log(float(cfg.slotgraph_temp_init))))
        self.edge_combine = nn.Linear(2 * dn, dn)   # endpoint-derived edge id from [id_src ; id_dst]

    def _mha(self, q, k, v, qh, kh, vh, oh, kv_mask=None):
        B, S, _ = q.shape; U = k.shape[1]
        Q = qh(q).view(B, S, self.heads, -1).transpose(1, 2)
        K = kh(k).view(B, U, self.heads, -1).transpose(1, 2)
        V = vh(v).view(B, U, self.heads, -1).transpose(1, 2)
        am = None
        if kv_mask is not None:
            am = torch.zeros(B, 1, 1, U, device=q.device, dtype=torch.float32).masked_fill(
                ~kv_mask.view(B, 1, 1, U), float("-inf"))
        o = F.scaled_dot_product_attention(Q, K, V, attn_mask=am).transpose(1, 2).reshape(B, S, -1)
        return oh(o)

    def _structure(self, G):
        """Predict each edge's src/dst over the node pool. Returns hard src,dst [B,E,N] + softs + picks.
        Keys/queries read the ids via content (ids are already added into G upstream)."""
        gn = self.ns(G)
        nodes = gn[:, :self.N]; edges = gn[:, self.N:]
        k = self.k_node(nodes)                                   # [B,N,dk]
        qs = self.q_src(edges); qd = self.q_dst(edges)           # [B,E,dk]
        scale = 1.0 / math.sqrt(self.d_key); temp = self.log_temp.exp().clamp_min(1e-2)
        sc_src = torch.einsum("bed,bnd->ben", qs, k) * scale / temp
        ls = _sinkhorn1(sc_src); src = _st_onehot(ls, 1.0)       # [B,E,N]
        sc_dst = torch.einsum("bed,bnd->ben", qd, k) * scale / temp - 1e4 * src.detach()
        ld = _sinkhorn1(sc_dst); dst = _st_onehot(ld, 1.0)
        # telemetry = the POST-Sinkhorn routing distribution (the competition-adjusted one the ST pick
        # uses), not the raw pre-competition softmax.
        return src, dst, ls.softmax(-1), ld.softmax(-1)

    def forward(self, G, ctx, ctx_keep, node_id):
        # cross-attn: graph units query the passage features
        G = G + self.co(self._mha(self.n1(G), ctx, ctx, self.cq, self.ck, self.cv, lambda x: x, ctx_keep))
        # self-attn: units mix
        gn = self.n2(G)
        G = G + self.so(self._mha(gn, gn, gn, self.sq, self.sk, self.sv, lambda x: x))
        src = dst = soft_s = soft_d = None
        if self.use_structure:
            src, dst, soft_s, soft_d = self._structure(G)
            if self.use_id:
                # endpoint-derived edge id: gather node ids by the materialized endpoints
                id_src = torch.einsum("ben,nd->bed", src, node_id)   # [B,E,dn]
                id_dst = torch.einsum("ben,nd->bed", dst, node_id)
                edge_id = self.edge_combine(torch.cat([id_src, id_dst], dim=-1))
                G = torch.cat([G[:, :self.N], G[:, self.N:] + edge_id], dim=1)
        G = G + self.ff(self.n3(G))
        return G, src, dst, soft_s, soft_d


class SlotGraphEncoder(nn.Module):
    is_conditioned_read = False          # not the biomem query-conditioned read
    wants_surprise = False
    wants_prepend_refresh = False
    reinforce_prepend_each_layer = False
    wants_graph_xattn = True             # NEW: model.py installs per-layer gated cross-attn read hooks

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
        self.M = self.N + self.E
        self.use_structure = bool(cfg.slotgraph_use_structure)
        self.use_id = bool(cfg.slotgraph_use_id)
        self.xattn_every = int(cfg.slotgraph_xattn_every)
        self.node_drop_p = 0.0           # set by the trainer each step (annealed node-dropout)
        self.read_ablate = None          # diagnostics: None | "edges" | "nodes" (read ablation gate)

        emb = base.get_input_embeddings()
        with torch.no_grad():
            emb_std = emb.weight.float().std().item()
        # small-unit seeds + learned ids + learned role
        self.slot_init = nn.Parameter(emb_std * torch.randn(self.M, dn) * 0.1)
        self.node_id = nn.Parameter(torch.randn(self.N, dn) / math.sqrt(dn))   # learned node ids
        self.id_scale = nn.Parameter(torch.tensor(1.0))                         # learnable, modest (NOT √d)
        # GCNII initial-residual weight (anti-over-smoothing): each GT layer mixes back the distinct-id
        # init G0, anchoring per-unit identity so the full 288×288 self-attn can't smooth units together.
        # sigmoid(-1.7)≈0.15 init (GCNII default α~0.1); learnable. This is the design's "re-inject id/role
        # each layer" made principled. Without it, pairwise cosine climbs to ~0.95 and the read can't
        # discriminate units.
        self.gcnii_alpha = nn.Parameter(torch.tensor(-1.7))
        self.role_embed = nn.Parameter(torch.randn(2, dn) / math.sqrt(dn))      # node / edge role
        is_node = torch.zeros(self.M, dtype=torch.bool); is_node[:self.N] = True
        self.register_buffer("is_node", is_node, persistent=False)

        # passage perception → graph space
        self.ctx_proj = nn.Linear(d, dn)
        self.in_norm = nn.LayerNorm(dn)
        self.gt_layers = nn.ModuleList([_GTLayer(cfg, dn, self.N, self.E) for _ in range(int(cfg.slotgraph_enc_layers))])
        self.out_norm = nn.LayerNorm(dn)

        # decoder read: one gated cross-attn per decoder layer (owned here so params are trainable;
        # applied to the decoder's layers via hooks in model.py)
        n_dec_layers = base.config.num_hidden_layers
        self._hook_layers = list(range(0, n_dec_layers, self.xattn_every))   # decoder layers that get a read
        self.read_xattn = nn.ModuleList([GatedGraphXAttn(d, dn, cfg.slotgraph_xattn_heads)
                                         for _ in self._hook_layers])         # one per HOOKED layer (no dead modules)
        print(f"[slotgraph v2] {self.N} nodes + {self.E} edges @ d_node={dn} (graph-utterance); "
              f"{int(cfg.slotgraph_enc_layers)}-layer GT encoder, per-layer gated cross-attn read "
              f"({n_dec_layers} dec layers, every {self.xattn_every}), encoder-LoRA r{cfg.slotgraph_lora_rank} "
              f"({n_wrapped} layers), use_structure={self.use_structure}, use_id={self.use_id}")

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

    def _init_graph(self, B):
        G = self.slot_init.unsqueeze(0).expand(B, -1, -1).clone()              # [B,M,dn]
        G = G + self.role_embed[self.is_node.long()].unsqueeze(0)             # role (0/1 → embed)
        if self.use_id:
            id_full = torch.zeros(self.M, self.dn, device=G.device, dtype=G.dtype)
            id_full[:self.N] = self.node_id
            G = G + self.id_scale * id_full.unsqueeze(0)                      # node ids (edges start id-less)
        return G

    def finalize_memory(self, state):
        emb, mask = state["emb"], state["mask"]                               # [B,T,d], [B,T]
        B = emb.shape[0]
        attn = mask.long()
        # perceive with the frozen LM (encoder-LoRA, bf16)
        H = self.base.model(inputs_embeds=emb, attention_mask=attn, use_cache=False).last_hidden_state
        with torch.autocast("cuda", enabled=False):                          # graph in fp32 (small + sensitive)
            ctx = self.in_norm(self.ctx_proj(H.float()))                     # [B,T,dn]
            G0 = self._init_graph(B).float()                                 # the distinct-id init (anchor)
            G = G0
            alpha = torch.sigmoid(self.gcnii_alpha)                          # GCNII initial-residual weight
            src = dst = soft_s = soft_d = None
            for layer in self.gt_layers:
                G, src, dst, soft_s, soft_d = layer(G, ctx, mask.bool(), self.node_id.float())
                G = (1.0 - alpha) * G + alpha * G0                           # anchor identity each layer (anti-over-smoothing)
            G = self.out_norm(G)
        memory = emb.new_zeros(B, 0, self.d)                                  # NO prepend; read = cross-attn
        aux = self._canaries(G, src, dst, soft_s, soft_d, emb.device)
        aux["graph_G"] = G                                                    # consumed by the read hooks
        return memory, aux

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        st = self.init_streaming_state(token_embeds.shape[0], token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)

    # ── read KV-keep mask: ablation gate (edges/nodes off) OR node-dropout curriculum ──
    def node_keep_mask(self, B, device, training: bool):
        keep = torch.ones(B, self.M, dtype=torch.bool, device=device)
        if self.read_ablate == "edges":          # diagnostics: read NODES only (the edges-off gate)
            keep[:, self.N:] = False
            return keep
        if self.read_ablate == "nodes":          # diagnostics: read EDGES only
            keep[:, :self.N] = False
            return keep
        if training and self.node_drop_p > 0.0 and self.use_structure:        # anti-bypass curriculum
            drop = torch.rand(self.N, device=device) < self.node_drop_p       # per-batch (shared across B)
            keep[:, :self.N] = keep[:, :self.N] & (~drop).view(1, self.N)
        return keep

    @torch.no_grad()
    def _canaries(self, G, src, dst, soft_s, soft_d, device):
        aux = {}
        aux["slotgraph_mem_effrank"] = torch.tensor(
            _participation_ratio(G.reshape(-1, G.shape[-1])), device=device)
        aux["slotgraph_edge_frac"] = torch.tensor(float(self.E) / self.M, device=device)
        # mean read gate (tanh) across decoder layers — watch the cross-attn bootstrap open over training
        aux["slotgraph_read_gate"] = torch.stack([x.gate for x in self.read_xattn]).detach().tanh().mean()
        if src is not None:
            N = self.N
            sp = src.argmax(-1); dp = dst.argmax(-1)                          # [B,E]
            aux["slotgraph_src"] = sp; aux["slotgraph_dst"] = dp
            aux["slotgraph_selfloop_frac"] = (sp == dp).float().mean()
            # within-batch cross-input routing diversity (THE topology signal): per edge, entropy of its
            # src pick across the B examples / ln N. ↑ over training ⇒ the graph responds to the input.
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
            aux["slotgraph_node_entropy"] = -(pu.clamp_min(1e-9).log() * pu).sum()
            aux["slotgraph_id_scale"] = self.id_scale.detach()
        return aux
