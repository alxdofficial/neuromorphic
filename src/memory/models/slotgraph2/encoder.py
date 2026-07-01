"""slotgraph2 — per-layer graph-transformer slot memory over ICAE-encoded input; PREPEND read.

The intermediate design between the simple (ICAE one-shot) slotgraph and the chain-furl idea. Fixed
partition: K node slots + (M-K) edge slots, each at d = d_llama (576). Per streaming window the frozen
LM (own copy + encoder-LoRA) encodes that window's tokens to hiddens H; then L graph-transformer layers
REWRITE the persistent graph state:

  tokenize:  node_tok = node_latent + id_scale·node-id            + node-role
             edge_tok = edge_latent + id_scale·(id_home + id_dst) + edge-role
             source is FIXED (edge e is anchored to its home node e); id_dst is the SOFT painted target.
  mix:       (1) graph tokens CROSS-attend to the input hiddens; (2) self-attend among themselves.
  heads:     node → next-latent delta (additive → residual highway)
             edge → next-latent delta (additive) + dst-query scored against the learnable node-id keys
                    → SOFTMAX over destinations (paintbrush; NO straight-through; dst ≠ home).

The graph state (latents + the soft dst distribution) PERSISTS across windows. Read = PREPEND the final M
graph tokens (norm-matched to the LM embedding scale). Binding rides the outer-product address
id_dst = Σ_k t_e,k·node_id[k] carried in each edge token (superposition of s⊗t, not a value-bag);
id_scale≈√d keeps that address legible against the LM-scale latent, and the additive latent updates give
an identity gradient highway so early actions (window 1 / layer 1) get credit as healthy as late ones.
Peakedness of t_e (dst_entropy canary) is the crosstalk↔gradient knob; routing_diversity = input-
dependence of the wiring (the metric that sat dead at ~0.02 in every prior arm).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...common import _NormMatch
from ...config import ReprConfig


def _participation_ratio(x: Tensor) -> float:
    x = x.detach().float()
    if x.shape[0] < 2:
        return 0.0
    xc = x - x.mean(0, keepdim=True); C = xc.t() @ xc
    return float((torch.diagonal(C).sum() ** 2 / (C * C).sum().clamp_min(1e-12)).item())


class _SG2Layer(nn.Module):
    """One write layer = TWO sub-steps, keeping the input sequence separate (never rewritten):
      (1) graph tokens CROSS-attend to the (input-role-tagged) input sequence — pull in input info;
      (2) graph tokens SELF-attend among themselves — mix the structure;
    then FFN + delta heads (node/edge next-latent, ADDITIVE residual) + a single dst-endpoint head
    (edge → dst query → SOFTMAX over node-id keys; the source endpoint is fixed to the edge's home node)."""

    def __init__(self, d: int, n_heads: int, d_key: int):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.hd, self.dk = n_heads, d // n_heads, d_key
        # sub-step 1 — cross-attn: graph queries over input keys/values
        self.nq_c = nn.LayerNorm(d); self.nkv_c = nn.LayerNorm(d)
        self.wq_c = nn.Linear(d, d); self.wk_c = nn.Linear(d, d); self.wv_c = nn.Linear(d, d); self.wo_c = nn.Linear(d, d)
        # sub-step 2 — self-attn over the graph tokens
        self.ns = nn.LayerNorm(d)
        self.wq_s = nn.Linear(d, d); self.wk_s = nn.Linear(d, d); self.wv_s = nn.Linear(d, d); self.wo_s = nn.Linear(d, d)
        # FFN + heads
        self.nff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.n_head = nn.LayerNorm(d)             # normed head input → bounds each additive delta to ~unit scale
        self.head_node = nn.Linear(d, d)          # node → next-latent delta (additive)
        self.head_edge = nn.Linear(d, d)          # edge → next-latent delta (additive)
        self.q_dst = nn.Linear(d, d_key)          # edge → dst endpoint query (source is the fixed home node)
        self.beta_node = nn.Parameter(torch.tensor(-1.2))   # gated-delta rate (sigmoid(-1.2)≈0.23)
        self.beta_edge = nn.Parameter(torch.tensor(-1.2))

    def _mha(self, q_in, kv_in, kv_keep, wq, wk, wv, wo):
        B, Mq, d = q_in.shape; U = kv_in.shape[1]
        q = wq(q_in).view(B, Mq, self.h, self.hd).transpose(1, 2)
        k = wk(kv_in).view(B, U, self.h, self.hd).transpose(1, 2)
        v = wv(kv_in).view(B, U, self.h, self.hd).transpose(1, 2)
        am = None
        if kv_keep is not None:
            am = torch.zeros(B, 1, 1, U, device=q_in.device, dtype=q.dtype).masked_fill(
                ~kv_keep.view(B, 1, 1, U), float("-inf"))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=am).transpose(1, 2).reshape(B, Mq, d)
        return wo(o)

    def forward(self, node_lat, edge_lat, dst_soft, node_id, id_home, role, id_scale, H_in, H_keep, k_keys, self_mask):
        B, K, d = node_lat.shape; E = edge_lat.shape[1]
        id_dst = torch.einsum("bek,kd->bed", dst_soft, node_id)      # soft painted dst address (Σ_k t_e,k·id_k)
        node_tok = node_lat + id_scale * node_id.unsqueeze(0) + role[0]           # [B,K,d]
        edge_tok = edge_lat + id_scale * (id_home.unsqueeze(0) + id_dst) + role[1]  # [B,E,d] (home = fixed src)
        graph = torch.cat([node_tok, edge_tok], dim=1)               # [B,M,d]
        # sub-step 1: cross-attend to the input sequence (H_in already carries the input-role tag)
        graph = graph + self._mha(self.nq_c(graph), self.nkv_c(H_in), H_keep,
                                  self.wq_c, self.wk_c, self.wv_c, self.wo_c)
        # sub-step 2: self-attend over the graph tokens (all graph tokens valid → no mask)
        gs = self.ns(graph)
        graph = graph + self._mha(gs, gs, None, self.wq_s, self.wk_s, self.wv_s, self.wo_s)
        graph = graph + self.ff(self.nff(graph))
        gh = self.n_head(graph); mn, me = gh[:, :K], gh[:, K:]       # normed → bounded deltas
        node_lat = node_lat + torch.sigmoid(self.beta_node) * self.head_node(mn)   # ADDITIVE residual highway
        edge_lat = edge_lat + torch.sigmoid(self.beta_edge) * self.head_edge(me)
        sc = torch.einsum("bed,kd->bek", self.q_dst(me), k_keys) / math.sqrt(self.dk)   # dst query vs node-id keys
        dst_soft = (sc + self_mask).softmax(-1)                      # SOFT paintbrush; -inf on home → dst ≠ home
        return node_lat, edge_lat, dst_soft


class SlotGraph2Encoder(nn.Module):
    is_conditioned_read = False              # READ = PREPEND the M graph tokens

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        from ...decoder import load_frozen_llama, apply_lora_to_llama
        base, _ = load_frozen_llama(cfg.llama_model)
        for p in base.parameters():
            p.requires_grad_(False)
        n_wrapped = apply_lora_to_llama(base, rank=cfg.slotgraph2_lora_rank, alpha=cfg.slotgraph2_lora_alpha,
                                        target_names=tuple(cfg.llama_lora_target_names))
        self.base = base
        d = cfg.d_llama; self.d = d
        self.M = int(cfg.slotgraph2_n_slots); self.K = int(cfg.slotgraph2_n_nodes); self.E = self.M - self.K
        self.L = int(cfg.slotgraph2_n_layers); self.window = int(cfg.slotgraph2_window)
        self.dk = int(cfg.slotgraph2_d_key); self.recurrent = bool(cfg.slotgraph2_recurrent)
        assert 1 <= self.K < self.M, f"need 1<=K({self.K})<M({self.M})"

        # learnable per-node identity embeddings (orthonormal init: distinct but trainable; re-normalized
        # to the unit sphere each forward so id_scale alone controls the address magnitude)
        nid = torch.empty(self.K, d); nn.init.orthogonal_(nid)
        self.node_id = nn.Parameter(F.normalize(nid, dim=-1))
        self.id_scale = nn.Parameter(torch.tensor(math.sqrt(d)))       # √d: keep the unit-norm address legible vs LM-scale latent
        self.role = nn.Parameter(torch.randn(3, d) / math.sqrt(d))     # [node-role, edge-role, input-role]
        # fixed 1:1 source anchoring — edge e is homed at node (e mod K); breaks edge-permutation symmetry
        # (the free id-tag anti-collapse win). self_mask forbids an edge painting its own home (dst ≠ home).
        home_idx = torch.arange(self.E) % self.K
        self.register_buffer("home_idx", home_idx, persistent=False)
        sm = torch.zeros(self.E, self.K); sm[torch.arange(self.E), home_idx] = float("-inf")
        self.register_buffer("self_mask", sm, persistent=False)
        # latent seeds (centered in the LM token region, icae-style)
        embed = base.get_input_embeddings()
        with torch.no_grad():
            mean_vec = embed.weight.float().mean(0); emb_std = embed.weight.float().std().item()
        self.node_lat_init = nn.Parameter(mean_vec.view(1, d).repeat(self.K, 1) + emb_std * torch.randn(self.K, d))
        self.edge_lat_init = nn.Parameter(mean_vec.view(1, d).repeat(self.E, 1) + emb_std * torch.randn(self.E, d))
        self.k_proj = nn.Linear(d, self.dk)                            # project node ids → selection keys
        n_layers = 1 if self.recurrent else self.L
        self.layers = nn.ModuleList([_SG2Layer(d, int(cfg.slotgraph2_heads), self.dk) for _ in range(n_layers)])
        self._trace = None                                             # set to [] to record per-(window,layer) grad credit

        self.norm = _NormMatch(d)
        with torch.no_grad():
            self.norm.scale.data.fill_(embed.weight.float().norm(dim=-1).mean().item())
        print(f"[slotgraph2] {self.K} nodes + {self.E} edges @ d={d}; {self.L} graph layers "
              f"({'recurrent' if self.recurrent else 'distinct'}); window={self.window}; PREPEND read; "
              f"encoder-LoRA r{cfg.slotgraph2_lora_rank} ({n_wrapped} layers)")

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

    def _layer(self, i):
        return self.layers[0] if self.recurrent else self.layers[i]

    def finalize_memory(self, state):
        emb, mask = state["emb"], state["mask"]                       # [B,T,d], [B,T]
        B, T, d = emb.shape
        if T == 0:
            raise ValueError("slotgraph2.finalize_memory: empty context (T=0)")
        node_lat = self.node_lat_init.float().unsqueeze(0).expand(B, -1, -1).contiguous()
        edge_lat = self.edge_lat_init.float().unsqueeze(0).expand(B, -1, -1).contiguous()
        dst_soft = torch.zeros(B, self.E, self.K, device=emb.device)  # no destination painted yet
        nid = F.normalize(self.node_id.float(), dim=-1)               # unit-sphere addresses
        role = self.role.float(); id_scale = self.id_scale.float()
        id_home = nid[self.home_idx]                                  # [E,d] fixed source per edge
        self_mask = self.self_mask.unsqueeze(0)                       # [1,E,K]
        for w in range(0, T, self.window):                            # streaming windows
            wm = mask[:, w:w + self.window].bool()
            active = wm.any(dim=1)                                     # [B] rows with real tokens in THIS window
            if not bool(active.any()):                                 # fully-pad window (batch-max padding) → skip
                continue                                              # else short rows eat input-free updates = length confound
            we = emb[:, w:w + self.window]
            H = self.base.model(inputs_embeds=we, attention_mask=wm.long(),
                                use_cache=False).last_hidden_state     # bf16 LM encode of this window
            with torch.autocast("cuda", enabled=False):               # graph in fp32 (small + sensitive)
                nl0, el0, ds0 = node_lat, edge_lat, dst_soft           # snapshot → freeze rows idle this window
                H_in = H.float() + role[2]                             # tag input tokens with the input-role
                kk = self.k_proj(nid)
                for i in range(self.L):                                # L graph-transformer layers / window
                    node_lat, edge_lat, dst_soft = self._layer(i)(
                        node_lat, edge_lat, dst_soft, nid, id_home, role, id_scale, H_in, wm, kk, self_mask)
                    if self._trace is not None:                        # opt-in early↔late grad-credit tracer
                        for t in (node_lat, edge_lat, dst_soft):
                            t.retain_grad()
                        self._trace.append((w // self.window, i, node_lat, edge_lat, dst_soft))
                if not bool(active.all()):                             # mixed-length batch → restore idle rows to pre-window state
                    a = active[:, None, None]
                    node_lat = torch.where(a, node_lat, nl0)
                    edge_lat = torch.where(a, edge_lat, el0)
                    dst_soft = torch.where(a, dst_soft, ds0)
        with torch.autocast("cuda", enabled=False):
            id_dst = torch.einsum("bek,kd->bed", dst_soft, nid)
            node_tok = node_lat + id_scale * nid.unsqueeze(0) + role[0]
            edge_tok = edge_lat + id_scale * (id_home.unsqueeze(0) + id_dst) + role[1]
            memory = self.norm(torch.cat([node_tok, edge_tok], dim=1))   # [B,M,d] prepend tokens
        aux = self._canaries(memory, node_lat, edge_lat, dst_soft, nid, emb.device)
        return memory, aux

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        st = self.init_streaming_state(token_embeds.shape[0], token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)

    @torch.no_grad()
    def _canaries(self, memory, node_lat, edge_lat, dst_soft, nid, device):
        def _within_cos(x):                                          # mean off-diagonal cosine among a sample's slots (→1 = collapsed)
            S = x.shape[1]
            if S < 2:
                return 0.0
            xn = F.normalize(x, dim=-1); cos = xn @ xn.transpose(-1, -2)
            off = cos.sum((-1, -2)) - cos.diagonal(dim1=-2, dim2=-1).sum(-1)
            return float((off / (S * (S - 1))).mean())
        aux = {  # READ — rank of the prepended memory across slots×batch
               "slotgraph2_mem_effrank": torch.tensor(_participation_ratio(memory.reshape(-1, self.d)), device=device),
               # VOCAB/RELATION content rank
               "slotgraph2_node_effrank": torch.tensor(_participation_ratio(node_lat.reshape(-1, self.d)), device=device),
               "slotgraph2_edge_effrank": torch.tensor(_participation_ratio(edge_lat.reshape(-1, self.d)), device=device),
               # VOCAB — do the learnable ADDRESSES stay distinct (analog of graph_bank_effrank)
               "slotgraph2_nodeid_effrank": torch.tensor(_participation_ratio(nid), device=device),
               # RELATION/VOCAB collapse — all slots → the same vector (analog of graph_edge_cos)
               "slotgraph2_edge_cos": torch.tensor(_within_cos(edge_lat), device=device),
               "slotgraph2_node_cos": torch.tensor(_within_cos(node_lat), device=device)}
        dp = dst_soft.argmax(-1)                                      # [B,E] hard-picked destination
        # SELECTION sharpness — peakedness of the paintbrush = the crosstalk↔gradient knob (0 sharp → lnK blur)
        aux["slotgraph2_dst_entropy"] = (-(dst_soft.clamp_min(1e-9).log() * dst_soft).sum(-1)).mean()
        # SELECTION spread — hub-collapse detector (↑→lnK spread, ↓ hub) + raw count of destinations used
        use = torch.bincount(dp.reshape(-1), minlength=self.K).float()
        pu = use / use.sum().clamp_min(1e-9)
        aux["slotgraph2_node_entropy"] = -(pu.clamp_min(1e-9).log() * pu).sum()
        aux["slotgraph2_nodes_used"] = torch.tensor(float(dp.reshape(-1).unique().numel()), device=device)
        # additive-residual growth watch (latents accumulate across windows) + address legibility vs content
        aux["slotgraph2_node_lat_norm"] = node_lat.norm(dim=-1).mean()
        aux["slotgraph2_edge_lat_norm"] = edge_lat.norm(dim=-1).mean()
        id_dst = torch.einsum("bek,kd->bed", dst_soft, nid)          # ‖id_scale·id_dst‖/‖edge_lat‖ → is the address readable?
        aux["slotgraph2_addr_legibility"] = (self.id_scale.detach() * id_dst.norm(dim=-1)
                                             / edge_lat.norm(dim=-1).clamp_min(1e-6)).mean()
        if dp.shape[0] > 1:
            # KEY — input-dependence of the WIRING across the batch (sat dead ~0.02 in every prior arm)
            oh = F.one_hot(dp, self.K).float().mean(0)               # [E,K] per-edge dst distribution across batch
            aux["slotgraph2_routing_diversity"] = (-(oh.clamp_min(1e-9).log() * oh).sum(-1)).mean() / math.log(self.K)
            # WRITE input-dependence at the CONTENT level — eff-rank of per-example mean latent (→1 = ignores input)
            aux["slotgraph2_node_input_sens"] = torch.tensor(_participation_ratio(node_lat.mean(1)), device=device)
            aux["slotgraph2_edge_input_sens"] = torch.tensor(_participation_ratio(edge_lat.mean(1)), device=device)
        return aux
