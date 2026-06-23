"""slotgraph — fixed-partition graph slot memory (ICAE write + multi-hop message-passing read).

Architecture: start from the proven binder — ICAE (own frozen base + encoder-LoRA, M learnable slots
APPENDED to the passage, run through the LM's OWN layers) — and read the slots out through a graph.

  • FIXED partition: slots 0..K-1 are NODES, slots K..M-1 are EDGES (role by position, not predicted).
  • CONTENT-ADDRESSED routing with COMPETITION: each edge emits src/dst QUERIES; each node emits a KEY
    (from [node content ; orthonormal id] → distinct keys even if contents collapse). Endpoints are
    chosen by query·key match — NOT a position-classifier — and a single-step Sinkhorn (row-norm over
    nodes + column-norm over edges) makes edges COMPETE for nodes so they spread instead of hub-collapsing.
    Straight-through argmax per edge → a hard node→node edge. Scores are only over the node pool, so
    edges→nodes holds by construction (no masked-softmax sentinel).
  • READ = a multi-hop residual message-passing GNN over the predicted graph: each node relays its own
    state + edge-routed neighbour messages; output is still M vectors (same prepend budget as icae).
    use_structure=False ⇒ plain prepend of the id-tagged slots (id-tagged ICAE control).

MAGNITUDE / GRADIENT POLICY (no free-floating scale coefficients, no gates; everything measured/bounded):
  1. The structure + read run in FP32 (the LM stays bf16). They are small (M=32 slots) and the sensitive
     part numerically; fp32 removes the stochastic bf16 backward-overflow that poisoned training.
  2. Competition is a single Sinkhorn step in LOG-space (logsumexp, no divide → no 1/‖·‖ singularity).
  3. The MP read's per-node update is bounded by an ELEMENTWISE clamp to [-1,1] (no norm/division → clean
     0/1 backward), so h ≤ ‖slot_final‖ + K·√d with no gate; mean (degree-normalized) aggregation.
  4. The only boundary scaling is the MEASURED rescale of the output to the LM embedding norm (_NormMatch);
     head inputs are balanced to √d (id_head_scale). Cold-start = zero-init update (step-0 read = plain read).
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
    """Effective rank via (tr C)²/‖C‖_F² (no eigendecomp; autocast-safe). PR≈1 ⇒ rank-1 collapse."""
    x = x.detach().float()
    if x.shape[0] < 2:
        return 0.0
    xc = x - x.mean(0, keepdim=True); C = xc.t() @ xc
    return float((torch.diagonal(C).sum() ** 2 / (C * C).sum().clamp_min(1e-12)).item())


def _st_onehot(logits: Tensor, temp: float) -> Tensor:
    """Straight-through hard selection: one-hot(argmax) forward, softmax(logits/temp) gradient backward."""
    soft = (logits / temp).softmax(-1)
    idx = soft.argmax(-1, keepdim=True)
    hard = torch.zeros_like(soft).scatter_(-1, idx, 1.0)
    return hard + (soft - soft.detach())


class SlotGraphEncoder(nn.Module):
    is_conditioned_read = False              # READ = PREPEND the M slots' (message-passed) final hiddens

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        from ...decoder import load_frozen_llama, apply_lora_to_llama
        base, _ = load_frozen_llama(cfg.llama_model)
        for p in base.parameters():
            p.requires_grad_(False)
        n_wrapped = apply_lora_to_llama(
            base, rank=cfg.slotgraph_lora_rank, alpha=cfg.slotgraph_lora_alpha,
            target_names=tuple(cfg.llama_lora_target_names))
        self.base = base
        d = cfg.d_llama
        self.M = cfg.slotgraph_n_slots
        self.d = d
        self.K = max(1, min(int(getattr(cfg, "slotgraph_n_nodes", self.M // 2)), self.M - 1))  # node slots 0..K-1
        self.use_structure = bool(getattr(cfg, "slotgraph_use_structure", True))
        self.max_hops = int(getattr(cfg, "slotgraph_max_hops", 5))

        # slot content seeds (appended to the passage, icae-style: centered in Llama's token region)
        embed = base.get_input_embeddings()
        with torch.no_grad():
            mean_vec = embed.weight.float().mean(0)
            emb_std = embed.weight.float().std().item()
        slot_init = mean_vec.view(1, d).repeat(self.M, 1) + emb_std * torch.randn(self.M, d)
        self.slot_init = nn.Parameter(slot_init)

        # FIXED near-orthonormal per-position identity codes (buffer → distinct + reload-stable)
        assert 2 <= self.M <= d, (f"slotgraph needs 2 <= M ({self.M}) <= d_llama ({d}): M≥2 for a valid "
                                  f"node/edge partition, M≤d for orthonormal id codes.")
        gen = torch.Generator().manual_seed(0)
        q = torch.linalg.qr(torch.randn(d, self.M, generator=gen))[0]    # [d,M] orthonormal cols
        self.register_buffer("id_embed", q.t().contiguous(), persistent=True)   # [M,d] unit-norm rows
        is_node = torch.zeros(self.M); is_node[:self.K] = 1.0
        self.register_buffer("is_node", is_node, persistent=False)              # [M] 1=node, 0=edge
        role_fixed = torch.zeros(self.M, 2); role_fixed[:self.K, 0] = 1.0; role_fixed[self.K:, 1] = 1.0
        self.register_buffer("role_fixed", role_fixed, persistent=False)        # [M,2] (canaries/viz)

        # content-addressed routing heads: edge→query, node→key (input = [struct_norm(slot) ; √d·id])
        self.struct_norm = nn.LayerNorm(d)
        self.id_head_scale = math.sqrt(d)
        self.d_k = int(getattr(cfg, "slotgraph_d_key", 64))
        self.q_src_head = nn.Linear(2 * d, self.d_k)   # edge → src query
        self.q_dst_head = nn.Linear(2 * d, self.d_k)   # edge → dst query
        self.k_head = nn.Linear(2 * d, self.d_k)       # node → key
        self.log_temp = nn.Parameter(torch.tensor(math.log(float(cfg.slotgraph_temp_init))))

        # ── multi-hop message-passing READ ──────────────────────────────────
        self.msg = nn.Linear(2 * d, d, bias=False)   # [source node ; edge content] → message
        self.update = nn.Linear(d, d, bias=False)    # summed incoming → node-state delta
        nn.init.zeros_(self.update.weight)           # zero-init ⇒ step-0 read = plain read (cold-start)

        self.norm = _NormMatch(d)
        with torch.no_grad():
            self.norm.scale.data.fill_(embed.weight.float().norm(dim=-1).mean().item())
        print(f"[slotgraph] icae-write + FIXED partition ({self.K} nodes / {self.M - self.K} edges) + "
              f"content-addressed routing (d_k={self.d_k}, Sinkhorn competition) + fp32 MP read "
              f"(≤{self.max_hops} hops), encoder-LoRA r{cfg.slotgraph_lora_rank} ({n_wrapped} layers), "
              f"use_structure={self.use_structure}")

    def train(self, mode: bool = True):
        super().train(mode); self.base.train(False); return self

    # ── streaming: accumulate the passage embeds (icae pattern) ──
    def init_streaming_state(self, batch_size, device, dtype):
        d = self.cfg.d_llama
        return {"emb": torch.zeros(batch_size, 0, d, device=device, dtype=dtype),
                "mask": torch.zeros(batch_size, 0, device=device, dtype=torch.bool)}

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0, **extra):
        del chunk_offset, extra
        if attention_mask is None:
            attention_mask = torch.ones(token_embeds.shape[:2], device=token_embeds.device, dtype=torch.bool)
        return {"emb": torch.cat([state["emb"], token_embeds], dim=1),
                "mask": torch.cat([state["mask"], attention_mask.bool()], dim=1)}, {}

    def _head_in(self, slot_h: Tensor) -> Tensor:
        """Routing-head input = [struct_norm(slot) ; √d·id] (both streams at the measured √d scale)."""
        return torch.cat([self.struct_norm(slot_h),
                          self.id_head_scale * self.id_embed.unsqueeze(0).expand(slot_h.shape[0], -1, -1)],
                         dim=-1)                                                              # [B,M,2d]

    @staticmethod
    def _sinkhorn1(scores: Tensor) -> Tensor:
        """One Sinkhorn step in LOG-space (logsumexp, no divide → no 1/‖·‖ singularity): row-normalize
        over nodes (each edge commits) then column-normalize over edges (the COMPETITION — a node demanded
        by many edges is pushed down, so edges spread). scores [B,E,N] → balanced log-assignment [B,E,N].
        Single step (not iterated): discourages hubs without forcing a permutation (right for E≈N)."""
        logA = scores - torch.logsumexp(scores, dim=-1, keepdim=True)     # row: per edge over nodes
        logA = logA - torch.logsumexp(logA, dim=-2, keepdim=True)         # col: per node over edges (competition)
        return logA

    def _structure(self, slot_h: Tensor):
        """Content-addressed endpoints with competition + NO self-loops. src is chosen first; dst then
        forbids the src node (mask it out) so dst≠src always → every edge is a genuine two-node relation.
        Returns (src, dst) [B,M,M] hard one-hots (edge rows → node cols) + (soft_src,soft_dst) [B,E,N]."""
        B = slot_h.shape[0]
        he = self._head_in(slot_h)                                # [B,M,2d]
        k = self.k_head(he[:, :self.K])                           # [B,N,dk] node keys
        q_src = self.q_src_head(he[:, self.K:])                   # [B,E,dk] edge src-query
        q_dst = self.q_dst_head(he[:, self.K:])                   # [B,E,dk] edge dst-query
        scale = 1.0 / math.sqrt(self.d_k)
        temp = self.log_temp.exp().clamp_min(1e-2)

        def _to_full(e_oh):                                        # [B,E,N] → [B,M,M] (edge rows, node cols)
            return F.pad(F.pad(e_oh, (0, self.M - self.K)), (0, 0, self.K, 0))

        sc_src = torch.einsum("bed,bnd->ben", q_src, k) * scale / temp     # [B,E,N]
        src_oh = _st_onehot(self._sinkhorn1(sc_src), 1.0)                  # [B,E,N] one-hot src
        # dst: forbid the src node (detached, finite constant → no self-loops, no 0·inf gradient trap)
        sc_dst = torch.einsum("bed,bnd->ben", q_dst, k) * scale / temp - 1e4 * src_oh.detach()
        dst_oh = _st_onehot(self._sinkhorn1(sc_dst), 1.0)                  # [B,E,N] one-hot dst (≠ src)
        return _to_full(src_oh), _to_full(dst_oh), sc_src.softmax(-1), sc_dst.softmax(-1)

    @torch.no_grad()
    def _adaptive_hops(self, src: Tensor, dst: Tensor) -> int:
        """#hops = the predicted graph's DIAMETER (reachability saturation), max over batch, capped."""
        M, B = self.M, src.shape[0]
        eh = (1.0 - self.is_node).view(1, M).expand(B, M)          # [B,M] edge indicator
        sh = (src > 0).float(); dh = (dst > 0).float()
        A = torch.einsum("bm,bmi,bmj->bij", eh, sh, dh)           # [B,M,M] directed adjacency
        eye = torch.eye(M, device=src.device, dtype=A.dtype).unsqueeze(0)
        A = ((A + A.transpose(1, 2)) > 0).float()                  # symmetric, binary
        reach = ((A + eye) > 0).float()
        K = 1
        for k in range(2, self.max_hops + 1):
            nxt = ((reach @ (A + eye)) > 0).float()
            if torch.equal(nxt, reach):
                break
            reach = nxt; K = k
        return max(1, min(K, self.max_hops))

    def _mp_read(self, slot_final: Tensor):
        """Multi-hop residual message-passing read (fp32). Returns (memory, info-with-canaries)."""
        src, dst, soft_src, soft_dst = self._structure(slot_final)
        edge_w = (1.0 - self.is_node).view(1, self.M, 1)          # [1,M,1] edge slots emit
        node_recv = self.is_node.view(1, self.M, 1)              # [1,M,1] node slots receive
        K = self._adaptive_hops(src, dst)
        deg = torch.einsum("bmn,bm->bn", (src + dst).detach(),
                           (1.0 - self.is_node).view(1, self.M).expand(src.shape[0], -1)).clamp(min=1.0)
        pre = self.norm(slot_final)                               # plain read (for the inert-MP canary)

        h = slot_final
        for _ in range(K):
            h_src = torch.einsum("bmn,bnd->bmd", src, h)          # source-endpoint node state per edge
            h_dst = torch.einsum("bmn,bnd->bmd", dst, h)          # dst-endpoint node state per edge
            m_to_dst = self.msg(torch.cat([h_src, h], -1)) * edge_w
            m_to_src = self.msg(torch.cat([h_dst, h], -1)) * edge_w
            agg = (torch.einsum("bmn,bmd->bnd", dst, m_to_dst)       # mean-aggregate at endpoint nodes
                   + torch.einsum("bmn,bmd->bnd", src, m_to_src)) / deg.unsqueeze(-1)
            u = self.update(agg).clamp(-1.0, 1.0)                  # bounded delta (no norm → safe backward)
            h = h + u * node_recv                                  # residual relay, only node slots update
        memory = self.norm(h)

        # ── canaries (detached) ──
        with torch.no_grad():
            def _ent(p):                                           # per-edge confidence (↓=sharp)
                return (-(p.clamp_min(1e-9).log() * p).sum(-1)).mean()
            node_use = (src + dst).sum(dim=(0, 1))[:self.K]        # [N] how many edge-endpoints hit each node
            pu = node_use / node_use.sum().clamp_min(1e-9)
            node_entropy = -(pu.clamp_min(1e-9).log() * pu).sum()  # ↑(→lnK)=spread; ↓=hub-collapse
            src_pick = src[:, self.K:, :self.K].argmax(-1)         # [B,E] ACTUAL src node per edge (Sinkhorn pick)
            dst_pick = dst[:, self.K:, :self.K].argmax(-1)         # [B,E] ACTUAL dst node (≠ src by the mask)
            info = {"hops": K,
                    "delta": (1.0 - F.cosine_similarity(pre, memory, dim=-1)).mean(),
                    "src_entropy": _ent(soft_src), "dst_entropy": _ent(soft_dst),
                    "node_entropy": node_entropy, "ent_max": math.log(self.K),
                    "selfloop": (src_pick == dst_pick).float().mean(),
                    "src_arg": src_pick, "dst_arg": dst_pick}
        return memory, info

    def finalize_memory(self, state):
        emb, mask = state["emb"], state["mask"]                    # [B,T,d], [B,T]
        B, _T, d = emb.shape
        slots0 = (self.slot_init.unsqueeze(0).expand(B, self.M, d)   # content seed
                  + self.id_embed.unsqueeze(0)).to(emb.dtype)        # + fixed identity
        inp = torch.cat([emb, slots0], dim=1)                       # [B,T+M,d]
        attn = torch.cat([mask, torch.ones(B, self.M, device=mask.device, dtype=mask.dtype)], dim=1).long()
        h = self.base.model(inputs_embeds=inp, attention_mask=attn,
                            use_cache=False).last_hidden_state       # bf16 LM forward, no KV cache
        slot_final = h[:, -self.M:]                                 # [B,M,d]
        mp_info = None
        if self.use_structure:
            with torch.autocast("cuda", enabled=False):            # structure + read in FP32 (small + sensitive)
                memory, mp_info = self._mp_read(slot_final.float())
        else:
            memory = self.norm(slot_final.float())                 # plain prepend (id-tagged ICAE control)

        aux = {}
        with torch.no_grad():
            aux["slotgraph_edge_frac"] = torch.tensor(float(1.0 - self.is_node.mean()), device=emb.device)
            aux["slotgraph_mem_effrank"] = torch.tensor(
                _participation_ratio(memory.reshape(-1, memory.shape[-1])), device=emb.device)
            aux["slotgraph_role"] = self.role_fixed[:, 1].long().unsqueeze(0).expand(B, -1)   # [B,M] 0 node 1 edge
            if mp_info is not None:
                aux["slotgraph_invalid_edge_frac"] = torch.zeros((), device=emb.device)  # 0 by construction
                aux["slotgraph_src_entropy"] = mp_info["src_entropy"]
                aux["slotgraph_dst_entropy"] = mp_info["dst_entropy"]
                aux["slotgraph_endpoint_entropy_max"] = torch.tensor(mp_info["ent_max"], device=emb.device)
                aux["slotgraph_node_entropy"] = mp_info["node_entropy"]   # ↑(→lnK)=edges spread; ↓=hub
                aux["slotgraph_selfloop_frac"] = mp_info["selfloop"]
                aux["slotgraph_mp_hops"] = torch.tensor(float(mp_info["hops"]), device=emb.device)
                aux["slotgraph_mp_delta"] = mp_info["delta"]
                aux["slotgraph_src"] = mp_info["src_arg"]
                aux["slotgraph_dst"] = mp_info["dst_arg"]
        return memory, aux

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        st = self.init_streaming_state(token_embeds.shape[0], token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)
