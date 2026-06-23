"""slotgraph — slot memory with a FIXED node/edge partition + multi-hop message-passing read.

Architecture: start from the proven binder — ICAE (own frozen base + encoder-LoRA, M learnable slots
APPENDED to the passage, run through the LM's OWN layers) — and read the slots out through a graph.

  • FIXED partition: slots 0..K-1 are NODES, slots K..M-1 are EDGES. Role is assigned by POSITION, not
    predicted (a predicted role head is unreliable for most of training; fixing it makes "who is a node"
    100%-reliable from step 0. Faithful to TokenGT: types are given, only connectivity is learned).
  • Each EDGE slot predicts, HARD via straight-through, which two NODE slots it links (src, dst), masked
    to the fixed node pool → every edge is a valid node→node link by construction.
  • READ = a multi-hop residual message-passing GNN over the predicted graph: each node relays its own
    state + edge-routed neighbour messages (msg = [source-node ; edge-content]); #hops = the graph's
    diameter (capped). Output is still M vectors (same prepend budget as icae). The prepended memory is a
    FUNCTION of the topology, so the endpoint heads + MP modules get real loss gradient.
    use_structure=False ⇒ plain prepend of the id-tagged slots (id-tagged ICAE control).

MAGNITUDE / GRADIENT POLICY (unified — no free-floating scale coefficients, no gates):
  1. Bound every internal magnitude with a NORMALIZATION, not a gate or a squashing activation. The MP
     read RMSNorm's the node state each hop: bounded by construction (no overflow → sum aggregation is
     safe) and gradient-alive (rescales, never saturates — unlike tanh/sigmoid or a stuck scalar gate).
  2. The ONLY boundary scaling is a MEASURED rescale of the output to the LM's token-embedding norm
     (_NormMatch). Unit-RMS ≠ the LM's input scale, so this one conversion is necessary; it is measured,
     not tuned. Head inputs are likewise balanced to a measured scale (√d) so content and id streams
     match (id_head_scale).
  3. Cold-start is ZERO-INIT (the update output projection starts at 0 → step-0 read = the plain read),
     not a gentle gate. The update grows in via gradient (LoRA-style); no scalar bottleneck to get stuck.
  4. No write into the frozen LM's intermediate residual stream (the old per-layer injection is removed:
     it was inert, unstable, and the one channel a normalization couldn't bound). The only memory write
     is the post-LM, RMSNorm-bounded read.
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


def _rmsnorm(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Magnitude bound that preserves gradient: rescale each row to unit RMS (no learnable gain, no
    saturation). Unlike a scalar gate or tanh, this can't get stuck or kill gradient — it just rescales."""
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


def _st_onehot(logits: Tensor, temp: float) -> Tensor:
    """Straight-through hard selection: one-hot(argmax) in the forward, softmax(logits/temp) gradient
    in the backward. Discrete commitment that still trains."""
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
        # FIXED node/edge partition: slots 0..K-1 are nodes, K..M-1 are edges.
        self.K = max(1, min(int(getattr(cfg, "slotgraph_n_nodes", self.M // 2)), self.M - 1))
        self.use_structure = bool(getattr(cfg, "slotgraph_use_structure", True))
        self.max_hops = int(getattr(cfg, "slotgraph_max_hops", 5))

        # slot content seeds (appended to the passage, icae-style: centered in Llama's token region)
        embed = base.get_input_embeddings()
        with torch.no_grad():
            mean_vec = embed.weight.float().mean(0)
            emb_std = embed.weight.float().std().item()
        slot_init = mean_vec.view(1, d).repeat(self.M, 1) + emb_std * torch.randn(self.M, d)
        self.slot_init = nn.Parameter(slot_init)

        # FIXED near-orthonormal per-position identity codes (buffer, persistent → distinct + stable)
        assert self.M <= d, (f"slotgraph needs M ({self.M}) <= d_llama ({d}) for orthonormal id codes; "
                             f"got M>d → QR would return a [d,d] table, not [M,d].")
        gen = torch.Generator().manual_seed(0)
        q = torch.linalg.qr(torch.randn(d, self.M, generator=gen))[0]    # [d,M] orthonormal cols
        self.register_buffer("id_embed", q.t().contiguous(), persistent=True)   # [M,d] unit-norm rows
        # FIXED role assignment by position.
        is_node = torch.zeros(self.M); is_node[:self.K] = 1.0
        self.register_buffer("is_node", is_node, persistent=False)              # [M] 1=node, 0=edge
        role_fixed = torch.zeros(self.M, 2); role_fixed[:self.K, 0] = 1.0; role_fixed[self.K:, 1] = 1.0
        self.register_buffer("role_fixed", role_fixed, persistent=False)        # [M,2] (for canaries/viz)

        # endpoint heads (shared, applied to the final slot hiddens): edge-slot hidden → src/dst logits.
        # Condition on [struct_norm(slot) ; √d·id]: the id (scaled to the content stream's measured norm)
        # tells the head which slot it is; masked to the node pool so every edge points at a node.
        self.struct_norm = nn.LayerNorm(d)
        self.id_head_scale = math.sqrt(d)
        self.src_head = nn.Linear(2 * d, self.M); self.dst_head = nn.Linear(2 * d, self.M)
        self.log_temp = nn.Parameter(torch.tensor(math.log(float(cfg.slotgraph_temp_init))))

        # ── multi-hop message-passing READ ──────────────────────────────────
        # Residual GNN, shared across hops (recurrent → params don't scale with #hops). msg = [source
        # node ; edge content]; update transforms the summed messages; the node state is RMSNorm-bounded
        # each hop (magnitude policy #1) so sum aggregation can't overflow and gradient stays alive.
        self.msg = nn.Linear(2 * d, d, bias=False)   # [source node ; edge content] → message
        self.update = nn.Linear(d, d, bias=False)    # summed incoming → node-state delta
        nn.init.zeros_(self.update.weight)           # zero-init ⇒ step-0 read = plain read (LoRA-style cold-start)

        self.norm = _NormMatch(d)
        with torch.no_grad():
            self.norm.scale.data.fill_(embed.weight.float().norm(dim=-1).mean().item())
        print(f"[slotgraph] icae-write + FIXED partition ({self.K} nodes / {self.M - self.K} edges) + "
              f"RMSNorm-bounded MP read (≤{self.max_hops} hops, ungated), encoder-LoRA "
              f"r{cfg.slotgraph_lora_rank} ({n_wrapped} layers), use_structure={self.use_structure}")

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
        """Endpoint-head input = [struct_norm(slot) ; √d·id] (both streams at the measured √d scale)."""
        return torch.cat([self.struct_norm(slot_h),
                          self.id_head_scale * self.id_embed.unsqueeze(0).expand(slot_h.shape[0], -1, -1)],
                         dim=-1)                                                              # [B,M,2d]

    def _endpoint_logits(self, hh: Tensor):
        """src/dst logits MASKED to the FIXED node pool (slots 0..K-1) so every edge points at a node."""
        block = (self.is_node < 0.5).view(1, 1, self.M)            # [1,1,M] edge (non-node) positions
        s = self.src_head(hh); d = self.dst_head(hh)              # [B,M,M] (bf16 under autocast)
        s = s.masked_fill(block, torch.finfo(s.dtype).min)        # OUTPUT dtype (bf16 min fits)
        d = d.masked_fill(block, torch.finfo(d.dtype).min)
        return s, d

    def _structure(self, slot_h: Tensor):
        """slot_h [B,M,d] → (src, dst) hard-ST one-hot endpoints over the node pool. Roles are fixed."""
        hh = self._head_in(slot_h)
        temp = self.log_temp.exp().clamp_min(1e-2)
        s_logits, d_logits = self._endpoint_logits(hh)
        return _st_onehot(s_logits, temp), _st_onehot(d_logits, temp)     # [B,M,M] each

    @torch.no_grad()
    def _adaptive_hops(self, src: Tensor, dst: Tensor) -> int:
        """#hops = the predicted graph's DIAMETER (smallest count at which reachability saturates), max
        over the batch, capped at max_hops. Pure control-flow (no grad): beyond the diameter MP adds
        nothing and risks over-smoothing, so this is the principled count, read from the actual wiring."""
        M, B = self.M, src.shape[0]
        eh = (1.0 - self.is_node).view(1, M).expand(B, M)          # [B,M] edge indicator (slots K..M-1)
        sh = (src > 0).float(); dh = (dst > 0).float()
        A = torch.einsum("bm,bmi,bmj->bij", eh, sh, dh)           # [B,M,M] directed adjacency
        eye = torch.eye(M, device=src.device, dtype=A.dtype).unsqueeze(0)
        A = ((A + A.transpose(1, 2)) > 0).float()                  # symmetric (bind both ways), binary
        reach = ((A + eye) > 0).float()
        K = 1
        for k in range(2, self.max_hops + 1):
            nxt = ((reach @ (A + eye)) > 0).float()
            if torch.equal(nxt, reach):
                break
            reach = nxt; K = k
        return max(1, min(K, self.max_hops))

    def _mp_read(self, slot_final: Tensor):
        """Multi-hop residual message-passing read. Nodes relay along the predicted edges (edges→nodes by
        construction); each edge slot conditions the message with its own content; the node state is
        RMSNorm-bounded each hop (no gate). Output is still M vectors. Returns (memory, info)."""
        src, dst = self._structure(slot_final)                    # [B,M,M] hard-ST, over the node pool
        edge_w = (1.0 - self.is_node).view(1, self.M, 1)          # [1,M,1] edge slots emit messages
        node_recv = self.is_node.view(1, self.M, 1)              # [1,M,1] node slots receive
        K = self._adaptive_hops(src, dst)
        pre = self.norm(slot_final.float())                       # plain read (for the inert-MP canary)

        h = slot_final
        for _ in range(K):
            h_src = torch.einsum("bmn,bnd->bmd", src, h)          # source-endpoint node state per edge
            h_dst = torch.einsum("bmn,bnd->bmd", dst, h)          # dst-endpoint node state per edge
            m_to_dst = self.msg(torch.cat([h_src, h], -1)) * edge_w   # src node + THIS edge's content → dst
            m_to_src = self.msg(torch.cat([h_dst, h], -1)) * edge_w   # symmetric: dst node + edge → src
            agg = (torch.einsum("bmn,bmd->bnd", dst, m_to_dst)       # SUM messages at endpoint nodes
                   + torch.einsum("bmn,bmd->bnd", src, m_to_src))    # (per-hop RMSNorm below bounds it)
            h = _rmsnorm(h + self.update(agg) * node_recv)         # bounded, gradient-alive, ungated relay
        memory = self.norm(h.float())
        info = {"hops": K, "delta": (1.0 - F.cosine_similarity(pre, memory, dim=-1)).mean().detach()}
        return memory, info

    def finalize_memory(self, state):
        emb, mask = state["emb"], state["mask"]                    # [B,T,d], [B,T]
        B, _T, d = emb.shape
        slots0 = (self.slot_init.unsqueeze(0).expand(B, self.M, d)   # content seed
                  + self.id_embed.unsqueeze(0)).to(emb.dtype)        # + fixed identity
        inp = torch.cat([emb, slots0], dim=1)                       # [B,T+M,d]
        attn = torch.cat([mask, torch.ones(B, self.M, device=mask.device, dtype=mask.dtype)], dim=1).long()
        h = self.base.model(inputs_embeds=inp, attention_mask=attn,
                            use_cache=False).last_hidden_state       # single fwd, no generation → no KV cache
        slot_final = h[:, -self.M:]                                 # [B,M,d]
        mp_info = None
        if self.use_structure:
            memory, mp_info = self._mp_read(slot_final)            # multi-hop topology-using read
        else:
            memory = self.norm(slot_final.float())                 # plain prepend (id-tagged ICAE control)

        aux = {}
        with torch.no_grad():
            temp = self.log_temp.exp().clamp_min(1e-2)
            s_logits, d_logits = self._endpoint_logits(self._head_in(slot_final))
            src_soft = (s_logits / temp).softmax(-1)
            dst_soft = (d_logits / temp).softmax(-1)

            def _ent(p):
                return (-(p.clamp_min(1e-9).log() * p).sum(-1)).mean()
            aux["slotgraph_edge_frac"] = torch.tensor(float(1.0 - self.is_node.mean()), device=emb.device)
            aux["slotgraph_invalid_edge_frac"] = torch.zeros((), device=emb.device)  # 0 by construction
            aux["slotgraph_src_entropy"] = _ent(src_soft)          # ↓ = confident edges; ↑ = arbitrary
            aux["slotgraph_dst_entropy"] = _ent(dst_soft)
            aux["slotgraph_temp"] = temp.detach()
            aux["slotgraph_mem_effrank"] = torch.tensor(
                _participation_ratio(memory.reshape(-1, memory.shape[-1])), device=emb.device)
            aux["slotgraph_role"] = self.role_fixed[:, 1].long().unsqueeze(0).expand(B, -1)   # [B,M] 0 node 1 edge
            aux["slotgraph_src"] = src_soft.argmax(-1)             # [B,M]
            aux["slotgraph_dst"] = dst_soft.argmax(-1)             # [B,M]
            if mp_info is not None:
                aux["slotgraph_mp_hops"] = torch.tensor(float(mp_info["hops"]), device=emb.device)
                aux["slotgraph_mp_delta"] = mp_info["delta"]       # 1-cos(plain,read): 0 ⇒ MP inert, ↑ ⇒ used
        return memory, aux

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        st = self.init_streaming_state(token_embeds.shape[0], token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)
