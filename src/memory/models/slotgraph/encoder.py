"""slotgraph — slot memory with a FIXED node/edge partition + multi-hop message-passing read.

Architecture: start from the proven binder — ICAE (own frozen base + encoder-LoRA, M learnable slots
APPENDED to the passage, run through the LM's OWN layers) — and impose a graph structure on the slots.

  • FIXED partition: slots 0..K-1 are NODES, slots K..M-1 are EDGES. Role is assigned by POSITION,
    not predicted. (An earlier version predicted node-vs-edge per slot, but that head is unreliable for
    most of training, which made the "edges only connect nodes" mask unreliable too. Fixing the roles
    makes "who is a node" 100%-reliable from step 0 and removes a whole collapse mode. This is also more
    faithful to TokenGT, where node/edge TYPES are given; only connectivity + content are learned.)
  • Per LM layer (from start_layer) each EDGE slot predicts, HARD via straight-through, which two NODE
    slots it links (src, dst) — masked to the fixed node pool, so every edge is a valid node→node link.
    That is concretized into a TokenGT-style structural embedding (role type + id codes) and RE-INJECTED
    into the slot positions before the next layer, so the next layer SEES the relationships just formed.
  • READ = a multi-hop residual message-passing GNN over the predicted graph: each node relays its own
    state + edge-routed neighbour messages (msg=[source-node ; edge-content]); #hops = the graph's
    diameter (capped). Output is still M vectors (same prepend budget as icae). This makes the prepended
    memory a FUNCTION of the topology, so the edge heads + MP modules get real loss gradient.

Design decisions:
  • The LM does the information mixing (its pretrained layers); only the tiny endpoint heads + the MP
    modules + encoder-LoRA are new → param-matched to icae. use_structure=False ⇒ id-tagged ICAE.
  • HARD (straight-through) endpoints: src/dst commit to one-hot in the forward (a real discrete edge,
    crisp & matchable) with a soft-surrogate gradient. (Soft = the membership-pool smear we keep losing.)
  • ID = slot POSITION via a FIXED near-orthonormal code table (a buffer, not learnable → distinct,
    reload-stable). Edge endpoint code = id[src] + id[dst] (transparent SUM) so attention can match an
    edge back to its two endpoint nodes. No aux loss, no pooling read.

MAGNITUDE POLICY (unified — no free-floating scale hyperparameters; everything is measured or bounded):
  1. Mixing weights are sigmoid(raw) ∈ [0,1] — naturally bounded, NO arbitrary `*_max` caps. Inits are a
     gentle start (≈0.1), not tuned ceilings.
  2. A learnable vector that combines ADDITIVELY with a fixed unit-norm reference is CLAMP-capped at the
     reference's scale (role_embed ≤ the id codes' unit norm). Clamp, NOT normalize: normalize has a
     1/‖·‖ gradient singularity as ‖·‖→0 AND forces full magnitude (kills self-regulation). Clamp caps
     the upside (stops the injection growing out-of-distribution → frozen-LM amplification → overflow)
     while still letting the model shrink it.
  3. Aggregates are mean/degree-normalized (the MP read) — magnitude is degree-invariant.
  4. Output / combined-stream magnitudes are matched to MEASURED references: the read → the LM's token
     embedding norm (_NormMatch); head inputs → √d so content and id streams are balanced (id_head_scale).
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
        n_layers = base.config.num_hidden_layers
        sl = int(getattr(cfg, "slotgraph_start_layer", 0))
        self.start_layer = sl if 0 <= sl < n_layers else 0
        self.use_structure = bool(getattr(cfg, "slotgraph_use_structure", True))
        # inject = write the predicted structure back into the slot hiddens per-layer. inject=False keeps
        # the structure heads + MP read but drops the per-layer injection → "MP-read-only" (the STABLE
        # ablation; injection-ONLY diverges because its objective over-drives the fragile injection).
        self.inject = bool(getattr(cfg, "slotgraph_inject", True))

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
        # FIXED role assignment by position: is_node[m]=1 for m<K (node), 0 for m>=K (edge).
        is_node = torch.zeros(self.M); is_node[:self.K] = 1.0
        self.register_buffer("is_node", is_node, persistent=False)              # [M] node indicator
        role_fixed = torch.zeros(self.M, 2); role_fixed[:self.K, 0] = 1.0; role_fixed[self.K:, 1] = 1.0
        self.register_buffer("role_fixed", role_fixed, persistent=False)        # [M,2] one-hot node/edge
        self.role_embed = nn.Parameter(torch.randn(2, d) / math.sqrt(d))        # [node, edge] type tags

        # endpoint heads (shared across layers): edge-slot hidden → src / dst logits over slot positions
        # (masked to the fixed node pool). Condition on [struct_norm(slot) ; scaled id] so the head sees
        # both content and the slot's own position; id scaled to the content stream's magnitude (√d).
        self.struct_norm = nn.LayerNorm(d)
        self.id_head_scale = math.sqrt(d)
        self.src_head = nn.Linear(2 * d, self.M); self.dst_head = nn.Linear(2 * d, self.M)
        self.log_temp = nn.Parameter(torch.tensor(math.log(float(cfg.slotgraph_temp_init))))
        # injection strength = sigmoid(raw) ∈ [0,1] (a naturally-bounded mixing weight — no arbitrary
        # cap; the injected e is itself bounded by the role_embed clamp). Init for a gentle ~0.1 start.
        self.inject_raw = nn.Parameter(torch.tensor(-2.197))  # sigmoid(-2.197) ≈ 0.1

        # ── multi-hop message-passing READ ──────────────────────────────────
        # Make the prepended memory a FUNCTION of the topology, so the edge heads get loss gradient (a
        # plain-prepend read leaves them inert). Canonical residual GNN, shared across hops (recurrent →
        # params don't scale with K_hops): each node relays its own state + edge-routed neighbour
        # messages. msg=[source-node ; edge-content]; update transforms the aggregate; the residual
        # self-term keeps each node's identity (the over-smoothing guard).
        self.use_mp_read = bool(getattr(cfg, "slotgraph_mp_read", True))
        self.max_hops = int(getattr(cfg, "slotgraph_max_hops", 5))
        # bias=False is load-bearing: with no edge messages agg=0, so update(0) MUST be 0 → the read is
        # then EXACTLY the identity (plain prepend). A bias would let the read move node slots WITHOUT
        # any graph message — a content-free bypass that defeats the point of a topology read.
        self.msg = nn.Linear(2 * d, d, bias=False)   # [source node ; edge content] → message
        self.update = nn.Linear(d, d, bias=False)    # aggregated incoming → node-state delta
        # MP read gate = sigmoid(raw) ∈ [0,1] (naturally bounded — no cap). Gentle ~0.1 start so step-0 ≈
        # plain read, but nonzero so gradient reaches msg/update/heads from step 0 (no zero-init block).
        self.mp_gate_raw = nn.Parameter(torch.tensor(-2.197))   # sigmoid(-2.197) ≈ 0.1

        self.norm = _NormMatch(d)
        with torch.no_grad():
            self.norm.scale.data.fill_(embed.weight.float().norm(dim=-1).mean().item())
        print(f"[slotgraph] icae-write + FIXED partition ({self.K} nodes / {self.M - self.K} edges), "
              f"encoder-LoRA r{cfg.slotgraph_lora_rank} ({n_wrapped} layers), structure from layer "
              f"{self.start_layer}/{n_layers}, use_structure={self.use_structure}, "
              f"mp_read={self.use_mp_read} (≤{self.max_hops} hops)")

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
        """Endpoint-head input = [struct_norm(slot) ; scaled fixed id]. The id is scaled to the content
        stream's magnitude (√d) so it isn't swamped → the head sees the slot's content AND its position."""
        return torch.cat([self.struct_norm(slot_h),
                          self.id_head_scale * self.id_embed.unsqueeze(0).expand(slot_h.shape[0], -1, -1)],
                         dim=-1)                                                              # [B,M,2d]

    def _endpoint_logits(self, hh: Tensor):
        """src/dst logits MASKED to the FIXED node pool (slots 0..K-1), so every edge points only at a
        node (a valid node→node link, by construction). The mask is a constant — no dependence on an
        unreliable role prediction."""
        block = (self.is_node < 0.5).view(1, 1, self.M)            # [1,1,M] edge (non-node) positions
        s = self.src_head(hh); d = self.dst_head(hh)              # [B,M,M] (bf16 under autocast)
        s = s.masked_fill(block, torch.finfo(s.dtype).min)        # use the OUTPUT dtype (bf16 min fits)
        d = d.masked_fill(block, torch.finfo(d.dtype).min)
        return s, d

    def _role_e(self) -> Tensor:
        """role_embed rows CLAMP-capped at the fixed id codes' unit scale (magnitude policy #2). Clamp,
        not normalize: normalize has a 1/‖·‖ gradient singularity as a row → cancels an id code, and it
        forces full magnitude (the model can no longer shrink the injection to self-regulate). Clamp caps
        the upside so the injected structure can't grow out-of-distribution (→ frozen-LM overflow)."""
        n = self.role_embed.norm(dim=-1, keepdim=True).clamp(min=1e-6)   # [2,1]
        return self.role_embed * (1.0 / n).clamp(max=1.0)               # rows capped at unit L2 (id scale)

    # ── structure concretization (fixed roles; per-layer-shared endpoint heads) ──
    def _structure(self, slot_h: Tensor):
        """slot_h [B,M,d] → (e [B,M,d] structural embed, role[B,M,2] fixed, src/dst[B,M,M] hard-ST).
        Node slots get a node type+id embed; edge slots get a type+SUM-of-endpoint-ids embed."""
        B = slot_h.shape[0]
        hh = self._head_in(slot_h)                                  # [B,M,2d]
        temp = self.log_temp.exp().clamp_min(1e-2)
        s_logits, d_logits = self._endpoint_logits(hh)             # edges→nodes ONLY (constant mask)
        src = _st_onehot(s_logits, temp)                           # [B,M,M] one-hot src (a node position)
        dst = _st_onehot(d_logits, temp)                           # [B,M,M] one-hot dst (a node position)
        role_e = self._role_e()                                    # [2,d] clamp-bounded type tags
        is_node = self.is_node.view(1, self.M, 1)                  # [1,M,1] fixed partition
        node_e = (role_e[0] + self.id_embed).unsqueeze(0)          # [1,M,d] node id = own position
        endp = torch.einsum("bmn,nd->bmd", src, self.id_embed) + torch.einsum("bmn,nd->bmd", dst, self.id_embed)
        edge_e = role_e[1] + endp                                  # [B,M,d] transparent SUM of endpoint ids
        e = is_node * node_e + (1.0 - is_node) * edge_e            # node slots → node_e, edge slots → edge_e
        role = self.role_fixed.unsqueeze(0).expand(B, -1, -1)      # [B,M,2] fixed (constant)
        return e, role, src, dst

    @torch.no_grad()
    def _adaptive_hops(self, role: Tensor, src_n: Tensor, dst_n: Tensor) -> int:
        """#hops = the predicted graph's DIAMETER (smallest K_hops at which reachability saturates),
        taken as the MAX over the batch and capped at max_hops. Pure control-flow (no grad): it only
        decides how many differentiable hops to run. Beyond the diameter, message passing adds nothing
        and risks over-smoothing — so this is the principled count. N and E alone don't determine it (a
        star and a path on the same counts have diameters 2 vs N-1), so we read the actual wiring."""
        M = self.M
        eh = (role[..., 1] >= 0.5).float()                          # [B,M] edge indicator
        sh = (src_n > 0).float(); dh = (dst_n > 0).float()          # node-masked hard endpoints
        A = torch.einsum("bm,bmi,bmj->bij", eh, sh, dh)            # [B,M,M] directed adjacency
        eye = torch.eye(M, device=A.device, dtype=A.dtype).unsqueeze(0)
        A = (((A + A.transpose(1, 2)) > 0).float())                 # symmetric (bind both ways), binary
        reach = ((A + eye) > 0).float()                            # 1-step reachability
        K = 1
        for k in range(2, self.max_hops + 1):
            nxt = ((reach @ (A + eye)) > 0).float()
            if torch.equal(nxt, reach):                            # saturated across the WHOLE batch
                break
            reach = nxt; K = k
        return max(1, min(K, self.max_hops))

    def _mp_read(self, slot_final: Tensor):
        """Multi-hop residual message-passing read over the predicted graph. Nodes relay along the
        predicted edges (edges→nodes by construction); each edge slot conditions the message with its
        own content; the residual self-term keeps node identity. Output is still M vectors (same prepend
        budget as icae) — node slots get neighbour-enriched, edge slots keep their relation content.
        Returns (memory [B,M,d], info dict for canaries)."""
        _, role, src, dst = self._structure(slot_final)            # role fixed; src/dst [B,M,M] hard-ST
        edge_w = role[..., 1:2]                                     # [B,M,1] edge gate (1 at edge slots)
        node_ind = self.is_node.view(1, self.M)                    # [1,M] fixed node indicator
        nm = node_ind.unsqueeze(1)                                 # [1,1,M] over endpoint POSITIONS
        src_n = src * nm                                           # (idempotent: src already node-masked)
        dst_n = dst * nm
        node_recv = self.is_node.view(1, self.M, 1)               # [1,M,1] only node slots receive

        K = self._adaptive_hops(role, src_n, dst_n)
        gate = torch.sigmoid(self.mp_gate_raw)                    # ∈ [0,1], naturally bounded
        pre = self.norm(slot_final.float())                       # icae read (for the inert-MP canary)

        h = slot_final
        for _ in range(K):
            h_src = torch.einsum("bmn,bnd->bmd", src_n, h)         # source-endpoint node state per edge
            h_dst = torch.einsum("bmn,bnd->bmd", dst_n, h)         # dst-endpoint node state per edge
            m_to_dst = self.msg(torch.cat([h_src, h], -1)) * edge_w   # src node + THIS edge's content → dst
            m_to_src = self.msg(torch.cat([h_dst, h], -1)) * edge_w   # symmetric: dst node + edge → src
            agg = (torch.einsum("bmn,bmd->bnd", dst_n, m_to_dst)      # gather messages at endpoint nodes
                   + torch.einsum("bmn,bmd->bnd", src_n, m_to_src))
            # MEAN aggregation (divide by in-degree): a node that many edges point to would otherwise get
            # a sum that scales with its degree, and over the multi-hop residual that compounds and can
            # overflow bf16 on a concentrated graph → NaN. Degree-normalizing bounds each hop to ~|msg|.
            deg = torch.einsum("bmn,bm->bn", (src_n + dst_n).detach(),
                               edge_w.squeeze(-1)).clamp(min=1.0)     # [B,M] incident-edge count per node
            agg = agg / deg.unsqueeze(-1)
            h = h + gate * self.update(agg) * node_recv            # residual relay; only nodes update
        memory = self.norm(h.float())
        info = {"hops": K, "gate": gate.detach(),
                "delta": (1.0 - F.cosine_similarity(pre, memory, dim=-1)).mean().detach()}
        return memory, info

    def _install_struct_hooks(self):
        """Per-LM-layer forward-pre-hooks (from start_layer): read slot hiddens → structure → inject
        into the slot positions before the layer runs. Differentiable → trains the endpoint heads."""
        handles = []
        layers = self.base.model.layers

        def _mk():
            def _hook(_module, args, kwargs):
                h = args[0] if args else kwargs.get("hidden_states")
                if h is None or h.shape[1] < self.M:
                    return None
                e, _, _, _ = self._structure(h[:, -self.M:])
                scale = torch.sigmoid(self.inject_raw)            # ∈ [0,1]; e is bounded by the role clamp
                new = h.clone()
                new[:, -self.M:] = new[:, -self.M:] + scale * e.to(h.dtype)
                if args:
                    return (new,) + tuple(args[1:]), kwargs
                kw = dict(kwargs); kw["hidden_states"] = new
                return args, kw
            return _hook
        for layer in layers[self.start_layer:]:
            handles.append(layer.register_forward_pre_hook(_mk(), with_kwargs=True))
        return handles

    def finalize_memory(self, state):
        emb, mask = state["emb"], state["mask"]                    # [B,T,d], [B,T]
        B, _T, d = emb.shape
        slots0 = (self.slot_init.unsqueeze(0).expand(B, self.M, d)   # content seed
                  + self.id_embed.unsqueeze(0)).to(emb.dtype)        # + fixed identity
        inp = torch.cat([emb, slots0], dim=1)                       # [B,T+M,d]
        attn = torch.cat([mask, torch.ones(B, self.M, device=mask.device, dtype=mask.dtype)], dim=1).long()

        handles = self._install_struct_hooks() if (self.use_structure and self.inject) else []
        try:
            h = self.base.model(inputs_embeds=inp, attention_mask=attn,
                                use_cache=False).last_hidden_state    # single fwd, no generation → no KV cache
        finally:
            for hh in handles:
                hh.remove()
        slot_final = h[:, -self.M:]                                 # [B,M,d]
        mp_info = None
        if self.use_structure and self.use_mp_read:
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
            edge_frac = float(1.0 - self.is_node.mean())           # FIXED = (M-K)/M
            aux["slotgraph_edge_frac"] = torch.tensor(edge_frac, device=emb.device)
            aux["slotgraph_invalid_edge_frac"] = torch.zeros((), device=emb.device)  # 0 by construction
            aux["slotgraph_src_entropy"] = _ent(src_soft)          # ↓ = confident edges; ↑ = arbitrary
            aux["slotgraph_dst_entropy"] = _ent(dst_soft)
            aux["slotgraph_role_entropy"] = torch.zeros((), device=emb.device)  # roles fixed → 0
            aux["slotgraph_temp"] = temp.detach()
            aux["slotgraph_inject_scale"] = torch.sigmoid(self.inject_raw).detach()
            aux["slotgraph_mem_effrank"] = torch.tensor(
                _participation_ratio(memory.reshape(-1, memory.shape[-1])), device=emb.device)
            # for viz: fixed role + predicted endpoints of the final structure
            aux["slotgraph_role"] = self.role_fixed[:, 1].long().unsqueeze(0).expand(B, -1)   # [B,M] 0 node 1 edge
            aux["slotgraph_src"] = src_soft.argmax(-1)             # [B,M]
            aux["slotgraph_dst"] = dst_soft.argmax(-1)             # [B,M]
            if mp_info is not None:                                # message-passing read canaries
                aux["slotgraph_mp_hops"] = torch.tensor(float(mp_info["hops"]), device=emb.device)
                aux["slotgraph_mp_gate"] = mp_info["gate"]         # injection strength of the MP delta
                aux["slotgraph_mp_delta"] = mp_info["delta"]       # 1-cos(pre,post): 0 ⇒ MP inert, ↑ ⇒ used
        return memory, aux

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        st = self.init_streaming_state(token_embeds.shape[0], token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)
