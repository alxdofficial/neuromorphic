"""slotgraph — emergent-topology slot memory (icae write + per-layer hard structure).

Architecture (a1): start from the actual proven binder — ICAE (own frozen base + encoder-LoRA, M
learnable slots APPENDED to the passage, run through the LM's OWN layers) — and let the model
DISCOVER graph structure on top. At each LM layer (from start_layer) a small head reads each slot's
hidden and predicts, HARD via straight-through:
  • role: is this slot a NODE or an EDGE,
  • for edges: which two slot-POSITIONS (src, dst) it connects.
That prediction is concretized into a TokenGT-style structural embedding and RE-INJECTED into the
slot positions before the next layer, so the next layer SEES the relationships just formed. The read
is a plain PREPEND of the slots' final hiddens (every decoder layer attends them, as usual).

Design decisions (from the design discussion / arrivalmem audit):
  • The LM does the information mixing (its 30 pretrained layers); only the tiny structure heads +
    encoder-LoRA are new → param-matched to icae, and use_structure=False degrades to *pure icae*.
  • HARD (straight-through): role/src/dst commit to one-hot in the forward (real discrete topology,
    crisp & matchable) with a soft-surrogate gradient — discreteness is the bit that beats flat, and
    the id-matching only works when sharp. (Soft = the membership-pool smear we keep losing to.)
  • ID = slot POSITION via a FIXED near-orthonormal code table (a buffer, not learnable → guaranteed
    distinct, reload-stable; the arrivalmem node_id lesson). Edge endpoint code = id[src] + id[dst]
    (transparent SUM, NOT an MLP) so attention can match an edge back to its two endpoint nodes.
  • Injection is gated by a small learnable scale (gentle cold-start; the structure heads still get
    gradient from step 0). No aux loss, no pooling read.
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
    is_conditioned_read = False              # READ = PREPEND the M slots' final hiddens
    # (no reinforce_prepend_each_layer: structure is injected during the ENCODER forward, baked in)

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
        n_layers = base.config.num_hidden_layers
        sl = int(getattr(cfg, "slotgraph_start_layer", 0))
        self.start_layer = sl if 0 <= sl < n_layers else 0
        self.use_structure = bool(getattr(cfg, "slotgraph_use_structure", True))

        # slot content seeds (appended to the passage, icae-style: centered in Llama's token region)
        embed = base.get_input_embeddings()
        with torch.no_grad():
            mean_vec = embed.weight.float().mean(0)
            emb_std = embed.weight.float().std().item()
        slot_init = mean_vec.view(1, d).repeat(self.M, 1) + emb_std * torch.randn(self.M, d)
        self.slot_init = nn.Parameter(slot_init)

        # FIXED near-orthonormal per-position identity codes (buffer, persistent → distinct + stable)
        gen = torch.Generator().manual_seed(0)
        q = torch.linalg.qr(torch.randn(d, self.M, generator=gen))[0]    # [d,M] orthonormal cols
        self.register_buffer("id_embed", q.t().contiguous(), persistent=True)   # [M,d] unit-norm rows
        self.role_embed = nn.Parameter(torch.randn(2, d) / math.sqrt(d))        # [node, edge]

        # structure heads (shared across layers): slot hidden → role / src / dst logits
        self.struct_norm = nn.LayerNorm(d)
        # heads condition on [slot hidden ; fixed id]: the orthonormal id guarantees a DISTINCT input
        # per slot, so the hard predictions vary across slots from init (a node/edge MIX, not the
        # all-one-class collapse a content-only head gives when the slots start similar — which would
        # starve the endpoint heads, since a hard all-node forward zeroes the edge branch).
        self.role_head = nn.Linear(2 * d, 2)
        self.src_head = nn.Linear(2 * d, self.M); self.dst_head = nn.Linear(2 * d, self.M)
        self.log_temp = nn.Parameter(torch.tensor(math.log(float(cfg.slotgraph_temp_init))))
        # BOUNDED injection: scale = inject_max·sigmoid(raw) ∈ (0, inject_max). Re-injecting an
        # unbounded learnable scalar before all 30 LM layers can distort the residual stream; cap it.
        self.inject_max = 0.5
        self.inject_raw = nn.Parameter(torch.tensor(-1.386))  # 0.5·sigmoid(-1.386) ≈ 0.1 at init

        self.norm = _NormMatch(d)
        with torch.no_grad():
            self.norm.scale.data.fill_(embed.weight.float().norm(dim=-1).mean().item())
        print(f"[slotgraph] icae-write + hard-ST structure: M={self.M}, encoder-LoRA r{cfg.slotgraph_lora_rank} "
              f"({n_wrapped} layers), structure from layer {self.start_layer}/{n_layers}, "
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

    # ── the hard-ST structure prediction + concretization (one shared head, used per layer) ──
    def _structure(self, slot_h: Tensor):
        """slot_h [B,M,d] → (e [B,M,d] structural embed, role/src/dst for canaries). HARD straight-through."""
        hh = torch.cat([self.struct_norm(slot_h),
                        self.id_embed.unsqueeze(0).expand(slot_h.shape[0], -1, -1)], dim=-1)  # [B,M,2d]
        temp = self.log_temp.exp().clamp_min(1e-2)
        role_soft = (self.role_head(hh) / temp).softmax(-1)         # [B,M,2] (needed for the grad-bypass)
        role = _st_onehot(self.role_head(hh), temp)                 # [B,M,2] hard role
        src = _st_onehot(self.src_head(hh), temp)                   # [B,M,M] one-hot src position
        dst = _st_onehot(self.dst_head(hh), temp)                   # [B,M,M] one-hot dst position
        node_e = (self.role_embed[0] + self.id_embed).unsqueeze(0)  # [1,M,d] node id = own position
        endp = torch.einsum("bmn,nd->bmd", src, self.id_embed) + torch.einsum("bmn,nd->bmd", dst, self.id_embed)
        edge_e = self.role_embed[1] + endp                          # [B,M,d] transparent SUM of endpoint ids
        e = role[..., 0:1] * node_e + role[..., 1:2] * edge_e       # [B,M,d] FORWARD = hard node XOR edge
        # gradient bypass: a zero-valued term whose gradient gives the endpoint heads a SOFT-weighted
        # signal even for node-classified slots, so src/dst don't freeze if role drifts all-node
        # (the hard gate alone zeroes the edge branch's gradient for node slots).
        bypass = role_soft[..., 1:2] * edge_e
        e = e + (bypass - bypass.detach())                         # forward += 0; backward feeds src/dst
        return e, role, src, dst

    def _install_struct_hooks(self):
        """Per-LM-layer forward-pre-hooks (from start_layer): read slot hiddens → hard structure →
        inject into the slot positions before the layer runs. Differentiable → trains the heads."""
        handles = []
        layers = self.base.model.layers

        def _mk():
            def _hook(module, args, kwargs):
                h = args[0] if args else kwargs.get("hidden_states")
                if h is None or h.shape[1] < self.M:
                    return None
                e, _, _, _ = self._structure(h[:, -self.M:])
                scale = self.inject_max * torch.sigmoid(self.inject_raw)   # bounded ∈ (0, inject_max)
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
        B, T, d = emb.shape
        slots0 = (self.slot_init.unsqueeze(0).expand(B, self.M, d)   # content seed
                  + self.id_embed.unsqueeze(0)).to(emb.dtype)        # + fixed identity
        inp = torch.cat([emb, slots0], dim=1)                       # [B,T+M,d]
        attn = torch.cat([mask, torch.ones(B, self.M, device=mask.device, dtype=mask.dtype)], dim=1).long()

        handles = self._install_struct_hooks() if self.use_structure else []
        try:
            h = self.base.model(inputs_embeds=inp, attention_mask=attn).last_hidden_state
        finally:
            for hh in handles:
                hh.remove()
        slot_final = h[:, -self.M:]                                 # [B,M,d]
        memory = self.norm(slot_final.float())                     # [B,M,d_llama]

        aux = {}
        with torch.no_grad():
            hh = torch.cat([self.struct_norm(slot_final),
                            self.id_embed.unsqueeze(0).expand(slot_final.shape[0], -1, -1)], dim=-1)
            temp = self.log_temp.exp().clamp_min(1e-2)
            role_soft = (self.role_head(hh) / temp).softmax(-1)
            src_soft = (self.src_head(hh) / temp).softmax(-1)
            dst_soft = (self.dst_head(hh) / temp).softmax(-1)

            def _ent(p):
                return (-(p.clamp_min(1e-9).log() * p).sum(-1)).mean()
            aux["slotgraph_edge_frac"] = (role_soft.argmax(-1) == 1).float().mean()   # hard edge fraction
            aux["slotgraph_src_entropy"] = _ent(src_soft)          # ↓ = confident edges; ↑ = arbitrary
            aux["slotgraph_dst_entropy"] = _ent(dst_soft)
            aux["slotgraph_role_entropy"] = _ent(role_soft)
            aux["slotgraph_temp"] = temp.detach()
            aux["slotgraph_inject_scale"] = (self.inject_max * torch.sigmoid(self.inject_raw)).detach()
            aux["slotgraph_mem_effrank"] = torch.tensor(
                _participation_ratio(memory.reshape(-1, memory.shape[-1])), device=emb.device)
            # for viz: discrete role + endpoints of the final structure
            aux["slotgraph_role"] = role_soft.argmax(-1)           # [B,M] 0=node 1=edge
            aux["slotgraph_src"] = src_soft.argmax(-1)             # [B,M]
            aux["slotgraph_dst"] = dst_soft.argmax(-1)             # [B,M]
        return memory, aux

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        st = self.init_streaming_state(token_embeds.shape[0], token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)
