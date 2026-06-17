"""Factorization experiments: does discreteness help (Exp 1) and does the graph WRITE
help (Exp 2b), each isolated vs a plain slot-attention control, at matched read-rate
(M prepend tokens × d_llama) and matched params. All three:
  • tap a FROZEN backbone layer for the observation (ICAE/CCM pattern, like GraphEncoder),
  • form M memory tokens, project to d_llama, and PREPEND (read by the frozen decoder),
  • are k-sliced like the rest of the MAE compressor cohort.

  SlotAttentionEncoder  — control: M free slots (Slot Attention, Locatello 2020).
  VocabSlotEncoder      — Exp 1: each slot constrained to a SPARSE (entmax) combination
                          of a learned node bank → tests DISCRETENESS (of content).
  FreeGraphEncoder      — Exp 2b: the GraphParser write with FREE-regressed endpoints
                          (no bank, no selection); snapshot → E edge tokens → prepend.
                          Tests the graph WRITE machinery, no discreteness/topology.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor

from ...config import ReprConfig
from ..graph.substrate import _Attn, _ParserBlock, _rmsnorm, entmax


# ── masked slot attention (Locatello 2020; softmax-over-slots competition + mask) ──
class _MaskedSlotAttn(nn.Module):
    def __init__(self, n_slots: int, d: int, iters: int = 3):
        super().__init__()
        self.n_slots, self.d, self.iters = n_slots, d, iters
        self.mu = nn.Parameter(torch.zeros(1, 1, d))
        self.log_sigma = nn.Parameter(torch.zeros(1, 1, d))
        self.norm_in = nn.LayerNorm(d); self.norm_slots = nn.LayerNorm(d); self.norm_mlp = nn.LayerNorm(d)
        self.to_q = nn.Linear(d, d, bias=False); self.to_k = nn.Linear(d, d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)
        self.gru = nn.GRUCell(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.scale = d ** -0.5

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        B, N, _ = inputs.shape
        x = self.norm_in(inputs)
        k = self.to_k(x); v = self.to_v(x)
        neg = torch.finfo(inputs.dtype).min
        col = (~mask.bool())[:, None, :]                                   # [B,1,N] pad columns
        slots = self.mu + self.log_sigma.exp() * torch.randn(
            B, self.n_slots, self.d, device=inputs.device, dtype=inputs.dtype)
        for _ in range(self.iters):
            q = self.to_q(self.norm_slots(slots)) * self.scale            # [B,K,d]
            logits = torch.einsum("bkd,bnd->bkn", q, k)                    # [B,K,N]
            logits = logits.masked_fill(col, neg)                         # don't attend pad
            attn = logits.softmax(dim=1) + 1e-8                           # over SLOTS → competition
            attn = attn.masked_fill(col, 0.0)
            w = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)     # per-slot mean over N
            updates = torch.einsum("bkn,bnd->bkd", w, v)                  # [B,K,d]
            slots = self.gru(updates.reshape(-1, self.d),
                             slots.reshape(-1, self.d)).reshape(B, self.n_slots, self.d)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots                                                      # [B,K,d]


# ── shared base: frozen obs tap + streaming + finalize → prepend memory ──────────
class _SlotMemBase(nn.Module):
    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        from ...decoder import load_frozen_llama
        base, _ = load_frozen_llama(cfg.llama_model)
        for p_ in base.parameters():
            p_.requires_grad_(False)
        self.base = base
        n_layers = base.config.num_hidden_layers
        ot = cfg.graph_obs_tap_layer
        self.obs_tap_layer = ot if 0 <= ot < n_layers else max(1, round(0.20 * n_layers))
        self.d_llama = cfg.d_llama
        self._build(cfg)

    def _build(self, cfg):           # subclass: create modules
        raise NotImplementedError

    def _form(self, hiddens: Tensor, mask: Tensor) -> Tensor:   # subclass: → [B,M,d_llama]
        raise NotImplementedError

    def train(self, mode: bool = True):
        super().train(mode); self.base.train(False); return self

    def init_streaming_state(self, batch_size, device, dtype):
        del batch_size, dtype
        return {"hiddens": None, "mask": None, "device": device}

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0, **extra):
        del chunk_offset, extra
        B, W = token_embeds.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones(B, W, device=token_embeds.device, dtype=torch.bool)
        with torch.no_grad():
            out = self.base.model(inputs_embeds=token_embeds, attention_mask=attention_mask.long(),
                                  output_hidden_states=True, use_cache=False)
            h = out.hidden_states[self.obs_tap_layer + 1]
        if state["hiddens"] is None:
            state["hiddens"] = h.float(); state["mask"] = attention_mask.bool()
        else:
            state["hiddens"] = torch.cat([state["hiddens"], h.float()], dim=1)
            state["mask"] = torch.cat([state["mask"], attention_mask.bool()], dim=1)
        return state, {}

    def finalize_memory(self, state):
        memory = self._form(state["hiddens"], state["mask"])     # [B,M,d_llama]
        return memory, {}

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        st = self.init_streaming_state(token_embeds.shape[0], token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)


# ── control: M free slots ────────────────────────────────────────────────────────
class SlotAttentionEncoder(_SlotMemBase):
    def _build(self, cfg):
        d = cfg.slotmem_d_slot
        self.obs_proj = nn.Linear(cfg.d_llama, d)
        self.slots = _MaskedSlotAttn(cfg.slotmem_n_slots, d, cfg.slotmem_iters)
        self.out = nn.Linear(d, cfg.d_llama)
        print(f"[slotattn] control: M={cfg.slotmem_n_slots} free slots, d_slot={d}, iters={cfg.slotmem_iters}")

    def _form(self, hiddens, mask):
        obs = self.obs_proj(hiddens.float())
        slots = self.slots(obs, mask)
        return self.out(slots)


# ── Exp 1: each slot = sparse (entmax) combination of a learned node bank ─────────
class VocabSlotEncoder(_SlotMemBase):
    def _build(self, cfg):
        d = cfg.slotmem_vocab_d_slot; N = cfg.slotmem_vocab_n
        self.obs_proj = nn.Linear(cfg.d_llama, d)
        self.slots = _MaskedSlotAttn(cfg.slotmem_n_slots, d, cfg.slotmem_iters)
        self.node_bank = nn.Parameter(torch.randn(N, d) / math.sqrt(d))
        self.bank_key = nn.Linear(d, d, bias=False)
        self.alpha = cfg.slotmem_vocab_entmax
        self.out = nn.Linear(d, cfg.d_llama)
        print(f"[vocabslot] Exp1: M={cfg.slotmem_n_slots} slots = entmax-{self.alpha} combo of "
              f"N={N} bank, d_slot={d}")

    def _form(self, hiddens, mask):
        obs = self.obs_proj(hiddens.float())
        slots = self.slots(obs, mask)                                     # [B,M,d] queries
        qn = _rmsnorm(slots); kn = _rmsnorm(self.bank_key(self.node_bank))   # [B,M,d],[N,d]
        scores = (qn @ kn.t()) * (slots.shape[-1] ** -0.5)               # [B,M,N]
        w = entmax(scores, alpha=self.alpha, dim=-1) if self.alpha > 1.0 else scores.softmax(-1)
        slot_v = w @ self.node_bank                                       # sparse combo of vocab
        return self.out(slot_v)


# ── Exp 2b: free-endpoint TokenGT write (no bank), snapshot → E edge tokens ──────
class _FreeGraphParser(nn.Module):
    """GraphParser write machinery with FREE-regressed endpoints (no node bank, no
    selection, no node cross-attention). 2-part working set + predict-off-Part-2 kept."""
    def __init__(self, cfg):
        super().__init__()
        d, E = cfg.slotmem_d_graph, cfg.slotmem_n_edges
        self.E, self.d = E, d
        self.obs_proj = nn.Linear(cfg.d_llama, d)
        self.role = nn.Parameter(torch.randn(3, d) / math.sqrt(d))
        self.tag = nn.Parameter(torch.randn(E, d) / math.sqrt(d))
        self.part = nn.Parameter(torch.randn(2, d) / math.sqrt(d))
        self.init_graph = nn.Parameter(torch.randn(3, E, d) / math.sqrt(d))
        self.blocks = nn.ModuleList(
            _ParserBlock(d, cfg.slotmem_heads, cfg.slotmem_ffn_mult, use_nodes=False)
            for _ in range(cfg.slotmem_write_layers))
        self.src_head = nn.Linear(d, d); self.dst_head = nn.Linear(d, d); self.edge_head = nn.Linear(d, d)

    def forward(self, obs, obs_mask):
        B, E, d = obs.shape[0], self.E, self.d
        obs_kv = self.obs_proj(obs.float())
        rt = self.role[:, None, :] + self.tag[None, :, :]                 # [3,E,d]
        p1 = (self.init_graph + rt + self.part[0]).reshape(3 * E, d).unsqueeze(0).expand(B, 3 * E, d).contiguous()
        p2 = (rt + self.part[1]).reshape(3 * E, d).unsqueeze(0).expand(B, 3 * E, d).contiguous()
        x = torch.cat([p1, p2], dim=1)
        for blk in self.blocks:
            x = blk(x, None, obs_kv, obs_mask)                            # nodes=None → no bank cross-attn
        src_t, edge_t, dst_t = x[:, 3 * E:].view(B, 3, E, d).unbind(1)
        return self.src_head(src_t), self.edge_head(edge_t), self.dst_head(dst_t)   # free endpoints


class FreeGraphEncoder(_SlotMemBase):
    def _build(self, cfg):
        d = cfg.slotmem_d_graph
        self.parser = _FreeGraphParser(cfg)
        self.edge_proj = nn.Linear(3 * d, cfg.d_llama)                    # [src;edge;dst] → one token
        print(f"[freegraph] Exp2b: E={cfg.slotmem_n_edges} edges (free endpoints), d_graph={d}, "
              f"write×{cfg.slotmem_write_layers}; M={cfg.slotmem_n_edges} edge tokens")

    def _form(self, hiddens, mask):
        src, edge, dst = self.parser(hiddens, mask)                      # [B,E,d] each (free)
        tok = torch.cat([src, edge, dst], dim=-1)                        # [B,E,3d]
        return self.edge_proj(tok)                                       # [B,E,d_llama]
