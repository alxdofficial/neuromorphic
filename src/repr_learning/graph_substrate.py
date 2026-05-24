"""Exp 1: graph substrate — bounded budget of continuous edges.

Memory is K_max edges, each (src, dst, state) ∈ R^d_node × R^d_node × R^d_state.
All endpoints and states are continuous (no codebook constraint — relaxes V2.1's
4096-node quantization). Nodes are implicit in edge endpoints; reusing an
endpoint vector across edges = sharing a node.

Per 1024-token window, a TransformerUpdater reads (encoded pin tokens +
current edge tokens) and outputs for each edge slot:
    - proposed (src, dst, state)
    - keep_gate ∈ [0,1]: preserve old vs overwrite with proposed
    - saliency ∈ [0,1]: per-edge confidence, used in prepend tokens

Connectivity (the distinguishing ingredient): proposed endpoints get
soft-snapped to existing endpoint vectors via attention, with a learned
snap_gate per emission. L_connectivity penalizes "novel but similar to
existing" — encourages a graph with shared nodes rather than 2·K_max
disjoint endpoints.

Read: K_max edges projected to d_llama via fused MLP (concat(src, dst, state)
→ d_llama), prepended to Llama input (same pattern as B/MT, no MemInject hook).

See docs/exp1_graph_baseline.md for full design.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────
# Substrate state
# ─────────────────────────────────────────────────────────────────────────

def init_graph_state(
    B: int, K_max: int, d_node: int, d_state: int,
    device: torch.device, dtype: torch.dtype,
    init_src: Optional[Tensor] = None,        # [K_max, d_node] learned init
    init_dst: Optional[Tensor] = None,
    init_state: Optional[Tensor] = None,
    init_saliency_logit: float = -4.0,         # sigmoid(-4) ≈ 0.018
) -> dict:
    """Initial graph state, deterministic given (B, init_*).

    Caller passes learned init parameters from the encoder (one set per
    K_max slot). These are expanded across the batch — same input must
    produce same memory in eval mode (no per-forward randomness).

    saliency_logit defaults to -4 so new/untrained slots start LOW
    priority (≈0.02). The updater can raise saliency via its delta head
    when it decides a slot is informative; high-saliency edges then
    become "sticky" via L_adjust weighting.
    """
    if init_src is None:
        src = torch.zeros(B, K_max, d_node, device=device, dtype=dtype)
    else:
        src = init_src.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).contiguous()
    if init_dst is None:
        dst = torch.zeros(B, K_max, d_node, device=device, dtype=dtype)
    else:
        dst = init_dst.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).contiguous()
    if init_state is None:
        state = torch.zeros(B, K_max, d_state, device=device, dtype=dtype)
    else:
        state = init_state.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).contiguous()
    saliency_logit = torch.full(
        (B, K_max), float(init_saliency_logit), device=device, dtype=dtype,
    )
    return {
        "src": src,                          # [B, K_max, d_node]
        "dst": dst,                          # [B, K_max, d_node]
        "state": state,                      # [B, K_max, d_state]
        "saliency_logit": saliency_logit,    # [B, K_max]  (sigmoid → [0,1])
    }


# ─────────────────────────────────────────────────────────────────────────
# Soft snap to existing endpoints
# ─────────────────────────────────────────────────────────────────────────

def soft_snap(
    predicted_v: Tensor,         # [B, K_max, d_node]   fresh proposals
    endpoint_bank: Tensor,       # [B, 2*K_max, d_node] all current src+dst
    snap_gate: Tensor,           # [B, K_max] in [0,1]  learned snap coefficient
    *,
    temperature: float = 0.1,
    top_k: int = 4,
    exclude_self: bool = True,
) -> tuple[Tensor, Tensor]:
    """Soft snap proposed endpoint to nearest existing endpoint.

    output = snap_gate · (attention over endpoint_bank by similarity to predicted)
           + (1 − snap_gate) · predicted_v

    With (temperature=0.1, top_k=4) the attention concentrates ~95%+ mass
    on the nearest single endpoint — actually behaves like a snap rather
    than a diffuse average. Top-k masking before softmax bounds gradient
    flow to the K nearest candidates.

    exclude_self: endpoint_bank is conventionally arranged as
        [src(0), src(1), ..., src(K-1), dst(0), dst(1), ..., dst(K-1)]
    Slot i's proposal is a residual from its own old src/dst, so the
    nearest match would always be itself unless we mask out positions
    {i, K+i} per slot. Required for L_connectivity to actually push
    toward sharing across slots, not self-stickiness.

    Returns: (out_v, max_sim)
        out_v  : [B, K_max, d_node] resolved endpoint vector
        max_sim: [B, K_max] cosine similarity to the *nearest* non-self
                            existing endpoint, used by L_connectivity.
    """
    B, K, d = predicted_v.shape
    K2 = endpoint_bank.shape[1]

    pred_norm = F.normalize(predicted_v, dim=-1, eps=1e-6)             # [B, K, d]
    bank_norm = F.normalize(endpoint_bank, dim=-1, eps=1e-6)           # [B, 2K, d]
    sim = pred_norm @ bank_norm.transpose(-1, -2)                      # [B, K, 2K]

    if exclude_self and K2 == 2 * K:
        slot = torch.arange(K, device=sim.device)
        self_mask = torch.zeros(K, K2, dtype=torch.bool, device=sim.device)
        self_mask[slot, slot] = True               # self-src
        self_mask[slot, K + slot] = True           # self-dst
        sim = sim.masked_fill(self_mask.unsqueeze(0), float("-inf"))

    # max_sim is computed AFTER self-mask so L_connectivity sees the
    # nearest *other* endpoint, not the slot's own residual seed.
    max_sim = sim.max(dim=-1).values                                   # [B, K]

    # Top-k attention: zero out non-top-k similarities, then softmax with
    # low temperature for sharp peak. K2 might be < top_k in degenerate
    # configs, so guard the topk call.
    k_eff = min(top_k, K2) if top_k is not None else K2
    if k_eff < K2:
        topk_vals, _ = sim.topk(k_eff, dim=-1)
        threshold = topk_vals[..., -1:].detach()                       # [B, K, 1]
        sim_attn = torch.where(
            sim >= threshold, sim, torch.full_like(sim, float("-inf")),
        )
    else:
        sim_attn = sim

    attn = F.softmax(sim_attn / temperature, dim=-1)                   # [B, K, 2K]
    attended = attn @ endpoint_bank                                    # [B, K, d]

    sg = snap_gate.unsqueeze(-1)                                       # [B, K, 1]
    out_v = sg * attended + (1.0 - sg) * predicted_v                   # [B, K, d]

    return out_v, max_sim


# ─────────────────────────────────────────────────────────────────────────
# Auxiliary losses
# ─────────────────────────────────────────────────────────────────────────

def _row_mean(per_row: Tensor, has_real: Optional[Tensor]) -> Tensor:
    """Mean over batch rows, masking all-padded rows."""
    if has_real is None:
        return per_row.mean()
    mask = has_real.to(per_row.dtype)
    n_real = mask.sum().clamp(min=1.0)
    if per_row.dim() == 1:
        return (per_row * mask).sum() / n_real
    # broadcast mask across trailing dims
    expand = (-1,) + (1,) * (per_row.dim() - 1)
    return (per_row * mask.view(*expand)).sum() / (n_real * per_row[0].numel())


def loss_connectivity(
    snap_gate_src: Tensor,       # [B, K_max] ∈ [0,1]
    snap_gate_dst: Tensor,       # [B, K_max]
    max_sim_src: Tensor,         # [B, K_max]
    max_sim_dst: Tensor,         # [B, K_max]
    has_real: Optional[Tensor] = None,
) -> Tensor:
    """L_connectivity = mean over emitted endpoints of:
                       max(sim_to_nearest, 0) · (1 − snap_gate)

    Penalizes "you generated something near an existing endpoint but didn't
    snap to it." Pushes the model toward graph connectivity — same vector
    reused across multiple edges instead of two-near-clones.
    """
    pen_src = max_sim_src.clamp(min=0.0) * (1.0 - snap_gate_src)       # [B, K_max]
    pen_dst = max_sim_dst.clamp(min=0.0) * (1.0 - snap_gate_dst)       # [B, K_max]
    per_row = (pen_src.mean(dim=-1) + pen_dst.mean(dim=-1)) * 0.5     # [B]
    return _row_mean(per_row, has_real)


def loss_adjust(
    edges_new: dict, edges_old: dict,
    has_real: Optional[Tensor] = None,
) -> Tensor:
    """L_adjust = saliency-weighted L2 change in edge state between old/new.

    Established (high-saliency) edges are stickier — penalized more for
    being modified. Newly-created/low-saliency edges can change freely.
    """
    # Use OLD saliency as the stickiness weight (established knowledge is sticky)
    weight = torch.sigmoid(edges_old["saliency_logit"])                # [B, K_max]
    denom = weight.sum(dim=-1).clamp(min=1e-6)                         # [B]

    delta_src = (edges_new["src"] - edges_old["src"]).pow(2).sum(-1)   # [B, K_max]
    delta_dst = (edges_new["dst"] - edges_old["dst"]).pow(2).sum(-1)
    delta_st = (edges_new["state"] - edges_old["state"]).pow(2).sum(-1)
    delta = delta_src + delta_dst + delta_st                            # [B, K_max]

    per_row = (delta * weight).sum(dim=-1) / denom                     # [B]
    return _row_mean(per_row, has_real)


# ─────────────────────────────────────────────────────────────────────────
# Transformer updater
# ─────────────────────────────────────────────────────────────────────────

class GraphUpdater(nn.Module):
    """Cross-attention updater. Takes encoded pins + current edges, outputs
    per-slot edge proposals + gates + saliency.

    Architecture:
      - Encode current edges as K_max tokens: concat(src, dst, state) → MLP → d
      - K_max learned positional embeddings (per-slot identity)
      - n_layers of (cross-attn to pins, self-attn among slots, FFN)
      - Output head: per-slot (proposed_src, proposed_dst, proposed_state,
                              snap_gate_src, snap_gate_dst,
                              keep_gate, saliency_logit_delta)
    """

    def __init__(
        self,
        d: int,
        K_max: int,
        d_node: int,
        d_state: int,
        n_layers: int = 3,
        n_heads: int = 4,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.d = d
        self.K_max = K_max
        self.d_node = d_node
        self.d_state = d_state

        # Encode old edge params as d-dim tokens
        edge_in_dim = 2 * d_node + d_state + 1   # +1 for saliency_logit
        self.edge_in_proj = nn.Linear(edge_in_dim, d)

        # K_max learned positional embeddings (stable slot identity)
        self.slot_pos = nn.Parameter(torch.randn(K_max, d) * (d ** -0.5))

        # Layer stack
        self.cross_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(d, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.self_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(d, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, ffn_mult * d),
                nn.GELU(),
                nn.Linear(ffn_mult * d, d),
            )
            for _ in range(n_layers)
        ])

        # Output head — per-slot. Small init so first call ≈ identity-ish.
        out_dim = (
            d_node + d_node + d_state    # proposed src, dst, state
            + 2                            # snap_gate_src, snap_gate_dst (sigmoid)
            + 1                            # keep_gate (sigmoid)
            + 1                            # saliency_logit delta
        )
        self.out_norm = nn.LayerNorm(d)
        self.out_head = nn.Linear(d, out_dim)
        nn.init.normal_(self.out_head.weight, std=0.01)
        nn.init.zeros_(self.out_head.bias)

        # Bias keep_gate init to 0.0 → sigmoid(0)=0.5 (neutral). With
        # learned-init edges (audit C1), the init is meaningful but not
        # sacred — first writes should be free to overwrite ~50%. The
        # 0.85 bias used previously biased keep≈0.7 and let random init
        # garbage persist across writes (and we now have non-random init,
        # so the "preserve" bias loses its rationale).
        self._keep_idx = d_node + d_node + d_state + 2
        with torch.no_grad():
            self.out_head.bias[self._keep_idx] = 0.0

    def forward(
        self,
        pins: Tensor,                          # [B, N, d]
        edges_old: dict,
        pins_pad_mask: Optional[Tensor] = None,  # [B, N] True = padded
    ) -> tuple[dict, dict]:
        """Returns (proposed_components, gates_dict).

        proposed_components: dict with raw 'src', 'dst', 'state' before snap
        gates_dict: 'snap_gate_src', 'snap_gate_dst', 'keep_gate', 'saliency_logit_delta'
        """
        B = pins.shape[0]

        # 1. Encode old edge params as token features
        old_concat = torch.cat([
            edges_old["src"],                              # [B, K_max, d_node]
            edges_old["dst"],
            edges_old["state"],
            edges_old["saliency_logit"].unsqueeze(-1),     # [B, K_max, 1]
        ], dim=-1)                                          # [B, K_max, edge_in_dim]
        edge_tok = self.edge_in_proj(old_concat)            # [B, K_max, d]
        edge_tok = edge_tok + self.slot_pos.unsqueeze(0)

        # 2. Layer stack
        for L in range(len(self.cross_attns)):
            q = self.cross_norms[L](edge_tok)
            attn_out, _ = self.cross_attns[L](
                q, pins, pins,
                key_padding_mask=pins_pad_mask,
            )
            edge_tok = edge_tok + attn_out

            q = self.self_norms[L](edge_tok)
            attn_out, _ = self.self_attns[L](q, q, q)
            edge_tok = edge_tok + attn_out

            edge_tok = edge_tok + self.ffns[L](self.ffn_norms[L](edge_tok))

        # 3. Output head
        raw = self.out_head(self.out_norm(edge_tok))        # [B, K_max, out_dim]

        d_node, d_state = self.d_node, self.d_state
        cursor = 0
        proposed_src = raw[..., cursor : cursor + d_node]; cursor += d_node
        proposed_dst = raw[..., cursor : cursor + d_node]; cursor += d_node
        proposed_state = raw[..., cursor : cursor + d_state]; cursor += d_state
        snap_logit_src = raw[..., cursor]; cursor += 1
        snap_logit_dst = raw[..., cursor]; cursor += 1
        keep_logit = raw[..., cursor]; cursor += 1
        saliency_delta = raw[..., cursor]; cursor += 1

        # The proposed deltas are added to the encoded-old quantities so
        # output ≈ old at init (small-init head). For continuous fields
        # this is the residual-style init we used for plastic and splat.
        proposed_src = edges_old["src"] + proposed_src
        proposed_dst = edges_old["dst"] + proposed_dst
        proposed_state = edges_old["state"] + proposed_state

        # New saliency = old + delta, kept as a logit
        new_saliency_logit = edges_old["saliency_logit"] + saliency_delta

        proposed = {
            "src": proposed_src,
            "dst": proposed_dst,
            "state": proposed_state,
            "saliency_logit": new_saliency_logit,
        }
        gates = {
            "snap_gate_src": torch.sigmoid(snap_logit_src),
            "snap_gate_dst": torch.sigmoid(snap_logit_dst),
            "keep_gate": torch.sigmoid(keep_logit),
        }
        return proposed, gates
