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
) -> dict:
    """Initial graph: K_max random edges with low initial saliency.

    All edges start with random src/dst/state and saliency ≈ 0.5 (sigmoid
    of zero — agnostic about which edges are important until training
    teaches the updater).
    """
    src = torch.randn(B, K_max, d_node, device=device, dtype=dtype) * (d_node ** -0.5)
    dst = torch.randn(B, K_max, d_node, device=device, dtype=dtype) * (d_node ** -0.5)
    state = torch.randn(B, K_max, d_state, device=device, dtype=dtype) * (d_state ** -0.5)
    saliency_logit = torch.zeros(B, K_max, device=device, dtype=dtype)
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
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Soft snap proposed endpoint to nearest existing endpoint.

    output = snap_gate · (attention over endpoint_bank by similarity to predicted)
           + (1 − snap_gate) · predicted_v

    When snap_gate ≈ 1 and the predicted vector is close to some existing
    endpoint, the output IS that endpoint (graph stays connected). When
    snap_gate ≈ 0, the output is the raw proposal (a new node is born).

    Returns: (out_v, max_sim)
        out_v  : [B, K_max, d_node] resolved endpoint vector
        max_sim: [B, K_max] cosine similarity to the *nearest* existing
                            endpoint, used by L_connectivity to penalize
                            "low snap when something close exists"
    """
    pred_norm = F.normalize(predicted_v, dim=-1, eps=1e-6)             # [B, K_max, d]
    bank_norm = F.normalize(endpoint_bank, dim=-1, eps=1e-6)           # [B, 2K, d]
    sim = pred_norm @ bank_norm.transpose(-1, -2)                      # [B, K_max, 2K]

    attn = F.softmax(sim / temperature, dim=-1)                        # [B, K_max, 2K]
    attended = attn @ endpoint_bank                                    # [B, K_max, d]

    sg = snap_gate.unsqueeze(-1)                                       # [B, K_max, 1]
    out_v = sg * attended + (1.0 - sg) * predicted_v                   # [B, K_max, d]

    max_sim = sim.max(dim=-1).values                                   # [B, K_max]
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

        # Bias keep_gate's pre-sigmoid input so init ≈ "preserve old" (gate ≈ 0.7)
        # Index of keep_gate in the output dim:
        self._keep_idx = d_node + d_node + d_state + 2
        with torch.no_grad():
            # bias[keep_idx] = logit(0.7) ≈ 0.85, biases initial gate toward "keep"
            self.out_head.bias[self._keep_idx] = 0.85

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
