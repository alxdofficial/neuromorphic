"""Graph substrate v4: free-target proposals + per-slot learned gate.

Per-window protocol:
  1. GraphUpdater emits K_max (src, dst, state) proposals via cross-attention
     to the input pins. Proposals are FREE TARGETS (not residuals from old) —
     each is an absolute position in node-space, not a perturbation of slot k's
     current state. This breaks the v3 "blend with most-similar neighbor"
     contraction that collapsed the substrate to a 1D ridge.
  2. Per-slot routing: each slot k picks ONE proposal j* (no diagonal self-mask).
     Self-pick is now meaningful: "the updater's recommendation for me is the
     best target for me — no cross-slot merge this window." Affinity is the
     average cosine over slot k's two endpoints vs proposal j's two endpoints.
  3. Learned per-slot gate g_k ∈ [0,1]: an MLP looking at (state_old, picked_state,
     pick_affinity, u) outputs how much this slot should update this window.
     Replaces v3's `alpha = sigmoid(scale·(cos - margin))` (which conflated
     "merge confidence" with "update magnitude") AND v3's `recycle_dead_slots`
     (which is just g_k ≈ 1 triggered by hand-tuned thresholds).
       g_k ≈ 0 → anchor: preserve, no movement
       g_k ≈ 0.3 → load-bearer: partial merge of new info
       g_k ≈ 1 → jumped ship: full overwrite (== recycle, learned)
     Init bias makes g_k ≈ 0.05 at start (anchor-by-default), so the model has
     to learn when to unlock updates rather than rate-limiting churn after the
     fact.
  4. Saliency u: EMA of per-slot pick popularity (how many other slots picked
     this slot's proposal). Now informational only — feeds the gate MLP — no
     longer triggers eviction. Kept for diagnostics + future use.

Read (handled in encoder.GraphReadout): directional W_src / W_dst transforms
(R-GCN-style) then a cross-edge message-passing block, then projection to
d_llama. Saliency gate multiplies each token's contribution.

See docs/exp1_graph_baseline.md for full design + citations.
"""
from __future__ import annotations

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
    grace_windows: int = 4,
) -> dict:
    """Initial state for expert-choice graph.

    Slot state:
      src, dst, state — same as before, from learned init parameters.
      u   — EMA of pick-affinity (init 1.0 so first windows have max-priority).
      age — windows since last overwrite (init = grace_windows so initial slots
            are immediately eligible after grace expires).
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

    # u init 0.5 = neutral midpoint (was 1.0 — v3 needed it as grace protection
    # against eviction; v4 has no eviction so the inflated init no longer serves
    # a purpose and just biases the gate/readout toward "all slots important").
    u = torch.full((B, K_max), 0.5, device=device, dtype=dtype)
    age = torch.full(
        (B, K_max), float(grace_windows), device=device, dtype=torch.float32,
    )
    return {
        "src": src,
        "dst": dst,
        "state": state,
        "u": u,
        "age": age,
    }


# ─────────────────────────────────────────────────────────────────────────
# Per-slot routing (v4) — picks ONE proposal per slot, no self-mask
# ─────────────────────────────────────────────────────────────────────────

def slot_routing(
    edges_old: dict,                  # 'src'/'dst' [B, K, d_node]
    proposals: dict,                  # 'src'/'dst' [B, K, d_node]
) -> tuple[Tensor, Tensor, Tensor]:
    """Each slot k picks ONE proposal j* by combined cosine affinity over both
    endpoints. Self-pick is allowed and meaningful (the updater's recommendation
    for slot k may itself be the right target — no cross-slot merge needed).

    Affinity score:
        a[k, j] = 0.5 · (cos(old_src_k, proposed_src_j) + cos(old_dst_k, proposed_dst_j))

    Returns:
        picked_idx       : [B, K] int — which proposal each slot picked
        pick_affinity    : [B, K] in [-1, 1] — combined affinity of the pick
        pick_count       : [B, K] int — how many slots picked each proposal
                           (column reduction; feeds u EMA + gate MLP)
    """
    src_old = F.normalize(edges_old["src"], dim=-1, eps=1e-6)         # [B, K, d]
    dst_old = F.normalize(edges_old["dst"], dim=-1, eps=1e-6)
    src_prop = F.normalize(proposals["src"], dim=-1, eps=1e-6)        # [B, K, d]
    dst_prop = F.normalize(proposals["dst"], dim=-1, eps=1e-6)

    aff_src = src_old @ src_prop.transpose(-1, -2)                    # [B, K, K]
    aff_dst = dst_old @ dst_prop.transpose(-1, -2)
    affinity = 0.5 * (aff_src + aff_dst)                              # [B, K, K]

    pick_affinity, picked_idx = affinity.max(dim=-1)                  # [B, K]

    K = picked_idx.shape[-1]
    one_hot = F.one_hot(picked_idx, num_classes=K).to(affinity.dtype) # [B, K(slot), K(prop)]
    pick_count = one_hot.sum(dim=1)                                   # [B, K(prop)]
    return picked_idx, pick_affinity, pick_count


def gather_picked_per_slot(field: Tensor, picked_idx: Tensor) -> Tensor:
    """Fetch field-value of each slot's picked proposal. Field [B, K, D] → [B, K, D]."""
    D = field.shape[-1]
    idx = picked_idx.unsqueeze(-1).expand(-1, -1, D)
    return field.gather(dim=1, index=idx)


# ─────────────────────────────────────────────────────────────────────────
# Per-slot update gate (v4) — learned, replaces alpha + recycle
# ─────────────────────────────────────────────────────────────────────────

class GraphGate(nn.Module):
    """Per-slot update gate g_k ∈ [0,1] computed from semantic + geometric context.

    Inputs per slot:
      - state_old_k       [d_state]   "what info am I currently carrying"
      - picked_state_k    [d_state]   "what info would I become"
      - pick_affinity_k   [1]         "how confident am I in this merge target"
      - u_old_k           [1]         "is my current content popular"
      - cos_src_k         [1]         "would src endpoint move much?"  (1=no move, -1=opposite)
      - cos_dst_k         [1]         "would dst endpoint move much?"
      - self_pick_k       [1]         "did I pick my own proposal? (1=yes,0=no)"

    Output: g_k ∈ [0,1], anchor-default biased so g_init ≈ sigmoid(-init_bias).

    The 3 extra geometric features (cos_src, cos_dst, self_pick) let the gate
    see WHAT it's about to change, not just the semantic state info. A gate
    that wants to anchor can choose g≈0 when cos_src/dst are high (small move)
    AND when state would barely change; can choose g≈1 when both endpoints are
    about to swing far AND state is novel.

    The init_bias makes the model stubborn-by-default; gradient from downstream
    loss has to push the MLP to UNLOCK updates for specific slots/contexts.
    """

    def __init__(self, d_state: int, hidden: int = 64, init_bias: float = 1.0):
        super().__init__()
        in_dim = 2 * d_state + 5         # state_old + picked_state + 5 scalars
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Output-layer bias init: negative → sigmoid(neg) ≈ small → anchor-leaning.
        nn.init.normal_(self.net[2].weight, std=0.01)
        nn.init.constant_(self.net[2].bias, -float(init_bias))

    def forward(
        self,
        state_old: Tensor,              # [B, K, d_state]
        picked_state: Tensor,           # [B, K, d_state]
        pick_affinity: Tensor,          # [B, K]
        u_old: Tensor,                  # [B, K]
        cos_src: Tensor,                # [B, K]
        cos_dst: Tensor,                # [B, K]
        self_pick: Tensor,              # [B, K] in {0, 1}
    ) -> Tensor:
        """Returns g ∈ [0,1] per slot, shape [B, K]."""
        inp = torch.cat([
            state_old, picked_state,
            pick_affinity.unsqueeze(-1), u_old.unsqueeze(-1),
            cos_src.unsqueeze(-1), cos_dst.unsqueeze(-1),
            self_pick.unsqueeze(-1),
        ], dim=-1)                                                    # [B, K, in_dim]
        logit = self.net(inp).squeeze(-1)                              # [B, K]
        return torch.sigmoid(logit)


# ─────────────────────────────────────────────────────────────────────────
# Saliency update (derived from observable signals)
# ─────────────────────────────────────────────────────────────────────────

def update_pick_affinity_ema(
    u_old: Tensor,                     # [B, K]
    adopted_popularity: Tensor,        # [B, K] — popularity of THIS slot's adopted proposal
    decay: float = 0.5,
) -> Tensor:
    """Per-slot u = EMA of the popularity of the proposal THIS slot adopted.

    v4 semantic fix: after slot k adopts proposal j, u[k] should reflect the
    popularity of proposal j (the content slot k now carries), not the
    popularity of proposal k (slot k's own pitched proposal that was probably
    NOT what slot k ended up with). Caller computes
    `adopted_popularity = pop.gather(1, picked_idx)` before passing in.

    Default decay 0.5 (was 0.95): for 4-window chunks, 0.5 gives
    u_w4 ≈ 0.0625·u_init + 0.94·recent_pop → real adoption-popularity tracking.
    0.95 left u_w4 ≈ 0.81·u_init + 0.19·recent_pop → init-dominated, useless.

    Normalization pick_count → popularity in [0, 1) via pc/(pc+1):
      pc=0 (orphan)   → pop=0
      pc=1 (typical)  → pop=0.5
      pc=2            → pop=0.67
      pc=5 (very hot) → pop=0.83
    """
    return decay * u_old + (1.0 - decay) * adopted_popularity.to(u_old.dtype)


# ─────────────────────────────────────────────────────────────────────────
# Transformer updater — emits FREE-TARGET proposals (v4: not residuals)
# ─────────────────────────────────────────────────────────────────────────

class GraphUpdater(nn.Module):
    """Cross-attention updater. Takes encoded pins + current edges, outputs
    per-slot proposals (src, dst, state) as FREE TARGETS — absolute positions
    in node-space, not perturbations of the slot's current state.

    Architecture:
      - Encode current edges as K_max tokens: concat(src, dst, state) → MLP → d
      - NO slot_pos — slots distinguished by content alone
      - n_layers of (cross-attn to pins, self-attn among slots, FFN)
      - Output head: per-slot (proposed_src, proposed_dst, proposed_state)
                     as absolute targets. NOT residuals (v3 → v4 change).
                     Init std=0.1 + zero bias → proposals start as small random
                     vectors (diverse across slots), so initial routing is
                     non-degenerate even before training.
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

        edge_in_dim = 2 * d_node + d_state                         # no saliency field
        self.edge_in_proj = nn.Linear(edge_in_dim, d)

        # Fix A: NO slot_pos parameter. Slot identity must come from content.

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

        # Output head: free-target proposals (NOT residuals — v4).
        # Init std=0.1 produces small random vectors per slot, distinct across
        # slots → non-degenerate cosine routing from step 0. Old v3 used std=0.01
        # because outputs were residuals; v4 outputs are absolute, so we need
        # enough magnitude variance that they don't all read as ~0 vectors.
        out_dim = d_node + d_node + d_state
        self.out_norm = nn.LayerNorm(d)
        self.out_head = nn.Linear(d, out_dim)
        nn.init.normal_(self.out_head.weight, std=0.1)
        nn.init.zeros_(self.out_head.bias)

    def forward(
        self,
        pins: Tensor,                          # [B, N, d]
        edges_old: dict,
        pins_pad_mask: Optional[Tensor] = None,  # [B, N] True = padded
    ) -> dict:
        """Returns proposals dict with keys 'src', 'dst', 'state',
        each [B, K_max, d_node|d_state]."""
        old_concat = torch.cat([
            edges_old["src"], edges_old["dst"], edges_old["state"],
        ], dim=-1)
        edge_tok = self.edge_in_proj(old_concat)                    # [B, K, d]
        # No slot_pos addition (fix A).

        for L in range(len(self.cross_attns)):
            q = self.cross_norms[L](edge_tok)
            attn_out, _ = self.cross_attns[L](
                q, pins, pins, key_padding_mask=pins_pad_mask,
            )
            edge_tok = edge_tok + attn_out

            q = self.self_norms[L](edge_tok)
            attn_out, _ = self.self_attns[L](q, q, q)
            edge_tok = edge_tok + attn_out

            edge_tok = edge_tok + self.ffns[L](self.ffn_norms[L](edge_tok))

        raw = self.out_head(self.out_norm(edge_tok))                # [B, K, out_dim]
        d_node, d_state = self.d_node, self.d_state
        # Free-target proposals (v4): output IS the proposal, not a delta.
        # No structural anchoring to edges_old; the model decides per slot
        # whether to propose something near old (then routing self-picks +
        # gate stays low = stubborn anchor mode) or something fresh.
        return {
            "src":   raw[..., :d_node],
            "dst":   raw[..., d_node:2 * d_node],
            "state": raw[..., 2 * d_node:2 * d_node + d_state],
        }
