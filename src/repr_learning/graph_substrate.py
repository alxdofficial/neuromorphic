"""Graph substrate: expert-choice routing + derived saliency.

Per-window protocol:
  1. GraphUpdater emits K_max (src, dst, state) proposals via cross-attention
     to the input pins. NO gates, NO saliency_delta — just the proposals.
  2. Expert-choice routing: every existing endpoint (2·K_max of them, src+dst
     across all slots) picks the proposal with which it has highest cosine
     affinity. Each endpoint then updates toward its pick, with strength
     gated by the affinity score (high score = full overwrite, low = no-op).
     Reuse falls out of geometry: similar existing endpoints pick the same
     proposal and converge, like k-means cluster centers.
  3. Saliency = EMA of POPULARITY (no learned scalar). Popularity =
     pick_count / (pick_count + 1) ∈ [0, 1), where pick_count is how many
     endpoints picked this slot's proposed endpoints (column reduction of
     the routing matrix). A slot whose proposals are repeatedly picked has
     high saliency; one whose proposals are ignored decays toward zero.
     Used to gate readout and to identify eviction victims when novel
     proposals arrive. Was previously wired to update_alpha (selectivity,
     a row property of the picker) — fixed.
  4. Eviction/creation: top-N most novel proposals and bottom-N deadest
     eligible slots are paired; each pair is admitted iff `novelty > u`
     (the proposal is more novel than the slot is alive). Fully
     comparative — no absolute thresholds. Cap N = ⌈5% × K_max⌉ ≈ 3.

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

    # u init 1.0 = max priority. Combined with age >= grace at init, all initial
    # slots are protected for one full grace cycle, then compete on merit.
    u = torch.ones(B, K_max, device=device, dtype=dtype)
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
# Expert-choice routing (B)
# ─────────────────────────────────────────────────────────────────────────

def expert_choice_routing(
    endpoints: Tensor,                # [B, 2K, d_node] — existing endpoints (src then dst stacked)
    proposed_endpoints: Tensor,       # [B, 2K, d_node] — fresh proposals (proposed_src then proposed_dst)
    strength_scale: float = 8.0,
    margin: Optional[float] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Each existing endpoint picks its top-1 NON-SELF proposal by cosine affinity.

    CRITICAL — self-mask. Proposals are emitted as residuals from old slot content
    (proposed_endpoint[i] = old_endpoint[i] + Δ from a small-init head). Without
    masking, endpoint i would always pick proposal i (its own near-identical
    residual), defeating the entire expert-choice clustering mechanism. We mask
    the diagonal so each endpoint must choose among OTHER slots' proposals.

    Update-strength margin. Without a margin, cosine=0 (random) maps to α=0.5
    (large update toward noise). We use sigmoid(scale · (cos - margin)) with
    margin = sqrt(2·log(2K)/d) ≈ 0.28 for our config — the expected nearest-of-N
    random cosine. So α stays near 0 unless the pick is clearly above noise.

    Novelty signal — pick-count based.
      novelty[p] = 1 / (1 + pick_count[p])
        where pick_count[p] = number of endpoints whose argmax landed on p.
      A proposal nobody picked → novelty = 1.0 (orphan, admit candidate).
      A proposal popular with many endpoints → novelty → 0 (already covered).
    This is a COLUMN reduction of the affinity matrix (vs u's ROW reduction),
    structurally independent of u so the admission gate `novelty > u` can fire.
    Same precedent as VQ-VAE codebook dead-code revival (EMA-usage thresholding).
    Earlier draft used `1 - max_affinity` (row reduction), which was coupled
    with u and made admission structurally impossible — bug fixed here.

    Returns:
        picked_idx   : [B, 2K] int — which proposal each endpoint picked (≠ self)
        update_alpha : [B, 2K] in [0,1] — soft update strength
        novelty      : [B, 2K] in (0, 1] — per-proposal novelty for admission
        pick_count   : [B, 2K] int — # endpoints that picked each proposal
                       (column reduction of the picks). Decoupled from update_alpha
                       so u (popularity) and admission can use independent signals.
    """
    import math
    B, twoK, d = endpoints.shape

    e_norm = F.normalize(endpoints, dim=-1, eps=1e-6)            # [B, 2K, d]
    p_norm = F.normalize(proposed_endpoints, dim=-1, eps=1e-6)   # [B, 2K, d]
    affinity = e_norm @ p_norm.transpose(-1, -2)                  # [B, 2K (endpoint), 2K (proposal)]

    # Self-mask diagonal — endpoint i CANNOT pick proposal i (its own residual)
    diag_mask = torch.eye(twoK, dtype=torch.bool, device=affinity.device)
    affinity = affinity.masked_fill(diag_mask.unsqueeze(0), float("-inf"))

    # Pick top-1 non-self proposal per endpoint
    pick_strength, picked_idx = affinity.max(dim=-1)              # [B, 2K]

    # Update strength = sigmoid(scale · (cos - margin)).
    if margin is None:
        margin = math.sqrt(2.0 * math.log(max(twoK, 2)) / max(d, 1))
    update_alpha = torch.sigmoid(strength_scale * (pick_strength - margin))

    # Pick count: # endpoints whose argmax landed on each proposal
    # one_hot[B, endpoint, proposal] = 1 iff this endpoint picked this proposal
    one_hot = F.one_hot(picked_idx, num_classes=twoK).to(affinity.dtype)
    pick_count = one_hot.sum(dim=1)                               # [B, 2K proposal]
    novelty = 1.0 / (1.0 + pick_count)                            # in (0, 1]

    return picked_idx, update_alpha, novelty, pick_count


def gather_picked_proposals(
    proposed_endpoints: Tensor,       # [B, 2K, d_node]
    picked_idx: Tensor,                # [B, 2K] int
) -> Tensor:
    """For each endpoint, fetch the proposal it picked. Returns [B, 2K, d_node]."""
    B, twoK, d = proposed_endpoints.shape
    idx = picked_idx.unsqueeze(-1).expand(-1, -1, d)              # [B, 2K, d]
    return proposed_endpoints.gather(dim=1, index=idx)            # [B, 2K, d]


# ─────────────────────────────────────────────────────────────────────────
# Saliency update (derived from observable signals)
# ─────────────────────────────────────────────────────────────────────────

def update_pick_affinity_ema(
    u_old: Tensor,                     # [B, K_max]
    pick_count: Tensor,                # [B, 2K] — per-proposal popularity (column sum of picks)
    decay: float = 0.95,
) -> Tensor:
    """Per-slot u = EMA of per-slot POPULARITY (how often others picked this slot).

    Architecture P1 fix (was buggy): u previously took update_alpha, which is
    "how strongly THIS slot's endpoints picked OTHER proposals" — that's
    SELECTIVITY (a property of the picker), not popularity (a property of the
    picked). The result: a slot that picked aggressively but was never picked
    BACK would read as alive, while an unpicked-but-passive slot would die.

    The correct signal is pick_count (column reduction): how many endpoints
    chose this slot's proposal endpoints. Same intuition as DNC usage tracking
    and VQ-VAE codebook EMA-usage — survive if you're being USED, not if
    you're using.

    Normalization: pick_count → popularity in [0, 1) via pc/(pc+1):
      pc=0 (orphan)   → pop=0     (eviction candidate)
      pc=1 (typical)  → pop=0.5
      pc=2            → pop=0.67
      pc=5 (very hot) → pop=0.83
    This matches u's prior [0, 1] range so existing thresholds still make sense.

    Slot k's two proposal endpoints (positions k and K+k in the proposal bank)
    each have their own pick_count. We use the max so a slot stays alive if
    EITHER side is being picked — biases toward retention, consistent with
    using max alpha previously.
    """
    K = u_old.shape[1]
    pc_src = pick_count[:, :K]                                     # [B, K]
    pc_dst = pick_count[:, K:]                                     # [B, K]
    per_slot_pc = torch.maximum(pc_src, pc_dst)                    # [B, K]
    popularity = per_slot_pc / (per_slot_pc + 1.0)                 # [B, K], in [0, 1)
    return decay * u_old + (1.0 - decay) * popularity.to(u_old.dtype)


# ─────────────────────────────────────────────────────────────────────────
# Slot recycling (eviction + creation)
# ─────────────────────────────────────────────────────────────────────────

def recycle_dead_slots(
    edges: dict,
    proposals: dict,                   # dict with src/dst/state, all [B, K, *]
    novelty: Tensor,                   # [B, 2K] — pick-count-based novelty from expert_choice_routing
    grace_windows: int = 4,
    max_overwrites_fraction: float = 0.05,
) -> tuple[dict, Tensor]:
    """Fully percentile-based top-N variable-count overwrite.

    Pipeline:
      1. Compute per-proposal novelty = 1 - mean(max_affinity over src/dst).
      2. Take the top-N most novel proposals as CANDIDATES (N=ceil(frac·K)).
      3. Take the bottom-N most dead eligible (past-grace) slots as CANDIDATE victims.
      4. Pair them in order (most-novel proposal ↔ deadest slot).
      5. ADMIT each pair only if novelty > u — i.e., the proposal is more
         novel than the slot is alive. Pure value comparison, no absolute
         threshold. Both quantities are in roughly [0, 1]; tie defaults to
         keeping memory (bias toward stability).

    Why fully comparative (no absolute floors):
      The earlier draft had `novelty > 0.5 AND u < 0.05` as admission gates.
      Both numbers were defensible but arbitrary. `novelty > u` is purely
      structural — "the proposal is worth more in attention than what's
      already there." No magic numbers; scales naturally with whatever u/novelty
      distributions the model produces.

    Cold-start behavior is automatic:
      In the first window, all slots have u=1.0 (PER max-priority init).
      No novelty value can exceed 1.0, so zero overwrites until u values
      begin to decay. Grace period (age >= grace_windows) is kept as a
      structural integer-level guard, independent of the value comparison.

    Why not one-per-window:
      Too conservative for dense text — a window introducing 5 new entities
      could only admit one. 5% of K_max (≈3 for K_max=68) matches typical
      cache eviction batch sizes and biological consolidation rates.

    Returns: (new_edges_dict, overwrite_mask[B, K] bool — True where overwrites landed)
    """
    B, K, _ = edges["src"].shape
    device = edges["src"].device
    N = max(1, int(max_overwrites_fraction * K))                    # eviction cap

    # 1. Per-slot proposal novelty = max(novelty_src, novelty_dst).
    # Each "proposal slot k" emits both a proposed src (index k in the bank)
    # and a proposed dst (index K+k). If EITHER side is orphaned (nobody
    # picked it), the proposal as a whole represents new content — admit.
    # Using max (not mean) means we don't lose half-novel proposals just
    # because the other side fits an existing cluster.
    novelty_src = novelty[:, :K]                                     # [B, K]
    novelty_dst = novelty[:, K:]
    per_slot_novelty = torch.maximum(novelty_src, novelty_dst)       # [B, K]
    novelty = per_slot_novelty  # rename for clarity downstream

    # 2. Top-N most novel proposals (regardless of absolute novelty).
    top_novelty, top_prop_idx = novelty.topk(N, dim=-1)             # [B, N]

    # 3. Bottom-N eligible slots by u (most-dead first). Mask grace-protected
    # slots to +inf so they can't be selected.
    eligible = edges["age"] >= grace_windows                        # [B, K] bool
    u_masked = torch.where(
        eligible, edges["u"], torch.full_like(edges["u"], float("inf")),
    )
    bot_u_neg, victim_idx = (-u_masked).topk(N, dim=-1)             # negative trick: top-N of -u = bottom-N of u
    bot_u = -bot_u_neg                                              # [B, N] actual u values

    # 4. Per-pair admission — fully comparative. A pair is admitted iff the
    # proposal's novelty is at least the slot's aliveness. Both quantities are
    # roughly in [0, 1]; the criterion is "the new content is worth at least
    # as much in attention as what's already there."
    #
    # `>=` (not `>`) is intentional: at cold start, all slots have u=1.0 and
    # orphan proposals have novelty=1.0; under `>` cold-start admission never
    # fires for that exact tie. With `>=` an orphan can claim a fresh slot
    # immediately. In steady state ties are essentially zero-measure (continuous
    # popularity values), so the "stability bias" of `>` was decorative — the
    # real safety guard is the grace_windows + max_overwrites_fraction cap.
    pair_admit = top_novelty >= bot_u.to(top_novelty.dtype)
    # [B, N]

    # 5. Apply overwrites. Loop over the N pair positions (small, vectorized).
    new_edges = {k: v.clone() for k, v in edges.items()}
    overwrite_mask = torch.zeros(B, K, dtype=torch.bool, device=device)

    # Pre-compute max u per row for PER-style cold-start seeding (consistent
    # across all overwrites in this window; doesn't get re-computed after each).
    u_max_per_row = edges["u"].max(dim=-1).values                   # [B]

    for n in range(N):
        rows_admit = pair_admit[:, n]                               # [B] bool
        if not rows_admit.any():
            continue
        active_rows = torch.where(rows_admit)[0]
        prop_idx = top_prop_idx[active_rows, n]                     # [N_active]
        slot_idx = victim_idx[active_rows, n]                       # [N_active]
        new_edges["src"][active_rows, slot_idx] = proposals["src"][active_rows, prop_idx]
        new_edges["dst"][active_rows, slot_idx] = proposals["dst"][active_rows, prop_idx]
        new_edges["state"][active_rows, slot_idx] = proposals["state"][active_rows, prop_idx]
        new_edges["age"][active_rows, slot_idx] = 0.0
        new_edges["u"][active_rows, slot_idx] = u_max_per_row[active_rows].to(new_edges["u"].dtype)
        overwrite_mask[active_rows, slot_idx] = True

    return new_edges, overwrite_mask


# ─────────────────────────────────────────────────────────────────────────
# Transformer updater — emits raw proposals only, no gates
# ─────────────────────────────────────────────────────────────────────────

class GraphUpdater(nn.Module):
    """Cross-attention updater. Takes encoded pins + current edges, outputs
    per-slot raw proposals (src, dst, state). No snap_gate, no keep_gate, no
    saliency_delta — those concepts are subsumed by expert-choice routing
    (snap=affinity, keep=alpha-gating, saliency=usage EMA).

    Architecture (unchanged from v1/v2 except output head):
      - Encode current edges as K_max tokens: concat(src, dst, state) → MLP → d
      - NO slot_pos (fix A) — slots distinguished by content alone
      - n_layers of (cross-attn to pins, self-attn among slots, FFN)
      - Output head: per-slot (proposed_src, proposed_dst, proposed_state),
                     applied as residuals on the encoded-old.
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

        # Output head: just the three residual deltas
        out_dim = d_node + d_node + d_state
        self.out_norm = nn.LayerNorm(d)
        self.out_head = nn.Linear(d, out_dim)
        nn.init.normal_(self.out_head.weight, std=0.01)
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
        delta_src = raw[..., :d_node]
        delta_dst = raw[..., d_node:2 * d_node]
        delta_state = raw[..., 2 * d_node:2 * d_node + d_state]

        return {
            "src":   edges_old["src"]   + delta_src,
            "dst":   edges_old["dst"]   + delta_dst,
            "state": edges_old["state"] + delta_state,
        }
