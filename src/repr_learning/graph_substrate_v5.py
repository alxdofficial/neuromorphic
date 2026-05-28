"""Graph substrate v5.1: shared node bank + soft-pointer edges + holistic updater.

Per-window pipeline:
  1. HolisticUpdater(pins, N_old, edges_old) → (node_update, edge_proposals).
     One transformer with [K_node node tokens, K_edge edge tokens] doing
     cross-attn(pins) + self-attn(nodes ↔ edges) + FFN. Single forward pass
     produces both bank updates AND edge proposals coordinated.
  2. NodeGate applies the node update to N (per-slot anchor-biased blend).
  3. Materialize ALL edge queries (existing + proposals) against N_new.
  4. slot_routing_on_endpoints: each existing edge picks ONE proposal by
     cosine on materialized endpoints; multiple edges CAN pick the same
     proposal → convergence pressure (v4.2 k-means dynamic).
  5. EdgeGate per-edge anchor-biased blend → new q_src, q_dst, state.
  6. RMSNorm + per-row all-pad protection.

State per chunk:
  N         [B, K_node, d_node]   — shared node bank
  q_src     [B, K_edge, d_node]   — per-edge src query
  q_dst     [B, K_edge, d_node]   — per-edge dst query
  state     [B, K_edge, d_state]  — per-edge state

Initialized at chunk start from learned (μ, σ) sampled fresh per forward
pass. NO per-slot trained params — slot identity is ephemeral (per-chunk only).

Read: materialize endpoints via soft-pointer attention into N, then
directional readout (W_src/W_dst — same N[k] gets projected role-specifically
when multiple edges share a node).

What carries from v4.2:
  - QK-Norm + KV-LN + post-attn-LN (AttnBlock).
  - Sinusoidal PE on pins (handled in encoder).
  - Slot routing pattern (multiple edges can pick same proposal).
  - Anchor-biased gate init.
  - Directional readout.

What's new in v5.1 (vs v4.2):
  - Shared N bank with chunk-fresh (μ, σ) init.
  - Edges store query vectors, materialize endpoints via soft pointer.
  - Routing operates on materialized endpoints under N_new (sees the bank).
  - Holistic updater fuses pin + node + edge information in one pass.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .graph_substrate import AttnBlock  # reuse v4.2 normalized attention block


# ─────────────────────────────────────────────────────────────────────────
# Substrate state — chunk-fresh sample from learned (μ, σ)
# ─────────────────────────────────────────────────────────────────────────

def init_graph_v5_state(
    B: int,
    K_node: int,
    K_edge: int,
    d_node: int,
    d_state: int,
    mu_node: Tensor,           # [d_node] learned mean
    log_sigma_node: Tensor,    # [d_node] learned log-std (param stored as log)
    mu_state: Tensor,          # [d_state] learned mean for edge state
    log_sigma_state: Tensor,   # [d_state] learned log-std for edge state
    mu_q: Tensor,              # [d_node] learned mean for edge query init
    log_sigma_q: Tensor,       # [d_node] learned log-std for edge query init
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,  # for deterministic eval
) -> dict:
    """Chunk-start state. Resampled per forward pass — no cross-chunk persistence.

    Pass a torch.Generator to make noise deterministic (for eval). When None,
    samples from the global RNG state (training default — keeps chunk-fresh
    init acting as a regularizer). Without this, same model + same val batch
    produces ~0.2 loss variance — exceeds inter-variant gaps we care about.

    Returns dict with:
      N         : [B, K_node, d_node]  — shared node bank
      q_src     : [B, K_edge, d_node]  — per-edge src query
      q_dst     : [B, K_edge, d_node]  — per-edge dst query
      state     : [B, K_edge, d_state] — per-edge state
    """
    sigma_node = log_sigma_node.exp().to(dtype)
    sigma_state = log_sigma_state.exp().to(dtype)
    sigma_q = log_sigma_q.exp().to(dtype)

    # Per-pass noise breaks symmetry; learned (μ, σ) controls scale/center.
    eps_n = torch.randn(B, K_node, d_node, device=device, dtype=dtype, generator=generator)
    eps_s = torch.randn(B, K_edge, d_state, device=device, dtype=dtype, generator=generator)
    eps_qs = torch.randn(B, K_edge, d_node, device=device, dtype=dtype, generator=generator)
    eps_qd = torch.randn(B, K_edge, d_node, device=device, dtype=dtype, generator=generator)

    N = mu_node.to(dtype).view(1, 1, -1) + sigma_node.view(1, 1, -1) * eps_n
    state = mu_state.to(dtype).view(1, 1, -1) + sigma_state.view(1, 1, -1) * eps_s
    q_src = mu_q.to(dtype).view(1, 1, -1) + sigma_q.view(1, 1, -1) * eps_qs
    q_dst = mu_q.to(dtype).view(1, 1, -1) + sigma_q.view(1, 1, -1) * eps_qd
    return {"N": N, "q_src": q_src, "q_dst": q_dst, "state": state}


# ─────────────────────────────────────────────────────────────────────────
# Holistic updater (v5.1) — single transformer fuses pins + N + edges
# ─────────────────────────────────────────────────────────────────────────

class HolisticUpdater(nn.Module):
    """One updater that takes pins + current N + current edges, jointly
    processes them, and emits BOTH node updates AND edge proposals.

    Token stream is [K_node node tokens, K_edge edge tokens]. Each layer:
      1. Cross-attn to pins — both nodes and edges see text content.
      2. Self-attn over (nodes ∪ edges) — nodes and edges coordinate:
         nodes see what edges want, edges see what nodes look like.
      3. Position-wise FFN.

    Output heads:
      - Node head: per-node update vector [B, K_node, d_node].
        Goes through a per-slot gate downstream (anchor-biased blend).
      - Edge head: per-edge proposal (q_src, q_dst, state).
        Goes through downstream routing — each existing edge picks one
        proposal by cosine on materialized endpoints (under the NEW bank).

    Node updates apply directly (one-to-one with slots) — they're not
    "proposals" routed by other entities; the model decided what each
    slot should become. Edge proposals ARE routed because existing edges
    have persistent identity and choose which proposal to absorb.
    """

    def __init__(
        self,
        d: int,                        # internal token dim (was d_updater)
        K_node: int,
        K_edge: int,
        d_node: int,
        d_state: int,
        d_pin: int,                    # pin dim — needed for competitive node write
        K_proposal: int = 0,           # 0 = legacy (K_proposal == K_edge, edge tokens ARE the proposals)
        n_layers: int = 3,
        n_heads: int = 16,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.d = d
        self.K_node = K_node
        self.K_edge = K_edge
        self.K_proposal = K_proposal if K_proposal > 0 else K_edge
        self.proposal_separate = K_proposal > 0
        self.d_node = d_node
        self.d_state = d_state
        self.d_pin = d_pin

        # Token projections: nodes have just d_node dim; edges have
        # 2·d_node + d_state. Project each into the shared updater dim d.
        self.node_in_proj = nn.Linear(d_node, d)
        self.edge_in_proj = nn.Linear(2 * d_node + d_state, d)

        # Proposal tokens (separate-proposal mode only). Sampled fresh per
        # forward pass from learned (μ, σ) — same philosophy as N init.
        # Proposals are window-local scratch: not persistent, not decoder-
        # facing, so they don't count toward bottleneck.
        if self.proposal_separate:
            self.mu_proposal = nn.Parameter(torch.zeros(d))
            self.log_sigma_proposal = nn.Parameter(torch.zeros(d))

        # Per-position embeddings for the joint token stream.
        # Stream layout: [K_node node tokens, K_edge edge tokens, K_proposal proposal tokens]
        # In legacy mode (K_proposal == K_edge and not separate), proposals
        # share the edge token positions (edge_out_head produces proposals).
        n_pos = K_node + K_edge + (self.K_proposal if self.proposal_separate else 0)
        # CRITICAL: without these, all tokens are statistically identical
        # samples from the same (μ, σ) distribution and the transformer's
        # cross-attn to pins gives near-identical outputs at every position,
        # which compounds across layers + windows → both N and edge queries
        # collapse to a shared direction (cross-slot cos→0.99 within 4 windows).
        #
        # Init std=1.0 (not standard transformer 0.02!) because the node_tok /
        # edge_tok content magnitudes after node_in_proj/edge_in_proj are ~1
        # per element. A 0.02 pos_emb is dwarfed and doesn't actually
        # differentiate positions through the attention layers. std=1.0 makes
        # the positional signal comparable to the content signal, which is
        # what the transformer needs to keep positions distinguishable.
        #
        # These don't bake in semantic identity — they're just position IDs
        # that let the transformer process distinct positions distinctly. The
        # CONTENT of slot k is still entirely set per-chunk by the (μ, σ)
        # sample + updater dynamics; node 5 in chunk A still has nothing in
        # common with node 5 in chunk B.
        self.pos_emb = nn.Parameter(torch.zeros(n_pos, d))
        nn.init.normal_(self.pos_emb, std=1.0)

        # Joint stack: every layer is (cross-attn-to-pins, self-attn-over-tokens, FFN).
        # AttnBlock is the v4.2 QK-Norm + KV-LN + post-attn-LN building block.
        self.cross_blocks = nn.ModuleList([
            AttnBlock(d, n_heads) for _ in range(n_layers)
        ])
        self.self_blocks = nn.ModuleList([
            AttnBlock(d, n_heads) for _ in range(n_layers)
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

        # Proposal output head: standard projection. Edge specialization is
        # ensured by the downstream slot routing (multiple edges can pick
        # the same proposal → convergence).
        #
        # In legacy mode (proposal_separate=False), this head consumes the
        # edge tokens' output — proposals are derived from existing edge
        # tokens (1:1 mapping). In separate mode, this head consumes the
        # dedicated proposal tokens' output (K_proposal can differ from
        # K_edge), and the edge tokens act as PURE CONTEXT — their output
        # is discarded.
        self.edge_out_norm = nn.LayerNorm(d)
        self.edge_out_head = nn.Linear(d, 2 * d_node + d_state)
        nn.init.normal_(self.edge_out_head.weight, std=0.1)
        nn.init.zeros_(self.edge_out_head.bias)

        # Node output: Slot Attention-style COMPETITIVE WRITE.
        #
        # The transformer above (cross-attn pins + self-attn nodes ↔ edges)
        # gives the model global context — node tokens see what edges want,
        # edges see what nodes look like, both see the text. But on its own
        # the transformer DOES NOT differentiate node positions: standard
        # cross-attn lets every slot attend to all pins, so all slots get
        # similar updates and N collapses to a shared direction within a
        # few windows (probe showed cross-slot cos→0.99 by window 4 without
        # this step).
        #
        # The fix: after the transformer, use node_tok_out as QUERIES into
        # pins with SOFTMAX OVER SLOTS (not over tokens). This is the
        # Locatello 2020 Slot Attention trick — pins compete to be absorbed
        # by slots, which forces specialization. Slots with similar queries
        # split the pin mass; slots with distinct queries claim distinct pin
        # subsets.
        self.node_q_proj = nn.Linear(d, d_node, bias=False)
        self.node_k_proj = nn.Linear(d_pin, d_node, bias=False)
        self.node_v_proj = nn.Linear(d_pin, d_node, bias=False)
        self.node_in_norm = nn.LayerNorm(d)
        self.pin_in_norm = nn.LayerNorm(d_pin)

    def forward(
        self,
        pins: Tensor,                              # [B, T_w, d]
        N_old: Tensor,                             # [B, K_node, d_node]
        edges_old: dict,                           # q_src/q_dst [B, K_edge, d_node], state [B, K_edge, d_state]
        pins_pad_mask: Optional[Tensor] = None,    # [B, T_w] True=padded
    ) -> tuple[Tensor, dict]:
        """Returns (node_update, edge_proposals).

        node_update: [B, K_node, d_node] — additive update applied to N
                     downstream with a per-slot gate.
        edge_proposals: dict with 'q_src', 'q_dst', 'state' — same shape
                        as edges_old; downstream routing picks one per edge.
        """
        # Tokenize: node tokens from N, edge tokens from concat(q_src, q_dst, state),
        # proposal tokens (separate mode only) from fresh per-pass noise.
        B = N_old.shape[0]
        node_tok = self.node_in_proj(N_old)                            # [B, K_node, d]
        edge_concat = torch.cat([
            edges_old["q_src"], edges_old["q_dst"], edges_old["state"],
        ], dim=-1)
        edge_tok = self.edge_in_proj(edge_concat)                      # [B, K_edge, d]

        if self.proposal_separate:
            # Fresh-noise proposal tokens — encoder-internal scratch.
            sigma_p = self.log_sigma_proposal.exp().to(node_tok.dtype)
            eps_p = torch.randn(B, self.K_proposal, self.d,
                                device=node_tok.device, dtype=node_tok.dtype)
            prop_tok = self.mu_proposal.to(node_tok.dtype).view(1, 1, -1) + sigma_p.view(1, 1, -1) * eps_p
            tokens = torch.cat([node_tok, edge_tok, prop_tok], dim=1)  # [B, K_node+K_edge+K_proposal, d]
        else:
            tokens = torch.cat([node_tok, edge_tok], dim=1)            # [B, K_node + K_edge, d]

        tokens = tokens + self.pos_emb.to(tokens.dtype).unsqueeze(0)

        for L in range(len(self.cross_blocks)):
            # Cross-attn: both node and edge tokens attend to pins
            tokens = self.cross_blocks[L](tokens, pins, kv_pad_mask=pins_pad_mask)
            # Self-attn: nodes and edges see each other → coordination
            tokens = self.self_blocks[L](tokens, tokens, kv_pad_mask=None)
            # FFN with pre-LN
            tokens = tokens + self.ffns[L](self.ffn_norms[L](tokens))

        # Cross-position whitening — addresses rank collapse in deep
        # transformer stacks (Dong et al. 2021 — pure attention exponentially
        # collapses to rank-1 subspace as depth grows; LayerNorm + FFN
        # counteract but don't eliminate). Subtracts the mean ACROSS
        # positions for each channel, forcing the output to have zero
        # cross-position homogeneity. Probe without this: node_update cross-
        # slot cos = 0.96 at chunk start, edge cross-pos cos = 0.89 after
        # 4 windows. With this + Slot Attention competition: 0.24 / TBD.
        #
        # Reference: σReparam (Zhai et al. 2023) does this more principledly
        # via spectral normalization; whitening is the cheap version.
        tokens = tokens - tokens.mean(dim=1, keepdim=True)

        # Split back into per-group outputs
        node_tok_out = tokens[:, : self.K_node, :]
        if self.proposal_separate:
            # edge tokens are context only — output discarded
            prop_tok_out = tokens[:, self.K_node + self.K_edge:, :]
            edge_raw = self.edge_out_head(self.edge_out_norm(prop_tok_out))  # [B, K_proposal, 2d_n+d_s]
        else:
            # legacy: edge tokens become proposals (1:1 with K_edge)
            edge_tok_out = tokens[:, self.K_node:, :]
            edge_raw = self.edge_out_head(self.edge_out_norm(edge_tok_out))  # [B, K_edge, 2d_n+d_s]

        d_n, d_s = self.d_node, self.d_state
        proposals = {
            "q_src": edge_raw[..., :d_n],
            "q_dst": edge_raw[..., d_n:2 * d_n],
            "state": edge_raw[..., 2 * d_n:2 * d_n + d_s],
        }

        # Node update — Slot Attention competitive write using post-transformer
        # node tokens as queries. Forces specialization (different slots claim
        # different pin subsets).
        slot_q = self.node_q_proj(self.node_in_norm(node_tok_out))   # [B, K_node, d_node]
        pin_ln = self.pin_in_norm(pins)
        pin_k = self.node_k_proj(pin_ln)                             # [B, T_w, d_node]
        pin_v = self.node_v_proj(pin_ln)                             # [B, T_w, d_node]

        scores = torch.matmul(slot_q, pin_k.transpose(-1, -2)) * (self.d_node ** -0.5)
        # softmax OVER SLOTS — pins compete for slots
        attn = scores.softmax(dim=1)                                 # [B, K_node, T_w]
        if pins_pad_mask is not None:
            pad_keep = (~pins_pad_mask).to(attn.dtype).unsqueeze(1)
            attn = attn * pad_keep
        # per-slot renormalize so each slot is a weighted average of its winnings
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
        node_update = torch.matmul(attn, pin_v)                      # [B, K_node, d_node]

        return node_update, proposals


# ─────────────────────────────────────────────────────────────────────────
# Node gate — per-slot anchor-biased blend for the node bank update
# ─────────────────────────────────────────────────────────────────────────

class NodeGate(nn.Module):
    """Per-slot gate g_n ∈ [0,1] for blending the holistic-updater's node
    proposal into the existing bank. Anchor-biased init (sigmoid(-bias))
    so the bank is stable by default; the model has to push g_n up for
    a slot to update meaningfully in a given window.

    Inputs per slot: (N_old[k], node_update[k]). Same MLP shape as the
    edge gate, just over a different vector.
    """

    def __init__(
        self,
        d_node: int,
        hidden: int = 64,
        init_bias: float = 0.5,        # sigmoid(-0.5) ≈ 0.38 — load-bearing
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_node, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.net[2].weight, std=0.01)
        nn.init.constant_(self.net[2].bias, -float(init_bias))

    def forward(self, N_old: Tensor, node_update: Tensor) -> Tensor:
        """Returns g_n ∈ [0,1], shape [B, K_node]."""
        inp = torch.cat([N_old, node_update], dim=-1)
        return torch.sigmoid(self.net(inp).squeeze(-1))


# ─────────────────────────────────────────────────────────────────────────
# Endpoint-space routing (v5.1) — cosine on materialized endpoints
# ─────────────────────────────────────────────────────────────────────────

def slot_routing_on_endpoints(
    endpoint_src_old: Tensor,         # [B, K_edge, d_node]
    endpoint_dst_old: Tensor,
    endpoint_src_prop: Tensor,        # [B, K_proposal, d_node] — materialized proposal endpoints
    endpoint_dst_prop: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Each existing edge picks ONE proposal by combined cosine affinity on
    the MATERIALIZED endpoints (under the new bank). Multiple edges CAN
    pick the same proposal → convergence pressure (k-means dynamic).

    K_proposal can differ from K_edge (the proposal pool can be larger than
    the existing-edge count, giving routing more candidates to choose from).

    Returns:
      picked_idx       : [B, K_edge] int in [0, K_proposal)
      pick_affinity    : [B, K_edge] in [-1, 1]
      pick_count       : [B, K_proposal] int — how many edges picked each proposal
    """
    es_old = F.normalize(endpoint_src_old, dim=-1, eps=1e-6)
    ed_old = F.normalize(endpoint_dst_old, dim=-1, eps=1e-6)
    es_prop = F.normalize(endpoint_src_prop, dim=-1, eps=1e-6)
    ed_prop = F.normalize(endpoint_dst_prop, dim=-1, eps=1e-6)

    aff_src = es_old @ es_prop.transpose(-1, -2)        # [B, K_edge, K_proposal]
    aff_dst = ed_old @ ed_prop.transpose(-1, -2)
    affinity = 0.5 * (aff_src + aff_dst)

    pick_affinity, picked_idx = affinity.max(dim=-1)    # [B, K_edge]

    K_proposal = endpoint_src_prop.shape[1]
    one_hot = F.one_hot(picked_idx, num_classes=K_proposal).to(affinity.dtype)
    pick_count = one_hot.sum(dim=1)                     # [B, K_proposal]
    return picked_idx, pick_affinity, pick_count


def gather_picked_per_slot(field: Tensor, picked_idx: Tensor) -> Tensor:
    """Fetch field-value of each slot's picked proposal. Field [B, K, D] → [B, K, D]."""
    D = field.shape[-1]
    idx = picked_idx.unsqueeze(-1).expand(-1, -1, D)
    return field.gather(dim=1, index=idx)


# ─────────────────────────────────────────────────────────────────────────
# Edge gate — per-edge anchor-biased blend
# ─────────────────────────────────────────────────────────────────────────

class EdgeGate(nn.Module):
    """Per-edge gate g_e ∈ [0,1] for blending new proposals into edge state.

    Mirrors v4's GraphGate (state_old, picked_state, scalars) but reformulated
    for the query-pointer setting:
      inputs per edge:
        - q_src_old, q_dst_old   [d_node × 2]
        - q_src_new, q_dst_new   [d_node × 2]
        - state_old              [d_state]
        - state_new              [d_state]
        - cos(q_src_old, q_src_new)   scalar
        - cos(q_dst_old, q_dst_new)   scalar
    """

    def __init__(
        self,
        d_node: int,
        d_state: int,
        hidden: int = 64,
        init_bias: float = 1.0,        # sigmoid(-1) ≈ 0.27 — anchor-leaning
    ):
        super().__init__()
        in_dim = 4 * d_node + 2 * d_state + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.net[2].weight, std=0.01)
        nn.init.constant_(self.net[2].bias, -float(init_bias))

    def forward(
        self,
        q_src_old: Tensor, q_dst_old: Tensor, state_old: Tensor,
        q_src_new: Tensor, q_dst_new: Tensor, state_new: Tensor,
    ) -> Tensor:
        # cosine_similarity in autocast dtype is safe: outputs are bounded in
        # [-1, 1] and `eps=1e-6` guards the divisor; the bf16-vs-fp32 spread
        # at that magnitude is below the gate MLP's noise floor.
        cos_qs = F.cosine_similarity(q_src_old, q_src_new, dim=-1, eps=1e-6)
        cos_qd = F.cosine_similarity(q_dst_old, q_dst_new, dim=-1, eps=1e-6)
        inp = torch.cat([
            q_src_old, q_dst_old, q_src_new, q_dst_new,
            state_old, state_new,
            cos_qs.unsqueeze(-1), cos_qd.unsqueeze(-1),
        ], dim=-1)
        return torch.sigmoid(self.net(inp).squeeze(-1))


# ─────────────────────────────────────────────────────────────────────────
# Soft-pointer materialization
# ─────────────────────────────────────────────────────────────────────────

def materialize_endpoints(
    q: Tensor,         # [B, K_edge, d_node] — edge queries (src or dst)
    N: Tensor,         # [B, K_node, d_node] — shared node bank
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Stateless soft pointer attention: edge query → distribution over bank → mixture.

    No learnable parameters — used by diagnostic/probe scripts that need to
    test specific temperatures or inspect raw pointer behavior. The trained
    model uses the SoftPointer module below.

    Returns:
      endpoint : [B, K_edge, d_node]  — α @ N
      attn     : [B, K_edge, K_node]  — soft pointer weights (telemetry)
    """
    d_node = q.shape[-1]
    scores = torch.matmul(q, N.transpose(-1, -2)) * (d_node ** -0.5)
    attn = (scores / max(temperature, 1e-3)).softmax(dim=-1)
    endpoint = torch.matmul(attn, N)
    return endpoint, attn


class SoftPointer(nn.Module):
    """v5.3 trained soft-pointer attention from edge queries to the node bank.

    Two additions over the stateless function:
    1) Key/value separation — N is projected through separate W_k (for
       scoring) and W_v (for aggregation). Csordás et al. 2019 identified
       that without K/V separation the address distribution from content-
       based lookup is "noisy and flat, since the value influences the score
       calculation, although only the key should." This is the standard DNC
       pathology. W_k and W_v init to identity so the module starts equal
       to the un-projected function.
    2) Learnable temperature — log_tau is a trained scalar. The model
       self-tunes sharpness (CLIP / Focal Attention style). Init from a
       passed temperature; clamped to avoid pathological extremes.

    Shared across all soft-pointer call sites in the encoder (src/dst,
    existing/proposal). Role specialization lives in the query content,
    not the pointer mechanism.
    """

    def __init__(
        self,
        d_node: int,
        init_temperature: float = 1.0,
        log_tau_floor: float = -3.0,    # τ_min = exp(-3) ≈ 0.05 → very sharp
        log_tau_ceiling: float = 3.0,   # τ_max = exp(3) ≈ 20 → very flat
        kv_split: bool = True,
    ):
        super().__init__()
        import math
        self.kv_split = kv_split
        if kv_split:
            self.W_k = nn.Linear(d_node, d_node, bias=False)
            self.W_v = nn.Linear(d_node, d_node, bias=False)
            nn.init.eye_(self.W_k.weight)
            nn.init.eye_(self.W_v.weight)
        init_log_tau = math.log(max(float(init_temperature), 1e-3))
        self.log_tau = nn.Parameter(torch.tensor(float(init_log_tau)))
        self.log_tau_floor = float(log_tau_floor)
        self.log_tau_ceiling = float(log_tau_ceiling)

    @property
    def temperature(self) -> Tensor:
        """Current τ as a tensor (for telemetry)."""
        return self.log_tau.clamp(self.log_tau_floor, self.log_tau_ceiling).exp()

    def project_kv(self, N: Tensor) -> tuple[Tensor, Tensor]:
        """Precompute K and V projections of the node bank.

        Cache once per `N` and reuse across multiple queries (src/dst/proposal/
        proposal) to avoid redundant W_k(N) / W_v(N) Linear passes. Saves 3×
        the projection cost when streaming_write calls soft_pointer 4 times
        against the same N_new.
        """
        if self.kv_split:
            k = self.W_k(N)
            v = self.W_v(N)
        else:
            k = N
            v = N
        return k, v

    def attend(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Compute soft-pointer attention given precomputed (k, v).

        Use when you have multiple q against the same N: call project_kv(N)
        once, then attend(q_i, k, v) for each query batch.
        """
        d_node = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-1, -2)) * (d_node ** -0.5)
        tau = self.log_tau.clamp(self.log_tau_floor, self.log_tau_ceiling).exp()
        attn = (scores / tau.to(scores.dtype)).softmax(dim=-1)
        endpoint = torch.matmul(attn, v)
        return endpoint, attn

    def forward(self, q: Tensor, N: Tensor) -> tuple[Tensor, Tensor]:
        """
        q : [B, K_edge, d_node] — edge queries (src or dst)
        N : [B, K_node, d_node] — shared node bank
        Returns:
          endpoint : [B, K_edge, d_node]  — α @ V
          attn     : [B, K_edge, K_node]  — soft pointer weights
        """
        k, v = self.project_kv(N)
        return self.attend(q, k, v)


# ─────────────────────────────────────────────────────────────────────────
# Readout (carried over from v4.2 GraphReadout — directional R-GCN style)
# ─────────────────────────────────────────────────────────────────────────
# Kept as v4.2: separate W_src and W_dst project from the SAME shared N[k]
# vector to role-specific contributions. When two edges' soft pointers land
# on the same N[k], both edges get the SAME underlying node vector that the
# readout then projects role-specifically. Sharing happens at the substrate
# (N), not at the projection.

class MessagePassingReadoutV5(nn.Module):
    """v5.4 message-passing readout — graph structure is load-bearing.

    Q/K/V split (designed 2026-05-27):
      Q = per-edge q_src, q_dst (persistent edge queries)
      K = N (stable bank — the "address book")
      V = msg_buf (evolving content buffer at each node)

    The bank N provides stable addresses; msg_buf accumulates content across
    T rounds. α_src and α_dst are computed against N (stable geometry); the
    materialization for messages reads msg_buf (evolving values).

    Per round:
      1. src_ctx = α_src @ msg_buf            # read evolving values
      2. msg     = MLP(src_ctx, edge_state)   # construct message
      3. agg     = α_dst.T @ msg              # route to dst
      4. msg_buf = pre-norm residual update

    Output: K_node memory tokens to Llama (one per bank entry).
    Inactive nodes (no incoming edges) → small agg → small contribution →
    Llama's cross-attention naturally ignores. Implicit activity gating.

    Notes:
      - msg_mlp shared across rounds (RNN-style unroll, parameter-efficient).
      - Pre-norm structure (GPT-style) for stability through T rounds.
      - Edge state static across rounds (standard MPNN convention).
      - W_init identity when d_msg == d_node so round-0 msg_buf is just N.
      - msg_mlp output init small so round-1 perturbation doesn't destroy
        the seed identity in the residual.
    """

    def __init__(
        self,
        d_node: int,
        d_state: int,
        d_llama: int,
        T: int = 4,
        d_mlp_hidden: Optional[int] = None,
        anchor_strength: float = 0.1,
        degree_normalize: bool = True,
    ):
        super().__init__()
        self.d_node = d_node
        self.d_state = d_state
        self.d_msg = d_node                       # symmetric with bank dim
        # Invariant: d_msg == d_node is required for eye_ init of W_init and for
        # the seed re-blend in the loop (anchor term has shape [..., d_msg]).
        # Codified here so a future config knob can't silently break it.
        assert self.d_msg == self.d_node, "d_msg must equal d_node for eye_ init"
        self.T = T
        # GCNII-style anchor strength: at each round, blend (1-α)·updated +
        # α·seed. Prevents msg_buf from drifting away from the seed identity
        # over T rounds (oversmoothing defense). α=0 disables.
        self.anchor_strength = float(anchor_strength)
        # GAT-style per-node degree normalization: divide agg by Σ_i α_dst_i[k]
        # so hub nodes don't dominate the residual stream. Without this, a node
        # touched by 10 edges gets ~10× larger agg than an isolated node →
        # variance imbalance + gradient asymmetry + hub-driven oversmoothing.
        self.degree_normalize = bool(degree_normalize)

        # Init seed: msg_buf_0 = W_init(N). Identity init → starts equal to N.
        self.W_init = nn.Linear(d_node, d_node, bias=False)
        nn.init.eye_(self.W_init.weight)

        # Pre-norm before reading msg_buf each round (GPT-style stability).
        self.pre_norm = nn.LayerNorm(d_node)

        # v5.5: PER-ROUND message-construction MLPs (un-shared across rounds).
        # Earlier versions shared one MLP across all T rounds (RNN-style param
        # efficiency). The decode probe at v5.4 showed memory tokens producing
        # gibberish when fed directly to lm_head — confirming the readout was
        # the bottleneck: information was in the memory but not packaged in a
        # Llama-readable form. Un-sharing lets each round learn a distinct
        # message-construction stage (e.g. round 0-1 aggregate, round 2-3
        # compose, round 4-5 format), at the cost of T× the msg_mlp params.
        d_h = d_mlp_hidden if d_mlp_hidden is not None else 2 * d_node
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_node + d_state, d_h),
                nn.GELU(),
                nn.Linear(d_h, d_node),
            )
            for _ in range(T)
        ])
        # Small init on output linear so round-1 doesn't smash the seed in
        # the residual (msg_buf = msg_buf + agg). The model can scale it up
        # from there. Applied identically per round.
        for mlp in self.msg_mlps:
            nn.init.normal_(mlp[-1].weight, std=0.02)
            nn.init.zeros_(mlp[-1].bias)

        # v5.5: post-MP FFN block — a "thinking" pass on msg_buf after the T
        # MP rounds finish, before W_out projects to d_llama. GPT-style:
        # LayerNorm + (Linear → GELU → Linear) with zero-init residual so it
        # starts as identity (no effect until trained). This gives the readout
        # an extra non-linear transformation between the graph-evolving content
        # and the Llama-input projection — the missing piece in v5.4's readout
        # that left memory readable to MP but not to Llama.
        self.post_ffn_norm = nn.LayerNorm(d_node)
        self.post_ffn = nn.Sequential(
            nn.Linear(d_node, 4 * d_node),
            nn.GELU(),
            nn.Linear(4 * d_node, d_node),
        )
        nn.init.zeros_(self.post_ffn[-1].weight)
        nn.init.zeros_(self.post_ffn[-1].bias)

        # Final output projection to Llama hidden.
        self.out_norm = nn.LayerNorm(d_node)
        self.W_out = nn.Linear(d_node, d_llama)

    def forward(
        self,
        N: Tensor,              # [B, K_node, d_node]
        alpha_src: Tensor,      # [B, K_edge, K_node] — soft pointer (from soft_pointer)
        alpha_dst: Tensor,      # [B, K_edge, K_node]
        edge_state: Tensor,     # [B, K_edge, d_state]
        compute_telemetry: bool = False,
    ) -> tuple[Tensor, dict]:
        """Returns (memory, telemetry).
        memory: [B, K_node, d_llama]
        telemetry: per-round diagnostic dict (msg_buf norms, cross-node cos, agg magnitudes)
          - Cross-node cosine is a K_node×K_node matmul per round, ~T·B·K² FLOPs.
            Gate via `compute_telemetry=True` (caller passes only during eval).
            Training-mode returns zero-filled stacks of length T so downstream
            logging keys stay present and shape-compatible.
        """
        seed = self.W_init(N)                                          # [B, K_node, d_msg]
        msg_buf = seed

        per_round_buf_norm: list[Tensor] = []
        per_round_agg_norm: list[Tensor] = []
        per_round_buf_cos: list[Tensor] = []

        for t in range(self.T):
            # Pre-norm read
            norm_buf = self.pre_norm(msg_buf)                          # [B, K_node, d_msg]
            # Materialize src context from msg_buf (NOT from N — that's the key insight)
            src_ctx = torch.matmul(alpha_src, norm_buf)                # [B, K_edge, d_msg]
            # Build messages: src_ctx ⊕ edge_state → d_msg
            msg_in = torch.cat([src_ctx, edge_state], dim=-1)
            msg = self.msg_mlps[t](msg_in)                             # [B, K_edge, d_msg]
            # Aggregate at dst — α_dst[k] is "how much each edge points at k"
            agg = torch.matmul(alpha_dst.transpose(-1, -2), msg)       # [B, K_node, d_msg]

            # GAT/mean-style per-node degree normalization: prevents hub
            # nodes from dominating the residual stream. Mathematically:
            # agg[b,k] := (Σ_i α_dst_i[k] · msg_i) / max(Σ_i α_dst_i[k], ε).
            # A node touched by 5 edges gets the AVERAGE incoming message,
            # not the SUM. Inactive nodes (deg≈0) get agg≈0 either way.
            if self.degree_normalize:
                deg = alpha_dst.sum(dim=1).unsqueeze(-1)               # [B, K_node, 1]
                agg = agg / deg.clamp_min(1e-6)

            # Residual update with GCNII-style initial residual: blend updated
            # buf with seed at every round. (1-α)·(buf+agg) + α·seed. Keeps
            # node identity from drifting too far over T rounds; combined with
            # degree-norm, this is the standard defense against deep-GNN
            # oversmoothing. anchor_strength=0 disables (pure residual).
            updated = msg_buf + agg
            if self.anchor_strength > 0:
                a = self.anchor_strength
                msg_buf = (1.0 - a) * updated + a * seed
            else:
                msg_buf = updated

            if compute_telemetry:
                with torch.no_grad():
                    per_round_buf_norm.append(msg_buf.norm(dim=-1).mean().to(torch.float32))
                    per_round_agg_norm.append(agg.norm(dim=-1).mean().to(torch.float32))
                    bn = F.normalize(msg_buf, dim=-1, eps=1e-6)
                    cos = torch.matmul(bn, bn.transpose(-1, -2))           # [B, K_node, K_node]
                    K_n = bn.shape[1]
                    off = ~torch.eye(K_n, dtype=torch.bool, device=bn.device)
                    per_round_buf_cos.append(cos[:, off].mean().to(torch.float32))

        # v5.5: post-MP FFN block — Transformer-style residual "thinking" pass
        # on the final msg_buf state. Zero-init residual means this starts as
        # identity and only contributes once trained, so it can't destabilize
        # the seed-recovery dynamics from the T MP rounds.
        msg_buf = msg_buf + self.post_ffn(self.post_ffn_norm(msg_buf))
        memory = self.W_out(self.out_norm(msg_buf))                    # [B, K_node, d_llama]

        if compute_telemetry:
            telemetry = {
                "mp_buf_norm_per_round": torch.stack(per_round_buf_norm),
                "mp_agg_norm_per_round": torch.stack(per_round_agg_norm),
                "mp_buf_cross_node_cos_per_round": torch.stack(per_round_buf_cos),
            }
        else:
            zero = torch.zeros(self.T, device=memory.device, dtype=torch.float32)
            telemetry = {
                "mp_buf_norm_per_round": zero,
                "mp_agg_norm_per_round": zero,
                "mp_buf_cross_node_cos_per_round": zero,
            }
        return memory, telemetry


class GraphReadoutV5(nn.Module):
    """v4.2-style directional readout, now operating on materialized
    endpoints from the shared bank.

    Same architecture as v4.2 GraphReadout (W_src/W_dst + cross-edge attn +
    saliency-style projection), just dropped the `u`-gate since there's no
    per-edge popularity EMA in v5 (no eviction mechanism to inform).
    """

    def __init__(
        self,
        d_node: int,
        d_state: int,
        d_llama: int,
        d_hidden: int,
        n_heads: int = 4,
    ):
        super().__init__()
        self.W_src = nn.Linear(d_node, d_hidden)
        self.W_dst = nn.Linear(d_node, d_hidden)
        self.W_state = nn.Linear(d_state, d_hidden)

        self.cross_edge_norm = nn.LayerNorm(d_hidden)
        self.cross_edge_attn = nn.MultiheadAttention(
            d_hidden, n_heads, batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_hidden)
        self.ffn = nn.Sequential(
            nn.Linear(d_hidden, 4 * d_hidden), nn.GELU(),
            nn.Linear(4 * d_hidden, d_hidden),
        )

        self.proj_norm = nn.LayerNorm(d_hidden)
        self.proj = nn.Sequential(
            nn.Linear(d_hidden, d_llama),
            nn.LayerNorm(d_llama),
        )

    def forward(
        self,
        endpoint_src: Tensor,    # [B, K_edge, d_node]
        endpoint_dst: Tensor,    # [B, K_edge, d_node]
        state: Tensor,           # [B, K_edge, d_state]
    ) -> Tensor:
        # Directional encoding — same N[k] gets projected differently for
        # src vs dst, but the underlying vector is the SAME shared node.
        h = (
            self.W_src(endpoint_src)
            + self.W_dst(endpoint_dst)
            + self.W_state(state)
        )                                                              # [B, K_edge, d_hidden]

        # Cross-edge message passing
        q = self.cross_edge_norm(h)
        attn_out, _ = self.cross_edge_attn(q, q, q)
        h = h + attn_out
        h = h + self.ffn(self.ffn_norm(h))

        h = self.proj_norm(h)
        return self.proj(h)                                            # [B, K_edge, d_llama]


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _rmsnorm(x: Tensor, eps: float = 1e-6) -> Tensor:
    d = x.shape[-1]
    return x * (d ** 0.5) / (x.pow(2).sum(-1, keepdim=True).sqrt() + eps)
