"""Trajectory walker — the core read/write algorithm.

Walks through the vocabulary graph for K hops, scoring edge candidates
against the routing query and falling back to the global vocab. Same
machinery for read and write; they differ only on whether edge states
get updated (write) or not (read), and on what `window_hiddens` they
cross-attend to.

For v1: **signature = step_query** (the routing query is the stored
content). No separate `signature_fn`. Avoids the gradient flow issue
in @torch.no_grad() edge updates because the query itself gets gradient
via its use in vocab_score and the contrastive losses.

Gradient flow summary:
  step_query  ─── (grad) ──→ used in vocab_score (matches concept_ids)
              ─── (grad) ──→ used in edge_score (RMS-norm cosine vs edge_state buffer)
              ─── (grad) ──→ used as STE input for softmax_top1_ste
              ─── (no_grad) ─→ written to edge_state buffer (intentional;
                              prevents autograd accumulation across windows)

concept_ids: gets gradient through every vocab_score AND through every
trajectory's concept_ids[next_node] gather. Stays learnable throughout.

edge_state: buffer, never gets optimizer steps. EMA-updated under
no_grad. Gradient flows through reads of it but doesn't propagate back
to past writes (intentional — see manifold.py docstring).

Design doc: docs/design_vocabulary_trajectory.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.trajectory_memory_v2._shared import (
    CrossAttention,
    EntryProjector,
    per_j_attn,
    routing_aux_losses,
    softmax_top1_ste,
)
from src.trajectory_memory_v2.config import TrajMemV2Config
from src.trajectory_memory_v2.manifold import VocabularyManifold, rms_norm_last


@dataclass
class WalkerResult:
    """Output of one trajectory walk."""
    visited_ids: Tensor       # [BS, J, K]   long — node indices visited (incl. entry)
    visited_embeds: Tensor    # [BS, J, K, D] — concept_ids gathered along trajectory
    step_queries: Tensor      # [BS, J, K, D] — per-step queries (for contrastive)
    aux_lb: Tensor            # scalar — load-balance aux loss
    aux_z: Tensor             # scalar — z-loss aux loss
    # Diagnostics (no grad)
    entry_logits_max: float  # mean of max entry logit (routing confidence)
    edge_score_active_frac: float  # fraction of hop steps where edge_score > 0


class TrajectoryWalker(nn.Module):
    """Shared walker logic. Used by both Read and Write modules.

    The Walker owns the per-walker parameters: step_mlp, cross_attn,
    history_attn, pos_enc, and a learnable λ that weights edge_score
    vs. vocab_score. Read and Write modules differ in:
        - which `window_hiddens` they cross-attend to (passed in by caller)
        - whether they call `manifold.update_edges` (write_mode=True)
        - K_read vs K_write (the trajectory length)
        - entry projection lookup (entry node is global; passed in pooled)

    The Walker does NOT own `entry_proj` or the manifold itself — those
    are shared singletons in IntegratedLM. The Walker takes them as
    forward() arguments.
    """

    def __init__(self, cfg: TrajMemV2Config, K: int):
        super().__init__()
        self.cfg = cfg
        self.K = K  # trajectory length (K_read or K_write)

        D = cfg.D_concept
        d_lm = cfg.d_lm

        # Cross-attention over window hiddens (shared by all K hops).
        # `precompute_kv` caches K,V once per window — same as v1.
        # Single-head per v1 convention (q=D, kv=d_lm, attn=D).
        self.cross_attn = CrossAttention(d_q=D, d_kv=d_lm, d_attn=D)

        # History-attention over visited trajectory so far.
        # Uses v1's _CrossAttn: (q in D, kv in D) → out in D.
        # per_j_attn handles the per-trajectory batching.
        self.history_attn = CrossAttention(d_q=D, d_kv=D, d_attn=D)

        # Positional encoding for the K-position dim in history.
        pe = torch.zeros(K, D)
        position = torch.arange(0, K).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, D, 2).float() * (-math.log(10000.0) / D),
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_enc", pe, persistent=False)

        # Step query MLP: combines (current_embed, history_attn_out,
        # cross_attn_out, running_cue) → step_query.
        self.step_mlp = nn.Sequential(
            nn.Linear(4 * D, D, bias=True),
            nn.GELU(),
            nn.Linear(D, D, bias=True),
        )
        # Smaller-than-Xavier init — same rationale as v1 entry_proj
        # (prevents initial-routing concentration onto few cells).
        nn.init.normal_(self.step_mlp[0].weight, std=0.05)
        nn.init.zeros_(self.step_mlp[0].bias)
        nn.init.normal_(self.step_mlp[-1].weight, std=0.05)
        nn.init.zeros_(self.step_mlp[-1].bias)

        # Projection that the running cue uses to absorb each visited
        # node's embedding. Keeps the cue from growing in magnitude
        # without bound.
        self.cue_proj = nn.Linear(D, D, bias=False)
        nn.init.normal_(self.cue_proj.weight, std=0.05)

        # Learnable λ that weights edge_score vs vocab_score in the
        # combined score. Initialized to cfg.lambda_edge_init.
        # Allowed to go negative (model can decide).
        self.lambda_edge = nn.Parameter(
            torch.tensor(cfg.lambda_edge_init, dtype=torch.float32),
        )
        # Learnable reuse bonus added to existing-edge dsts in the combined
        # logits. Encourages routing toward already-allocated cells so the
        # walker reuses the abstract vocabulary instead of constantly
        # allocating new edges. Default init = 1.0 (~ one std-dev of vocab
        # logits); model can adjust if reuse pressure should be higher or
        # lower based on training signal.
        self.reuse_bonus = nn.Parameter(
            torch.tensor(cfg.reuse_bonus_init, dtype=torch.float32),
        )

    def forward(
        self,
        window_hiddens: Tensor,          # [BS, T, d_lm]
        entry_proj: EntryProjector,      # shared with paired write/read module
        manifold: VocabularyManifold,
        window_mask: Optional[Tensor] = None,  # [BS, T] bool — True=real, False=pad
        write_mode: bool = False,
        hard_routing: bool = True,
    ) -> WalkerResult:
        """Run one trajectory walk over the window.

        Args:
            window_hiddens: [BS, T, d_lm] — Llama hiddens for this window.
                For writes: current passage's hiddens.
                For reads: question hiddens (Wave 1) or prev-window
                            hiddens (streaming).
            entry_proj: shared EntryProjector (Hopfield-tied with paired module).
            manifold: VocabularyManifold (vocabulary + edge memory).
            write_mode: if True, update edge states after each traversal.
                If False, just walk (read mode).
            hard_routing: if True, softmax-top-1 STE; if False, argmax (eval).

        Returns:
            WalkerResult with visited_ids, visited_embeds, step_queries,
            aux losses, and diagnostics.
        """
        cfg = self.cfg
        BS, T, d_lm = window_hiddens.shape
        D = cfg.D_concept
        J = cfg.J
        K = self.K
        N = cfg.N

        # ── ENTRY: global lookup ────────────────────────────────────
        # Masked mean-pool. Without the mask, pad-token hiddens dilute
        # the pooled context — for a 50-token passage padded to T=256,
        # ~80% of the pool would be pad-token noise.
        if window_mask is not None:
            mask_f = window_mask.to(window_hiddens.dtype).unsqueeze(-1)   # [BS, T, 1]
            denom = mask_f.sum(dim=1).clamp_min(1.0)                      # [BS, 1]
            pooled = (window_hiddens * mask_f).sum(dim=1) / denom         # [BS, d_lm]
        else:
            pooled = window_hiddens.mean(dim=1)            # [BS, d_lm]
        Q_entry = entry_proj(pooled)                       # [BS, J, D]

        ids_normed = manifold.concept_ids_normed           # [N, D] (computed every call)
        entry_logits_raw = torch.einsum(
            "bjd,nd->bjn", rms_norm_last(Q_entry), ids_normed,
        )
        eff_scale = D ** -0.5
        entry_logits = entry_logits_raw * eff_scale        # [BS, J, N]

        entry_one_hot, entry_idx = softmax_top1_ste(
            entry_logits, hard=hard_routing,
        )                                                  # [BS, J, N], [BS, J]
        # Accumulate routing aux losses on entry decision.
        aux_e = routing_aux_losses(entry_logits, entry_one_hot)
        aux_lb_total = aux_e["load_balance"]
        aux_z_total = aux_e["z_loss"]

        # Diagnostic: routing confidence
        entry_logits_max = float(entry_logits.max(dim=-1).values.mean().detach())

        # Gather entry embeddings. Use einsum with one-hot to keep gradient
        # flowing back to concept_ids via the soft routing path.
        concept_ids = manifold.concept_ids                  # [N, D] (with grad)
        # one_hot @ concept_ids = gather but differentiable through STE
        entry_embed = torch.einsum("bjn,nd->bjd", entry_one_hot, concept_ids)  # [BS, J, D]
        current = entry_idx                                 # [BS, J] long

        # Initialize running cue with Q_entry (the D-dim projection of
        # pooled context). The cue accumulates a leaky integration of
        # each visited node's embedding across hops — see the per-hop
        # update below.
        cue_D = Q_entry                                     # [BS, J, D]

        # Accumulators
        visited_ids = [current]                             # list of [BS, J]
        visited_embeds = [entry_embed]                      # list of [BS, J, D]
        step_queries = [Q_entry]                            # entry's "step query" is Q_entry

        # Precompute cross_attn KV for the window (one-time per call).
        cross_K, cross_V = self.cross_attn.precompute_kv(window_hiddens)
        # cross_attn key mask: per-batch [BS, T] → broadcast over J inside
        # forward_with_kv. Bool: True = real token (attendable).
        cross_key_mask = window_mask  # may be None

        # Edge-score diagnostic
        edge_active_steps = 0

        # ── K-1 HOPS ────────────────────────────────────────────────
        for t in range(1, K):
            # Step query construction
            current_embed = visited_embeds[-1]              # [BS, J, D]

            # History attention — query is current; keys/values are visited so far + pos_enc
            history_kv = torch.stack(visited_embeds, dim=2) # [BS, J, t, D]
            pos = self.pos_enc[:t].unsqueeze(0).unsqueeze(1)  # [1, 1, t, D]
            history_kv = history_kv + pos
            history_attn_out = per_j_attn(
                self.history_attn, current_embed, history_kv,
            )                                               # [BS, J, D]

            # Cross-attention over the window's Llama hiddens
            cross_attn_out = self.cross_attn.forward_with_kv(
                current_embed, cross_K, cross_V, key_mask=cross_key_mask,
            )                                               # [BS, J, D]

            # Step query: combine all four signals
            step_input = torch.cat(
                [current_embed, history_attn_out, cross_attn_out, cue_D], dim=-1,
            )                                               # [BS, J, 4D]
            step_query = self.step_mlp(step_input)          # [BS, J, D]

            # ── Score against vocab AND edges ────────────────────
            # Vocab score: SDPA against all N concept_ids
            vocab_logits = torch.einsum(
                "bjd,nd->bjn", rms_norm_last(step_query), ids_normed,
            ) * eff_scale                                   # [BS, J, N]

            # Edge score: zero-centered cosine over edges from current
            edge_states, edge_dsts, edge_active = manifold.lookup_edges(current)
            # edge_states: [BS, J, K_max, D]
            # edge_dsts:   [BS, J, K_max]  long
            # edge_active: [BS, J, K_max]  bool

            # RMS-norm the edge states; q_norm already computed for vocab.
            # Apply eff_scale = D^-0.5 so the dot product is unit-variance
            # (equivalent to cosine similarity in [-1, 1] expectation).
            # Without this scale, edge scores have variance D and blow up
            # the z-loss via the scatter_add into combined logits.
            q_norm = rms_norm_last(step_query)              # [BS, J, D]
            e_norm = rms_norm_last(edge_states)             # [BS, J, K_max, D]
            edge_score_raw = torch.einsum(
                "bjd,bjkd->bjk", q_norm, e_norm,
            ) * eff_scale                                   # [BS, J, K_max]
            # Mask inactive slots to 0 contribution
            edge_score = edge_score_raw * edge_active.float()  # [BS, J, K_max]

            # Scatter edge_score onto N-space (using edge_dsts as indices)
            # safe_dsts: replace -1 with 0 (will contribute 0 since masked)
            safe_dsts = edge_dsts.clamp_min(0)              # [BS, J, K_max]
            # Match dtypes for scatter_add_ — under bf16 autocast
            # vocab_logits is bf16; edge_score may be fp32 (came from
            # an einsum whose inputs included fp32 buffers). Cast src
            # to match self.
            edge_contrib = torch.zeros_like(vocab_logits)   # [BS, J, N]
            edge_contrib.scatter_add_(
                dim=2, index=safe_dsts, src=edge_score.to(edge_contrib.dtype),
            )                                               # [BS, J, N]

            # Reuse bonus: existing-edge dsts get a constant boost so the
            # walker prefers reusing vocab over allocating new edges. Build
            # a 0/1 mask at active-edge dst positions, multiply by the
            # learnable bonus.
            existing_dst_mask = torch.zeros_like(vocab_logits)
            existing_dst_mask.scatter_(
                dim=2, index=safe_dsts,
                src=edge_active.to(existing_dst_mask.dtype),
            )

            combined = (
                vocab_logits
                + self.lambda_edge * edge_contrib
                + self.reuse_bonus * existing_dst_mask
            )

            # Top-1 selection with STE
            combined_one_hot, next_idx = softmax_top1_ste(
                combined, hard=hard_routing,
            )                                               # [BS, J, N], [BS, J]

            # Accumulate routing aux losses on this hop
            aux_h = routing_aux_losses(combined, combined_one_hot)
            aux_lb_total = aux_lb_total + aux_h["load_balance"]
            aux_z_total = aux_z_total + aux_h["z_loss"]

            # Differentiable gather for next node's embedding
            next_embed = torch.einsum("bjn,nd->bjd", combined_one_hot, concept_ids)

            # Update running cue: leaky integrator. Each new visit's
            # contribution decays the previous cue by `cue_decay`, then
            # adds the freshly-projected next_embed. Bounds steady-state
            # magnitude to ~|cue_proj output| / (1 - cue_decay), preserving
            # recency-weighted relative magnitude information across hops
            # (older visits weight less, but their direction persists).
            # The decay is a static hyperparameter; it could be made
            # learnable per-position later if needed.
            cue_D = cfg.cue_decay * cue_D + self.cue_proj(next_embed)

            # ── WRITE / READ side effects on manifold (no-grad) ──
            src_flat = current.reshape(-1)                   # [BS*J]
            dst_flat = next_idx.reshape(-1)                  # [BS*J]
            if write_mode:
                # signature = step_query (v1 simplification; no signature_fn).
                sig_flat = step_query.reshape(-1, D)         # [BS*J, D]
                # Detach is intentional — update is no_grad anyway,
                # but be explicit so future readers don't get confused.
                manifold.update_edges(
                    src_flat.detach(), dst_flat.detach(), sig_flat.detach(),
                )
            else:
                # Read mode: the chosen next_idx might match an existing
                # edge's dst. If so, that edge was "retrieved" — record it
                # for the W-TinyLFU eviction signal.
                manifold.record_read_touch(
                    src_flat.detach(), dst_flat.detach(),
                )

            # Track diagnostic
            edge_active_steps += int(edge_active.any(dim=-1).float().mean().item() > 0.0)

            # Advance
            current = next_idx
            visited_ids.append(current)
            visited_embeds.append(next_embed)
            step_queries.append(step_query)

        # ── PACKAGE OUTPUT ──────────────────────────────────────────
        # Average aux losses over the K decisions (1 entry + K-1 hops)
        aux_lb_total = aux_lb_total / K
        aux_z_total = aux_z_total / K

        result = WalkerResult(
            visited_ids=torch.stack(visited_ids, dim=2),     # [BS, J, K]
            visited_embeds=torch.stack(visited_embeds, dim=2),  # [BS, J, K, D]
            step_queries=torch.stack(step_queries, dim=2),   # [BS, J, K, D]
            aux_lb=aux_lb_total,
            aux_z=aux_z_total,
            entry_logits_max=entry_logits_max,
            edge_score_active_frac=float(edge_active_steps) / max(K - 1, 1),
        )
        return result
