"""ReadTrajectoryGenerator — J parallel autoregressive hops at start of window.

Conditions only on `prev_window_hiddens: [BS, T_window, D_lm]` (Llama hiddens
from the previous window). Outputs `[BS, J*K_read, D_concept]` — J·K_read
visited concept states, flattened, ready for cross-attention into Llama.

Per-hop Q construction takes `current_state[j], history_attn, cross_attn`
(no `id_vec[current]` — redundant with the K side of the QK match).
Order is encoded by sinusoidal positional encoding added to history-attn
keys/values.

Discrete next-concept selection uses Gumbel-top-1 STE during training
(hard forward, soft backward) and argmax at inference.

See docs/plan_trajectory_memory.md §2.2, §3.1, §4.8.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.manifold import Manifold


def make_pos_enc(K: int, D: int) -> Tensor:
    """Sinusoidal positional encoding [K, D]. Standard Transformer-style."""
    pos = torch.arange(K, dtype=torch.float32).unsqueeze(1)        # [K, 1]
    div = torch.exp(
        torch.arange(0, D, 2, dtype=torch.float32) * (-math.log(10000.0) / D)
    )                                                              # [D/2]
    pe = torch.zeros(K, D)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def routing_aux_losses(
    logits: Tensor,
    one_hot: Tensor,
    *,
    z_loss_logits: Tensor | None = None,
) -> dict[str, Tensor]:
    """Compute Switch-Transformer load-balance + ST-MoE z-loss at a routing
    decision. Returns scalars suitable for adding to the training loss.

    Args:
        logits:        [..., N] post-softmax-input routing scores
                       (differentiable, typically scaled by logit_scale)
        one_hot:       [..., N] hard one-hot selection from softmax-STE
                       (forward-detached; its sum-over-vocab is 1 per row)
        z_loss_logits: optional [..., N] — if provided, z_loss uses these
                       instead of `logits`. Pass the UNSCALED cosine logits
                       so z_loss penalizes raw logit magnitude (per ST-MoE
                       convention) without fighting `logit_scale_raw`
                       learning to grow. With scaled logits, z_loss would
                       drive logit_scale toward zero — the opposite of
                       what we want.

    Returns:
        {
          'load_balance': scalar  — Switch loss: N · Σᵢ fᵢ · Pᵢ
                                    minimized when routing is uniform.
                                    Uses `logits` (the scaled version is
                                    correct — penalizes actual softmax
                                    distribution after temperature).
          'z_loss':       scalar  — mean of (logsumexp(z_loss_logits))²
                                    keeps RAW logit magnitudes from
                                    saturating; orthogonal to logit_scale.
        }
    """
    N = logits.shape[-1]
    # Flatten everything but the last dim — treats each row as one routing
    # decision. Works for entry-router [BS, J, N] and per-hop [BS, J, K_max].
    flat_logits = logits.reshape(-1, N)              # [M, N]
    flat_oh = one_hot.reshape(-1, N)                 # [M, N]
    if flat_logits.shape[0] == 0:
        # Degenerate — return zero scalars with grad to logits to keep
        # autograd happy.
        z = (flat_logits * 0.0).sum()
        return {"load_balance": z, "z_loss": z}

    # Pᵢ: mean softmax probability mass on concept i across the batch.
    # Differentiable — gradient flows back through softmax to logits.
    P = F.softmax(flat_logits, dim=-1).mean(dim=0)   # [N]
    # fᵢ: fraction of HARD selections that picked concept i.
    # Detached — treated as a constant load indicator.
    f = flat_oh.detach().mean(dim=0)                 # [N]
    # Switch loss: N · Σᵢ fᵢ · Pᵢ. Uniform routing → Σ (1/N)(1/N) · N = 1.
    # Fully one-hot on a single concept → 1 · 1 · N = N. So loss grows
    # linearly with concentration severity.
    load_balance = N * (f * P).sum()

    # z-loss: penalize magnitude of logsumexp (= log normalizer of softmax).
    # Compute on UNSCALED logits when provided — this controls raw cosine
    # magnitudes without fighting the learnable logit_scale_raw.
    z_logits = z_loss_logits if z_loss_logits is not None else logits
    z_flat = z_logits.reshape(-1, N)
    z_loss = (torch.logsumexp(z_flat, dim=-1) ** 2).mean()

    return {"load_balance": load_balance, "z_loss": z_loss}


def softmax_top1_ste(
    logits: Tensor,
    *,
    hard: bool,
) -> tuple[Tensor, Tensor]:
    """Softmax-top-1 with straight-through estimator (no Gumbel noise).

    Modern MoE pattern (Mixtral / DeepSeek). Forward picks argmax; backward
    routes through softmax probabilities.

    Why no Gumbel: with cosine routing the logits are bounded in [-1, 1],
    so Gumbel(0,1) noise (std ~1.28) dominates the signal — routing is
    random regardless of weights. Pre-multiplying logits by a learnable
    `logit_scale` (caller's responsibility) restores signal dominance.
    Without the noise term, exploration comes from random init + the
    load-balance aux loss, same as standard MoE.

    `hard=False` returns the argmax one-hot directly without an STE path.
    That hard one-hot has no autograd link back to `logits` (argmax is
    non-differentiable) — so this branch silently kills gradient into
    routing. It's safe inside `torch.no_grad()` (inference / eval) but
    a trap during training. The assert below catches the trap.
    """
    soft = F.softmax(logits, dim=-1)
    idx = soft.argmax(dim=-1)
    hard_one_hot = F.one_hot(idx, num_classes=logits.shape[-1]).to(soft.dtype)
    if not hard:
        # Non-differentiable argmax — no STE. Callers using hard=False
        # are responsible for knowing they get no gradient through
        # routing. Used in Wave 3 prompt prefill (grad enabled for the
        # adapter, but routing-grad intentionally not used during the
        # frozen-policy generation) and eval paths (under no_grad).
        return hard_one_hot, idx
    # Straight-through: forward hard, backward soft probabilities.
    one_hot = (hard_one_hot - soft).detach() + soft
    return one_hot, idx


def gumbel_top1_ste(
    logits: Tensor,
    tau: Tensor | float,
    *,
    hard: bool,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Gumbel-top-1 with straight-through estimator.

    During training (hard=True): hard one-hot forward, soft backward.
    During inference (hard=False): just argmax (no Gumbel noise).

    Returns:
        one_hot:  [..., V]  one-hot selection (forward); soft probs in backward
        index:    [...]     int64 selected index
    """
    if not hard:
        idx = logits.argmax(dim=-1)
        one_hot = F.one_hot(idx, num_classes=logits.shape[-1]).to(logits.dtype)
        return one_hot, idx

    # Gumbel noise. `torch.rand_like` sometimes lacks a generator kwarg
    # depending on torch version; use empty + uniform_ for portability.
    if generator is None:
        u = torch.rand_like(logits)
    else:
        u = torch.empty_like(logits).uniform_(0.0, 1.0, generator=generator)
    # NB: parens are load-bearing. `-torch.log(x).clamp_min(1e-20)` parses as
    # `-(torch.log(x).clamp_min(1e-20))`, which clamps the *negative* log to
    # 1e-20 (i.e. all values become 1e-20), then negates → all `-1e-20` →
    # `log(-1e-20)` = NaN. We need to clamp AFTER negating.
    log_u = torch.log(u.clamp_min(1e-20))                # in (-46, 0]
    neg_log_u = (-log_u).clamp_min(1e-20)                # positive
    g = -torch.log(neg_log_u)                            # standard Gumbel(0,1)
    y = (logits + g) / tau
    soft = F.softmax(y, dim=-1)
    idx = soft.argmax(dim=-1)
    hard_one_hot = F.one_hot(idx, num_classes=logits.shape[-1]).to(soft.dtype)
    # Straight-through: forward = hard, backward = soft
    one_hot = (hard_one_hot - soft).detach() + soft
    return one_hot, idx


def per_j_attn(attn_module: "_CrossAttn", q: Tensor, kv: Tensor) -> Tensor:
    """Apply cross-attention separately for each j in the J dim.

    Used by both read_module and write_module to attend per-trajectory
    without mixing across j.

    Args:
        attn_module: a `_CrossAttn` instance.
        q:  [BS, J, d_q]
        kv: [BS, J, NK, d_kv]

    Returns:
        [BS, J, d_q] — attention readout per (BS, J).

    Implementation: fold J into BS, apply attn, fold back.
    """
    BS, J, D = q.shape
    d_kv = kv.shape[-1]
    NK = kv.shape[-2]
    q_flat = q.reshape(BS * J, 1, D)
    kv_flat = kv.reshape(BS * J, NK, d_kv)
    out = attn_module(q_flat, kv_flat)                            # [BS*J, 1, D]
    return out.reshape(BS, J, D)


class _CrossAttn(nn.Module):
    """Single-head cross-attention. Q in d_q, KV in d_kv, both project to d.

    Used in two places:
    - history_attn: Q = current state, KV = visited list + pos_enc
                    KV genuinely differs per (b, j) trajectory; use
                    `forward(q, kv)` with kv shape [BS, J, NK, d_kv].
    - cross_attn:   Q = current state, KV = window hiddens
                    KV is the SAME for all j and all hops within a window.
                    Use `precompute_kv(kv)` once per window, then
                    `forward_with_kv(q, K, V)` per hop. This avoids
                    materializing a J-broadcasted copy of `kv` and skips
                    redundant W_k / W_v projections at each hop.

    Both fast paths run in bf16 inside autocast — Linear weights stay
    fp32, but activations and matmuls are bf16 (~50% memory cut). The
    fp32 caller boundary is preserved (output is cast back).

    Single-head suffices because trajectory hops are small and we don't
    need multi-head expressivity; if benching shows it's too lossy we can
    bump to multi-head.
    """

    def __init__(self, d_q: int, d_kv: int, d_attn: int):
        super().__init__()
        self.d_attn = d_attn
        self.W_q = nn.Linear(d_q, d_attn, bias=False)
        self.W_k = nn.Linear(d_kv, d_attn, bias=False)
        self.W_v = nn.Linear(d_kv, d_attn, bias=False)
        for m in (self.W_q, self.W_k, self.W_v):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        """Per-(b,j) attention path. kv shape includes the same leading
        dims as q (the per_j_attn caller folds J into BS).

        Args:
            q:  [BS, *Q, d_q]
            kv: [BS, *KV, d_kv]
        Returns:
            [BS, *Q, d_attn]
        """
        out_dtype = q.dtype
        with torch.autocast(
            device_type="cuda" if q.is_cuda else "cpu",
            dtype=torch.bfloat16, enabled=q.is_cuda,
        ):
            Q = self.W_q(q)                                          # [BS, *Q, d_attn]
            K = self.W_k(kv)                                         # [BS, *KV, d_attn]
            V = self.W_v(kv)
            # Flatten Q-extra-dims and KV-extra-dims for bmm-friendly attention.
            BS = q.shape[0]
            Q2 = Q.reshape(BS, -1, self.d_attn)                      # [BS, NQ, d]
            K2 = K.reshape(BS, -1, self.d_attn)                      # [BS, NK, d]
            V2 = V.reshape(BS, -1, self.d_attn)
            scores = torch.bmm(Q2, K2.transpose(1, 2)) / math.sqrt(self.d_attn)
            attn = F.softmax(scores, dim=-1)
            out2 = torch.bmm(attn, V2)                               # [BS, NQ, d]
            out = out2.reshape(*Q.shape)
        return out.to(out_dtype)

    def precompute_kv(self, kv: Tensor) -> tuple[Tensor, Tensor]:
        """Project KV once per window. Used by `cross_attn` (KV = window
        hiddens, identical across all j and all hops). Returns bf16 tensors
        on CUDA so downstream `forward_with_kv` stays in bf16.

        Args:
            kv: [BS, NK, d_kv]  — no J dim. Typically window hiddens.
        Returns:
            K, V: each [BS, NK, d_attn]
        """
        with torch.autocast(
            device_type="cuda" if kv.is_cuda else "cpu",
            dtype=torch.bfloat16, enabled=kv.is_cuda,
        ):
            K = self.W_k(kv)
            V = self.W_v(kv)
        return K, V

    def forward_with_kv(self, q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        """Per-hop attention reusing precomputed K, V (no J broadcast,
        no W_k / W_v re-projection). Returns dtype matching `q`.

        Args:
            q: [BS, J, d_q] or [BS, *Q, d_q]
            K, V: [BS, NK, d_attn] (from `precompute_kv`)
        Returns:
            [BS, *Q, d_attn]
        """
        out_dtype = q.dtype
        with torch.autocast(
            device_type="cuda" if q.is_cuda else "cpu",
            dtype=torch.bfloat16, enabled=q.is_cuda,
        ):
            Q = self.W_q(q)                                          # [BS, *Q, d_attn]
            BS = q.shape[0]
            Q2 = Q.reshape(BS, -1, self.d_attn)                      # [BS, NQ, d]
            scores = torch.bmm(Q2, K.transpose(1, 2)) / math.sqrt(self.d_attn)
            attn = F.softmax(scores, dim=-1)
            out2 = torch.bmm(attn, V)                                # [BS, NQ, d]
            out = out2.reshape(*Q.shape)
        return out.to(out_dtype)


class EntryProjector(nn.Module):
    """Shared entry-point projection used by BOTH read and write modules.

    Hopfield-tied keys: write deposits into the manifold at slot
    `argmax(W·pooled)`, and the next window's read retrieves at the same
    `argmax(W·pooled')` — when the underlying hiddens are the same window
    (write[d] and read[d+1] both pool window d), shared `W` makes the
    address agree by construction. This guarantees write's parameters get
    gradient through the read's gather.

    Owns `head_query` and `entry_mlp`. Surprise is intentionally NOT an
    input here — it controls *write strength* (step_mlp / mutate_mlp), not
    write *location*. Putting surprise in the address would re-introduce
    read/write divergence.
    """

    def __init__(self, cfg: TrajMemConfig):
        super().__init__()
        D = cfg.D_concept
        d_lm = cfg.d_lm
        self.J = cfg.J
        self.D = D
        # std=1.0 to dominate the projected pooled term so J trajectories
        # don't collapse to identical paths at init.
        self.head_query = nn.Parameter(torch.empty(cfg.J, D))
        # Init std lowered from 1.0 → 0.1 (2026-05-12). At std=1.0 the
        # per-head bias norm (~sqrt(D)≈16) overwhelmed mean-pooled LM
        # content (norm ~sqrt(D/T_window)≈1) by ~16× → entry routing
        # was mostly head-bias-driven at init, with content as a small
        # perturbation. Lower std makes content dominate from the start;
        # heads still differ enough at init for J trajectories to diverge.
        nn.init.normal_(self.head_query, std=0.1)
        self.entry_mlp = nn.Sequential(
            nn.Linear(d_lm + D, D, bias=True),
            nn.GELU(),
            nn.Linear(D, D, bias=True),
        )

    def forward(self, pooled: Tensor) -> Tensor:
        """pooled: [BS, d_lm] → Q_entry: [BS, J, D_concept]."""
        BS, d_lm = pooled.shape
        J, D = self.J, self.D
        pooled_j = pooled.unsqueeze(1).expand(BS, J, d_lm)        # [BS, J, d_lm]
        hq = self.head_query.unsqueeze(0).expand(BS, J, D)        # [BS, J, D]
        return self.entry_mlp(torch.cat([pooled_j, hq], dim=-1))  # [BS, J, D]


class ReadTrajectoryGenerator(nn.Module):
    """Generates J parallel read trajectories per window.

    Forward:
        prev_window_hiddens: [BS, T_window, d_lm]
        current_states:      [BS, N, D_concept] — manifold state at window start
        manifold:            Manifold instance (for concept_ids, edge_indices)

        returns visited:     [BS, J, K_read, D_concept]
                visited_ids: [BS, J, K_read] (int64) — for telemetry/debug
    """

    def __init__(
        self, cfg: TrajMemConfig, *, entry_proj: "EntryProjector | None" = None,
    ):
        super().__init__()
        self.cfg = cfg
        D = cfg.D_concept
        d_lm = cfg.d_lm

        # Entry projection: shared with the write module when constructed
        # via IntegratedLM (Hopfield-tied keys); standalone when None
        # (test-only). Sharing the projection makes write deposit at the
        # address that the next window's read will retrieve from.
        self.entry_proj = entry_proj if entry_proj is not None else EntryProjector(cfg)

        # Per-hop attention modules:
        d_attn = D
        self.history_attn = _CrossAttn(d_q=D, d_kv=D, d_attn=d_attn)
        self.cross_attn = _CrossAttn(d_q=D, d_kv=d_lm, d_attn=d_attn)

        # Per-hop step MLP: takes (current_state, history_attn_out, cross_attn_out)
        # produces a query in id-space for the neighbor scoring.
        self.step_mlp = nn.Sequential(
            nn.Linear(D * 3, D, bias=True),
            nn.GELU(),
            nn.Linear(D, D, bias=True),
        )

        # Positional encoding for the visited list (used in history_attn keys).
        # Buffer so it moves to GPU with the module.
        # Size K_read: indexed up to pos_enc[:K_read] in the longest hop.
        pe = make_pos_enc(cfg.K_read, D) * cfg.pos_enc_scale
        self.register_buffer("pos_enc", pe, persistent=False)

        # Learnable logit scale (CLIP-style). Cosine routing produces logits
        # in [-1, 1]; without scaling, softmax is near-uniform and routing
        # can't depend on the cosine signal. effective_scale =
        # exp(logit_scale_raw).clamp(max=20) so the optimizer can sharpen
        # routing as training proceeds.
        self.logit_scale_raw = nn.Parameter(
            torch.tensor(float(cfg.logit_scale_init))
        )

    def forward(
        self,
        prev_window_hiddens: Tensor,
        current_states: Tensor,
        manifold: Manifold,
        *,
        hard: bool = True,
        tau: Tensor | float | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Generate J parallel trajectories.

        Args:
            prev_window_hiddens: [BS, T_window, d_lm]
            current_states:      [BS, N, D_concept]
            manifold:            Manifold (concept_ids + edge_indices)
            hard:                Gumbel-STE if True, argmax if False
            tau:                 Gumbel-softmax temperature override. None
                                 falls back to cfg.gumbel_tau (1.0). Trainer
                                 passes a scheduled value (with a floor of
                                 0.5) — see Phase1Trainer's _current_tau.

        Returns:
            visited:     [BS, J, K_read, D_concept]
            visited_ids: [BS, J, K_read] int64
            aux:         {'load_balance': scalar, 'z_loss': scalar}
                         summed across the entry routing and all K_read
                         hops. The trainer multiplies by coefficients and
                         adds to the main loss before backward.
        """
        cfg = self.cfg
        BS, T, d_lm = prev_window_hiddens.shape
        D = cfg.D_concept
        J, K = cfg.J, cfg.K_read
        N = cfg.N
        assert current_states.shape == (BS, N, D)
        # Tau: leave as tensor if caller passed a tensor (dynamo treats it as
        # a dynamic input → 1 compile reused forever). Only cast to float if
        # absent (use cfg default — also a Python scalar, but compiled once).
        tau_eff = cfg.gumbel_tau if tau is None else tau

        # ── 1. Pick J entry concepts ──────────────────────────────────
        pooled = prev_window_hiddens.mean(dim=1)                  # [BS, d_lm]
        Q_entry = self.entry_proj(pooled)                         # [BS, J, D]

        # Precompute cross_attn K,V on prev_window_hiddens — identical for
        # all j and all K_read hops in this window. Avoids re-projecting
        # K/V at each hop and avoids the J-broadcast materialization that
        # `per_j_attn` would force.
        cross_K, cross_V = self.cross_attn.precompute_kv(prev_window_hiddens)

        # Score each entry query against all N concept_ids.
        # L2-normalize both Q and K → cosine similarity, bounded ∈ [-1, 1].
        # Decouples entry selection from raw vector magnitudes so the
        # Gumbel τ schedule operates on a stable, interpretable scale.
        # concept_ids_normed: [N, D] → broadcast over BS, J.
        ids = manifold.concept_ids_normed                         # [N, D]
        entry_logits_raw = torch.einsum(
            "bjd,nd->bjn", F.normalize(Q_entry, dim=-1), ids,
        )                                                         # [BS, J, N] ∈ [-1,1]
        # Scale cosine logits so signal dominates softmax temperature.
        # exp(logit_scale_raw).clamp(max=20) — learnable, init exp(1.5)≈4.5.
        eff_scale = self.logit_scale_raw.exp().clamp(max=20.0)
        entry_logits = entry_logits_raw * eff_scale
        entry_one_hot, entry_idx = softmax_top1_ste(
            entry_logits, hard=hard,
        )                                                         # [BS, J, N], [BS, J]
        # Accumulate routing aux losses (Switch load-balance + ST-MoE z-loss).
        # z_loss on the raw (unscaled) cosine logits so it doesn't oppose
        # logit_scale_raw learning to grow.
        _aux_lb = routing_aux_losses(
            entry_logits, entry_one_hot, z_loss_logits=entry_logits_raw,
        )
        aux_lb_total = _aux_lb["load_balance"]
        aux_z_total = _aux_lb["z_loss"]

        # ── 2. K_read autoregressive hops ────────────────────────────
        current = entry_idx                                       # [BS, J] int (for next neighbor lookup)
        # Gather initial state via the one-hot mixture so gradient flows
        # back to entry_logits → entry_mlp → head_query.
        # entry_one_hot: [BS, J, N]; current_states: [BS, N, D] → [BS, J, D].
        # In forward, one_hot is one-hot, so this picks current_states[entry_idx];
        # in backward, gradient flows via the soft probs (Gumbel-STE).
        current_state = torch.einsum(
            "bjn,bnd->bjd", entry_one_hot, current_states,
        )                                                         # [BS, J, D]

        # Pre-allocate visited list. We accumulate via list-append for
        # readability; could be replaced with a pre-allocated tensor for
        # torch.compile friendliness if profiling motivates.
        visited_list: list[Tensor] = []
        visited_id_list: list[Tensor] = []

        for t in range(K):
            # Append current to visited (raw state, no mutation in read path).
            visited_list.append(current_state)                    # [BS, J, D]
            visited_id_list.append(current)                       # [BS, J]

            # Build keys/values for history attention from visited so far.
            # visited so far has length t+1 (including the just-appended).
            # We attend on visited[:t+1] + pos_enc[:t+1].
            hist_kv = torch.stack(visited_list, dim=2)            # [BS, J, t+1, D]
            pos = self.pos_enc[: t + 1].unsqueeze(0).unsqueeze(1) # [1, 1, t+1, D]
            hist_kv = hist_kv + pos                               # broadcast

            # history_attn: query per j attends over its own visited list.
            history_attn_out = per_j_attn(
                self.history_attn, current_state, hist_kv,
            )                                                     # [BS, J, D]

            # cross_attn: query per j attends over the same prev_window_hiddens.
            # Use the shared-KV fast path — no J broadcast, K/V already cached.
            cross_attn_out = self.cross_attn.forward_with_kv(
                current_state, cross_K, cross_V,
            )                                                     # [BS, J, D]

            # Step MLP → query in id-space.
            step_input = torch.cat(
                [current_state, history_attn_out, cross_attn_out], dim=-1,
            )                                                     # [BS, J, 3D]
            Q_t = self.step_mlp(step_input)                       # [BS, J, D]

            # Score against neighbors of current[j]. Cosine similarity (L2-
            # normalized) — same rationale as the entry-point scoring.
            nbr_idx = manifold.get_neighbor_indices(current)      # [BS, J, K_max]
            nbr_ids = manifold.concept_ids_normed[nbr_idx]        # [BS, J, K_max, D]
            nbr_logits_raw = torch.einsum(
                "bjd,bjkd->bjk", F.normalize(Q_t, dim=-1), nbr_ids,
            )                                                     # [BS, J, K_max] ∈ [-1,1]
            nbr_logits = nbr_logits_raw * eff_scale
            nbr_one_hot, next_local = softmax_top1_ste(
                nbr_logits, hard=hard,
            )                                                     # [BS, J, K_max], [BS, J]
            _aux_hop = routing_aux_losses(
                nbr_logits, nbr_one_hot, z_loss_logits=nbr_logits_raw,
            )
            aux_lb_total = aux_lb_total + _aux_hop["load_balance"]
            aux_z_total = aux_z_total + _aux_hop["z_loss"]

            # Map local neighbor index to absolute concept id (int, for next
            # iter's neighbor lookup).
            next_global = torch.gather(
                nbr_idx, dim=2, index=next_local.unsqueeze(-1),
            ).squeeze(-1)                                         # [BS, J]
            current = next_global

            # Differentiable next-state: soft gather over neighbor states by
            # the Gumbel-STE one-hot. Gradient flows back to nbr_logits →
            # Q_t → step_mlp / history_attn / cross_attn / current_state.
            nbr_states_full = manifold.gather_states(
                current_states, nbr_idx,
            )                                                     # [BS, J, K_max, D]
            current_state = torch.einsum(
                "bjk,bjkd->bjd", nbr_one_hot, nbr_states_full,
            )                                                     # [BS, J, D]

        visited = torch.stack(visited_list, dim=2)                # [BS, J, K, D]
        visited_ids = torch.stack(visited_id_list, dim=2)         # [BS, J, K]
        # Average aux losses over the (entry + K_read) routing decisions so
        # the magnitude doesn't depend on K_read. Trainer applies the
        # coefficient on top.
        n_routes = 1 + K
        aux = {
            "load_balance": aux_lb_total / n_routes,
            "z_loss": aux_z_total / n_routes,
        }
        return visited, visited_ids, aux
