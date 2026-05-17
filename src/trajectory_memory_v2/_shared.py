"""Shared routing/attention primitives used by the walker, read, write, and
integrated_lm modules.

Inlined from the v1 `trajectory_memory` package (read_module.py and
integrated_lm.py) so the v2 architecture is self-contained. The originals
are preserved on the `abandoned/trajectory-memory-v1` branch.

Public surface:
- `routing_aux_losses` — Switch load-balance + ST-MoE z-loss
- `softmax_top1_ste`   — argmax with straight-through estimator
- `per_j_attn`         — per-trajectory cross-attn helper
- `CrossAttention`     — single-head cross-attn with precomputed-KV fast path
- `EntryProjector`     — Hopfield-tied entry projector (shared read/write)
- `TrajectoryReadAttn` — per-token Llama-hidden → trajectory cross-attn
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.trajectory_memory_v2.config import TrajMemV2Config


# ── Routing aux losses ────────────────────────────────────────────────────


def routing_aux_losses(
    logits: Tensor,
    one_hot: Tensor,
    *,
    z_loss_logits: Tensor | None = None,
) -> dict[str, Tensor]:
    """Switch-Transformer load-balance + ST-MoE z-loss at a routing decision.

    Args:
        logits:        [..., N] differentiable routing scores (typically post
                       scaling by `eff_scale`).
        one_hot:       [..., N] hard one-hot selection from softmax-STE.
        z_loss_logits: optional override for the z-loss input — pass the
                       UNSCALED logits when the routing logits have been
                       multiplied by a learnable scale, so z_loss penalizes
                       raw magnitudes (per ST-MoE) without fighting the
                       scale parameter.

    Returns:
        {
          'load_balance': scalar — Switch loss N · Σᵢ fᵢ · Pᵢ; 1 at uniform,
                          grows linearly with concentration.
          'z_loss':       scalar — mean of logsumexp(z_loss_logits)²; keeps
                          raw routing magnitudes from blowing up.
        }
    """
    N = logits.shape[-1]
    flat_logits = logits.reshape(-1, N)
    flat_oh = one_hot.reshape(-1, N)
    if flat_logits.shape[0] == 0:
        z = (flat_logits * 0.0).sum()
        return {"load_balance": z, "z_loss": z}

    P = F.softmax(flat_logits, dim=-1).mean(dim=0)
    f = flat_oh.detach().mean(dim=0)
    load_balance = N * (f * P).sum()

    z_logits = z_loss_logits if z_loss_logits is not None else logits
    z_flat = z_logits.reshape(-1, N)
    z_loss = (torch.logsumexp(z_flat, dim=-1) ** 2).mean()

    return {"load_balance": load_balance, "z_loss": z_loss}


def softmax_top1_ste(
    logits: Tensor,
    *,
    hard: bool,
    noise_std: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Softmax-top-1 with straight-through estimator.

    Forward picks argmax; backward routes through softmax probabilities.

    `noise_std > 0` adds Gaussian noise to logits before softmax (Shazeer
    2017 noisy gating). `hard=False` returns the argmax one-hot without an
    STE path — safe inside `no_grad`, but cuts gradient into routing if
    used during training.
    """
    if noise_std > 0:
        logits = logits + noise_std * torch.randn_like(logits)
    soft = F.softmax(logits, dim=-1)
    idx = soft.argmax(dim=-1)
    hard_one_hot = F.one_hot(idx, num_classes=logits.shape[-1]).to(soft.dtype)
    if not hard:
        return hard_one_hot, idx
    one_hot = (hard_one_hot - soft).detach() + soft
    return one_hot, idx


# ── Cross-attention ───────────────────────────────────────────────────────


def per_j_attn(attn_module: "CrossAttention", q: Tensor, kv: Tensor) -> Tensor:
    """Apply cross-attention separately for each j in the J dim.

    Folds J into BS so per-trajectory attention does not mix across j.

    Args:
        attn_module: a `CrossAttention` instance.
        q:  [BS, J, d_q]
        kv: [BS, J, NK, d_kv]
    Returns:
        [BS, J, d_q] — attention readout per (BS, J).
    """
    BS, J, D = q.shape
    d_kv = kv.shape[-1]
    NK = kv.shape[-2]
    q_flat = q.reshape(BS * J, 1, D)
    kv_flat = kv.reshape(BS * J, NK, d_kv)
    out = attn_module(q_flat, kv_flat)
    return out.reshape(BS, J, D)


class CrossAttention(nn.Module):
    """Single-head cross-attention. Q in d_q, KV in d_kv, both project to d.

    Two use sites:
    - history_attn: Q = current state, KV = visited list + pos_enc. Use
                    `forward(q, kv)` (KV genuinely differs per trajectory).
    - cross_attn:   Q = current state, KV = window hiddens. KV is the SAME
                    for all j and all hops; call `precompute_kv(kv)` once
                    per window, then `forward_with_kv(q, K, V)` per hop.

    Both fast paths run in bf16 inside autocast on CUDA. Single-head is
    sufficient for the small trajectory hops; bump to multi-head if
    benching shows lossy attention.
    """

    def __init__(self, d_q: int, d_kv: int, d_attn: int):
        super().__init__()
        self.d_attn = d_attn
        self.W_q = nn.Linear(d_q, d_attn, bias=False)
        self.W_k = nn.Linear(d_kv, d_attn, bias=False)
        self.W_v = nn.Linear(d_kv, d_attn, bias=False)
        for m in (self.W_q, self.W_k, self.W_v):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, q: Tensor, kv: Tensor, key_mask: Tensor | None = None) -> Tensor:
        """key_mask: [BS, NK] bool — True = real, False = pad. If given,
        pad-position scores are masked to -inf before softmax so they
        contribute zero to attention output."""
        out_dtype = q.dtype
        with torch.autocast(
            device_type="cuda" if q.is_cuda else "cpu",
            dtype=torch.bfloat16, enabled=q.is_cuda,
        ):
            Q = self.W_q(q)
            K = self.W_k(kv)
            V = self.W_v(kv)
            BS = q.shape[0]
            Q2 = Q.reshape(BS, -1, self.d_attn)
            K2 = K.reshape(BS, -1, self.d_attn)
            V2 = V.reshape(BS, -1, self.d_attn)
            scores = torch.bmm(Q2, K2.transpose(1, 2)) / math.sqrt(self.d_attn)
            if key_mask is not None:
                # key_mask: [BS_outer, NK]; broadcast to [BS, NQ, NK].
                scores = scores.masked_fill(
                    ~key_mask.reshape(BS, 1, -1), float("-inf"),
                )
            attn = F.softmax(scores, dim=-1)
            out2 = torch.bmm(attn, V2)
            out = out2.reshape(*Q.shape)
        return out.to(out_dtype)

    def precompute_kv(self, kv: Tensor) -> tuple[Tensor, Tensor]:
        with torch.autocast(
            device_type="cuda" if kv.is_cuda else "cpu",
            dtype=torch.bfloat16, enabled=kv.is_cuda,
        ):
            K = self.W_k(kv)
            V = self.W_v(kv)
        return K, V

    def forward_with_kv(
        self, q: Tensor, K: Tensor, V: Tensor, key_mask: Tensor | None = None,
    ) -> Tensor:
        out_dtype = q.dtype
        with torch.autocast(
            device_type="cuda" if q.is_cuda else "cpu",
            dtype=torch.bfloat16, enabled=q.is_cuda,
        ):
            Q = self.W_q(q)
            BS = q.shape[0]
            Q2 = Q.reshape(BS, -1, self.d_attn)
            scores = torch.bmm(Q2, K.transpose(1, 2)) / math.sqrt(self.d_attn)
            if key_mask is not None:
                # Guard against rows where ALL keys are pad — softmax over
                # all -inf is NaN. Force such rows to attend over all
                # positions (best-effort fallback, contributes whatever
                # the projected pad-hiddens encode but doesn't poison).
                any_valid = key_mask.any(dim=-1, keepdim=True)        # [BS, 1]
                safe_key_mask = key_mask | (~any_valid)               # [BS, NK]
                scores = scores.masked_fill(
                    ~safe_key_mask.reshape(BS, 1, -1), float("-inf"),
                )
            attn = F.softmax(scores, dim=-1)
            out2 = torch.bmm(attn, V)
            out = out2.reshape(*Q.shape)
        return out.to(out_dtype)


# ── Entry projector (Hopfield-tied between read and write) ────────────────


class EntryProjector(nn.Module):
    """Shared entry-point projection used by BOTH read and write modules.

    Hopfield-tied: write deposits at slot `argmax(W·pooled)` and the next
    window's read retrieves at the same argmax — when the underlying
    hiddens come from the same window, shared `W` makes the addresses
    agree by construction, so write's parameters get gradient through the
    read's gather.
    """

    def __init__(self, cfg: TrajMemV2Config):
        super().__init__()
        D = cfg.D_concept
        d_lm = cfg.d_lm
        self.J = cfg.J
        self.D = D
        self.head_query = nn.Parameter(torch.empty(cfg.J, D))
        # Scale init by 1/√D (Xavier-style). With std=0.1 fixed, the J
        # head biases would dominate the entry_mlp content at init for
        # large D (magnitude ~√D × 0.1 = 3.2 at D=1024), collapsing the
        # J trajectories toward the same direction before training. The
        # 1/√D scale keeps the head bias magnitude unit-order regardless
        # of D, leaving room for the content signal to differentiate
        # trajectories quickly.
        nn.init.normal_(self.head_query, std=1.0 / math.sqrt(D))
        self.entry_mlp = nn.Sequential(
            nn.Linear(d_lm + D, D, bias=True),
            nn.GELU(),
            nn.Linear(D, D, bias=True),
        )
        # Small-init router (Switch / GShard inspiration). Default Xavier
        # over-concentrated routing onto ~9% of cells; std=0.05 softens
        # initial routing without zeroing entry_mlp outputs.
        nn.init.normal_(self.entry_mlp[0].weight, std=0.05)
        nn.init.zeros_(self.entry_mlp[0].bias)
        nn.init.normal_(self.entry_mlp[-1].weight, std=0.05)
        nn.init.zeros_(self.entry_mlp[-1].bias)

    def forward(self, pooled: Tensor) -> Tensor:
        """pooled: [BS, d_lm] → Q_entry: [BS, J, D_concept]."""
        BS, d_lm = pooled.shape
        J, D = self.J, self.D
        pooled_j = pooled.unsqueeze(1).expand(BS, J, d_lm)
        hq = self.head_query.unsqueeze(0).expand(BS, J, D)
        return self.entry_mlp(torch.cat([pooled_j, hq], dim=-1))


# ── Per-token Llama-hidden → trajectory cross-attn ────────────────────────


class TrajectoryReadAttn(nn.Module):
    """Cross-attention from Llama hidden (in D_concept space) to the read
    trajectory KV. Invoked inside MemInjectLayer's `memory_fn` closure.
    """

    def __init__(self, D_concept: int):
        super().__init__()
        self.D = D_concept
        self.W_q = nn.Linear(D_concept, D_concept, bias=False)
        self.W_k = nn.Linear(D_concept, D_concept, bias=False)
        self.W_v = nn.Linear(D_concept, D_concept, bias=False)
        for m in (self.W_q, self.W_k, self.W_v):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, h_mem: Tensor, read_trajectory: Tensor) -> Tensor:
        """
        Args:
            h_mem:           [BS, T, D_concept]    Llama hidden in memory space
            read_trajectory: [BS, J*K_read, D_concept]
        Returns:
            readout: [BS, T, D_concept]
        """
        Q = self.W_q(h_mem)
        K = self.W_k(read_trajectory)
        V = self.W_v(read_trajectory)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.D)
        attn = F.softmax(scores, dim=-1)
        return torch.bmm(attn, V)
