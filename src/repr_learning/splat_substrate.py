"""Exp 3: Gaussian Splat substrate.

Memory is a fixed-K signed Gaussian mixture in a shared latent space:
  ρ(x) = Σ_k  s_k · w_k · N(x; μ_k, diag(σ_k²))
with K constant (bounded memory) and s_k ∈ (-1, +1) via tanh.

Writes (per 1024-token window): a small TransformerUpdater takes
(encoded pins + current blob state) → outputs target blob parameters
for the next call. 4 calls per 4096-token chunk; blob state evolves
across calls within a chunk.

Reads (per Llama position): the probe emits K_rays directions and one
origin. Each ray's response is the closed-form line integral of the
signed density along the ray. K_rays scalars per position form a vector
response, injected into Llama via a forward pre-hook.

Numerics:
  - Diagonal Σ throughout (full Σ at d=256 is O(d³) — non-starter).
  - Log-space arithmetic; the (2π)^{-(d-1)/2} prefactor is dropped
    (constant across (ray, blob), absorbed by the learned W_out).
  - Signed-sum stability: subtract max log-term before exp.
  - Aux losses pre-normalized so coefficient = relative importance.

See docs/exp3_gaussian_splat_baseline.md for the design.
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

def init_splat_state(
    B: int, K: int, d: int,
    device: torch.device, dtype: torch.dtype,
) -> dict:
    """Initial blob state.

    Naming convention (subtle — read carefully):
      `log_diag_sigma` stores **log σ²** (log of the variance, i.e.
      log of the diagonal entries of Σ). So exp(log_diag_sigma) = σ²
      and exp(0.5 * log_diag_sigma) = σ (std dev). The "sigma" in the
      name refers to Σ (the covariance), not σ (std dev) — Σ_ii = σ²_i.
    """
    # μ_k ~ N(0, 1/d) — small random
    mu = torch.randn(B, K, d, device=device, dtype=dtype) * (d ** -0.5)
    # log σ²_i ~ N(0, 0.1) — σ² ≈ 1 ⇒ σ ≈ 1 with small variance
    log_diag_sigma = torch.randn(B, K, d, device=device, dtype=dtype) * 0.1
    # w_raw such that softplus(w_raw) ≈ 1/K (uniform mass over K blobs)
    target_w = 1.0 / K
    # softplus^{-1}(y) = log(exp(y) - 1)
    w_raw = torch.full((B, K), math.log(math.exp(target_w) - 1.0),
                       device=device, dtype=dtype)
    # s_logit ~ N(0, 0.1) — signs near 0 (tanh ≈ 0), high-gradient region
    s_logit = torch.randn(B, K, device=device, dtype=dtype) * 0.1
    return {
        "mu": mu,
        "log_diag_sigma": log_diag_sigma,    # log σ² (not log σ!)
        "w_raw": w_raw,
        "s_logit": s_logit,
    }


def derive_blob_quantities(blobs: dict):
    """Derive quantities from raw parameters.

    `log_diag_sigma` stores log σ² (log of variance). So:
      diag_sigma_sq  = exp(log_diag_sigma)       = σ²    (variance)
      inv_diag_sigma_sq = exp(-log_diag_sigma)   = 1/σ²  (precision)
      log_det_sigma  = sum(log_diag_sigma)       = log|Σ|
    """
    mu = blobs["mu"]                                    # [B, K, d]
    log_diag_sigma = blobs["log_diag_sigma"]           # [B, K, d]  log σ²
    diag_sigma = torch.exp(log_diag_sigma)              # σ², element-wise
    inv_diag_sigma = torch.exp(-log_diag_sigma)         # 1/σ², element-wise
    log_det_sigma = log_diag_sigma.sum(dim=-1)         # [B, K]  Σ log σ²_i = log |Σ|
    w = F.softplus(blobs["w_raw"])                     # [B, K]  ≥ 0
    s = torch.tanh(blobs["s_logit"])                   # [B, K]  ∈ (-1, +1)
    return mu, diag_sigma, inv_diag_sigma, log_det_sigma, w, s


# ─────────────────────────────────────────────────────────────────────────
# Line-integral read (closed form)
# ─────────────────────────────────────────────────────────────────────────

def _log_kernel(
    o: Tensor,                  # [B, T, d]
    dirs: Tensor,               # [B, T, K_rays, d]    unit vectors
    mu: Tensor,                 # [B, K, d]
    inv_diag_sigma: Tensor,     # [B, K, d]
    log_det_sigma: Tensor,      # [B, K]
) -> Tensor:
    """Per-(batch, position, ray, blob) log line-integral kernel.

    log_kernel = -½ log|Σ| - ½ log(dᵀΣ⁻¹d) - ½ m²,
    where m² = (o-μ)ᵀΣ⁻¹(o-μ) - [(o-μ)ᵀΣ⁻¹d]²/(dᵀΣ⁻¹d).

    The shared (2π)^{-(d-1)/2} prefactor is dropped — constant across
    (ray, blob), absorbed by the downstream W_out.

    Returns: [B, T, K_rays, K]
    """
    # delta = o - μ — does not depend on ray
    delta = o.unsqueeze(2) - mu.unsqueeze(1)           # [B, T, K, d]
    delta_norm = delta * inv_diag_sigma.unsqueeze(1)   # [B, T, K, d]
    m_full = (delta * delta_norm).sum(dim=-1)          # [B, T, K]  (o-μ)ᵀΣ⁻¹(o-μ)

    # A = dᵀΣ⁻¹d — per ray, per blob
    A = (dirs.unsqueeze(-2) ** 2
         * inv_diag_sigma.unsqueeze(1).unsqueeze(1)).sum(dim=-1)
    # A: [B, T, K_rays, K]

    # δᵀΣ⁻¹d — per ray, per blob
    d_dot = (dirs.unsqueeze(-2) * delta_norm.unsqueeze(2)).sum(dim=-1)
    # d_dot: [B, T, K_rays, K]

    # m² = m_full - d_dot² / A
    A_safe = A.clamp(min=1e-8)
    m_squared = m_full.unsqueeze(2) - (d_dot ** 2) / A_safe
    m_squared = m_squared.clamp(min=0.0)               # numerical floor

    log_kernel = (
        -0.5 * log_det_sigma.unsqueeze(1).unsqueeze(1)   # [B, 1, 1, K]
        - 0.5 * torch.log(A_safe)                         # [B, T, K_rays, K]
        - 0.5 * m_squared                                 # [B, T, K_rays, K]
    )
    return log_kernel                                     # [B, T, K_rays, K]


def ray_features(
    o: Tensor,                  # [B, T, d]
    dirs: Tensor,               # [B, T, K_rays, d]
    blobs: dict,
) -> Tensor:
    """Per-ray features (4 channels):
        [I_total_norm, I_pos_norm, I_neg_norm, log_max_norm]

    The first three are signed-softmax-style attention over blobs (in (-K,+K]
    range). The fourth is `max_log / d` — the dominant blob's log-affinity,
    normalized by latent dim so it's O(1).

    Why normalized rather than reabsorbing exp(max_log) scale:
    - In d=256, m² is O(d)=~250 for random vectors at init
    - exp(max_log) underflows to literal 0 in fp32
    - Reabsorbing kills the signal AND the gradient
    - The normalized terms preserve the relative blob contributions and
      pass useful gradient back; max_log itself enters as a separate
      O(1) feature so absolute scale info isn't lost.

    Returns: [B, T, K_rays, 4]
    """
    mu, _diag, inv_diag, log_det, w, s = derive_blob_quantities(blobs)
    d = mu.shape[-1]

    log_kernel = _log_kernel(o, dirs, mu, inv_diag, log_det)  # [B, T, K_rays, K]

    # Per-blob signed weight; log of its magnitude
    sw = s * w                                                  # [B, K]
    abs_sw = sw.abs().clamp(min=1e-12)                          # [B, K]
    log_abs_sw = torch.log(abs_sw)                              # [B, K]
    sign_sw = torch.sign(sw)                                    # [B, K]  ∈ {-1, 0, +1}

    # log of |contribution_k| at each (B, T, K_rays, K)
    log_terms = log_abs_sw.unsqueeze(1).unsqueeze(1) + log_kernel  # [B, T, K_rays, K]

    # Subtract per-(B,T,K_rays) max before exp for stability
    max_log = log_terms.max(dim=-1, keepdim=True).values          # [B, T, K_rays, 1]
    # Detach max_log when subtracting — we don't want gradient flowing
    # through the max-selection (the typical softmax stability trick).
    norm_terms = torch.exp(log_terms - max_log.detach())          # [B, T, K_rays, K]

    sign_per_k = sign_sw.unsqueeze(1).unsqueeze(1)                # [B, 1, 1, K]
    pos_mask = (sign_per_k > 0).to(norm_terms.dtype)
    neg_mask = (sign_per_k < 0).to(norm_terms.dtype)

    # Sum positive and negative branches separately (NORMALIZED sums)
    pos_sum_norm = (norm_terms * pos_mask).sum(dim=-1)            # [B, T, K_rays]
    neg_sum_norm = (norm_terms * neg_mask).sum(dim=-1)
    total_sum_norm = pos_sum_norm - neg_sum_norm                  # signed, in (-K, +K]

    # max_log itself is O(d); normalize by d so it's O(1) as a feature.
    # (max_log carries the "how big is the strongest log-affinity" info.)
    max_log_feat = max_log.squeeze(-1) / float(d)                 # [B, T, K_rays]

    feats = torch.stack(
        [total_sum_norm, pos_sum_norm, neg_sum_norm, max_log_feat],
        dim=-1,
    )                                                              # [B, T, K_rays, 4]
    return feats


# ─────────────────────────────────────────────────────────────────────────
# Auxiliary losses (pre-normalized, coefficients = relative importance)
# ─────────────────────────────────────────────────────────────────────────

def density_at_points(points: Tensor, blobs: dict) -> Tensor:
    """ρ(p) = Σ_k s_k · w_k · N(p; μ_k, diag σ_k²).

    Used by L_pin (compute ρ at each pin). The same log-space technique:
    drop the (2π)^{-d/2} prefactor (constant); compute per-blob log-density
    minus its max for stability.

    points: [B, N, d]
    Returns: [B, N]  signed density at each point (in arbitrary units —
             the absolute scale is consistent across points so the L_pin
             margin/hinge still makes sense relatively).
    """
    mu, _diag, inv_diag, log_det, w, s = derive_blob_quantities(blobs)

    # δ = p - μ — [B, N, K, d]
    delta = points.unsqueeze(2) - mu.unsqueeze(1)
    delta_norm = delta * inv_diag.unsqueeze(1)
    m_sq = (delta * delta_norm).sum(dim=-1)                         # [B, N, K]

    # log N(p; μ_k, Σ_k) without (2π)^{-d/2} = -½ log|Σ| - ½ m²
    log_gauss = -0.5 * log_det.unsqueeze(1) - 0.5 * m_sq            # [B, N, K]

    sw = s * w                                                       # [B, K]
    abs_sw = sw.abs().clamp(min=1e-12)
    log_abs_sw = torch.log(abs_sw)
    sign_sw = torch.sign(sw)                                         # [B, K]

    log_terms = log_abs_sw.unsqueeze(1) + log_gauss                  # [B, N, K]
    max_log = log_terms.max(dim=-1, keepdim=True).values             # [B, N, 1]
    # Detach max_log in the subtraction (standard softmax stability trick:
    # the argmax index shouldn't carry gradient). The exp(max_log) on the
    # outside re-applies the absolute scale; that one is NOT detached, so
    # the gradient through the scale is preserved.
    norm_terms = torch.exp(log_terms - max_log.detach())             # [B, N, K]

    signed_norm = (sign_sw.unsqueeze(1) * norm_terms).sum(dim=-1)    # [B, N]
    rho = signed_norm * torch.exp(max_log).squeeze(-1)               # [B, N]
    return rho


def loss_pin(
    pins: Tensor,               # [B, N, d]
    pins_mask: Optional[Tensor],  # [B, N] True = real pin
    blobs: dict,
    margin: float = 1.0,
) -> Tensor:
    """L_pin = mean over real pins of softplus(margin - ρ(p)).

    A pin in negative density is penalized hardest. A pin in flat (ρ ≈ 0)
    region gets margin pressure. A pin already in a well-supported region
    (ρ > margin) is satisfied.
    """
    rho = density_at_points(pins, blobs)                             # [B, N]
    penalty = F.softplus(margin - rho)                               # [B, N]
    if pins_mask is not None:
        m = pins_mask.to(penalty.dtype)
        denom = m.sum().clamp(min=1.0)
        loss = (penalty * m).sum() / denom
    else:
        loss = penalty.mean()
    return loss


def loss_proportional(
    pins_mask: Optional[Tensor],   # [B, N]
    blobs: dict,
    target_mass_per_pin: float = 1.0,
) -> Tensor:
    """L_proportional = ((Σ_k s_k · w_k − N_pins) / N_pins)²

    Mass-balance: total signed blob mass should match the number of pins
    (under the unit target_mass_per_pin convention). Order-1 in normalized
    form.
    """
    _, _, _, _, w, s = derive_blob_quantities(blobs)
    blob_mass = (s * w).sum(dim=-1)                                  # [B]

    if pins_mask is not None:
        N = pins_mask.to(w.dtype).sum(dim=-1).clamp(min=1.0)         # [B]
    else:
        N = torch.full_like(blob_mass, w.shape[1])

    target = target_mass_per_pin * N
    rel_err = (blob_mass - target) / N                               # [B]
    return rel_err.pow(2).mean()


def loss_adjust(blobs_new: dict, blobs_old: dict) -> Tensor:
    """L_adjust = mass-weighted sum of W₂² between matched old/new blobs.

    For diagonal Σ:
        W₂²(N(μ₁, σ₁²), N(μ₂, σ₂²))  =  ‖μ₁ − μ₂‖²  +  ‖σ₁ − σ₂‖²

    γ_k = w_k_old — established blobs are stickier.
    Normalized by Σ_k γ_k so the coefficient is a pure importance weight.

    Plus a small per-blob (Δw)² term and (Δs_logit)² term.
    """
    _, _, _, _, w_old, s_old = derive_blob_quantities(blobs_old)
    _, _, _, _, w_new, s_new = derive_blob_quantities(blobs_new)

    mu_old = blobs_old["mu"]                                         # [B, K, d]
    mu_new = blobs_new["mu"]
    # log_diag_sigma stores log σ² (variance). exp(0.5 · log σ²) = σ (std dev).
    # The W₂² formula for diagonal Gaussians uses the per-dim std dev.
    sigma_old = torch.exp(0.5 * blobs_old["log_diag_sigma"])
    sigma_new = torch.exp(0.5 * blobs_new["log_diag_sigma"])

    # W₂² per blob for diagonal Σ:
    w2_mu = (mu_new - mu_old).pow(2).sum(dim=-1)                     # [B, K]
    w2_sigma = (sigma_new - sigma_old).pow(2).sum(dim=-1)            # [B, K]
    w2_per_blob = w2_mu + w2_sigma                                   # [B, K]

    dw = (w_new - w_old).pow(2)                                      # [B, K]
    ds = (s_new - s_old).pow(2)                                      # [B, K]

    gamma = w_old                                                    # [B, K]  stickiness
    denom = gamma.sum(dim=-1).clamp(min=1e-6)                        # [B]

    weighted = gamma * (w2_per_blob + dw + ds)                       # [B, K]
    per_batch = weighted.sum(dim=-1) / denom                         # [B]
    return per_batch.mean()


def loss_sign_saturation(blobs: dict, target: float = 0.3) -> Tensor:
    """Tiny penalty: keep tanh(s_logit) bounded away from ±1 to preserve
    gradient on the sign channel. Penalize tanh² exceeding `target`.
    """
    s = torch.tanh(blobs["s_logit"])
    excess = (s.pow(2) - target).clamp(min=0.0)
    return excess.mean()


# ─────────────────────────────────────────────────────────────────────────
# TransformerUpdater — learned blob updater
# ─────────────────────────────────────────────────────────────────────────

class TransformerUpdater(nn.Module):
    """Cross-attention updater. Takes encoded pins + current blobs,
    outputs target parameters for the next blob state.

    Architecture:
      - Encode old blobs as K tokens (concat μ, log_σ, w_raw, s_logit, project to d)
      - Add K learned positional embeddings (stable blob identity)
      - For n_layers:
          blob_tok  ← LayerNorm → cross-attn(blob_tok as Q, pins as KV) → +residual
          blob_tok  ← LayerNorm → self-attn(among blob_tok) → +residual
          blob_tok  ← LayerNorm → FFN → +residual
      - Per-blob output head: blob_tok → (Δμ, Δlog_σ, Δw_raw, Δs_logit)

    Initialization: output head's final Linear is small-init so the first
    forward call ≈ old blobs (near-identity behavior).
    """

    def __init__(
        self,
        d: int,
        K: int,
        n_layers: int = 3,
        n_heads: int = 4,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.d = d
        self.K = K
        self.n_layers = n_layers

        # Encode old blob params (μ, log_σ, w_raw, s_logit) → d
        # input dim per blob = d + d + 1 + 1 = 2d + 2
        self.blob_in_proj = nn.Linear(2 * d + 2, d)

        # K learned positional embeddings (stable blob identity)
        self.blob_pos = nn.Parameter(torch.randn(K, d) * (d ** -0.5))

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

        # Output head — per-blob target params: (μ, log_σ, w_raw, s_logit)
        # Init small so the first call's output is dominated by the
        # additive skip from the encoded old params (see forward).
        self.out_norm = nn.LayerNorm(d)
        self.out_head = nn.Linear(d, 2 * d + 2)
        nn.init.normal_(self.out_head.weight, std=0.01)
        nn.init.zeros_(self.out_head.bias)

    def forward(
        self,
        pins: Tensor,                          # [B, N, d]
        blobs_old: dict,
        pins_pad_mask: Optional[Tensor] = None,  # [B, N] True = padded
    ) -> dict:
        B, N, _ = pins.shape
        K = self.K

        # 1. Encode old blob params as token features
        old_concat = torch.cat([
            blobs_old["mu"],                                 # [B, K, d]
            blobs_old["log_diag_sigma"],                     # [B, K, d]
            blobs_old["w_raw"].unsqueeze(-1),                # [B, K, 1]
            blobs_old["s_logit"].unsqueeze(-1),              # [B, K, 1]
        ], dim=-1)                                           # [B, K, 2d+2]
        blob_tok = self.blob_in_proj(old_concat)             # [B, K, d]
        blob_tok = blob_tok + self.blob_pos.unsqueeze(0)     # add per-blob positional

        # 2. Layer stack
        for L in range(self.n_layers):
            # Cross-attention: blob_tok queries pins
            q = self.cross_norms[L](blob_tok)
            attn_out, _ = self.cross_attns[L](
                q, pins, pins,
                key_padding_mask=pins_pad_mask,
            )
            blob_tok = blob_tok + attn_out

            # Self-attention among blob tokens
            q = self.self_norms[L](blob_tok)
            attn_out, _ = self.self_attns[L](q, q, q)
            blob_tok = blob_tok + attn_out

            # FFN
            blob_tok = blob_tok + self.ffns[L](self.ffn_norms[L](blob_tok))

        # 3. Output head — per-blob deltas. Add to encoded old params
        # so that small-init delta ≈ identity on the input concat.
        delta = self.out_head(self.out_norm(blob_tok))       # [B, K, 2d+2]
        new_concat = old_concat + delta

        # Split target tensor back into (μ, log_σ, w_raw, s_logit)
        mu_new = new_concat[..., : self.d]
        log_diag_sigma_new = new_concat[..., self.d : 2 * self.d]
        w_raw_new = new_concat[..., 2 * self.d]
        s_logit_new = new_concat[..., 2 * self.d + 1]

        # Clamp log_σ for numerical stability (σ ∈ [e^{-3}, e^{3}] = [0.05, 20])
        log_diag_sigma_new = log_diag_sigma_new.clamp(min=-3.0, max=3.0)
        # Clamp s_logit so tanh stays in the high-gradient region (|tanh| ≲ 0.995)
        s_logit_new = s_logit_new.clamp(min=-3.0, max=3.0)

        return {
            "mu": mu_new,
            "log_diag_sigma": log_diag_sigma_new,
            "w_raw": w_raw_new,
            "s_logit": s_logit_new,
        }
