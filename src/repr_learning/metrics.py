"""Training-health metrics for repr_learning.

Ports the relevant subset of `src/trajectory_memory_v2.trainer.V2RetrievalMetrics`
to the single-window representation-learning setup. Drops fields that
don't apply (R↔W overlap, edge-memory state, walker diagnostics, QA
accuracy) and adds V2.1-specific ones (modifier_delta_norm).

Public surface:
- `ReprMetrics`: dataclass of per-step metrics, JSON-serializable via asdict()
- `compute_metrics()`: build a ReprMetrics from a forward+backward step
- `module_grad_norm()`: per-module grad norm helper
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ReprMetrics:
    """Per-step health metrics for representation-learning training."""

    # Step bookkeeping
    step: int = 0
    variant: str = ""

    # Loss components — universal
    loss: float = 0.0
    loss_recon: float = 0.0
    loss_aux: float = 0.0
    loss_orth: float = 0.0           # V2.1 codebook orthogonality penalty
    loss_z: float = 0.0              # V2.1 router z-loss (logit-magnitude cap)
    modifier_delta_clip_frac: float = 0.0  # fraction of V2.1 modifier deltas clipped
    grad_norm: float = 0.0

    # Memory token health — universal (all 5 variants produce 96 mem tokens)
    mem_rms_mean: float = 0.0        # per-token RMS, averaged over tokens + batch
    mem_dispersion: float = 0.0      # mean off-diag cosine within sample (lower = better)

    # Per-module gradient norms (variant-dependent; 0 if module absent)
    grad_norm_encoder: float = 0.0   # bi-transformer + slot_attn + slot queries
    grad_norm_codebook: float = 0.0  # concept_id table (V2.1 + A)
    grad_norm_modifier: float = 0.0  # src/dst modifier MLPs (V2.1)
    grad_norm_proj: float = 0.0      # projection MLPs (all variants)
    grad_norm_mask_embed: float = 0.0
    grad_norm_mamba: float = 0.0     # Mamba blocks (recurrent_baseline)

    # Routing health (V2.1 + A: discrete codebook variants)
    routing_entropy: float = 0.0     # avg softmax entropy over picks (nats)
    unique_codes_per_batch: int = 0  # distinct codebook entries picked across batch

    # Codebook health (V2.1 + A): canary for collapse / runaway scales
    codebook_norm_mean: float = 0.0  # mean ||concept_id[i]||
    codebook_norm_cv: float = 0.0    # std/mean (low = uniform norms, high = imbalance)
    codebook_pairwise_cos: float = 0.0  # mean off-diag cosine of concept vectors

    # V2.1-specific
    modifier_delta_norm: float = 0.0  # mean ||modifier(...)|| residual delta
    edge_vec_norm: float = 0.0        # mean ||edge_vec||

    # Throughput
    step_ms: float = 0.0
    text_tok_per_sec: float = 0.0


def module_grad_norm(module: Optional[nn.Module]) -> float:
    """L2 grad norm across all parameters of a module. 0 if module is None
    or has no parameters with gradients."""
    if module is None:
        return 0.0
    sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            sq += p.grad.detach().float().pow(2).sum().item()
    return math.sqrt(sq)


def _param_grad_norm(p: Optional[nn.Parameter]) -> float:
    if p is None or p.grad is None:
        return 0.0
    return p.grad.detach().float().norm().item()


def _memory_dispersion(memory: Tensor) -> float:
    """Mean off-diagonal cosine similarity within each sample's memory tokens.
    Lower = more dispersed (good); higher = collapsed (bad). Returns 0.0 for
    variants that emit zero or one memory token (NullEncoder / vanilla_llama)
    where dispersion is undefined."""
    B, M, _ = memory.shape
    if M < 2:
        return 0.0
    m = F.normalize(memory.detach().float(), dim=-1)   # [B, M, d]
    cos = m @ m.transpose(1, 2)                         # [B, M, M]
    mask = ~torch.eye(M, dtype=torch.bool, device=cos.device)
    return cos[:, mask].mean().item()


def _codebook_health(codebook: Tensor, sample_n: int = 256) -> tuple[float, float, float]:
    """Returns (norm_mean, norm_cv, pairwise_cos).

    Pairwise cosine sampled from sample_n random entries to avoid O(N^2)
    where N is typically 4096.
    """
    cb = codebook.detach().float()
    norms = cb.norm(dim=-1)                             # [N]
    norm_mean = norms.mean().item()
    norm_cv = (norms.std() / norms.mean().clamp(min=1e-8)).item()

    N = cb.shape[0]
    n = min(sample_n, N)
    idx = torch.randperm(N, device=cb.device)[:n]
    sample = F.normalize(cb[idx], dim=-1)               # [n, d]
    cos = sample @ sample.T                             # [n, n]
    mask = ~torch.eye(n, dtype=torch.bool, device=cb.device)
    pairwise_cos = cos[mask].mean().item()

    return norm_mean, norm_cv, pairwise_cos


def compute_metrics(
    model: nn.Module,
    out: dict,
    memory: Tensor,
    aux: dict,
    grad_norm: float,
    step: int,
    variant: str,
    step_ms: float,
    text_tok_per_sec: float,
) -> ReprMetrics:
    """Build a ReprMetrics from a completed forward+backward step.

    Call AFTER backward() and AFTER grad clipping (so grad_norm matches
    the value the optimizer sees) but BEFORE optimizer.step() (so grads
    are still on parameters).

    Args:
        model: the ReprLearningModel
        out: forward output dict (has loss, loss_recon, loss_aux)
        memory: [B, M, d_llama] memory tokens
        aux: encoder's aux dict (variant-specific keys)
        grad_norm: post-clip grad norm scalar
        step: current training step
        variant: variant name string
        step_ms: wall-time of this step in milliseconds
        text_tok_per_sec: throughput estimate

    Returns:
        ReprMetrics with all applicable fields populated.
    """
    m = ReprMetrics(step=step, variant=variant)

    # Universal loss + grad
    m.loss = float(out["loss"].item() if isinstance(out["loss"], Tensor) else out["loss"])
    m.loss_recon = float(out["loss_recon"].item() if isinstance(out["loss_recon"], Tensor) else out["loss_recon"])
    m.loss_aux = float(out["loss_aux"].item() if isinstance(out["loss_aux"], Tensor) else out["loss_aux"])
    if "loss_orth" in out:
        m.loss_orth = float(out["loss_orth"].item() if isinstance(out["loss_orth"], Tensor) else out["loss_orth"])
    if "loss_z" in out:
        m.loss_z = float(out["loss_z"].item() if isinstance(out["loss_z"], Tensor) else out["loss_z"])
    if "modifier_delta_clip_frac" in aux:
        v = aux["modifier_delta_clip_frac"]
        m.modifier_delta_clip_frac = float(v.item() if isinstance(v, Tensor) else v)
    m.grad_norm = float(grad_norm)
    m.step_ms = float(step_ms)
    m.text_tok_per_sec = float(text_tok_per_sec)

    # Memory token health
    with torch.no_grad():
        rms = memory.detach().float().pow(2).mean(dim=-1).sqrt()
        m.mem_rms_mean = rms.mean().item()
        m.mem_dispersion = _memory_dispersion(memory)

    # Per-module gradient norms — probe by attribute name where available
    enc = model.encoder
    m.grad_norm_encoder = (
        module_grad_norm(getattr(enc, "bi_transformer", None))
        + module_grad_norm(getattr(enc, "slot_attn", None))
        + _param_grad_norm(getattr(enc, "edge_queries", None))
        + _param_grad_norm(getattr(enc, "code_queries", None))
        + _param_grad_norm(getattr(enc, "cont_queries", None))
    )
    # Codebook (V2.1 + A) lives as a Parameter at enc.concept_id
    if hasattr(enc, "concept_id"):
        m.grad_norm_codebook = _param_grad_norm(enc.concept_id)
    # Modifier MLPs (V2.1 only)
    if hasattr(enc, "src_modifier"):
        m.grad_norm_modifier = (
            module_grad_norm(enc.src_modifier) + module_grad_norm(enc.dst_modifier)
        )
    # Projection MLPs — sum across all proj_* attributes the variant has
    proj_sq = 0.0
    for name in ("proj_src", "proj_dst", "proj_edge", "proj_code", "proj_cont",
                 "proj_value", "proj_to_llama"):
        sub = getattr(enc, name, None)
        if sub is not None:
            proj_sq += module_grad_norm(sub) ** 2
    m.grad_norm_proj = math.sqrt(proj_sq)
    # mask_embed lives on the decoder
    m.grad_norm_mask_embed = _param_grad_norm(
        getattr(model.decoder, "mask_embed", None),
    )
    # Mamba blocks
    if hasattr(enc, "mamba_blocks"):
        mamba_sq = sum(module_grad_norm(b) ** 2 for b in enc.mamba_blocks)
        m.grad_norm_mamba = math.sqrt(mamba_sq)

    # Routing entropy + unique codes (V2.1 + A)
    if "routing_entropy" in aux:
        m.routing_entropy = float(aux["routing_entropy"].item())
    if "picked_ids" in aux:
        m.unique_codes_per_batch = int(aux["picked_ids"].unique().numel())

    # Codebook health (V2.1 + A)
    if hasattr(enc, "concept_id"):
        norm_mean, norm_cv, pairwise_cos = _codebook_health(enc.concept_id)
        m.codebook_norm_mean = norm_mean
        m.codebook_norm_cv = norm_cv
        m.codebook_pairwise_cos = pairwise_cos

    # V2.1-specific
    if "modifier_delta_norm" in aux:
        m.modifier_delta_norm = float(aux["modifier_delta_norm"].item())
    if "edge_vec_norm" in aux:
        m.edge_vec_norm = float(aux["edge_vec_norm"].item())

    return m


def metrics_to_jsonl_row(m: ReprMetrics) -> dict:
    """Convert ReprMetrics to a JSONL-serializable dict."""
    return asdict(m)
