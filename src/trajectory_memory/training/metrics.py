"""Per-step / per-batch diagnostics for live training monitoring.

Functions are cheap (~µs to ms): suitable to call every step.
Heavier diagnostics (loss-by-source, trajectory diversity histograms)
live in the trainer entrypoints since they need access to dataset
metadata.
"""

from __future__ import annotations

from typing import Iterable

import torch

from src.trajectory_memory.integrated_lm import IntegratedLM


# Module-name → display name used in plots / JSON dump.
_COMPONENT_PREFIXES: list[tuple[str, str]] = [
    # (prefix, label). First match wins; order matters because
    # `llama.layers.<L>.mem_inject.*` would otherwise be classified as
    # generic llama.
    ("llama.model.layers.", "bridge_in_llama"),  # MemInjectLayer adapter
    ("manifold.", "manifold"),
    ("read_module.", "read"),
    ("read_attn.", "read"),
    ("write_module.", "write"),
    ("llama.", "llama_other"),                    # frozen, but in case anything trainable lands here
]


def classify_param(name: str) -> str:
    """Map a named-parameter key to a component label."""
    for prefix, label in _COMPONENT_PREFIXES:
        if name.startswith(prefix):
            return label
    return "other"


def grad_norms_by_component(model: IntegratedLM) -> dict[str, float]:
    """Compute per-component gradient L2 norm in a single pass.

    Call AFTER backward, BEFORE optimizer.step (so .grad is populated
    and not zeroed). Returns dict {component_label: l2_norm}.

    Components without any grad-bearing params return 0.0.
    """
    sums: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None or not p.requires_grad:
            continue
        label = classify_param(name)
        # Sum of squares; sqrt at end.
        sums[label] = sums.get(label, 0.0) + float(p.grad.detach().pow(2).sum())
    return {label: total ** 0.5 for label, total in sums.items()}


def param_norms_by_component(model: IntegratedLM) -> dict[str, float]:
    """Per-component parameter L2 norm — for tracking weight-scale drift
    over the course of training (especially useful for the bridge,
    where a divergence here is an early "model is going off the rails"
    signal)."""
    sums: dict[str, float] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        label = classify_param(name)
        sums[label] = sums.get(label, 0.0) + float(p.detach().pow(2).sum())
    return {label: total ** 0.5 for label, total in sums.items()}


def trajectory_diversity_stats(
    visited_ids: torch.Tensor, N: int,
) -> dict[str, float]:
    """Stats over which concepts a trajectory visited.

    Args:
        visited_ids: int64 tensor of visited concept indices, any shape.
        N: total number of concepts in the manifold (for % unique).

    Returns:
        dict with:
          unique_count: int — distinct concepts visited across all trajectories
          unique_frac:  float — unique_count / N
          self_overlap_rate: float — fraction of (j, t) pairs that revisit
                            an already-visited concept within the same
                            trajectory (degenerate routing if high)
    """
    flat = visited_ids.detach().reshape(-1).cpu()
    if flat.numel() == 0:
        return {"unique_count": 0, "unique_frac": 0.0, "self_overlap_rate": 0.0}
    unique = flat.unique().numel()

    # Self-overlap: per (BS, J) trajectory, what fraction of hops revisit?
    # Reshape if visited_ids has structure [BS, J, K]; otherwise estimate.
    if visited_ids.dim() == 3:
        BS, J, K = visited_ids.shape
        revisit_count = 0
        total_hops = BS * J * K
        for b in range(BS):
            for j in range(J):
                seen = set()
                for k in range(K):
                    cid = int(visited_ids[b, j, k])
                    if cid in seen:
                        revisit_count += 1
                    else:
                        seen.add(cid)
        self_overlap = revisit_count / max(total_hops, 1)
    else:
        self_overlap = 0.0

    return {
        "unique_count": unique,
        "unique_frac": unique / max(N, 1),
        "self_overlap_rate": self_overlap,
    }


def surprise_stats(surprises: torch.Tensor) -> dict[str, float]:
    """Mean / std / min / max of per-window surprise scalars.

    `surprises` is typically [BS] or [BS, D]. We flatten to a 1-D
    distribution before reducing.
    """
    if surprises.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    s = surprises.detach().float().reshape(-1)
    return {
        "mean": float(s.mean()),
        "std": float(s.std(unbiased=False)) if s.numel() > 1 else 0.0,
        "min": float(s.min()),
        "max": float(s.max()),
    }


def vram_stats() -> dict[str, float]:
    """Current allocated VRAM (GB) and peak since last reset."""
    if not torch.cuda.is_available():
        return {"alloc_gb": 0.0, "peak_gb": 0.0}
    return {
        "alloc_gb": torch.cuda.memory_allocated() / 1e9,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def routing_entropy(
    visited_ids: torch.Tensor, N: int,
) -> float:
    """Visit-distribution entropy across the N concepts.

    For each concept c in [0, N), count how often it appears in
    `visited_ids`. Compute entropy = -Σ p(c) log p(c) where p(c) is the
    visit fraction. Returns entropy in nats.

    Use as a routing-collapse detector. Healthy memory routing has
    high entropy (visits spread across many concepts). Pathological
    collapse (routing always picks the same K concepts) has low
    entropy. For uniform visits across all N concepts, entropy is
    log(N) ≈ 8.3 for N=4096. For collapse onto 4 concepts, entropy
    is log(4) ≈ 1.4.

    Returns 0.0 if visited_ids is empty.
    """
    flat = visited_ids.detach().reshape(-1)
    if flat.numel() == 0:
        return 0.0
    counts = torch.bincount(flat.cpu(), minlength=N).float()
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    # entropy = -Σ p log p, ignoring p=0 (0*log0 = 0)
    nonzero = p > 0
    entropy = -(p[nonzero] * p[nonzero].log()).sum()
    return float(entropy)


def concept_state_drift(
    current_states: torch.Tensor,        # [BS, N, D] OR [N, D]
    state_init: torch.Tensor,            # [N, D]
) -> dict[str, float]:
    """Per-concept drift from state_init. Detects manifold runaway —
    if mean drift grows unboundedly, the manifold is exploding (writes
    aren't being well-regulated). If max drift is much greater than
    mean, a few concepts are absorbing all the writes (over-attractor).

    Returns:
        mean_drift: average L2(current[c] - init[c]) across concepts
        max_drift:  max L2 drift across concepts
    """
    if current_states.dim() == 3:
        # [BS, N, D] → average over batch first
        current = current_states.mean(dim=0)
    else:
        current = current_states
    delta = (current - state_init).detach()    # [N, D]
    per_concept_l2 = delta.float().pow(2).sum(dim=-1).sqrt()    # [N]
    return {
        "mean_drift": float(per_concept_l2.mean()),
        "max_drift": float(per_concept_l2.max()),
    }


def effective_lr_by_component(
    grad_norms: dict[str, float],
    param_norms: dict[str, float],
    lr_per_group: list[float],
) -> dict[str, float]:
    """Effective LR = (lr · ||grad||) / ||param|| per component. The
    "fractional update size" each component sees per step. Healthy is
    1e-5 to 1e-3; >1e-2 is unstable; <1e-7 is not learning.

    Uses lr_per_group[0] as the effective LR for all components (a
    coarse approximation — components actually map to either group 0
    [memory] or group 1 [adapter] depending on `build_optimizer`'s
    classification, but for plotting purposes the order-of-magnitude
    is what matters).
    """
    if not lr_per_group:
        return {}
    lr = lr_per_group[0]
    out: dict[str, float] = {}
    for label, gn in grad_norms.items():
        pn = param_norms.get(label, 0.0)
        if pn > 0:
            out[label] = (lr * gn) / pn
        else:
            out[label] = 0.0
    return out


def logit_stats(logits: torch.Tensor) -> dict[str, float]:
    """Per-token logit std + range. Detects logit collapse (std → 0)
    or explosion (huge max-min range). Healthy logits have std ~1-3
    and range a few × std. Pre-softmax."""
    if logits.numel() == 0:
        return {"std": 0.0, "range": 0.0, "max": 0.0}
    flat = logits.detach().float().reshape(-1)
    return {
        "std": float(flat.std()),
        "range": float(flat.max() - flat.min()),
        "max": float(flat.max()),
    }
