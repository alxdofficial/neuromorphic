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
