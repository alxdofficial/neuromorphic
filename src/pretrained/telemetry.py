"""Telemetry helpers for the pretrained training loops.

Wraps the existing memory-graph helper methods (`compute_memory_health`,
`compute_plasticity_rates`, `compute_component_grad_norms`,
`compute_param_norms`, `compute_lane_divergence`) and adds the pivot-
specific pieces the memory module can't know about:

    - W_in / W_out / scale grad norms and param stats
    - scale's mean/std/min/max (catches silent collapse toward zero
      — the failure mode from the v9 mem_gate bug)
    - codebook live-code count (dead-code detection)
    - throughput (tok/s, ms/step, peak VRAM)
    - RL-specific reward / advantage / log_pi_sum distribution stats

Each collector returns a flat dict of float-valued metrics ready to
concat into a JSONL record. Callers opt in per log step — the full
collection does a couple of CUDA syncs and isn't free.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch
from torch import Tensor


def collect_inject_stats(wrapper) -> dict:
    """MemInjectLayer grad + param stats (pivot-specific trainables)."""
    mi = wrapper.mem_inject
    out: dict[str, float] = {}
    for name, p in (("W_in", mi.W_in.weight),
                    ("W_out", mi.W_out.weight),
                    ("scale", mi.scale)):
        out[f"{name}_norm"] = p.detach().float().norm().item()
        out[f"grad_{name}"] = (0.0 if p.grad is None
                                else p.grad.detach().float().norm().item())
    # Scale distribution. Catches slow collapse toward zero (= memory
    # contribution fading out silently).
    s = mi.scale.detach().float()
    out["scale_mean"] = s.mean().item()
    out["scale_std"] = s.std().item()
    out["scale_min"] = s.min().item()
    out["scale_max"] = s.max().item()
    out["scale_abs_mean"] = s.abs().mean().item()
    return out


def collect_codebook_stats(wrapper, usage_threshold: float = 1e-3) -> dict:
    """Dead-code detection. `usage_count` is maintained by
    `DiscreteActionPolicy.update_usage` during phase-1 Gumbel training."""
    if wrapper.memory is None:
        return {}
    pol = wrapper.memory.discrete_policy
    usage = pol.usage_count.detach().float()
    total = pol.usage_total.item() if pol.usage_total.numel() > 0 else 0.0
    K = usage.numel()
    live = (usage / max(total, 1.0) > usage_threshold).sum().item()
    return {
        "codebook_K": int(K),
        "codebook_live_codes": int(live),
        "codebook_dead_codes": int(K - live),
        "codebook_usage_total": float(total),
        "codebook_usage_max": usage.max().item(),
        "codebook_usage_std": usage.std().item(),
    }


def collect_memory_stats(wrapper, include_slow: bool = True) -> dict:
    """Memory graph health + plasticity + per-component grad/param norms.

    `include_slow=False` skips `compute_memory_health` + `compute_plasticity_rates`
    (both do CUDA syncs over every state tensor). Use for per-step logs
    where only cheap stats are needed.
    """
    if wrapper.memory is None:
        return {}
    mem = wrapper.memory
    out: dict[str, float] = {}
    out.update(mem.compute_component_grad_norms())
    out.update(mem.compute_param_norms())
    if include_slow:
        out.update(mem.compute_memory_health())
        out.update(mem.compute_plasticity_rates())
        out.update(mem.compute_lane_divergence())
    return out


def collect_throughput_stats(
    tok_per_s: float | None = None,
    ms_per_step: float | None = None,
    device: torch.device | None = None,
) -> dict:
    """Throughput + VRAM. Peak VRAM is measured on CUDA via
    `torch.cuda.max_memory_allocated`; caller is responsible for
    resetting the peak counter at checkpoint/eval boundaries if they
    want per-segment numbers (by default it's lifetime-max)."""
    out: dict[str, float] = {}
    if tok_per_s is not None:
        out["tok_s"] = float(tok_per_s)
    if ms_per_step is not None:
        out["ms_step"] = float(ms_per_step)
    if device is not None and device.type == "cuda":
        out["vram_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return out


def collect_lr_stats(optimizer: torch.optim.Optimizer) -> dict:
    """LR per param group. Phase-1 uses one group but the legacy trainer
    uses four; logging the list prevents silent LR-schedule drift from
    being invisible when pool LRs diverge."""
    lrs = [g["lr"] for g in optimizer.param_groups]
    out = {f"lr_g{i}": float(lr) for i, lr in enumerate(lrs)}
    out["lr"] = float(lrs[0])  # convenience: "lr" = first group
    return out


def summarize_tensor(x: Tensor, *, prefix: str) -> dict:
    """`{prefix}_min/max/mean/std/abs_mean` for a 1-D tensor of rewards
    / advantages / log_pi. Used by GRPO telemetry."""
    xf = x.detach().float()
    return {
        f"{prefix}_min": xf.min().item(),
        f"{prefix}_max": xf.max().item(),
        f"{prefix}_mean": xf.mean().item(),
        f"{prefix}_std": xf.std().item() if xf.numel() > 1 else 0.0,
        f"{prefix}_abs_mean": xf.abs().mean().item(),
    }


def collect_rollout_stats(generated: Tensor, reference: Tensor) -> dict:
    """Quality diagnostics on a [K, gen_length] rollout matrix. Catches
    degenerate generation: e.g., the model emits the same token K times
    (repetition rate = 1.0), or all K rollouts converge to the same
    token stream (`rollout_token_agreement` = 1.0 → no divergence to
    drive REINFORCE variance)."""
    K, L = generated.shape
    if L == 0:
        return {"rollout_K": K, "rollout_gen_len": 0}
    # Repetition rate: fraction of adjacent token pairs (within a
    # rollout) that are identical. 0 = all distinct, 1 = constant.
    if L > 1:
        reps = (generated[:, 1:] == generated[:, :-1]).float().mean().item()
    else:
        reps = 0.0
    # Rollout disagreement at each position, averaged.
    # position-wise unique count / K → 1.0 if rollouts agree, ~1/K if not.
    agreement = (generated == generated[0:1]).float().mean().item()
    # Match rate to reference (same as token_match_reward).
    if reference.dim() == 1 and reference.shape[0] == L:
        match = (generated == reference.unsqueeze(0)).float().mean(dim=-1)
        match_mean = match.mean().item()
        match_std = match.std().item() if K > 1 else 0.0
    else:
        match_mean = float("nan")
        match_std = float("nan")
    return {
        "rollout_K": int(K),
        "rollout_gen_len": int(L),
        "rollout_repetition_rate": reps,
        "rollout_agreement_rate": agreement,
        "rollout_match_reference_mean": match_mean,
        "rollout_match_reference_std": match_std,
    }


# ----------------------------------------------------------------------
# JSONL writer
# ----------------------------------------------------------------------


class JsonlLogger:
    """Append-only JSONL writer with a flush on every write. Safe to call
    from training loops — the cost is a single syscall per step. Creates
    the parent directory on first write."""

    def __init__(self, path: str | None):
        self.path = path
        self._opened = False
        if path is not None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def write(self, record: dict[str, Any]):
        if self.path is None:
            return
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def close(self):
        pass  # we open-close per write so there's nothing persistent
