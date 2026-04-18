"""Phase 1 (bootstrap) training loop — Gumbel-softmax backprop on top of a
frozen pretrained LM.

Per-step contract:
    1. batch = next(data_iter)  — provides input_ids [BS, T] + target_ids
    2. wrapper.forward(input_ids) returns LM output + stashes mem_pred_loss
    3. loss = CE(logits, target_ids) + mem_pred_weight * mem_pred_loss
    4. loss.backward()
    5. clip grads
    6. optimizer.step()
    7. wrapper.detach_memory() for TBPTT boundary
    8. anneal Gumbel tau if requested

No autocast wrapping here — callers pass a wrapper already in the right
dtype (bf16 on CUDA, fp32 on CPU). The phase-1 run script sets this up.

This module is intentionally small. No telemetry JSONL, no checkpointing,
no distributed. Call it from a driver script that adds those layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
import torch.nn.functional as F

from src.pretrained.llm_wrapper import PretrainedLMWithMemory


@dataclass
class Phase1Batch:
    input_ids: torch.Tensor          # [BS, T]
    target_ids: torch.Tensor         # [BS, T] — typically input_ids shifted
    prev_token: torch.Tensor | None  # [BS] — token at position -1 (for valid-mask)


@dataclass
class Phase1StepLog:
    step: int
    loss: float
    ce: float
    mem_pred_loss: float
    gumbel_tau: float
    grad_norm: float


def anneal_gumbel_tau(step: int, total_steps: int, tau_start: float,
                      tau_end: float) -> float:
    """Linear anneal from tau_start down to tau_end across `total_steps`.
    Clamped to tau_end after saturation."""
    if total_steps <= 0:
        return tau_end
    frac = min(1.0, step / total_steps)
    return tau_start + (tau_end - tau_start) * frac


def run_phase1(
    wrapper: PretrainedLMWithMemory,
    optimizer: torch.optim.Optimizer,
    data_iter: Iterator[Phase1Batch],
    *,
    steps: int,
    mem_pred_weight: float = 0.1,
    max_grad_norm: float = 1.0,
    gumbel_tau_start: float = 1.0,
    gumbel_tau_end: float = 0.3,
    anneal_across_steps: int | None = None,
    on_step: Callable[[Phase1StepLog], None] | None = None,
):
    """Run phase-1 training for `steps` optimizer steps.

    `anneal_across_steps` defaults to `steps`. Set it shorter if you want tau
    to hit its floor before the end of training (matches the from-scratch
    run's schedule where tau saturates before lr decay completes).
    """
    anneal_across_steps = anneal_across_steps or steps
    wrapper.train()

    for step in range(steps):
        batch = next(data_iter)
        input_ids = batch.input_ids
        target_ids = batch.target_ids
        prev_token = batch.prev_token

        tau = anneal_gumbel_tau(step, anneal_across_steps,
                                gumbel_tau_start, gumbel_tau_end)
        wrapper.memory.gumbel_tau = tau

        kwargs = {}
        if prev_token is not None:
            kwargs["prev_token"] = prev_token
        out = wrapper(input_ids, **kwargs)

        ce = F.cross_entropy(
            out.logits.reshape(-1, out.logits.shape[-1]).float(),
            target_ids.reshape(-1))
        mem_loss = wrapper._last_mem_pred_loss
        if mem_loss is None:
            mem_loss = torch.zeros((), device=ce.device)
        loss = ce + mem_pred_weight * mem_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainable = [p for _, p in wrapper.trainable_parameters()]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm).item()
        optimizer.step()
        wrapper.detach_memory()

        log = Phase1StepLog(
            step=step,
            loss=float(loss.item()),
            ce=float(ce.item()),
            mem_pred_loss=float(mem_loss.item() if torch.is_tensor(mem_loss) else 0.0),
            gumbel_tau=tau,
            grad_norm=float(grad_norm) if math.isfinite(grad_norm) else float("inf"),
        )
        if on_step is not None:
            on_step(log)
