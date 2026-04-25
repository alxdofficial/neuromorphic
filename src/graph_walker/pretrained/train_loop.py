"""Cycle orchestrator: bootstrap → (phase-1 AR ↔ phase-2 GRPO) cycles.

Mirrors the v2 cycle loop with adapted freeze surfaces:
- Bootstrap (parallel phase-1):  full surface (walker + W_in/W_out/scale).
- Cycle phase-1 AR:              full surface; AR unroll trains the walker
                                 to use prefix writes for continuation logits.
- Cycle phase-2 GRPO:            phase-2 minimum surface (`memory.neuromod.*`
                                 only). Routes hard-Categorical, REINFORCE
                                 on rewards.

Caller owns the data iterators (production data pipelines vary too much
for a single CLI). The orchestrator just sequences stages and constructs
fresh optimizers per stage so stale Adam momentum from now-frozen params
doesn't leak across stages.

Per-stage logging is written via `StatsCollector` if `telemetry` is provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

from src.graph_walker.pretrained.llm_wrapper import GraphWalkerPretrainedLM
from src.graph_walker.pretrained.train_phase1 import (
    Phase1Batch,
    phase1_pretrained_step,
)
from src.graph_walker.pretrained.train_phase1_ar import (
    Phase1ARBatch,
    phase1_ar_pretrained_step,
)
from src.graph_walker.pretrained.train_phase2 import grpo_step
from src.graph_walker.telemetry import StatsCollector


@dataclass
class CycleConfig:
    work_dir: str
    bootstrap_steps: int = 1000
    cycles: int = 5
    cycle_phase1_steps: int = 500
    cycle_phase2_steps: int = 500

    # Optimizer
    lr: float = 1e-4
    grad_clip: float = 1.0

    # Phase-2 GRPO
    grpo_K: int = 8
    grpo_rollout_len: int = 128
    grpo_temperature: float = 1.0
    grpo_top_p: float = 1.0
    grpo_adv_std_floor: float = 1e-3


def _make_optimizer(wrapper: GraphWalkerPretrainedLM, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [p for _, p in wrapper.trainable_parameters()], lr=lr,
    )


def run_cycle_loop(
    wrapper: GraphWalkerPretrainedLM,
    bootstrap_iter: Iterable[Phase1Batch],
    cycle_p1_iter: Iterable[Phase1ARBatch],
    cycle_p2_iter: Iterable[tuple[torch.Tensor, torch.Tensor]],   # (prefix_ids[1,T_pre], reference_cont[L])
    *,
    reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    cfg: CycleConfig,
    on_step: Callable[[dict], None] | None = None,
) -> None:
    """Run the bootstrap → cycles training loop.

    Data iterators are caller-owned. They must yield enough items to cover
    `bootstrap_steps + cycles * (cycle_phase1_steps + cycle_phase2_steps)`.
    Iterators that return fewer items terminate the corresponding stage early.
    """
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    boot_iter = iter(bootstrap_iter)
    p1_iter = iter(cycle_p1_iter)
    p2_iter = iter(cycle_p2_iter)
    global_step = 0

    with StatsCollector(work_dir=work_dir) as collector:
        # ---- BOOTSTRAP ----
        wrapper.unfreeze_all()
        opt = _make_optimizer(wrapper, cfg.lr)
        for _ in range(cfg.bootstrap_steps):
            try:
                batch = next(boot_iter)
            except StopIteration:
                break
            stats = phase1_pretrained_step(wrapper, opt, batch)
            row = collector.snapshot(
                wrapper, step=global_step, phase="bootstrap", stats=stats,
            )
            if on_step is not None:
                on_step(row)
            global_step += 1

        # ---- CYCLES ----
        for cyc in range(cfg.cycles):
            # Phase-1 AR.
            wrapper.unfreeze_all()
            opt = _make_optimizer(wrapper, cfg.lr)
            for _ in range(cfg.cycle_phase1_steps):
                try:
                    batch = next(p1_iter)
                except StopIteration:
                    break
                stats = phase1_ar_pretrained_step(wrapper, opt, batch)
                row = collector.snapshot(
                    wrapper, step=global_step,
                    phase=f"cycle{cyc}-p1ar", stats=stats,
                )
                if on_step is not None:
                    on_step(row)
                global_step += 1

            # Phase-2 GRPO.
            wrapper.freeze_all_but_E_bias_and_neuromod()
            opt = _make_optimizer(wrapper, cfg.lr)
            for _ in range(cfg.cycle_phase2_steps):
                try:
                    prefix, reference = next(p2_iter)
                except StopIteration:
                    break
                stats = grpo_step(
                    wrapper, opt,
                    prefix_ids=prefix, reference_cont=reference,
                    reward_fn=reward_fn,
                    num_rollouts=cfg.grpo_K,
                    gen_length=cfg.grpo_rollout_len,
                    temperature=cfg.grpo_temperature,
                    top_p=cfg.grpo_top_p,
                    adv_std_floor=cfg.grpo_adv_std_floor,
                    grad_clip=cfg.grad_clip,
                )
                row = collector.snapshot(
                    wrapper, step=global_step,
                    phase=f"cycle{cyc}-p2grpo", stats=stats,
                )
                if on_step is not None:
                    on_step(row)
                global_step += 1


if __name__ == "__main__":
    raise NotImplementedError(
        "run_cycle_loop is library code — wire your own data loaders + reward "
        "function and call it from a script. See "
        "src/graph_walker/pretrained/__init__.py for the public API."
    )
