"""Bootstrap + iterative-cycle training driver for Llama + memory graph.

Mirrors the old `src/train_loop.py` on `abandoned/main-v2` but adapted
for the frozen-LM setup:

    1. Bootstrap phase-1 — one-time, ~500M tokens. Everything in the
       memory graph is trainable (modulator, codebook, decoder, dynamics)
       plus W_in / W_out / scale. Llama is always frozen on this branch.
       Uses the parallel teacher-forced loop (`run_phase1`) because it's
       faster and enough to shape the codebook.

    2. Per cycle (×N):
        a. Cycle phase-1 — ~10M tokens. `freeze_codebook_decoder()` pins
           the code vocabulary and the ΔW emission so phase-2's trained
           modulator doesn't see a moving target. Uses the autoregressive
           unrolled loop (`run_phase1_ar`) so memory learns to rely on
           the writes it made during the prefix in order to predict the
           continuation — the actual long-horizon-memory signal.
        b. Cycle phase-2 — ~40M tokens. `freeze_all_but_logit_head()`
           locks everything except the modulator's final linear. GRPO
           rollouts with hard Categorical sampling produce REINFORCE
           gradient on the code policy, stabilized by the minimal
           trainable surface.

The three sub-loops (`run_phase1`, `run_phase1_ar`, `grpo_step`) remain
fully runnable standalone for debugging; this driver just interleaves them.

Usage:
    python -m src.pretrained.train_loop \\
        --work-dir outputs/pretrained_cycle \\
        --bootstrap-steps 5000 --cycles 5 \\
        --cycle-phase1-steps 1000 --cycle-phase2-steps 2000
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass


@dataclass
class CycleConfig:
    work_dir: str
    bootstrap_steps: int = 5000
    cycles: int = 5
    cycle_phase1_steps: int = 1000
    cycle_phase2_steps: int = 2000
    bs: int = 16
    # Autoregressive unroll split: T_pre tokens teacher-forced prefix, then
    # T_cont tokens unrolled one-at-a-time. Total sequence length per batch
    # is T_pre + T_cont. Cap T_cont to control VRAM — each unroll step
    # holds a full forward graph until backward.
    T_pre: int = 256
    T_cont: int = 64
    grpo_K: int = 8
    grpo_rollout_len: int = 128
    lr: float = 1e-4


def parse_args() -> CycleConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--bootstrap-steps", type=int, default=5000)
    p.add_argument("--cycles", type=int, default=5)
    p.add_argument("--cycle-phase1-steps", type=int, default=1000)
    p.add_argument("--cycle-phase2-steps", type=int, default=2000)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--T-pre", type=int, default=256)
    p.add_argument("--T-cont", type=int, default=64)
    p.add_argument("--grpo-K", type=int, default=8)
    p.add_argument("--grpo-rollout-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    ns = p.parse_args()
    return CycleConfig(
        work_dir=ns.work_dir,
        bootstrap_steps=ns.bootstrap_steps,
        cycles=ns.cycles,
        cycle_phase1_steps=ns.cycle_phase1_steps,
        cycle_phase2_steps=ns.cycle_phase2_steps,
        bs=ns.bs,
        T_pre=ns.T_pre,
        T_cont=ns.T_cont,
        grpo_K=ns.grpo_K,
        grpo_rollout_len=ns.grpo_rollout_len,
        lr=ns.lr,
    )


def run_cycle_loop(
    wrapper,
    bootstrap_iter,
    cycle_p1_iter,
    cycle_p2_iter,
    reward_fn,
    cfg: CycleConfig,
    *,
    log: callable = print,
):
    """Programmatic entrypoint for the cycle loop. Callers provide:

      - `wrapper`: a `PretrainedLMWithMemory` already on the target device.
      - `bootstrap_iter`: yields `Phase1Batch` for the bootstrap phase.
      - `cycle_p1_iter`: yields `Phase1ARBatch` for cycle phase-1.
      - `cycle_p2_iter`: yields (prefix_ids, reference_cont) pairs for
        cycle phase-2 GRPO.
      - `reward_fn`: (generated, reference) → rewards [K]. Used by grpo_step.

    Nothing about this function is file-system-aware; callers pass in the
    already-constructed iterators. That keeps the driver testable and
    avoids duplicating data-loading plumbing.
    """
    import torch
    from src.pretrained.train_phase1 import run_phase1
    from src.pretrained.train_phase1_ar import run_phase1_ar
    from src.pretrained.train_phase2 import grpo_step

    os.makedirs(cfg.work_dir, exist_ok=True)
    metrics_path = os.path.join(cfg.work_dir, "metrics.jsonl")
    opt = torch.optim.AdamW(
        [p for _, p in wrapper.trainable_parameters()], lr=cfg.lr)

    # --- 1. Bootstrap ---
    log(f"\n=== BOOTSTRAP — {cfg.bootstrap_steps} steps, full trainable surface ===")
    wrapper.unfreeze_all()
    wrapper.current_phase = "phase1"
    wrapper.reset_memory(bs=cfg.bs)
    run_phase1(wrapper, opt, bootstrap_iter, steps=cfg.bootstrap_steps,
               metrics_path=metrics_path,
               on_step=lambda lg: log(
                   f"  [boot {lg.step:>5}] loss={lg.loss:.3f} "
                   f"ce={lg.ce:.3f} mem={lg.mem_pred_loss:.3f} "
                   f"tau={lg.gumbel_tau:.2f} |g|={lg.grad_norm:.2f}") if lg.step % 50 == 0 else None)

    # --- 2. Cycles ---
    for c in range(cfg.cycles):
        log(f"\n=== CYCLE {c + 1}/{cfg.cycles} ===")

        # 2a. Cycle phase-1: AR unroll with codebook+decoder frozen.
        log(f"--- cycle {c + 1} phase-1 (AR unroll, "
            f"{cfg.cycle_phase1_steps} steps, codebook+decoder frozen) ---")
        wrapper.unfreeze_all()
        wrapper.freeze_codebook_decoder()
        wrapper.current_phase = "phase1"
        # Rebuild optimizer so frozen params don't carry Adam momentum that
        # would re-apply the moment new trainables come online in later stages.
        opt = torch.optim.AdamW(
            [p for _, p in wrapper.trainable_parameters()], lr=cfg.lr)
        wrapper.reset_memory(bs=cfg.bs)
        run_phase1_ar(wrapper, opt, cycle_p1_iter,
                      steps=cfg.cycle_phase1_steps,
                      metrics_path=metrics_path,
                      on_step=lambda lg: log(
                          f"  [c{c+1} p1 {lg.step:>5}] loss={lg.loss:.3f} "
                          f"tau={lg.gumbel_tau:.2f} |g|={lg.grad_norm:.2f}"
                      ) if lg.step % 50 == 0 else None)

        # 2b. Cycle phase-2: GRPO with only the modulator's logit_head trainable.
        log(f"--- cycle {c + 1} phase-2 (GRPO on logit_head only, "
            f"{cfg.cycle_phase2_steps} steps, K={cfg.grpo_K}) ---")
        wrapper.unfreeze_all()
        wrapper.freeze_all_but_logit_head()
        opt = torch.optim.AdamW(
            [p for _, p in wrapper.trainable_parameters()], lr=cfg.lr)
        for p2_step in range(cfg.cycle_phase2_steps):
            prefix, ref = next(cycle_p2_iter)
            lg = grpo_step(
                wrapper, opt,
                prefix_ids=prefix, reference_cont=ref,
                num_rollouts=cfg.grpo_K,
                gen_length=cfg.grpo_rollout_len,
                reward_fn=reward_fn,
                metrics_path=metrics_path,
                step_idx=p2_step,
            )
            if p2_step % 50 == 0:
                log(f"  [c{c+1} p2 {p2_step:>5}] loss={lg.loss:.3f} "
                    f"r_mean={lg.reward_mean:.3f} r_std={lg.reward_std:.3f} "
                    f"log_pi={lg.log_pi_mean:.2f} A_max={lg.advantage_max_abs:.2f}")

    # Restore a neutral training surface so callers that continue to use
    # the wrapper after the cycle loop aren't stuck in the final
    # phase-2 freeze configuration.
    wrapper.unfreeze_all()
    log("\n=== CYCLE LOOP COMPLETE ===")


if __name__ == "__main__":
    _cfg = parse_args()
    raise NotImplementedError(
        "The CLI entry point requires a data-loader wiring step. Import "
        "`run_cycle_loop` from this module and pass your own iterators + "
        "reward_fn — there is no single data pipeline baked in yet. See "
        "`docs/pretrained_lm_memory.md` for the contract.")
