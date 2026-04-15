"""Outer orchestration driver for bootstrap + iterative cycle training.

Runs the full pipeline:
    1. Bootstrap phase 1 (one-time, ~500M tokens, everything trainable —
       codebook, decoder, logit head, LM, memory dynamics)
    2. For each cycle:
        a. Phase 1 (~10M tokens, codebook + decoder FROZEN; logit head
           still trains so the modulator adapts to LM improvements)
        b. Phase 2 curriculum (~40M tokens, GRPO on the logit head only;
           reward windows 512 → 1024 → 2048 → 4096)
    3. Repeat cycles

The architecture no longer has a separate action-collection sub-phase or
standalone RVQ-VAE codebook fit. Quantization is integrated into the
neuromodulator (DiscreteActionPolicy): encoder → logits → Gumbel-softmax
(phase 1) or hard Categorical (phase 2) → shared codebook → shared decoder.

This is a thin orchestrator that shells out to the existing sub-trainers
(src.train, src.train_phase2) via subprocess. Each sub-step is fully
runnable standalone for debugging; the outer loop just wires them together.

Usage:
    python -m src.train_loop \
        --work-dir outputs/v14/loop \
        --bootstrap-tokens 500_000_000 \
        --cycles 5
"""

import argparse
import glob
import os
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True,
                   help="Directory for all checkpoints and intermediate files")
    p.add_argument("--bootstrap-tokens", type=int, default=500_000_000)
    p.add_argument("--phase1-tokens-per-cycle", type=int, default=10_000_000,
                   help="Phase-1 tokens per cycle (all backprop; no action "
                        "collection in the new architecture — codebook+decoder "
                        "are frozen after bootstrap).")
    p.add_argument("--cycles", type=int, default=5)
    p.add_argument("--bs", type=int, default=80)
    p.add_argument("--phase2-bs", type=int, default=8)
    p.add_argument("--phase2-group-size", type=int, default=8)
    p.add_argument("--phase2-stage1-tokens", type=int, default=None)
    p.add_argument("--phase2-stage2-tokens", type=int, default=None)
    p.add_argument("--phase2-stage3-tokens", type=int, default=None)
    p.add_argument("--phase2-stage4-tokens", type=int, default=None)
    p.add_argument("--skip-bootstrap", action="store_true",
                   help="Skip bootstrap (if already done once)")
    p.add_argument("--start-cycle", type=int, default=0,
                   help="Resume at this cycle index")
    # Phase 2 hyperparams — forwarded to src.train_phase2. Defaults match
    # train_phase2.py; override here to tune without bypassing the outer loop.
    p.add_argument("--phase2-lr", type=float, default=1e-4)
    p.add_argument("--phase2-tau", type=float, default=1.0)
    p.add_argument("--phase2-entropy-coeff", type=float, default=0.01)
    p.add_argument("--phase2-warmup-batches", type=int, default=8)
    p.add_argument("--phase2-eval-interval", type=int, default=50)
    # Phase 1 eval hyperparams — forwarded to src.train. Previously these
    # defaulted to whatever src.train defined, with no way to tune from the
    # outer loop. Useful to dial down during long bootstrap runs where eval
    # cost is non-trivial.
    p.add_argument("--phase1-eval-interval", type=int, default=500,
                   help="src.train default is 500; --phase1-eval-interval 0 disables")
    p.add_argument("--phase1-eval-batches", type=int, default=8)
    p.add_argument("--phase1-eval-warmup-batches", type=int, default=4)
    return p.parse_args()


def tokens_to_steps(tokens: int, bs: int, T: int = 128) -> int:
    return max(1, tokens // (bs * T))


def run(cmd: list[str]):
    print(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"!!! Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def latest_ckpt(dir_path: str, before: set[str] | None = None) -> str | None:
    """Return the newest ckpt_*.pt in dir_path, excluding any in `before`."""
    matches = sorted(glob.glob(os.path.join(dir_path, "ckpt_*.pt")))
    matches = [m for m in matches if before is None or m not in before]
    return matches[-1] if matches else None


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    bootstrap_ckpt = os.path.join(args.work_dir, "bootstrap.pt")
    python = sys.executable

    # Track cumulative phase-1 step counter, since train.py uses --steps as
    # an ABSOLUTE target (not "more steps"). Each phase-1 launch must pass
    # target = start_step + steps_to_run.
    cumulative_step = 0

    # LR schedule target: cosine decays over this many optimizer steps.
    # Same schedule across all phase-1 calls (bootstrap + cycles).
    phase1_steps_per_cycle = tokens_to_steps(args.phase1_tokens_per_cycle, args.bs)
    bootstrap_steps = tokens_to_steps(args.bootstrap_tokens, args.bs)
    lr_target_step = bootstrap_steps + args.cycles * phase1_steps_per_cycle
    print(f"LR schedule target (total optimizer steps across all phase-1 "
          f"calls): {lr_target_step:,}")

    # Shared phase-1 eval CLI knobs forwarded to both bootstrap + cycle runs.
    phase1_eval_flags = [
        "--eval-interval", str(args.phase1_eval_interval),
        "--eval-batches", str(args.phase1_eval_batches),
        "--eval-warmup-batches", str(args.phase1_eval_warmup_batches),
    ]

    # ---- Bootstrap ----
    if not args.skip_bootstrap and not os.path.exists(bootstrap_ckpt):
        print(f"\n=========== BOOTSTRAP ({args.bootstrap_tokens:,} tokens) ===========")
        steps = tokens_to_steps(args.bootstrap_tokens, args.bs)
        before = set(glob.glob(os.path.join(args.work_dir, "ckpt_*.pt")))
        run([
            python, "-m", "src.train",
            "--bs", str(args.bs),
            "--steps", str(steps),
            "--lr-target-step", str(lr_target_step),
            "--save-dir", args.work_dir,
            "--save-interval", str(steps),
            *phase1_eval_flags,
        ])
        latest = latest_ckpt(args.work_dir, before=before)
        if latest is not None:
            os.rename(latest, bootstrap_ckpt)
            print(f"Bootstrap saved to {bootstrap_ckpt}")
            cumulative_step = steps
        else:
            print("!!! Phase 1 produced no checkpoint. Check output.")
            sys.exit(1)
    else:
        print(f"Bootstrap already exists at {bootstrap_ckpt}, skipping.")
        # Load bootstrap to pick up the cumulative step counter. A silent
        # fallback would produce 0-step cycles that "succeed" vacuously —
        # exit with a clear error instead.
        try:
            import torch
            ck = torch.load(bootstrap_ckpt, map_location="cpu", weights_only=False)
            cumulative_step = ck.get("step", 0)
            del ck
        except Exception as e:
            print(f"!!! FATAL: could not load {bootstrap_ckpt} to recover "
                  f"the cumulative step counter: {e}")
            print(f"!!! If the file is genuinely valid, re-verify with "
                  f"`torch.load({bootstrap_ckpt!r}, weights_only=False)`.")
            print(f"!!! Otherwise remove/rename it and re-run without "
                  f"--skip-bootstrap.")
            sys.exit(1)

    current_ckpt = bootstrap_ckpt

    # When resuming at --start-cycle N, find the last completed cycle's
    # phase2 checkpoint so we resume from the right model, not bootstrap.
    if args.start_cycle > 0:
        for prev in range(args.start_cycle - 1, -1, -1):
            prev_p2 = os.path.join(args.work_dir, f"cycle_{prev:02d}", "phase2.pt")
            if os.path.exists(prev_p2):
                current_ckpt = prev_p2
                try:
                    import torch
                    ck = torch.load(prev_p2, map_location="cpu", weights_only=False)
                    cumulative_step = ck.get("step", cumulative_step)
                    del ck
                except Exception:
                    pass
                print(f"Resuming from cycle {prev} checkpoint: {prev_p2} "
                      f"(step {cumulative_step})")
                break

    # ---- Iterative cycles ----
    for cycle in range(args.start_cycle, args.cycles):
        print(f"\n\n=========== CYCLE {cycle}/{args.cycles-1} ===========")
        cycle_dir = os.path.join(args.work_dir, f"cycle_{cycle:02d}")
        os.makedirs(cycle_dir, exist_ok=True)

        # Phase 1 (cycle): freeze codebook + decoder to preserve code
        # semantics against which phase 2 GRPO trained. The neuromod's
        # logit head + memory dynamics + LM all stay trainable.
        # Action collection / codebook fit / separate VQ-VAE training are
        # gone in the new architecture — the code vocabulary is trained
        # jointly during bootstrap and frozen after.
        phase1_steps = tokens_to_steps(args.phase1_tokens_per_cycle, args.bs)
        target_step_p1 = cumulative_step + phase1_steps
        phase1_end_ckpt = os.path.join(cycle_dir, "phase1_end.pt")
        print(f"\n--- Cycle {cycle} phase 1 ({args.phase1_tokens_per_cycle:,} tokens, "
              f"{phase1_steps} steps, absolute target {target_step_p1}) ---")
        before = set(glob.glob(os.path.join(cycle_dir, "ckpt_*.pt")))
        run([
            python, "-m", "src.train",
            "--bs", str(args.bs),
            "--steps", str(target_step_p1),
            "--lr-target-step", str(lr_target_step),
            "--save-dir", cycle_dir,
            "--save-interval", str(target_step_p1),
            "--resume", current_ckpt,
            "--freeze-codebook-decoder",
            *phase1_eval_flags,
        ])
        latest = latest_ckpt(cycle_dir, before=before)
        if latest is not None:
            os.rename(latest, phase1_end_ckpt)
            cumulative_step = target_step_p1
        else:
            print("!!! Phase 1 produced no checkpoint.")
            sys.exit(1)

        # Phase 2 curriculum (factored categorical GRPO on logit head).
        phase2_ckpt = os.path.join(cycle_dir, "phase2.pt")
        print(f"\n--- Cycle {cycle} phase 2 ---")
        phase2_seed = 42 + cycle * 100_000
        phase2_cmd = [
            python, "-m", "src.train_phase2",
            "--checkpoint", phase1_end_ckpt,
            "--out", phase2_ckpt,
            "--bs", str(args.phase2_bs),
            "--group-size", str(args.phase2_group_size),
            "--seed", str(phase2_seed),
            "--lr", str(args.phase2_lr),
            "--tau", str(args.phase2_tau),
            "--entropy-coeff", str(args.phase2_entropy_coeff),
            "--warmup-batches", str(args.phase2_warmup_batches),
            "--eval-interval", str(args.phase2_eval_interval),
        ]
        for i, v in enumerate(
            (args.phase2_stage1_tokens, args.phase2_stage2_tokens,
             args.phase2_stage3_tokens, args.phase2_stage4_tokens),
            start=1,
        ):
            if v is not None:
                phase2_cmd += [f"--stage{i}-tokens", str(v)]
        run(phase2_cmd)

        current_ckpt = phase2_ckpt

        # Auto-regenerate the cross-cycle health dashboard after each cycle
        # so the user can watch overall training health as it evolves.
        # Best-effort: a plot failure should not abort the cycle loop.
        plot_cmd = [python, "-m", "scripts.plot_health", args.work_dir,
                    "--bs", str(args.bs)]
        print(f"\n$ {' '.join(plot_cmd)}")
        r = subprocess.run(plot_cmd, check=False)
        if r.returncode != 0:
            print(f"  (health plot regen failed, exit={r.returncode})")

    print(f"\n=== All cycles complete. Final checkpoint: {current_ckpt} ===")


if __name__ == "__main__":
    main()
