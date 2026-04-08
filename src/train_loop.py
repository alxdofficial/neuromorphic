"""Outer orchestration driver for bootstrap + iterative cycle training.

Runs the full pipeline:
    1. Bootstrap phase 1 (one-time, ~200M tokens, modulator trains)
    2. For each cycle:
        a. Phase 1 (50M tokens, modulator FROZEN)
        b. Action collection (last ~2M tokens of phase 1)
        c. RVQ codebook fit
        d. Phase 2 curriculum (512 -> 2048 -> 4096 reward windows)
    3. Repeat cycles

This is a thin orchestrator that shells out to the already-existing
sub-trainers (src.train, src.train_phase2, scripts.train_codebook) via
subprocess. Each sub-step is fully runnable standalone for debugging;
the outer loop just wires them together.

Usage:
    python -m src.train_loop \
        --work-dir outputs/v12/loop \
        --bootstrap-tokens 200_000_000 \
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
    p.add_argument("--bootstrap-tokens", type=int, default=200_000_000)
    p.add_argument("--phase1-tokens-per-cycle", type=int, default=50_000_000)
    p.add_argument("--action-collection-tokens", type=int, default=2_000_000)
    p.add_argument("--cycles", type=int, default=5)
    p.add_argument("--bs", type=int, default=96)
    p.add_argument("--phase2-bs", type=int, default=8)
    p.add_argument("--phase2-group-size", type=int, default=8)
    p.add_argument("--skip-bootstrap", action="store_true",
                   help="Skip bootstrap (if already done once)")
    p.add_argument("--start-cycle", type=int, default=0,
                   help="Resume at this cycle index")
    p.add_argument("--codebook-epochs", type=int, default=20)
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

    # ---- Bootstrap ----
    if not args.skip_bootstrap and not os.path.exists(bootstrap_ckpt):
        print(f"\n=========== BOOTSTRAP ({args.bootstrap_tokens:,} tokens) ===========")
        steps = tokens_to_steps(args.bootstrap_tokens, args.bs)
        before = set(glob.glob(os.path.join(args.work_dir, "ckpt_*.pt")))
        run([
            python, "-m", "src.train",
            "--bs", str(args.bs),
            "--steps", str(steps),
            "--save-dir", args.work_dir,
            "--save-interval", str(steps),
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
        # Best-effort: load to pick up the cumulative step counter
        try:
            import torch
            ck = torch.load(bootstrap_ckpt, map_location="cpu", weights_only=False)
            cumulative_step = ck.get("step", 0)
            del ck
        except Exception:
            pass

    current_ckpt = bootstrap_ckpt

    # ---- Iterative cycles ----
    for cycle in range(args.start_cycle, args.cycles):
        print(f"\n\n=========== CYCLE {cycle}/{args.cycles-1} ===========")
        cycle_dir = os.path.join(args.work_dir, f"cycle_{cycle:02d}")
        os.makedirs(cycle_dir, exist_ok=True)

        # Phase 1 with frozen modulator (regular steps)
        phase1_main_tokens = args.phase1_tokens_per_cycle - args.action_collection_tokens
        phase1_main_steps = tokens_to_steps(phase1_main_tokens, args.bs)
        target_step_main = cumulative_step + phase1_main_steps
        phase1_main_ckpt = os.path.join(cycle_dir, "phase1_main.pt")
        print(f"\n--- Cycle {cycle} phase 1 main ({phase1_main_tokens:,} tokens, "
              f"{phase1_main_steps} new steps, absolute target {target_step_main}) ---")
        before = set(glob.glob(os.path.join(cycle_dir, "ckpt_*.pt")))
        run([
            python, "-m", "src.train",
            "--bs", str(args.bs),
            "--steps", str(target_step_main),
            "--save-dir", cycle_dir,
            "--save-interval", str(target_step_main),
            "--resume", current_ckpt,
            "--freeze-modulator",
        ])
        latest = latest_ckpt(cycle_dir, before=before)
        if latest is not None:
            os.rename(latest, phase1_main_ckpt)
            cumulative_step = target_step_main
        else:
            print("!!! Phase 1 main produced no checkpoint.")
            sys.exit(1)

        # Action collection sub-phase (still phase 1 TBPTT, but with collection on)
        ac_steps = tokens_to_steps(args.action_collection_tokens, args.bs)
        target_step_ac = cumulative_step + ac_steps
        action_db = os.path.join(cycle_dir, "action_database.pt")
        phase1_end_ckpt = os.path.join(cycle_dir, "phase1_end.pt")
        print(f"\n--- Cycle {cycle} action collection ({ac_steps} new steps, "
              f"target {target_step_ac}) ---")
        before = set(glob.glob(os.path.join(cycle_dir, "ckpt_*.pt")))
        run([
            python, "-m", "src.train",
            "--bs", str(args.bs),
            "--steps", str(target_step_ac),
            "--save-dir", cycle_dir,
            "--save-interval", str(target_step_ac),
            "--resume", phase1_main_ckpt,
            "--freeze-modulator",
            "--collect-actions",
            "--action-db-out", action_db,
        ])
        latest = latest_ckpt(cycle_dir, before=before)
        if latest is not None:
            os.rename(latest, phase1_end_ckpt)
            cumulative_step = target_step_ac
        else:
            print("!!! Action collection produced no checkpoint.")
            sys.exit(1)

        # Codebook fit
        codebook_path = os.path.join(cycle_dir, "codebook.pt")
        print(f"\n--- Cycle {cycle} codebook fit ---")
        run([
            python, "-m", "scripts.train_codebook",
            "--actions", action_db,
            "--out", codebook_path,
            "--epochs", str(args.codebook_epochs),
        ])

        # Phase 2 curriculum
        phase2_ckpt = os.path.join(cycle_dir, "phase2.pt")
        print(f"\n--- Cycle {cycle} phase 2 ---")
        run([
            python, "-m", "src.train_phase2",
            "--checkpoint", phase1_end_ckpt,
            "--codebook", codebook_path,
            "--out", phase2_ckpt,
            "--bs", str(args.phase2_bs),
            "--group-size", str(args.phase2_group_size),
        ])

        current_ckpt = phase2_ckpt

    print(f"\n=== All cycles complete. Final checkpoint: {current_ckpt} ===")


if __name__ == "__main__":
    main()
