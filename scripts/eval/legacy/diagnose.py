#!/usr/bin/env python3
"""diagnose.py — single-command training-run health check.

Synthesizes liveness, speed, loss, routing, memory, and checkpoint
state into a brief Green/Yellow/Red report.

**SAFE TO RUN DURING ACTIVE TRAINING:**
- All torch.load calls use map_location="cpu" — never touches CUDA.
- Only reads training.log, training.json, and ckpt state_dict on CPU.
- Skips the checkpoint inspection if the file was written < 30s ago
  (avoids reading a half-flushed save).
- Total impact: ~3 GB of CPU RAM transiently (the checkpoint), a few
  seconds of disk I/O. Zero impact on the GPU or the live training
  process.

Designed for an agent (or human) to call this during a long training
run and get a single-screen answer to "is this training healthy?"

Usage (from repo root):
    python scripts/diagnostics/diagnose.py                # auto-pick latest wave
    python scripts/diagnostics/diagnose.py --wave 1
    python scripts/diagnostics/diagnose.py --wave 2
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


GREEN = "\033[92m✓\033[0m"
YELLOW = "\033[93m⚠\033[0m"
RED = "\033[91m✗\033[0m"
DIM = "\033[2m"
RESET = "\033[0m"


def _run(cmd: list[str], timeout: int = 5) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
        return (out.stdout or "") + (out.stderr or "")
    except Exception as e:
        return f"<error: {e}>"


def _smooth(arr, win):
    return [statistics.mean(arr[max(0, i - win + 1):i + 1]) for i in range(len(arr))]


def section(title: str):
    print(f"\n{title}\n" + "─" * len(title))


def check_liveness(wave: int) -> dict:
    """Is training actually running?"""
    section(f"Liveness — wave {wave}")
    tmux = _run(["tmux", "ls"])
    log_path = Path(f"outputs/wave{wave}/training.log")
    log_exists = log_path.exists()
    log_age_s = time.time() - log_path.stat().st_mtime if log_exists else None

    pid_alive = False
    if "wave" in tmux:
        # Look for python train_wave process
        ps = _run(["pgrep", "-f", f"train_wave{wave}.py"])
        pid_alive = bool(ps.strip())

    status = GREEN
    if not pid_alive:
        status = RED
    elif log_age_s is None or log_age_s > 600:
        status = YELLOW

    print(f"  {status} tmux session for wave{wave}: "
          f"{'running' if 'wave' in tmux else 'NOT FOUND'}")
    print(f"  {status} train process: "
          f"{'alive' if pid_alive else 'DEAD'}")
    if log_exists:
        print(f"  {status} log last modified: {log_age_s:.1f}s ago")
    return {"alive": pid_alive, "log_age_s": log_age_s}


def check_gpu():
    """GPU util + VRAM."""
    section("GPU")
    out = _run([
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ])
    if "<error>" in out or not out.strip():
        print(f"  {RED} nvidia-smi failed")
        return {}
    line = out.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    util, mem_used, mem_total, temp = map(int, parts[:4])
    mem_pct = 100 * mem_used / mem_total

    util_sig = GREEN if util > 50 else (YELLOW if util > 10 else RED)
    mem_sig = GREEN if mem_pct < 85 else (YELLOW if mem_pct < 95 else RED)

    print(f"  {util_sig} util:    {util}%")
    print(f"  {mem_sig} VRAM:    {mem_used} / {mem_total} MiB ({mem_pct:.1f}%)")
    print(f"  {GREEN if temp < 80 else RED} temp:    {temp}°C")
    return {"util": util, "vram_mb": mem_used, "vram_pct": mem_pct, "temp_c": temp}


def check_log_state(wave: int, num_steps: int | None = None) -> dict:
    """Tail of training.log: latest step, step time, errors."""
    section(f"Log state")
    log_path = Path(f"outputs/wave{wave}/training.log")
    if not log_path.exists():
        print(f"  {RED} no training.log at {log_path}")
        return {}

    log = log_path.read_text(errors="replace")
    # Look for FATAL / Traceback / OOM
    fatal = [l for l in log.splitlines()
             if any(s in l for s in ("FATAL", "Traceback", "OutOfMemoryError"))]
    if fatal:
        print(f"  {RED} CRASH detected:")
        for l in fatal[-3:]:
            print(f"    {l[:120]}")
        return {"crash": True}

    # Count recompile_limit warnings (should be ≤1 after tau-tensor fix)
    recompile_count = log.count("recompile_limit")
    rsig = GREEN if recompile_count <= 1 else (
        YELLOW if recompile_count < 5 else RED)
    print(f"  {rsig} recompile_limit warnings: {recompile_count} (≤1 = healthy)")

    # Last step lines
    step_lines = [l for l in log.splitlines() if " step " in l and "loss=" in l]
    if not step_lines:
        print(f"  {YELLOW} no step prints yet (still in cold-start?)")
        return {"latest_step": None}

    last = step_lines[-1]
    # Parse step number + step time
    import re
    m_step = re.search(r"step\s+(\d+)\s+loss=([\d.]+)", last)
    m_st = re.search(r"\(([\d.]+)s/step\)", last)
    m_uf = re.search(r"r_uf=([\d.]+)\s+w_uf=([\d.]+)\s+r_ent=([\d.]+)", last)
    latest_step = int(m_step.group(1)) if m_step else None
    step_time_s = float(m_st.group(1)) if m_st else None
    r_uf = float(m_uf.group(1)) if m_uf else None
    w_uf = float(m_uf.group(2)) if m_uf else None

    print(f"  • latest step: {latest_step}")
    if step_time_s is not None:
        speed_sig = GREEN if step_time_s < 1.5 else (
            YELLOW if step_time_s < 3.0 else RED)
        print(f"  {speed_sig} step time:   {step_time_s:.2f} s/step")
        if num_steps and latest_step:
            remaining_s = (num_steps - latest_step) * step_time_s
            eta_h = remaining_s / 3600
            print(f"  • ETA:         {eta_h:.1f}h "
                  f"({num_steps - latest_step} steps remaining)")

    # Routing diversity
    if r_uf is not None:
        rsig = GREEN if r_uf > 0.05 else (YELLOW if r_uf > 0.02 else RED)
        wsig = GREEN if w_uf > 0.05 else (YELLOW if w_uf > 0.02 else RED)
        print(f"  {rsig} r_uf:        {r_uf:.4f} (>0.05 healthy, <0.02 collapse)")
        print(f"  {wsig} w_uf:        {w_uf:.4f}")

    return {
        "latest_step": latest_step,
        "step_time_s": step_time_s,
        "r_uf": r_uf, "w_uf": w_uf,
        "crash": False,
        "recompile_count": recompile_count,
    }


def check_training_history(wave: int) -> dict:
    """Parse training.json for per-component grads + loss trajectory."""
    section("Training history (training.json)")
    json_path = Path(f"outputs/wave{wave}/training.json")
    if not json_path.exists():
        print(f"  {YELLOW} no training.json yet")
        return {}

    h = json.load(open(json_path))
    n = len(h.get("step", []))
    if n < 50:
        print(f"  {YELLOW} only {n} steps logged — too early for trends")
        return {}

    smoothed_loss = _smooth(h["loss"], 100)
    early_loss = smoothed_loss[100] if n > 100 else smoothed_loss[-1]
    cur_loss = smoothed_loss[-1]
    min_loss = min(smoothed_loss[100:]) if n > 100 else cur_loss

    print(f"  • avg100 loss: now={cur_loss:.3f}, "
          f"min-so-far={min_loss:.3f}, early={early_loss:.3f}")
    drop = early_loss - cur_loss
    drop_sig = GREEN if drop > 0.3 else (YELLOW if drop > 0.05 else RED)
    print(f"  {drop_sig} loss drop from step 100: {drop:+.3f}")

    # Per-component grad norms — look for dead components
    print()
    for comp in ["read", "write", "manifold", "entry", "bridge_in_llama"]:
        key = f"grad_norm_{comp}"
        if key not in h or not h[key]:
            continue
        # Take last-100 mean
        recent = h[key][-100:]
        avg = sum(recent) / len(recent)
        sig = GREEN if avg > 1e-6 else RED
        print(f"  {sig} grad_norm_{comp:20s}: avg100={avg:.2e}")

    # Drift trajectory
    if "concept_drift_max" in h and h["concept_drift_max"]:
        drift_max = h["concept_drift_max"][-1]
        dsig = GREEN if drift_max < 30 else (YELLOW if drift_max < 100 else RED)
        print(f"\n  {dsig} concept_drift_max: {drift_max:.2f} "
              f"(<30 healthy, >100 likely exploding)")

    # Val state
    val_steps = h.get("val_step", [])
    val_loss = h.get("val_loss", {})
    if val_steps and val_loss:
        print()
        for src, vals in val_loss.items():
            if not vals:
                continue
            best = min(vals)
            best_idx = vals.index(best)
            best_step = val_steps[best_idx]
            stale_passes = len(vals) - 1 - best_idx
            stale_sig = GREEN if stale_passes < 3 else (
                YELLOW if stale_passes < 8 else RED)
            print(f"  {stale_sig} val[{src:>14}]: latest={vals[-1]:.3f}, "
                  f"best={best:.3f} at step {best_step} "
                  f"({stale_passes} val passes stale)")
    return {"cur_loss": cur_loss, "min_loss": min_loss}


def check_concept_usage(wave: int) -> dict:
    """Load latest ckpt's state_dict + inspect manifold.usage_ema.

    CPU-only: `map_location="cpu"` forces host-RAM allocation, never
    touches CUDA. To stay safe against reading a half-flushed save,
    skip if the ckpt was written < 30s ago.
    """
    section("Concept usage (manifold.usage_ema from latest ckpt)")
    ckpt = Path(f"outputs/wave{wave}/ckpt.pt")
    if not ckpt.exists():
        print(f"  {YELLOW} no ckpt.pt yet")
        return {}

    age_s = time.time() - ckpt.stat().st_mtime
    if age_s < 30:
        print(f"  {YELLOW} skipping (ckpt is only {age_s:.1f}s old — "
              f"may be mid-save; rerun in a minute)")
        return {}

    try:
        import torch
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        u = ck["model_state_dict"]["manifold.usage_ema"]
    except Exception as e:
        print(f"  {RED} couldn't load usage_ema: {e}")
        return {}

    N = u.shape[0]
    threshold_active = 1.0 / (100 * N)
    threshold_dominant = 10.0 / N
    n_alive = (u > 0).sum().item()
    n_active = (u > threshold_active).sum().item()
    n_dominant = (u > threshold_dominant).sum().item()

    asig = GREEN if n_active >= 0.9 * N else (
        YELLOW if n_active >= 0.5 * N else RED)
    dsig = GREEN if n_dominant < 100 else (
        YELLOW if n_dominant < 500 else RED)

    print(f"  • N = {N} concepts")
    print(f"  {asig} active:    {n_active} / {N} ({100*n_active/N:.1f}%) "
          f"(usage > 1/(100N))")
    print(f"  {dsig} dominant:  {n_dominant} (usage > 10/N)")
    print(f"  • max usage on any one concept: {u.max().item():.5f}")
    return {"n_active": n_active, "n_dominant": n_dominant, "N": N}


def check_checkpoints(wave: int) -> dict:
    section("Checkpoints")
    base = Path(f"outputs/wave{wave}")
    files = {}
    for name in ["ckpt.pt", "ckpt.best.pt"]:
        p = base / name
        if p.exists():
            age = time.time() - p.stat().st_mtime
            files[name] = age
            sig = GREEN if age < 7200 else (YELLOW if age < 21600 else RED)
            print(f"  {sig} {name}: {age/60:.1f} min old")
        else:
            print(f"  {YELLOW} {name}: missing")

    # Staleness gap between ckpt.pt and ckpt.best.pt
    if "ckpt.pt" in files and "ckpt.best.pt" in files:
        gap = files["ckpt.best.pt"] - files["ckpt.pt"]
        if gap > 0:
            stale_min = gap / 60
            sig = GREEN if stale_min < 30 else (
                YELLOW if stale_min < 120 else RED)
            print(f"  {sig} val improvement stale: {stale_min:.1f} min "
                  f"(best.pt is {stale_min:.0f} min older than ckpt.pt)")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=None,
                    help="1 or 2; auto-pick latest if omitted")
    args = ap.parse_args()

    # Auto-pick wave: if wave2/training.log is newer + active, use it.
    if args.wave is None:
        w1 = Path("outputs/wave1/training.log")
        w2 = Path("outputs/wave2/training.log")
        if w2.exists() and (not w1.exists() or
                             w2.stat().st_mtime > w1.stat().st_mtime):
            args.wave = 2
        else:
            args.wave = 1

    # Approximate num-steps from the launch script (hardcoded for now;
    # could parse from training.log "Starting Wave 1 training: N steps").
    num_steps = {1: 29044, 2: 15413}.get(args.wave)

    print(f"\n{'═' * 60}")
    print(f"  Training diagnostic: wave {args.wave}")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 60}")

    check_liveness(args.wave)
    check_gpu()
    log = check_log_state(args.wave, num_steps=num_steps)
    check_training_history(args.wave)
    check_concept_usage(args.wave)
    check_checkpoints(args.wave)

    # ── Synthesis ────────────────────────────────────────────────────
    section("Overall")
    if log.get("crash"):
        print(f"  {RED} TRAINING CRASHED — see traceback above")
        return
    issues = []
    if log.get("r_uf") is not None and log["r_uf"] < 0.02:
        issues.append("routing collapsed (r_uf < 0.02)")
    if log.get("recompile_count", 0) > 5:
        issues.append("excessive recompiles")
    if log.get("step_time_s") and log["step_time_s"] > 3.0:
        issues.append("step time slow")
    if not issues:
        print(f"  {GREEN} Healthy. No flags.")
    else:
        print(f"  {YELLOW} Flags: " + "; ".join(issues))
    print()


if __name__ == "__main__":
    main()
