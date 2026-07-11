#!/usr/bin/env python3
"""
WATCHDOG — the billing backstop. Polls each pod's R2 _STATUS marker and the vast.ai
instance list, and DESTROYS an instance as soon as it is done, failed, or has run
past the wall-clock cap. Run this locally right after launch and leave it running.

Kill conditions (any one triggers destroy):
  * R2 _STATUS is DONE / FAILED / TIMEOUT   (bootstrap finished, results uploaded)
  * instance age ≥ cap_hours (cfg.max_hours + grace)   (hard billing cap)
  * instance is gone / errored in vast's list           (already dead; drop tracking)

The vast.ai key stays here (never on the pod). The watchdog is authoritative for
teardown; bootstrap.sh's MAX_HOURS timeout and the account credit ceiling are the
independent backstops if this process dies.

Usage:
  python scripts/pod/watchdog.py <run_id>                 # poll+reap until all gone
  python scripts/pod/watchdog.py <run_id> --once          # single pass, then exit
  python scripts/pod/watchdog.py <run_id> --destroy-all   # PANIC: nuke every pod now
  python scripts/pod/watchdog.py <run_id> --grace 0.5 --interval 60
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE_DIR = HERE / "state"
VAST_KEY_FILE = Path(os.path.expanduser("~/.config/vastai/api_key"))
R2_CRED_FILE = Path(os.path.expanduser("~/.config/r2/credentials"))

TERMINAL = ("DONE", "FAILED", "TIMEOUT")


def _vast_env() -> dict:
    env = dict(os.environ)
    env["VAST_API_KEY"] = VAST_KEY_FILE.read_text().strip()
    return env


def _vastai(args, env, check=True) -> str:
    exe = str((HERE.parents[1] / ".venv/bin/vastai"))
    if not Path(exe).exists():
        exe = "vastai"
    p = subprocess.run([exe, *args], env=env, capture_output=True, text=True)
    if check and p.returncode != 0:
        print(f"[watchdog] vastai {' '.join(args)} failed: {p.stderr or p.stdout}")
    return p.stdout


def _r2_env() -> dict:
    env = dict(os.environ)
    for line in R2_CRED_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    env["AWS_ACCESS_KEY_ID"] = env.get("R2_ACCESS_KEY_ID", "")
    env["AWS_SECRET_ACCESS_KEY"] = env.get("R2_SECRET_ACCESS_KEY", "")
    env["AWS_DEFAULT_REGION"] = "auto"
    return env


def r2_status(run_id, arm, r2env) -> str | None:
    """Read the pod's _STATUS marker from R2, or None if not written yet."""
    prefix = r2env.get("R2_PREFIX", "neuromorphic")
    bucket = r2env["R2_BUCKET"]
    endpoint = r2env["R2_ENDPOINT"]
    key = f"s3://{bucket}/{prefix}/results/{run_id}/{arm}/_STATUS"
    p = subprocess.run(["aws", "s3", "cp", "--endpoint-url", endpoint, key, "-"],
                       env=r2env, capture_output=True, text=True)
    if p.returncode != 0:
        return None
    return p.stdout.strip()


def vast_instances(env) -> dict:
    """Map instance_id -> vast record for all my instances."""
    out = _vastai(["show", "instances", "--raw"], env)
    try:
        rows = json.loads(out)
    except json.JSONDecodeError:
        return {}
    return {str(r.get("id")): r for r in rows}


def destroy(instance_id, env):
    print(f"[watchdog] destroying instance {instance_id}")
    _vastai(["destroy", "instance", str(instance_id)], env, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--interval", type=int, default=60, help="poll seconds")
    ap.add_argument("--grace", type=float, default=0.5, help="hours added to cfg.max_hours before hard-kill")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--destroy-all", action="store_true", help="PANIC: destroy every tracked pod now")
    args = ap.parse_args()

    state_path = STATE_DIR / f"{args.run_id}.json"
    if not state_path.exists():
        sys.exit(f"[watchdog] no state at {state_path}")
    state = json.loads(state_path.read_text())
    env = _vast_env()
    cap_h = float(state.get("cfg", {}).get("max_hours", 8)) + args.grace

    tracked = {i["instance_id"]: i for i in state["instances"] if i.get("instance_id")}
    if args.destroy_all:
        for iid in tracked:
            destroy(iid, env)
        print("[watchdog] PANIC destroy issued for all tracked instances.")
        return

    r2env = _r2_env()
    done: set = set()
    print(f"[watchdog] run={args.run_id} tracking {len(tracked)} pods; cap={cap_h}h "
          f"interval={args.interval}s")

    while True:
        vast = vast_instances(env)
        for iid, info in tracked.items():
            if iid in done:
                continue
            arm = info["arm"]
            rec = vast.get(str(iid))
            # (1) already gone from vast → stop tracking
            if rec is None:
                print(f"[watchdog] {arm} ({iid}) not in vast list — assumed destroyed.")
                done.add(iid)
                continue
            status = r2_status(args.run_id, arm, r2env)
            age_h = float(rec.get("duration", 0) or 0) / 3600.0
            cost = float(rec.get("dph_total", info.get("dph", 0)) or 0)
            tag = f"{arm} ({iid}) vast={rec.get('actual_status','?')} age={age_h:.2f}h r2={status or '—'}"
            # (2) terminal status from the pod → reap
            if status and status.split()[0] in TERMINAL:
                print(f"[watchdog] {tag} → TERMINAL, destroying")
                destroy(iid, env)
                done.add(iid)
            # (3) wall-clock cap → reap (billing backstop)
            elif age_h >= cap_h:
                print(f"[watchdog] {tag} → past {cap_h}h cap, destroying")
                destroy(iid, env)
                done.add(iid)
            else:
                print(f"[watchdog] {tag} (${cost:.3f}/hr)")
        remaining = [i for i in tracked if i not in done]
        if not remaining:
            print("[watchdog] all pods reaped. done.")
            return
        if args.once:
            print(f"[watchdog] --once: {len(remaining)} still running.")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
