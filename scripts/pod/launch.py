#!/usr/bin/env python3
"""
LAUNCH — pick vast.ai GPU offers and spin up one pod per training arm.

DEFAULT IS DRY-RUN: prints the plan + a worst-case cost bound and does NOT spend a
cent. Pass --go to actually create instances.

Two modes:
  --mode ssh   (default) create bare SSH pods; secrets are NOT sent to vast.ai.
               After launch, run  scripts/pod/drive.sh <run_id>  to SSH in (over an
               encrypted channel), copy R2 creds, and start bootstrap.sh in a remote
               tmux. This matches the "local tmux → ssh into pod" workflow.
  --mode auto  create pods whose onstart clones the repo and runs bootstrap.sh with
               secrets passed via --env. Fully hands-off, but R2 creds are stored in
               vast.ai's instance metadata. Use only if you accept that exposure.

Billing safety (independent of mode):
  * every pod is wall-clock-capped by bootstrap.sh (MAX_HOURS, default 8h),
  * the vast.ai key stays local — pods never get it,
  * scripts/pod/watchdog.py destroys instances on DONE/FAILED/TIMEOUT/wall-cap,
  * the account credit itself ($ balance) is a hard ceiling; vast stops pods at $0.

State is written to scripts/pod/state/<run_id>.json (offers, instance ids, ssh
endpoints) for drive.sh / watchdog.py / pull_results.sh to consume.

Examples:
  # see the plan + cost, spend nothing:
  python scripts/pod/launch.py --arms slotgraph_baseline icae_baseline
  # actually launch (ssh mode), then drive + watch:
  python scripts/pod/launch.py --arms slotgraph_baseline --go
  scripts/pod/drive.sh podrun-XXXX
  python scripts/pod/watchdog.py podrun-XXXX
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

# The trainable cohort (eval-only arms need no pod — they run locally at analysis time).
DEFAULT_ARMS = [
    "icae_baseline", "autocompressor_baseline", "titans_baseline",
    "gisting_baseline", "memoryllm_baseline", "slotgraph_baseline",
]
# GPU tiers. Each sets a compute-capability floor (×100) AND the pod image, because
# the two are coupled: Blackwell (5090, sm_120) needs a CUDA-12.8/torch-≥2.7 build,
# while Ada/Ampere run fine on the older cu121 image.
#   4090 (Ada, sm_89) — RECOMMENDED: big speed jump over Ampere, zero image risk,
#                       ~$0.30/hr, 24GB is plenty at small batch. Default.
#   5090 (Blackwell)  — marginally faster for our small-batch/135M workload (latency-
#                       bound → extra SMs/bw underused), needs the newer image.
#   ampere (3090/A-series) — cheapest ($0.12/hr) but slower clocks/bf16.
# Filter by GPU *name*, not compute-capability: capability is a poor speed proxy
# (a 4080S and a 4090 share sm_89; an "RTX PRO 4000 Blackwell" and a 5090 share sm_120
# but the workstation card has a fraction of the SMs/bandwidth). Pin the exact card so
# "we want speed" gets the card we mean. min_cc stays as a bf16 sanity guard.
# Images: use VAST'S OWN pre-cached image (vastai/pytorch), NOT raw Docker Hub (pytorch/pytorch).
# Vast hosts keep the vastai/* images warm → the container is ready in seconds; a raw Docker Hub
# pull on a cold host took ~8.4 min (505s) in a live test (image download + apt + locale-gen) — pure
# billed idle. The `-auto` tag selects the torch build matching the host driver. cuda-13.0.3-auto was
# VALIDATED on a 5090 (torch 2.12/cu130, sm_120/Blackwell + bf16 OK). NOTE: the vast image ships torch
# in a venv at /venv/main (bootstrap activates it) and sets HF_HOME=/workspace/.hf_home.
_VAST_IMG = "vastai/pytorch:cuda-13.0.3-auto"
GPU_TIERS = {
    "4090":   {"gpu_name": "RTX_4090", "min_cc": 890,  "image": _VAST_IMG},
    "5090":   {"gpu_name": "RTX_5090", "min_cc": 1200, "image": _VAST_IMG},
    "3090":   {"gpu_name": "RTX_3090", "min_cc": 800,  "image": _VAST_IMG},
}
DEFAULT_GPU = "5090"   # 32GB Blackwell — the per-layer-KV arms (gisting/memoryllm) need >24GB
                       # for the full 5-task behavioral_kl mix at B=8; 5090 fits all arms uniformly.


def _vast_env() -> dict:
    if not VAST_KEY_FILE.exists():
        sys.exit(f"no vast.ai key at {VAST_KEY_FILE}")
    env = dict(os.environ)
    env["VAST_API_KEY"] = VAST_KEY_FILE.read_text().strip()
    return env


def _vastai(args: list[str], env: dict) -> str:
    """Run the vastai CLI (from the repo venv if present) and return stdout."""
    exe = str((HERE.parents[1] / ".venv/bin/vastai"))
    if not Path(exe).exists():
        exe = "vastai"
    p = subprocess.run([exe, *args], env=env, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"vastai {' '.join(args)} failed:\n{p.stderr or p.stdout}")
    return p.stdout


def search_offers(env, min_ram, max_dph, min_disk, cuda_min, min_compute_cap,
                  gpu_name, limit) -> list[dict]:
    # NOTE: the vast CLI's server-side numeric filters are unreliable for large
    # thresholds — `gpu_ram>=24000` wrongly returns 0 even for 24564MB 4090s. So we
    # query the reliable constraints server-side (single GPU, exact card name, price
    # cap) and apply the VRAM / disk / cuda / compute-capability / bandwidth cuts
    # CLIENT-SIDE. gpu_name pins the exact card (speed ≠ compute-capability).
    name_q = f"gpu_name={gpu_name} " if gpu_name else ""
    query = f"rentable=true num_gpus=1 {name_q}dph_total<={max_dph}"
    out = _vastai(["search", "offers", query, "-o", "dph_total", "--raw",
                   "--limit", str(limit)], env)
    try:
        offers = json.loads(out)
    except json.JSONDecodeError:
        sys.exit(f"could not parse offers json:\n{out[:500]}")

    def ok(o):
        # compute_cap is capability×100 (V100=700, A100=800, 3090=860, 4090=890).
        # Require ≥800 (Ampere) so bf16 autocast has hardware support — a V100 (sm_70)
        # would error or silently fall back to fp32 under our bf16 training path.
        return (float(o.get("gpu_ram", 0) or 0) >= min_ram
                and float(o.get("disk_space", 0) or 0) >= min_disk
                and float(o.get("inet_down", 0) or 0) >= 200
                and float(o.get("cuda_max_good", 0) or 0) >= cuda_min
                and float(o.get("compute_cap", 0) or 0) >= min_compute_cap
                and float(o.get("dph_total", 1e9) or 1e9) <= max_dph)
    kept = [o for o in offers if ok(o)]
    kept.sort(key=lambda o: float(o.get("dph_total", 1e9)))
    return kept


def _fmt_offer(o: dict) -> str:
    return (f"id={o['id']} {o.get('gpu_name','?')} {int(o.get('gpu_ram',0))}MB "
            f"${o.get('dph_total',0):.3f}/hr {o.get('geolocation','?')} "
            f"inet↓{int(o.get('inet_down',0))}Mbps disk{int(o.get('disk_space',0))}GB")


def choose_offers(offers, n) -> list[dict]:
    """Cheapest n DISTINCT machines (avoid stacking arms on one host)."""
    chosen, seen_hosts = [], set()
    for o in offers:                      # already price-sorted
        h = o.get("host_id")
        if h in seen_hosts:
            continue
        seen_hosts.add(h)
        chosen.append(o)
        if len(chosen) >= n:
            break
    return chosen


def _run_id() -> str:
    # No wall-clock available in this process context is fine; use a short counter file.
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    ctr = STATE_DIR / ".counter"
    n = (int(ctr.read_text()) + 1) if ctr.exists() else 1
    ctr.write_text(str(n))
    return f"podrun-{n:04d}"


def build_onstart_cmd(arm, run_id, cfg, secrets: dict | None, run_bootstrap: bool) -> str:
    """onstart script for the pod.

    auto mode (run_bootstrap=True, secrets set): clone repo, export env + R2 secrets,
      run bootstrap.sh — fully hands-off.
    ssh mode (run_bootstrap=False, secrets=None): only pre-clone the repo so drive.sh
      can immediately start bootstrap over SSH. NOTHING sensitive is baked into vast
      metadata; training doesn't start until drive.sh injects creds over the SSH channel."""
    clone = ("cd /workspace && rm -rf neuromorphic && "
             "git clone --filter=blob:none " + cfg["repo_url"] + " neuromorphic")
    if not run_bootstrap:
        return clone
    exports = [
        f"export ARM={arm}", f"export OUT_TAG={run_id}", f"export RUN_ID={run_id}",
        f"export REPO_REF={cfg['ref']}", f"export STEPS={cfg['steps']}",
        f"export BATCH={cfg['batch']}", f"export MAX_HOURS={cfg['max_hours']}",
    ]
    if secrets:
        exports += [f"export {k}={v}" for k, v in secrets.items()]
    return " ; ".join(exports) + " ; " + clone + " ; bash neuromorphic/scripts/pod/bootstrap.sh"


def create_instance(env, offer_id, arm, image, disk, onstart_cmd, mode) -> dict:
    args = ["create", "instance", str(offer_id), "--image", image,
            "--disk", str(disk), "--label", arm, "--raw"]
    if mode == "ssh":
        args += ["--ssh", "--direct"]
    args += ["--onstart-cmd", onstart_cmd]
    out = _vastai(args, env)
    try:
        res = json.loads(out)
    except json.JSONDecodeError:
        res = {"raw": out.strip()}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--mode", choices=["ssh", "auto"], default="ssh")
    ap.add_argument("--go", action="store_true", help="actually create pods (default: dry-run)")
    ap.add_argument("--gpu", choices=list(GPU_TIERS), default=DEFAULT_GPU,
                    help="GPU tier: sets the compute-cap floor + matching pod image "
                         "(4090=recommended, 5090=Blackwell+newer image, ampere=cheapest)")
    ap.add_argument("--image", default=None, help="override the tier's pod image")
    ap.add_argument("--repo-url", default="https://github.com/alxdofficial/neuromorphic.git")
    ap.add_argument("--ref", default="main", help="git branch/sha the pods train (MUST contain the fixes)")
    ap.add_argument("--steps", type=int, default=2000,
                    help="training steps. DEFAULT 2000 = a PILOT; the full declared protocol is 8000 — "
                         "label any <8000-step run a pilot in results.")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-hours", type=int, default=8, help="hard per-pod wall-clock cap")
    ap.add_argument("--min-gpu-ram", type=int, default=32000, help="min GPU VRAM in MB (32GB — 5090; "
                    "the per-layer-KV arms need >24GB at B=8 on the full mix)")
    ap.add_argument("--max-dph", type=float, default=0.80, help="max $/hr per pod (0.80 widens the "
                    "5090 offer pool vs the old 0.60 4090 cap)")
    ap.add_argument("--min-disk", type=int, default=40, help="min disk GB (image+data+ckpts)")
    ap.add_argument("--cuda-min", type=float, default=12.0)
    ap.add_argument("--min-compute-cap", type=int, default=None,
                    help="override the tier's min compute capability ×100 (890=4090, 1200=5090)")
    ap.add_argument("--disk", type=int, default=40, help="disk GB to allocate on the pod")
    args = ap.parse_args()

    # Resolve GPU tier → card name + compute-cap floor + pod image (explicit flags win).
    tier = GPU_TIERS[args.gpu]
    gpu_name = tier["gpu_name"]
    min_cc = args.min_compute_cap if args.min_compute_cap is not None else tier["min_cc"]
    image = args.image if args.image is not None else tier["image"]
    if args.gpu == "5090":
        print(f"[launch] NOTE: 5090 (Blackwell sm_120) uses the vast pre-cached image {image} "
              "— validated: torch 2.12/cu130, sm_120 + bf16 OK, instant load.")

    env = _vast_env()
    run_id = _run_id()
    cfg = dict(ref=args.ref, steps=args.steps, batch=args.batch,
               max_hours=args.max_hours, repo_url=args.repo_url)

    print(f"[launch] run_id={run_id} mode={args.mode} arms={args.arms}")
    print(f"[launch] gpu tier={args.gpu} (card={gpu_name}, compute_cap≥{min_cc}, image={image})")
    print(f"[launch] searching offers: ≥{args.min_gpu_ram}MB VRAM, ≤${args.max_dph}/hr, "
          f"≥{args.min_disk}GB disk, cuda≥{args.cuda_min}")
    offers = search_offers(env, args.min_gpu_ram, args.max_dph, args.min_disk,
                           args.cuda_min, min_cc, gpu_name, limit=800)
    if not offers:
        sys.exit("[launch] no offers matched — loosen --max-dph / --min-gpu-ram.")
    chosen = choose_offers(offers, len(args.arms))
    if len(chosen) < len(args.arms):
        print(f"[launch] WARNING only {len(chosen)} distinct machines for {len(args.arms)} arms")

    plan = list(zip(args.arms, chosen))
    total_dph = sum(o["dph_total"] for _, o in plan)
    worst = total_dph * args.max_hours
    print(f"\n[launch] PLAN ({len(plan)} pods):")
    for arm, o in plan:
        print(f"  {arm:<26} → {_fmt_offer(o)}")
    if args.steps < 8000:
        print(f"[launch] NOTE: {args.steps} steps < 8000 → this is a PILOT, not the full protocol.")
    print(f"\n[launch] cost: Σ${total_dph:.3f}/hr; worst case (all run the full "
          f"{args.max_hours}h cap) = ${worst:.2f}")
    print(f"[launch] account credit gates spend; watchdog + {args.max_hours}h cap + "
          f"$credit ceiling bound the bill.")

    if not args.go:
        print("\n[launch] DRY-RUN — nothing created. Re-run with --go to launch.")
        if args.ref == "main":
            print("[launch] NOTE: pods train --ref main. Ensure the training fixes are "
                  "committed+pushed to main (or pass --ref <pod-branch>) BEFORE --go.")
        return

    # ── actually create ──────────────────────────────────────────────────────
    secrets = None
    if args.mode == "auto":
        secrets = _load_r2_secrets()     # baked into vast metadata (documented exposure)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {"run_id": run_id, "mode": args.mode, "cfg": cfg, "image": image,
             "gpu_tier": args.gpu, "instances": []}
    for arm, o in plan:
        onstart = build_onstart_cmd(arm, run_id, cfg, secrets,
                                    run_bootstrap=(args.mode == "auto"))
        print(f"[launch] creating {arm} on offer {o['id']} (${o['dph_total']:.3f}/hr) …")
        res = create_instance(env, o["id"], arm, image, args.disk, onstart, args.mode)
        inst_id = res.get("new_contract") or res.get("id")
        print(f"           → instance {inst_id}  {res if inst_id is None else ''}")
        state["instances"].append({
            "arm": arm, "offer_id": o["id"], "instance_id": inst_id,
            "dph": o["dph_total"], "gpu": o.get("gpu_name"), "create_raw": res,
        })
    state_path = STATE_DIR / f"{run_id}.json"
    state_path.write_text(json.dumps(state, indent=2))
    print(f"\n[launch] created {len(state['instances'])} pods; state → {state_path}")
    if args.mode == "ssh":
        print(f"[launch] next: scripts/pod/drive.sh {run_id}   # ssh in + start bootstrap")
    print(f"[launch] then:  python scripts/pod/watchdog.py {run_id}   # billing safety")


def _load_r2_secrets() -> dict:
    if not R2_CRED_FILE.exists():
        sys.exit(f"[launch] --mode auto needs R2 creds at {R2_CRED_FILE}")
    out = {}
    for line in R2_CRED_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.startswith("R2_"):
            out[k] = v
    return out


if __name__ == "__main__":
    main()
