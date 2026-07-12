#!/usr/bin/env python3
"""RunPod remote-training helper — create / drive / list / reap / terminate pods for the arch sweep.

RunPod is the remote GPU provider for the arch sweep. `bootstrap.sh` is provider-agnostic — it just
needs the R2 env + ARM/STEPS/BATCH + the HF token — so this file only handles pod lifecycle + the drive.

Auth: `~/.runpod/config.toml` apikey (matches the account SSH key at `~/.ssh/id_ed25519`).
R2 creds: `~/.config/r2/credentials`. HF token: `~/.cache/huggingface/token`.

Usage:
  python scripts/pod/runpod.py create <arm> [--gpu 4090]      # one pod, GPU fallback chain
  python scripts/pod/runpod.py drive  <pod_id> <arm> <run_id> [ref]  # ship creds + launch bootstrap in tmux (ref=SHA to pin; default HEAD)
  python scripts/pod/runpod.py list                           # pods + ssh endpoints
  python scripts/pod/runpod.py reap   <run_id>                # terminate pods whose R2 _STATUS is terminal
  python scripts/pod/runpod.py terminate <pod_id|all>

Full runbook (GPU selection, the launch-bound perf finding, gotchas): docs/ops/runpod_workflow.md.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Image is Ampere/Ada only (CUDA 12.4.1) — do NOT pick Blackwell (5090 / RTX PRO 45xx), which need cu128.
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")
# GPU fallback chain: every entry fits B=6 (>=24GB) AND runs on the cu124 image (Ampere/Ada). 4090 first
# (fastest/cheapest when in stock); A40/A6000 (48GB Ampere) are the reliable fallback when 4090s are out.
GPU_CHAIN = [
    ("NVIDIA GeForce RTX 4090", "COMMUNITY"), ("NVIDIA GeForce RTX 4090", "SECURE"),
    ("NVIDIA A40", "SECURE"), ("NVIDIA RTX A6000", "SECURE"), ("NVIDIA RTX A6000", "COMMUNITY"),
    ("NVIDIA RTX A5000", "COMMUNITY"), ("NVIDIA GeForce RTX 3090", "COMMUNITY"),
]
GPU_ALIAS = {"4090": "NVIDIA GeForce RTX 4090", "a40": "NVIDIA A40", "a6000": "NVIDIA RTX A6000",
             "a5000": "NVIDIA RTX A5000", "3090": "NVIDIA GeForce RTX 3090"}

DEFAULT_ARMS = ["icae_baseline", "autocompressor_baseline", "titans_baseline",
                "gisting_baseline", "memoryllm_baseline", "slotgraph_baseline"]


def _rp():
    import runpod
    cfg = Path("~/.runpod/config.toml").expanduser().read_text()
    m = re.search(r'apikey\s*=\s*"?([^"\n]+)', cfg)
    if not m:
        sys.exit("no apikey in ~/.runpod/config.toml")
    runpod.api_key = m.group(1).strip()
    return runpod


def _endpoint(pod: dict):
    """Direct SSH host:port from a pod's runtime ports (needs support_public_ip). None if not ready."""
    rt = pod.get("runtime") or {}
    for p in (rt.get("ports") or []):
        if p.get("privatePort") == 22 and p.get("ip"):
            return p["ip"], p["publicPort"]
    return None


def _r2_env():
    """Load R2_* from ~/.config/r2/credentials into a dict for aws-cli calls."""
    env = dict(os.environ)
    cred = Path("~/.config/r2/credentials").expanduser()
    if cred.exists():
        for line in cred.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def cmd_create(arm: str, gpu: str | None):
    rp = _rp()
    chain = [(GPU_ALIAS[gpu], "COMMUNITY"), (GPU_ALIAS[gpu], "SECURE")] if gpu else GPU_CHAIN
    for gpu_id, cloud in chain:
        try:
            pod = rp.create_pod(
                name=f"neuro-{arm.replace('_baseline', '')}", image_name=IMAGE, gpu_type_id=gpu_id,
                cloud_type=cloud, gpu_count=1, container_disk_in_gb=20, volume_in_gb=40,
                volume_mount_path="/workspace", ports="22/tcp,8888/http",
                support_public_ip=True, start_ssh=True)
            print(f"CREATED {arm} -> {pod.get('id')}  [{gpu_id} / {cloud}]")
            return pod.get("id")
        except Exception as e:
            print(f"  {gpu_id}/{cloud}: {str(e)[:80]}")
    sys.exit(f"no GPU available for {arm} across the fallback chain — retry later or widen GPU_CHAIN")


def cmd_list():
    rp = _rp()
    pods = rp.get_pods()
    if not pods:
        print("(no pods)")
        return
    for p in pods:
        gpu = (p.get("machine") or {}).get("gpuDisplayName") or "?"
        ep = _endpoint(p)
        ssh = f"root@{ep[0]}:{ep[1]}" if ep else "(no ssh-22 endpoint yet — still booting)"
        print(f"{p.get('id')}  {p.get('name'):22}  {gpu:14}  {p.get('desiredStatus'):8}  {ssh}")


def _ssh(host, port, cmd, timeout=60):
    opts = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=15", "-i", SSH_KEY, "-p", str(port)]
    return subprocess.run(["ssh", *opts, f"root@{host}", cmd], capture_output=True, text=True, timeout=timeout)


def _scp(host, port, src, dst):
    opts = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=15", "-i", SSH_KEY, "-P", str(port)]
    return subprocess.run(["scp", *opts, src, f"root@{host}:{dst}"], capture_output=True, text=True, timeout=120)


def _resolve_ref(ref: str | None) -> str:
    """Pin to an immutable SHA so every arm of a sweep trains the SAME code. Default = local HEAD (the
    commit you just pushed). Passing a floating branch (`main`) is allowed but discouraged — arms driven
    at different times could then clone different tips and silently mix code versions in the bake-off."""
    import subprocess as _sp
    r = _sp.run(["git", "rev-parse", ref or "HEAD"], capture_output=True, text=True,
                cwd=str(Path(__file__).resolve().parents[2]))
    return r.stdout.strip() or (ref or "HEAD")


def cmd_drive(pod_id: str, arm: str, run_id: str, ref: str | None = None, steps=8000, batch=6):
    """Ship creds + code, then launch bootstrap.sh in a detached tmux (the ONLY reliable detach on RunPod —
    setsid/`& disown` die when the one-shot ssh closes). Pins the training commit via REPO_REF=<sha> so
    all arms of a run_id train identical code (defaults to local HEAD; pass a SHA to pin explicitly)."""
    rp = _rp()
    sha = _resolve_ref(ref)
    print(f"  [pin] {arm}: REPO_REF={sha}")
    pod = next((p for p in rp.get_pods() if p.get("id") == pod_id), None)
    if not pod:
        sys.exit(f"pod {pod_id} not found")
    ep = None
    for _ in range(30):                                  # pods boot slowly; wait for the ssh-22 endpoint
        pod = next((p for p in rp.get_pods() if p.get("id") == pod_id), None)
        ep = _endpoint(pod) if pod else None
        if ep and _ssh(*ep, "echo ok", timeout=25).stdout.strip() == "ok":
            break
        print("  waiting for sshd…"); time.sleep(10)
    else:
        sys.exit("pod never became ssh-reachable")
    host, port = ep
    print(f"driving {arm} on {pod_id} ({host}:{port})")
    _ssh(host, port, "mkdir -p /root/.config/r2")
    _scp(host, port, os.path.expanduser("~/.config/r2/credentials"), "/root/.config/r2/credentials")
    tok = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(tok):
        _scp(host, port, tok, "/root/.config/hf_token")
    _scp(host, port, str(Path(__file__).parent / "bootstrap.sh"), "/workspace/bootstrap.sh")
    _ssh(host, port, "apt-get update -y >/dev/null 2>&1 && apt-get install -y tmux >/dev/null 2>&1")
    launch = (
        f"tmux new-session -d -s {arm} \"bash -lc 'set -a; source /root/.config/r2/credentials; set +a; "
        f"export ARM={arm} STEPS={steps} BATCH={batch} OUT_TAG={run_id} RUN_ID={run_id} REPO_REF={sha} MAX_HOURS=8; "
        f"exec bash /workspace/bootstrap.sh' > /workspace/{arm}.log 2>&1\"")
    _ssh(host, port, launch)
    r = _ssh(host, port, "tmux ls")
    print(f"  tmux: {r.stdout.strip() or r.stderr.strip()}")
    print(f"  watch:  ssh -i {SSH_KEY} -p {port} root@{host} 'tail -f /workspace/bootstrap.log'")


def cmd_reap(run_id: str):
    """Billing safety: terminate any pod whose arm's R2 _STATUS is terminal (DONE/FAILED/TIMEOUT)."""
    rp = _rp()
    env = _r2_env()
    prefix = env.get("R2_PREFIX", "neuromorphic")
    bucket, endpoint = env.get("R2_BUCKET"), env.get("R2_ENDPOINT")
    reaped = 0
    for p in rp.get_pods():
        arm = (p.get("name") or "").replace("neuro-", "") + "_baseline"
        key = f"s3://{bucket}/{prefix}/results/{run_id}/{arm}/_STATUS"
        out = subprocess.run(["aws", "s3", "--endpoint-url", endpoint, "cp", key, "-"],
                             capture_output=True, text=True, env=env).stdout.strip()
        if out.startswith(("DONE", "FAILED", "TIMEOUT")):
            rp.terminate_pod(p["id"])
            print(f"reaped {p['id']} ({arm}): {out[:40]}"); reaped += 1
        else:
            print(f"keep   {p['id']} ({arm}): {out[:40] or 'no status yet'}")
    print(f"reaped {reaped} pod(s)")


def cmd_terminate(target: str):
    rp = _rp()
    pods = rp.get_pods()
    ids = [p["id"] for p in pods] if target == "all" else [target]
    for pid in ids:
        rp.terminate_pod(pid); print(f"terminated {pid}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    cmd = a[0]
    if cmd == "create":
        gpu = a[a.index("--gpu") + 1] if "--gpu" in a else None
        cmd_create(a[1], gpu)
    elif cmd == "drive":
        cmd_drive(a[1], a[2], a[3], a[4] if len(a) > 4 else None)   # drive <pod> <arm> <run> [ref/sha]
    elif cmd == "list":
        cmd_list()
    elif cmd == "reap":
        cmd_reap(a[1])
    elif cmd == "terminate":
        cmd_terminate(a[1])
    else:
        sys.exit(__doc__)
