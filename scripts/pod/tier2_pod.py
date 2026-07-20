#!/usr/bin/env python3
"""Tier-2 pod interaction layer — the seamless agent<->pod bridge for the baseline eval run.

Unlike runpod.py (training: one-arm-per-pod, fire-and-forget through R2), the Tier-2 baseline run is
INTERACTIVE: the agent runs shell commands, launches the eval panel, tails buffered output, and rsyncs
scores / caches / checkpoints back — all against ONE multi-GPU Secure pod (3-4x H100 SXM; the panel
launcher run_pod_panel.sh puts one method per GPU on a single host, so one pod covers the whole panel and
the agent only ever juggles ONE ssh endpoint).

Design goals (why each subcommand exists):
  * seamless command exec         -> `exec`  : one-shot ssh, full stdout+stderr returned to the agent
  * non-blocking long runs        -> `run`   : detached tmux, UNBUFFERED (python -u) + tee to a logfile
  * see any buffered output       -> `logs`  : tail that logfile; the run never hides in a pipe buffer
  * pull scores/caches/ckpts back -> `sync`  : incremental rsync pod->local (or --push local->pod for edits)
  * one-glance progress           -> `status`: nvidia-smi + tmux ls + per-method cache row counts + disk

The pod id/endpoint is saved to state/tier2.json by `up`, so every other subcommand defaults to it — the
agent never hand-copies host:port. All ssh/rsync use the account key at ~/.ssh/id_ed25519 (injected by
RunPod into every pod).

Usage:
  tier2_pod.py up [--gpus 3] [--disk 120]      # create a multi-GPU Secure H100 pod, save its id
  tier2_pod.py bootstrap                        # copy tier2_bootstrap.sh up + run it (idempotent setup)
  tier2_pod.py exec "nvidia-smi"                # run a command, stream output back        (main hands-on)
  tier2_pod.py run panel "DATASET=longmemeval scripts/baselines/tier2/run_pod_panel.sh"   # detached tmux
  tier2_pod.py logs panel [-n 60]               # tail ~/logs/panel.log
  tier2_pod.py sync                             # rsync pod:outputs/baselines -> local
  tier2_pod.py sync --push                       # rsync local repo -> pod (ship uncommitted edits)
  tier2_pod.py status                           # gpu util + tmux sessions + cache counts + disk
  tier2_pod.py ssh                              # print the raw ssh command for manual poking
  tier2_pod.py down                             # terminate the pod
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# --- load the RunPod SDK, NOT this directory's runpod.py (same-dir name collision: sys.path[0] is
# scripts/pod, so a bare `import runpod` finds our sibling helper file instead of the pip package). Drop the
# script dir from sys.path for the import, then restore it. (runpod.py itself trips over this bug.) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_saved = list(sys.path)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
import runpod as _sdk                                                       # the pip SDK  # noqa: E402
sys.path[:] = _saved

REPO_LOCAL = Path(__file__).resolve().parents[2]                            # /home/alex/code/neuromorphic
REPO_REMOTE = "/root/neuromorphic"
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")
STATE = Path(__file__).parent / "state" / "tier2.json"
GPU_ID = "NVIDIA H100 80GB HBM3"                                            # H100 SXM (multi-GPU Secure, High stock)
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"         # Hopper-OK; nvcc for flash-attn/kernels


def _rp():
    import re
    cfg = Path("~/.runpod/config.toml").expanduser().read_text()
    m = re.search(r'apikey\s*=\s*"?([^"\n]+)', cfg)
    if not m:
        sys.exit("no apikey in ~/.runpod/config.toml")
    _sdk.api_key = m.group(1).strip()
    return _sdk


def _endpoint(pod: dict):
    """Direct ssh host:port from a pod's runtime ports (needs support_public_ip). None until it boots."""
    for p in ((pod.get("runtime") or {}).get("ports") or []):
        if p.get("privatePort") == 22 and p.get("ip"):
            return p["ip"], p["publicPort"]
    return None


def _save(pod_id, host=None, port=None):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"pod_id": pod_id, "host": host, "port": port}, indent=1))


def _load():
    if not STATE.exists():
        sys.exit("no active tier2 pod — run `tier2_pod.py up` first")
    return json.loads(STATE.read_text())


def _resolve(refresh=False):
    """Return (pod_id, host, port), refreshing the endpoint from the API if missing/asked."""
    st = _load()
    if st.get("host") and st.get("port") and not refresh:
        return st["pod_id"], st["host"], st["port"]
    rp = _rp()
    pod = next((p for p in rp.get_pods() if p.get("id") == st["pod_id"]), None)
    if not pod:
        sys.exit(f"pod {st['pod_id']} not found (terminated?)")
    ep = _endpoint(pod)
    if not ep:
        sys.exit("pod has no ssh endpoint yet — still booting; retry in ~30s")
    _save(st["pod_id"], ep[0], ep[1])
    return st["pod_id"], ep[0], ep[1]


def _ssh_opts(port):
    return ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-i", SSH_KEY, "-p", str(port)]


def _ssh(cmd, timeout=900, check=False):
    _pid, host, port = _resolve()
    r = subprocess.run(["ssh", *_ssh_opts(port), f"root@{host}", cmd],
                       capture_output=True, text=True, timeout=timeout)
    if r.stdout:
        sys.stdout.write(r.stdout)
    if r.stderr:
        sys.stderr.write(r.stderr)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    return r


def _rsync(src, dst, port, extra=()):
    ssh = "ssh " + " ".join(_ssh_opts(port))
    return subprocess.run(["rsync", "-az", "--info=stats1", "-e", ssh, *extra, src, dst],
                          text=True)


# --------------------------------------------------------------------------- subcommands
def cmd_list():
    rp = _rp()
    pods = rp.get_pods() or []
    if not pods:
        print("(no pods)")
    for p in pods:
        gpu = (p.get("machine") or {}).get("gpuDisplayName") or "?"
        ep = _endpoint(p)
        ssh = f"root@{ep[0]}:{ep[1]}" if ep else "(booting — no ssh endpoint yet)"
        print(f"{p.get('id')}  {p.get('name'):24}  x{p.get('gpuCount','?')} {gpu:14}  {p.get('desiredStatus'):8}  {ssh}")


def cmd_attach(pod_id=None):
    """Adopt an existing (e.g. hand-created) pod into the tier2 workflow: save its id + resolve its ssh
    endpoint. With no id, auto-picks the sole running pod (errors if there are several)."""
    rp = _rp()
    pods = rp.get_pods() or []
    if pod_id is None:
        if len(pods) == 1:
            pod_id = pods[0]["id"]
        else:
            ids = ", ".join(p["id"] for p in pods) or "(none)"
            sys.exit(f"multiple/zero pods — pass an id explicitly. Pods: {ids}")
    pod = next((p for p in pods if p.get("id") == pod_id), None)
    if not pod:
        sys.exit(f"pod {pod_id} not found on the account")
    _save(pod_id)
    ep = _endpoint(pod)
    if ep:
        _save(pod_id, ep[0], ep[1])
        print(f"attached {pod_id}  ->  root@{ep[0]}:{ep[1]}")
    else:
        print(f"attached {pod_id}  (still booting — ssh endpoint not ready; `status` will resolve it)")


def cmd_up(gpus=3, disk=120):
    rp = _rp()
    pod = rp.create_pod(
        name=f"neuro-tier2-{gpus}xh100", image_name=IMAGE, gpu_type_id=GPU_ID,
        cloud_type="SECURE", gpu_count=gpus, container_disk_in_gb=disk, volume_in_gb=0,
        ports="22/tcp,8888/http", support_public_ip=True, start_ssh=True)
    pid = pod.get("id")
    _save(pid)
    print(f"CREATED {pid}  [{gpus}x {GPU_ID} / SECURE, {disk}GB disk]")
    print("  wait ~60-90s for boot, then: tier2_pod.py status   (auto-resolves the ssh endpoint)")


def cmd_bootstrap():
    _pid, host, port = _resolve(refresh=True)
    boot = Path(__file__).parent / "tier2_bootstrap.sh"
    if not boot.exists():
        sys.exit(f"missing {boot}")
    print(f"[bootstrap] shipping creds + {boot.name} -> {host}:{port}")
    subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                    "-i", SSH_KEY, "-P", str(port), str(boot), f"root@{host}:/root/tier2_bootstrap.sh"],
                   check=True)
    hf = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(hf):
        _ssh("mkdir -p /root/.cache/huggingface")
        subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                        "-i", SSH_KEY, "-P", str(port), hf, f"root@{host}:/root/.cache/huggingface/token"])
    # run it in a detached tmux so a dropped ssh doesn't kill the ~30-40 min build; tail via `logs bootstrap`
    _ssh("apt-get update -y >/dev/null 2>&1 && apt-get install -y tmux rsync >/dev/null 2>&1; mkdir -p /root/logs")
    _ssh("tmux new-session -d -s bootstrap "
         "\"bash -lc 'bash /root/tier2_bootstrap.sh 2>&1 | tee /root/logs/bootstrap.log'\"")
    print("[bootstrap] launched in tmux. Watch:  tier2_pod.py logs bootstrap -n 40")


def cmd_exec(cmd):
    _ssh(f"cd {REPO_REMOTE} 2>/dev/null; {cmd}")


def cmd_run(name, cmd):
    """Launch cmd in a detached tmux 'name', unbuffered + teed to ~/logs/name.log (poll with `logs name`)."""
    _ssh("mkdir -p /root/logs")
    inner = f"cd {REPO_REMOTE} && PYTHONUNBUFFERED=1 {cmd}"
    esc = inner.replace("'", "'\\''")
    _ssh(f"tmux new-session -d -s {name} \"bash -lc '{esc} 2>&1 | tee /root/logs/{name}.log'\"", check=True)
    print(f"[run] '{name}' launched in tmux. Tail:  tier2_pod.py logs {name}")


def cmd_logs(name, n=60):
    _ssh(f"tail -n {n} /root/logs/{name}.log 2>/dev/null || echo '(no log yet for {name})'")


def cmd_sync(push=False):
    _pid, host, port = _resolve()
    if push:
        print(f"[sync] local repo -> {host}:{REPO_REMOTE}")
        # NB: anchor big-dir excludes to the repo ROOT (leading /) — a bare `data`/`outputs` would also
        # match src/memory/data/ (the dataset loaders!) and silently break load_items on the pod.
        _rsync(f"{REPO_LOCAL}/", f"root@{host}:{REPO_REMOTE}/", port,
               extra=["--exclude", "/.git", "--exclude", "/.venv", "--exclude", "/outputs",
                      "--exclude", "/data", "--exclude", "__pycache__", "--exclude", "*.pyc"])
    else:
        print(f"[sync] {host}:{REPO_REMOTE}/outputs/baselines -> local")
        (REPO_LOCAL / "outputs" / "baselines").mkdir(parents=True, exist_ok=True)
        _rsync(f"root@{host}:{REPO_REMOTE}/outputs/baselines/",
               f"{REPO_LOCAL}/outputs/baselines/", port)


def cmd_status():
    _ssh(
        "echo '=== GPUs ==='; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader 2>/dev/null; "
        "echo; echo '=== tmux sessions ==='; tmux ls 2>/dev/null || echo '(none)'; "
        f"echo; echo '=== cache row counts ==='; for f in {REPO_REMOTE}/outputs/baselines/cache/*.jsonl; do "
        "[ -e \"$f\" ] && echo \"$(wc -l <\"$f\") $(basename \"$f\")\"; done 2>/dev/null || echo '(no caches yet)'; "
        "echo; echo '=== disk ==='; df -h / 2>/dev/null | tail -1")


def cmd_ssh():
    _pid, host, port = _resolve()
    print(f"ssh {' '.join(_ssh_opts(port))} root@{host}")


def cmd_down():
    st = _load()
    _rp().terminate_pod(st["pod_id"])
    print(f"terminated {st['pod_id']}")
    STATE.unlink(missing_ok=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    c = a[0]
    if c == "up":
        g = int(a[a.index("--gpus") + 1]) if "--gpus" in a else 3
        d = int(a[a.index("--disk") + 1]) if "--disk" in a else 120
        cmd_up(g, d)
    elif c == "list":
        cmd_list()
    elif c == "attach":
        cmd_attach(a[1] if len(a) > 1 else None)
    elif c == "bootstrap":
        cmd_bootstrap()
    elif c == "exec":
        cmd_exec(a[1])
    elif c == "run":
        cmd_run(a[1], a[2])
    elif c == "logs":
        n = int(a[a.index("-n") + 1]) if "-n" in a else 60
        cmd_logs(a[1], n)
    elif c == "sync":
        cmd_sync(push=("--push" in a))
    elif c == "status":
        cmd_status()
    elif c == "ssh":
        cmd_ssh()
    elif c == "down":
        cmd_down()
    else:
        sys.exit(__doc__)
