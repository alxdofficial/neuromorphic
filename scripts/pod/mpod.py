#!/usr/bin/env python3
"""Multi-pod orchestrator for data-parallel Tier-2 baseline runs (N pods, 1 GPU each, 1 shard each).

Unlike single-GPU instance-packing (which fails for M+ — injection is compute-bound and serializes),
data-parallel across SEPARATE pods gives true linear N× speedup. Each pod runs
`run_memoryllm.py --num-shards N --shard-idx K` over a disjoint 1/N of the contexts; merge_shards.py
combines them at the end. All storage is on each pod's LOCAL container disk (fast; no slow network volume).

Commands:
  create N [--gpu 'NVIDIA GeForce RTX 4090'] [--secure]   # create N pods, save state
  endpoints                                               # resolve+save ssh host:port for all (wait boot)
  setup                                                   # parallel: base tools + repo + env + weights
  setup-status                                            # env-done + weights per pod
  launch <dataset>                                        # shard K -> pod K (full run, no subsample)
  status [--rate]                                         # progress + alive + budget across pods
  sync                                                    # pull every pod's outputs/baselines -> local
  respawn K                                               # replace dead pod K + re-setup + resume its shard
  down                                                    # terminate ALL pods
State: scripts/pod/.mpod_state.json  (list of pods + rate/budget refs).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_LOCAL = HERE.parents[1]                    # /home/alex/code/neuromorphic/code
REPO_REMOTE = "/root/neuromorphic"
SSH_KEY = str(Path("~/.ssh/id_ed25519").expanduser())
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
STATE = HERE / ".mpod_state.json"
SETUP = f"{REPO_REMOTE}/scripts/pod/tier2_setup/setup_memoryllm.sh"
ENV_PY = "/root/micromamba/envs/memoryllm/bin/python"
REPODIR = "/root/tier2_repos/MemoryLLM"


def _rp():
    import re
    # avoid this dir's runpod.py (same-dir name collision) — drop script dir from sys.path for the import
    saved = sys.path[:]
    sys.path = [p for p in sys.path if p not in ("", str(HERE))]
    try:
        import runpod as sdk
    finally:
        sys.path = saved
    cfg = Path("~/.runpod/config.toml").expanduser().read_text()
    sdk.api_key = re.search(r'apikey\s*=\s*"?([^"\n]+)', cfg).group(1).strip()
    return sdk


def _load() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"pods": []}


def _save(st: dict):
    STATE.write_text(json.dumps(st, indent=1))


def _opts(port):
    return ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-i", SSH_KEY, "-p", str(port)]


def _ssh(host, port, cmd, timeout=900):
    return subprocess.run(["ssh", *_opts(port), f"root@{host}", cmd],
                          capture_output=True, text=True, timeout=timeout)


def _rsync_up(host, port):
    ssh = "ssh " + " ".join(_opts(port))
    subprocess.run(["rsync", "-az", "--info=stats1", "-e", ssh,
                    "--exclude", "/.git", "--exclude", "/.venv", "--exclude", "/outputs",
                    "--exclude", "/data", "--exclude", "__pycache__", "--exclude", "*.pyc",
                    f"{REPO_LOCAL}/", f"root@{host}:{REPO_REMOTE}/"], text=True)


def _rsync_down(host, port, k):
    ssh = "ssh " + " ".join(_opts(port))
    dst = REPO_LOCAL / "outputs" / "baselines"
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-az", "-e", ssh,
                    f"root@{host}:{REPO_REMOTE}/outputs/baselines/", f"{dst}/"], text=True)


def _endpoint(pod):
    for p in ((pod.get("runtime") or {}).get("ports") or []):
        if p.get("privatePort") == 22 and p.get("ip"):
            return p["ip"], p["publicPort"]
    return None


# --------------------------------------------------------------------------------------------------
def cmd_create(args):
    rp = _rp()
    cloud = "SECURE" if args.secure else "COMMUNITY"
    pods = []
    for k in range(args.n):
        p = None
        for attempt in range(5):                    # Community availability is flaky → retry across machines
            try:
                p = rp.create_pod(name=f"neuro-mplus-sh{k}", image_name=IMAGE, gpu_type_id=args.gpu,
                                  cloud_type=cloud, gpu_count=1, container_disk_in_gb=args.disk, volume_in_gb=0,
                                  ports="22/tcp", support_public_ip=True, start_ssh=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"[create] shard {k} attempt {attempt + 1}/5 failed: {str(e)[:90]}; retry ...")
                time.sleep(6)
        if p is None:
            print(f"[create] shard {k} could NOT be provisioned after 5 tries — {len(pods)} pods up so far")
            break
        pods.append({"idx": k, "id": p["id"], "host": None, "port": None})
        print(f"[create] shard {k} -> pod {p['id']} ({args.gpu}, {cloud})")
    _save({"pods": pods, "num_shards": args.n, "gpu": args.gpu, "secure": args.secure,
           "ref_epoch": time.time(), "rate_per_pod": 0.34 if not args.secure else 0.69})
    print(f"[create] {args.n} pods created. Next: mpod.py endpoints")


def cmd_endpoints(args):
    rp = _rp()
    st = _load()
    for _try in range(40):
        live = {p["id"]: p for p in rp.get_pods()}
        missing = 0
        for pod in st["pods"]:
            if pod.get("host"):
                continue
            ep = _endpoint(live.get(pod["id"], {}))
            if ep:
                pod["host"], pod["port"] = ep
                print(f"[endpoints] shard {pod['idx']} -> {ep[0]}:{ep[1]}")
            else:
                missing += 1
        _save(st)
        if not missing:
            print("[endpoints] all resolved."); return
        print(f"[endpoints] {missing} still booting; retry in 20s ...")
        time.sleep(20)
    sys.exit("[endpoints] timed out waiting for ssh endpoints")


def _setup_one(pod):
    h, p, k = pod["host"], pod["port"], pod["idx"]
    log = f"[setup sh{k}]"
    # base tools (fresh image has no rsync/tmux)
    _ssh(h, p, "apt-get update -y >/dev/null 2>&1 && apt-get install -y rsync tmux git >/dev/null 2>&1", timeout=300)
    _rsync_up(h, p)
    # env setup in tmux (all-local: WORKDIR=/root), then weights download in a second tmux
    _ssh(h, p, f"tmux new-session -d -s setup 'WORKDIR=/root REPO={REPO_REMOTE} HF_HOME=/root/hf "
               f"bash {SETUP} 2>&1 | tee /root/logs_setup.txt'", timeout=60)
    return f"{log} launched (base tools + repo synced + setup tmux)"


def cmd_setup(args):
    st = _load()
    with ThreadPoolExecutor(max_workers=len(st["pods"])) as ex:
        for r in ex.map(_setup_one, st["pods"]):
            print(r)
    print("[setup] all launched. Poll: mpod.py setup-status")


def _dl_weights(pod):
    h, p, k = pod["host"], pod["port"], pod["idx"]
    _ssh(h, p, f"tmux new-session -d -s dlw 'HF_HOME=/root/hf {ENV_PY} -c "
               f"\"from huggingface_hub import snapshot_download; snapshot_download(\\'YuWangX/mplus-8b\\')\"'",
         timeout=60)
    return f"[dlw sh{k}] weights download launched"


def cmd_setup_status(args):
    st = _load()
    for pod in st["pods"]:
        h, p, k = pod["host"], pod["port"], pod["idx"]
        r = _ssh(h, p, "test -f /root/.setup_memoryllm.env.done && echo ENVDONE || echo env-building; "
                       f"du -sh /root/hf/hub/models--YuWangX--mplus-8b 2>/dev/null | cut -f1 || echo no-weights",
                 timeout=60)
        out = r.stdout.replace("\n", " ").strip()
        env_done = "ENVDONE" in out
        # kick off weights download once env is done and not started
        if env_done and "no-weights" in out:
            _dl_weights(pod)
        print(f"[setup-status] sh{k}: {out}")


def cmd_launch(args):
    st = _load()
    n = st["num_shards"]
    for pod in st["pods"]:
        h, p, k = pod["host"], pod["port"], pod["idx"]
        cmd = (f"cd {REPO_REMOTE} && HF_HOME=/root/hf CUDA_VISIBLE_DEVICES=0 {ENV_PY} "
               f"scripts/baselines/tier2/run_memoryllm.py --dataset {args.dataset} "
               f"--num-shards {n} --shard-idx {k} --no-bem --repo-dir {REPODIR}")
        _ssh(h, p, f"tmux kill-session -t run 2>/dev/null; tmux new-session -d -s run '{cmd} 2>&1 | tee /root/logs_run.txt'", timeout=60)
        print(f"[launch sh{k}] {args.dataset} --num-shards {n} --shard-idx {k}")
    st["dataset"] = args.dataset
    _save(st)


def cmd_status(args):
    st = _load()
    rp = _rp()
    live = {p["id"]: p for p in rp.get_pods()}
    up_h = (time.time() - st["ref_epoch"]) / 3600
    spend = 9.52 + up_h * st["rate_per_pod"] * len(st["pods"])   # 9.52 = sunk (A40+A100)
    print(f"[budget] {len(st['pods'])} pods x ${st['rate_per_pod']}/hr, up {up_h:.2f}h → total spend ~${spend:.2f}"
          f"  {'### OVER $23 — sync+merge+down ###' if spend >= 23 else 'OK'}")
    total = 0
    for pod in st["pods"]:
        h, p, k = pod["host"], pod["port"], pod["idx"]
        alive_api = pod["id"] in live
        r = _ssh(h, p, "tmux ls 2>/dev/null | grep -q run && echo RUN || echo done; "
                       f"cat {REPO_REMOTE}/outputs/baselines/cache/*sh{k}of*.jsonl 2>/dev/null | wc -l; "
                       "tmux capture-pane -pt run -S -40 2>/dev/null | grep -E 'Traceback|OutOfMemory' | tail -1",
                 timeout=60)
        lines = [x for x in r.stdout.strip().split("\n") if x]
        run_state = lines[0] if lines else "SSH-FAIL"
        rows = lines[1] if len(lines) > 1 and lines[1].isdigit() else "?"
        err = next((x for x in lines if "Trace" in x or "Memory" in x), "")
        try:
            total += int(rows)
        except (ValueError, TypeError):
            pass
        print(f"  sh{k}: {run_state:8} rows={rows:>4} api_alive={alive_api} {('ERR:'+err) if err else ''}")
    print(f"[status] total rows across shards: {total}")


def cmd_sync(args):
    st = _load()
    for pod in st["pods"]:
        print(f"[sync] pulling sh{pod['idx']} ({pod['host']}) ...")
        _rsync_down(pod["host"], pod["port"], pod["idx"])


def cmd_down(args):
    st = _load()
    rp = _rp()
    for pod in st["pods"]:
        try:
            rp.terminate_pod(pod["id"]); print(f"[down] terminated sh{pod['idx']} {pod['id']}")
        except Exception as e:  # noqa: BLE001
            print(f"[down] sh{pod['idx']} {pod['id']}: {e}")
    STATE.unlink(missing_ok=True)


def cmd_respawn(args):
    """Replace a dead pod K, re-setup, resume its shard (resumable cache)."""
    st = _load()
    rp = _rp()
    pod = next(p for p in st["pods"] if p["idx"] == args.k)
    try:
        rp.terminate_pod(pod["id"])
    except Exception:  # noqa: BLE001
        pass
    cloud = "SECURE" if st.get("secure") else "COMMUNITY"
    np = rp.create_pod(name=f"neuro-mplus-sh{args.k}", image_name=IMAGE, gpu_type_id=st["gpu"],
                       cloud_type=cloud, gpu_count=1, container_disk_in_gb=100, volume_in_gb=0,
                       ports="22/tcp", support_public_ip=True, start_ssh=True)
    pod["id"], pod["host"], pod["port"] = np["id"], None, None
    _save(st)
    print(f"[respawn] shard {args.k} -> new pod {np['id']}. Run: endpoints, setup, setup-status, then launch")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create"); c.add_argument("n", type=int); c.add_argument("--gpu", default="NVIDIA GeForce RTX 4090"); c.add_argument("--secure", action="store_true"); c.add_argument("--disk", type=int, default=50)
    sub.add_parser("endpoints")
    sub.add_parser("setup")
    sub.add_parser("setup-status")
    lp = sub.add_parser("launch"); lp.add_argument("dataset", choices=["longmemeval", "memoryagentbench"])
    st = sub.add_parser("status"); st.add_argument("--rate", action="store_true")
    sub.add_parser("sync")
    sub.add_parser("down")
    rs = sub.add_parser("respawn"); rs.add_argument("k", type=int)
    args = ap.parse_args()
    {"create": cmd_create, "endpoints": cmd_endpoints, "setup": cmd_setup, "setup-status": cmd_setup_status,
     "launch": cmd_launch, "status": cmd_status, "sync": cmd_sync, "down": cmd_down, "respawn": cmd_respawn}[args.cmd](args)


if __name__ == "__main__":
    main()
