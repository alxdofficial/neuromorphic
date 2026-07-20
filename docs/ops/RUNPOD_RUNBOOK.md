# RunPod Runbook — hard-won operational lessons

Portable, project-agnostic runbook for renting and driving RunPod GPUs. Written to be **exported and reused
on other projects** — nothing here depends on this repo. Project-specific glue lives in
`docs/ops/runpod_workflow.md` and `scripts/pod/`.

Every rule below cost real time or money to learn. They are ordered by how much pain they save.

---

## 1. Disk: put everything on the LOCAL SSD, never the network volume

RunPod gives you two very different disks:

| | flag | mount | backing | speed | survives terminate? |
|---|---|---|---|---|---|
| **Container disk** | `container_disk_in_gb` | `/root`, `/tmp` | **local NVMe SSD** | fast | ❌ no |
| **Network volume** | `volume_in_gb` | `/workspace` | **moosefs over network** | **~3 MB/s writes** (measured) | ✅ yes |

- **Allocate your space as container disk.** `container_disk_in_gb=50..200`, `volume_in_gb=0` unless you
  genuinely need persistence across pod restarts.
- The network volume is **not** "slower disk" — for many-small-file workloads it is *pathologically* slow.
  Measured **~3 MB/s** on writes. A pip install of a torch stack there takes **20+ minutes**; the same
  install on the container SSD takes **~2 minutes**. ~10× difference, sometimes worse.
- Only put on `/workspace` what must survive: final results, and *maybe* big model weights (few large files,
  download-bound, so the FS penalty is small). Never envs, never repos, never pip caches.

### 1a. Corollary: ALL installs go on the SSD

Env, virtualenv/micromamba root, pip cache, and TMPDIR must all point at local disk:

```bash
export MAMBA_ROOT_PREFIX=/root/micromamba   # NOT /workspace/micromamba
export PIP_CACHE_DIR=/root/pipcache
export TMPDIR=/tmp
```

Write setup scripts so these are **overridable** and default to local disk. If a script hardcodes
`/workspace/...` for the env, you will silently eat the 10× penalty on every pod you ever launch.

---

## 2. Use RunPod's ALREADY-CACHED images — never a custom PyTorch image

RunPod pre-caches its own official images on host machines. Using one means the container starts in
**seconds**; anything else must be pulled over the network first (**minutes to tens of minutes**, and it is
billed time).

- **Rule: the image name must start with `runpod/`.** e.g.
  `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Pods created **manually in the RunPod web console default to these cached images** — which is exactly why
  console-created pods come up fast. Programmatic creation does *not* pick one for you; you must name it.
- **Before launching a fleet:** browse RunPod console → *Explore / Pytorch templates* and copy the exact tag
  you want. Match the tag to a pod you previously created by hand and know starts fast.
- **Match the CUDA build to the GPU generation.** A `cuda12.4` image is **Ampere/Ada only** (3090, 4090, A40,
  A6000, A5000). Blackwell (5090, RTX PRO 45xx) needs a **cu128** image or nothing will run.
- Do not build/push a custom image for a one-off campaign. Start from the cached image and install on top.

---

## 3. Loosen version pins rather than compiling from source

**A slightly different dependency version is almost always better than a 30–40 minute source compile.**

- If pip wants to build a wheel from source, **stop** and find a way out: relax the pin, take the nearest
  prebuilt wheel, or fetch an official prebuilt release artifact.
- **flash-attn is the canonical trap** — it has no PyPI wheel and compiles for 30–45 min. Use the project's
  prebuilt release wheels instead, matched on `(torch, cuda, python, cxx11abi)`:
  ```
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/
    flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
  ```
  Strip the source pin out of `requirements.txt` and install the wheel explicitly.
- **Install pinned torch ONCE from the right CUDA index, before `-r requirements.txt`.** Otherwise an
  unpinned `pip install torch` grabs the newest wheel (~3 GB, possibly wrong CUDA), and the requirements file
  then *downgrades* it — a second multi-GB download for nothing. Observed cost: **20+ wasted minutes.**
  ```bash
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
      --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt      # torch line already satisfied → not re-downloaded
  ```
- Record *why* each pin exists. An unexplained pin gets brute-forced by the next person; a documented one
  gets relaxed safely.

---

## 4. "The host has stalled" is usually OUTPUT BUFFERING

Before concluding a remote job is hung, **prove it**. Python buffers stdout when it is not a TTY (which is
exactly the case under `nohup`, tmux-piped-to-file, and ssh one-shots), so a perfectly healthy job can print
*nothing* for many minutes.

**Prevention** (do all of these by default):
```bash
python -u script.py                 # unbuffered
export PYTHONUNBUFFERED=1
stdbuf -oL -eL <cmd>                # for non-Python tools
grep --line-buffered ...            # every pipe stage must flush, or matches sit in a buffer
```

**Diagnosis — distinguish a real stall from a quiet one:**
```bash
nvidia-smi                          # GPU util >0 / memory held  → it is working
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
top -b -n1 | head -20               # CPU burning → working
py-spy dump --pid <PID>             # exactly which line it is on
ls -l --time-style=full-iso out.log # is the file still growing?
cat /proc/<PID>/status | grep State
```
A job with GPU memory held and CPU active is **not** stalled — you just cannot see it. Only if util is 0,
CPU is idle, *and* the log mtime is frozen should you suspect a genuine hang.

---

## 5. Detaching: tmux is mandatory

`setsid` and `nohup … & disown` **die** when a one-shot ssh session closes — RunPod's sshd tears down the
session's process group. Only a **tmux server** reliably survives disconnect.

```bash
ssh -i ~/.ssh/id_ed25519 -p <port> root@<host> \
  "tmux new-session -d -s work 'bash -lc \"cd /root && python -u run.py 2>&1 | tee /root/run.log\"'"
```
RunPod images ship **no tmux** and stale apt lists → `apt-get update && apt-get install -y tmux` first.

Also: a `&`-backgrounded prefetch inside a setup script dies when the script exits. Give long downloads their
own tmux session, or `wait` on them.

---

## 6. Billing hygiene

- **Poll every 1–2 minutes** while a pod is up. Background watchers frequently fail to wake you, and an idle
  pod bills exactly like a busy one.
- **Terminate the moment work completes.** Have a reaper that polls a status file and terminates on
  done/fail/cap, but treat it as a backstop, not a substitute for watching.
- **Set a hard spend cap up front**, and abort + terminate when hit.
- Confirm termination — a "stopped" pod with a network volume can still accrue storage charges.
- Retry pod creation: `"This machine does not have the resources"` is common. Loop a few attempts, then fall
  back to another GPU type or to Secure cloud.

---

## 7. GPU selection: measure, don't assume

- **Measure peak VRAM on one item before renting a fleet.** Sample during a real run:
  ```bash
  while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 3; done
  ```
- **Measure throughput per card, not just price.** The right metric is **cost per unit work**
  (`$/hr × seconds/op`), not `$/hr`. A card that is 2.3× faster at 1.6× the price is *cheaper*.
  Observed: an **Ada RTX 4090 ran ~2.3× faster than an Ampere A40** on the same 8B workload, making it
  ~30% cheaper per operation despite the higher hourly rate.
- **Bigger/pricier is not automatically faster.** An A100 gave no meaningful speedup over an A40 on a
  latency/overhead-bound single-instance job. Benchmark before paying the premium.
- **Community vs Secure:** Community is cheaper but frequently out of stock and can be reclaimed. Secure
  costs more and is reliably available. Probe availability before planning around a price.

---

## 8. Scaling: many small GPUs beat one big GPU

For workloads that **cannot batch within one process** (single-stream, overhead- or serialization-bound):

- **Packing multiple instances onto one big GPU does not work** — they serialize on compute. Measured only
  **~1.2×** from 3 co-located instances instead of the hoped 3×.
- **Data-parallel across separate GPUs is the only real speedup, and it is linear.** Total GPU-hours are
  fixed, so **wall-time ÷ N at ~constant total cost**. Renting 6 GPUs for 1 hour ≈ the price of 1 GPU for
  6 hours.
- **Shard at the unit that owns the cache.** If a built-up state is reused across many queries, shard by
  *that state*, never by query — otherwise every shard rebuilds the same expensive state. Assign whole
  cache-units to exactly one worker.
- **Watch shard granularity + balance.** With few, uneven units, round-robin creates stragglers and everyone
  waits on the slowest. Prefer **longest-first assignment to the lightest worker (LPT)**. Parallelism is
  capped by the number of units.
- **Prefer one multi-GPU host over N single-GPU hosts** when available: model weights download **once** into
  a shared cache and there is one env build and one ssh target. Across N hosts you pay the (often tens of GB)
  download N times.
- Make runs **resumable and idempotent** (append results per item, skip already-done ids) so a reclaimed pod
  costs minutes, not the whole run. Merge shard outputs and score the union once.

---

## 9. Pre-flight checklist

```
[ ] Image is runpod/* and CUDA build matches the GPU generation
[ ] container_disk_in_gb sized for the job; volume_in_gb = 0 unless persistence needed
[ ] Env / pip cache / TMPDIR all on local SSD (not /workspace)
[ ] torch pinned + installed from the correct CUDA index BEFORE requirements.txt
[ ] No source compiles in the install path (flash-attn etc. use prebuilt wheels)
[ ] Peak VRAM measured on one item; fits with headroom
[ ] Throughput measured; cost-per-unit-work compared across candidate cards
[ ] Long jobs launched inside tmux; python -u / PYTHONUNBUFFERED=1
[ ] Run is resumable + shard-mergeable
[ ] Spend cap set; reaper armed; poll cadence 1-2 min
```
