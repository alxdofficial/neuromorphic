# vast.ai + Cloudflare R2 remote-training runbook

A runbook for offloading the arch-sweep training to rented GPUs. Scripts live in
`scripts/pod/`; this doc is the *why* + the gotchas that aren't obvious from the code.
Read `scripts/pod/README.md` for the terse command list; read this for the reasoning,
the setup, and the bugs already solved so you don't re-discover them.

TL;DR of the model: **one pod per training arm.** Code comes from public GitHub, data
from R2, results go back to R2. A **local** watchdog reaps pods. The billing-capable
vast.ai key **never leaves the local machine**. Account credit is itself a hard spend
ceiling.

---

## 0. VALIDATED end-to-end (2026-07-11 live test) + the gotchas it found

A full single-arm run (icae_baseline, B=8, behavioral_kl, 5090) was driven on a real pod:
clone → deps → R2 data → 30 steps → checkpoint. **TRAIN_EXIT=0, peak_vram 22.1 GB, 30 steps
in 1.1 min** (matches local). Four setup gotchas were found and folded into the scripts:

1. **Use vast's OWN image, not Docker Hub.** `vastai/pytorch:cuda-13.0.3-auto` is pre-cached on
   hosts → container ready in **seconds**. The raw `pytorch/pytorch:2.8.0-cuda12.8` image on a cold
   host took **505 s (8.4 min)** to become ready (pull + apt + locale-gen) = pure billed idle.
   `launch.py` now uses `vastai/pytorch:cuda-13.0.3-auto` for every tier (torch 2.12/cu130, Blackwell
   sm_120 + bf16 verified on a 5090). The `-auto` tag picks the torch build matching the host driver.
2. **The vast image ships torch in a venv at `/venv/main`** — bare `python3` is the *system*
   interpreter with **no torch**, so unactivated it would pip-reinstall torch (GBs, minutes).
   `bootstrap.sh` now `source /venv/main/bin/activate` before anything, and uses **`uv pip install`**
   (present on the image) → all repo deps in **~55 s**.
3. **Gated HF dep:** the FineWeb `--src-tokenizer` is **`meta-llama/Llama-3.2-1B` (GATED)**. Needs a
   Meta-approved HF token or the reconstruct/continuation tasks 401. `drive.sh` ships the local token
   (`~/.cache/huggingface/token`) to the pod over SSH; bootstrap exports it as `HF_TOKEN`.
4. **The vast image sets `HF_HOME=/workspace/.hf_home`** → a token file at `~/.cache/huggingface/token`
   is IGNORED. Must use the **`HF_TOKEN` env var** (path-independent, highest priority) — which is what
   bootstrap now does. (Also: SmolLM2-135M backbone is NOT gated, downloads fine; R2 819 MB pull = ~75 s;
   vast SSH can be transiently flaky — retry.)

---

## 1. One-time setup

### 1a. vast.ai
- CLI: `pip install vastai` (in the repo `.venv`). Version 1.3.0 tested.
- Key file: `~/.config/vastai/api_key` (chmod 600). Our scripts read it and pass it as
  `VAST_API_KEY` env to each `vastai` call — **single source of truth**.
- The CLI *also* has its own store at `~/.config/vastai/vast_api_key` (written by
  `vastai set api-key`). If `VAST_API_KEY` env is set it **takes precedence** over that
  store (the CLI even prints "Unset VAST_API_KEY to use the saved key"). We always set
  the env, so the store is irrelevant.
- Auth check: `VAST_API_KEY=$(cat ~/.config/vastai/api_key) vastai show user`.

> ⚠️ **Two accounts exist; only one is funded.** One has **$0** credit (useless — pods
> won't spawn); the other holds the working credit and is the one saved locally. If pods
> refuse to launch, first check `vastai show user` → which account + credit. A $0 account
> fails silently-ish (creates a *stopped* instance).

### 1b. Cloudflare R2 (S3-compatible, **zero egress fees**)
- Creds file: `~/.config/r2/credentials` (chmod 600), shell-sourceable `K=V` lines:
  ```
  R2_ACCOUNT_ID=<R2_ACCOUNT_ID>
  R2_ACCESS_KEY_ID=...
  R2_SECRET_ACCESS_KEY=...
  R2_ENDPOINT=https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com
  R2_BUCKET=gen-purp-bucket
  R2_PREFIX=neuromorphic
  ```
- Access via the plain `aws` CLI with `--endpoint-url $R2_ENDPOINT` and
  `AWS_ACCESS_KEY_ID/SECRET` set from the R2 keys, `AWS_DEFAULT_REGION=auto`.
- Wrapper: `scripts/pod/r2.sh {up|down|ls|rm|raw}` namespaces everything under
  `$R2_BUCKET/$R2_PREFIX/`. Use `raw` as an escape hatch for arbitrary `aws s3` args.
- **Streaming works** and we rely on it for status markers:
  `echo msg | aws s3 cp - s3://…/_STATUS` (upload) and `aws s3 cp s3://…/_STATUS -`
  (download; nonzero exit if the key is missing → treat as "not written yet").
- HuggingFace has no relevant limits for our pull sizes; R2 zero-egress is why we stage
  the 820 MB `data.tar.gz` there instead of re-pulling from HF on every pod.

### 1c. SSH key (for the tmux→ssh flow)
- Your local pubkey must be registered with vast **once**:
  `scripts/pod/drive.sh --register-key` (uploads `~/.ssh/id_ed25519.pub` or `id_rsa.pub`).
- Without it, `--mode ssh` pods accept no connection.

---

## 2. The data tarball (already uploaded)
`scripts/pod/pack_data.sh` tars the 5-task **training** subset and uploads it to
`neuromorphic/data.tar.gz` (~820 MB). Sources: `fineweb_edu pile redpajama code squad
triviaqa hotpot_train musique_train multiwoz`. `babi` is pulled from HF at runtime;
`bio` is procedurally generated — neither is in the tarball. Phase-2/eval sources are
deliberately excluded to keep it lean. Re-run only when the training data changes.

---

## 3. The workflow (end to end)

```bash
# 0. dry-run: plan + worst-case cost, spends nothing (DEFAULT)
python scripts/pod/launch.py --arms slotgraph_baseline icae_baseline

# 1. PRECONDITION: pods clone a git ref → push the code first
git add scripts/pod && git commit -m "pod scripts" && git push origin main

# 2. launch bare ssh pods (spends money)
python scripts/pod/launch.py \
  --arms icae_baseline autocompressor_baseline titans_baseline \
         gisting_baseline memoryllm_baseline slotgraph_baseline \
  --gpu 4090 --ref main --steps 2000 --go
#   → writes scripts/pod/state/podrun-NNNN.json

# 3. start training on every pod (ssh + creds over the encrypted channel)
scripts/pod/drive.sh podrun-NNNN

# 4. billing backstop — leave running in its own terminal
python scripts/pod/watchdog.py podrun-NNNN

# 5. when _STATUS says DONE, pull + evaluate
scripts/pod/pull_results.sh podrun-NNNN --status     # quick poll
scripts/pod/pull_results.sh podrun-NNNN              # download results
python scripts/diagnostics/mixed/mixed_band_gate_eval.py --out-tag podrun-NNNN
```

### The tmux → ssh interactive flow (what the user recommends)
Run everything **inside a local tmux session** so a dropped laptop connection doesn't
kill the launcher/watchdog. To watch or debug a single pod live:

```bash
# scripted: opens ssh -t and attaches the pod's remote tmux session named after the arm
scripts/pod/drive.sh podrun-NNNN --attach slotgraph_baseline    # Ctrl-b d to detach

# manual equivalent:
VAST_API_KEY=$(cat ~/.config/vastai/api_key) vastai ssh-url <instance_id>   # ssh://root@HOST:PORT
ssh -p PORT root@HOST
tmux attach -t slotgraph_baseline        # bootstrap runs inside this session
tail -f /workspace/boot.log              # or watch the log directly
```

`drive.sh` does the non-obvious parts for you: it waits for sshd (pods boot slowly),
`scp`s the R2 creds **over the encrypted channel** (so secrets never touch vast
metadata), and starts `bootstrap.sh` in a detached remote `tmux` per arm.

---

## 4. GPU selection

**Workload shape matters more than the spec sheet.** We train a *frozen 135M* model at
*small batch* over *8 sequential streaming windows* (each depends on the previous). That
is **latency-bound on many small kernels**, not throughput-bound — so a big datacenter
GPU (A100/H100) sits mostly idle, and even a 5090's extra SMs/bandwidth are only partly
used. Clock speed + per-kernel latency + `torch.compile`/CUDA-graphs matter most.

Recommended tiers (`--gpu` flag; sets card + compute-cap floor + matching image):

| tier | card | ~$/hr (×6) | VRAM | image / torch | notes |
|------|------|-----------|------|---------------|-------|
| `4090` (default) | RTX 4090 | ~$2.1/hr | 24 GB | torch 2.4.1 / cu121 | **safe pick**, huge jump over Ampere, zero image risk |
| `5090` | RTX 5090 | ~$2.1/hr | 32 GB | torch 2.8.0 / cu128 | faster + more VRAM + torch closer to local 2.10; Blackwell freshness risk |
| `3090` | RTX 3090 | ~$0.8/hr | 24 GB | torch 2.4.1 / cu121 | cheapest, but Ampere clocks/bf16 are noticeably slower |

- At current prices **4090 ≈ 5090** (both ~$2.1/hr for 6 pods, worst-case ~$17 of the
  $23.65 credit at the 8h cap; a real 2000-step pilot finishes in ~1–3h ≈ $4–9).
- **4090 = de-risked default.** For a first run where the *comparison's* correctness
  matters and debugging happens on paid boxes, take the known-good path.
- **5090 = valid, one flag away** (`--gpu 5090`). Same price, ~10–30% faster in practice
  (not the 1.5–2× the specs imply, because of the latency bound), 32 GB headroom, and
  torch 2.8 is *closer to local torch 2.10* → better fidelity to reference numbers. Only
  residual risk: Blackwell + `torch.compile` edge cases.
- The bigger speed lever on a 135M model is often **`torch.compile` + dataloader**, not
  the card — worth confirming that's on before assuming the GPU is the bottleneck.

---

## 5. Billing safety (four independent layers)
1. **wall-clock cap** — `bootstrap.sh` wraps training in `timeout ${MAX_HOURS}h` (default 8h).
2. **watchdog** — destroys any pod past `max_hours + grace` or on terminal `_STATUS`.
   Panic button: `python scripts/pod/watchdog.py <run_id> --destroy-all`.
3. **key locality** — the vast key stays local; pods can't spawn or extend themselves.
4. **credit ceiling** — the account holds only its balance; vast stops pods at $0. The
   worst-case blast radius is the account credit, full stop.

---

## 6. Findings & bugs solved (the gold — read before touching launch.py)

- **`vastai search offers` numeric filters break for large thresholds.**
  `gpu_ram>=24000` returns **0 offers even for 24564 MB 4090s**. `gpu_ram>=24` returns
  everything (all cards clear 24 MB), which masks the bug. Fix: query only cheap reliable
  constraints server-side (`num_gpus=1`, `gpu_name=`, `dph_total<=`) and filter
  VRAM/disk/inet/compute-cap **client-side** in Python. See `search_offers()`.

- **`compute_cap` is a *capability* field, not a *speed* field.** It's compute
  capability ×100: V100=700, A100=800, RTX 3090=860, RTX 4090=890, Blackwell (5090 /
  RTX PRO 4000 Blackwell)=1200. Filtering by `compute_cap>=890` grabbed **RTX 4080
  Supers and workstation cards**, not 4090s; `>=1200` grabbed **RTX PRO 4000 Blackwells,
  not a single real 5090**. Fix: filter by **`gpu_name`** for a specific card; keep
  `compute_cap` only as a **bf16 sanity floor** (`>=800`).

- **V100 (sm_70) has no hardware bf16.** Our training uses bf16 autocast; a V100 errors
  or silently falls back to fp32. The `compute_cap>=800` floor excludes it (and anything
  pre-Ampere).

- **Blackwell (sm_120 / 5090) needs CUDA 12.8 + torch ≥2.7.** The default cu121/torch-2.4.1
  image fails on a 5090 with *"no kernel image is available for execution"*. Verified
  existing tags: `pytorch/pytorch:{2.7.0,2.7.1,2.8.0}-cuda12.8-cudnn9-runtime` exist;
  `2.6.0-cuda12.8` does **not** (torch 2.6 shipped no cu128). The `--gpu 5090` tier pins
  torch 2.8.0/cu128 automatically.

- **Two vast accounts, only one funded** (see §1a). Always verify with `show user`.

- **`vastai show instances` is deprecated** (→ `show instances-v1`, paginated). The old
  command still works and returns the fields the watchdog uses (`id`, `duration`,
  `actual_status`, `dph_total`); migrate if it ever stops.

- **`vastai ssh-url` returns empty until the instance is actually running** — pods take a
  minute+ to boot. `drive.sh` polls sshd with a timeout before giving up.

- **`create instance` returns the new id under `new_contract`** in the `--raw` JSON
  (not `id`). launch.py reads `new_contract` first.

- **`--env` / `--onstart-cmd` are stored in vast's instance metadata.** That's why
  `--mode ssh` (default) does **not** pass secrets there — it pre-clones the repo in
  onstart and lets `drive.sh` deliver R2 creds over SSH. Only `--mode auto` bakes R2
  creds into `--env` (documented exposure; skip `drive.sh`).

- **Launch precondition: push first.** Pods `git clone` the `--ref` (default `main`).
  If the training fixes or the pod scripts aren't pushed, pods train **stale code**.
  Commit + push before `--go`, and match `--ref`.

- **`dataclasses.asdict`/`replace` silently drop dynamically-attached cfg attrs** (a
  project-wide landmine that also bit checkpointing and the titans auto-disable). Not
  vast-specific, but relevant when the pod loads a checkpoint's `cfg_all`.

---

## 7. R2 result layout
```
neuromorphic/data.tar.gz                                # training data (820 MB)
neuromorphic/results/<run_id>/<arm>/run.jsonl           # per-step metrics
neuromorphic/results/<run_id>/<arm>/<arm>.last.pt       # final weights
neuromorphic/results/<run_id>/<arm>/<arm>.best.pt       # early-stop best
neuromorphic/results/<run_id>/<arm>/summary.json
neuromorphic/results/<run_id>/<arm>/bootstrap.log       # full pod stdout/stderr
neuromorphic/results/<run_id>/<arm>/_STATUS             # RUNNING|DONE|FAILED|TIMEOUT
```
`pull_results.sh` maps these back into `outputs/memory/<run_id>_<arm>/…` so the local
band/gate eval and diagnostics see them exactly as a local run would.

---

## 8. Security note
The R2 credentials and both vast keys were pasted into chat during setup → they are
**exposed** and should be rotated when convenient (the user overrode rotation at the time
with "go ahead and use it"). Design keeps the *billing-capable* vast key local-only; in
ssh mode R2 creds travel over the encrypted SSH channel, not vast metadata.
