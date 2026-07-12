# RunPod + Cloudflare R2 remote-training runbook

Runbook for offloading the arch-sweep training to rented RunPod GPUs. Tooling lives in `scripts/pod/`
(`runpod.py` + the provider-agnostic `bootstrap.sh`); this doc is the *why* + the gotchas.

Host quality matters: pick Secure/Community hosts with recent drivers, and match the CUDA image to the
GPU generation (see §2 — the cu124 image is Ampere/Ada only, not Blackwell).

TL;DR of the model: **one pod per training arm.** Code comes from public GitHub, data + models from R2,
results go back to R2. The billing-capable RunPod key stays on the local machine; `runpod.py reap`
terminates pods on completion so a stuck job can't bleed credit.

```
 local ──(1 create)──▶ RunPod pod ──(2 drive: ssh+creds+tmux)──▶ bootstrap.sh
   ▲                        │                                        │
   │                        │ train one arm (behavioral_kl)          │
   └──(4 pull results)── R2 ◀──(3 push results + _STATUS)────────────┘
                          ▲
        runpod.py reap polls _STATUS, terminates on done/fail/cap
```

## 0. One-time setup
- **RunPod auth:** `~/.runpod/config.toml` with `apikey = "..."`. The account's SSH key must match
  `~/.ssh/id_ed25519` (RunPod injects it into every pod). `pip install runpod`.
- **R2 creds:** `~/.config/r2/credentials` (`R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` / `R2_ENDPOINT` /
  `R2_BUCKET` / `R2_PREFIX`). **Never leaves the local machine except shipped to a pod over SSH.**
- **HF token:** `~/.cache/huggingface/token` (gated `meta-llama/Llama-3.2-1B` FineWeb src-tokenizer).
- **Data + models on R2:** `bash scripts/pod/pack_data.sh` (uploads `data.tar.gz`) and the
  `models.tar.gz` pre-seed (SmolLM2 + Llama tokenizer). Re-pack after any data change.

## 1. The flow
```bash
# create one pod per arm (GPU fallback chain; see §2)
python scripts/pod/runpod.py create icae_baseline
python scripts/pod/runpod.py list                       # confirm RUNNING + ssh endpoint

# drive: ship creds + code, launch bootstrap in a detached tmux
python scripts/pod/runpod.py drive <pod_id> icae_baseline podrun-0014

# monitor (poll FREQUENTLY — every 1–2 min; RunPod bills while idle)
ssh -i ~/.ssh/id_ed25519 -p <port> root@<host> 'tail -20 /workspace/bootstrap.log'

# reap finished pods (billing safety) + pull results
python scripts/pod/runpod.py reap podrun-0014
bash scripts/pod/pull_results.sh podrun-0014
```

## 2. GPU selection (validated 2026-07-12)
The image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1` is **Ampere/Ada only** — do NOT pick Blackwell
(5090 / RTX PRO 45xx); they need a cu128 image. `runpod.py`'s fallback chain (all ≥24 GB, all cu124-safe):
**4090 → A40 → A6000 → A5000 → 3090.**
- **RTX 4090** (24 GB, ~$0.34 community / $0.69 secure) — first choice when in stock, but **stock is
  frequently "Low"/out on both clouds**; `create` falls through automatically.
- **A40 / A6000** (48 GB Ampere, ~$0.44 / $0.49 secure) — the reliable fallback.
- **`BATCH=6`** on any 24 GB card (B=8 OOMs — a 24 GB 4090 has ~23.5 GB usable, and B=8 peaks ~22 GB +
  fragmentation). bootstrap sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid the frag OOM.

## 3. Gotchas already solved
- **tmux is mandatory for the detach.** `setsid` and `nohup … & disown` DIE when the one-shot ssh session
  closes (RunPod's sshd tears down the session's processes). `runpod.py drive` launches bootstrap inside a
  detached tmux (`bash -lc '… exec bash bootstrap.sh' > <arm>.log`) — the tmux server survives disconnect.
  Needs `apt-get update && apt-get install -y tmux` first (RunPod images ship no tmux; apt lists are stale).
- **FD limit.** Multi-worker DataLoaders + pin_memory exhaust the default 1024 open-file soft limit →
  "Too many open files" / "received 0 items of ancdata". Fixed in-code (`base.py` raises soft→hard) + in
  `bootstrap.sh` (`ulimit -n`).
- **HF offline.** bootstrap pre-seeds the HF cache from `models.tar.gz` so pods don't depend on a gated-HF
  fetch. The container CPU is cgroup-capped (e.g. 12 vcpu even on a 96-core host).
- **Poll frequently.** Background watchers often don't wake you; check running pods every 1–2 min or idle
  billing accrues. `runpod.py reap` is the backstop, not a substitute for watching.

## 4. Performance note — it is CPU-launch-bound, not GPU-bound
Pods run **~0.8 step/s vs ~1.4 local on the SAME 4090** (GPU ~36% util at near-max clock / low power).
Profiled to root: the step is **CPU-launch-bound** — SmolLM2-135M @ B=6 makes GPU kernels too small to hide
kernel-launch latency, and the rented container's 2.6 GHz cores dispatch ~2× slower than a local 5.7 GHz
desktop core. It is NOT data (data_wait = 0 ms), NOT throttling, NOT tokenization — so pre-tokenization and
`torch.compile --compile-decoder` do **not** help. Accepted for the one-time campaign (~2.8 h/arm). Levers
if it ever matters: a faster-CPU host, a bigger batch on a 48 GB card (amortizes launches), or CUDA graphs
(big refactor). See the session notes / `project_pod_pipeline` memory.

## 5. Files (`scripts/pod/`)
| file | runs where | does |
|------|-----------|------|
| `runpod.py` | local | create (GPU fallback) / drive (ssh+creds+tmux) / list / reap / terminate via the RunPod SDK |
| `bootstrap.sh` | pod | **provider-agnostic** — pull code+data+models, train one arm, push results + `_STATUS` to R2 |
| `pack_data.sh` | local (once) | tar the training data subset → upload `data.tar.gz` to R2 |
| `pull_results.sh` | local | download a run's results from R2 into `outputs/memory/` |
| `r2.sh` | local+pod | thin `aws s3` wrapper for the R2 bucket |
