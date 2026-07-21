# Tier-2 pod — resume / don't-retread checklist

Everything below is already baked into the setup scripts. This file is the human checklist for the restart
scenarios + the exact fixes we hit, so we never re-debug them.

## Current shape (2026-07-21): a SHARDED FLEET, not one big pod
The single 3× H100 pod named in earlier revisions is **gone** — M+ cannot batch, so a big multi-GPU pod buys
nothing. The campaign runs **N single-GPU pods, one disjoint shard each**, driven by
**`scripts/pod/mpod.py`** (state: `scripts/pod/.mpod_state.json`).

- Last fleet: **14 pods**, mixed **RTX 4090 / A40 / RTX A6000** (Secure Cloud), `rate_per_pod` **$0.69/hr**.
- All storage on each pod's **local container disk** (`/root/...`), NOT a network volume — see
  [`docs/ops/RUNPOD_RUNBOOK.md`](../../docs/ops/RUNPOD_RUNBOOK.md) §1 for why (~3 MB/s on `/workspace`).
- Per-pod env at `/root/micromamba/envs/memoryllm`, repo at `/root/neuromorphic`, MemoryLLM clone at
  `/root/tier2_repos/MemoryLLM`; image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.

```bash
python scripts/pod/mpod.py create 14 --gpu 'NVIDIA A40' --secure   # rent
python scripts/pod/mpod.py endpoints && python scripts/pod/mpod.py setup
python scripts/pod/mpod.py setup-status
python scripts/pod/mpod.py launch longmemeval        # shard K -> pod K
python scripts/pod/mpod.py status --rate             # progress + alive + burn
python scripts/pod/mpod.py respawn 7                 # replace dead pod 7, resume its shard
python scripts/pod/mpod.py sync                      # pull outputs/baselines from every pod
python scripts/pod/mpod.py down                      # terminate ALL
python scripts/baselines/tier2/merge_shards.py ...   # merge shard artifacts -> one result JSON
```
Poll `status` **every 1–2 min** while a run is finishing — a fleet idling after completion bills at
14 × $0.69/hr.

`scripts/pod/tier2_pod.py attach <pod_id>` (single-pod `exec`/`run`/`logs`/`sync`) still works for one-off
debugging on an individual pod.

## Fleet-specific gotchas (measured 2026-07-21)
- **Runner ETAs read ~1.5× optimistic.** Measured on the fleet: **0.44–0.64 s/inject**, **4.1–4.5 s/answer**;
  `run_memoryllm.py` defaults are `--cost-inject-s 0.375` / `--cost-answer-s 2.3` (from a local 4090 n=1/n=2
  smoke). Retune per card or the ETA lies.
- **Ampere is not slower here.** A40 / A6000 measure **~1.24× an isolated 4090** on this workload — it is
  snapshot/PCIe-bandwidth-bound, not compute-bound. Buy the cheap 48GB Ampere cards.
- **M+ on MemoryAgentBench needs a ≥116GB-RAM container** (host RAM, not VRAM). MAB's post-injection snapshot
  carries `cached_dropped_*` buffers that grow with inject count; **47GB pods are OOM-killed (SIGKILL, rc=137,
  no traceback).** LongMemEval is unaffected — singleton contexts skip the snapshot entirely. Filter on RAM.
- **Merging shards:** `merge_shards.py` globs the cache **recursively** and resolves duplicate `question_id`s
  by newest mtime. Duplicates are expected across re-partitionings (context grouping changes M+'s injection
  schedule, so the same id can have different generations) — the M+ LongMemEval merge hit **72**. Use
  `--shard-glob` to merge exactly one partitioning when you want that instead.

## Legacy single-pod scenarios (`tier2_pod.py` + `/workspace` network volume)
Kept for the KVzip/SnapKV path, which is still a one-big-GPU job. **Not** how the M+ fleet runs.

### Restart scenario A — same volume reattached (common)
Envs (`/workspace/micromamba/envs/{kvzip,kvcache,memoryllm,lclm}` plus `/workspace/venvs/h2o`) + weights
(`/workspace/hf`) are already there. **No bootstrap needed.** Just:
1. `python scripts/pod/tier2_pod.py attach <new_pod_id>`
2. `python scripts/pod/tier2_pod.py sync --push`   # re-ship the repo to the fresh overlay `/root/neuromorphic`
3. launch the panel (below).

### Restart scenario B — fresh pod, no volume (must rebuild)
`python scripts/pod/tier2_pod.py bootstrap` — the hardened `tier2_bootstrap.sh` applies every fix below.
Run with `WORKDIR=/workspace` so nothing lands on the 20GB overlay.

## The fixes (all now in tier2_bootstrap.sh / the runners) — DO NOT re-discover
1. **Everything on `/workspace`** (WORKDIR), incl. `PIP_CACHE_DIR`/`TMPDIR` — the 20GB overlay overflows otherwise.
2. **KVzip kernel**: needs `cuda-nvcc` + **`cuda-cccl=12.4.*`** (unpinned pulls cccl 13.x → `fatal error: nv/target`).
   Build with `CUDA_HOME=<env prefix>`; `make` (csrc) then `pip install -e .`. Import needs **torch first**
   (`import torch, tiny_api_cuda`) or you get a spurious `libc10.so` error.
3. **KVzip also needs flash-attn** (`flash_attn_varlen_func`) → prebuilt wheel `cu12torch2.3…cp310`.
4. **M+ REQUIRES flash_attention_2** — its eager/sdpa attn classes return 4 values, decoder unpacks 5 →
   crash. Prebuilt wheel `cu12torch2.5…cp311`. `run_memoryllm.py --attn-impl` default = `flash_attention_2`.
5. **M+ requirements pin a from-source flash-attn** → strip it (`grep -v flash`) and use the prebuilt wheel.
6. **M+ is a BASE model** (rambles, never emits EOS) → `run_memoryllm.py` takes the **first line** as the
   answer + reports `stop`, else every item is a "length" cutoff and coverage = 0.
7. **kvcache + lclm run under sdpa** on 80GB. H2O uses its own bounded eager attention path. None needs
   flash-attn.
8. **Datasets HF-auto-download** (`xiaowu0162/longmemeval-cleaned`); LCLM checkpoint prefetched too. The
   `sync --push` exclude is **anchored** (`/data`, not `data`) so `src/memory/data/` (loaders) ships.
9. **Llama-3.1-8B is GATED** (SnapKV/H2O only) — the HF token must have accepted the Meta license, else it
   SKIPs; the 3 core methods (kvzip/memoryllm/lclm) are unaffected.

## Legacy launch — multiple methods on ONE multi-GPU pod (superseded)
The commands below launch several methods on one multi-GPU pod (GPU 0/1/2). That shared-pod shape is
**superseded**: the current campaign (`docs/baselines/PHASE2_HUB.md` §2) shards M+ across a fleet via
`mpod.py launch <dataset>`, and LCLM is dropped from the panel entirely. Keep these only for a one-off
KVzip/SnapKV pod.
```
C="MAMBA_ROOT_PREFIX=/workspace/micromamba HF_HOME=/workspace/hf"
tier2_pod.py run kvzip_lme   "$C CUDA_VISIBLE_DEVICES=0 micromamba run -n kvzip     python scripts/baselines/tier2/run_kvcompress.py --method kvzip --dataset longmemeval     --repo-dir /workspace/tier2_repos/KVzip"
tier2_pod.py run memllm_lme  "$C CUDA_VISIBLE_DEVICES=1 micromamba run -n memoryllm python scripts/baselines/tier2/run_memoryllm.py            --dataset longmemeval     --repo-dir /workspace/tier2_repos/MemoryLLM"
tier2_pod.py run lclm_lme    "$C CUDA_VISIBLE_DEVICES=2 micromamba run -n lclm      python scripts/baselines/tier2/run_lclm.py                  --dataset longmemeval     --repo-dir /workspace/tier2_repos/LCLM"
# then --dataset memoryagentbench for each. Pull results: tier2_pod.py sync
```
Runs are **resumable** (per-question JSONL cache under `outputs/baselines/cache/`), so a mid-run restart
continues where it left off after `sync --push`.
