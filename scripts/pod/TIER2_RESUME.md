# Tier-2 pod — resume / don't-retread checklist

Everything below is already baked into `tier2_bootstrap.sh` (fresh pod) and persisted on the `/workspace`
network volume (envs + model weights survive a pod stop/restart). This file is the human checklist for the
two restart scenarios + the exact fixes we hit, so we never re-debug them.

## Current pod (2026-07-20)
- id `tvuw5lqjn2r6ov` · 3× H100 SXM 80GB Secure · volume mounted at `/workspace`
- container overlay `/` = 20GB (tiny — repo only); **all envs/models/caches live on `/workspace`**
- attach with: `python scripts/pod/tier2_pod.py attach <pod_id>` (then `exec`/`run`/`logs`/`sync`)

## Restart scenario A — same volume reattached (common)
Envs (`/workspace/micromamba/envs/{kvzip,kvcache,memoryllm,lclm}`) + weights (`/workspace/hf`) are already
there. **No bootstrap needed.** Just:
1. `python scripts/pod/tier2_pod.py attach <new_pod_id>`
2. `python scripts/pod/tier2_pod.py sync --push`   # re-ship the repo to the fresh overlay `/root/neuromorphic`
3. launch the panel (below).

## Restart scenario B — fresh pod, no volume (must rebuild)
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
7. **kvcache + lclm run under sdpa** on 80GB — no flash-attn needed there.
8. **Datasets HF-auto-download** (`xiaowu0162/longmemeval-cleaned`); LCLM checkpoint prefetched too. The
   `sync --push` exclude is **anchored** (`/data`, not `data`) so `src/memory/data/` (loaders) ships.
9. **Llama-3.1-8B is GATED** (SnapKV/H2O only) — the HF token must have accepted the Meta license, else it
   SKIPs; the 3 core methods (kvzip/memoryllm/lclm) are unaffected.

## Launch the panel (per method, one GPU each; env prefix matters)
```
C="MAMBA_ROOT_PREFIX=/workspace/micromamba HF_HOME=/workspace/hf"
tier2_pod.py run kvzip_lme   "$C CUDA_VISIBLE_DEVICES=0 micromamba run -n kvzip     python scripts/baselines/tier2/run_kvcompress.py --method kvzip --dataset longmemeval     --repo-dir /workspace/tier2_repos/KVzip"
tier2_pod.py run memllm_lme  "$C CUDA_VISIBLE_DEVICES=1 micromamba run -n memoryllm python scripts/baselines/tier2/run_memoryllm.py            --dataset longmemeval     --repo-dir /workspace/tier2_repos/MemoryLLM"
tier2_pod.py run lclm_lme    "$C CUDA_VISIBLE_DEVICES=2 micromamba run -n lclm      python scripts/baselines/tier2/run_lclm.py                  --dataset longmemeval     --repo-dir /workspace/tier2_repos/LCLM"
# then --dataset memoryagentbench for each. Pull results: tier2_pod.py sync
```
Runs are **resumable** (per-question JSONL cache under `outputs/baselines/cache/`), so a mid-run restart
continues where it left off after `sync --push`.
