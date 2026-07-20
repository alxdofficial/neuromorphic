#!/usr/bin/env bash
# Tier-2 pod bootstrap — idempotent provisioning for the baseline eval panel. Safe to re-run: each step
# is guarded (skips if already present) and fails LOUD but does NOT abort the others, so one method's build
# surprise (flash-attn, KVzip's custom kernel, gated Llama) doesn't block the rest. Refine live on the pod
# with `tier2_pod.py exec` and re-run.
#
#   env layout: micromamba per method (KVzip pins py3.10; the others py3.11) — they pin mutually
#   incompatible torch/transformers/flash-attn (see scripts/baselines/tier2/README.md "Environment
#   isolation"). Our eval core (src/memory/eval/tier2_common.py) is pure-python + lazy imports, so a small
#   shared "harness" dep set (numpy/datasets/sentence-transformers/bert-score/rank-bm25) goes into EACH env.
#
# WORKDIR: put envs+HF cache on a persistent path so they survive across pods when a network volume is
# mounted (set WORKDIR=/workspace). Defaults to /root (container disk; gone when the pod is terminated).
set -uo pipefail
WORKDIR="${WORKDIR:-/root}"
REPO="${REPO:-/root/neuromorphic}"
export HF_HOME="${HF_HOME:-$WORKDIR/hf}"
export MAMBA_ROOT_PREFIX="$WORKDIR/micromamba"
REPOS="$WORKDIR/tier2_repos"
# Route ALL scratch (pip cache, build tmp, flash-attn compile) onto WORKDIR — the container overlay `/` is
# often tiny (~20GB) and a flash-attn build alone can blow it. HF_HOME/envs/repos already live on WORKDIR.
export PIP_CACHE_DIR="$WORKDIR/pipcache"
export TMPDIR="$WORKDIR/tmp"
mkdir -p "$HF_HOME" "$REPOS" "$PIP_CACHE_DIR" "$TMPDIR" /root/logs
step(){ echo -e "\n=== [bootstrap] $* ==="; }
ok(){ echo "  ok: $*"; }

step "system deps"
apt-get update -y >/dev/null 2>&1
apt-get install -y git rsync tmux build-essential ninja-build >/dev/null 2>&1 && ok "apt"

step "micromamba"
if ! command -v micromamba >/dev/null 2>&1; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -C /usr/local/bin --strip-components=1 -xvj bin/micromamba >/dev/null 2>&1
fi
eval "$(micromamba shell hook -s bash)" 2>/dev/null || true
command -v micromamba >/dev/null 2>&1 && ok "micromamba $(micromamba --version 2>/dev/null)"

# harness deps every env needs to import our eval core + run the deterministic scorers
HARNESS="numpy datasets huggingface_hub rank-bm25 sentence-transformers bert-score"

mkenv(){  # <name> <python-version>
  local name="$1" py="$2"
  if [ ! -d "$MAMBA_ROOT_PREFIX/envs/$name" ]; then
    micromamba create -y -n "$name" "python=$py" pip >/dev/null 2>&1 && ok "env $name (py$py)"
  else ok "env $name exists"; fi
}
pipin(){ micromamba run -n "$1" pip install -q "${@:2}" ; }

step "clone method repos"
clone(){ [ -d "$REPOS/$2" ] || git clone -q "$1" "$REPOS/$2" && ok "$2"; }
clone https://github.com/Zefan-Cai/KVCache-Factory.git KVCache-Factory
clone https://github.com/snu-mllab/KVzip.git            KVzip
clone https://github.com/wangyu-ustc/MemoryLLM.git      MemoryLLM
clone https://github.com/LeonLixyz/LCLM.git             LCLM

# flash-attn: use PREBUILT wheels (NO from-source compile — no PyPI wheel exists and compiling is slow +
# fragile). Two methods NEED it: M+'s eager/sdpa attention classes return 4 values but its decoder unpacks 5
# (encoder_retriever_weights) → only flash_attention_2 works; KVzip imports flash_attn_varlen_func at load.
# KVCache-Factory + our own reader run fine under sdpa on 80GB, so no flash-attn there. Wheels are matched to
# each env's (torch, cuda, python, cxx11abi=FALSE) — bump the tags if an env's torch version changes.
FA_MEMLLM="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
FA_KVZIP="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# ---- KVzip (py3.10; custom eviction kernel needs nvcc + cccl 12.4 + flash_attn) ----
step "env: kvzip"
mkenv kvzip 3.10
# cuda-cccl MUST be pinned to 12.4 to match cuda-nvcc 12.4 — unpinned pulls cccl 13.x whose nv/target header
# is incompatible and the kernel compile dies with 'fatal error: nv/target: No such file or directory'.
micromamba install -y -n kvzip -c nvidia cuda-nvcc "cuda-cccl=12.4.*" cuda-cudart-dev >/dev/null 2>&1 && ok "cuda-nvcc + cccl 12.4 in kvzip"
pipin kvzip torch --index-url https://download.pytorch.org/whl/cu121 2>/dev/null
pipin kvzip -r "$REPOS/KVzip/requirements.txt" 2>/dev/null
pipin kvzip "$FA_KVZIP" 2>/dev/null && ok "flash-attn (prebuilt) in kvzip"
# build the AdaKV-derived kernel; CUDA_HOME must point at the env (nvcc + cccl 12.4 live there). Upstream
# `make i` = `cd csrc && make` (compiles+installs tiny_api_cuda) then `pip install -e .` — done explicitly:
( cd "$REPOS/KVzip/csrc" && CUDA_HOME="$MAMBA_ROOT_PREFIX/envs/kvzip" micromamba run -n kvzip make 2>&1 | tail -3 ) || echo "  WARN kvzip kernel compile"
( cd "$REPOS/KVzip" && micromamba run -n kvzip pip install -e . >/dev/null 2>&1 )
micromamba run -n kvzip python -c 'import torch, tiny_api_cuda' 2>/dev/null && ok "kvzip kernel imports (torch-first)" || echo "  WARN kvzip kernel import — inspect on pod"
pipin kvzip $HARNESS

# ---- KVCache-Factory (SnapKV; LongMemEval only; sdpa OK on 80GB) ----
step "env: kvcache"
mkenv kvcache 3.11
pipin kvcache torch transformers accelerate 2>/dev/null
pipin kvcache -r "$REPOS/KVCache-Factory/requirements.txt" 2>/dev/null
pipin kvcache $HARNESS

# ---- H2O (our online Llama-3.1 adapter; reuse image torch, no upstream clone/build) ----
step "env: h2o"
H2O_ENV="$WORKDIR/venvs/h2o"
[ -x "$H2O_ENV/bin/python" ] || python3 -m venv --system-site-packages "$H2O_ENV"
"$H2O_ENV/bin/pip" install -q -r "$REPO/requirements.txt" 2>/dev/null
"$H2O_ENV/bin/python" "$REPO/scripts/baselines/tier2/smoke_h2o.py" --device cuda >/root/logs/smoke_h2o.log 2>&1 \
  && ok "h2o CUDA smoke" || echo "  WARN h2o smoke — inspect /root/logs/smoke_h2o.log"

# ---- MemoryLLM / M+ (flash-attn REQUIRED; the repo pins a from-source flash-attn — strip it, use prebuilt) ----
step "env: memoryllm"
mkenv memoryllm 3.11
grep -viE 'flash[-_]attn' "$REPOS/MemoryLLM/requirements_infer_only.txt" > "$TMPDIR/memreq.txt"
pipin memoryllm torch transformers accelerate 2>/dev/null
pipin memoryllm -r "$TMPDIR/memreq.txt" 2>/dev/null
pipin memoryllm "$FA_MEMLLM" 2>/dev/null && ok "flash-attn (prebuilt) in memoryllm"
pipin memoryllm $HARNESS

# ---- LCLM (soft-token compressor) ----
step "env: lclm"
mkenv lclm 3.11
pipin lclm -r "$REPOS/LCLM/requirements.txt" 2>/dev/null
pipin lclm $HARNESS

step "HF auth + model prefetch (backgrounded; big)"
[ -f /root/.cache/huggingface/token ] && export HF_TOKEN="$(cat /root/.cache/huggingface/token)"
# Prefetch the default checkpoints so the eval doesn't stall on first-token download. Gated Llama needs the
# token above to have accepted the Meta license. Backgrounded — check with `du -sh $HF_HOME`.
micromamba run -n memoryllm python - <<'PY' >/root/logs/prefetch.log 2>&1 &
from huggingface_hub import snapshot_download
# kvzip=Qwen2.5-7B, memoryllm=mplus-8b, lclm=0.6b-4b-LCLM-16x, snapkv/h2o=Llama-3.1-8B (GATED: the token
# above must have accepted the Meta license, else it SKIPs and only SnapKV/H2O are affected — core 3 are fine).
for r in ["Qwen/Qwen2.5-7B-Instruct-1M", "YuWangX/mplus-8b", "latent-context/0.6b-4b-LCLM-16x",
          "meta-llama/Llama-3.1-8B-Instruct"]:
    revision = "0e9e39f249a16976918f6564b8830bc894c89659" if r.startswith("meta-llama/") else None
    ignore = ["original/*"] if r.startswith("meta-llama/") else None
    try: snapshot_download(r, revision=revision, ignore_patterns=ignore); print("fetched", r, revision or "main")
    except Exception as e: print("SKIP", r, str(e)[:120])
PY
ok "prefetch pid $!"

step "readiness"
echo "  repos:   $(ls "$REPOS" 2>/dev/null | tr '\n' ' ')"
echo "  envs:    $(micromamba env list 2>/dev/null | awk 'NR>2{print $1}' | tr '\n' ' ')"
echo "  GPUs:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ',')"
echo "  next:    smoke each method (--max-examples 5) then run_pod_panel.sh — see tier2/README.md"
echo "=== [bootstrap] done (re-run safe; refine failed steps via tier2_pod.py exec) ==="
