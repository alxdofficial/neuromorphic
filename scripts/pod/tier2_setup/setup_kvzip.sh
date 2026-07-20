#!/usr/bin/env bash
# Tier-2 pod setup — KVzip (query-agnostic KV compression, Qwen2.5-7B). Idempotent; re-run safe.
#   WORKDIR=/workspace bash scripts/pod/tier2_setup/setup_kvzip.sh
# Pins CUDA 12.1 / py3.10 / flash-attn 2.7.4 + a custom AdaKV-derived eviction kernel (make + pip install -e).
# Qwen2.5-7B's 4-head KV makes ~100-111k-token LongMemEval-S borderline on a 24GB 4090;
# full MAB reaches ~746k tokens and needs an 80GB-class GPU. Query-agnostic → encode once/context.
# Runner: scripts/baselines/tier2/run_kvcompress.py --method kvzip.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/common.sh"
ensure_base
step "repo + env: kvzip"
clone_repo https://github.com/snu-mllab/KVzip.git KVzip
mkenv kvzip 3.10
# cuda-cccl MUST be pinned to 12.4 to match cuda-nvcc 12.4 — unpinned pulls cccl 13.x whose nv/target header
# is incompatible → kernel compile dies 'fatal error: nv/target: No such file or directory'.
micromamba install -y -n kvzip -c nvidia cuda-nvcc "cuda-cccl=12.4.*" cuda-cudart-dev >/dev/null 2>&1 && ok "cuda-nvcc + cccl 12.4"
pipin kvzip torch --index-url https://download.pytorch.org/whl/cu121 2>/dev/null
pipin kvzip -r "$REPOS/KVzip/requirements.txt" 2>/dev/null
pipin kvzip "$FA_KVZIP" 2>/dev/null && ok "flash-attn (prebuilt) in kvzip"
# Upstream's tiny extension only emits sm_80 and sm_90 cubins. Add the 4090's sm_89 target when absent.
# This changes build portability only; the CUDA source and KVzip algorithm remain exactly upstream.
if ! grep -q 'compute_89' "$REPOS/KVzip/csrc/build.py"; then
  sed -i '/# 2\. Target NVIDIA H100/i\# Target NVIDIA Ada GPUs (RTX 4090)\ncc_flag.append("-gencode")\ncc_flag.append("arch=compute_89,code=sm_89")\n' \
    "$REPOS/KVzip/csrc/build.py"
  ok "KVzip CUDA build target sm_89 added"
fi
# build the AdaKV kernel; CUDA_HOME must point at the env (nvcc + cccl 12.4 live there). Upstream `make i` =
# `cd csrc && make` (compiles+installs tiny_api_cuda) then `pip install -e .` — done explicitly:
( cd "$REPOS/KVzip/csrc" && CUDA_HOME="$MAMBA_ROOT_PREFIX/envs/kvzip" micromamba run -n kvzip make 2>&1 | tail -3 ) || warn "kvzip kernel compile"
( cd "$REPOS/KVzip" && micromamba run -n kvzip pip install -e . >/dev/null 2>&1 )
# import needs torch FIRST (import torch, tiny_api_cuda) else a spurious libc10.so error.
micromamba run -n kvzip python -c 'import torch, tiny_api_cuda' 2>/dev/null && ok "kvzip kernel imports (torch-first)" || warn "kvzip kernel import — inspect on pod"
pipin kvzip $HARNESS
prefetch kvzip "Qwen/Qwen2.5-7B-Instruct-1M"
step "kvzip ready — smoke: cd $REPOS/KVzip && micromamba run -n kvzip python $REPO/scripts/baselines/tier2/run_kvcompress.py --method kvzip --dataset longmemeval --max-examples 5"
