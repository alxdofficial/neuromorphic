#!/usr/bin/env bash
# KVzip setup for RTX PRO 6000 / GeForce Blackwell (sm_120) pods.
# Preserves the pod image's CUDA 12.8+ PyTorch because upstream's torch 2.3/cu121 pin predates Blackwell.
set -euo pipefail

REPO="${REPO:-/root/neuromorphic}"
WORKDIR="${WORKDIR:-/root}"
KVZIP_REPO="${KVZIP_REPO:-$WORKDIR/tier2_repos/KVzip}"
VENV="${KVZIP_VENV:-$WORKDIR/venvs/kvzip-blackwell}"
HF_HOME="${HF_HOME:-$WORKDIR/hf}"
FLASH_ATTN_COMMIT="${FLASH_ATTN_COMMIT:-2409214a03797b168f648ea30df1adbc09ce658a}"
export HF_HOME MAX_JOBS="${MAX_JOBS:-12}" TORCH_CUDA_ARCH_LIST="12.0"
# FlashAttention otherwise builds Ampere, Hopper, and both Blackwell families as well.
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-120}"

mkdir -p "$(dirname "$KVZIP_REPO")" "$(dirname "$VENV")" "$HF_HOME" /root/logs
if [ ! -d "$KVZIP_REPO/.git" ]; then
  git clone -q https://github.com/snu-mllab/KVzip.git "$KVZIP_REPO"
fi

# Reuse the image's Blackwell-enabled torch rather than downloading an incompatible upstream pin.
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip setuptools wheel packaging ninja
"$VENV/bin/pip" install -q \
  'transformers==4.51.3' 'datasets==4.5.0' 'accelerate==1.7.0' \
  'huggingface-hub>=0.30,<1.0' 'numpy==1.26.4' 'pandas>=2.2,<3' \
  'scipy==1.14.1' 'scikit-learn==1.5.2' future decorator matplotlib \
  rouge-score fuzzywuzzy rich rouge 'tree-sitter==0.21.3' \
  'tree-sitter-languages==1.10.2' rank-bm25 sentence-transformers bert-score

# Current FlashAttention main contains the sm_120 forward and varlen kernels needed by KVzip.
# Pin the source commit so a repeated pod setup cannot silently drift.
FLASH_SRC="$WORKDIR/src/flash-attention"
if [ ! -d "$FLASH_SRC/.git" ]; then
  git clone -q --recursive https://github.com/Dao-AILab/flash-attention.git "$FLASH_SRC"
fi
git -C "$FLASH_SRC" fetch -q origin "$FLASH_ATTN_COMMIT"
git -C "$FLASH_SRC" checkout -q "$FLASH_ATTN_COMMIT"
git -C "$FLASH_SRC" submodule update -q --init --recursive
FLASH_ATTENTION_FORCE_BUILD=TRUE "$VENV/bin/pip" install -q --no-build-isolation "$FLASH_SRC"

# Upstream emits Ampere and Hopper cubins only. Add a native sm_120 cubin without changing kernel logic.
if ! grep -q 'compute_120' "$KVZIP_REPO/csrc/build.py"; then
  sed -i '/# 2\. Target NVIDIA H100/i\# Target NVIDIA client Blackwell\ncc_flag.append("-gencode")\ncc_flag.append("arch=compute_120,code=sm_120")\n' \
    "$KVZIP_REPO/csrc/build.py"
fi
(cd "$KVZIP_REPO/csrc" && "$VENV/bin/python" build.py install)
(cd "$KVZIP_REPO" && "$VENV/bin/pip" install -q -e . --no-deps)

"$VENV/bin/python" - <<'PY'
import flash_attn
import torch
import tiny_api_cuda
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(), "capability", torch.cuda.get_device_capability())
print("flash_attn", flash_attn.__version__)
print("tiny_api_cuda", tiny_api_cuda.__file__)
PY

"$VENV/bin/pip" freeze > /root/logs/kvzip-blackwell-freeze.txt
git -C "$KVZIP_REPO" rev-parse HEAD > /root/logs/kvzip-upstream-commit.txt
echo "KVzip Blackwell environment ready: $VENV"
