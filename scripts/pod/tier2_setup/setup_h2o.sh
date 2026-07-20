#!/usr/bin/env bash
# Tier-2 pod setup for our modern Llama-3.1 H2O adapter. No upstream clone,
# CUDA extension, or flash-attn build is required. The venv reuses the pod
# image's CUDA-enabled PyTorch instead of downloading another torch wheel.
#
#   WORKDIR=/workspace bash scripts/pod/tier2_setup/setup_h2o.sh
set -euo pipefail
WORKDIR="${WORKDIR:-/root}"
REPO="${REPO:-/root/neuromorphic}"
export HF_HOME="${HF_HOME:-$WORKDIR/hf}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKDIR/pipcache}"
export TMPDIR="${TMPDIR:-$WORKDIR/tmp}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$WORKDIR/venvs" /root/logs
step(){ echo -e "\n=== [h2o-setup] $* ==="; }
ok(){ echo "  ok: $*"; }

step "env: h2o (reuse image torch)"
H2O_ENV="$WORKDIR/venvs/h2o"
if [ ! -x "$H2O_ENV/bin/python" ]; then
  python3 -m venv --system-site-packages "$H2O_ENV"
fi
"$H2O_ENV/bin/pip" install -q --upgrade pip
"$H2O_ENV/bin/pip" install -q -r "$REPO/requirements.txt"
"$H2O_ENV/bin/python" - <<'PY'
import torch, transformers
assert torch.cuda.is_available(), "pod PyTorch cannot see CUDA"
assert transformers.__version__ == "5.1.0", transformers.__version__
print(f"  ok: torch={torch.__version__} transformers={transformers.__version__} gpu={torch.cuda.get_device_name(0)}")
PY

step "offline adapter smoke"
"$H2O_ENV/bin/python" "$REPO/scripts/baselines/tier2/smoke_h2o.py" --device cuda

step "prefetch Llama-3.1-8B-Instruct"
[ -f /root/.cache/huggingface/token ] && export HF_TOKEN="$(cat /root/.cache/huggingface/token)"
"$H2O_ENV/bin/python" - <<'PY' >>/root/logs/prefetch_h2o.log 2>&1 &
from huggingface_hub import snapshot_download
snapshot_download(
    "meta-llama/Llama-3.1-8B-Instruct",
    revision="0e9e39f249a16976918f6564b8830bc894c89659",
    ignore_patterns=["original/*"],
)
print("fetched meta-llama/Llama-3.1-8B-Instruct")
PY
ok "prefetch pid $! (tail /root/logs/prefetch_h2o.log)"
step "h2o ready — activate: source $H2O_ENV/bin/activate"
