#!/usr/bin/env bash
# Shared helpers for the per-baseline Tier-2 pod setup scripts (setup_<method>.sh).
# SOURCE this (`. common.sh`), don't execute it. Pod-side only — needs apt + micromamba + CUDA wheels; it
# does NOT run on the local dev box (no micromamba there). All the hard-won fixes (prebuilt flash-attn wheels,
# KVzip cccl-12.4 pin, M+ flash-attn strip) live here / in the per-method scripts so they never drift from the
# monolithic tier2_bootstrap.sh (which now just calls these). See scripts/pod/TIER2_RESUME.md for the why.
set -uo pipefail

# WORKDIR: envs + HF cache + repos go here so they survive across pods on a mounted network volume
# (set WORKDIR=/workspace). Defaults to /root (container disk; gone on terminate).
WORKDIR="${WORKDIR:-/root}"
REPO="${REPO:-/root/neuromorphic}"
export HF_HOME="${HF_HOME:-$WORKDIR/hf}"
export MAMBA_ROOT_PREFIX="$WORKDIR/micromamba"
REPOS="$WORKDIR/tier2_repos"
# Route ALL scratch onto WORKDIR — the container overlay `/` is often ~20GB and a flash-attn build alone blows it.
export PIP_CACHE_DIR="$WORKDIR/pipcache"
export TMPDIR="$WORKDIR/tmp"
mkdir -p "$HF_HOME" "$REPOS" "$PIP_CACHE_DIR" "$TMPDIR" /root/logs

step(){ echo -e "\n=== [tier2-setup] $* ==="; }
ok(){ echo "  ok: $*"; }
warn(){ echo "  WARN: $*"; }

# harness deps every env needs to import our eval core (src/memory/eval/tier2_common.py) + run the scorers
HARNESS="numpy datasets huggingface_hub rank-bm25 sentence-transformers bert-score"

# Prebuilt flash-attn wheels (NO from-source compile — none on PyPI, compiling is slow + fragile). Matched to
# each env's (torch, cuda, python, cxx11abi=FALSE); bump tags if an env's torch changes.
FA_MEMLLM="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
FA_KVZIP="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

ensure_base(){  # apt deps + micromamba (idempotent; safe to call from any per-method script)
  step "base: system deps + micromamba"
  apt-get update -y >/dev/null 2>&1
  apt-get install -y git rsync tmux build-essential ninja-build >/dev/null 2>&1 && ok "apt"
  if ! command -v micromamba >/dev/null 2>&1; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -C /usr/local/bin --strip-components=1 -xvj bin/micromamba >/dev/null 2>&1
  fi
  eval "$(micromamba shell hook -s bash)" 2>/dev/null || true
  command -v micromamba >/dev/null 2>&1 && ok "micromamba $(micromamba --version 2>/dev/null)" || warn "micromamba missing"
}

clone_repo(){ [ -d "$REPOS/$2/.git" ] || git clone -q "$1" "$REPOS/$2" && ok "repo $2 @ $(git -C "$REPOS/$2" rev-parse --short HEAD 2>/dev/null)"; }

mkenv(){  # <name> <python-version>
  local name="$1" py="$2"
  if [ ! -d "$MAMBA_ROOT_PREFIX/envs/$name" ]; then
    micromamba create -y -n "$name" "python=$py" pip >/dev/null 2>&1 && ok "env $name (py$py)"
  else ok "env $name exists"; fi
}
pipin(){ micromamba run -n "$1" pip install -q "${@:2}"; }

prefetch(){  # <env> <repo_id...>  — background-fetch checkpoints so eval doesn't stall on first token
  local env="$1"; shift
  local ids="$*"
  micromamba run -n "$env" python - "$@" <<'PY' >>/root/logs/prefetch.log 2>&1 &
import sys
from huggingface_hub import snapshot_download
for r in sys.argv[1:]:
    try: snapshot_download(r); print("fetched", r)
    except Exception as e: print("SKIP", r, str(e)[:160])
PY
  ok "prefetch [$ids] pid $! (tail /root/logs/prefetch.log)"
}
