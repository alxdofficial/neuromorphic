#!/usr/bin/env bash
# Tier-2 pod setup — M+ / MemoryLLM (mplus-8b). Idempotent; re-run safe.
#   WORKDIR=/workspace bash scripts/pod/tier2_setup/setup_memoryllm.sh
# Fits a 24GB card (~16GB weights + pool) but is OVERHEAD-bound → cheapest on an A40 (memory
# project_mplus_batching_verdict). REQUIRES flash_attention_2: M+'s eager/sdpa attn classes return 4 values
# but its decoder unpacks 5 → only flash_attention_2 works. The repo pins a from-source flash-attn → we STRIP
# it and install a matched prebuilt wheel. Runner: scripts/baselines/tier2/run_memoryllm.py.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/common.sh"
ensure_base
step "repo + env: memoryllm"
clone_repo https://github.com/wangyu-ustc/MemoryLLM.git MemoryLLM
mkenv memoryllm 3.11
grep -viE 'flash[-_]attn' "$REPOS/MemoryLLM/requirements_infer_only.txt" > "$TMPDIR/memreq.txt"
pipin memoryllm torch transformers accelerate 2>/dev/null
pipin memoryllm -r "$TMPDIR/memreq.txt" 2>/dev/null
pipin memoryllm "$FA_MEMLLM" 2>/dev/null && ok "flash-attn (prebuilt) in memoryllm"
pipin memoryllm $HARNESS
micromamba run -n memoryllm python -c 'import torch, flash_attn; print("flash_attn", flash_attn.__version__)' 2>/dev/null \
  && ok "memoryllm env imports flash_attn" || warn "flash_attn import failed — check wheel tags in common.sh"
prefetch memoryllm "YuWangX/mplus-8b"
step "memoryllm ready — smoke: micromamba run -n memoryllm python scripts/baselines/tier2/run_memoryllm.py --dataset longmemeval --max-examples 5"
