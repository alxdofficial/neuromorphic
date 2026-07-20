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
# strip flash-attn (no wheel → installed separately) AND torch/vision/audio (installed pinned from cu121 below;
# leaving them here would let `-r` re-resolve against PyPI). KEEP torchmetrics — it's a real repo dep.
grep -viE 'flash[-_]attn|^(torch|torchvision|torchaudio)==' "$REPOS/MemoryLLM/requirements_infer_only.txt" > "$TMPDIR/memreq.txt"
# Install the repo's PINNED torch from the cu121 index ONCE (unpinned `pip install torch` pulls the latest
# ~3GB wheel, then memreq.txt downgrades it to 2.5.1 = a wasted second ~2.5GB download). Matches the local venv.
pipin memoryllm torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 2>/dev/null
pipin memoryllm -r "$TMPDIR/memreq.txt" 2>/dev/null   # torch line already satisfied → not re-downloaded
pipin memoryllm "$FA_MEMLLM" 2>/dev/null && ok "flash-attn (prebuilt) in memoryllm"
pipin memoryllm $HARNESS
micromamba run -n memoryllm python -c 'import torch, flash_attn; print("flash_attn", flash_attn.__version__)' 2>/dev/null \
  && ok "memoryllm env imports flash_attn" || warn "flash_attn import failed — check wheel tags in common.sh"
prefetch memoryllm "YuWangX/mplus-8b"
# env-ready sentinel (cheap completion check for the driver; weights prefetch continues in the background —
# verify $HF_HOME/hub/models--YuWangX--mplus-8b size separately before smoke).
touch "$WORKDIR/.setup_memoryllm.env.done"
step "memoryllm ENV ready (weights still prefetching) — smoke: micromamba run -n memoryllm python scripts/baselines/tier2/run_memoryllm.py --dataset longmemeval --max-examples 5"
