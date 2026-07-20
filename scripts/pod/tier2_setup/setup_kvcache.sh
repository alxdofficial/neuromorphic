#!/usr/bin/env bash
# Tier-2 pod setup — KVCache-Factory (SnapKV; query-aware KV compression, Llama-3.1-8B). Idempotent.
#   WORKDIR=/workspace bash scripts/pod/tier2_setup/setup_kvcache.sh
# Runs fine under sdpa on an 80GB card → NO flash-attn build needed (the monkeypatch patches the sdpa attn
# class too, transformers 4.44.2). Peak ~35-45GB (full KV before eviction) → wants ≥48GB. LongMemEval ONLY
# (query-aware → can't reuse a compressed cache across MAB's ~85 Q/context). Base model is GATED (Llama-3.1-8B):
# the HF token must have accepted the Meta license or prefetch SKIPs. Runner: run_kvcompress.py --method snapkv.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/common.sh"
ensure_base
step "repo + env: kvcache"
clone_repo https://github.com/Zefan-Cai/KVCache-Factory.git KVCache-Factory
mkenv kvcache 3.11
pipin kvcache torch transformers accelerate 2>/dev/null
pipin kvcache -r "$REPOS/KVCache-Factory/requirements.txt" 2>/dev/null
pipin kvcache $HARNESS
[ -f /root/.cache/huggingface/token ] && export HF_TOKEN="$(cat /root/.cache/huggingface/token)"
prefetch kvcache "meta-llama/Llama-3.1-8B-Instruct"
step "kvcache ready — smoke: micromamba run -n kvcache python scripts/baselines/tier2/run_kvcompress.py --method snapkv --dataset longmemeval --max-examples 5"
