#!/usr/bin/env bash
# Tier-2 pod setup — LCLM (end-to-end soft-token compressor; 0.6B enc + 4B dec). Idempotent; re-run safe.
#   WORKDIR=/workspace bash scripts/pod/tier2_setup/setup_lclm.sh
# ~9-10GB bf16 → fits a 24GB card; runs under sdpa (no flash-attn build). Our closest concurrent competitor
# (soft-token memory into a decoder — but LCLM TRAINS its decoder; we keep ours frozen). Checkpoints load ONLY
# with the repo on PYTHONPATH (run_lclm.py --repo-dir). Runner: scripts/baselines/tier2/run_lclm.py.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/common.sh"
ensure_base
step "repo + env: lclm"
clone_repo https://github.com/LeonLixyz/LCLM.git LCLM
mkenv lclm 3.11
pipin lclm torch transformers accelerate 2>/dev/null
pipin lclm -r "$REPOS/LCLM/requirements.txt" 2>/dev/null
pipin lclm $HARNESS
prefetch lclm "latent-context/0.6b-4b-LCLM-16x"
step "lclm ready — smoke: micromamba run -n lclm python scripts/baselines/tier2/run_lclm.py --dataset longmemeval --max-examples 5"
