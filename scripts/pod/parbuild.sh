#!/usr/bin/env bash
# Build all 4 Tier-2 method envs CONCURRENTLY (the bootstrap does them serially; they're independent).
# Idempotent: micromamba create skips an existing env; pip re-runs are ~no-ops. Each env logs separately.
export MAMBA_ROOT_PREFIX=/workspace/micromamba HF_HOME=/workspace/hf
export PIP_CACHE_DIR=/workspace/pipcache TMPDIR=/workspace/tmp
REPOS=/workspace/tier2_repos
H='numpy datasets huggingface_hub rank-bm25 sentence-transformers bert-score'
mm(){ micromamba "$@"; }

b_kvzip(){
  mm create -y -n kvzip python=3.10 pip
  mm install -y -n kvzip -c nvidia cuda-nvcc cuda-toolkit
  mm run -n kvzip pip install -q torch --index-url https://download.pytorch.org/whl/cu121
  mm run -n kvzip pip install -q -r "$REPOS/KVzip/requirements.txt"
  ( cd "$REPOS/KVzip" && mm run -n kvzip make i ) 2>&1 | tail -3
  mm run -n kvzip pip install -q $H
  echo KVZIP_DONE; }

b_kvcache(){
  mm create -y -n kvcache python=3.11 pip
  mm run -n kvcache pip install -q torch transformers accelerate
  mm run -n kvcache pip install -q -r "$REPOS/KVCache-Factory/requirements.txt"
  mm run -n kvcache pip install -q $H
  echo KVCACHE_DONE; }

b_memoryllm(){
  mm create -y -n memoryllm python=3.11 pip
  mm run -n memoryllm pip install -q -r "$REPOS/MemoryLLM/requirements_infer_only.txt"
  mm run -n memoryllm pip install -q $H
  echo MEMORYLLM_DONE; }

b_lclm(){
  mm create -y -n lclm python=3.11 pip
  mm run -n lclm pip install -q -r "$REPOS/LCLM/requirements.txt"
  mm run -n lclm pip install -q $H
  echo LCLM_DONE; }

b_kvzip     > /root/logs/env_kvzip.log 2>&1 &
b_kvcache   > /root/logs/env_kvcache.log 2>&1 &
b_memoryllm > /root/logs/env_memoryllm.log 2>&1 &
b_lclm      > /root/logs/env_lclm.log 2>&1 &
wait
echo ALL_ENVS_DONE
