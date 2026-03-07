#!/bin/bash
# Quick setup for RunPod (or any fresh Ubuntu + CUDA machine).
# Clones the repo, installs deps, and runs the throughput benchmark.
#
# Usage (on RunPod):
#   # After cloning:
#   cd /workspace/neuromorphic && bash scripts/setup_runpod.sh
#
#   # Or one-liner (clone + setup):
#   git clone https://github.com/alexredsmith/neuromorphic.git /workspace/neuromorphic \
#     && cd /workspace/neuromorphic && bash scripts/setup_runpod.sh

set -euo pipefail

PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "Neuromorphic LM — RunPod Setup"
echo "============================================"
echo "Repo root: $REPO_ROOT"
echo ""

# --- 1. System check ---
echo "[1/4] Checking system..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || {
    echo "ERROR: No GPU detected"; exit 1
}
$PYTHON --version
echo ""

# --- 2. Install Python deps ---
echo "[2/4] Installing dependencies..."
# PyTorch should already be installed on RunPod images.
pip install -q transformers">=4.35.0" datasets">=2.14.0" tokenizers">=0.15.0" \
    tqdm">=4.65.0" numpy">=1.24.0" pyarrow 2>&1 | tail -5

# FLA (flash-linear-attention) for RWKV-7 baseline
pip install -q fla 2>/dev/null || echo "  (FLA not available — RWKV-7 benchmarks will be skipped)"

# Triton kernel for our model's scan
pip install -q triton 2>/dev/null || echo "  (Triton not available — will use CPU scan fallback)"

echo "  torch=$($PYTHON -c 'import torch; print(torch.__version__)')"
echo "  CUDA=$($PYTHON -c 'import torch; print(torch.version.cuda)')"
echo ""

# --- 3. Quick smoke test ---
echo "[3/4] Smoke test (import check)..."
cd "$REPO_ROOT"
$PYTHON -c "
import sys; sys.path.insert(0, '.')
from src.model.config import ModelConfig
from src.model.model import NeuromorphicLM
for name, fn in [('A', ModelConfig.tier_a), ('B', ModelConfig.tier_b), ('C', ModelConfig.tier_c)]:
    cfg = fn()
    cfg.vocab_size = 32000
    m = NeuromorphicLM(cfg)
    n = sum(p.numel() for p in m.parameters())
    print(f'  Tier {name}: {n/1e6:.1f}M params (D={cfg.D}, D_embed={cfg.D_embed})')
    del m
print('  Import OK')
"
echo ""

# --- 4. Run throughput benchmark ---
echo "[4/4] Running throughput benchmark..."
echo "  This takes ~10-30 minutes depending on GPU and tiers."
echo "  Monitor with: tail -f results/throughput_*.json"
echo ""

TIERS="${TIERS:-A}"
$PYTHON scripts/benchmark_throughput.py \
    --tiers $TIERS \
    --json "results/throughput_$(hostname)_$(date +%Y%m%d_%H%M%S).json"

echo ""
echo "============================================"
echo "Setup complete! Results in results/"
echo "============================================"
echo ""
echo "To benchmark specific tiers:"
echo "  TIERS='A B C' bash scripts/setup_runpod.sh"
echo "  python scripts/benchmark_throughput.py --tiers A B C --json results/all.json"
