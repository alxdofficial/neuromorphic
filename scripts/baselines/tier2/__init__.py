"""Phase-2 Tier-2 GPU baselines (KV-cache compression + MemoryLLM/M+).

Pod-only runners — these manipulate KV cache / model internals with real weights on a GPU, so
(unlike Tier-1's `run_api_eval.py`) they cannot run via the OpenRouter API and cannot run on this
machine. See `docs/baselines/TIER2_GPU_INTEGRATION.md` (spec) and `scripts/baselines/tier2/README.md`
(pod bring-up).
"""
