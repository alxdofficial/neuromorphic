# Tier-2 GPU baselines — pod bring-up

Runners for the **2a architectural / weight-level** memory baselines the Phase-2 panel compares us against:
KV-compression (SnapKV/H2O via KVCache-Factory, KVzip), parametric memory (MemoryLLM/M+), and the soft-token
compressor **LCLM** (`run_lclm.py`, our closest concurrent competitor). (Cartridges was considered but
**dropped** — per-corpus training doesn't fit LongMemEval's private haystacks; cite-only.)
Unlike Tier-1 (`scripts/baselines/run_api_eval.py`, OpenRouter API), these manipulate KV cache / model
internals with real weights — they need a rented GPU and **cannot run on this machine or via any API**.

> The **2b agent-memory** baselines (A-MEM / MemoryOS) are NOT here — they run over a frozen LLM via the
> OpenRouter path with only a CPU embedder: `scripts/baselines/run_agentmem.py --method {a-mem,memoryos}`
> (`pip install` the framework + `OPENROUTER_API_KEY`; no GPU).

Spec + provenance + exact entry points: `docs/baselines/TIER2_GPU_INTEGRATION.md` (§4 LCLM, §5 Cartridges, §6 2b).

## 1. Pick the pod

| GPU | VRAM | RunPod $/hr (community) | KVCache-Factory (SnapKV/H2O) | KVzip | MemoryLLM/M+ |
|---|---|---|---|---|---|
| RTX 4090 | 24GB | $0.34 | too small (~35-45GB peak) | too small (~33-38GB peak) | fits (est.) |
| **RTX A6000** | **48GB** | **$0.33** | fits | fits | fits |
| L40S | 48GB | $0.79 | fits | fits | fits |
| A100 80GB | 80GB | $1.19 | fits, headroom | fits, headroom | fits |

**Use a single 48GB RTX A6000** (same community price as a 24GB 4090, 2x the VRAM) — it's the only card in
the cheap tier that covers all three methods. The forcing function is that the KV-compression methods
materialize the FULL ~115k-token KV cache before compressing it (peak 33-45GB); MemoryLLM/M+ never holds a
full KV so it would fit a 24GB card on its own, but sharing one A6000 for the whole panel is simpler and
still only ~$1-2/method (~3-6 hrs compute) — see the integration doc's pod-plan table for the arithmetic.

Follow `scripts/pod/README.md` / `docs/ops/runpod_workflow.md` for the actual `runpod.py create` /
`drive` / `reap` mechanics (SSH key, tmux, billing-safety wall-clock cap) — this doc only covers what's
Tier-2-specific: which repos to clone and how to launch each runner once you're on the pod.

## 2. Clone the method repos (on the pod)

Each runner does `sys.path.insert(0, <repo-dir>)` instead of `pip install`-ing these (none are on PyPI as
importable packages) — clone them under `~/tier2_repos/` (the runners' `--repo-dir` defaults) or pass
`--repo-dir` explicitly.

```bash
mkdir -p ~/tier2_repos && cd ~/tier2_repos

# --- KVCache-Factory (SnapKV / H2O) ---
git clone https://github.com/Zefan-Cai/KVCache-Factory.git
cd KVCache-Factory && pip install -r requirements.txt && cd ..

# --- KVzip ---
# Pinned: CUDA 12.1, Python 3.10, flash-attn==2.7.4.post1. Use a matching base image/conda env —
# this repo has its own custom CUDA kernel (`make i`) that will NOT build against arbitrary torch/CUDA.
git clone https://github.com/snu-mllab/KVzip.git
cd KVzip
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
make i                                    # builds the AdaKV-derived custom eviction kernel
cd ..

# --- MemoryLLM / M+ ---
git clone https://github.com/wangyu-ustc/MemoryLLM.git
cd MemoryLLM && pip install -r requirements_infer_only.txt && cd ..

# --- LCLM (soft-token compressor; ⚠ license not stated on repo/HF — clear before use) ---
# checkpoints are NOT vanilla-transformers loadable → the repo MUST be on PYTHONPATH (--repo-dir).
git clone https://github.com/LeonLixyz/LCLM.git
cd LCLM && pip install -r requirements.txt && cd ..
```

Common baseline deps (KVCache-Factory + MemoryLLM; KVzip pins its own versions above):
```bash
pip install torch transformers accelerate flash-attn --no-build-isolation
```
`flash_attention_2` is REQUIRED (not just faster) for both KVCache-Factory and M+ — eager/sdpa prefill on
a ~115k-token context multiplies the already-tight VRAM budget further.

## 3. HF auth + gated weights

`meta-llama/Llama-3.1-8B-Instruct` (KVCache-Factory default) is gated — `huggingface-cli login` with a
token that has accepted the Meta license before running. `Qwen/Qwen2.5-7B-Instruct-1M` (KVzip default) and
`YuWangX/mplus-8b` (MemoryLLM default) are ungated.

## 4. Launch

```bash
cd /path/to/neuromorphic   # repo root — the runners insert it onto sys.path themselves

# smoke test each method on a handful of items first (no BEM download needed for a quick sanity check)
python scripts/baselines/tier2/run_kvcompress.py --method snapkv --max-examples 5
python scripts/baselines/tier2/run_kvcompress.py --method h2o    --max-examples 5
python scripts/baselines/tier2/run_kvcompress.py --method kvzip  --max-examples 5
python scripts/baselines/tier2/run_memoryllm.py                  --max-examples 5
python scripts/baselines/tier2/run_lclm.py --checkpoint latent-context/0.6b-4b-LCLM-16x --max-examples 5

# full LongMemEval-S panel (500 questions)
python scripts/baselines/tier2/run_kvcompress.py --method snapkv --max-capacity-prompt 2048
python scripts/baselines/tier2/run_kvcompress.py --method h2o    --max-capacity-prompt 2048
python scripts/baselines/tier2/run_kvcompress.py --method kvzip  --ratio 0.3
python scripts/baselines/tier2/run_memoryllm.py
python scripts/baselines/tier2/run_lclm.py    --checkpoint latent-context/0.6b-4b-LCLM-{4x,8x,16x}

# 2b agent-memory (NO GPU — runs from anywhere with a key):
OPENROUTER_API_KEY=... python scripts/baselines/run_agentmem.py --method a-mem --max-examples 5
```

Each run prints `overall_acc` / `task_averaged_accuracy` / `abstention_accuracy` and writes one JSON to
`outputs/baselines/longmemeval__<method>__<model-slug>.json` (same scorer + shape as Tier-1's
`outputs/baselines/*.json`, so the two panels line up directly). Pull that directory back over R2 the same
way `scripts/pod/pull_results.sh` does for training runs.

## 5. Gotchas (see the integration doc for the full detail)

- **KVCache-Factory: Llama + Mistral only** (no Qwen). `--model` must be a Llama/Mistral checkpoint.
- **KVCache-Factory's 7,500-tok cap** lives only in ITS OWN `run_longbench.py` — `run_kvcompress.py` never
  imports that file, so the runner sees the full ~115k-token history.
- **KVzip CWD assumption** — `from model import ModelKVzip` resolves relative imports inside the KVzip repo;
  if you see import errors, `cd` into the cloned `KVzip/` dir before invoking the runner (in addition to
  `--repo-dir`).
- **MemoryLLM-8B (not M+) tops out ~20k tokens** — cannot hold LongMemEval-S's ~115k-token haystack.
  `run_memoryllm.py` defaults to M+ for this reason; only override `--model` to MemoryLLM-8B/-chat if you
  want a (roundly unfair, context-truncated) sanity comparison.
- **MemoryLLM's >16-token injection minimum** — `run_memoryllm.py` merges short sessions until each
  `inject_memory()` chunk clears `--min-inject-tokens` (default 17).
- **Per-item memory reset (MemoryLLM/M+)** — the repo exposes no public "reset memory" call; the runner
  snapshots `model.memory.data` once after load and restores it before every LongMemEval item (each item
  has its own private haystack). Flagged as a POD-ONLY TODO in `run_memoryllm.py` — verify on the pod that
  this is the complete state to reset (M+'s LTM store in particular).
- **VRAM headroom** — if a KV-compression run OOMs on the A6000, first try a smaller `--max-capacity-prompt`
  / larger `--ratio` (evict more), then fall back to the A100 80GB row in the table above.
- **LCLM** — the context MUST be wrapped in `<|memory_start|> … <|memory_end|>` (the runner does this); the
  checkpoints only load with the repo on PYTHONPATH (`--repo-dir`); ~9–10GB bf16 → fits a 24GB card,
  inference-only. ⚠ **license not stated** — clear redistribution before publishing numbers.
- **A-MEM / MemoryOS** (`run_agentmem.py`, top-level, no GPU) — point them at OpenRouter (`OPENROUTER_API_KEY`);
  a-mem retrieves and WE generate the answer via the same panel model; memoryos generates internally. Only the
  local sentence-embedder runs (CPU-fine). `pip install` the framework first (A-mem repo / `memoryos-pro`).
