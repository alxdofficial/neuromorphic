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

**Campaign runs ONE tailored pod PER MODEL (not a shared pod), on Secure Cloud** — current per-model GPU/cost
choices live in [`docs/baselines/PHASE2_HUB.md`](../../../docs/baselines/PHASE2_HUB.md) §2. The VRAM table above
is reference only. Key points: the KV-compression methods materialize the FULL ~115k-token KV before compressing
(peak 33-45GB → want ≥48GB); M+ never holds a full KV (fits 24GB but is **overhead-bound → cheapest on an A40**,
not a big GPU); **LCLM runs FULL locally on the 4090** (no rental).

Follow `scripts/pod/README.md` / `docs/ops/runpod_workflow.md` for the actual `runpod.py create` /
`drive` / `reap` mechanics (SSH key, tmux, billing-safety wall-clock cap) — this doc only covers what's
Tier-2-specific: which repos to clone and how to launch each runner once you're on the pod.

> ⚠ **Environment isolation.** KVzip (CUDA 12.1 / py3.10 / flash-attn 2.7.4.post1 + custom kernel),
> KVCache-Factory, MemoryLLM/M+, and LCLM pin **mutually incompatible** torch / transformers / numpy /
> python versions. Do **not** install them into one shared env — give each its own venv/conda env (or run
> them on separate pod sessions). Our harness code is pure-python + lazy-imported, so it rides along in any
> of them.

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
`flash_attention_2` is REQUIRED for **M+** only (its eager/sdpa attention classes return 4 values but the
decoder unpacks 5 → crash), installed via a prebuilt wheel. **KVCache-Factory (SnapKV/H2O) and KVzip run fine
under `sdpa` on an 80GB card** (verified) — no flash-attn build needed there. See `scripts/pod/TIER2_RESUME.md`.

## 3. HF auth + gated weights

`meta-llama/Llama-3.1-8B-Instruct` (KVCache-Factory default) is gated — `huggingface-cli login` with a
token that has accepted the Meta license before running. `Qwen/Qwen2.5-7B-Instruct-1M` (KVzip default) and
`YuWangX/mplus-8b` (MemoryLLM default) are ungated.

## 4. Launch

**All three runners take `--dataset {longmemeval,memoryagentbench}`** and share one eval core
(`src/memory/eval/tier2_common.py`). That core does the **per-context reuse** that makes MAB affordable: it
groups questions by distinct context and encodes each context ONCE (MAB = 36 encodes for 3,071 Q; LongMemEval
= one per question, unique histories). This is the local analog of the Tier-1 prompt-cache win — see
`docs/baselines/TIER2_HOSTING.md`. Same deterministic scorers + JSON shape as Tier-1, so panels line up in
`report.py`.

```bash
cd /path/to/neuromorphic   # repo root — the runners insert it onto sys.path themselves

# --- smoke each method (5-20 items) BEFORE the full run ---
python scripts/baselines/tier2/run_kvcompress.py --method kvzip --dataset longmemeval --max-examples 5
python scripts/baselines/tier2/run_memoryllm.py               --dataset memoryagentbench --max-examples 20
python scripts/baselines/tier2/run_lclm.py                    --dataset longmemeval --max-examples 5

# --- OPTIONAL: multi-GPU shared pod (run_pod_panel.sh, one method per GPU). The CURRENT campaign runs ONE
#     MODEL PER TAILORED POD instead (docs/baselines/PHASE2_HUB.md §2); LCLM runs LOCALLY, not on a pod.
#     Kept for the case where a multi-GPU pod is rented to overlap methods. ---
DATASET=longmemeval scripts/baselines/tier2/run_pod_panel.sh          # override KVZIP_ENV/KVZIP_GPU/... to taste
DATASET=memoryagentbench scripts/baselines/tier2/run_pod_panel.sh

# --- or run methods individually ---
python scripts/baselines/tier2/run_kvcompress.py --method kvzip  --dataset memoryagentbench --ratio 0.3
python scripts/baselines/tier2/run_kvcompress.py --method snapkv --dataset longmemeval --max-capacity-prompt 2048  # LongMemEval only
python scripts/baselines/tier2/run_memoryllm.py  --dataset longmemeval
python scripts/baselines/tier2/run_lclm.py       --dataset longmemeval --checkpoint latent-context/0.6b-4b-LCLM-16x

# 2b agent-memory (NO GPU — runs from anywhere with a key; NOT part of the pod panel):
OPENROUTER_API_KEY=... python scripts/baselines/run_agentmem.py --method a-mem --max-examples 5
```

Each run writes `outputs/baselines/<dataset>__<method>__…​.json` (+ a resumable per-question JSONL under
`cache/`). Pull that directory back over R2 the same way `scripts/pod/pull_results.sh` does for training runs.

## 5. Gotchas (see the integration doc for the full detail)

- **KV method for MAB = kvzip, NOT snapkv/h2o.** KVzip is query-AGNOSTIC → one reusable compressed cache per
  context (correct for MAB's ~85 Q/context). SnapKV/H2O are query-AWARE → they'd re-prefill per question
  (slow AND degraded); the runner **refuses `snapkv|h2o` + `--dataset memoryagentbench`**. Use them on
  LongMemEval only. (docs/baselines/TIER2_HOSTING.md.)
- **LCLM on MAB — latent-reuse is a POD-VERIFY.** `run_lclm.py` currently encodes+decodes per question
  (correct, but re-runs the 0.6B encoder ~85×/context). To hit the MAB time budget, wire `_encode_memory`
  (encode once) + the cached-latent decode against the repo's `inference/hf.py` on the pod. LongMemEval is
  unaffected (unique histories → nothing to reuse).
- **Per-context reuse verify (KVzip / MemoryLLM).** KVzip: confirm `generate(kv=...)` treats the pruned cache
  read-only (query-agnostic by design; if answers degrade after the 1st question in a group, clone the cache
  per question). MemoryLLM: `_snapshot_memory_state` must find the real M+ LTM buffers (it HARD-WARNS if not)
  so a context's memory doesn't leak into the next.
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
