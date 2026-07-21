# Tier-2 GPU baselines — pod bring-up

Runners for the **2a architectural / weight-level** memory baselines the Phase-2 panel compares us against:
KV-compression (SnapKV via KVCache-Factory, infinite-streaming H2O via our Llama-3.1 adapter, KVzip), parametric memory
(MemoryLLM/M+), and the soft-token
compressor **LCLM** (`run_lclm.py`, our closest concurrent competitor). (Cartridges was considered but
**dropped** — per-corpus training doesn't fit LongMemEval's private haystacks; cite-only.)
Unlike Tier-1 (`scripts/baselines/run_api_eval.py`, OpenRouter API), these manipulate KV cache / model
internals with real weights — they need a GPU and cannot run via an API. H2O-2048 and KVzip on LME-S can run
on the local 4090; SnapKV and full-MAB KVzip need larger rented GPUs.

> The **2b agent-memory** baselines (A-MEM / MemoryOS) are NOT here — they run over a frozen LLM via the
> OpenRouter path with only a CPU embedder: `scripts/baselines/run_agentmem.py --method {a-mem,memoryos}`
> (`pip install` the framework + `OPENROUTER_API_KEY`; no GPU).

Spec + provenance + exact entry points: `docs/baselines/TIER2_GPU_INTEGRATION.md` (§4 LCLM, §5 Cartridges, §6 2b).

## 1. Pick the pod

| GPU | VRAM | RunPod $/hr (community) | H2O cap=2048 | KVCache-Factory (SnapKV) | KVzip | MemoryLLM/M+ |
|---|---|---|---|---|---|---|
| RTX 4090 | 24GB | $0.34 | fits (18.40GB max measured) | too small (~35-45GB peak) | LME-S fits at score-chunk=1k; MAB too large | fits (est.) |
| **RTX A6000** | **48GB** | **$0.33** | fits | fits | LME-S fits; full MAB too large | fits |
| L40S | 48GB | $0.79 | fits | fits | LME-S fits; full MAB too large | fits |
| A100 80GB | 80GB | $1.19 | fits, headroom | fits, headroom | fits, headroom | fits |

**Campaign runs a SHARDED FLEET of single-GPU Secure-Cloud pods** (`scripts/pod/mpod.py`; 14 pods, mixed
4090/A40/A6000 at $0.69/pod-hr for the M+ LongMemEval run) — status and cost live in
[`docs/baselines/PHASE2_HUB.md`](../../../docs/baselines/PHASE2_HUB.md) §2, pod ops in
[`docs/ops/RUNPOD_RUNBOOK.md`](../../../docs/ops/RUNPOD_RUNBOOK.md). The VRAM table above
is reference only. Key points: SnapKV/KVzip materialize the full KV before compressing. Qwen2.5-7B KVzip
uses fewer KV heads than Llama, making LME-S borderline on 24GB, but full MAB still needs 80GB. Online H2O
stays bounded; M+ never holds a full KV (fits 24GB but is **overhead-bound → shard across many cheap Ampere
pods**, not one big GPU; A40/A6000 measure ~1.24× a 4090). **M+ on MemoryAgentBench additionally needs a
≥116GB-RAM container** — the post-injection snapshot's `cached_dropped_*` buffers grow with inject count and
47GB pods get SIGKILLed (rc=137). **LCLM is dropped from the panel.**

Follow `scripts/pod/README.md` / `docs/ops/runpod_workflow.md` for the actual `runpod.py create` /
`drive` / `reap` mechanics (SSH key, tmux, billing-safety wall-clock cap) — this doc only covers what's
Tier-2-specific: which repos to clone and how to launch each runner once you're on the pod.

> ⚠ **Environment isolation.** H2O uses our Transformers 5.1 adapter; KVzip (CUDA 12.1 / py3.10 /
> flash-attn 2.7.4.post1 + custom kernel), KVCache-Factory, MemoryLLM/M+, and LCLM pin **mutually incompatible** torch / transformers / numpy /
> python versions. Do **not** install them into one shared env — give each its own venv/conda env (or run
> them on separate pod sessions). Our harness code is pure-python + lazy-imported, so it rides along in any
> of them. KVzip's environment deliberately overrides its upstream `datasets==3.6.0` pin with
> `datasets==4.5.0` after
> installation because current MemoryAgentBench metadata uses the 4.x `List` schema feature; this affects
> benchmark loading only, not model or KV-cache behavior.

## 2. Clone the method repos (on the pod)

**H2O needs no third-party clone or CUDA build.** Its setup reuses the pod image's CUDA PyTorch, installs the
validated Transformers 5.1 harness, runs a no-download CUDA smoke, and prefetches Llama in the background:

```bash
WORKDIR=/workspace bash scripts/pod/tier2_setup/setup_h2o.sh
source /workspace/venvs/h2o/bin/activate
```

Each runner does `sys.path.insert(0, <repo-dir>)` instead of `pip install`-ing these (none are on PyPI as
importable packages) — clone them under `~/tier2_repos/` (the runners' `--repo-dir` defaults) or pass
`--repo-dir` explicitly.

```bash
mkdir -p ~/tier2_repos && cd ~/tier2_repos

# --- KVCache-Factory (SnapKV only) ---
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

Common upstream baseline deps (KVCache-Factory + MemoryLLM; KVzip pins its own versions above):
```bash
pip install torch transformers accelerate flash-attn --no-build-isolation
```
`flash_attention_2` is REQUIRED for **M+** only (its eager/sdpa attention classes return 4 values but the
decoder unpacks 5 → crash), installed via a prebuilt wheel. **KVCache-Factory (SnapKV) and KVzip run fine
under `sdpa` on an 80GB card** (verified) — no flash-attn build needed there. See `scripts/pod/TIER2_RESUME.md`.

## 3. HF auth + gated weights

`meta-llama/Llama-3.1-8B-Instruct` (H2O/SnapKV default) is gated — `huggingface-cli login` with a
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
# 4090 KVzip: validated low-memory setting; score-1k is a documented paper-ablation setting.
python scripts/baselines/tier2/run_kvcompress.py --method kvzip --dataset longmemeval --max-examples 5 \
  --kvzip-prefill-chunk-size 2048 --kvzip-scoring-chunk-size 1000
python scripts/baselines/tier2/run_kvcompress.py --method h2o  --dataset longmemeval --max-examples 1 \
  --max-capacity-prompt 2048 --prefill-chunk-size 512 --h2o-head-mode query_head
python scripts/baselines/tier2/run_kvcompress.py --method h2o  --dataset memoryagentbench --max-examples 20 \
  --max-capacity-prompt 2048 --prefill-chunk-size 512 --h2o-head-mode query_head
python scripts/baselines/tier2/run_memoryllm.py               --dataset memoryagentbench --max-examples 20
python scripts/baselines/tier2/run_lclm.py                    --dataset longmemeval --max-examples 5

# --- OPTIONAL: multi-GPU shared pod (run_pod_panel.sh, one method per GPU). The CURRENT campaign runs ONE
#     MODEL PER TAILORED POD instead (docs/baselines/PHASE2_HUB.md §2); LCLM runs LOCALLY, not on a pod.
#     Kept for the case where a multi-GPU pod is rented to overlap methods. ---
DATASET=longmemeval scripts/baselines/tier2/run_pod_panel.sh          # override KVZIP_ENV/KVZIP_GPU/... to taste
DATASET=memoryagentbench scripts/baselines/tier2/run_pod_panel.sh

# --- or run methods individually (full MAB KVzip requires 80GB) ---
python scripts/baselines/tier2/run_kvcompress.py --method kvzip  --dataset memoryagentbench --ratio 0.3
python scripts/baselines/tier2/run_kvcompress.py --method h2o   --dataset longmemeval --max-capacity-prompt 2048
python scripts/baselines/tier2/run_kvcompress.py --method h2o   --dataset memoryagentbench --max-capacity-prompt 2048
python scripts/baselines/tier2/run_kvcompress.py --method snapkv --dataset longmemeval --max-capacity-prompt 2048  # LongMemEval only
python scripts/baselines/tier2/run_memoryllm.py  --dataset longmemeval
python scripts/baselines/tier2/run_lclm.py       --dataset longmemeval --checkpoint latent-context/0.6b-4b-LCLM-16x

# 2b agent-memory (NO GPU — runs from anywhere with a key; NOT part of the pod panel):
OPENROUTER_API_KEY=... python scripts/baselines/run_agentmem.py --method a-mem --max-examples 5
```

Each run writes `outputs/baselines/<dataset>__<method>__…​.json` (+ a resumable per-question JSONL under
`cache/`). Pull that directory back over R2 the same way `scripts/pod/pull_results.sh` does for training runs.

## 5. Gotchas (see the integration doc for the full detail)

- **H2O and KVzip both support MAB context reuse; SnapKV does not.** Infinite H2O streams each context once,
  snapshots raw retained K/V plus heavy-hitter scores, and forks that state per question. The runner derives
  and verifies an exact token-level common prefix aligned to the chunk boundary. SnapKV remains query-aware
  and is refused on MAB because it has no equivalent reusable state path.
- **H2O's primary setting is fixed 2,048 KV entries/layer:** 1,024 cumulative-attention heavy hitters +
  1,024 recent tokens, selected independently per query head. The official paper commonly used a 20% budget,
  so 2,048 is a more aggressive fixed-footprint comparison on ~100k-token LongMemEval prompts. The artifact
  records this distinction. `--h2o-head-mode kv_head` is a 4x-smaller GQA approximation, not the primary row.
- **H2O prefill is infinite and chunked:** retained keys are stored before RoPE and re-rotated at compact cache
  positions, matching the official repository's `H2OLlamaAttention_streaming` design. A 512-token chunk raises
  attention-time KV from 2,048 to 2,560, then post-attention pruning returns it to 2,048. The eager attention
  matrix is bounded to `chunk × (2048 + chunk)`. Position rolling discards original absolute token distances;
  retained order, K/V content, and accumulated importance survive. A tail question can reweight survivors but
  cannot recover context evicted during streaming.
- **Measured locally on RTX 4090:** a 105,146-token LME-S prompt prefills in 23.29s (4,514 tok/s), 17.62GB peak.
  The longest MAB prompt (744,639 tokens, versus Llama's 131,072 native window) prefills once in 164.38s;
  snapshot forks cost about 2.5ms and peak at 18.40GB. Every layer retained exactly 2,048 entries in both checks.
- **H2O does not use KVCache-Factory.** The official FMInference/H2O cache is the semantic oracle; our modern
  adapter is `src/memory/eval/h2o_llama.py`. Run `smoke_h2o.py --device cuda` before loading 8B weights.
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
- **KVzip peak is before pruning.** Increasing compression (`--ratio`) does not make an oversized context fit.
  Qwen2.5-7B uses about 56 KiB of BF16 KV per token: LME-S is 99,837-111,192 tokens (~5.7-6.4GB raw KV),
  while MAB reaches 745,586 tokens (~42.8GB raw KV). `--kvzip-prefill-chunk-size` can reduce temporary
  prefill memory. The paper uses a 2,000-token reconstruction chunk; its ablation reports under 2% average
  relative performance difference at 1,000, which is the principled fallback (`--kvzip-scoring-chunk-size
  1000`) when the longest LME-S contexts do not fit on 24GB.
- **Measured KVzip on the local RTX 4090:** the longest 111,192-token LME-S context succeeds with prefill
  chunk 2,048 and scoring chunk 1,000 at 22.03GB allocated / 24.73GB reserved, taking 81.5s to compress.
  The paper-default 2,000 scoring chunk OOMs on that item; a 4,096 prefill chunk also leaves too little
  scoring headroom. Use `--kvzip-prefill-chunk-size 2048 --kvzip-scoring-chunk-size 1000` locally.
- **MemoryLLM-8B (not M+) tops out ~20k tokens** — cannot hold LongMemEval-S's ~115k-token haystack.
  `run_memoryllm.py` defaults to M+ for this reason; only override `--model` to MemoryLLM-8B/-chat if you
  want a (roundly unfair, context-truncated) sanity comparison.
- **MemoryLLM's >16-token injection minimum** — `run_memoryllm.py` merges short sessions until each
  `inject_memory()` chunk clears `--min-inject-tokens` (default 17).
- **Per-item memory reset (MemoryLLM/M+)** — the repo exposes no public "reset memory" call; the runner
  snapshots `model.memory.data` once after load and restores it before every LongMemEval item (each item
  has its own private haystack). Flagged as a POD-ONLY TODO in `run_memoryllm.py` — verify on the pod that
  this is the complete state to reset (M+'s LTM store in particular).
- **VRAM headroom** — KVzip's peak precedes pruning, so a smaller retained ratio cannot rescue an oversized
  context. Full MAB needs an A100/H100 80GB; smoke its 745,586-token maximum before launching all 3,071 Q.
- **LCLM** — the context MUST be wrapped in `<|memory_start|> … <|memory_end|>` (the runner does this); the
  checkpoints only load with the repo on PYTHONPATH (`--repo-dir`); ~9–10GB bf16 → fits a 24GB card,
  inference-only. ⚠ **license not stated** — clear redistribution before publishing numbers.
- **A-MEM / MemoryOS** (`run_agentmem.py`, top-level, no GPU) — point them at OpenRouter (`OPENROUTER_API_KEY`);
  a-mem retrieves and WE generate the answer via the same panel model; memoryos generates internally. Only the
  local sentence-embedder runs (CPU-fine). `pip install` the framework first (A-mem repo / `memoryos-pro`).
