# Tier-2 mechanism baselines — integration map (build reference)

Research-verified 2026-07-18 (sub-agents, against live repos). Source of truth for `scripts/baselines/tier2/`
and the pod plan. Two kinds (mechanism cut, see `PHASE2_BASELINES.md` §2.5):
- **2a weight/KV-level** (§1–4: KVCache-Factory, KVzip, MemoryLLM/M+, LCLM) — manipulate KV cache / model
  internals or load their own weights → **cannot run via API**, need a GPU (RunPod).
- **2b agent-memory** (§6: A-MEM, MemoryOS) — orchestration layers over a **frozen** chat LLM → **run via
  our Tier-1 OpenRouter path**, GPU-optional (only a small local embedder).
- **Cartridges (§5): DROPPED as a runnable baseline** (per-corpus training doesn't fit private haystacks) →
  cite-only.

Deferred until we assemble the matched-decoder table (gated on our layer binding). **The newly-added
mechanisms report NO LongMemEval number → we generate them under our harness (never quote paper numbers).**

## 1. KVCache-Factory — SnapKV / H2O / PyramidKV / StreamingLLM
- Repo [Zefan-Cai/KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory) · **MIT** (note `csrc/` has its own license) · last commit 2026-07-10 (active).
- **Entry point = monkey-patch (not CLI-locked):** `from pyramidkv.monkeypatch import replace_llama; replace_llama("snapkv")` BEFORE model load, then set per-layer config on each `layer.self_attn.config` (`window_size`, `max_capacity_prompt`, `kernel_size`, `pooling`), then plain `model.generate(...)`.
- **7,500-tok cap** lives ONLY in `run_longbench.py`'s data path (`model2maxlen`) — bypass by tokenizing/generating yourself; the full 115k flows through.
- **Base models: Llama + Mistral ONLY** (no Qwen support). Use `attn_implementation="flash_attention_2"` or prefill VRAM balloons.
- **VRAM:** 8B bf16 (~16GB) + full 115k KV materialized (~15GB) before compression → **peak ~35–45GB** (needs >24GB).

## 2. KVzip — query-agnostic KV compression (NeurIPS'25)
- Repo [snu-mllab/KVzip](https://github.com/snu-mllab/KVzip) · **MIT** · last commit 2026-02-11.
- **Entry point (compress-then-query):**
  ```python
  from model import ModelKVzip
  m = ModelKVzip("Qwen/Qwen2.5-7B-Instruct-1M")
  kv = m.prefill(context, load_score=False, do_score=True)   # chunked internally (16k), handles 115k
  kv.prune(ratio=0.3)                                         # ratio = fraction RETAINED
  ans = m.generate(m.apply_template(question), kv=kv)
  ```
- **Base models:** Llama-3.1-8B, Qwen2.5-7B/14B-1M, Qwen3 0.6–32B, Gemma3.
- **VRAM:** ~16GB weights + ~15GB resident KV → **peak ~33–38GB**; ~20GB after prune(0.3).
- **Gotcha:** pinned CUDA 12.1 / py3.10 / `flash-attn==2.7.4.post1` (`--no-build-isolation`) + `make i` custom-kernel build. Import assumes CWD = repo root.

## 3. MemoryLLM / M+ — parametric memory
- Repo [wangyu-ustc/MemoryLLM](https://github.com/wangyu-ustc/MemoryLLM) · code **MIT**, weights `YuWangX/{memoryllm-8b,mplus-8b}` (HF cards say apache-2.0 but Llama-3-derived → check Meta license) · last commit 2025-07-28 (stale-ish).
- **Write:** per session, `model.inject_memory(ids, update_memory=True)` (each chunk must be >16 tokens). **Query:** plain `model.generate(...)` (memory already fused into hidden states).
- **Capacities:** MemoryLLM-8B = 12,800 mem-tok/layer, useful retention **~20k tokens** (random Ebbinghaus-style eviction). **M+** adds CPU-offloaded LTM (`put_ltm_to_numpy()`, 153,600 tok) + co-trained retriever → **~160k tokens** at similar VRAM.
- **VRAM:** ~16GB weights + ~3.3GB pool → plausibly fits 24GB (unverified; authors used H100-80GB). M+ CPU-offload → not materially more VRAM than 8B.
- **No LongMemEval runner exists** — hand-write the write-then-query loop (chunk history → inject per session → generate for the question). **MemoryLLM-8B (~20k) cannot hold 115k → use M+ only.**

## 4. LCLM — end-to-end soft-token compressor at scale (★ closest concurrent competitor)
- Repo [LeonLixyz/LCLM](https://github.com/LeonLixyz/LCLM) · HF org [latent-context](https://huggingface.co/latent-context) · ⚠ **LICENSE NOT STATED** (no LICENSE file, no model-card license) → **redistribution blocker; clarify before use/citation.**
- **Checkpoints:** `latent-context/0.6b-4b-LCLM-{4x,8x,16x}` (= 0.6B encoder + 4B decoder, compression 4×/8×/16×). Eval data `latent-context/lclm-eval` (configs ruler/gsm8k/longhealth5/longbench).
- **Base models:** encoder `Qwen/Qwen3-Embedding-0.6B`, decoder `Qwen/Qwen3-4B-Instruct-2507`.
- **Entry point (`inference/hf.py`) — repo MUST be on PYTHONPATH (not loadable via vanilla `transformers`/`vllm`):**
  ```python
  from inference.hf import load_model, generate_text
  model, dec_tok, processor = load_model("latent-context/0.6b-4b-LCLM-16x", device="cuda", dtype="bf16")
  prompt = f"<|memory_start|>{long_history}<|memory_end|> {question}"   # context MUST be wrapped
  ans = generate_text(model, dec_tok, processor, prompt, max_tokens=512, temperature=0.0)
  ```
  (CLI: `python -m inference.examples.example_hf --checkpoint ... --prompt "..."`; two-stage vLLM `encode`→`decode` runs encoder+decoder in SEPARATE processes.)
- **Decoder is TRAINED end-to-end** (frozen in stages 0–1, unfrozen with small LR in stage 2, SFT stage 3) — this is the axis that differentiates us (we keep the decoder FROZEN); engage in related work (`PHASE2_BASELINES.md` §2.5).
- **VRAM:** ~4.6B params ≈ 9–10GB bf16 → fits one 24GB GPU, inference-only (weights released; NO training needed). *[VRAM inferred.]*
- **Numbers:** RULER / LongBench / LongHealth / GSM8K; **no LongMemEval.** RULER 4×≈91.8 / 16×≈75.1 + "8.8× faster than KV baselines" are from SECONDARY coverage (VentureBeat) — verify against the paper's Tables 6/8/9 before citing.

## 5. Cartridges — CITE-ONLY (dropped as a runnable baseline, 2026-07-18b)
[HazyResearch/cartridges](https://github.com/HazyResearch/cartridges) · **Apache-2.0** · Qwen3-4b · trainable
soft-KV "cartridge" into a **FROZEN** LM ("self-study"): the closest analog to us on the frozen-decoder axis,
and worth citing as such. **Not run** because a cartridge is trained PER CORPUS — LongMemEval's 500 private
haystacks = 500 cartridges (infeasible & pointless: train, then ask one question). It fits an "inject-once,
query-many" shape (one big corpus / MAB), not our headline; the per-corpus training cost isn't worth it here.
(Entry points, if ever revisited: synth `cartridges/synthesize.py::SynthesizeConfig`+`SelfStudySynthesizer`
→ train `cartridges/train.py::TrainConfig` w/ `KVFromRandomText(max_tokens=p)` → query the Tokasaurus
`/v1/cartridge/chat/completions` endpoint; all need a live Tokasaurus/SGLang server.)

## 6. 2b agent-memory — A-MEM / MemoryOS (run via the Tier-1 OpenRouter path, GPU-optional)
Orchestration layers over a **frozen** chat LLM (retrieval + prompting) — NOT weight-level. Point them at our OpenRouter panel; ingest LongMemEval sessions then query; score with the SAME `score_longmemeval`. Only a small local embedder runs (CPU-fine). **Add exactly ONE** (`PHASE2_BASELINES.md` §2.5).
- **A-MEM** (default) — [WujiangXu/A-mem](https://github.com/WujiangXu/A-mem) (repro) / `A-mem-sys` (agent) · **MIT**. `from memory_layer import AgenticMemorySystem`; `add_note(content, ...)` to ingest, `find_related_memories(query, k=5)` to retrieve; backends `{openai, vllm, ollama, sglang}`; local embedder `all-MiniLM-L6-v2`. Eval driver `test_advanced_robust.py --backend openai --model ...`.
- **MemoryOS** (alt) — [BAI-LAB/MemoryOS](https://github.com/BAI-LAB/MemoryOS) · **Apache-2.0** · `pip install memoryos-pro`. `from memoryos import Memoryos`; `add_memory(user_input, agent_response)` to ingest, `get_response(query)` to answer; OpenAI-compatible backends; local embedder `BAAI/bge-m3`.
- Both report LoCoMo only (A-MEM: no absolute cells in README; MemoryOS: +49.11% F1 *relative*) → **no LongMemEval; we generate it.** Integration effort ≈ a thin adapter that maps LongMemEval sessions → `add_*` calls and the question → the query call, backend = our OpenRouter key.

## 7. Pod plan
| GPU | VRAM | RunPod $/hr (community) | KVCache-Factory | KVzip | MemoryLLM/M+ | our 135M |
|---|---|---|---|---|---|---|
| RTX 4090 | 24GB | $0.34 | ✗ (~35–45GB) | ✗ (~33–38GB) | ✓ (est.) | ✓ |
| **RTX A6000** | **48GB** | **$0.33** | ✓ | ✓ | ✓ | ✓ |
| L40S | 48GB | $0.79 | ✓ | ✓ | ✓ | ✓ |
| A100 80GB | 80GB | $1.19 | ✓ (headroom) | ✓ (headroom) | ✓ | ✓ |

**Recommendation: a single 48GB A6000 ($0.33/hr community — same as a 4090 but 2× VRAM) covers all four.** The KV methods (must materialize the full 115k KV before compressing → peak 33–45GB) are the >24GB forcing function; MemoryLLM never holds a full KV so it's fine on 24GB. Per-pass compute ≈ $1–2 each (~3–6 hrs on A6000); all Tier-2 + our model sequentially < $6, or use A100 80GB to remove OOM risk.

## Scaffolding plan (`scripts/baselines/tier2/`)
Each runner: load base model on GPU → for each LongMemEval item, ingest the 115k history via the method's
write/compress API → generate the answer → score with `src/memory/eval/score_longmemeval` (SAME scorer as
Tier-1, so numbers are directly comparable). `run_kvcompress.py` (SnapKV/KVzip), `run_memoryllm.py` (M+),
plus `README.md` with the pod bring-up + pinned-deps notes above. Build when we go to the matched-decoder table.
