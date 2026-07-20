# Tier-2 hosting best-practices — KV/memory reuse (the local analog of API prompt-caching)

**Question this answers:** do we host these local memory models "as normal," or is there special
inference-style hosting (caching) to exploit — the way we used OpenRouter prefix-caching to cut the MAB API
cost ~50×? **Answer: yes, and it's mandatory for MemoryAgentBench.**

## The principle

The API win came from MAB's structure: only **36 distinct contexts** underlie all 3,071 questions (~85 Q per
context), so re-sending each context per question was ~85× redundant — prefix-caching collapsed it. The
**local analog** is identical: **compute each context's memory representation ONCE, reuse it across every
question that shares it.** Hosting each method "as normal" (re-encode per question) re-does the expensive
prefill 85× — the same waste, now paid in GPU-hours instead of dollars.

This lever is **huge for MemoryAgentBench** (36 contexts × ~85 Q) and **negligible for LongMemEval** (each of
the 500 questions has its own unique history → nothing to reuse; the levers there are FlashAttention-2 prefill
+ batching, not caching).

## Per-method reuse mechanism

| method | reuse unit | recipe | note |
|---|---|---|---|
| **KV-eviction** | compressed KV cache | prefill context → compress → **cache the compressed KV** → every question decodes from it | **use KVzip, not SnapKV** (see below) |
| **MemoryLLM / M+** | memory-pool snapshot | write context session-by-session → **snapshot memory state** → answer all its questions → restore for next context | `_snapshot_memory_state` already scaffolded in `run_memoryllm.py` |
| **LCLM** | encoded latents | encoder compresses context → **cache the latent tokens** → decoder consumes cached latents + each question | LCLM compresses *before* decoder prefill → cheap per-question |

## Multi-query KV baselines: KVzip and infinite H2O

[KVzip (arXiv:2505.23416, ICML/NeurIPS'25)](https://arxiv.org/abs/2505.23416) is **query-agnostic**: it
compresses a context into a **single reusable KV cache** by scoring KV importance via context-reconstruction,
independent of any query. Its explicit selling point is the multi-query setting:

> "query-aware KV eviction methods (SnapKV, H2O) exhibit substantial performance degradation in multi-query
> settings, as they overfit to the initial queries and require **repetitive cache prefills**."

That is *exactly* MemoryAgentBench (one context, ~85 queries). The comparison now includes:
- **KVzip** (compress once per context, reuse across its ~85 questions). Tested on LLaMA-3.1-8B to 170k
  ctx; 3–4× cache reduction, ~2× decode speedup, negligible QA loss.
- **Infinite H2O** (stream once with position rolling, snapshot raw retained K/V and heavy-hitter scores, fork
  per question). Its context-only heavy-hitter decisions cannot use later questions to recover evicted tokens,
  but the cache is reusable and tests enforce equivalence to independent streaming replay.
- **SnapKV** still has no reusable path and remains LongMemEval-only.

## Standard tooling vs. our custom models

- **vLLM / SGLang automatic prefix caching** ([vLLM APC](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/),
  SGLang RadixAttention) is the off-the-shelf way to "process the long document once, reuse its KV for all
  later questions." SGLang's radix tree is best for shared-context multi-query.
- **But** MemoryLLM / LCLM are custom architectures (custom `modeling_*.py`) and KV-eviction patches attention
  → **none are drop-in vLLM/SGLang-servable.** We implement reuse manually via each method's own snapshot /
  latent-cache API + HF `past_key_values`. (The vanilla long-context ceiling that *would* benefit from SGLang
  we already have from the Tier-1 API, so no need to serve it locally.)

## Fidelity knobs (do NOT trade these for speed)

- **bf16, not 4-bit**, unless VRAM forces it — these are *baselines*; quantization would handicap them and
  invite "you under-ran the competitor." 48GB fits bf16 for all three, so no need.
- **FlashAttention-2 / SDPA** for the 100k+ prefills — required for both memory and speed.

## Scope + timing implication

With per-context reuse, **MAB local drops from ~30 GPU-hr/method to ~1.5–2 hr/method** (≈36 heavy context
encodes + 3,071 cheap question-decodes, vs. 3,071 heavy encodes). So MAB can ride along on the pod, not just
LongMemEval — *if* we wire the reuse. Caveat unchanged: 8B methods still **truncate MAB contexts > their
window** (AR up to 881k → 128k), so those competencies are truncation-limited exactly as the API llama was;
KVzip/MemoryLLM are meaningful on the ≤128k competencies, LCLM (compresses first) can reach further.

**Net:** wire the reuse before renting. It's the difference between a feasible ~2-hr pod and a multi-day one.
