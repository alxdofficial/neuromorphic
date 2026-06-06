# Stage-2+ Llama speed — research plan

**Why this matters.** Once the encoder is bf16+compiled (stage-1), stage-2 (and stage-3 RL) are **~90% frozen-Llama**. Every memory mechanism shares the *same* frozen Llama-3.2-1B, so optimizing it once speeds up training for **all 5 baselines**. This doc is the menu + plan.

## Measured baseline (graph_v6, BS=8, bf16, sdpa)

| read mode | fwd | bwd | total | note |
|---|---|---|---|---|
| **inject@13** (production) | 22ms | 20ms | **42ms** (~190 samp/s) | memory enters at layer 13; bwd flows only 13→16 |
| prepend (stopgap) | 36ms | 64ms | 100ms | memory at input → full-16 fwd+bwd |

The prepend→inject gap (2.4×) is the short backward (bwd reaches only the inject layer) + no memory tokens stuck on the sequence.

## The key structural fact

**Llama is frozen AND (in inject mode) memory enters high (layer 13).** So **layers 1–12 are a static, gradient-free feature extractor over a fixed context.** That's an unusual amount of exploitable structure. Two consequences:

1. **`no_grad` on the lower stack is already implicit for inject mode.** With frozen params + frozen (no-grad) input embeddings, autograd builds *no* graph below layer 13 — nothing there requires grad. So `no_grad` adds ~0 speed for inject@13 (it only helps memory/cleanliness). **Correction to an earlier claim: the win is NOT `no_grad`, it's caching the lower-stack *forward compute*.**
2. **Caching the lower stack is the real lever.** The context below layer 13 produces identical activations every epoch (deterministic frozen forward) → precompute the layer-13 input hiddens for the dataset **once**, then each step runs only **layers 13–16 + the memory**.

### Read-mode dependence (answers "is this for all baselines?")
- **inject@13 (graph_v6 today):** lower stack static → cacheable; short backward. The exploit applies.
- **prepend (vqvae/slot/mt/mamba today):** memory at the *input* mixes into every layer via attention → lower stack is **not** static (depends on per-step memory) and backward flows through all 16 layers (~100ms). The cache/short-backward **do not apply.**

➡️ **Enabler: unify all baselines onto the inject@13 cross-attention read.** This (a) gives every baseline the cacheable lower stack + short backward, (b) is *fairer* — a common read isolates differences to the *write* mechanism (as the MQAR gate already does), and (c) aligns stage-2 with the gate and the K/V-split (which needs cross-attention anyway). **This is the unlock that makes "optimize Llama once → all baselines benefit" actually true.**

## The menu (prioritized)

### Tier 1 — exploit frozen + high-inject (biggest, specific to us)
1. **Lower-stack activation cache.** Precompute the layers-below-13 hiddens over the fixed context once; train on only 13–15 + memory. **MEASURED (graph_v6 BS8, inject@13): caching layers 0–12 cuts the forward 22.7→12.0ms = 47% of the forward (1.9×).** More modest than first estimated: the residual 12ms is dominated by the **lm_head (128K-vocab matmul)** + the 3 upper layers, and the backward is already short. Net step est. ~42→~32ms (~1.3×) for inject mode. **The bigger payoff is for the PREPEND baselines** — moving them to inject (the enabler) is 2.4× *before* the cache. Cost: storage for cached hiddens (CPU/disk, fixed dataset). **Requires inject mode for all.**
2. ~~**Restrict the lm_head + CE to supervised (answer) positions.**~~ **ALREADY DONE** — `model.py:869–894` ("Selective lm_head") gathers prediction positions before the 128K-vocab projection, and `:1084–1109` skips HF's unconditional lm_head. So the measured 42ms already includes this win. (Kept here for the record.)
3. **Weight-quantize the frozen Llama (int8 / fp8 / int4).** ⚠️ **DEFERRED — stay bf16 for now (decision 2026-06-06).** Not a *training-stability* risk (frozen weights never update), BUT the frozen Llama is in the **gradient path to the trainable memory**, so quant error injects bias/noise into the memory's gradients → an accuracy/learnability question, not free. Especially wrong to add while the design's learnability is still unproven. Keep as strictly *test-before-trust*, revisit only if speed becomes a hard blocker. Every other lever here is bf16-preserving. (torchao not installed.)

### Tier 2 — compilation (infra already in repo: #334 regional-compile, #278/#371 cudagraphs)
3. **Regional `torch.compile` + CUDA graphs.** Whole-module compile breaks on the layer-13 forward-hook (measured 1.01×); compile decoder layers individually + cudagraph the static forward → ~1.3–2×. (Or fold the inject into the forward so whole-module compile works — also the read-path work the split needs.)

### Tier 3 — attention
4. **FlashAttention-2/3** (CUDA build) — modest at short context, grows with Stage-A context.
5. **Verify SDPA uses the flash backend** — a custom attention mask from the inject path can silently downgrade to the math kernel.

### Stage-3 (RL) — different bottleneck = generation
6. **KV-cache + cudagraph the per-token decode** (#371 exists), and if rollout-heavy, a **vLLM/paged-attention** generation path while training on logged trajectories.

## Recommended prototype order (revised after measuring)
1. **Unify-on-inject** for the baselines (the enabler) — modest plumbing; biggest single win *for them* (prepend ~100ms → inject ~42ms = 2.4×) + unlocks Tier-1 for all + fairness.
2. **lm_head/CE answer-position restriction** — cheap (check-then-fix); the 128K-vocab projection is a big slice of both the residual forward and the backward. Verify current behavior first.
3. **int8 weight-quant** of frozen Llama — high ROI at small BS (bandwidth-bound), all baselines; test LM-head accuracy.
4. **Lower-stack cache** — measured ~1.3× for inject (47% of forward); bigger for the freshly-injected baselines.
5. Regional compile + cudagraphs (proven infra, lower novelty).

**Realistic stack:** for inject mode, lm_head-restrict + int8 + cache ≈ ~42 → ~20–25ms. The headline is **unify-on-inject brings the 4 prepend baselines from ~100ms to that same range** — so the real win is *uniformity*: optimize once, all 5 land together (and fairer, since they'd share the read).
