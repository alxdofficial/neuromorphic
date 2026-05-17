# Wave 1 v3 — Memory-anchored training protocol

**Status:** design doc (2026-05-13), not yet implemented.

**Problem with Wave 1 v2 (current):** plain TF NTP gives weak gradient
signal for memory. Most tokens are predictable from immediate Llama
context alone; memory has no marginal contribution to optimize against.
Established by Daniluk et al. 2017 — memory-augmented LMs trained on
TF NTP empirically attend to "the 5 most recent representations." Our
own Wave 1 v2 showed transient memory use at step 1500 (needle
answer-only 3.76 < vanilla-2K 4.08), then drift away by step 8000.
Memory has no incentive to *stay* useful under TF NTP.

## Goal

Replace TF NTP loss with a training objective that **forces memory to
be load-bearing**, while remaining as cheap as Wave 1.

## Design: Far-token MLM auxiliary

Inspired by REALM (Guu et al. 2020), which pretrained their retriever
via masked-LM rather than NTP. Same pattern, adapted to our trajectory
memory.

### Core protocol

Per training step, on a 1024-token chunk (D=4 windows × 256 tokens):

1. **Standard TF NTP forward** (as Wave 1 v2). Memory writes happen
   during each window's read+predict+write cycle. Loss = NTP CE on the
   chunk.

2. **Mask a fraction of tokens in the OLDEST window** (window 0) before
   the chunk forward begins. Mask rate: 15-20%. Replace masked tokens
   with `[MASK]` (or pad-token; whatever doesn't collide with
   Llama's vocab).

3. **At the LAST window's exit (window D-1)**, run a small "memory
   recall head" that attempts to predict the masked tokens using ONLY
   memory readout:

   ```
   for each masked position p in window 0:
       q_p = position_encoding(p)              # query for this masked slot
       attn_out = cross_attn(q_p, memory_readout_at_window_D-1)
       logits_p = vocab_head(attn_out)         # vocab-sized output
       mlm_loss += CE(logits_p, target_p)
   ```

   The `vocab_head` is a small new trainable component: ~256 × 128K =
   33M params. Could alternatively reuse Llama's frozen lm_head if we
   project memory readout into d_lm first.

4. **Total loss:**
   ```
   loss = NTP_CE + α * MLM_CE
   ```
   Start α = 0.5; tune from there.

### Why this works

- **Masked tokens cannot be predicted from local context.** They're
  beyond Llama's effective_lm_context window at prediction time.
- **Memory is the only information path.** Memory was written during
  window 0 (when those tokens were originally seen). Now at window
  D-1, memory must retain enough info to recall the masked positions.
- **Direct training signal.** Each masked token's loss is directly
  attributable to whether memory encoded enough info about that
  token's content. Gradient flows back through:
  `MLM_CE → vocab_head → cross_attn → memory_readout → manifold states →
  write_module → ... → backprop chain`.
- **Doesn't break NTP path.** Standard NTP loss still trains the
  bridge + read modules to "use memory at inference." The MLM aux just
  ADDS pressure for memory to retain content.

### What this fixes vs Wave 1 v2

| failure mode | Wave 1 v2 (TF NTP) | Wave 1 v3 (TF NTP + far-MLM) |
|---|---|---|
| Memory has nothing to do on bulk tokens | yes (loss dominated by local-context predictions) | partially fixed (MLM aux concentrates on memory-required tokens) |
| Memory drifts away from useful regime | yes (no pressure to stay) | fixed (MLM keeps demanding memory retain content) |
| Routing collapse risk | mitigated by aux losses | same |
| Architecture changes | none | none (only aux loss + new vocab head) |

### Implementation cost

- **Code changes:** ~200 lines
  - `Phase1Trainer.step_wave1`: add mask sampling logic
  - `IntegratedLM.forward_window`: optionally return memory readout
    at last window
  - New `RecallHead` module: cross_attn + linear projection to vocab
  - Loss accumulation in trainer
- **Tests:** ~150 lines (3 new test cases)
- **Smoke test:** 10 steps on toy data, verify MLM loss is finite + drops
- **Training cost:** maybe +10-20% step time (one extra cross-attn +
  vocab projection on ~50 masked positions per step)

Roughly 1-2 days end to end.

### Variants worth considering

- **a) Variable mask distance.** Instead of always masking window 0,
  sometimes mask window 1 or 2. Mix of "very-far" and "moderately-far"
  recall pressure. Closer to curriculum learning.
- **b) Reuse Llama's lm_head.** Instead of a new vocab_head, project
  memory readout up to d_lm and feed through Llama's frozen lm_head.
  Saves 33M params; loses some specialization room for the recall head.
- **c) Mask only "content tokens."** Skip stopwords / common tokens
  (which are predictable from grammar regardless of memory). Focus
  masking on rare tokens (names, numbers, entities) where memory
  retention actually matters.

### Eval criteria

This protocol succeeds if:
1. **MLM CE drops** during training (memory is encoding old content).
2. **Memory-off ablation hurts MLM CE** (forces memory_readout = 0,
   should make MLM accuracy collapse).
3. **Needle EM improves** at >2K distances (the actual downstream
   task — verbatim recall benefits from this aux pressure).
4. **Bulk-token NTP CE doesn't regress** (MLM aux shouldn't break
   normal training).

### Fallbacks if this doesn't work

If far-MLM doesn't give the desired specialization:
- **Reconstruction loss** (AE-style on entire prior window) — stronger
  pressure but more memory bandwidth required.
- **Hindsight-augmented data** — synthetic queries about prior content
  embedded in training data.
- **Direct skip to GRPO** — Wave 3 reward signal may be sufficient on
  its own; we may not need Wave 1 v3 at all if Phase 2 GRPO works.

## Open questions

- Should the MLM auxiliary apply during Wave 2 (chat SFT) too? The
  long-prior chat data has the same "predict-from-far" structure.
- Does mask rate matter? At low rates (5%) we get clean signal but
  noisy; at high rates (50%) we starve the NTP path. 15-20% is a
  guess; should sweep.
- How does the "vocab head" interact with manifold-state semantics?
  If the head learns to read manifold states as token predictions,
  this might inadvertently encourage the manifold to encode token-
  level info instead of higher-level concepts. Worth probing post-hoc.

## References

- REALM (Guu et al. 2020): retriever pretrained via MLM, the original
  precedent.
- Daniluk et al. 2017: memory-augmented LMs trained on NTP attend to
  only the last ~5 tokens.
- LongMem (Wang et al. 2023): "memory-augmented adaptation training"
  with joint objectives.
- Structured Memory Mechanisms (Xing 2025): explicit joint training
  objective for memory writing + forgetting.
