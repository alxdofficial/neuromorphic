# Meeting Prep — 2026-05-12

Status update for the trajectory-memory project.

---

## This week in my own words

- Almost all the time this week went to **training-stability bug fixes**, not new features.
- **Architecture-level redesign at the start** of the week. The old graph_walker did a memory read/write at every token. Two problems with that:
  - At every-token frequency, the walker picks up many small unrelated bits — e.g. how should it route on the word "the"?
  - Read and write start positions were forced to be the same node (wherever the walker happened to be). We wanted the model to *choose* where to start reading/writing trajectories.
  - **Fix:** walker now has separate read/write start positions, chosen periodically (per window) instead of every token.
- **Bug 1 — representation collapse.** After the redesign, gradient from token-prediction loss → "memory writes" only exists if a later read touches the *same* intersection of nodes the write used. Weak gradient signal in early training → only a few nodes ever activated → those few became the only active nodes.
  - **Fix attempt:** added an **entropy aux loss** (load-balance + z-loss) to encourage spreading logits across nodes.
- **Bug 2 — opposite problem from the fix.** The aux loss was too strong → routing became uniformly random; the model couldn't *select* any concept based on input.
- **Bug 3 — KV cache didn't respect `effective_lm_context=2048`.** Cap was supposed to force memory to carry long-range info; in KV-cache mode `past_key_values` grew unbounded so Llama saw the whole doc and memory had no functional role.
  - **Fix:** added `_crop_kv_cache` helper, wired into `forward_window`.
- **Diagnostic benches:** vanilla Llama vs our model on needle-in-haystack (EM, CE, per-distance stratification). Confirmed the training failure was structural, not just hyperparameter tuning.
- **Bug 4 (the deepest, found today)** — the **Gumbel noise** in the routing's straight-through estimator had std ≈ 1.28 while the cosine routing signal was bounded in [-1, 1] with typical magnitudes ~0.06. Noise was ~20× larger than signal → routing was random regardless of training, regardless of aux loss. Explains the `r_uf=0.22` (random-routing math) stuck across all of Wave 1.
  - **Fix:** dropped Gumbel-STE, used vanilla softmax-STE (modern MoE pattern), added a learnable `logit_scale` (CLIP-style) before softmax so cosine signal dominates.
- **Audit pass + 3 more bug fixes** after the Gumbel finding: `softmax_top1_ste(hard=False)` was silently gradient-killing (added assertion); z_loss was opposing the new logit_scale's growth (now uses unscaled logits); added a test that verifies routing decisions actually depend on input content (this would have caught the Gumbel bug if it existed).
- **Preflight (500 steps) with all fixes** confirms the routing fix works: `r_uf` dropped from 0.22 (uniform) to ~0.05 (specialized) within 60 steps; needle val_loss at step 500 already beats Wave 1's full-run val (4.16 vs 6.85).
- **Re-launching full 8K-step retrain now** with the corrected architecture.

---

## The 30-second story

We trained Wave 1 (15K steps), found unexpected loss above vanilla
Llama, traced it through gap decomposition + scale sweep, and
identified that the architecture currently produces a memory readout
that is **pure noise** — uniform random concept routing makes the
trajectory aggregate equal to the corpus marginal. Fix is mechanical
(loosen aux-loss coefficients) and is being retested now.

---

## What we built

Llama-3.2-1B (frozen) + side-car **trajectory-memory** module:
- **N=4096 concepts** on a small-world ring graph (sparse adjacency).
- **Per 256-token window:** read J=4 parallel trajectories of K=8 hops
  on the manifold → cross-attn aggregate → inject at Llama layer 8.
  Then write J trajectories that mutate concept state (scatter-mean).
- **Cross-window TBPTT (D=4)** trains the write module via NTP CE.
- Effective Llama context capped at 2048 tokens so memory must carry
  long-range info.

Architecture diagram and training-wave details in
`docs/plan_trajectory_memory.md` and Notion (Architecture & Model
Design, Training Strategy pages).

---

## Headline numbers

| Setup | bulk val CE | answer-CE on needle | EM% |
|---|---|---|---|
| Vanilla Llama, 1024-token chunks | **2.4367** | 5.13 | — |
| Vanilla Llama, full context (needle) | — | **1.18** | 6% |
| Vanilla Llama, 2K cap (needle) | — | 4.08 | 3% |
| Wave 1 ckpt, full Llama context | — | 3.12 | 5% |
| Wave 1 ckpt, scaffold @ scale=0 (identity) | **2.3739** | 5.32 | — |
| Wave 1 ckpt, scaffold @ trained scale | **2.4639** | 5.27 | — |
| Wave 1 ckpt, 2K cap enforced at inference | — | ~6.5 | 0% |

Numbers are weighted (answer tokens × 100 weight in valid_mask) where
applicable. Full per-source / per-distance breakdown in
`docs/baseline_numbers.md`.

---

## Issue we solved (and what it taught us)

**Bug:** `cfg.effective_lm_context = 2048` was *not* enforced in
KV-cache mode (Phase 2 rollout, Wave 1's primary execution path during
generation). `past_key_values` grew unbounded — Llama silently attended
over the full document. Memory had no functional role during training.

**Fix:** `_crop_kv_cache(cache, max_length)` helper in
`src/trajectory_memory/tbptt.py`, wired into `forward_window` after
every Llama call. All 48 tests pass.

**What we then expected:** retrain under enforced cap → memory carries
long-range info → big gain.

**What we actually found** (gap decomposition + scale sweep):

1. **Memory injection is symmetric, not asymmetric, on bulk vs needle
   tokens.** Bulk cost is +0.09 nats; needle cost is similarly small
   per-token. The 1.94-nat hit on `em_ours` is because the trained
   model can't *retrieve* facts from memory, not because injection
   degrades Llama's own attention.
2. **Scale sweep is strictly monotonic.** At every positive
   scale_raw multiplier, val CE goes up. The bridge "would have" gated
   to zero but couldn't optimize there in 7K steps.
3. **Root cause: routing is uniform-random.** `r_uf=0.224 ± 0.002`
   across 7000 steps — exactly the random-routing math
   (`1 − (4095/4096)^1024 = 0.22`). `usage_ema.max() = 1/4096 =
   0.0006`. Every concept used equally. Trajectory aggregate ≈ corpus
   marginal. Cross-attn pulls in noise, not signal.

**So the bug was real, but it wasn't the load-bearing issue.** The
load-bearing issue is upstream: the aux losses
(`load_balance_coef=1e-2`, `z_loss_coef=1e-3`) push routing to
maximally-uniform when combined with cosine-similarity logit bounding.

![scale sweep](../outputs/scale_sweep.png)

---

## What's next (in flight)

Retrain Wave 1 v2 with looser aux losses:
- `--load-balance-coef 1e-3` (10× lower)
- `--z-loss-coef 0` (redundant with bounded cosine logits;
  revival mechanism at threshold 1e-5 already prevents dead concepts)
- Everything else identical (D=4, J=4, K=8, N=4096, D_concept=256,
  scale_init=0.1, effective_lm_context=2048).

Gate before commit: 500-step preflight from scratch. Pass criteria:
`r_uf` drops below 0.20, loss decreases monotonically, no NaN.

Full retrain: 8000 steps (~6h at 0.79s/step) if preflight passes.

**Watch list after retrain:**
- `r_uf` and `usage_ema.max()` — routing specialization
- `scale_raw` mean — does the bridge move from init?
- Bulk val CE — should stay near the 2.37 floor
- Needle-distance EM stratification — does memory now help past 2K?

---

## Bigger questions for the meeting

1. **Is the trajectory architecture worth the compositional complexity?**
   Alternative: flat cross-attn over all 4096 concept states (RETRO /
   Memorizing-Transformers pattern). Loses "grammar" but is a
   stronger inductive baseline.

2. **If routing specializes, do trajectories actually compose into
   useful "grammar"?** Need an experiment: visualize trained
   trajectories and check if they have semantic structure (concept ↔
   token-distribution analysis).

3. **Memory probe suite** (MemoryAgentBench, LongMemEval, NarrativeQA)
   from plan §6.5 — not started yet. Worth doing once Wave 1 v2 looks
   non-degenerate.

---

## Files referenced

- `docs/baseline_numbers.md` — full numerical record
- `docs/plan_trajectory_memory.md` — architecture + wave plan
- `docs/research_backlog.md` — open items (#10 inner-module compile,
  #11 StaticCache, #12 routing-specialization)
- `outputs/scale_sweep.{json,png}` — sweep curve
- `outputs/gap_decomp_scale_{zero,trained}.json` — ablation results
- `outputs/wave1/ckpt.best.pt` — current Wave 1 checkpoint (step 7000)
- `outputs/wave1_v2/` — retrain (in progress)
