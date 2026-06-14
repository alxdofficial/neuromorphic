# Research backlog — post-Wave-1+2

Consolidated index of ideas surfaced during Wave 1 monitoring that we
agreed to defer until the production training runs finish. Each item
points at the existing scaffolding (stub script, eval doc section, or
project memory) where the design already lives.

Date opened: 2026-05-11. Updated 2026-05-12 with perf-optimization + routing-specialization items.

---

## 10. torch.compile via inner-module wrap — landed; verify Wave 2 speedup

**Status:** landed 2026-05-12. Verify the predicted speedup holds in Wave 2.

`IntegratedLM.compile_inner_modules()` compiles `read_module.forward`,
`write_module.forward`, and `MemInjectLayer.forward` separately (1-2
stable input variants each), leaving the branchy `forward_window`
orchestrator eager. `forward_window` was thrashing torch.compile with
64+ distinct call signatures (None-or-tensor inputs, KV-cache mutation,
train/val grad differences) — hitting the recompile cap and falling back
to eager regardless.

Expected speedup per `docs/profile_analysis.md`: 10-15% step-time
reduction. Wave 1 (the run that's underway now) launched in eager mode
before the fix landed; Wave 2 picks it up automatically.

**When Wave 2 finishes:**
- Compare Wave 2 step time against the eager baseline (Wave 1 at 0.82-0.90 s/step).
- Expected: ~0.70-0.80 s/step in Wave 2.
- If gap is smaller than expected, investigate which inner module is still hitting cap (check `outputs/wave2/training.log` for `recompile_limit` warnings).

## 11. Llama compile via StaticCache — the bigger perf win (deferred)

**Status:** deferred; ~1 day effort.

The inner-module compile gives 10-15%. The bigger win is compiling
Llama itself, which is **99.3% of compute** per the profile data
(`docs/profile_analysis.md`). Currently Llama runs eager-with-flash-attn
because we use HF `DynamicCache`, which:
- Mutates a Python list of K/V tensors in-place each call
- Causes `torch.cat`-driven shape growth (different shape every call)
- Both of these thrash torch.compile and prevent cudagraph capture

Fix: switch `DynamicCache` → `StaticCache` (pre-allocate K/V at
`max_cache_len = effective_lm_context = 2048`, write into indexed
slots). Then `model.llama.model = torch.compile(..., mode="reduce-overhead")`
becomes viable, unlocking cudagraphs.

Expected speedup: **25-40% on Llama time → ~25% on step time** (Llama
is 80%+ of step). Combined with item #10, total expected step time:
**0.55-0.65 s/step** vs current eager 0.90 s/step.

**Implementation outline:**
1. In `src/pretrained/hosts/llama.py`: construct a `StaticCache` instance
   at model init, attach to the host wrapper.
2. In `src/trajectory_memory/training/phase1.py` + `phase2.py`: replace
   `DynamicCache()` instantiation sites with the pre-allocated static one.
   Reset (not reallocate) between sequences via `cache.reset()`.
3. In `src/trajectory_memory/integrated_lm.py forward_window`: replace
   `past_key_values.get_seq_length()` with the static cache's current
   length attribute; ensure `cache_position` indexing handles wraparound
   correctly for TXL-style continuous training.
4. Compile: `model.llama.model = torch.compile(model.llama.model, mode="reduce-overhead", fullgraph=True)`.

**Risk:** static cache wraparound logic for cross-document training
(TXL-style continuous, no doc-boundary resets) needs careful testing.
HF docs have recipes for this.

Bigger win for Phase 2 (AR rollout) than Phase 1 (TF), because AR
rollout fires Llama forward per-token where Python overhead dominates —
cudagraphs eliminate it.

## 12. Routing-specialization investigation — concepts may not be specializing

**Status:** observation logged 2026-05-12, post-fix Wave 1 run; needs investigation post-Wave-2.

The post-normalization-fix Wave 1 run shows **perfectly uniform routing**:
- `r_uf = 0.222` (essentially constant across 7000+ steps; std=0.002)
- This matches the math for uniform-random routing: `unique_per_step ≈ N · (1 - (1-1/N)^M) = 4096 × (1 - 0.778) ≈ 907`, so `0.907/4096 ≈ 0.22`.
- `usage_ema`: 4096/4096 active, **0 dominant** (>10/N usage), max single-concept usage 0.0006.
- `r_self_overlap = 0.000` — no within-trajectory revisits.

This is technically healthy (no collapse) but means **concepts are not
specializing** — every concept gets visited approximately equally, and
the manifold acts as a uniform-random scratchpad rather than a semantic
vocabulary.

**Likely cause:** combination of (a) Switch load-balance loss α=1e-2 +
ST-MoE z-loss β=1e-3 explicitly pushing toward uniform routing, and
(b) L2-normalized routing producing bounded cosine logits ∈ [-1, 1]
that are difficult to specialize without explicit per-concept biases.

**Why this matters for the grammar thesis:** if concepts are functionally
interchangeable (no specialization), the architectural argument that
"discrete vocab + grammar > flat banks" is undermined — we'd have a
4096-slot manifold that behaves like a 4096-slot random scratchpad.

**Two experiments post-Wave-2:**
1. **Aux-loss coefficient sweep** (cheap, single training run per
   config): try α=1e-3, β=1e-4 (10× lower). Less regularization pressure,
   more freedom for the model to specialize concepts on natural-data
   priors.
2. **Per-concept biases** (architectural): add a small learnable
   per-concept bias to routing logits, initialized at 0. Lets the model
   develop "hot" concepts for common content patterns while still
   preventing magnitude-driven runaway.

**How to verify post-experiment:** Tier 2 viz
(`scripts/diagnostics/viz_concept_language.py`) — if concepts are
semantic, each one should have a coherent top-K-tokens distribution.
If concepts are uniform-random, the top-K tokens per concept will be
indistinguishable from the corpus marginal.

---

## 1. Memory probe ablation — clean needle-eval signal

**Status:** deferred from current Wave 1.
**Where:** `scripts/training/train_wave1.py` `run_answer_only_val()` already does
half of this (answer-span-only loss). Need to extend with:

a. **Memory-off val pass.** Same `eval_wave1` loop but inject zeroed
   (or `prev_states=None` reset per chunk). Cost: one extra ~30s val
   pass per save.

b. **Per-distance stratification.** Stratify `val_answer_loss` by
   `target_distance` bin (3K, 8K, 16K, 32K). Doc metadata already has
   this in `NeedleDoc.target_distance`. Track separate series
   `val_answer_loss_{3k,8k,16k,32k}`.

c. **Llama-only baseline.** One offline run, vanilla Llama frozen, no
   memory module, same val docs. Provides the lower-bound "what context
   alone can do" signal.

**Key metric:** Δ(distance) = answer_loss[memory_off] − answer_loss[memory_on].
If Δ ≈ 0 at 3K but Δ ≫ 0 at 16K/32K — that's the memory-contribution
proof. Required for any honest paper claim.

**Confounds to handle:**
- Force tau→0 (deterministic routing) in this eval to remove Gumbel noise.
- Answer-class priors (alphanum_short ≈ log(32) bits/char) can drive
  loss down without memory doing anything — see the conversation thread
  for the full analysis.

---

## 2. Compositional-generalization benchmark (custom dataset)

**Status:** new; tests the core "grammar > flat-bank" thesis.
**Where:** would live alongside `src/trajectory_memory/data/needle_haystack.py`
as a new generator.

Train the model on docs covering a Cartesian product of {entities A..Z}
× {actions α..ω} × {objects 1..N}, but **hold out specific
(entity, action, object) triples**. At eval, test held-out triples.

- Flat-KV baselines (Larimar, kNN-LM stub) should memorize seen triples
  but degrade on held-out — they have no compositional bias.
- A truly compositional manifold should generalize: same entity concept
  + same action concept = correct prediction even for novel combinations.

**Why this matters:** This is the cleanest direct test for the
[grammar thesis](../.claude-memory/project_thesis_grammar.md) we've
adopted. Without this, "concepts compose" is a hand-wave.

**Cost:** ~1 week of dataset design + a few training runs. Defer to
after main waves complete.

---

## 3. Capacity-per-parameter vs flat-KV baseline

**Status:** new; quantifies the architectural bet.
**Where:** new bench script `scripts/bench/bench_capacity_vs_flat.py`.

Match parameter counts:
- **Our manifold:** N=4096 × D=256 × 2 (ids + state_init) = 2.1M params.
- **Flat-KV equivalent:** N'=8192 token-keys × d_lm=2048 = 16.8M params.
  Better matchup: N'=4096 × d_lm=512 = 2.1M.

Both bolted onto frozen Llama. Measure long-doc retention at matched
param budget. If trajectory access really gives N^K vs N states-per-
param, our manifold should hold strictly more info per byte on long-
context retention tasks (BABILong, MemoryAgentBench CR-MH).

**Open design question:** what's the "flat-KV equivalent"? Larimar's
matrix is the closest published thing; Memorizing Transformers' per-
layer kNN cache is bigger. Match against the smallest published
baseline that demonstrates the function.

---

## 4. Visualization Tier 2 — concept ↔ language correspondence

**Status:** stub.
**Where:** [`scripts/diagnostics/viz_concept_language.py`](../scripts/diagnostics/viz_concept_language.py)

Produces: concept→top-K tokens dictionary, per-source usage barplot,
trajectory traces for picked docs. The "are concepts semantic?" test.
~30 min GPU run.

---

## 5. Visualization Tier 3 — needle write→read overlap

**Status:** stub.
**Where:** [`scripts/diagnostics/viz_needle_recall.py`](../scripts/diagnostics/viz_needle_recall.py)

Per-doc Jaccard overlap between write_visited@needle_pos and
read_visited@answer_pos, stratified by distance. The cleanest
mechanistic memory-contribution test. ~10 min GPU.

---

## 6. Visualization Tier 4 — trajectory decoder (inversion probe)

**Status:** stub.
**Where:** [`scripts/diagnostics/viz_trajectory_decoder.py`](../scripts/diagnostics/viz_trajectory_decoder.py)

Train a small `trajectory → text` decoder, use it to *speak* the
manifold's contents. Produces the "concept dictionary" — the
interpretability headline. Modes:

- **Per-concept definitions** (length-1 trajectories visiting only
  concept #i, decode 100 samples).
- **Interpolations** along the small-world graph (concept A → B
  geodesic).
- **Counterfactuals** (swap one hop, decode).
- **Direct needle readout** (decode write_trajectory at needle position
  — if the decoder emits the needle's content, that's direct proof of
  memory encoding).

~3-5 hour GPU training cost; could end up being the paper money figure.

Design choice to pin before implementing: from-scratch decoder (clean
causal story but slower to train) vs Llama-with-second-adapter (fast,
but Llama's pretraining can shortcut). Recommend from-scratch ~50M
transformer.

---

## 7. Hopfield energy landscape

**Status:** speculative.
**Where:** would extend Tier 4.

Treat `concept_ids` as stored patterns in a modern-Hopfield-net energy:
`E(q) = -logsumexp(q @ concept_ids.T / β)`. Plot energy along a real
document's token sequence. Should dip at high-content tokens (proper
nouns, technical terms) and stay flat in filler — mirrors the modern-
Hopfield-attention story (Ramsauer 2020). If the manifold has organized
coherently, this gives a clean "memory recall" signal.

---

## 8. Manifold evolution across training (animation)

**Status:** speculative, low-cost.
**Where:** would extend `scripts/diagnostics/viz_manifold.py`.

We already have ckpts at steps 5000, 6000, 7000, 8000, 9000, 10000,
11000+. Load them, run UMAP on concatenated `state_init` with a shared
embedding, color points by which ckpt + draw arrows per concept. Either
a multi-panel grid or an animated GIF. Tells us if concepts are still
moving or have settled.

---

## 9. Causal intervention probe

**Status:** speculative, paper-grade.
**Where:** would need a new hook in `IntegratedLM.forward_window`.

Take a high-usage concept i. Manually zero its state vector before a
forward pass. Re-run inference on a held-out doc. Measure downstream
NTP-loss change. If concept i is load-bearing for predicting certain
tokens, zeroing it should hurt those tokens specifically. The
interpretability gold standard for "what does concept i encode."

Anthropic's SAE causal-intervention work (Templeton 2024 "Scaling
Monosemanticity") is the closest analog.

---

## Cross-references

- Architectural thesis: [`project_thesis_grammar.md`](../.claude-memory/project_thesis_grammar.md)
- Track A/B benchmarks: [`docs/eval_plan.md`](eval_plan.md)
- Architecture spec: [`docs/plan_trajectory_memory.md`](plan_trajectory_memory.md)
- Viz tier roadmap: [`scripts/diagnostics/README.md`](../scripts/diagnostics/README.md)

## Items deferred from 2026-05-12 audit

13. **State row norms unbounded by GRUCell tanh.** The `||current_state||
    <= 1` comment in `write_module.py:89` is wrong — GRUCell's tanh
    bounds per-element, not per-row, so row norms can reach `sqrt(D)≈16`
    (observed: ~8 after many self-writes). Affects pos_enc relative
    magnitude and cross-attn input scale. Not load-bearing (downstream
    cross-attn has `1/sqrt(d_attn)` compensation), but worth a row-norm
    clamp toward `cfg.state_init_norm` if downstream attn signal looks
    weak.

14. **Wave 1 train doesn't reset state at doc boundaries; Wave 1 val
    does.** TXL-style continuous training in train, doc-local in val —
    asymmetric semantics. Probably intentional but should be documented
    in `plan_trajectory_memory.md` explicitly.

15. **`record_visits` decay 0.99 applied 8 times per step.** Effective
    decay per step = `0.99^8 ≈ 0.923`, a ~13-step horizon, not the
    ~100-step implied by `0.99`. Consider one concatenated update per
    step instead of per-window.

16. **Add tests for `softmax_top1_ste` directly + logit_scale
    sensitivity.** Current gumbel tests are legacy; active primitive is
    `softmax_top1_ste`.

17. **Bridge/read-attn gradient test on the real path.** Existing tests
    use the test-mode fallback (no Llama). A test that wires a small
    LM and asserts grad flows from final loss back through bridge →
    read-attn → manifold would catch a future regression the current
    tests miss.

## Priority order when Wave 1+2 complete

1. **Item 1 (memory probe ablation)** — required for any honest claim. Cheap.
2. **Item 4 (Tier 2 viz)** — confirms or refutes "concepts are semantic." Cheap.
3. **Item 5 (Tier 3 viz)** — direct memory-contribution test on needle. Cheap.
4. **Item 6 (Tier 4 trajectory decoder)** — paper headline figure. Costly but worth it.
5. **Item 2 (compositional generalization)** — the thesis-validation experiment. Most ambitious.
6. **Item 3 (capacity vs flat-KV)** — needed to claim parameter efficiency.
7. **Items 7-9** — speculative; revisit after the above land signals.
