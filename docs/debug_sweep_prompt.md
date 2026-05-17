# Debug Sweep Prompt — Trajectory-Memory v2

> Hand this prompt to a fresh AI agent (Claude, GPT, etc.) that has filesystem
> access to this repo. The agent should produce a list of high-confidence bugs
> + a stability-risk assessment. Under 1000 words back.

---

## What this codebase is

A from-scratch rewrite (`src/trajectory_memory_v2/`) of a side-car memory
system for Llama-3.2-1B. The original v1 lives at `src/trajectory_memory/`
and had two specific failure modes we are paranoid about repeating:

1. **Routing collapse** — over training, all reads and writes collapsed to
   the same ~1% of memory cells. v1 used aux load-balance + z-loss to fight
   this. v2 routes over the full N=4096 vocabulary at every hop (v1 only
   at the entry), so the same coefficients would scale ~400× larger; v2
   therefore drops coefs from 1e-2/1e-3 to 1e-4/1e-4.

2. **R↔W partition problem** — reads converged on a cell set disjoint from
   what writes had populated, so memory retrieval was structurally broken.
   v1 fixed it with InfoNCE contrastive loss. v2 inherits the contrastive
   loss + Hopfield-ties read and write through a shared `EntryProjector`.

The v2 design lives in `docs/design_vocabulary_trajectory.md`. Read it first.

## What you should audit

Read these files end to end:

- `src/trajectory_memory_v2/config.py` — hyperparameters
- `src/trajectory_memory_v2/manifold.py` — VocabularyManifold + edge buffers
  (EMA + 4-feature eviction + protection floors)
- `src/trajectory_memory_v2/walker.py` — TrajectoryWalker (entry + K-1 hops,
  step_query construction, edge contribution scatter)
- `src/trajectory_memory_v2/read_module.py`, `write_module.py` — thin wrappers
- `src/trajectory_memory_v2/integrated_lm.py` — Llama + manifold integration,
  forward_window (the entry point used by every trainer)
- `src/trajectory_memory_v2/trainer.py` — Wave 1 retrieval trainer
- `src/trajectory_memory_v2/wave2_trainer.py` — Wave 2 streaming SFT trainer
- `scripts/training/train_wave1_v2.py` and `scripts/training/train_wave2_v2.py`

For comparison, the v1 trainer is at `src/trajectory_memory/training/phase1_retrieval.py`.

## Specific things to check

### 1. Gradient flow

Every learnable parameter must receive non-zero gradient under the standard
loss. Trace the path from `total_loss.backward()` back through:

- `manifold.id_basis` + `manifold.id_proj.weight` (the concept vocab,
  via SimVQ reparameterization)
- `walker.lambda_edge` (the scalar mixing edge-score into routing)
- `walker.cross_attn` + `walker.history_attn` + `walker.step_mlp` +
  `walker.cue_proj` + `walker.pos_enc`
- `entry_proj.head_query`
- `mem_inject_layer.bridge_in / bridge_out / scale` (inside the Llama stack)
- `read_attn` (per-token cross-attn from Llama hiddens to trajectory)

Any `.detach()` or `@torch.no_grad()` on a tensor that should carry
gradient is a bug. Edge buffer updates are intentionally `@torch.no_grad()`
(they use a separate signature path, not the autograd graph) — that is
correct. But `step_query` going into vocab scoring must NOT be detached.

### 2. Routing-collapse risk

v2 routes over all N=4096 cells at every hop. With `load_balance_coef=1e-4`
and `z_loss_coef=1e-4`, are these coefficients strong enough? Look at the
raw magnitudes printed by smoke tests (`/tmp/wave1_v2_smoke.jsonl`,
`/tmp/wave2_v2_smoke.jsonl`).

Specifically: at random init the raw aux_lb is ~1500 and aux_z ~340, giving
contributions of ~0.15 + 0.034 = 0.18 nats. Compare to answer loss ~3 nats.
Is 6% adequate to prevent collapse? Cite the v1 baseline that worked.

If you think coefs are too low, give a justified target.

### 3. R∩W overlap mechanics

The contrastive loss is supposed to align read and write entry/step queries.
Verify:

- `EntryProjector` is genuinely shared between `read_module` and `write_module`
  (look at `IntegratedLMV2.__init__` lines around `self.read_module = ...`).
- The InfoNCE positives/negatives in `Phase1RetrievalTrainerV2._entry_contrastive`
  and `_per_step_contrastive` are constructed correctly. Pay attention to
  in-batch negatives, temperature, and the masking.
- Per-step contrastive aligns the *step queries* — which are the routing
  queries, not the cell embeddings visited. This is intentional but means
  the loss does not directly constrain the visited-cell sets. Is the
  `rw_overlap_*` metric the only proof that alignment is working? Should
  there be a more direct loss?

### 4. EMA + eviction dynamics

- `α = max(α_base / (1 + log(1 + visit_count)), α_min)` — at α_base=0.1 and
  visit=100 this gives α≈0.022. Is that enough to keep edges learnable for
  popular nodes, or do they silently freeze?
- Eviction score uses 4 normalized features (visit, stale, norm, specificity)
  with protection floors. Check the math in
  `manifold._select_eviction_victim`: are clip/normalization steps right?
  Could you construct a pathological case where useful edges are evicted?
- `protect_max_frac=0.30` — what happens when the protected set fills?
  Does eviction silently no-op, causing the buffer to lock?

### 5. Wave 2 streaming BPTT

`Wave2TrainerV2._forward` splits the prior into `max_prior_windows` windows
and chains `prev_window_hiddens` between them. Verify the carry-over is
`.detach()`'d so backward doesn't unroll through all prior windows. If it
is not, the trainer will OOM and silently waste compute on no-op gradients
through the frozen Llama backbone.

### 6. Numerical stability

- `edge_score` is scaled by `eff_scale = D^-0.5` to put it on the same scale
  as `vocab_logits`. Verify (`walker.py:265` area).
- Aux losses at init: `aux_lb` should be O(log(N)) when routing is uniform,
  O(big) when collapsed. The raw 1500 reported is plausible for random
  routing over N=4096 at J=4. Sanity-check this on paper.

### 7. Test mode (`attach_lm=False`)

`IntegratedLMV2(cfg, attach_lm=False)` is used by some unit tests. Verify
the test-mode forward_window path doesn't reference undefined locals
(`read_result`, `write_result`).

## What "high confidence" means

Only report issues you would stake your reputation on. For each, give:
- File + line number
- One sentence on what's wrong
- One sentence on the fix
- Confidence percentage (only report ≥80%)

If something looks suspicious but you can't fully verify, list it
separately as "needs follow-up", but don't pad the main findings.

## Out of scope

- Style nits, naming preferences, doc improvements
- Performance unless it crosses an obvious cliff (e.g. quadratic loop in hot path)
- Anything that requires running the model (you have read-only filesystem access)

Report under 1000 words. Bug list first, then a short stability assessment.
