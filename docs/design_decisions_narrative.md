# Trajectory-Memory: design decisions narrative

A chronological record of architectural decisions, each motivated by what
the diagnostic suite revealed about the prior version. Format per
version: **Design** (what + why) → **Observed trends** → **Problem + analysis** → fix.

Status: 2026-05-17. Numbers for V1.0 are from the original measurement
runs (recovered from `docs/baseline_numbers_historical.md`). Numbers for
V1.1–V1.5 are currently being regenerated on composite_v1 (retrains in
progress) — placeholders + historical predictions in the meantime.
V2.13 numbers are fresh from `outputs/wave1_v2/eval_full_10000.{json,md}`.

The unifying observation across **every** version we measured: **the
memory side-car has provided either zero or negative contribution to
language-modelling loss**. Each design iteration was a hypothesis about
*why* — and each retrain has been an attempt to make the memory
actually do work.

---

## V1.0 — Trajectory + softmax-STE routing  (`dcc61d4`, 2026-05-12)

### Design

- **Architecture**: per-cell state vectors `[N=4096, D_concept=1024]` arranged on a
  small-world ring graph. Per-window write trajectory updates cell states via
  scatter_mean; per-window read trajectory reads via cross-attention over visited
  cells.
- **Read decoupled from write** by separate `read_module` and `write_module`.
  Both walk the graph independently (different MLPs).
- **Routing change**: replaced Gumbel-STE with softmax-STE. Earlier diagnostics
  showed Gumbel noise was dominating the cosine-similarity routing signal — the
  router was effectively choosing cells at random.
- **KV-cache cap**: training set `cfg.effective_lm_context = 2048` to bound
  Llama's attention range. Motivation: memory should be the **only** path for
  information further than 2K tokens.

### Observed trends

(from the original Wave 1 measurement runs on the 1024-token train mix
+ needle eval; recovered numbers)

| Metric | Value | Source |
|---|---:|---|
| Vanilla Llama NTP CE (bulk) | 2.4367 | `outputs/vanilla_llama_train_floor.json` (lost) |
| Ours scaffold @ scale=0 (identity) | 2.3739 | `outputs/gap_decomp_scale_zero.json` (lost) |
| Ours scaffold @ trained scale | 2.4639 | `outputs/gap_decomp_scale_trained.json` (lost) |
| **Memory injection cost (bulk)** | **+0.09 nat** | scale_trained − scale_zero |
| Vanilla full-context needle answer-CE | 1.18 | `outputs/em_vanilla.json` (lost) |
| Ours full-context needle answer-CE | 3.12 | `outputs/em_ours.json` (lost) |
| **Memory injection cost (needle)** | **+1.94 nat** | the asymmetry |
| Vanilla 2K-cap needle answer-CE | 4.08 | `outputs/em_vanilla_2k.json` (lost) |
| Ours 2K-cap needle answer-CE | ~6.5 | `outputs/em_ours_capped.json` (lost) |
| `r_uf` (read uniqueness fraction) | 0.224 ± 0.002 | matches `1 − (4095/4096)^1024` exactly = uniform random |
| `usage_ema` max | 0.0006 = 1/N | fully flat across cells |
| `scale_raw` final mean (bridge gate) | 0.0964 (init was 0.1) | **never moved** |
| `inject_snr` (training telemetry) | ≈ 0.65 | injection has 65% of hidden-state norm |
| Scale sweep | strictly monotonic, no sweet spot | memory readout adds noise |

### Problem and analysis

1. **KV-cache cap was bypassed by HF generate**. `past_key_values` grew unbounded
   during training; Llama silently attended over the entire document. Memory
   was functionally redundant — the LM could solve needle recall via attention
   alone, so gradient pressure to make memory useful was near zero on those
   tokens.

2. **Routing was uniform-random**. `r_uf = 0.224` exactly matches the formula
   for random hits with K_read=1024 trials over N=4096 cells. The entry
   projection and graph walk had not learned to specialize. Diagnostic: every
   cell had usage 1/4096, no specialization.

3. **Bridge gate never moved**. `scale_raw` started at 0.1, ended at 0.0964
   after 7K steps. On bulk tokens, gradient pressure to reduce scale was weak
   (only +0.09 nat penalty). On needle tokens, pressure was strong but rare.
   Net displacement: negligible.

4. **Injection magnitude was the wrong scale**. `inject_snr ≈ 0.65` meant the
   memory readout had 65% of Llama's hidden-state norm — that's not a "gentle
   gating" perturbation, that's a major rewrite of the residual stream. On
   bulk text Llama could absorb it; on tasks needing precise residuals, it
   broke the calculation.

5. **Asymmetric injection cost**: +0.09 nat on bulk text (small) but +1.94 nat
   on in-context needle recall (large). Memory was hurting *exactly when it
   needed to help*.

**Conclusion**: the architecture had four compounding issues. The KV-cache
bug masked the architectural test; even without the bug, memory wasn't
specialized enough to help; even if it had been, the injection was too
large; and even if all that were fixed, the read/write signal didn't have a
training task that forced retrieval.

### Fix → V1.1

Build a **write-then-retrieve** task that **forces** memory to be load-bearing:
the LM at QA time cannot see the passages — only the question + whatever
memory routed in. If memory doesn't retrieve, the answer-loss stays at
the vanilla-no-context floor.

---

## V1.1 — + Wave 1 v4 retrieval protocol  (`d95f4b9`, 2026-05-13)

### Design

- **Architecture unchanged from V1.0**.
- **New training task**: synthetic biographical universe. Each chunk = 8 facts
  written sequentially, then a target QA over one of those facts. The QA
  window contains ONLY the question + answer; no passage repeated. So Llama
  *must* go through memory to answer.
- **`answer_content_token_positions`** in the data marks just the content
  tokens of each answer (e.g., positions [6,7,8] for "ember-compass-swan"),
  so the loss is computed only on the load-bearing tokens.

### Observed trends — current retrain in progress

Expected behavior from the V1.0 diagnostic story:
- val answer_loss should drop from `vanilla nc floor (≈ 4.8)` toward `~1` if
  retrieval actually works
- Cell utilization (`w_unique_per_window`, `r_unique_per_window`) should start
  showing differentiation if routing is learning
- `rw_overlap_*` metrics: should rise from random baseline

What we *historically* observed (per memory `project_routing_uniformity.md`):
the model achieved val loss ~1.23 — better than vanilla floor — but routing
was over-regularized: 8.7% of cells ever written, 3.7% ever read (mode
collapse).

### Problem and analysis — predicted from V1.0 lessons

The retrieval task alone doesn't fix routing collapse. The architectural
machinery (entry_proj → hop_step → softmax-STE) tends to converge to a small
high-confidence subset of cells regardless of input. Without
load-balance pressure + noise injection during training, the system is in
a bad equilibrium: highly-used cells get reinforced because their state
already encodes more, dead cells stay dead because they never get a
gradient signal.

The retrieval task *did* make the LM's answer-loss respond to memory, which is
why val dropped to 1.23. But the model is using a tiny **memorization
subset** (350 cells out of 4096) — not exploiting the full 4096-cell
capacity. The graph topology was being ignored.

### Fix → V1.2

Add a **flat-bank ablation**: replace the trajectory-walker with top-K
cell attention (no graph). Same parameter count. This isolates whether
the graph topology is what's broken vs the entire routing approach.

---

## V1.2 — + Flat-bank ablation + decode probe  (`f2f77ea`, 2026-05-13)

### Design

- **Architecture unchanged for trajectory**.
- **Flat-bank baseline** added as a sibling module (selectable via `--flat-bank`).
  Replaces the per-window walker with top-K attention over all N cells.
- **Decode probe**: feeds the read trajectory output directly to a separate
  Llama decoder, samples 100 tokens, summarizes what the trajectory "speaks".

### Observed trends — current retrain in progress

What we *historically* observed (per memory
`project_trajectory_underperformance_diagnosis.md`):

| Metric | Trajectory | Flat-bank |
|---|---:|---:|
| val_loss @ step 10K | 1.23 | **1.15** ← won by 0.08 nat |
| Cells ever written | 357 / 4096 (8.7%) | 2,214 / 4096 (54.1%) |
| Cells ever read | 153 (3.7%) | 686 (16.7%) |
| Unique cells per K=8 trajectory | 7.02 | 8.00 |
| Distinct entries per 8-fact chunk | 6.59 | 7.69 |
| Cross-fact write overlap | 0.153 | 0.098 |

### Problem and analysis

1. **The flat-bank competitor won, but for the wrong reason**: during the
   debug sweep that introduced flat-bank, it got three Mixtral-style
   routing fixes applied (small init, noisy gating, top-K renormalization).
   The trajectory model didn't get these fixes. So the comparison was
   unfair: trajectory's routing was the limiter, not the graph itself.

2. **Within-chunk routing for trajectory was fine** — 6-7 unique cells per
   8-fact chunk, low cross-fact overlap. The trajectory model can *route
   8 facts to 8 cells* within a chunk. The problem is across chunks:
   91% of inputs end up routed to the same 350-cell subset.

3. **Decode probe outcome** (from memory): the trajectory readout was
   less discriminating than flat-bank's. Llama, given the trajectory
   readout alone, decoded generic tokens rather than fact-specific
   content.

**Conclusion**: the architectural ablation (graph vs flat) cannot be
trusted until the routing fixes are applied to BOTH sides. The current
result (flat-bank wins by 0.08 nat) is a routing-difference artifact, not
an architectural verdict.

### Fix → V1.3/V1.4

Apply the same Mixtral-style fixes to the trajectory model and rerun the
head-to-head.

---

## V1.3 / V1.4 — + Mixtral routing fixes on trajectory  (`311c2aa` → `c103ffe`, 2026-05-14)

### Design

Three changes from the MoE literature (Switch Transformer / Mixtral):

- **Small init for entry_proj**: std = 0.01 instead of default. Reduces
  early over-confident routing decisions.
- **Noisy gating during training**: add Gaussian noise (std = 0.5) to
  per-hop softmax logits before `softmax_top1_ste`. Forces the router to
  not lock in early, gives dead cells a chance.
- **Lower temperature, stronger load-balance loss**. Pushes the routing
  distribution toward uniformity early in training.

V1.4 was a tuning iteration: noise std 0.5 → 0 and init std 0.01 → 0.05
("dial back the over-correction"). The noise was making the gradient
signal too noisy in early training; smaller noise + slightly bigger init
was a better operating point.

### Observed trends — current retrain in progress

What we *expected* but never measured (the retrain was deferred for the
v2 rewrite):

- Cell utilization should jump from ~8.7% to 40-60% (matching flat-bank).
- Trajectory val_loss should approach or beat flat-bank's 1.15.
- If the graph topology actually carries information, trajectory should
  *exceed* flat-bank once routing is fair.

### Problem and analysis

This is the gap in our knowledge. We've never measured V1.3/V1.4 cleanly
on composite_v1. The current retrain will tell us:

- If trajectory ≈ flat-bank: graph topology is decorative; flat attention
  is sufficient.
- If trajectory > flat-bank: graph helps, multi-hop matters.
- If trajectory < flat-bank: even with fair routing, graph overhead hurts.

### Fix → V1.5

Even with fair routing, the read trajectory and write trajectory might not
*align* (read for a question goes to different cells than write for the
matching fact). Add explicit contrastive losses that pull read and write
trajectories together.

---

## V1.5 — + Per-hop trajectory-state contrastive loss  (`7bec656`, ~2026-05-14)

### Design

Two contrastive losses train read↔write alignment explicitly:

- **Entry-cell InfoNCE**: for each chunk, the read entry-cell should be the
  same as the write entry-cell for the target fact. Positive: same fact.
  Negatives: other facts in the same chunk. Temperature 0.07.
- **Per-hop trajectory-state InfoNCE**: at each hop k of the read trajectory,
  pull the read's state at hop k toward the write's state at hop k for the
  target fact. Pull away from other facts. Coefficient 0.05.

The hypothesis: if the read trajectory is *trained* to match the write
trajectory pointwise, then it must visit the same cells and end up in the
same place. This removes the freedom for read to develop a separate
internal language from write.

### Observed trends — current retrain in progress

Predictions:
- `l_contrast_entry` should drop from ~3.7 nat (random) to <1.5 nat by step 5K
- `l_contrast_per_step` should drop from ~2.0 to <0.7
- `rw_overlap_entry` should rise from random ~0.2 to >0.5
- `rw_overlap_hop` should rise from ~0 to >0.1
- val answer_loss should drop further than V1.4 IF the alignment translates
  into the readout actually carrying fact-specific signal

### Problem and analysis — what we'll learn

Most important question: does explicit alignment training translate into
**memory contribution** (non-zero `with_mem − no_mem` NLL gap)?

If yes: contrastive losses are the key, and V2 was probably the wrong
direction.

If no: alignment of read/write trajectories doesn't fix the underlying
issue — the bridge MLP between the readout and Llama's residual stream is
the bottleneck (it can produce question-conditioned outputs that Llama
ignores). This was the motivation for V2's complete redesign.

### Fix → V2

The v1 architecture has *per-cell* state — every cell carries a
D_concept-dim vector that gets EMA-updated on every write. With 4096 cells
this is a 4M-parameter recurrent state — hard to train, hard to specialize.
V2 replaces this with a **vocabulary** (concept IDs) + **sparse edges**:
each cell is anchored by a learnable embedding (the "vocabulary"), and
content is stored in **edges** between cells (sparse).

---

## V2.0 → V2.13 — Vocabulary-trajectory rewrite  (`6cb713c` → `65fe2f1`, 2026-05-17)

### Design

Complete architectural reframing:

- **Vocabulary**: N=4096 concept IDs as a learnable matrix
  `concept_ids = id_proj(id_basis)` with SimVQ reparameterization
  (Vector Quantized — the cells are anchored points in embedding space).
- **Sparse edges**: instead of per-cell state, each cell has up to K_max=32
  outgoing edges (each edge = `[dst_id, edge_state]`). Edge_state is
  EMA-updated only on the specific edges the write trajectory traversed.
  Total memory: 131K possible edges vs 4M D_concept-dim cell states.
- **W-TinyLFU eviction**: edges are bounded; eviction policy tracks
  read/write touch ratios with EMA decay + age floor.
- **Walker step is per-WINDOW, not per-token**: walker fires once per
  256-token window, dramatically reducing kernel-launch overhead.
- **Hopfield-tied EntryProjector**: read and write share entry projection
  weights (forces alignment by construction).
- **NPMI co-activation tracker** + **revive_dead_concepts**: catches
  routing collapse early and rebuilds dead cells from co-activated pairs.

Together these address every diagnostic failure from V1.x:
- Vocabulary anchors → routing can't drift everywhere
- Sparse edges → bounded memory, eviction handles long-tail capacity
- Per-window walker → faster training, more passes
- Hopfield tying → no read/write alignment loss needed (theoretically)
- NPMI + revival → catches collapse signals

### Observed trends — measured this session

(`outputs/wave1_v2/eval_full_10000.{json,md}`, 800 val chunks paired
across modes)

**Memory contribution (THE headline):**

| Probe | NLL/tok | First-token NLL |
|---|---:|---:|
| v2 (memory active) | 1.5325 | 1.8865 |
| v2 (manifold empty, no writes) | **1.5330** | **1.8859** |
| Vanilla Llama no-context | 4.7783 | 4.3396 |
| Vanilla Llama full-context (8 P) | 3.7313 | 4.3269 |

- **Memory contribution: +0.0005 nat** (effectively zero, exactly equal
  to the noise floor on this metric).
- Gap to vanilla no-ctx: -3.2458 nat (v2 wins by a lot)
- Gap to vanilla full-ctx: -2.1988 nat (v2 even beats Llama-with-everything-in-context)

**Cross-question read divergence: Jaccard = 0.084.** Read trajectories
DO route differently per question (low Jaccard = high divergence). So
the routing is doing the right thing. But the readout doesn't change
Llama's output.

**Routing health (training-time MA):**

| Metric | Value | Healthy? |
|---|---:|---|
| `w_unique_per_window` | 7.8 (24% of K_max) | ✅ way above v1's 0.003 collapse |
| `aux_lb` | 50 (down from 710 at step 1K) | ✅ converged |
| `mean_edge_state_norm` | 29.4 (target √D = 32) | ✅ stable |
| `n_active_edges` | 108K (83% of N×K_max cap) | borderline; eviction active |

**Per-task NLL (every task shows v2 ≡ v2_no_mem to 3 decimals):**

| Task | v2 | v2_no_mem | vanilla_nc |
|---|---:|---:|---:|
| biographical | 3.903 | 3.903 | 5.049 |
| boxes | 1.605 | 1.605 | 4.342 |
| calendar | 0.845 | 0.845 | 4.021 |
| knights | 0.519 | 0.519 | 3.771 |
| passphrase | 2.122 | 2.122 | 6.759 |
| preferences | 1.188 | 1.188 | 5.583 |
| revisions | 1.387 | **1.392** | 3.496 |
| theory_of_mind | 1.185 | 1.185 | 3.425 |
| triage | 0.880 | 0.880 | 6.341 |

### Problem and analysis

**The architecture as built does not use the memory side-car.** Every diagnostic
points to the same conclusion:

1. **Headline metric is identical with/without memory** (+0.0005 nat).
2. **Per-task breakdown identical** across all 9 task families.
3. **First-token NLL identical** (which kills the teacher-forcing leak
   explanation).
4. **Cross-question Jaccard 0.084** confirms the read module IS receiving
   the question and routing accordingly — the failure is downstream, in
   how the readout connects back to Llama.
5. **v2 still beats vanilla by 3.25 nat** — but that gain is entirely from
   the trained adapter (format-priors: "calendar answers are dates",
   "knights answers are knight/knave", etc.), NOT from retrieval.

**The root cause hypothesis**: `mem_inject`'s W_in/W_out bridge MLP can
produce question-conditioned readouts (we know this because Jaccard is
low), but it gets absorbed by Llama's normalization or attention without
informing the next-token logits. Need to probe the bridge MLP directly:
- log readout magnitude per question
- inject random readouts and see if the loss changes
- ablate mem_inject entirely (force memory_fn → zero) and compare

These probes are next on the list. If `mem_inject` is the bottleneck, V3
might need to inject memory at a different layer / use a stronger gating
mechanism / route through cross-attention instead of residual addition.

---

## Cross-cutting observations

What the diagnostic series across V1.0 → V2.13 establishes:

1. **Memory side-car contributing to LM loss requires more than architectural
   correctness.** Every design we've measured shows the side-car routing
   improving (cell utilization went 8.7% → 83%; routing collapsed early,
   un-collapsed late) without the LM loss responding. The plumbing between
   memory and prediction is the limiting factor, not memory's internal
   correctness.

2. **Training metrics can mislead.** V2.13's val_loss dropped to 1.42
   (best in series) — looks like progress — but the entire gap to
   vanilla came from format-learning + teacher-forced AR continuation,
   not memory retrieval. Without paired with-mem vs no-mem probes, we
   wouldn't have caught this. **Future runs must include this probe
   from day 1.**

3. **Capacity isn't the problem.** V2.13 has 108K active edges (83% of
   cap), strong routing diversity, stable edge norms — the architecture
   *has* the capacity. The signal isn't being USED.

4. **The right test is generation, not teacher-forced NLL.** The current
   eval still uses TF on the answer span, which leaks via the gold prior
   tokens. The next eval iteration needs autoregressive decode + exact
   match.

5. **The architecture's success criterion has been clarified by the
   diagnostic work.** The right bar isn't "val_loss drops" or even "we
   beat vanilla no-context". The right bar is **"v2 NLL < v2_no_mem NLL
   by ≥ 0.5 nat on retrieval tasks"**. By that standard, no design we
   have built passes yet.

---

## Status of evidence

| Version | Numbers | Source |
|---|---|---|
| V1.0 | Historical, lost from disk but in `baseline_numbers_historical.md` | needs no retrain (architecture obsolete) |
| V1.1 | Pending retrain on composite_v1 | `outputs/v1.1/` (running) |
| V1.2 trajectory | Pending retrain | `outputs/v1.2_trajectory/` (running) |
| V1.2 flat-bank | Pending retrain | `outputs/v1.2_flatbank/` (running) |
| V1.4 | Pending retrain | `outputs/v1.4/` (running) |
| V1.5 | Pending retrain | `outputs/v1.5/` (running) |
| V2.13 | **Measured** in `outputs/wave1_v2/eval_full_10000.{json,md}` | this session |

The retrain queue is running in `outputs/v1_chain.log`. Each row above
becomes concrete as ckpts land. Until then, the V1.x rows use historical
predictions from the diagnostic work documented in
`baseline_numbers_historical.md` and the memory notes
`project_routing_uniformity.md`, `project_trajectory_underperformance_diagnosis.md`,
and `project_capacity_concern.md`.
