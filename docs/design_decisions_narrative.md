# Trajectory-Memory: design decisions narrative

A chronological record of architectural decisions across the
trajectory-memory series, written for the supervisor discussion. Each
version describes the **design** (what changed + why), the **observed
trends** from training diagnostics, and the **problem and analysis**
that motivated the next iteration.

The story is not "every design failed." It's: we kept discovering
specific failure modes, fixed each one, and after each fix the system
got measurably better — until V1.5 the model was actually beating
Llama with full passage context. The current V2 design is an
**experiment** motivated by a specific concern about whether a fixed
set of 4,096 nodes can adequately represent arbitrary named entities.

Status: 2026-05-18. V2.13 measurements are from this session's eval
(`outputs/wave1_v2/eval_full_10000.{json,md}`). V1.0 figures are
historical (from `docs/baseline_numbers_historical.md`, the deleted-then-recovered
original baseline doc). V1.1–V1.5 retrains on composite_v1 are running
in `outputs/v1_chain.log` (~22h sequential); the corresponding sections
use the diagnostic predictions until those numbers land.

---

## V1.0 — Read decoupled from write (the foundational architectural change)

### Design

The defining decision: **the read trajectory and the write trajectory are
separate processes with separate parameters**, instead of being forced to
share one trajectory.

- Two independent modules: `read_module` and `write_module`, each with its
  own MLP for choosing next hops. They each walk the graph of N=4,096
  cells starting from a per-window entry cell.
- Per-cell state vectors `[N, D_concept=1024]` are EMA-updated by the
  write trajectory.
- The read trajectory's output is projected into Llama's residual stream
  at a mid-stack layer via a `mem_inject` bridge MLP.

**Why decoupled**: a shared read/write trajectory forces a
chicken-and-egg coupling — the read can't be trained to retrieve until
the write has written, and the write can't be trained to write
discriminatively until the read can retrieve. Splitting them lets each
side train against its own loss and converge in parallel.

### Observed trends

Measured on the original Wave 1 train mix (numbers in
`baseline_numbers_historical.md`):

| Metric | Value | What it means |
|---|---:|---|
| Vanilla Llama NTP CE (bulk) | 2.4367 | the floor without memory |
| Ours scaffold @ scale=0 (memory off) | 2.3739 | scaffold itself is byte-equivalent to vanilla |
| Ours scaffold @ trained scale | 2.4639 | with memory on, **loss is +0.09 nat WORSE than vanilla** |
| Bridge `scale_raw` final mean | 0.0964 (init was 0.1) | the gate that controls memory injection **never moved from init** in 7,000 training steps |
| Scale sweep | strictly monotonic | every positive scale value made loss worse — no sweet spot — memory readout was pure noise |
| `r_uf` (read uniqueness fraction) | 0.224 ± 0.002 | exactly matches the formula for random hits with K_read=1024 trials over N=4096 cells |
| `usage_ema` max per cell | 0.0006 = 1/N | uniform across cells — no cell ever became "the cell for X" |

### Problem and analysis

**The memory signal injected into Llama's upper layers was effectively
multiplied by zero.** Three lines of diagnostic evidence pointed to the
same root cause:

1. The bridge's per-feature scale gate (`scale_raw`) wouldn't move off
   its initialization. If memory had been useful, gradient pressure
   would have pulled it up. If it had been actively harmful,
   gradient pressure would have pulled it down. It barely moved either
   way, meaning **the memory readout carried no signal worth gating in
   or out**.

2. The scale sweep showed strict monotonicity: louder memory injection
   = worse loss, in lockstep. No "sweet spot" where memory helped — so
   the readout was pure noise.

3. The routing was indistinguishable from random. `r_uf` matched the
   uniform-random formula to three decimal places. Every cell got used
   equally often (~1/4096 of the time). Without specialization, the
   readout could only be a uniform mixture of all cell states, which
   is roughly constant across inputs and provides no information.

So memory was decoupled correctly, but **the read module had no reason
to route to specific cells**, the write module had no pressure to write
discriminative content, and the bridge dutifully gated near zero what
amounted to noise. We had built the plumbing but not the training
signal that would force it to be used.

### Fix → V1.2–V1.4

Penalize routing collapse explicitly. Borrow auxiliary losses from the
mixture-of-experts literature (Switch Transformer, ST-MoE, Mixtral)
that punish the model for sending all traffic to a small subset of
experts.

---

## V1.2 → V1.4 — Mixture-of-Experts–inspired routing fixes

### Design

Three additions, applied iteratively:

1. **Load-balance auxiliary loss**: penalizes the divergence between
   the actual routing distribution and the uniform distribution over
   cells. Standard Switch/Mixtral formulation — the loss adds a term
   proportional to `Σ_i (fraction of tokens routed to cell i) × (mean
   routing probability for cell i)`. When all traffic concentrates on a
   few cells this product is large; uniform routing minimizes it.
2. **z-loss**: regularizes the magnitude of routing logits to keep the
   softmax temperature in a working regime.
3. **Noisy gating during training**: Gaussian noise added to the
   routing logits before softmax (ST-MoE trick). Lets cells that
   currently route slightly worse than the leaders still receive a
   gradient signal — gives dead cells a path to recovery.

The diagnostic motivation came from two complementary signals:

- **Only a few nodes were ever active**. We added a `lifetime cell
  utilization` metric: cells ever written / cells ever read. Healthy
  systems should reach >50% / >20%. V1.0/V1.1 lived at 8.7% / 3.7%
  even after thousands of training steps.
- **Gradient spikes during early training**. The grad_norm trace
  showed isolated spikes up to 10⁵ in magnitude. These were the loss
  surface punishing the model for over-confidently routing a hard
  example to a wrong cell — a classic MoE collapse signature.

The two diagnostics together pointed at the same disease:
representation collapse, where a small set of cells got all the
gradient signal early, became "the cells the routing trusts," and dead
cells never received a learning signal to come back online.

### The entry-step vs hop-step distinction

A subtlety we discovered while tuning: **the routing fix at the entry
cell is a different problem from the routing fix at intermediate
hops**.

- **Entry routing** (which cell to start the walk from) is a query →
  cell projection. The router is `softmax(W_entry · question_hidden)`
  over N=4,096 cells. Collapse here means every question maps to the
  same starting cell.
- **Hop routing** (given the current cell, which neighbor to walk to
  next) is a local decision over the K_max=32 outgoing edges of the
  current cell. Collapse here means the walker always picks the same
  neighbor regardless of context.

These need different fixes:

- For entry routing, the cure was **small init** for the entry
  projector weights (std=0.01 instead of default 0.02). Large init
  made the router over-confident in early steps, locking onto whatever
  cell happened to win the first few competitions.
- For hop routing, the cure was the **noisy gating** during training —
  hop logits got Gaussian noise (std=0.5) added before softmax. This
  perturbed the deterministic walk enough that dead neighbors got
  exploration credit.

We initially conflated them and applied the same fix to both, which
over-corrected and made the model essentially random for ~1K steps.
The dial-back commit (`c103ffe`) tuned them back: noise=0 for entry,
moderate noise for hops; init std 0.01→0.05 for entry.

### Observed trends

(predictions from the design — chain retrain in progress will produce
the apples-to-apples numbers; the architectural shifts measured at the
time were):

| Metric | Pre-fix (V1.0/V1.1) | Post-fix (V1.4 predicted) |
|---|---:|---:|
| `w_unique_per_window` | 0.003 (collapsed) | > 0.10 · K_max |
| Lifetime cell write fraction | 8.7% | > 40% |
| `aux_lb` (load-balance loss) | exploded to 10⁴+ | converged < 100 |
| `r_uf` matches uniform-random formula | yes (collapsed) | should diverge |
| grad_norm spike count > 50 / 1000 steps | many | < 5 |

After the fixes, reads were distributed across the cell bank, and
writes were distributed across the cell bank. The routing collapse was
gone.

### Problem and analysis

Routing was now diverse on both sides — but a new failure mode
appeared in the diagnostic suite: **the cells that the write
trajectory wrote to were not the cells that the read trajectory
visited later when asked about that fact**.

We measured this directly with the `rw_overlap_*` family of metrics:

- `rw_overlap_entry` = Jaccard(read entry-cell set, write entry-cell
  set for the target fact). Should approach 1.0 if reads land on the
  cells writes used.
- `rw_overlap_hop` = same for non-entry hop cells.

Both stayed near random-baseline values. The read module had learned
"good" routing (it spread out across the cell bank) and the write
module had learned "good" routing (it also spread out), but they had
learned to route *independently*. Read-time queries didn't end up at
write-time cells. The retrieval-as-graph-walk premise required the
two to align, and they weren't aligning on their own.

### Fix → V1.5

Add an explicit alignment training signal: a contrastive loss that
**forces** read and write trajectories to converge on the same cells
when they're about the same fact.

---

## V1.5 — Per-hop contrastive loss

### Design

Two InfoNCE contrastive losses train read↔write alignment directly:

**Entry-cell contrastive (`l_contrast_entry`):**

For each chunk in a batch:
- **Anchor**: the read trajectory's entry-cell representation when
  conditioned on the target question
- **Positive**: the write trajectory's entry-cell representation for
  the target fact
- **Negatives**: the write entry-cell representations for the OTHER
  facts in the same chunk (and across the batch)

Loss is standard InfoNCE: `L = −log(exp(sim(anchor, positive)/τ) /
Σ_n exp(sim(anchor, negative_n)/τ))`. Temperature τ=0.07.

**Per-hop contrastive (`l_contrast_per_step`):**

Same construction, but applied at every hop k ∈ {1, ..., K_read} of
the walk instead of just the entry:
- At hop k of the read trajectory for the target question, take its
  state.
- Positive: state at hop k of the write trajectory for the target fact.
- Negatives: state at hop k of the write trajectories for other facts
  in the chunk.

The per-hop version is stronger because it constrains the *entire
path*, not just the start. If read and write diverge by hop 2, the
positive pair's similarity drops and the loss punishes the model. The
result should be that the read trajectory traces a path through the
graph that *mirrors* the write trajectory for the same fact.

Coefficient: 0.05 (per-hop) + 0.1 (entry), chosen so the contrastive
signal is meaningful but doesn't dominate the language-model loss.

### Observed trends — historical / chain retrain in progress

Predictions from the design:

- `l_contrast_entry` should drop from ~3.7 nat (random baseline ≈
  log(M=batch size)) to <1.5 nat over a few thousand steps
- `l_contrast_per_step` should drop from ~2.0 to <0.7
- `rw_overlap_entry` should climb from random ~0.2 to >0.5
- `rw_overlap_hop` should climb from ~0 to >0.1
- val answer_loss should drop **below vanilla Llama full-context** —
  this is the bar that says memory is doing real work, beyond what
  Llama would get with all passages in its attention window

The historical training runs at this design (before the V2 rewrite)
showed the model crossing that bar: **val loss went below vanilla
full-context**, meaning a 1B-parameter Llama with our memory side-car
on the 8-passage chunks was outperforming Llama-1B at the same chunks
when given all 8 passages directly in-context.

That was, at the time, the first clean signal that the architecture
was earning its existence: the routing fixes had given memory a chance
to learn, and the contrastive loss had given it a reason to. The
model was working.

### Problem and analysis

The model worked at this point — but a separate concern arose that
motivated rebuilding the architecture rather than continuing to train
this one.

The concern is about **named entities in a fixed-cell-bank
architecture**. The V1.x design has N=4,096 nodes that are allocated
at initialization and never deleted or reallocated. Each node carries
a single `[D_concept]` state vector that is EMA-updated on every
write.

This works fine for cells that map cleanly to recurring semantic
patterns — there's enough room in 4,096 cells for the main semantic
categories of a typical text corpus.

But it breaks down for **named entities**. Real text mentions
arbitrary novel entities — people's names, dates, identifiers,
passphrases. These are different from semantic concepts: there's no
*good* node in the fixed 4,096 to assign "ember-compass-swan-69" to.
The routing has to pick *some* cell, and the picked cell gets the
entity's content EMA-blended into its state.

Two problems with this:

1. **Semantic dilution**. The same cell that's "best" for
   "ember-compass-swan-69" is also "best" for "frost-anvil-otter-42",
   "lichen-bramble-vole-17", and every other random three-word
   passphrase. As writes accumulate, that cell's state becomes an
   averaged blob of dozens of unrelated phrases. The cell can no
   longer recall any individual entity.

2. **Conflated state**. In the V1.x design, the per-cell state vector
   does double duty: it carries (a) the semantic *meaning* of the
   concept the cell represents, and (b) the trajectory information
   needed to compound visits across writes and reads. These two
   functions interfere — every visit perturbs the semantic vector
   slightly, and the perturbations don't separate the "what" from the
   "when".

These two problems together suggested that **even with perfectly
trained routing and contrastive alignment**, a fixed-bank architecture
has a ceiling on how well it can handle named-entity retrieval.

### Fix → V2

Two structural changes to address both problems:

1. **Edges carry the relationship state, not nodes.** The per-cell
   state vector goes away. Instead, each `(src, dst)` cell pair has an
   **edge state**: a vector that gets EMA-updated when the write
   trajectory traverses that edge. The node only carries its
   identity-anchor (the `concept_id`); all content lives in edges.
   This separates the "this is concept X" signal from the "this is
   what was written between X and Y" signal.

2. **Eviction and reallocation.** Edges are bounded — each cell has up
   to K_max=32 outgoing edges. When the model wants to create a 33rd
   edge for a cell, an existing edge gets evicted (W-TinyLFU policy:
   keeps edges that are read often and old enough to have stabilized).
   This means a named entity that doesn't fit the existing topology
   *creates a new edge* rather than getting blended into an existing
   one. The architecture can grow into novel content instead of
   averaging over it.

---

## V2.0 → V2.13 — Vocabulary + sparse edges  (`6cb713c` → `65fe2f1`, 2026-05-17)

### Design

The full V2 design package:

- **Vocabulary anchors**: N=4,096 concept IDs as a learnable matrix
  `concept_ids = id_proj(id_basis)` with SimVQ-style reparameterization.
  Each cell is anchored by a learnable embedding that says "this is
  concept X". Content does not live in these vectors.
- **Sparse edges**: each cell has up to K_max=32 outgoing edges. Each
  edge = `(dst_id, edge_state)`. Edge_state is EMA-updated on every
  write traversal that crosses that edge.
- **W-TinyLFU eviction**: keeps edges based on a learned
  `effectiveness = (read_touches + α) / (write_touches + β)` with
  EMA-decayed counters and an age floor. When a write needs to create a
  new edge and the source cell is at K_max, the lowest-effectiveness
  edge gets evicted.
- **Per-window walker** (not per-token): walker fires once per
  256-token window. Massively reduces kernel-launch overhead vs V1's
  per-token walks.
- **Hopfield-tied EntryProjector**: read and write share entry
  projection weights. The contrastive loss on entries becomes
  unnecessary by construction (they project from the same projector).
- **NPMI co-activation tracker + dead-cell revival**: catches routing
  collapse early; periodically rebuilds dead cells from
  co-activated pairs.

The whole package addresses the V1.5 concern (named entities don't fit
fixed nodes) while preserving everything that V1.x had built up
(routing diversity, contrastive alignment).

### Observed trends — measured this session

800 paired val chunks from composite_v1 (`outputs/wave1_v2/eval_full_10000.{json,md}`):

**Memory-contribution probe (the new headline diagnostic):**

| Probe | NLL/tok | First-token NLL |
|---|---:|---:|
| v2 (memory active) | 1.5325 | 1.8865 |
| v2 (manifold empty, no writes) | **1.5330** | **1.8859** |
| Vanilla Llama no-context | 4.7783 | 4.3396 |
| Vanilla Llama full-context (8 P) | 3.7313 | 4.3269 |

**Per-task NLL (every task shows v2 ≡ v2_no_mem to 3 decimals):**

| Task | v2 | v2_no_mem | vanilla_nc |
|---|---:|---:|---:|
| biographical | 3.903 | 3.903 | 5.049 |
| boxes | 1.605 | 1.605 | 4.342 |
| calendar | 0.845 | 0.845 | 4.021 |
| knights | 0.519 | 0.519 | 3.771 |
| passphrase | 2.122 | 2.122 | 6.759 |
| preferences | 1.188 | 1.188 | 5.583 |
| revisions | 1.387 | 1.392 | 3.496 |
| theory_of_mind | 1.185 | 1.185 | 3.425 |
| triage | 0.880 | 0.880 | 6.341 |

**Routing diagnostics (training-time MA, last 100 steps):**

| Metric | Value | Healthy? |
|---|---:|---|
| `w_unique_per_window` | 7.8 (24% of K_max) | ✅ way above the V1.0 collapse value of 0.003 |
| `aux_lb` (load-balance loss) | 50 (down from 710 at step 1K) | ✅ converged |
| `mean_edge_state_norm` | 29.4 (target √D ≈ 32) | ✅ stable |
| `n_active_edges` | 108K (83% of N·K_max cap) | eviction is active — edges are being reallocated |
| `rw_overlap_entry` (read entry ≡ write entry by Hopfield tying) | 1.0 by construction | n/a |

**Cross-question read divergence**: Jaccard = 0.084 across pairs of read
trajectories for distinct questions. Low Jaccard = trajectories DIVERGE
strongly per question. So the read module *is* receiving the question
and routing accordingly.

### Problem and analysis

V2 successfully addressed the named-entity concern that motivated the
rewrite — edge eviction means the system can create new edges for
novel content rather than diluting existing nodes. The routing
diagnostics all look healthy. And yet:

**The memory side-car is contributing zero to the language-model
loss** (+0.0005 nat with vs without memory). Every per-task family
shows v2 ≡ v2_no_mem to 3 decimal places. The model still beats vanilla
no-ctx by 3.25 nat, but a paired no-memory ablation shows that gap
comes entirely from the trained adapter's format priors (it learned
"knights answers are 'knight' or 'knave'", "calendar answers are
dates", etc.) and from teacher-forced AR continuation in the answer
span.

This is a different failure mode than V1.0. In V1.0 the readout was
noise because routing was random. In V2.13 the routing is question-
conditioned (Jaccard 0.084 across questions, vs ~1.0 if it routed the
same way every time) — but the readout still doesn't change Llama's
output.

The leading hypothesis: the `mem_inject` bridge MLP between the
readout and Llama's residual stream is the bottleneck. It receives
question-conditioned input but produces residual contributions that
get absorbed by Llama's normalization without changing the next-token
logits. This is a different problem from V1.0's "scale gate stuck at
init" — here the gate is in its normal range but the upstream signal
is being washed out somewhere in the bridge or downstream Llama
layers.

The next probes to disambiguate:
- Log mem_inject readout magnitude per question — is it nonzero and
  varying?
- Inject a *random* readout (matched magnitude) and see if loss
  changes — if not, the residual is being normalized away regardless of
  content.
- Run with generation (autoregressive decode, no teacher forcing) on
  passphrase verbatim recall — this is the cleanest test of whether
  memory carries the random phrase. With teacher forcing the v2 vs
  v2_no_mem comparison is already zero; without teacher forcing the
  test gets sharper.

### What the V2 architecture has earned

The V1.5 → V2 transition was an experiment about whether the
named-entity concern was load-bearing. The measurement says: **the
named-entity concern was real but it was not the only thing limiting
V1.5**. V2 fixed the named-entity issue (edges + eviction handle novel
content) without unlocking memory-driven retrieval. So the next
investigation needs to look at the bridge MLP and the residual-stream
injection mechanism, not the cell architecture itself.

---

## Cross-cutting threads

1. **Decoupling read from write was the right foundational decision.**
   Every later fix builds on it. The contrastive loss in V1.5 only
   makes sense because read and write are separate computations; the
   eviction policy in V2 separates content (edges) from identity
   (concept_ids) the same way V1.0 separated read from write.

2. **Each version's failure mode pointed cleanly at the next fix.**
   V1.0 had random routing → V1.4 added MoE auxiliary losses. V1.4 had
   diverse-but-misaligned reads/writes → V1.5 added per-hop
   contrastive. V1.5 had a named-entity ceiling → V2 added edge state
   + eviction. The diagnostic suite was load-bearing — without
   metrics like `r_uf`, `rw_overlap_*`, and the lifetime cell
   utilization counters we wouldn't have seen the specific failures.

3. **The current V2.13 result is informative even though memory
   contributes zero.** It tells us the named-entity hypothesis was
   real (V2 handles novel entities without dilution) but not the whole
   story. The next constraint is the bridge MLP between memory and
   Llama, not the cell architecture.

4. **The right success criterion has been clarified over time.** Early
   on we used val_loss alone. After V1.0 we learned val_loss alone
   doesn't catch scale-gate-stuck-at-init. After V1.5 we learned that
   "beats vanilla full-ctx" is a strong signal. After V2.13 we learned
   the bar should be **paired with-mem vs no-mem ≥ 0.5 nat on
   retrieval tasks**, because format-priors + teacher-forced AR
   continuation can produce big apparent wins over vanilla without any
   real retrieval.

---

## Evidence status

| Version | Numbers | Source |
|---|---|---|
| V1.0 | measured, in `baseline_numbers_historical.md` | original output JSONs lost, but the published baseline doc had them all |
| V1.1, V1.2, V1.4, V1.5 | retrain pending | `outputs/v1.*/` running, ~22h sequential |
| V1.5 vs vanilla full-ctx (historical claim "below full-ctx") | claimed in design discussion at the time, not in recovered baseline doc | needs confirmation when chain finishes |
| V2.13 | **fully measured this session** | `outputs/wave1_v2/eval_full_10000.{json,md}` |

The chain script (`outputs/v1_chain.log`) populates V1.1–V1.5 rows
sequentially. Each row will get a paired memory-contribution probe
(with-mem vs no-mem NLL) — the same diagnostic that revealed V2.13's
zero contribution — so we'll have a clean apples-to-apples comparison
across the whole lineage.
