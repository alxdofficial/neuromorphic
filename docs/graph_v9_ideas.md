# Graph V9 — Operator-Node Memory (current design)

**Status:** design COMPLETE (architecture level), not built. Supersedes the 2026-06-11 operator-graph
sketch (this file's previous content) after the 2026-06-12 design session. Open questions are listed
at the bottom; none block the flat probe.

**One-liner:** a fixed-footprint memory sidecar for a frozen Llama, organized as a pyramid of node
layers. Each node = a slow learnable key + a fast value, where the value is an ordered list of
**generalized Householder factors** (direction + strength pairs). Input tokens stream through the
pyramid: they activate nodes by key, get *transformed* by the activated nodes' factor chains, and the
operated sequence passes up. Nodes evolve by surprise-gated delta writes and by **within-layer
Hebbian absorption** (strength-conserving transfer between coactivating nodes). The read runs the
same circuit on the question and hands the transformed query back to Llama.

**The thesis (unchanged):** input-tailored vocabulary as memory. If the model can build, per input,
a small set of "words" that efficiently describe exactly what it is observing, then at matched float
budget it stores more than a flat bank — because a sufficiently specific description IS a memory.
Nodes are both the vocabulary and the storage; the v8 double-duty failure is fixed not by separating
them but by repairing the three actual root causes (written keys, contractive fusion, no content
channel).

---

## 1. Why v8 failed (carried forward — still the motivation)

emat_bio gate, 600 steps: every graph version flat (SHUF−REAL ≈ 0) with large OFF−REAL = generically
useful codebook, zero example-specific binding. Beacon (+2.07 SHUF−REAL at 102M params) proves the
task is winnable; its edge is a per-layer attention read.

Root causes, each with its v9 fix:

1. **Convex/contractive aggregation** (softmax pooling collapses to the hull) → fuse is now operator
   *composition* (productive: generates new elements; finite alphabet → unbounded language).
2. **Stacked linear aggregation flattens** (layer 2 = re-weighted layer 1) → layers are successive
   *re-descriptions of the data*, not re-mixings of the same state; composition is non-commutative.
3. **Value-linear read can't synthesize bindings** → read = apply the stored operator to the query
   (degree-2 in query × stored content).
4. **Input content never reached stored values** (input picked coefficients over a static dictionary)
   → factor directions/strengths are written per input (arm A) or selected-and-composed per input
   (arm B); either way REAL and SHUF produce different stored state.

Session addition: **keys were also written in v8c** → address drift (write-time address ≠ read-time
address). v9 keys are never written.

---

## 2. The primitive: generalized Householder factors

```
one factor  =  (direction v, strength β)      |v| = 1,  β ∈ [0, 2]
apply to x  :  x ← x − β·(vᵀx)·v
```

- Established since 1958 (QR decomposition); in ML: orthogonal RNNs (Mhammedi 2017), Householder
  flows (Tomczak & Welling 2016), and the modern linear-RNN line — DeltaNet's transition is exactly
  this, Grazzi 2024 shows β ∈ [0,2] (negative eigenvalues) unlocks state tracking beyond TC⁰,
  DeltaProduct composes several per step. We arrange a known-good primitive; we don't invent one.
- **Intuition:** a partially-silvered mirror owning one direction. β=0 skip, β=1 erase (projection),
  β=2 flip (reflection). Everything perpendicular passes untouched.
- **Stability for free:** spectral norm ≤ 1 at any β in range, at any composition depth. Nothing
  blows up; gradients pass through chains non-expansively (orthogonal-RNN argument). Works at init:
  strengths start at 0 → every operator is the identity → the pyramid is transparent at step 0.
- **β must keep its full [0,2] range.** Clipping to [0,1] "for stability" silently deletes the
  expressivity claim (negative eigenvalues are the TC⁰ escape). Stability never needed the clip.

**Where the expressivity edge actually lives (be precise):** NOT in one step, NOT in the 4-layer
stack (any fixed-length product is TC⁰-computable). It lives in composition length that GROWS with
the input — the persistent node state compounds multiplicatively across the stream (writes +
absorptions), so by the end of a document the effective composed operation has length ~ document
length. Non-commutative growing products = group word problems = NC¹-hard (cups-and-ball /
S₅ / parity — all provably beyond fixed-depth attention, all natural for this algebra; a swap is
literally one β=2 factor). On order-free tasks the edge must come from parameter efficiency instead.

---

## 3. Architecture

- **Pyramid of layers** (default 4). Layer ℓ has N_ℓ nodes (decreasing) and S_ℓ slots per node
  (increasing); default heuristic: N_ℓ × S_ℓ ≈ constant, so every layer carries the same fast state.
- **Node = slow key + fast value.**
  - key: learnable parameter (key dim). NEVER written at inference. Addressing is stateless
    recomputation: same content → same routing at write and read time.
  - value: ordered list of S_ℓ factor slots (direction in code dim, strength in [0,2]).
    Strength 0 = empty slot = identity. All directions live in ONE shared code space (composition
    across anything requires a common space). Key dim and code dim are separate config knobs.
- **One Llama tap** (mid-stack, ~the inject@13 lore), projected to seed codes.
- **Cross-layer interface = the operated token sequence ONLY.** No cross-layer node dynamics, no
  donors/recipients across layers, no doubling-ladder bookkeeping. Each layer is self-contained:
  its own keys, projections, coactivation table, plasticity MLP inputs.
- **Per-layer ephemeral state:** lagged coactivation table C (node × node, asymmetric:
  "i fires now × j fired recently"), built from a short score trace; learnable decay per layer for
  both table and trace (the v8c decay-ladder, repurposed). Reset per input.
- **No decay on slot strengths anywhere.** Strengths bounded by construction; turnover = gated
  overwriting (eviction by demand, not by clock); consolidation conserves strength (see §5).

Sizing convention: fit total fast state to the historical 26K-float budget for baseline
comparability (e.g. 4 layers × nodes×slots×(code dim+1) summed; code dim chosen to fit).
All gains/scales learnable or dimension-scaled (no bare constants).

---

## 3b. Why layers at all (vs one node layer + stacked attention)

Productive composition does NOT justify depth (consolidation is within-layer; one layer gets it
too). The pyramid's actual justifications:

1. **Higher layers can name things that don't exist below.** Layer 0 words form only about patterns
   in raw seed codes; layer 1 receives the stream AFTER layer-0 words acted on it → its words form
   about patterns in how the lower vocabulary was used. A flat layer can never coin a word about its
   own words.
2. **Multi-resolution memory** — surface bindings low, bundles/gist high, read out at matched
   decoder depths (Beacon's proven edge).
3. **Combinatorial reuse (the thesis):** flat = one node per distinct pattern; pyramid = small
   alphabet + composition = exponential expressible set at incremental cost (BPE economics).

"One layer + multiple rounds of attention" IS v8: attention rounds are convex re-mixings of the
same state → flattening (root cause 2). Our depth re-describes the DATA; it does not re-mix nodes.

Falsification: the pyramid must beat the flat operator memory AT MATCHED FLOATS on bundle-reuse
structure, or the flat one wins and the thesis is wrong (de-risking ladder §10 step 3).

## 4. Write path (per token, per layer)

```python
# code: the token's running code at this layer (layer 0: projected Llama hidden state)
scores = score(code, keys)                      # ephemeral; routing query proj is slow

# (a) WRITE — gated delta into activated nodes
v_prop = normalize(W_dir @ code)                # proposal from THIS layer's input
b_prop = squash_to_0_2(W_str @ code) * surprise # surprise = decoder NLL, free
for i in nodes:                                 # all soft; tiny gates = tiny writes
    gate = scores[i] * surprise * update_gate_mlp(...)
    deposit(node_i, v_prop, b_prop, gate)       # §6 unified primitive, no debit = injection
                                                # displacement IS the eviction policy

# (b) APPLY — transform the token by the activated nodes' chains
for i in fixed_node_index_order:                # NEVER score-sorted (argsort flips)
    for s in slot_order(i):
        v, b   = direction[i][s], strength[i][s]
        b_eff  = scores[i] * b                  # score scales strength, not position
        code   = code - b_eff * dot(v, code) * v
# one running vector, chained — node outputs are NEVER averaged

# (c) BOOKKEEPING
C     = table_decay * C + outer(scores, trace)  # lagged: row=fires-now, col=fired-earlier
trace = trace_decay * trace + scores

# (d) CONSOLIDATE (see §5)

# (e) PASS UP — operated code is the next layer's input
```

- Layer 1+ proposals are projections of *layer-below-operated* codes → higher-layer content is
  vocabulary-filtered by construction (flow-through). Higher layers never tap Llama directly:
  prevents shortcut routing around the vocabulary and makes lower factors load-bearing (dead
  operators degrade everything above → the loss feels it; endogenous pressure, no aux losses).
- Apply-vs-write order within a token step: minor open question (§9).

---

## 4b. Compute schedule: chunkwise execution (the v8c lesson)

Per-token SEMANTICS, chunk-boundary EXECUTION (default chunk = 128 tokens):

- **Parallel within a chunk** (values frozen at chunk start): scores (exact — keys are slow),
  applies (each token's chain independent), deposit proposals + surprise (held, not landed).
- **At each chunk boundary, in order:** (1) land the chunk's merged deposits; (2) update the
  coactivation table — EXACT in closed form (within a chunk, scores never depend on the table →
  the EMA scan over 128 score rows batches into one shot); (3) ONE absorption pass using the fresh
  table + just-landed writes; (4) next chunk reads updated state.
- The one approximation: within-chunk staleness (a fact written at token 10 isn't readable at
  token 90 of the same chunk — one-chunk self-read latency). Standard chunkwise trade; harmless
  for the gate protocol (passage chunks write, query chunks read). Chunk size = freshness vs
  throughput knob.
- **Cadence is part of the model:** learnable gate scales calibrate to the trained cadence —
  train-time and eval-time chunk size must match.
- Chunk boundaries = recurrence steps for backprop; full BPTT across the window's chunks
  (checkpointing), not truncation.

## 5. Consolidation: within-layer Hebbian absorption

The vocabulary-formation mechanism. When the lagged table says nodes keep co-firing, the later-firing
node absorbs its predecessors — so next occurrence, one node does the coalition's work.

```python
for (i, j) in all_pairs:                        # fully soft; batches into matmuls;
                                                # cadence (every k tokens) is the compute lever
    g = squash(C[i][j]) * surprise * plasticity_mlp(key[i], key[j],
                C[i][j], C[j][i], occupancy(i), occupancy(j))
    for s in slots(j):                          # per-slot migration, donor j → absorber i
        deposit(node_i, direction[j][s], strength[j][s], g,
                debit_from=(j, s))              # §6 unified primitive; debit = TRANSFER
        # matched part reinforces i's existing slot, novel part fills free slots,
        # donor debited exactly what landed — conservation
```

- **Transfer, not copy** (conservation law): the donor is debited exactly what the absorber gains.
  Kills the duplicate-application bug (a strong factor applied twice via two co-firing nodes
  partially CANCELS — reflections compose to identity), frees donor slots (turnover), and yields a
  per-layer invariant: total strength changes only via token injection + overwrite displacement;
  consolidation only relocates. Directly measurable (telemetry §8).
- **Plasticity MLP = the learned grammar.** Raw coactivation is frequency; the MLP (slow weights,
  fed both nodes' keys, both lag directions, occupancies) learns WHICH KINDS of pairings deserve
  merging. Frequency proposes, grammar disposes.
- **Direction convention:** later-firing absorbs earlier-firing (it fires with the pattern's full
  context behind it). The MLP sees both lag directions and can veto/reverse per pair-kind.
- **Order placement:** migrated factors land wherever routed; position is fuzzy only while strength
  is small (≈ identity), frozen as it grows. Strict temporal-order placement: deferred unless
  telemetry shows order errors.
- occupancy(i) = Σ_s strength[i][s] / (2 · S_ℓ) ∈ [0,1], soft.

---

## 6. The unified deposit primitive (ONE write mechanism, two callers)

Every fast-state change goes through one primitive — minimizes confusion, one implementation,
one unit-test surface, one telemetry hook:

```python
def deposit(node, v_in, b_in, gate, debit_from=None):
    matched_frac, matched_slot = soft_match(v_in, node.slots)  # read-before-write
    novel_frac = 1 - matched_frac
    amount = gate * b_in                                       # soft-capped at 2
    strengthen(matched_slot, amount * matched_frac, refine_toward=v_in)  # reinforce in place
    allocate_to_free_slots(node, v_in, amount * novel_frac)    # room-weighted soft assignment
    if debit_from is not None:
        debit(debit_from, amount)                              # conservation (transfer)
```

| caller            | source of (v, b)                  | gate                                  | debit |
|-------------------|-----------------------------------|---------------------------------------|-------|
| per-token write   | projection of layer's input code  | score × surprise × update-gate MLP    | no (injection) |
| absorption        | a donor node's slot               | squash(C) × surprise × plasticity MLP | yes (donor, equal) |

- **Matched fraction reinforces in place** (repeat mention → strength consolidates, direction
  refines); **novel fraction allocates to free space** (room-weighted softmax, learnable
  temperature). Duplicates can't pile up in either path (duplicates CANCEL under composition —
  this is load-bearing, not cosmetic). The earlier separate "novelty" multiplier is superseded:
  novelty = the allocation share, never a write-blocker.
- Full node + novel content: no room → almost nothing lands; with debit semantics the content
  stays with the donor. Occupancy feeds the gate MLPs so the grammar can learn to refuse early.
- Everything soft: no argmin, no thresholds, no hard top-k semantics (top-k only ever as a compute
  approximation, prefer cadence reduction instead). Strength cap at 2 via soft saturation, not clip.
- Telemetry falls out: injection flux = deposits w/o debit; transfer flux = deposits w/ debit;
  displacement = what allocation overwrote.

---

## 7. Read path

- The question flows through the SAME circuit — seed, score, apply, pass up — with write gates closed
  (no writes, no consolidation from the query segment by default).
- The read result per layer = the operated query code: the question as transformed by the bindings
  the input wrote. Nothing is "retrieved" — the memory answers by steering the question's direction
  (the stored mirror-arrangement rotates query-direction toward answer-direction).
- Addressing works because keys are slow + routing is stateless: question content re-activates the
  nodes statement content wrote into — THE central thing training must learn (W_r mapping statement
  and question phrasings to matching routing directions).
- Hand-off to Llama: per-layer un-projection (code dim → Llama hidden dim), then either
  (a) prepend as memory tokens (matched-read fairness with baselines — the harness decision), or
  (b) inject per layer at matched decoder depths via MemInjectLayer (Beacon-shaped — its proven
  edge). Default: (a) for the flat probe, (b) once the pyramid exists. Open question §9.
- Across nodes the apply is the same chain as the write path (no output averaging anywhere).
- **Multi-hop for free:** the query steered by node A's factors then meets node B already-steered —
  one pass through a layer can follow a chain of associations; a flat attend-and-retrieve read
  cannot compose hops in one shot.
- Question flows per-token (same granularity as the write); pooling the question to one seed is a
  probe-level simplification only (convex mixing before the productive machinery).

---

## 7b. Horizons: episodic mode vs lifelong mode

- The reset policy is a MODE, not a mechanic. Episodic (benchmark) mode: fast state resets per
  input. Lifelong mode: fast state persists across documents indefinitely. Writes, absorption,
  conservation, routing — all identical in both.
- No-decay is the lifelong-correct forgetting policy: displacement-by-demand forgets only under
  actual competition for space; time-based decay would erase rare-but-important old knowledge for
  being old. The per-layer strength-budget telemetry is the lifelong health dashboard.
- Over long horizons the surviving fast state BECOMES the general vocabulary: recurring coinages
  get continuously refined (novelty-gated reinforcement on re-encounter), non-recurring ones get
  displaced — BPE economics running forever. "General language vs input-specific vocabulary" =
  two AGES of the same state, not two tiers. (The pyramid runs slow→fast bottom→top; lifelong adds
  a consolidation gradient over time.)
- Honest caveat: training is short-horizon (TBPTT); lifelong behavior relies on the RULES (gates,
  grammar, routing) generalizing train-short → deploy-long. Cross-document interference at scale is
  untested; name it, don't assume it away.
- Read at lifelong scale requires sharp routing (irrelevant vocabulary costs capacity, never
  correctness, ONLY while routing stays discriminative — watch routing entropy + key
  discriminability as the lexicon grows).
- FUTURE DIRECTION (the reason lifelong matters): next-state prediction in the model's OWN
  vocabulary space — sequence modeling over the pyramid's activations/codes instead of raw tokens.
  BPE-one-level-up with a dynamic lexicon; higher layers are slowly-varying/abstract → cheap, long
  effective horizons. Payoffs: (a) endogenous surprise (memory predicts its own next state;
  prediction error replaces borrowed decoder NLL → self-contained gating, predictive-coding proper —
  the TEM / Dynamic Predictive Coding lineage in our research notes); (b) dense memory-local
  training pressure (fixes the weak-gradient-pressure problem the emat_bio floor exposed). This is
  a future PRIMARY objective for a standalone system — not an aux loss on current gate runs.

## 7c. Initialization & scale control (no magic numbers)

**Controlled by construction (init-proof):** fast state on bounded manifolds (unit directions via
normalize-parameterization; strengths in [0,2] via 2·sigmoid; convex per-slot blends) — no write
sequence can blow up; apply chain non-expansive at any depth (forward AND backward). Step 0 exactly
inert: strengths 0 → identity pyramid; empty C tables → absorption dormant; zero-init read
un-projection (LoRA-style) → Llama bit-identical to no-memory, gradients still flow. No warmup
schedules.

**The five managed spots:**
1. Llama tap: RMSNorm → project (1/√fan-in) → RMSNorm. Seed codes enter at unit RMS.
2. Inter-layer: RMSNorm between layers (mirrors only shrink → prevents cascade quieting).
   Direction = information; magnitude = incidental. Stack is norm-stationary.
3. Routing: cosine scores, temperature DERIVED: √d_k logit normalization + solve temperature for a
   target effective-k active nodes given N (structural choice, ~a handful). N changes → re-derive,
   never re-tune.
4. Surprise: relative, not raw — learnable affine of (NLL − running mean)/running spread through
   sigmoid. Frozen decoder → stationary stats.
5. C table: ROW-NORMALIZE before gating ("what fraction of my co-firing was you?") — scale-free,
   kills the squash's magic scale. Decay horizons: learnable, geometric ladder across layers tied
   to chunk size.

**Init table:** projections 1/√fan-in; un-projection out = 0; keys random-normalized
(quasi-orthogonal free in high dim); slot strengths 0; gate MLPs weights-0 bias-mid (constant gate,
grads via bias); strength-head bias mid; match-softmax temperature ∝ √d_c (random cosines ~1/√d_c).

**Scale taxonomy (enforced by review policy):** every scale is (a) derived from dims, (b) learnable
w/ principled init, or (c) an algebra bound. A bare config float outside these = bug. Subtle case:
chunk-boundary deposit merge must use the exact sequential-blend closed form (product of (1−gate)),
NOT a gate sum (sum silently scales with chunk size).

**Named risk: write-path gradient starvation** (project history: write-grad collapse). Init gate
magnitude ~ score(≈1/N) × surprise × bias-mid — alive but faint. From step 0, in the REAL
bf16+compile path, log gate magnitudes + grad norms per write-path param group; ~0 write-path grads
after ~100 steps = launch-abort signal. First knob: derived routing temperature (raise effective-k
at init), never an ad-hoc multiplier.

## 8. Telemetry (IMPLEMENTED 2026-06-12 — the full panel, logged to the gate jsonl)

Per layer unless noted; all `graph_v9_*` keys. Failure mode → metric:

**Routing collapse axes**
- `L*_route_tok_eff_k` — per-token softmax perplexity. ≈N = uniform routing (the v8
  disease); ≈1 = winner-take-all. Init calibrated to effective_k (≈8).
- `L*_route_usage_eff_frac` — effective fraction of nodes used over the window.
  → 0 = node collapse (few hot nodes get everything, the v8c5 finding).
- `L*_route_token_overlap` — mean pairwise cos of token score vectors. → 1.0 =
  every token routes identically = ADDRESSING COLLAPSE (the SHUF=REAL precursor).
- `key_nn_cos_L*` — nearest-neighbor key cosine. → 1.0 = keys merging.

**What the Householder factors are DOING**
- `L*_beta_eff_mean/_top`, `L*_beta_frac_{skip,erase,reflect}` — effective-β
  distribution (0 skip / 1 erase / 2 reflect) over a 32-token sample; _top =
  the strongest factor actually applied per token.
- `L*_apply_norm_ratio` — out/in RMS through the chain. 1.0 = rotation-dominated,
  <1 = erase-dominated.
- `L*_apply_rotation_cos` — cos(in, out). 1.0 = identity chain = DEAD memory.
- `L*_code_overlap_in/out` — token-code pairwise cos before/after the chain
  (homogenization watch: out ≫ in = the chain is blurring tokens).
- `dir_collapse_cos_L*` — factor-direction diversity (operator collapse).

**Coactivation / absorption (the write)**
- `coact_mass/entropy_L*`, `L*_coact_asymmetry` (0 = the lag carries no
  precedence signal → absorption direction arbitrary).
- `L*_absorb_gate_mean/max`, `L*_grammar_dev` (plasticity MLP's |dev from 1| —
  is the grammar learning anything), `L*_absorb_flux` (strength moved per
  boundary), `L*_absorb_refusal` (room-fraction refusals).
- `strength_budget_L*` (conservation invariant: constant per doc in arm C),
  `slot_occupancy_L*`, `strength_disp_L*` + `dirs_rot_L*` (displacement of the
  doc's state from the trained base — zero = the write did nothing).

**THE gate-deciding number**
- `state_sep_cos_L*` — pairwise cos of batchmates' end-of-doc strengths. 1.0 =
  every doc produces the same state = SHUF≡REAL structurally. Measured 1.0000 at
  init (template-dominated coact); training must drive it down.
- `state_doc_var_L*` — the doc-specific state component's magnitude.

**Read side** (`graph_v9_read_*`, harvested from the decoder forward)
- `p*_eff_k_L*` — query routing sharpness through the same keys.
- `p*_read_rotation_cos` — 1.0 = memory not steering queries (dead read).
- `p*_inj_ratio` — injection RMS / residual-stream RMS (the gate=1.0 loudness watch).
- `reader_gate_p*` — learned gate values.

## 9. Open questions

1. **The write-philosophy arms** (the big one) — three arms, decided by the probe, not argument.
   **Arm C (PRIMARY, pivot 2026-06-12): vocabulary-by-absorption.** NO deposits anywhere.
   Layer 0 = static trained atoms; apply input = PROJECTED SEED (unit-RMS projection of the
   Llama hiddens, per token — mirrors the read: queries must project into code space and
   travel up the pyramid as themselves; decision 2026-06-12). Projected content rides the
   STREAM but never lands in a NODE — the memory stays selection-pure. Writable layers initialize
   from a TRAINED random base vocabulary (slow params — random init then gradient-trained,
   NOT per-input noise, else the read could never learn to interpret relocated content);
   the ONLY write is within-layer conserving absorption. Binding IS the relocation pattern
   (Marcus's node absorbs the co-firing occupation node's factors — the displacement is the
   stored fact). Total strength per layer is CONSTANT per document (the strict invariant).
   Words store PROGRAMS (factor lists move as factor lists), never blended points — the
   composition-over-averaging principle applied to the write itself. Cost knowingly paid:
   writable layers are not identity-at-init (content must exist to relocate — the atoms'
   trade); Llama-identity at step 0 still holds via the reader's zero-init out-projection.
   No cross-layer state flow needed (within-layer absorption bootstraps from the base);
   no lift maps needed (independent per-layer bases decorrelate the layers).
   **Arm B:** static atoms + constant seed; layers >= 1 take DEPOSITS of the operated codes
   (composed-code deposits — words store points). **Arm A (control):** seed projection +
   deposits everywhere (content channel at the leaves).
   For arm C the probe runs absorb ON (absorption IS its write); arms A/B probe absorb OFF.
2. **Read/write phrasing asymmetry** — statement vs question routing through slow keys alone.
   Likeliest failure point; watch routing overlap REAL vs SHUF.
3. **Read injection form** — SETTLED at implementation (2026-06-12): per-layer hooks
   (GraphV9FlowReader), not prepend. Apply-to-query structurally REQUIRES the query as the read
   input — prepend hands the decoder static tokens and degenerates to a value-linear read (root
   cause 3 again). v8c set the harness precedent (M=0 + hooks; the REAL/SHUF/OFF differential
   stays read-fair since the read path is identical across the three).
4. **Consolidation cadence** — SETTLED: once per chunk boundary (§4b). Remaining: absorb during
   query segment? (default no).
5. **Sizing** — layer count, N_ℓ, S_ℓ, key dim, code dim, 26K-float fit.
6. Minor: apply-before-write vs write-before-apply; exact novelty/match definitions; position-aware
   slot placement (deferred).

---

## 10. De-risking ladder

1. **Flat probe, two arms** (arm A: one writable layer; arm B: static alphabet + one writable
   layer). emat_bio gate, 600 steps, BS=8, matched float budget. Entry ticket: SHUF−REAL > 0.
   Read SHUF−REAL only, never REAL alone. Compare floor (1.79) and Beacon (+2.07 at 102M).
2. **Algebra unit test** (cheap, before/alongside): stream of swaps → track the permutation
   (cups-and-ball / S₅). Verifies the operator chain end-to-end at init; the one directly
   falsifiable "beyond attention" demonstration.
3. If the probe binds → add layer 2 (flow-through + absorption). The pyramid must then beat the
   flat operator memory AT MATCHED FLOATS on bundle-reuse structure (emat_bio with multiple
   queries per entity; continuation/gist later) — that, not EMAT alone, tests the vocabulary
   thesis as distinct from "Householder writes work".
4. Then layers 3–4, per-layer reads, full telemetry.

Never debug hierarchy and binding at once.

---

## 11. Grounding (2024–2026)

- **DeltaNet / Gated DeltaNet / DeltaProduct / RWKV-7** — the primitive + WY parallel-training form.
- **Grazzi et al. 2024** — β ∈ [0,2] (negative eigenvalues) unlocks state tracking; do not clip.
- **Titans / Atlas / Miras** — surprise-gated write strength (we take NLL surprise free from the
  decoder); memory as online optimization.
- **Modern Hopfield** — capacity ⇔ key separation (what good vocabulary buys the storage).
- **Beacon (our 2026-06-07 result)** — per-layer attention read is the binding baseline's edge;
  adopted via per-layer reads.
- **Householder 1958; orthogonal RNNs; Householder flows** — the operation's pedigree.
- **STDP / Hebbian assemblies** — lagged coactivation = timing-asymmetric Hebb; absorption =
  assembly consolidation.
- **BPE / LZ dictionary coding** — the economics of the pyramid (combinatorial reuse; recurrence
  earns a word).

## 12. Success criteria

- Probe: SHUF−REAL > 0 at ≪ Beacon's params, present early in training (the operator algebra works
  from init; only addressing must be learned).
- Pyramid: beats flat operator memory at matched floats on bundle-reuse tasks.
- Algebra test: tracks S₅-style state where attention baselines provably degrade with length.
