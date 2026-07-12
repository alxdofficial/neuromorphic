# slotgraph — THE graph memory (design memo)

*The canonical slotgraph. Supersedes the exploratory line (slotgraph / slotgraph2 / slotgraph3 /
slotgraph4 — all removed 2026-07-11; their lessons are folded into §10 below and `docs/design/graph_thesis.md`).
One arm, converged after the full edge-substrate design thread (2026-07-10), then built + stabilized
2026-07-11 (§12). Read with `docs/design/graph_thesis.md` (why free structure collapses) and `docs/design/OBJECTIVES.md`
(why binding is an objective problem). References to "slotgraph3/4" below cite where a specific fix or
cost was first identified in that lineage — provenance, not live docs.*

## 0. TL;DR — what slotgraph IS

**N=96 node slots, NO explicit edge tokens, and a persistent plastic state for every ordered pair: a unit
relation vector `R[i,j]` plus scalar confidence `C[i,j]`. Their product lives on the attention value path and
is updated from how write-layer attention evolves across depth.**

- **A LoRA'd LM does the write AND the read.** The write is a single forward of a frozen SmolLM2
  (write-LoRA) over `[text ; 96 nodes]`; we HARVEST its per-layer attention rather than building custom
  attention layers (the project thesis: the LM already forms the graph — harvest it). The read is the same
  LM (read-LoRA) decoding over the prepended memory.
  > **Impl status (2026-07):** the *design intent* is one shared base + two adapters (a VRAM win). The
  > **current implementation keeps two frozen copies** — the encoder builds its own base
  > (`encoder.py` `load_frozen_llama`), the decoder its own (`model.py`) — i.e. ICAE **option-A**, which
  > avoids write/read adapter collision at the cost of a second frozen 135M. Collapsing to a genuine single
  > shared base (toggling write-LoRA vs read-LoRA on one instance) is an **unrealized optimization**, not
  > done yet (§2, §6).
- **Nodes only in the forward.** The write processes 96 node tokens (+ text as keys/values). There are NO
  N·k edge tokens — that was the dominant cost of slotgraph4 (288 edge tokens outnumbering the text), and
  it is gone. Edges are not tokens; they are a state that *modifies what nodes transmit*.
- **Edges live on the value path.** Query/receiver `i` receives sender `j`'s value plus the effective edge
  `C[i,j]R[i,j]`: `out_i = Σ_j a_ij·(V(x_j) + U·C[i,j]R[i,j])`. `R` carries relation type; `C∈[0,1]`
  carries existence/strength. This is inspired by Relational Attention, but narrower: SlotGraph adds only
  the edge value residual rather than conditioning all Q/K/V projections.
- **The pair state is persistent and plastic within an episode.** `R,C` recur across the 8 internal windows,
  but reset for each `finalize_memory` call. The semantic proposal comes from consecutive-layer attention-
  weighted endpoint features. A separate learned evidence head, calibrated by absolute pair attention,
  proposes confidence. Separate data-dependent gates commit both at each window boundary (§4).
- **Self-consistent read/write of the edge.** The same attention op reads `C[i,j]R[i,j]` and supplies evidence
  for its next update. No
  separate edge-materialization pass.

**No Watts-Strogatz topology.** Dense attention over the 96 nodes IS the topology; the graph is
emergent-but-persistent. (WS existed to dodge N²; we pay N² for the edge state anyway, so the scaffold
buys nothing — §3.)

**What slotgraph does NOT claim to fix:** binding/membership. A richer, more legible, more dynamical edge
feature makes the graph more *usable*; it does not make the model *use* it. That remains an OBJECTIVE
problem (§7), and it is unmeasured in the current (behavioral-KL) regime — the single most important open
gate (§9).

## 1. State

- **Nodes** `X : [N, d]`, N=96, d=576 (LM dim). Invented free latents; per-forward init noise breaks slot
  symmetry (Slot Attention). Frozen orthonormal id buffers give reusable entity identity (EntNet).
- **Relation semantics** `R : [N, N, d_e]`, d_e=**32**. Unit-norm when populated and initialized to zero.
- **Confidence/strength** `C : [N, N, 1]`, bounded to `[0,1]` and initialized to zero. The effective edge is
  `C·R`, so zero confidence is a real no-edge state and `||C·R||=C`.
- Both are directed: index `(i,j)` means sender `j →` receiver/query `i`, matching Relational Attention's
  convention. They recur only within one episode. Footprint is N²·(d_e+1) ≈ **304K floats/example** —
  internal STATE (modifies values), NOT read budget (never prepended). This sits on the
  "state floats" fairness axis where it is modest (Titans carries ~11M there). **Keep d_e small — do not
  let it creep to 64** (that blows past the prepend-equivalent budget and doubles BPTT memory).

## 2. Write — ONE LoRA'd LM forward, edges harvested from its attention

**We do NOT build custom attention layers. We run the encoder's frozen LM copy (with a write-side LoRA) over
`[text ; 96 nodes]` and HARVEST its per-layer attention.** This is the project's thesis made literal:
the LM already forms the in-context graph — harvest it, don't reinvent it. The write is one LM forward per
window; trainable write-side params are the write-LoRA plus the relation/confidence machinery.

The relational read/write of the edge decomposes into the LM's own output PLUS a cheap additive residual —
so it needs NO custom kernel. For query node i, per layer l:
```
out_i = Σ_j a_ij·(v_j + U·C[i,j]R[i,j])
      = Σ_j a_ij·v_j + U·(Σ_j a_ij·C[i,j]R[i,j])
        └ native output ┘   └ confidence-scaled relation residual ┘
```
- **Harvest, then correct.** Run the LoRA'd LM with `output_attentions` (cheap here — the write window is
  256 text + 96 nodes ≈ **352 tokens**, NOT the 2048-ctx forward that makes eager attention slow in H2O;
  <~1GB held-for-grad at B=4). Take the per-layer node-block scores `a_ij`, add the residual
  `U·(Σ_j a_ij·C[i,j]R[i,j])` to the node hiddens via a per-layer hook
  slotgraph3 already used). The edge aggregation is N²·d_e < N²·d; `U` hits only N vectors after.
- **The harvested `a_ij` do triple duty from ONE forward:** (1) contextualize the nodes, (2) form the
  emergent graph, (3) feed both the edge residual and depth-wise relation trace (§3) — all differentiable,
  free from the same forward. Gradient reaches the node states through the (frozen-but-differentiable +
  LoRA) q/k/v projections.
- **Why LoRA, not frozen-harvest-only OR custom layers.** Frozen-only can't *adapt* the graph toward
  memory-optimality (it bets the LM's linguistic next-token graph is right for recall); custom LM-init
  layers *drift away* from the LM's graph as they train and cost extra params. The **write-LoRA is the
  resolution**: authentic-LM-graph starting point (frozen base) + trainable adaptation (LoRA delta), no
  custom layers, same frozen-LM-adaptation pattern every other baseline uses. NOTE the honest consequence:
  with a LoRA + the edge residual both bending the forward, there is no longer a "pristine LM graph" being
  read — `a_ij` is the **LoRA-adapted, edge-conditioned** graph. That is what we want (the graph conditions
  on accumulated structure), but the "harvest the authentic graph" framing is retired: it is an adapted graph.
- **TWO LoRAs (write + read).** The write forward's job ("form a good graph over nodes+text") and the read
  forward's job ("answer from memory", §6) are different objectives; a shared adapter couples them. Use a
  separate write and read adapters so the write-graph adaptation is not pulled around by the read loss.
  Mixed capacity matching currently sets the write rank to 84; the decoder uses the cohort read adapter.
  The current option-A implementation keeps separate frozen encoder/decoder copies so their adapters cannot
  collide. Sharing one frozen base remains a possible VRAM optimization, not current behavior.
- `U` is **zero-init** (ReZero), so window 0 starts as a plain LM write. The nonzero confidence/semantic
  commit and direct graph read give `U, φ, W` a gradient path. This encoder does ordinary attention over
  text/nodes; it does **not** currently implement Slot Attention's token-over-slot competition.
- **Absorption stays controlled.** The write-LoRA *can* reshape `a_ij`, and the edge state also shapes node
  values — but the edge injection is an **additive value residual**, not a score bias, so it is not
  absorbable into the scores (§6). Keep it there.

## 3. Pair observations — relation semantics plus confidence evidence

The implementation averages the node-block attention over heads. It keeps two versions:
```
a_raw^l[i,j] = mean_h attention^l[h,i,j]                 # absolute mass in the full [text;nodes] softmax
a_rel^l[i,j] = a_raw^l[i,j] / Σ_k a_raw^l[i,k]           # topology conditional on attending to a node
e^l[i,j]     = a_rel^l[i,j] · (φ_i(y_i^l) + φ_j(y_j^l)) # d_e-dimensional relation observation
```
`y_i^l` is the self-attention branch output at node `i`, not the individual value `v_j` and not the full
post-MLP residual state. `a_rel` stabilizes value aggregation; `a_raw` preserves the absolute evidence that
renormalization would otherwise discard. Heads are currently averaged and there is no multi-timescale
filterbank. Those are possible extensions, not implemented properties.

**The inter-layer operator (do NOT use bare subtraction).** Given consecutive-layer edge features `e^l`,
`e^{l+1}`, combine them with a **learnable, content-in-the-inputs operator** — the survey verdict (§8):
```
feat(e^l, e^{l+1}) = W · [ (e^{l+1} − e^l)  ‖  (e^l ⊙ e^{l+1})  ‖  e^{l+1} ]   → d_e
```
- **difference** = the dynamical/derivative signal (how the edge changed across the layer).
- **element-wise product** = the co-activation/Hebbian-correlation term (the DIAGONAL of the true outer
  product `e^l ⊗ e^{l+1}`, at d_e cost not d_e²) — the term bare difference is missing.
- **raw** = lets a linear map recover a plain read.
- `W` is a **FIXED learned matrix** (content lives in the INPUTS, not the operator — see §6, the absorption
  trap). It learns *once, globally* how to weigh diff vs product vs raw; it can become pure-difference
  (TransE) or pure-product (DistMult) per-dim if the data wants.
- `W` has **no bias**, so every dense pair does not receive the same content-free relation proposal.

A separate evidence head reads the same concatenated feature plus `[a_raw^l, a_raw^{l+1}, node_mass^l,
node_mass^{l+1}]`. Its sigmoid is multiplied by `N·sqrt(a_raw^l a_raw^{l+1})`, clipped at one. Thus uniform
node attention gives evidence equal to the row's absolute node-attention mass; text-dominated rows stay weak.
The pairwise observations are averaged over harvested layer pairs to produce semantic proposal `S[i,j]` and
confidence observation `O[i,j]∈[0,1]`.

Bare subtraction is the *TransE* of this design: the simplest operator, and the exact thing the entire
relation-learning literature was built to improve on (it can't represent 1-to-many / symmetric relations;
the product captures the "same/co-active" case where difference vanishes). §8.

## 4. Persistence — separate semantic direction from confidence/strength

Per-layer observations remain scratch state; persistent `R,C` move only at window boundaries, keeping
recurrence depth equal to the number of windows. The commits are:
```
ΔR      = S[i,j] − K R[i,j]                              # K is identity-initialized, then learned
R[i,j]  = unit_norm( α_r⊙R[i,j] + β_r⊙O[i,j]⊙ΔR )        # semantic direction

retained = α_c·C[i,j]
C[i,j]   = retained + (1−retained)·β_c·O[i,j]             # bounded confidence in [0,1]
```
All four gates are factored per endpoint and data-dependent. Confidence evidence additionally opens writing
and replacement directly (`O` enters the confidence gate logits). With no evidence, confidence decays by
`α_c`; repeated evidence fills its remaining capacity. Relation magnitude cannot impersonate confidence:
populated `R` has unit norm and every use site consumes `C·R`. This makes weak, absent, and strong edges
representable while retaining a vector-valued relation type.

The semantic update is **delta-style**, not a key-addressed DeltaNet rule and not guaranteed to overwrite at
an exact fixed point. Identity-initializing `K` gives it an interpretable `S−R` starting point; overwrite,
contradiction, and lag behavior remain empirical canaries.

## 5. Initialization

- **Scratch `S`** → 0 at the start of every window (trivial).
- **Persistent `R,C`** → zero at window 0 of each episode. A learned pair-specific initial relation would
  assert structure before exchangeable slots have content. Window 1 observes the native write; later windows
  receive confidence-scaled feedback.
- **Confidence observation** starts low (`sigmoid(-2)≈0.12`) and is further scaled by absolute attention
  evidence. It is not zero, so relation/confidence heads receive gradient through the direct graph read.
- **`U` (the value-path injection)** → zero-init (ReZero). Semantic and confidence write gates initialize
  open; confidence retention initializes near 0.95. `K` starts as identity.

## 6. Read — the shared LM (read-LoRA), prepend + bidirectional

The read uses the **decoder's frozen LM copy with its read-LoRA** (§2) over prepended memory tokens. `R,C`
are internal graph state, not N² payload tokens:
- **Node-centric tokens** (N): each node receives `Σ_j A_ij·(X_j + up(C[i,j]R[i,j]))`. Routing depends on
  confidence-scaled relation keys and neighbor content. A second hop reuses the first-hop node states.
- **Top-k explicit pointer tokens** (optional): ranking is `learned_relation_salience(R)+log(C)`, so semantic
  importance cannot select a nonexistent edge and confidence alone need not define relation type. Pointers
  encode source `j` before receiver `i`, matching message direction. Content is resolved by the decoder
  attending back to the endpoints. This is exactly why the read is **prepend + bidirectional** (Set-LLM,
  `uniform_mem_pos`), NOT per-layer KV: the relational read needs intra-memory attention, which keys-only
  KV cannot provide.

`force_no_edges` sets `C=0`; the bias-free edge lift then produces exactly zero edge messages while retaining
the content-only `X_j` message path. This is the node-only graph-read control. `U=0` is a different ablation:
it removes edge feedback during writing but leaves the final edge read active.

## 7. The load-bearing part is STILL the objective

Every result in the design thread converges on: **legibility ≠ leverage.** Making the graph richer / more
dynamical / more bio-plausible makes it more USABLE; it does not make the model USE it. Under a
loss-neutral objective the edge state decays to zero exactly like slotgraph4's did (`docs/design/graph_thesis.md`;
NRI; Williams-2018). So slotgraph ships with the objective ladder (`docs/design/OBJECTIVES.md`):
- **behavioral-KL** (`E[KL] = I(context; answer)`) — the shared backbone. NOTE: this is a *different kind*
  of objective from the MAE-CE regime that produced every prior collapse result, so those priors are
  **void as evidence here** (§9). behavioral-KL directly charges for the memory carrying answer-relevant
  info — the first regime where a legible edge could plausibly convert to leverage.
- **exclusive read channel** (decoder reads only memory) + **provenance-InfoNCE** (address reward) +
  **bypass-gap** (Larimar: memory must beat the no-memory floor).

The learnable operator `W` (§3) is double-edged: it CAN represent the right relation, and it also has a
trivial optimum — zero the diff/product columns, keep only raw — if structure stays loss-neutral. That is
not a bug; it is a **canary** (§ below).

## 8. Operator survey (why learnable diff+product, not subtraction)

Three literatures ran essentially this experiment and converge:
- **NLI sentence-pairs** (InferSent, Conneau 2017; Sentence-BERT, Reimers 2019): ablations find
  `[u ; v ; |u−v| ; u⊙v]` best; difference and product are **complementary, not redundant** (difference =
  "how far", product = "do dims co-vary"); difference-only is insufficient.
- **KG embeddings** (TransE / TransH-2014 / DistMult / ComplEx): subtraction IS TransE; the whole field
  exists because difference-only **can't represent 1-to-many / many-to-1 / symmetric** relations — the
  "DistMult symmetry ceiling" this project already hit. Bilinear/product forms were invented to fix it.
- **Feature interaction** (xDeepFM-2018): explicit vector-wise **products** capture combinatorial
  interactions additive/difference forms miss.

Element-wise product is the diagonal of the lag-one outer product `e^l ⊗ e^{l+1}` at d_e rather than d_e²
cost. This is a depth-wise autocorrelation feature. It is Hebbian/STDP-inspired, but not literal STDP: layers
are not biological time, and the implementation does not compare separate pre-before-post and post-before-pre
events.

## 9. The decisive experiment + canaries

slotgraph is now BUILT (§12) but UNMEASURED — and per the "old objective voided the priors" argument,
**there is NO trustworthy binding baseline at all yet.** The build is done; the decisive experiment is not.
The highest-value move now is to get the first trustworthy number:
1. **Baseline:** train the arm with `force_no_edges=True` as the content-only node control
   to convergence and read `SHUF−REAL` + node/edge effective-rank. **Run this with the MEMBERSHIP+ADDRESSING
   objectives ACTIVE**
   (SHUF-contrastive Rung 4 + provenance-InfoNCE Rung 2), **not behavioral-KL alone (P10).** behavioral-KL
   charges *use* (`I(context;answer)`); the wall that killed babi was *membership/input-dependence*, which
   KL does not directly charge for — so a KL-only null would FALSE-NEGATIVE. Only a null under the objective
   that actually charges for membership licenses the "structure won't save it" verdict.
2. **Then** A/B the edge machinery — and the bar is to **BEAT, not tie (P9):** the edge arm must beat both
   (a) content-only nodes with the identical prepend+bidir read, and (b) a flat ICAE-style memory
   given the identical read. The readout-not-substrate lesson (`project_readout_not_substrate`) is that a
   rich query-conditioned read manufactures an advantage a flat memory gets too; beating the no-memory
   floor is the WEAK bar. Report SHUF−REAL + task metric for all three in one table.
3. **Gate #0 — path-ablation** (`project_graph_edge_state_bypass`): zero-confidence vs zero-nodes. Require
   node-only to beat the no-memory floor AND edge-only to be strictly WORSE than REAL. If 100% of the
   memory rides the free `C·R` bank and nodes are vestigial, the "graph" is a flat bank in disguise — the
   exact prior failure. This is a HARD gate before trusting any edge result.

### Canaries — LEADING indicators first (every prior collapse was visible here FIRST)
The three signals below caught every collapse in the slotgraph/furlgraph line *before* the outcome metrics
moved. §9 must ship them, not just the lagging outcomes:
1. **Input-dependence of `R`, `C`, `C·R`, and `a_ij` across examples** (per-pair participation ratios and the
   harvested `a_ij`, on BINDING tasks specifically — babi, not aggregate). THE leading indicator — the
   `routing_diversity` analog; SHUF−REAL *lags* it. Without this, a SHUF−REAL≈0 result is undiagnosable
   ("did semantics, confidence, or downstream use fail?"). If `S,O`/`a_ij` are input-invariant on babi,
   the edge is dead regardless of objective (`project_graph_write_collapse`: the parser emitted one
   input-invariant edge ×128, edge-cos
   ≈0.999, BEFORE the read).
2. **Gradient-norm per edge-module** — include `U, W, φ, confidence-observer, R/C gates` versus read-LoRA at
   every val step. `slotgraph_gradflow.py` now reports these groups separately. The prior 1000× write
   starvation (`project_graph_internals_diag`: q_dst 1.4e-4 vs decoder-LoRA 1.8e-1) was visible ONLY at the
   gradient level. Gate: edge-machinery grad/param >~100× below the read-LoRA in the first few hundred
   steps = fail (the P8 deadlock, live).
3. **Per-window streaming collapse trace** — emit effective-edge rank, `C` mean/std/active fraction, and
   fixed-pair relation drift. Over-smoothing COMPOUNDS across the 8-window
   recurrence — furlgraph's `node_wcos 0.34→0.94` was the #1 documented risk. (infra: `collect_layer_metrics`.)

### Canaries — lagging outcomes + design-validators (all cheap, @no_grad, GPU-scalar)
- **learned-`W` block weights** — did the diff/product columns survive, or did `W` collapse to raw-only?
- **edge-state effective rank** (ID-subtracted, content-only — NOT id-spoofed).
- **Three path ablations** — `U=0` (no write feedback), `C=0` (no edge read), and zero nodes (edge-only).
- **overwrite canary** — write fact → overwrite same pair → test old-fact suppression (validates the
  error-correcting delta of §4; a pure EMA would fail this).
- **direction canary** — `R[i,j]` vs `R[j,i]` distinguishability (the diagonal-product symmetry risk).
- **retention-vs-lag** — bio query-window sweep 0..7; uniform idle-decay shows as monotonic degradation with
  lag (validates the per-edge gated retention of §4).
- **week-0 persistence/BPTT** — bind at window 0, query at window 7, loss only at end; confirm ∂L reaches
  the window-0 write BEFORE trusting 8-window retention.
- **SHUF−REAL and OFF−REAL** per-task — does memory bind / contribute.
- **node-order shuffle invariance** — validates the set-invariance (no-PE) design (within-example
  order-invariance — DISTINCT from the cross-batch SHUF binding control).
- **write telemetry** — `‖S‖`, `O`, `C`, effective `‖C·R‖`, and both gate families per window.

## 10. Considered and rejected (the design thread, recorded)

- **Per-node feature instead of per-edge** — cheaper (O(N·d_e)) but can't carry pairwise relation TYPE
  (modulates j's value the same for every reader); and a per-node signal is largely recoverable from the
  residual stream → redundant-with-depth. Per-EDGE is provably non-redundant (the residual stream is
  per-node, can't store the disaggregated pairwise term). Rejected for multi-hop; kept as a fallback.
- **Pooling `Σ_j a_ij v_j` then differencing across layers** — this IS the attention output, recoverable
  from the residual increment → the redundancy-with-depth trap. Must NOT pool (keep the j index).
- **Watts-Strogatz fixed topology** — dropped: dense attention over 96 nodes IS the topology; WS only
  existed to dodge N², which the dense edge state pays anyway.
- **Explicit edge tokens in the write stack (slotgraph4)** — the dominant cost (288 tokens); replaced by
  value-path edges (edges leave the stack entirely).
- **Feeding the emergent attention graph back as an INPUT feature / attention bias** — MIXED→negative in
  the literature: (a) attention-logit bias is ABSORBABLE (Routing-Absorption 2603.02227 = the
  routing_diversity≈0.02 collapse; a scalar post-softmax gate is algebraically the same log-bias, no
  escape); (b) higher-order/rollout structure fed back as a feature is redundant-with-depth (Reuse
  Transformers 2021; rollout = the composition the stack already computes) — feeding attention back helps
  only when it adds DIVERSITY (RealFormer/Re-attention) or GATES compute (FastV). So the edge state must ride
  the VALUE path (non-absorbable), not the bias.
- **Content-dependent OPERATOR** (`W_ij = g(x_i,x_j)`) — the absorption trap in a new outfit: an operator
  generated from node content makes the edge reconstructable from the nodes → collapse. Keep `W` fixed;
  content lives in the inputs only.
- **Outer-product Hebbian `e^l ⊗ e^{l+1}`** — the true correlation form but d_e² per edge (×N² = millions);
  replaced by the element-wise product (its diagonal) at d_e.
- **Node-order-shuffle as a FEEDBACK feature** — ~zero by construction (set-invariant architecture); kept
  only as a canary.
- **Noise-ablation attribution** — off-manifold (Hase-Bansal 2021; Zhang-Nanda 2023 prefer mean/resample);
  if we ever do explicit interventions, use a mean/null baseline, not noise.
- **STDP/bio-plasticity as the memory mechanism, unqualified** — the project's biomem (LIF) went inert;
  bio-plausibility is not a performance argument. slotgraph keeps the *eligibility-trace* math (it is just
  a differentiable temporal filter) but makes no bio claim and gates everything on the objective.

## 11. Prior art
The individual ingredients are established:
- **Relational Attention** (Diao & Loynd, ICLR 2023) has directed vector edges, full Q/K/V conditioning, and
  per-layer edge updates from both endpoints and the reverse edge. It permits edge-presence flags inside the
  vector, but does not separate a recurrent confidence state or harvest relations from attention evolution.
- **EGT** has evolving edge channels and scalar edge gates; **NRI** infers categorical edge type/existence
  probabilities; temporal GNNs add recurrent node memory around timestamped edges.
- **Differentiable plasticity / Backpropamine / e-prop** motivate learned local traces and modulated writes;
  **Gated DeltaNet** supplies data-dependent decay plus residual correction; **EntNet** motivates normalized
  semantic memory; **Slot Attention** motivates competitive binding, which this implementation lacks.

SlotGraph's claim is therefore a **novel synthesis candidate**, not a first-principles invention: relation
semantics harvested from consecutive LM layers, explicit recurrent confidence, cross-window pair persistence,
and confidence-scaled value feedback in a compressed LM memory. Do not claim first-of-kind without a broader
systematic review. The diff/product operator itself is project-specific and should be presented as a learned
heuristic, not an established STDP rule.

## 12. Status
**BUILT; R/C REVISION FORWARD/BACKWARD VERIFIED (2026-07-11)** — `src/memory/models/slotgraph/`, the canonical
`slotgraph_baseline`.
96 nodes, no edge tokens; separate frozen write/read LM copies with separate LoRAs (mixed training overrides
write rank to 84 for capacity matching); persistent unit relation `R` plus scalar confidence `C`; confidence-aware value-path
feedback; learnable inter-layer diff/product semantics; and separate data-dependent commits (§4).

The earlier single-vector build stabilized node-block normalization, bounded value injection, factored pair
features, and memory use. The R/C revision additionally preserves absolute attention for confidence, removes
the semantic-operator bias, identity-initializes `K`, makes the edge lift bias-free, and defaults to full
BPTT. Its bounded recurrence and inactive-row behavior have focused CPU tests. A real bf16 encoder
forward/backward at the active geometry (`N=96`, `d_e=32`, rank-84 write LoRA, 2048 tokens/eight windows,
full BPTT, batch 1) was finite, reached every semantic/confidence module, and peaked at 4.15 GiB allocated on
an RTX 4090. End-to-end task-loss and batch-size sweeps remain required before convergence runs.

**Checkpoint boundary:** pre-R/C SlotGraph checkpoints are not valid resumptions for this architecture.
Non-strict loading would silently leave the confidence heads/gates untrained and cannot recover the old
single-vector normalization semantics. Start R/C experiments from a fresh initialization.

**Still open (the decisive experiment, §9):** slotgraph has NOT yet been trained to convergence — the
binding question (`SHUF−REAL`) is unmeasured. Run it under the MEMBERSHIP+ADDRESSING objectives (not
behavioral-KL alone — KL charges *use*, the wall is *membership*), armed with the three LEADING canaries
(input-dependence, gradient-norm, per-window collapse trace). The exploratory slotgraph 1–4 code + design
docs were removed 2026-07-11; this is THE slotgraph.
