# slotgraph — THE graph memory (design memo)

*The canonical slotgraph. Supersedes the exploratory line (slotgraph / slotgraph2 / slotgraph3 /
slotgraph4 — all retained on disk as history, none current). One arm, converged after the full
edge-substrate design thread (2026-07-10). Read with `docs/graph_thesis.md` (why free structure
collapses), `docs/OBJECTIVES.md` (why binding is an objective problem), and `docs/slotgraph4_design.md`
(the immediate predecessor whose write/read machinery this inherits).*

## 0. TL;DR — what slotgraph IS

**N=96 node slots, NO explicit edge tokens, and a persistent plastic per-edge state that lives INSIDE the
attention (on the value path), accumulated from how the write layers' attention evolves across depth.**

- **ONE LoRA'd LM does the write AND the read.** The write is a single forward of the shared frozen LM
  (write-LoRA) over `[text ; 96 nodes]`; we HARVEST its per-layer attention rather than building custom
  attention layers (the project thesis: the LM already forms the graph — harvest it). The read is the same
  shared frozen LM (read-LoRA) decoding over the prepended memory. Two rank-16 LoRAs, one shared base — so
  slotgraph4's two-frozen-copies (featurizer + decoder) collapse to one base + two adapters (§2, §6).
- **Nodes only in the forward.** The write processes 96 node tokens (+ text as keys/values). There are NO
  N·k edge tokens — that was the dominant cost of slotgraph4 (288 edge tokens outnumbering the text), and
  it is gone. Edges are not tokens; they are a state that *modifies what nodes transmit*.
- **Edges live on the value path.** When node i attends to node j, it receives j's value *modified by the
  edge state* `E[i,j]`: `out_i = Σ_j a_ij·(V(x_j) + U·E[i,j])` = the LM's own attention output + a cheap
  additive residual `U·(Σ_j a_ij·E[i,j])`. This is Relational Attention (Diao & Loynd, ICLR 2023) done as a
  harvest-plus-correction on the frozen LM, NOT an attention-logit bias (which is absorbable — §6).
- **The edge state is a persistent, plastic, temporal trace.** `E` is a streaming state (per-episode,
  carried across the 8 windows), NOT learned weights. Within a window it is accumulated from how the L
  write-layers' inter-node attention *changes across depth* (an STDP-flavored trace); at the window
  boundary ONE commit writes it into the single persistent `E` — and that commit is **error-correcting
  (delta, not raw EMA), content-gated per-edge (idle edges freeze), and magnitude-bounded (EntNet norm)**,
  re-adopting three fixes slotgraph4 / the write-audit already landed (§4). Two-timescale: fast node
  content `x` (per-window propose→commit) + slow relational memory `E`.
- **Self-consistent read/write of the edge.** The same attention op both *reads* `E[i,j]` (injects it into
  the value) and *writes* it (its score `a_ij` is the coincidence gate for `E`'s plasticity update). No
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
- **Edges** `E : [N, N, d_e]`, d_e=**32**. Persistent streaming state, **initialized to ZERO** at window 0
  (§5). Directed (`E[i,j] ≠ E[j,i]`). Dense (no fixed topology). Footprint = N²·d_e ≈ 96²·32 ≈ **295K
  floats/example** — internal STATE (modifies values), NOT read budget (never prepended). Sits on the
  "state floats" fairness axis where it is modest (Titans carries ~11M there). **Keep d_e small — do not
  let it creep to 64** (that blows past the prepend-equivalent budget and doubles BPTT memory).

## 2. Write — ONE LoRA'd LM forward, edges harvested from its attention

**We do NOT build custom attention layers. We run the shared frozen LM (with a write-side LoRA) over
`[text ; 96 nodes]` and HARVEST its per-layer attention.** This is the project's whole thesis made literal:
the LM already forms the in-context graph — harvest it, don't reinvent it. The write is one LM forward per
window; the only trainable write-side params are the write-LoRA + the small edge machinery (`U, φ, W, γ, η`).

The relational read/write of the edge decomposes into the LM's own output PLUS a cheap additive residual —
so it needs NO custom kernel. For query node i, per layer l:
```
out_i = Σ_j a_ij·(v_j + U·E[i,j])  =  Σ_j a_ij·v_j   +   U·(Σ_j a_ij·E[i,j])
                                      └ the LM's own attention output ┘   └ edge residual, added to the hidden ┘
```
- **Harvest, then correct.** Run the LoRA'd LM with `output_attentions` (cheap here — the write window is
  256 text + 96 nodes ≈ **352 tokens**, NOT the 2048-ctx forward that makes eager attention slow in H2O;
  <~1GB held-for-grad at B=4). Take the per-layer node-block scores `a_ij`, add the residual
  `U·(Σ_j a_ij·E[i,j])` to the node hiddens via a per-layer hook (the `_lm_suffix`/anchor-injection pattern
  slotgraph3 already used). The edge aggregation is N²·d_e < N²·d; `U` hits only N vectors after.
- **The harvested `a_ij` do triple duty from ONE forward:** (1) contextualize the nodes, (2) form the
  emergent graph, (3) feed BOTH the edge residual above AND the STDP trace (§3) — all differentiable, all
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
  separate rank-16 write-LoRA and read-LoRA (~0.9M each) so the write-graph adaptation is not pulled around
  by the read loss. Both wrap the SAME shared frozen base → slotgraph4's two-frozen-LM-copies problem
  (featurizer + decoder) collapses to **one shared base + two small adapters** (a real VRAM win).
- `U` is **zero-init** (ReZero) → with `E=0` at window 0 AND `U=0` at step 0, the write is a plain LM
  forward; the edge channel earns influence as `U, φ, W` train. Competitive assignment (softmax over the
  SLOT axis) is retained — the anti-duplication / reuse mechanism (§7), and the thing the diversity
  literature (RealFormer/Re-attention) says actually helps.
- **Absorption stays controlled.** The write-LoRA *can* reshape `a_ij`, and the edge state also shapes node
  values — but the edge injection is an **additive value residual**, not a score bias, so it is not
  absorbable into the scores (§6). Keep it there.

## 3. The edge feature — a small vector from consecutive layers' attention

The raw material per pair (i,j) is a **sequence of L scalars** `a_1[i,j] … a_L[i,j]` (plus heads). The
target is a **32-d vector**. Width does NOT come from the timing rule — it comes from three sources, which
factorize d_e:
1. **Heads × timescales (the backbone).** H attention heads give H independent score sequences; a small
   filterbank of learnable decay time-constants reads each at multiple temporal scales. `d_e ≈ n_heads ×
   n_timescales`. This is the cleanest width — genuinely distinct measurements, not correlated views.
2. **Multi-scale temporal kernels (STDP-native).** Per-channel decay constants integrate the score history
   over different spans (fast = late-depth structure, slow = early-depth). Learnable (e-prop eligibility
   traces; Backpropamine for the amplitudes). Alone, too thin over L≈4–6 layers — needs source 1.
3. **Content projection (relation TYPE).** `φ(x_i^l, x_j^l)` — a learned projection of the endpoint states,
   attention-gated. This is what lets an edge carry "is-capital-of" vs "is-located-in," not just strength.

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

Bare subtraction is the *TransE* of this design: the simplest operator, and the exact thing the entire
relation-learning literature was built to improve on (it can't represent 1-to-many / symmetric relations;
the product captures the "same/co-active" case where difference vanishes). §8.

## 4. Persistence — accumulate in scratch, commit once (ERROR-CORRECTING, per-edge gated, bounded)

Mirror the node write. One persistent `E`; the per-layer information is NOT persisted as L copies (that
defeats the inter-layer-change signal AND costs L× storage) and NOT updated inside the inner loop (that
makes E's recurrence depth #windows×L — the deep-BPTT + last-layer→first-layer wraparound problem). The
commit is NOT a plain EMA — that regresses three fixes slotgraph4 / the write-audit already landed. It is
an **error-correcting, content-gated, magnitude-bounded** write:
```
within window:   S[i,j]  ← Σ_l  feat(e_ij^l, e_ij^{l+1})            # scratch trace over the L layers; reset per window
at boundary:      α, β   = per-edge gates from the window's write hidden (content-dependent)   # NOT a global scalar
                  ΔE      = S[i,j] − read(E[i,j])                    # ERROR-correcting (retrieve, then write the residual)
                  E[i,j]  ← norm( α ⊙ E[i,j] + β ⊙ ΔE )             # gated commit + EntNet post-norm (bounds ‖E‖)
```
Three carry-forward fixes over a plain `E ← γE + ηS` EMA (all solved upstream — see §10, prior lessons):

- **Error-correcting delta, not raw additive (P3 / write-audit #5).** A plain EMA never reads `E` before
  writing, so a second fact on the same directed pair *superposes* onto the decayed old one → the
  averaging fixed point that gives `REAL==SHUF`. Subtracting the current read `S − read(E)` writes only the
  RESIDUAL, so a repeat/overwrite converges to the new value in place (DeltaNet / Gated-DeltaNet;
  `docs/OBJECTIVES.md` sidecar note). Drive it with the already-ported `mamba_delta_rule` primitive.
  (Titans' EMA is of a *surprise/error* term ‖Mk−v‖ — our raw-feature EMA was strictly weaker; this closes
  the gap.)
- **Content-gated per-edge retention, not a global decay (P4).** `α, β` are per-edge, content-dependent
  (computed from the window's write hidden), NOT global per-channel scalars. A global `γ` decays EVERY edge
  every window, so a fact written in window 1 and never re-mentioned is `γ⁷` by the query window — uniform
  forced-forgetting of exactly the retain-under-interference (T2) signal. Per-edge gating lets an IDLE edge
  (S≈0) FREEZE (α≈1) while an active one updates. This is slotgraph4's decoupled α/β, applied per-edge.
- **EntNet post-write norm bounds ‖E‖ (P12).** `α≈1` (which retention wants) makes an unbounded additive
  write grow without limit — additive saturation. Normalizing after the commit bounds magnitude (retention
  becomes directional, magnitude re-pinned). NOTE the consequence: under unit-norm E, a `‖E[i,j]‖`-based
  read selector is DEGENERATE (all ties) — so the read top-k uses a **learned salience head**, not ‖E‖ (§6).

Recurrence depth stays = #windows (wraparound dissolves — E only moves at boundaries); the scratch trace S
still captures all L layers' inter-layer dynamics; timing asymmetry in `feat` + directed `read(E[i,j])`
keeps E directed.

## 5. Initialization

- **Scratch `S`** → 0 at the start of every window (trivial).
- **Persistent `E`** → 0 at window 0 of each episode. NOT a learned per-pair init: at window 0 the slots are
  exchangeable (no content has landed), so a learned `E₀[i,j]` would assert a relationship between slots
  with no identity yet — meaningless, the same reason we dropped WS. Zero is the only per-pair value that
  respects the symmetry, and it gives a clean bootstrap (window-1 runs pure native attention → STDP
  observes it → window-2+ feels it). (Contrast Titans' learned `M₀`: fine there because it is a shared MLP
  not indexed by exchangeable pair-identity.)
- **`U` (the value-path injection)** → zero-init (ReZero) — so step-0 is a plain LM forward regardless of E.
- **AVOID the double-zero gradient deadlock (P8).** `E=0` AND `U=0` AND a throttled write gate is NOT clean
  ReZero — proper ReZero keeps the branch's *content* alive so the scalar still earns gradient, but here
  both the state (E) and the injection (U) are zero, so neither earns gradient from the read path at init.
  The ONLY thing that breaks the deadlock is the commit `E ← …β⊙ΔE`, which fires only if the write gate
  inits OPEN. So: init the **write gate β OPEN** (logit-space, effective ≈0.5–1.0; Jozefowicz +1 forget-init
  wisdom), init the **retention α near 1** (γ≈1), and apply ReZero to **at most ONE** of {U, β} — never both.
  Give `φ, W` an **independent gradient path** by routing E into the read-side graph-conv without a zero
  gain, so they don't wait on U's lift-off. Do **NOT** use a single learned scalar gain on the injection
  (scalars don't move under Adam — the frozen-scalar-temp trap, `feedback_frozen_scalar_temp`); put any
  gain in per-channel β or the q/k projections.

## 6. Read — the shared LM (read-LoRA), prepend + bidirectional

The read is the **same shared frozen LM with the read-LoRA** (§2's second adapter) decoding over the
prepended memory tokens — identical to how every other prepend arm reads. `E` is an OPERATOR (N×N weights),
not a payload — you do not prepend it. It shapes the node read, in the two roles slotgraph4's read already had:
- **Node-centric tokens** (N of them): each node token folds in its E-weighted neighbor blend
  `Σ_j E-driven aggregate of X_j` — one graph-conv readout with the learned relational adjacency. `E²` (a
  small persistent N×N op) gives 2-hop reachability for free.
- **Top-k explicit pointer tokens** (optional budget): the strongest edges by a **learned salience head**
  `Linear(d_e, 1)` (soft-gated), NOT by `‖E[i,j]‖` — under the EntNet post-write norm (§4) all edges have
  unit norm, so a ‖E‖ selector is degenerate (all ties). This is slotgraph4's settled answer. The selected
  edges become PURE pointers `tok_proj([content? ‖ id_i ‖ id_j ‖ type])` — content resolved by the decoder
  attending back to the endpoints. This is exactly why the read is **prepend + bidirectional** (Set-LLM,
  `uniform_mem_pos`), NOT per-layer KV: the relational read needs intra-memory attention, which keys-only
  KV cannot provide.

## 7. The load-bearing part is STILL the objective

Every result in the design thread converges on: **legibility ≠ leverage.** Making the graph richer / more
dynamical / more bio-plausible makes it more USABLE; it does not make the model USE it. Under a
loss-neutral objective the edge state decays to zero exactly like slotgraph4's did (`graph_thesis.md`;
NRI; Williams-2018). So slotgraph ships with the objective ladder (`docs/OBJECTIVES.md`):
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

Element-wise product = the diagonal of the Hebbian outer product `e^l ⊗ e^{l+1}` at d_e (not d_e²) cost —
so "learned diff + product" is the tractable, STDP-faithful form of the correlation we wanted.

## 9. The decisive experiment + canaries

slotgraph is unmeasured in the behavioral-KL regime — and per the "old objective voided the priors"
argument, **we currently have NO trustworthy binding baseline at all.** The highest-value move is therefore
NOT to build the full edge machinery first, but to get the first trustworthy number:
1. **Baseline:** train plain nodes (or slotgraph4 as-is) to convergence and read `SHUF−REAL` +
   node/edge effective-rank. **Run this baseline with the MEMBERSHIP+ADDRESSING objectives ACTIVE**
   (SHUF-contrastive Rung 4 + provenance-InfoNCE Rung 2), **not behavioral-KL alone (P10).** behavioral-KL
   charges *use* (`I(context;answer)`); the wall that killed babi was *membership/input-dependence*, which
   KL does not directly charge for — so a KL-only null would FALSE-NEGATIVE. Only a null under the objective
   that actually charges for membership licenses the "structure won't save it" verdict.
2. **Then** A/B the edge machinery — and the bar is to **BEAT, not tie (P9):** the edge arm must beat both
   (a) plain-nodes with the IDENTICAL prepend+bidir read (`U=0`), and (b) a flat ICAE/jun24-style memory
   given the identical read. The readout-not-substrate lesson (`project_readout_not_substrate`) is that a
   rich query-conditioned read manufactures an advantage a flat memory gets too; beating the no-memory
   floor is the WEAK bar. Report SHUF−REAL + task metric for all three in one table.
3. **Gate #0 — path-ablation** (`project_graph_edge_state_bypass`): zero-E vs zero-nodes. Require
   node-only to beat the no-memory floor AND edge-only to be strictly WORSE than REAL. If 100% of the
   memory rides the free E vector and nodes are vestigial, the "graph" is a flat bank in disguise — the
   exact prior failure. This is a HARD gate before trusting any edge result.

### Canaries — LEADING indicators first (every prior collapse was visible here FIRST)
The three signals below caught every collapse in the slotgraph/furlgraph line *before* the outcome metrics
moved. §9 must ship them, not just the lagging outcomes:
1. **Input-dependence of E / `a_ij` across examples** (per-pair participation-ratio of `E[i,j]` and the
   harvested `a_ij`, on BINDING tasks specifically — babi, not aggregate). THE leading indicator — the
   `routing_diversity` analog; SHUF−REAL *lags* it. Without this, a SHUF−REAL≈0 result is undiagnosable
   ("did E fail to become input-dependent, or did something downstream break?"). `depthtime_trace` called
   this "the key metric going forward." If `S`/`a_ij` is input-invariant on babi, E is DOA regardless of
   objective (`project_graph_write_collapse`: the parser emitted one input-invariant edge ×128, edge-cos
   ≈0.999, BEFORE the read).
2. **Gradient-norm per edge-module** — `‖∂L/∂{U, W, φ, E}‖` as grad/param ratios vs the read-LoRA, every
   val step (infra: `scripts/diagnostics/slotgraph/slotgraph_gradflow.py`). The prior 1000× write
   starvation (`project_graph_internals_diag`: q_dst 1.4e-4 vs decoder-LoRA 1.8e-1) was visible ONLY at the
   gradient level. Gate: edge-machinery grad/param >~100× below the read-LoRA in the first few hundred
   steps = fail (the P8 deadlock, live).
3. **Per-window streaming collapse trace** — per window emit ID-subtracted edge-effrank(`E_w`), inter-edge
   cosine, and fixed-pair drift `cos(E_w[i,j], E_0[i,j])`. Over-smoothing COMPOUNDS across the 8-window
   recurrence — furlgraph's `node_wcos 0.34→0.94` was the #1 documented risk. (infra: `collect_layer_metrics`.)

### Canaries — lagging outcomes + design-validators (all cheap, @no_grad, GPU-scalar)
- **learned-`W` block weights** — did the diff/product columns survive, or did `W` collapse to raw-only?
- **edge-state effective rank** (ID-subtracted, content-only — NOT id-spoofed).
- **`U=0` ablation** — are edges load-bearing or decorative?
- **overwrite canary** — write fact → overwrite same pair → test old-fact suppression (validates the
  error-correcting delta of §4; a pure EMA would fail this).
- **direction canary** — `E[i,j]` vs `E[j,i]` distinguishability (the diagonal-product DistMult symmetry risk).
- **retention-vs-lag** — bio query-window sweep 0..7; uniform idle-decay shows as monotonic degradation with
  lag (validates the per-edge gated retention of §4).
- **week-0 persistence/BPTT** — bind at window 0, query at window 7, loss only at end; confirm ∂L reaches
  the window-0 write BEFORE trusting 8-window retention.
- **SHUF−REAL and OFF−REAL** per-task — does memory bind / contribute.
- **node-order shuffle invariance** — validates the set-invariance (no-PE) design (within-example
  order-invariance — DISTINCT from the cross-batch SHUF binding control).
- **write telemetry** — `‖S‖`, `‖E‖`, α/β gate means per window (is the commit alive + bounded?).

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
  only when it adds DIVERSITY (RealFormer/Re-attention) or GATES compute (FastV), which competitive
  assignment already provides. So the edge state must ride the VALUE path (non-absorbable), not the bias.
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
Relational Attention (Diao & Loynd, ICLR 2023, 2210.05062 — edge vectors on the value path; slotgraph adds
PERSISTENT + PLASTIC edges); eligibility-trace learning (e-prop, Bellec 2020) / Backpropamine + differentiable
plasticity (Miconi 2018, 2002.10585) — the learnable temporal-trace machinery; EntNet (Henaff 2017) / Slot
Attention (Locatello 2020) — keyed competitive slots + anti-duplication; ReZero (Bachlechner 2020); Titans
(Behrouz 2024, 2501.00663) — in-cohort EMA-with-momentum memory (the persistence precedent); RealFormer
(He 2020) / Re-attention (Zhou 2021) — feeding attention forward helps via DIVERSITY; Reuse Transformers
(Bhojanapalli 2021) — cross-layer attention redundancy (why rollout-feedback is inert); TransE/DistMult/
ComplEx + InferSent/SBERT + xDeepFM — the operator survey (§8); Set Transformer / Perceiver — set read.

## 12. Status
Design complete + coherent; NOT built. slotgraph = 96 nodes, no edge tokens; ONE shared frozen LM with
**two rank-16 LoRAs** (write-harvest + read-decode); persistent plastic value-path edge state harvested from
the write forward's per-layer attention; learnable inter-layer diff+product operator; **error-correcting,
per-edge-gated, EntNet-bounded commit** (§4, re-adopting slotgraph4's fixes); zero-init with **write-gate-open
to avoid the double-zero deadlock** (§5); prepend+bidir read with a **learned salience selector** (§6).
**Gated on §9: get one binding baseline — under the MEMBERSHIP+ADDRESSING objectives, not behavioral-KL
alone — before building the edge machinery**, armed with the three LEADING canaries (input-dependence,
gradient-norm, per-window collapse trace) that caught every prior collapse first. That measurement is what
turns this from an elegant design into a known-worthwhile build. The exploratory slotgraph/2/3/4 docs remain
on disk as history; this is THE slotgraph.
