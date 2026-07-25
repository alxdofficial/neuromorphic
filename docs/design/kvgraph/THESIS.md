# SlotGraph v2 — Graph KV-Compression with Recoverable Eviction

_Design doc. Written 2026-07-23; §3–§5 (what the graph actually is, topology ops, memory tiers) added
2026-07-24 after a working session that settled the construction. Supersedes the entangled slotgraph line
(`slotgraph_design.md`, `graph_thesis.md`, `furlgraph_design.md`) as the current thesis. Those documents
remain on disk as the record of why the previous framing hit the membership/loss-neutrality wall; this one
is the response to that wall._

---

## 0. One-paragraph thesis

Take the objective that already **wins** our own Phase-2 benchmark (KVzip's query-agnostic
reconstruction-based KV compression, 0.519 on MemoryAgentBench, best of every compression method) and replace
its flat per-entry pruning with a **learned graph** over hidden/KV states that encourages **node reuse** and
carries **relational edges**. Do NOT ask this graph to be a persistent store. Instead add a **discrete,
input-dependent eviction** operation that moves cold subgraphs to disk while **keeping pointers** in the
active workspace, so eviction is recoverable rather than the greedy-irreversible eviction that makes H2O fail.
Pretrain under **multiple compression budgets** so a hierarchy emerges. The end state is a compressed,
reuse-encouraging, relationship-bearing memory with a naturally hierarchical, recoverable eviction mechanism —
and, eventually, a substrate expressive enough to do **JEPA-style latent reasoning in graph space** instead of
decoding to language every step.

The load-bearing claim is NOT "a graph compresses better." KVzip already compresses well. The claim is that
**recoverable eviction + a compression budget forces the edges to carry binding**, which is the one thing every
compressed memory in the literature fails at (multi-hop / associative recall), and which no independently
verified system currently beats attention or RAG on.

---

## 1. Why the previous slotgraph line died (the wall we must clear)

Documented across `project_slotgraph3_result.md`, `research_custom_vs_lm_posterior_collapse.md`,
`project_structure_vs_flat_proof.md`, `project_diagnosis_2x2.md`:

- **Binding is loss-neutral.** Under a reconstruction/plain-CE objective, all write mechanisms tie. The graph
  edges were decorative; a flat bank did the same job; message passing was inert; the model routed around the
  structure (posterior collapse / edge_state bypass).
- **The proof.** At fixed degrees of freedom, structure cannot beat flat **unless the topology carries
  discrete, input-dependent bits.** Every dead iteration violated this — the topology was static and carried
  no bits the flat baseline lacked.
- **Root cause = objective, not architecture.** The membership/binding wall was repeatedly diagnosed as an
  OBJECTIVE problem (loss-neutrality), not a read-surface or capacity problem.

**Retro-diagnosis added 2026-07-24 — the second root cause was a substrate/task mismatch.** sg3 stored edges
*implicitly*, as soft superposed key→value associations inside per-slot matrices (the T3 keyed delta-rule
write). A superposed associative store is formally closer to **flat** than to graph: its "topology" is soft
superposition, carrying no discrete input-dependent bits, so the structure-vs-flat proof says it *cannot* win.
localrun-0005 is the clean demonstration — collapse fixed (node_cos 0.41, effrank 29, distinct high-rank
slots) and binding **still** flat, with `edge_inputdep` essentially unchanged (2.79→2.90): distinct slots,
generic edges.

Deeper still: sg3 was asking a **distributed parametric substrate to store particulars** (bAbI facts, entity
bindings). That is precisely the job the fixed-state recall ceiling (§2.4) proves such a substrate is bad at —
82% of the SSM-vs-attention gap is associative recall. Particulars need an **explicit, addressable, discrete**
store; parametric distributed state is the right substrate for *generalities*. See §5: sg3's machinery is not
wasted, it was aimed at the wrong tier.

**v2's job is to make the graph topology carry input-dependent bits by construction, not by hoping an aux loss
installs them.** See §6.

---

## 2. External grounding (2026-07-23 literature sweep + our own Phase-2 results)

Three independent research passes (recurrent latent compression; test-time-updated parametric memory;
recurrent-state SSM/linear-attention) converged on findings that shape this design:

1. **Reconstruction-based KV compression is the validated objective.** KVzip (NeurIPS'25, query-agnostic,
   reconstruction-scored) was the strongest compression method in our own run (MAB 0.519 / LME 0.523),
   beating H2O@2% and — on LongMemEval — beating llama-3.1-8B reading the full context. Stage 1 inherits this
   objective rather than inventing one.
2. **Greedy irreversible eviction is the failure mode to fix.** H2O evicts before it knows what matters and
   cannot recover it → refused 95.8% of LongMemEval. Stage 2's recoverable eviction is aimed precisely here.
3. **Hybridization is the only fix that survived independent reproduction.** Qwen3-Next and Kimi Linear both
   ship ~75% linear/recurrent memory + ~25% full attention (3:1). Gated DeltaNet succeeded by NOT claiming to
   be a standalone memory. Our active-workspace + recoverable-disk is the same shape: cheap structured memory
   for the bulk, retrieval for the recall-critical tail.
4. **The fixed-state recall ceiling is proven, not incidental.** "Repeat After Me" (2402.01032):
   information-theoretic proof a bounded state cannot losslessly copy unbounded input. Zoology (2312.04927):
   **82% of the SSM-vs-attention gap is associative-recall failure.** "Illusion of State" (2404.08819): SSMs
   bounded by TC⁰. This is why M+ collapsed 0.423→0.286 (LME→MAB, 124k→793k) in our run — it is the ceiling,
   on real data, exactly as predicted.
5. **The open gap.** No architecture in any family has a documented, independently verified case of **beating
   strong long-context attention or RAG on a real (non-synthetic) multi-hop task.** ARMT beats GPT-4+RAG only
   on synthetic single-fact BABILong (multi-hop → 37%). Cartridges beats full-ICL on LongHealth but is
   per-document offline training, not streaming memory. **This gap is the target.**
6. **Bold claims don't reproduce.** Titans and Infini-attention both failed independent reimplementation.
   Discipline: v2 makes modest, falsifiable claims and de-risks the crux first.

---

## 3. What we mean by "graph" (the concrete construction)

Everything below was settled in the 2026-07-24 session by hand-constructing the same sentence twice and
comparing. It replaces the hand-wave "states become/attach to nodes."

### 3.1 Scope — the memory is an ABox, not a semantic theory

The graph stores **ground facts about particulars**: specific individuals and specific events. General,
quantified, universal knowledge is **delegated to the frozen LM's parameters**.

In description-logic terms the split is TBox (general axioms — *selling transfers ownership*, *brothers are
siblings*) vs ABox (assertions about individuals — *this Maria's brother allegedly sold this farm*). A frozen
Llama-3.1-8B already holds the TBox; what it lacks, and what a memory must supply, is the ABox.

This is a scope **commitment**, and it must be disclosed as a limitation rather than buried:

> _The memory represents ground facts about particulars; general and quantified knowledge is delegated to the
> frozen LM's parameters._

Three consequences:

- **It matches the workload.** LongMemEval, MemoryAgentBench, factconsolidation, WikiBigEdit, MSC are
  essentially all ground facts about specific individuals.
- **It retires the two hard corners.** Quantifier scope (*every boy loves a girl*) is out of scope by
  construction. Negation-over-a-region was hard mostly through its interaction with quantifiers; for ground
  facts, negation is a **polarity attribute on an event node**, which covers effectively every realistic case.
- **It gives a free admission filter.** If a statement is not about a particular, it does not enter the memory
  at all — the LM already knows it. That is compression applied *before* eviction runs.

**"Particulars", NOT "named entities".** *The rumour*, *the sale*, *the letter*, *the delivery* are none of
them named entities in the NER sense, yet all are nodes. Reaching for an off-the-shelf NER model would miss
most of the graph. The operative notion is groundedness/particularity — which is what **coreference + SRL**
identify, and why AMR is the scaffolding parser of choice (§8).

### 3.2 Nodes — entities and events, uniformly

Two node kinds, one representation:

- **Entity nodes** — particular individuals (Maria, the brother, the farm, the letter, the family).
- **Event nodes** — particular events/states, reified (the sale, the delivery, the believing).

The noun/verb distinction **dissolves at the graph level**, which is a feature: nominalizations (*the sale*,
*her refusal*, *the destruction of the city*) are exactly the case where an event must be referred to as a
thing, and they need no special handling because an event node is already addressable.

### 3.3 Edges — event hubs with role-labeled spokes (neo-Davidsonian)

**An event is a hub; all its arrows point out to its participants, each labelled with the participant's ROLE.**

```
                     Rumour
                       │ content
                       ▼
        Brother ◄── (Sale) ──► farm            Sale :manner secretly  :time t1
                 seller    merchandise
```

rather than `Brother --sold--> farm`. This is Davidson/Parsons event semantics
(`∃e. Sale(e) ∧ Seller(e,brother) ∧ Merchandise(e,farm) ∧ Manner(e,secretly)`), FrameNet frames + frame
elements, RDF's n-ary relation pattern, AMR's `:ARGn`, and formally the **incidence encoding of a hypergraph**
(an n-ary relation is a hyperedge; a hyperedge is drawn as a node joined to its members). Existing machinery —
SRL parsers, role inventories, bipartite-incidence GNNs — applies directly.

**Why it matters architecturally: reification moves the argument information out of edge DIRECTION and into
the edge LABEL.** With verbs-as-edges, orientation distinguishes agent from patient — 1 bit, and half the
argument structure is invisible to a type-keyed operator. With hubs, direction is uniform and semantically
empty (always hub→participant) and the role label carries log₂K bits. Our typed-operator mixer keys on edge
type, so hubs put the whole role in the place the mixer can see it.

**Two things this fixes that verbs-as-edges cannot:**

1. **Propositions as arguments.** *"the rumour **that** her brother sold the farm"* requires pointing at the
   selling. An edge cannot be the endpoint of another edge. Without the `Rumour --content--> Sale` edge, the
   graph asserts the sale as world-fact — a **truth-conditional error**, not a lossy compression. Concretely,
   these two sentences produce byte-identical entity-only graphs and opposite truth conditions:
   - *"Maria didn't believe the rumour that her brother sold the farm."*
   - *"Her brother sold the farm. Maria didn't believe the rumour about him."*

   **No traversal recovers a distinction that is not in the structure.** The general test: _could two
   sentences with different meanings produce this exact graph?_ If yes, retrieval cannot fix it. Neighbourhood
   retrieval recovers what was **compressed**, never what was **destroyed**.
2. **N-ary relations and modification.** *send(sender, recipient, payload)* does not fit a binary edge; nor do
   `:manner`, `:time`, `:polarity`, or a discourse link. All attach to a hub.

### 3.4 Edge types — a small canonical role inventory (the free-bits constraint on edges)

Roles come from a **fixed, shared inventory of ~20–25 types**, not per-verb invented labels. In the hand-built
graph, `payload` and `merchandise` were the same role (the thing transferred) under two names — exactly the
redundancy that prevents reuse.

Core roles: `agent, theme, recipient, experiencer, instrument, source, goal, manner, time, location, cause,
purpose, content, attribute, quantity`. Stative/entity–entity: `part-of, member-of, owns, located-in`.
Discourse (hub→hub): `because, although, contrast, then`.

The inventory size **K is the free-bits constraint on the edge channel**, the exact analogue of the node
budget: a small K forces relation reuse, and reuse is what makes the topology carry information a flat bank
does not have. It is also what keeps the typed-operator bank `{R_r}` small enough to share across the corpus.

### 3.5 Representation — discrete anchor + continuous residual, decodable by commitment

Nodes and edges carry **continuous vectors**, but anchored to a **discrete learned codebook** (~1–3k codes):

```
node_vector  =  anchor_code[c]            # discrete, shared, stable identity + type
             +  continuous_residual       # instance-specific detail
```

- The **anchor** supplies stable identity across mentions, a shared vocabulary that must be reused (a second
  free-bits constraint), and — critically — **`c` is a discrete input-dependent bit**, so the codebook
  contributes to the structure-vs-flat requirement independently of eviction.
- The **residual** supplies instance specificity, so nodes are not forced into codebook centroids.
- Use **FSQ / lookup-free quantization, not classic VQ.** We have already hit VQ codebook collapse in this
  project (`project_repr_codebook_collapse_fix`); FSQ has no codebook to collapse.
- A k-sparse combination over D≈3000 primitives yields C(3000,4) ≈ 10¹² distinct codes — capacity is a
  non-issue; the discreteness is what we are buying, not the compression.

**Decodability is a design commitment.** Node and edge vectors stay near the LM's token-embedding manifold so
the graph can be **decoded back to text and probed**. This is deliberate and it has a price and a payoff:

- _Price:_ it **caps compression** at roughly what text needs. We do not out-compress language (see §3.7).
- _Payoff:_ the memory is inspectable (dump the graph, read it), which is worth a great deal for debugging and
  for the paper; and node→source-span decoding is a free training signal that keeps nodes grounded.

### 3.6 How much abstraction — graduated, set by the budget

Abstraction is not a fixed level; it is a **ladder the budget walks down**, and this is what Stage 3's
hierarchy actually is:

| level | representation | cost |
|---|---|---|
| L0 | raw token KV | what we are replacing |
| L1 | node = anchor + residual, decodes to its source span | full |
| L2 | node = anchor only (identity + type; detail dropped) | cheap |
| L3 | **beacon** = bundle (superposition) of the anchor codes of an evicted subgraph | one node |
| L4 | on disk, full fidelity, recoverable via the beacon's pointer | free (not resident) |

Bundling anchor codes for a beacon is VSA-style superposition: membership is testable by dot product against
the bundle, degradation is graceful as more is bundled, and **beacons over beacons** give multi-resolution for
free. The ceiling on abstraction is decodability (§3.5); the floor is the node budget.

### 3.7 What the graph does NOT claim

Human vocabulary and word order are near several efficiency frontiers — Zipf's abbreviation law, Uniform
Information Density, ~39 bits/s across languages (Coupé 2019), near-Pareto-optimal semantic systems
(Kemp & Regier; Zaslavsky IB), dependency-length minimisation across 37 languages (Futrell & Gibson 2015).
**We do not claim to beat language at being language.** Language is optimal for a 1-D acoustic channel at
~39 bits/s to a receiver with a shared prior. A machine memory has none of those constraints.

The claim is therefore on the **structure/access axis, not the vocabulary axis**:

- the **KV cache** is demonstrably not an efficient memory (that is why KV compression works at all);
- language cannot **deduplicate** (every mention must be re-uttered) — the graph merges N mentions to 1 node;
- language cannot **update in place** — it can only append *"actually, X changed to Y"*. A graph modifies the
  edge. Per `project_eval_protocol`, good memory = compression × write/update, and the write/update axis is
  where every benchmark we are weakest on (factconsolidation, WikiBigEdit) lives.

**The differentiator is mutability and addressability, not size.** That is a far more defensible claim than a
compression claim, and it should be the paper's framing.

---

## 4. Topology operations (the discrete, input-dependent bits)

Four operations mutate the graph. All are discrete, input-dependent, and budget-driven — which is exactly what
§6 needs them to be.

| op | trigger | effect | dual of |
|---|---|---|---|
| **merge** | two nodes are the same particular (coref/entity resolution) | N mentions → 1 node | split |
| **promote** | an edge gets referenced, or needs >2 arguments | edge → event hub + role spokes | demote |
| **demote** | a binary hub is unreferenced and budget is tight | hub → plain edge | promote |
| **evict** | subgraph is cold | subgraph → beacon + pointer; body to disk | recall |

**An edge is a compressed event node; a beacon is a compressed subgraph — the same operation at two scales.**

**Construction is uniform; compression is learned.** A hand-written "reify only on reference or arity" rule
would be a *hand-designed compression heuristic*, and it requires knowing the future (a later sentence may
reference a relation already committed as an edge — *"Maria's refusal to believe surprised everyone"*). Our
thesis says compression decisions should be learned and budget-driven. So:

> **Parse to the uniform event-hub form (slightly over-complete), then let the budget demote and evict.**

This also lands on the working instinct from the session: **simple construction, learnable pruning** — and it
makes edge-vs-node one of the *learned* decisions rather than a parse-time guess.

### 4.1 Traversal, eviction score, and position are ONE mechanism

Use **personalized PageRank** (restart-walk) seeded at query-matching nodes:

```
r = α·s + (1−α)·W·r          s = query-similarity seed,  α = restart probability
```

This single mechanism does four jobs, which is the elegance argument for it:

1. **Read traversal** — take top-B by `r`. "How far do I traverse" becomes the single knob α (high = local,
   low = diffuse); there is no discrete depth decision and no hop limit.
2. **Eviction score** — accumulate `r` mass across queries = visit frequency = the coldness signal.
3. **Positional encoding** — random-walk structural encoding is the same family (see §4.2).
4. **Mixing** — graph diffusion `Σₖ αₖ Âᵏ` is the same operator.

Properties that matter: cycles are native (it is a Markov chain — see §4.3); local push variants
(Andersen–Chung–Lang) are sublinear so the whole graph is never materialised; it is differentiable (power
iteration; APPNP is exactly this as a propagation layer).

**Walk symmetrized, compute directed.** PPR runs on the symmetrized adjacency for reachability; the mixer uses
direction + role for the typed operator. This settles the inverse-edge question: **store no inverse edges.**
With hubs, co-participants are always exactly 2 hops apart (`Brother ← Sale → farm`), so symmetric walking
gives co-participant reachability without doubling the edge count.

### 4.2 Positional encoding — structural, relative, anchored to the persistent hierarchy

Positional encoding does three jobs that a single integer conflates on a line: break permutation symmetry,
encode relational geometry, and provide addressing. On a graph they separate.

Sinusoidal PE **is** the Laplacian eigenbasis of the path graph — so we generalise rather than abandon it:
Laplacian eigenvector PE, random-walk structural encoding (RWSE), shortest-path-distance bias (Graphormer's
`bias = f(spd(i,j))` is literally relative PE), or anchor/landmark distance vectors.

**Our eviction constraint drives the choice: PE must be an invariant of LOCAL structure, not of the global
embedding**, because the global structure keeps changing. That rules out absolute Laplacian eigen-coordinates
as primary (one edit shifts the whole spectrum) and favours graph-distance, anchor-relative coordinates, and
random-walk fingerprints.

**Anchors are the always-resident coarse-hierarchy nodes** (§3.6 L2/L3), so PE survives eviction by
construction: a leaf moves to disk but its gravestone still says *"I lived at distance-2 from anchor A."* The
positional encoding and the recovery pointer are the same object. Anchors also serve as **query entry points**
— PE and "where do I start reading" are one problem.

**Time is a demoted axis, not the spine.** Order matters (factconsolidation is fact-updates-over-time), but in
a graph time is one labelled coordinate among several — a recency scalar on an event hub, plus version-ordering
edges between successive states of a fact — rather than the organizing backbone.

**The trap:** a PE that is a learned per-node ID is the sg3 "id-tags are free" failure reborn — it
distinguishes nodes (job 1) but carries no structure (job 2), i.e. address-not-content, generic, loss-neutral.
PE must be a **function of the topology**.

### 4.3 Cycles

Cycles are allowed and expected (knowledge is cyclic). They only ever caused problems through **leaf-based
pruning**; §4's eviction is **score-based** (any node is evictable by PPR-mass × recency × centrality), so no
leaves are required and the objection dissolves. Traversal termination is a visited-set plus the budget.

Corollary: **do not use distance-from-root as an importance proxy.** In both hand-built graphs, the only node
everything was reachable from was the *least* important entity in the sentence (Bank, then Delivery). Keep
reachability-from-anchor as an invariant; use access frequency for importance.

### 4.4 The mixer — typed-operator message passing

```
h_i' = h_i  +  Σ_{j ∈ N(i)}  a_ij · ( R_{r(ij)} · h_j )   +  landmark/pointer term
```

- **`r(ij)` is a discrete role type selecting an operator `R_r` from a small learned bank.** Multiplicative
  (operator application), not additive with untyped scalar weights — the latter is the flat operation the
  structure-vs-flat proof says cannot beat a flat bank, and is what made sg3's message passing inert.
- **Sum aggregation, not mean.** Sum is injective (GIN); mean over-smooths hardest. Anti-collapse by
  construction.
- **Bounded long-range escape** — landmark/hub nodes reachable in ≤2 hops, plus eviction **pointers acting as
  long-range shortcut edges**. Over-squashing (k hops needs k layers, exponential receptive field through a
  fixed channel) is the graph-native restatement of our multi-hop wall; over-smoothing is the graph-native
  restatement of our collapse. Both must be designed against from step one or the mixer reproduces our old
  dead runs with new vocabulary.
- Cost is `O(E·d·r)` with `R_r` diagonal-plus-low-rank, versus attention's `O(N²·d)`.

**Reusable from sg3** (the machinery is not wasted): the delta-rule associative write primitive, the KBLaM
boundary-token rectangular-mask read surface (our audited, literature-correct read), and softmax+Gumbel
routing (exactly the discretization typed-edge selection needs). **Replaced:** the edge representation — soft
superposed associations → committed discrete typed edges.

---

## 5. Memory tiers (where each piece lives)

The graph is **one tier of three**, and separating them explains both what to build and what not to.

| tier | content | substrate | status |
|---|---|---|---|
| **L1 — episodic** | ground facts about particulars: entities, events, what happened | **the graph** (§3), non-parametric, bounded, recoverable-evictable | **the new work** |
| **L2 — semantic** | assumptions, biases, beliefs, dispositions; concepts and how they associate | parametric distributed state; nodes are ideas not individuals; **edges carry no meaning — they modulate interactions between nodes** | where the retired slotgraph line belongs |
| **L3 — archival** | files, external stores, anything unbounded | agentic tool calls / RAG | **off-the-shelf; a baseline, not something to build** |

Three things follow:

1. **This retro-explains sg3's failure** (§1). sg3 was an L2-shaped substrate asked to do L1's job — store
   particulars. Distributed parametric state is provably weak at exactly that (associative recall,
   82% of the SSM gap). The negative results are not wasted; they were aimed at the wrong tier.
2. **L2 needs a DIFFERENT mixer from L1.** L1 edges are typed roles carrying relational meaning; L2 edges are
   graded couplings/gates that modulate (Hopfield-weight-like). Do not reuse one edge mechanism for both.
3. **No tier should be pure.** M+ is a flat parametric pool *plus* a retriever over CPU-offloaded LTM — even
   the most parametric system in the panel is hybrid, and it still hit the ceiling (0.423→0.286). This is the
   same finding as the 3:1 hybridization consensus (§2.3).

**Scope discipline: build L1. L2 is deferred (and is where sg3 gets revived if it is revived at all); L3 is a
baseline we already run.** The falsification in §7 concerns L1 only.

---

## 6. The crux — why the topology carries input-dependent bits this time

The structure-vs-flat proof still applies. Reconstruction ALONE does not force binding (a flat VQ codebook
reconstructs fine; this is the AutoCompressor/ICAE trap — great reconstruction, 12.9/19.5 on real QA). The
escapes are **architectural, not aux losses**, and there are now three:

1. **The compression budget is the objective fix (free-bits / rate constraint).** A tight budget makes node
   **reuse** necessary — repeated substructure must share nodes or the budget is blown. Reuse stops being
   optional. The same constraint applies twice more: a small **role inventory K** (§3.4) and a small **anchor
   codebook** (§3.5) force reuse on the edge and vocabulary channels.
2. **Topology mutation is discrete and input-dependent, so the topology is functional, not decorative.**
   *Which* subgraph is evicted, *which* edge is promoted to a hub, *which* mentions merge — all depend on the
   input and all materially change what is computable downstream. These decisions **are** the input-dependent
   discrete bits the proof requires, carried by construction rather than installed by hope. §4's four ops are
   each a source of such bits; the previous version of this doc only had eviction.
3. **The anchor code `c` is itself a discrete input-dependent bit** (§3.5), independent of any topology op.

**These compose into a binding mechanism reconstruction-alone cannot provide:** if the model must reconstruct
**after** eviction using only retained nodes + pointers, the edges MUST encode binding — otherwise it pulls
back the wrong subgraph. **Reconstruction-after-eviction forces binding where reconstruction-before-eviction
never did.** This is the hypothesis the whole thesis rests on.

**Eviction must preserve transitive reachability.** Evicting B from A→B→C must leave A able to reach C, or
multi-hop collapses exactly when memory fills — which is literally the M+ failure we measured. A beacon is
therefore **not a prose summary**: it is a learned routing key + pointer + preserved pass-through relations.

**The known risk, named:** access frequency is query-dependent but the memory is built query-agnostically
(compute once, answer ~85 questions per context). Evicting on frequency *before* knowing what matters is
precisely H2O's greedy-irreversible failure. Recoverability is the fix **only if the beacon genuinely routes**
— so beacon routing quality is load-bearing, and forcing it is exactly what reconstruction-after-eviction is
for.

---

## 7. First falsification (the milestone that decides everything)

Do NOT build all four stages. Test the crux in Stage 1 (streaming, budgeted, evicting):

> **At a fixed compression budget, does a graph with recoverable eviction beat a flat bank at the same budget
> on a task requiring binding/multi-hop — and does ablating the edges (collapsing to flat) measurably hurt?**

- **If edge-ablation does NOT hurt** → the graph is inert again (the old failure), learned in ONE experiment
  instead of four stages deep. Kill or rethink.
- **If it DOES hurt, specifically on multi-hop / binding** — where every fixed-state model in the literature
  collapses (ARMT 37% on QA3; `factconsolidation_mh` ≤0.05 for every model in our own run, deepseek
  full-context included at 0.200) — then we have the thing the field does not have: an independently
  verifiable real-task win for a compressed memory.

**The specific ablation is TWO-SIDED**, because each edge now carries a discrete relation *and* a continuous
edge vector (BUILD.md §3), and either alone could be doing the work:
- collapse `R_{r(ij)}` to a single shared untyped operator → does the **relation typing** matter?
- zero the edge vector → does the **continuous refinement** matter?

If only the edge vector matters, the graph typing was decorative and we have learned that cleanly rather than
being fooled by a good reconstruction number. This is why the edge vector is bounded by construction
(low-rank, zero-init, decay): an unbounded residual reproduces loss-neutrality in a new idiom.

Anti-Goodhart co-gate (per `project_curriculum.md` discipline): the win must survive a SHUF−REAL control and
must not be reachable by the flat baseline at matched budget. Reconstruction fidelity is NOT the success
metric — downstream multi-hop accuracy under eviction is.

**Test target:** `factconsolidation_mh_{6k,32k,64k,262k}` (multi-hop, our measured wall) and a bound-pair
control. These are already in the MemoryAgentBench loader.

**Open question to resolve before building:** do `factconsolidation_mh`'s multi-hop chains run through
relations an SRL/AMR parse *keeps* (predicate-argument, coref) or through the temporal/discourse residue it
*drops*? If the latter, the parser gets us the backbone but the residue channel does the real work — worth
knowing before committing engineering.

---

## 8. Build order (de-risking sequence)

The two risks — *does graph structure help?* and *can we learn to induce it?* — are independent, and coupling
them is how the previous line died. Decouple:

1. **Parser-given graph first.** Get nodes+typed roles from an off-the-shelf **AMR** parse (SPRING/AMRBART,
   ~84–86 Smatch) or coref+SRL. Feed through the typed-operator mixer into the frozen decoder via the KBLaM
   rectangular mask. **Run the §7 edge-ablation gate now**, before writing any induction code.
   - If a near-oracle *given* graph does not help, learning to induce one is pointless — found out in the
     cheapest possible experiment.
   - AMR gives predicate-argument structure, typed roles, coreference-via-reentrancy, and paraphrase
     normalisation. It drops tense/aspect/number/definiteness, quantifier scope, and is sentence-level (a
     cross-sentence merge step is ours to add). Its losses are almost exactly the serial residue that does not
     fit hub-and-role — which validates a thin side-channel rather than pretending one representation holds
     everything.
2. **Then the learned inducer.** Attention-based induction (pretrained heads already track coreference and
   dependencies — Clark et al., *What Does BERT Look At?*) with slot-attention node pooling and Gumbel edge
   typing. The parser becomes the **weak-supervision target that breaks loss-neutrality** (Hungarian-match
   learned nodes to parse nodes, CE on role types), annealed down as budget+eviction take over.

**The discipline that separates this from the graveyard:** every "attention figures it out" step must name its
**anchor** — external supervision, a non-loss-neutral objective, or a hard bottleneck. Induction stacks five
unsupervised structural decisions (node clustering, role assignment, cross-paragraph merge, access scoring,
beacon routing); under plain reconstruction each is a fresh chance to go decorative.

**Two representational rules, learned the hard way:**

- **Node = identity key; facts live on edges.** Never pool all of an entity's mention-meaning into its node
  vector — merging then averages toward a centroid matching no context, and the node becomes membership-only
  (the PMA / pool-then-address trap). Merging entities merges *keys*; adding facts adds *edges*.
- **Install binding in the write, never pool-then-address** (`research_memory_sidecar_binding`).

**Entity resolution is now the critical path.** Scoping to particulars (§3.1) concentrates risk there:
over-merging fuses two individuals and a hop lands on a wrong fact; under-merging splits one and the hop
breaks. Both corrupt multi-hop specifically. Streaming forces per-paragraph-then-merge (greedy, locally
errorful) over global joint coref — an accepted cost that must be measured, not assumed small.

---

## 9. The four stages (each a distinct, separately-falsifiable thesis)

### Stage 0 — Warmup (plumbing only, NOT a regime)
- One paragraph (~256–512 tokens), graph built per §3, **all** nodes read, reconstruct.
- Sole purpose: verify the decoder can read graph-KV at all. Teaches **representation** before the streaming
  stage teaches **selection**. Proves nothing scientific.

### Stage 1 — Streaming compression with live eviction (the gate)
- Multi-window (ctx 2048 = 8 × 256), one persistent graph, nodes from different windows connect and
  **merge** (§4); node budget **96**, ~3× over-subscribed, so forgetting is real.
- **Eviction is a PRECONDITION of the objective, not a scaling feature.** The crux (§6) is
  reconstruction-*after*-eviction; with eviction deferred, this stage is KVzip-with-a-graph and will be
  loss-neutral exactly as predicted. Read a **retrieved subgraph**, seeded by the preceding window — a
  non-circular query the streaming setting supplies for free.
- Pruning = cache eviction scored by PPR mass (§4.1), **not** by leaf-ness or distance-from-root (§4.3).
- **Eviction ≠ deletion.** Cold subgraphs → beacon + pointer, body to disk, recoverable (§3.6 L3/L4).
- Grounding: KVzip's winning objective with learned graph reuse substituted for flat pruning.

### Stage 2 — Scale
- Long documents, disk offload, recall of evicted subgraphs. Engineering, not new science.

### Stage 3 — Hierarchy via multi-budget pretraining
- Pretrain under different compression budgets to force the §3.6 ladder to appear (Matryoshka-style nesting
  adapted to graph topology): coarse anchors always resident, fine detail evictable and recoverable.
- Grounding: Matryoshka representation learning; Compressive Transformer two-tier; Continuum Memory Systems.

### Stage 4 — Latent reasoning in graph space (JEPA-style)
- Predict **graph-state transitions** rather than decoding to language every step.
- **Discrete anchor codes rescue this from JEPA's own failure mode:** transition prediction becomes
  classification over the codebook (cross-entropy), so representational collapse is impossible and no
  anti-collapse aux loss is needed — consistent with `feedback_avoid_aux_losses`.
- **Graph comparison** (needed for the transition objective) is the Weisfeiler-Leman criterion: compare each
  node's neighbourhood multiset of role-labelled neighbours. The differentiable, permutation-invariant version
  is **Fused Gromov-Wasserstein** (matches nodes so pairwise relational structure is preserved, entropic /
  Sinkhorn-regularised). Shortcut: **with persistent identity keys, correspondence is a lookup, not a matching
  problem** — only genuinely new nodes need soft matching.
- **Grounding and warning.** Latent reasoning is real but **flat**: Coconut (2412.06769) chains hidden states,
  LCM (2412.08821) chains SONAR sentence embeddings. Coconut *loses* to text-CoT on GSM8K (34.1 vs 42.9),
  winning only on planning/search; an independent audit (2512.21711) finds its latents are unsteerable
  shortcut placeholders. Token Assorted (2502.03275) is the closest real "learned denser code" win (+4% Math)
  and is unreplicated. **Structured/graph latent reasoning is essentially absent** (one self-reported Jan-2026
  paper) — genuine white space. But every independent audit converges on *"the objective, not the medium, is
  the bottleneck"* (2510.15522 reads current latent reasoning as a fuzzy superposition over the existing
  vocabulary). **So Stage 4's contribution must be the training pressure, not the substrate.**
- **Explicitly a downstream bet.** Do not let it pull effort from de-risking Stage 1+2.

---

## 10. What is genuinely novel vs recombined (honest accounting)

- **Recombined (validated pieces):** reconstruction-based compression objective (= KVzip); neo-Davidsonian
  event reification (= formal semantics / AMR / FrameNet); personalized PageRank propagation (= APPNP);
  hybridization (= production consensus); multi-resolution memory (= Matryoshka / Compressive Transformer);
  latent reasoning (= JEPA / Coconut).
- **Genuinely novel (to verify, not assert):** (a) a learned graph over **hidden/KV states** with node reuse —
  existing graph-memory work (GraphRAG, Zep/Graphiti, AriGraph) builds **text-level** knowledge graphs, not
  KV/hidden-state graphs; (b) **recoverable subgraph eviction addressed by graph topology** rather than a
  separate vector index; (c) **reconstruction-after-eviction as a binding-forcing objective**;
  (d) **promote/demote/evict as learned, budget-driven topology operations** (edge-as-compressed-node and
  beacon-as-compressed-subgraph as the same operation at two scales); (e) graph as the substrate for latent
  reasoning.
- **The contribution, stated honestly:** the field's gap is not a better compressor (KVzip is already good) —
  it is that no compressed memory beats attention/RAG on a real multi-hop task. If recoverable-eviction +
  budget-forced-binding clears multi-hop under an anti-Goodhart control, **that** is the contribution.

_Novelty items (a)–(e) are claims to verify with a literature check before writing them into a paper, not
settled facts._
