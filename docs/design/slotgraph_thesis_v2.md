# SlotGraph v2 — Graph KV-Compression with Recoverable Eviction

_Design doc. Written 2026-07-23. Supersedes the entangled slotgraph line
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

**v2's job is to make the graph topology carry input-dependent bits by construction, not by hoping an aux loss
installs them.** See §4.

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

## 3. The four stages (each a distinct, separately-falsifiable thesis)

### Stage 1 — Graph KV compressor (representation only, NOT persistent)
- Encode a context window's hidden/KV states into a **graph**: states become/attach to **nodes**; the encoder
  is encouraged to **reuse nodes** for repeated substructure and to form **edges** that carry relations.
- Trained on the **existing mixed reconstruction objective** (the one already in the harness) under a
  compression budget. Accumulates over a window like our other models; **no persistence requirement** — this
  stage's only job is to learn the representation and, ideally, get binding working.
- Grounding: this is KVzip's winning objective with learned graph reuse substituted for flat pruning.

### Stage 2 — Cross-subgraph edges + recoverable eviction
- Nodes from **different subgraphs** may connect — the graph is one workspace, not disjoint per-window graphs.
- **Pruning = cache eviction, principled by visitation/modification.** Track which nodes are visited and
  modified (analogous to H2O heavy-hitters and M+ recall-frequency, but on graph topology).
- **Eviction ≠ deletion.** Cold subgraphs/leaves move **to disk** — out of the compute budget but not lost —
  while **pointers remain in the active workspace**. A future read can pull an evicted subgraph back.
- Grounding: fixes H2O's irreversible eviction; hybridizes compressed memory with retrieval-of-evicted-state.

### Stage 3 — Hierarchy via multi-budget pretraining
- Graphs are inherently hierarchical. **Pretrain under different compression budgets** to force a hierarchy to
  appear (Matryoshka-style nested structure, adapted to graph topology).
- Yields multi-resolution memory: coarse summary always resident, fine detail evictable and recoverable.
- Grounding: Matryoshka representation learning; Compressive Transformer two-tier; Continuum Memory Systems.

### Stage 4 — Latent reasoning in graph space (JEPA-style)
- Language output is one long chain = also a graph. With a sufficiently expressive learned representation, do
  **world-modeling / reasoning in the graph representation** rather than decoding to language every step.
- Grounding: JEPA (predict in representation space); Meta Coconut / chain-of-continuous-thought (2412.06769).
- **Explicitly a downstream bet.** Do not let it pull effort from de-risking Stage 1+2.

---

## 4. The crux — why the topology carries input-dependent bits this time

The structure-vs-flat proof still applies. Reconstruction ALONE does not force binding (a flat VQ codebook
reconstructs fine; this is the AutoCompressor/ICAE trap — great reconstruction, 12.9/19.5 on real QA). Two
mechanisms in this design are the intended escape, and they are **architectural, not aux losses**:

1. **The compression budget is the objective fix (free-bits / rate constraint).** A tight budget makes node
   **reuse** necessary — repeated substructure must share nodes or the budget is blown. Reuse stops being
   optional.
2. **Eviction is discrete and input-dependent, so the topology is functional, not decorative.** *Which*
   subgraph is evicted depends on the input, and eviction materially changes what is computable downstream.
   The pruning decisions **are** the input-dependent discrete bits the proof requires — carried by
   construction, not installed by hope.

**The two compose into a binding mechanism reconstruction-alone cannot provide:** if the model must reconstruct
**after** eviction using only retained nodes + pointers, the edges MUST encode binding — otherwise it pulls
back the wrong subgraph. **Reconstruction-after-eviction forces binding where reconstruction-before-eviction
never did.** This is the hypothesis the whole thesis rests on.

---

## 5. First falsification (the milestone that decides everything)

Do NOT build all four stages. Test the crux in Stage 1+2:

> **At a fixed compression budget, does a graph with recoverable eviction beat a flat bank at the same budget
> on a task requiring binding/multi-hop — and does ablating the edges (collapsing to flat) measurably hurt?**

- **If edge-ablation does NOT hurt** → the graph is inert again (the old failure), learned in ONE experiment
  instead of four stages deep. Kill or rethink.
- **If it DOES hurt, specifically on multi-hop / binding** — where every fixed-state model in the literature
  collapses (ARMT 37% on QA3; `factconsolidation_mh` ≤0.05 for every model in our own run, deepseek
  full-context included at 0.200) — then we have the thing the field does not have: an independently
  verifiable real-task win for a compressed memory.

Anti-Goodhart co-gate (per `project_curriculum.md` discipline): the win must survive a SHUF−REAL control and
must not be reachable by the flat baseline at matched budget. Reconstruction fidelity is NOT the success
metric — downstream multi-hop accuracy under eviction is.

**Test target:** `factconsolidation_mh_{6k,32k,64k,262k}` (multi-hop, our measured wall) and a bound-pair
control. These are already in the MemoryAgentBench loader.

---

## 6. What is genuinely novel vs recombined (honest accounting)

- **Recombined (validated pieces):** reconstruction-based compression objective (= KVzip); hybridization
  (= production consensus); multi-resolution memory (= Matryoshka / Compressive Transformer); latent reasoning
  (= JEPA / Coconut).
- **Genuinely novel (to verify, not assert):** (a) a learned graph over **hidden/KV states** with node reuse —
  existing graph-memory work (GraphRAG, Zep/Graphiti, AriGraph) builds **text-level** knowledge graphs, not
  KV/hidden-state graphs; (b) **recoverable subgraph eviction addressed by graph topology** rather than a
  separate vector index; (c) **reconstruction-after-eviction as a binding-forcing objective**; (d) graph as
  the substrate for latent reasoning.
- **The contribution, stated honestly:** the field's gap is not a better compressor (KVzip is already good) —
  it is that no compressed memory beats attention/RAG on a real multi-hop task. If recoverable-eviction +
  budget-forced-binding clears multi-hop under an anti-Goodhart control, **that** is the contribution.

_Novelty items (a)–(d) are claims to verify with a literature check before writing them into a paper, not
settled facts._
