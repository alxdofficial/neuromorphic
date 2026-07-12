# Why a graph memory — the thesis, and what the literature says about making it work

Standing rationale for the graph-memory line (slotgraph3 / FurlGraph), grounded in a 2026-07-09
literature sweep. Read alongside `docs/design/OBJECTIVES.md` (the binding ladder) and `docs/design/furlgraph_design.md`.

## Two lenses that motivate graphs

1. **Node reuse — entities as nodes, relations as edges.** A natural, compositional representation
   (knowledge graphs, semantic networks, scene graphs, **AMR** — Banarescu et al. 2013). Its defining
   property is *reentrancy*: one entity node addressed by many mentions (coreference *is* node reuse).
2. **Language is a latent graph linearized into a chain.** Dependency grammar / Universal Dependencies
   and AMR formalize this: the token chain is a linearization of a head-dependent graph. Not speculation —
   the standard linguistic position. (The "an alien language could be 2-D/multi-dimensional rather than a
   chain" intuition is real at the frontier — non-autoregressive / diffusion LMs show 1-D order is a
   *choice* — but there is **no empirical evidence** a multi-dimensional language is more efficient; keep
   it as motivation, not a claim.)

## Three honest cautions (why the graph must *earn its keep*)

- **The frozen LM already encodes the graph.** Hewitt & Manning, *A Structural Probe* (NAACL 2019): a
  single linear map recovers dependency-tree distances from BERT/ELMo geometry — whole syntax trees are
  implicit in the vectors. So SmolLM2-135M's residual stream *already* carries much of the passage's
  dependency/entity graph. **An external graph memory only earns its keep by supplying what the chain does
  NOT already expose: sparsity, discreteness, or PERSISTENT REUSABLE NODE IDENTITY across the compression
  bottleneck** — not "representing the graph" (the LM does that for free).
- **Self-attention IS a graph operator.** Joshi, *Transformers are GNNs*: attention = message passing on a
  fully-connected token graph. A bolt-on graph adds value only by imposing sparsity / discreteness /
  persistent identity that dense attention doesn't.
- **The KG-augmentation win is regime-dependent.** QA-GNN / GreaseLM / DRAGON / GNN-RAG help on
  **multi-hop / long-tail / small-LM**, but "larger LLMs show lower gains" and noisy memory can *degrade*;
  LMs already store relational facts (Petroni, *LMs as Knowledge Bases?*, EMNLP 2019). Our memory must beat
  the frozen LM's own knowledge **and** plain retrieval — not just a null baseline.

## The core problem: collapse is the *rational optimum* when structure is loss-neutral

This is the project's membership/binding wall, and the literature is unanimous and precise about it:

- **NRI** (Kipf et al., ICML 2018) names it exactly: *"a sub-optimal decoder that ignores the latent edges
  completely … achieves only a marginally-worse reconstruction loss."* When the structureless shortcut is
  within ε of the structured one, gradients never prefer topology → `routing_diversity ≈ 0.02`.
- **Williams, Drozdov & Bowman**, *Do latent tree learning models identify meaningful structure?* (TACL
  2018) — the **published analog of our wall**: trees induced from a *downstream* objective are inconsistent
  across seeds, match no linguistic formalism, and **task score can even IMPROVE under trivial structure.**
  Structure induction needs a signal that rewards the *structure itself*, not just the task.
- **Collapse-to-mean is a built-in attractor**, task-independent: GNN over-smoothing (Oono & Suzuki, ICLR
  2020 — embeddings → low-rank subspace exponentially), pure-attention rank collapse (Dong et al., ICML 2021
  — rank-1 doubly-exponentially without skip/MLP), VAE posterior collapse (a frozen capable decoder is the
  worst case), MoE routing collapse, VQ codebook collapse. A pooling/message-passing read *fights* this pull.

## What actually makes topology load-bearing (the recipe)

From NRI, Slot Attention, the latent-graph-learning line, and the entity-memory line — meaningful structure
emerges **only** when several of these hold (anti-collapse mechanics alone are necessary-not-sufficient):

1. **Make the graph the EXCLUSIVE inter-node channel + make its loss-effect LARGE** (NRI): predict far
   enough ahead / multi-hop that ignoring the structure is *costly*, and don't leave a parallel content path
   that absorbs it. (Our multi-horizon continuation + graph-only read is the lever.)
2. **Competition over slots** (Slot Attention, Locatello et al. 2020): softmax normalized *over slots* so
   slots compete to explain each input — a plain sum/bias pools; competition is what specializes. slotgraph3
   already does this (competitive slot-read); keep it.
3. **Reward the structure DIRECTLY**: supervision or posterior-regularization is what made latent trees work
   (URNNG's CRF guide, DIORA's inside-outside chart). Our version = **provenance-supervised InfoNCE**
   (`docs/design/OBJECTIVES.md` Rung 2) + tasks that *provably* can't be solved structureless (compositional
   generalization: SCAN, COGS, CFQ; bAbI multi-hop).
4. **Discrete, addressable, REUSABLE node identity** (the strongest positive precedent): **EntNet** (Henaff
   et al., ICLR 2017) — keyed, competitive, *independent* slots — first to solve all 20 bAbI. Entities as
   Experts, Mention-Memory/TOME, EntityNLM scale it: identity must be an addressable **key**, not a pooled
   average. Named entities = **node reentrancy** (AMR) + entity linking: one node, many mentions addressing
   it — exactly what pooled/mean memory destroys.

## Topology representation — the corrected guidance

Representing learnable topology as a **differentiable function of node states** (GAT, NRI, DGM /
latent-graph-inference) is sound and gradient-healthy. **But two sharp corrections** (a naive low-rank
bilinear attention-bias is *not* the answer):

- **A content-based bilinear score summed into the SAME softmax as content is ABSORBABLE** — the sum of two
  low-rank bilinear forms is one wider head, adding *no* structural expressivity. Worse, *Routing Absorption*
  (arXiv:2603.02227) shows learned gates converge within 2.2% of a **frozen random gate** (~91% of the
  routing benefit absorbed by Q/K/V co-adaptation) — a **mechanistic explanation of the membership wall**.
  → Topology must be a **competitive, exclusive** structural channel, not a content-parallel bias.
- **A scalar per pair provably cannot carry relation semantics** (DistMult is symmetric; can't model
  asymmetric or multiple relation types). For genuine relational reasoning, **edge vectors read+written each
  layer** win (Relational Attention, Diao & Loynd ICLR 2023, +~11% on CLRS-30). → slotgraph3's per-edge
  **edge-state is the literature-preferred expressive form**, not a liability to drop; the real fork is
  *relations-in-nodes* (content-free edges) **vs** *relations-in-edge-vectors* (more expressive, heavier).

**Net:** the fix for routing collapse is **not** a cleverer topology parameterization — it's (1) an
exclusive/competitive structural channel, (2) a large loss-effect for structure, (3) an objective that
rewards structure directly, and (4) discrete reusable node identity. Architecture buys anti-collapse;
**the objective buys binding.**

## The one-line thesis
You don't get topology by building a better graph module — you get it by making the **structureless
solution lose the objective**, on a substrate with competitive, discrete, reusable node identity, where the
graph is the exclusive channel the reader is forced through.
