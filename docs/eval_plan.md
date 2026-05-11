# Evaluation plan — baselines and benchmarks

Status: draft, 2026-05-11. Distilled from the literature review on
trajectory-memory vs established long-context / memory-augmented LM
architectures.

## TL;DR

- **Classical long-context QA is not our defensible territory.** A
  well-tuned RAG baseline at Llama-3.2-1B class will likely beat us on
  NarrativeQA / HotpotQA / NaturalQuestions. Reporting those is
  important for honesty, not for headline.
- **Defensible niche, if any: memory-evolution benchmarks** where every
  existing method scores <10%. Specifically MemoryAgentBench CR-MH
  (cross-session multi-hop, current SOTA <6%) and LongMemEval
  knowledge-update / multi-session subsets.
- **Most threatening competitor: RMT** (Bulatov 2022) — adds near-zero
  parameters and dominates needle-style benchmarks. If we cannot beat
  RMT on BABILong at our compute budget, the architecture is not
  justified.
- **DNC is the base-rate warning sign.** Architecturally closest
  cousin; failed to scale beyond synthetic bAbI; community largely
  abandoned the lineage. We have to articulate why we won't suffer
  the same fate.

## Baseline tiers

### Tier 1 — Sanity (must beat to justify the architecture at all)

| # | Baseline | What it isolates | Implementation cost |
|---|----------|-------------------|---------------------|
| 1 | Frozen Llama-3.2-1B, no memory, zero-shot | Baseline raw capability | Free (inference-only) |
| 2 | Frozen Llama-3.2-1B + adapter only | MemInjectLayer trained, manifold dummied (zeros or random states); checks that the manifold actually contributes vs the adapter doing all the work | ~5M params, one short training run |

If our system does not beat both Tier 1 baselines, the manifold +
graph traversal isn't adding anything — stop here and redesign.

### Tier 2 — Direct architectural competitors at our compute budget

| # | Baseline | Why it matters | Implementation cost |
|---|----------|----------------|---------------------|
| 3 | **RMT-1B** on Llama backbone (Bulatov et al. 2022) | The most embarrassing comparison: near-zero added parameters, dominates BABILong. If our 16M-param graph-routing apparatus loses to a few memory tokens, the architecture is not justified. | ~50 LOC plus a Wave 1-equivalent training run. Single biggest research priority. |
| 4 | Memorizing Transformers small (frozen Llama + kNN K/V cache + gated attention) | Closest "external memory" cousin. Non-differentiable cache vs our differentiable manifold. | ~300 LOC; one training run. |
| 5 | Larimar-style adapter (Bayesian-Kanerva memory + frozen LLM) | Closest published frozen-LM episodic memory. If we want to claim "graph traversal" as the novel piece, we need to beat Larimar's flat Kanerva read. | ~400 LOC; one training run. |

Tier 2 is where the research question lives. If we beat Tier 2 on at
least one benchmark, there's a story.

### Tier 3 — Production-grade comparators (likely we lose; quantify the gap)

| # | Baseline | Why include it | Implementation cost |
|---|----------|----------------|---------------------|
| 6 | RAG with e5-base encoder + top-5 chunks + Llama-3.2-1B reader | Standard external baseline. Establishes the ceiling we cannot reach on classical QA. The number to quote for honesty in the paper. | ~200 LOC; vector-DB setup. No training run needed (frozen encoder). |
| 7 | Long-context Llama-3.2-1B at 8K or 16K context | Baseline for "does our memory module add anything beyond just giving Llama more context?" | Free (HF supports it). |

### Tier 4 — Mode-specific competitors (per benchmark)

| Benchmark | Strongest published baseline at our scale |
|-----------|-------------------------------------------|
| BABILong / needle | RMT-1B (dominates) |
| NarrativeQA | RAG (DOS-RAG simple) — F1 ~25-32 at 1B |
| HotpotQA | ChatQA-class with retrieval — EM ~30-35 |
| MemoryAgentBench CR-MH | Everyone fails (<6%) — our window of opportunity |
| LongMemEval knowledge-update | Long-context LLM drops ~30% across sessions; RAG cannot patch this automatically |

### Tier 5 — Ablations of our architecture (cheap, high-value)

These are variants of our own training run with one component disabled
or replaced. They do not need separate baseline implementations — just
re-runs with different flags or a few-LOC patch.

| Ablation | Question answered |
|----------|-------------------|
| No graph (full N×N routing) | Does the small-world ring topology help, or is graph topology overhead with no benefit? |
| Single hop (K=1) | Does multi-hop add anything? Riskiest mechanism per Gumbel gradient-decay literature; this is THE ablation that tests our novelty claim. |
| Argmax instead of Gumbel-STE | Does stochastic routing help over deterministic? |
| No decay gate (additive write) | Does state-magnitude bounding actually help? |
| No load-balance loss | Replicate the prior plateau — validates the fix is load-bearing. |
| No dead-code revival | Is revival doing real work or vestigial? |
| Llama-3.2-3B backbone (size up) | Does the architecture scale with backbone capacity? |

## Recommended minimum viable evaluation set

To ship a credible result without spending six months on baselines:

**5 external baselines + our system + 4 ablations:**

External:
1. Frozen Llama-1B zero-shot (sanity floor)
2. Long-context Llama-1B at 8K (more-context baseline)
3. RAG with e5-base + Llama-1B (production floor; quantifies the gap)
4. RMT-1B on Llama backbone (embarrassing-if-we-lose comparator)
5. Memorizing Transformers small (closest external-memory cousin)

Our system + ablations:
6. Full architecture (proposed system)
7. No graph (full attention over N concepts)
8. Single hop (K=1)
9. No load-balance loss (replicates collapse)
10. No write-side gradient (replicates the pre-fix-#4 Phase-2 state)

## Recommended benchmark suite

**4 benchmarks covering different memory profiles:**

| Benchmark | What it tests | Expected outcome |
|-----------|---------------|-------------------|
| BABILong | Needle-in-haystack across long context | RMT wins; we plausibly lose. Important as sanity floor. |
| NarrativeQA | Holistic narrative QA over 8K-token docs | RAG wins; quantify our gap. |
| MemoryAgentBench (CR-MH subset) | Cross-session multi-hop with conflict | Everyone <6%. Our shot. |
| LongMemEval (knowledge-update subset) | Multi-session factual updates | Long-context LLMs drop ~30%; RAG can't auto-fix. Our other shot. |

## Implementation priority order

1. **First**: RMT-1B baseline. ~50 LOC. Cheapest comparator and biggest
   risk-reducer. If we lose to RMT, we save months of pointless scaling.
2. **Second**: Ablations of our own system (single hop, no graph). These
   are config-flag changes — running them while the main training is
   in flight just requires the GPU bandwidth for additional runs.
3. **Third**: RAG baseline. Production-grade but well-trodden — should
   take a long afternoon to set up with sentence-transformers + faiss.
4. **Fourth**: Memorizing Transformers. Most expensive baseline to
   build right; defer until the cheaper ones are in.
5. **Fifth**: Larimar — only do this if the early ablations show we
   have something to defend beyond what Larimar does.

## Honest narrative shape

After all the research:

> "Existing learned-memory architectures (DNC, RMT, Titans, Mamba) and
> retrieval baselines (RAG, Memorizing Transformers) all underperform
> RAG on classical long-context QA at the Llama-1B-class budget. We
> propose a learned graph-traversal memory side-car that, while not
> designed to compete with RAG on classical recall, demonstrates [N]×
> improvement over RAG and RMT on the cross-session multi-hop reasoning
> subset of MemoryAgentBench (CR-MH), where all existing methods score
> <6%. The architecture's small-world ring topology + multi-hop
> Gumbel-STE traversal enables differentiable composition of concepts
> across sessions that retrieval-based methods cannot perform."

This is publishable IF the CR-MH or LongMemEval numbers come out.
If they don't — if our CR-MH is also <6% — we have a negative
result, which is also publishable as "this combination doesn't work
either." Either outcome is a contribution; we just need to commit to
running the comparison.

## Cautionary notes

1. **Don't run a full RAG corpus that's larger than our 2M-float
   memory by 30×.** That's not a fair comparison. To compare honestly,
   either restrict the RAG corpus to a comparable bit-budget OR
   acknowledge upfront that RAG wins on raw capacity and our pitch
   is different.
2. **Don't cherry-pick benchmarks where RMT also wins easily.** If RMT
   beats RAG on BABILong with ~zero parameters, we're not actually
   pitching against RAG — we're pitching against RMT. Be explicit.
3. **The DNC failure-to-scale is the strongest base-rate evidence
   against us.** Articulate explicitly why we believe we won't suffer
   the same fate. Honest answer at this stage: we don't know yet.
4. **Multi-hop Gumbel-STE through 8 hops is the riskiest mechanism.**
   The single-hop ablation directly tests whether the multi-hop story
   carries any weight.

## References

- RMT: https://arxiv.org/abs/2207.06881
- Memorizing Transformers: https://arxiv.org/abs/2203.08913
- RETRO: https://arxiv.org/abs/2112.04426
- Titans: https://arxiv.org/abs/2501.00663
- Mamba: https://arxiv.org/abs/2312.00752
- DNC: https://www.nature.com/articles/nature20101
- Larimar: https://arxiv.org/html/2403.11901v2
- ChatQA-1.5: https://arxiv.org/html/2401.10225v5
- MemoryAgentBench: https://github.com/hust-ai-hyz/memoryagentbench
- LongMemEval: https://arxiv.org/abs/2410.10813v2
- DOS-RAG (simple-RAG-beats-fancy): https://arxiv.org/html/2506.03989v2
- MemoryBench (RAG-dominates-survey): https://arxiv.org/abs/2510.17281
