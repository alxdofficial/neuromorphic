# Motivation Document — Neuromorphic AI Prototype (Neural Memory + Neuromodulators)

> *Aligned with spec v2.0. For implementation details, see `spec/SPECIFICATION_v2.md`.*

## What we're building
We’re building a **next-generation sequence model** that behaves less like a static transformer and more like a **brain-inspired system** that can **adapt while it runs**.

At a high level, this model should:
- be **strong at general tasks** (language first, multimodal later),
- be **intrinsically multimodal** at the token interface level (text/image/audio tokens can all flow through the same backbone),
- **self-improve during use** without finetuning or retraining,
- support **agent-like behavior** with persistent, compressible memory,
- be designed in a way that can plausibly be **mapped to efficient silicon** in the long term.

This prototype is the first proof of concept: show the core mechanism works on language, then demonstrate online adaptation and stability. (The spec defines training Phases A–E; this document's "Phase 1" refers to the overall language-first project phase, not a specific training phase.)

---

## Why this is worth building (the problem)
Modern LLMs are extremely capable, but they have structural limitations that make them a poor match for how humans learn and how future consumer AI should behave:

### 1) LLMs are mostly “static”
Once deployed, an LLM’s weights don’t change. If you want it to improve:
- you finetune,
- you retrain,
- or you bolt on retrieval (RAG).

This creates a gap between:
- **short-term learning** (what the user just told you),
- **long-term personalization** (user preferences, style, habits),
- **robust agent memory** (what you did yesterday, what tools worked, what failed).

### 2) RAG is not the same as learning
Retrieval can fetch text, but it doesn’t automatically:
- integrate knowledge into the model’s internal computations,
- adapt behavior continuously,
- protect against repeated mistakes,
- compress and generalize experience.

### 3) Transformers are expensive for long context
Attention-based models become costly as context grows.
If we want “AI embedded everywhere” and eventually baked into silicon, we want:
- predictable compute,
- more local, update-friendly operations,
- and mechanisms that scale like **state** rather than like quadratic attention.

### 4) We want “agents” that get better by doing
A practical agent should:
- remember what it has done,
- improve tool-use patterns,
- adapt to domain workflows,
- and refine behavior over time without requiring model updates in the cloud.

---

## The core idea (what makes this different)
The model is built around **Neural Memory Layers (NMLs)** — recurrent layers that include an **online-updatable memory system**.

Instead of relying on unbounded attention over long token histories, the model maintains **multiple memory systems** that evolve during use. This is inspired by how brains combine:
- long-term learned structure ("genetics" / stable synapses),
- short-term precise context (working memory),
- procedural skills (implicit memory),
- and episodic recall (explicit memory).

### Four memory types (v2 architecture)
1) **Genetic memory / slow weights (stable competence)**
- Learned during training by gradient descent.
- Encodes general language understanding and skills.
- Frozen during normal deployment.

2) **Working memory (WM)** — sliding-window attention
- Short-term precision: copies, bindings, recent tokens.
- Bounded window (W=256 tokens), no post-training evolution.

3) **Procedural memory (PM)** — fast low-rank weights
- Updates while the model runs via neuromodulated commits.
- Encodes behavioral patterns, skills, habits.
- B×L=32 instances (one per layer per block), each with its own controller.

4) **Episodic memory (EM)** — vector store
- Updates via neuromodulated writes at plasticity boundaries.
- Encodes specific facts, events, user preferences.
- B instances (one per block), each with its own controller.
- Bounded + decaying + capacity-limited to prevent drift.

---

## How “learning during use” works (Hebbian plasticity + neuromodulation)
We want brain-like “fire together, wire together” behavior, but implemented in a way that’s:
- stable,
- controllable,
- and computationally cheap.

### Eligibility traces: “what could be learned”
Every token step produces a candidate learning signal based on local activity:
- pre activity (input features),
- post activity (hidden state features).

This signal is accumulated in an **eligibility trace**, which is a buffer of “potential updates” that does *not* automatically change memory.

Think of eligibility as:  
> “this experience is eligible to be stored.”

### Neuromodulator: “should we actually store it?”
A neuromodulator network outputs low-dimensional control signals that decide:
- **when** to commit memory (write event),
- **how strongly** to write,
- **how much** to decay old memory,
- **where** to store it (slot selection via softmax-weighted top-k).

Think of neuromodulation as:  
> “store this now, because it will help.”

This is crucial because naïvely updating memory every token causes drift and instability.

---

## Why the plastic memories are low-rank / item-based (efficiency + hardware story)

**Procedural Memory (PM)** uses low-rank slots (r=8 per instance):
- each slot has a key vector, value vector, and strength scalar.
- yields **O(r·D_h)** compute per token.
- clear capacity limits (only r slots per layer per block).

**Episodic Memory (EM)** uses an item-based vector store (M=256 per block):
- each item has a key vector, value vector, and strength scalar.
- supports top-k retrieval (k_ret=4 latent tokens).
- bounded capacity with weakness-based overwriting.

This design aligns with the silicon goal:
- dot products + weighted sums are hardware-friendly,
- sparse writes at plasticity boundaries are bandwidth-friendly,
- no O(D²) matrix updates.

---

## Why neuromodulators are trained via main-loss backprop
Backprop can train the slow weights to predict tokens. Memory commit/write decisions are also trained end-to-end through the main loss:
- neuromodulator outputs (write strength, decay rate, slot selection temperature, etc.) flow through differentiable memory operations,
- the gradient from future predictions reaches the neuromodulator heads through: commit/write → PM/EM state → retrieval → layer output → logits → loss,
- this naturally learns to write only when it increases future performance, avoid harmful writes, and stay within energy/budget constraints.

This approach is simpler and more stable than RL-based alternatives, and uses only a single optimizer for all parameters.

---

## What we prove in the language-first phase
The first goal is not to beat frontier transformers.
The goal is to show a compelling "new capability" story:

1) The model can learn language normally (baseline competence with WM + scan core).
2) With PM enabled, it can:
   - store behavioral patterns and skills,
   - adapt to new local patterns without finetuning.
3) With EM enabled, it can:
   - remember session facts explicitly,
   - retrieve relevant past experiences at long delays.
4) With learned controllers (`PMNeuromodulator` and `EMNeuromodulator`), it commits/writes more intelligently than heuristics.
5) All of this happens without destabilizing core competence.

This is the investor-facing proof:
- "the model gets better as you use it" (without retraining),
- "it can be embedded and personalized efficiently."

---

## What success looks like (concrete outcomes)
### Core metrics
- Validation perplexity improves during training (LM works).
- Memory benchmarks show strong gains with PM/EM ON vs OFF.

### Online adaptation metrics
- After seeing a user profile once, the model answers profile queries later more accurately (EM).
- After repeated tool/task patterns, the model reduces mistakes within-session (PM).

### Stability metrics
- After long adaptation runs, perplexity with PM/EM OFF remains stable (core competence preserved).
- Commit/write rates stay sparse and bounded per block.

---

## Design philosophy for implementation
This prototype should prioritize:
- **stability and debuggability** over maximum novelty,
- **mechanism isolation** (clear baselines and ablations),
- **4090-friendly engineering** (TBPTT, mixed precision, single optimizer).

We intentionally reuse mature components (tokenizers, dataset streaming, AMP training) so we spend effort only on the novel parts:
- fast memory representation,
- eligibility traces,
- neuromodulated commits,
- neuromodulated memory training via main-loss backprop.

---

## Long-term direction (what this enables later)
v2 already implements key upgrades from the roadmap:
- **scan-friendly recurrence** (affine: h_t = a_t ⊙ h_{t-1} + b_t),
- **episodic memory** (per-block EM with dedicated controllers),
- **working memory** (sliding-window attention).

Future directions include:
- intrinsic multimodality (image/audio tokens),
- consolidation (distilling PM/EM into slow weights during "sleep"),
- alternative similarity/compatibility functions beyond cosine,
- silicon mapping and quantization.

The language-first phase is the foundation:
> a model that can learn normally, then adapt online in a stable, controllable way.
