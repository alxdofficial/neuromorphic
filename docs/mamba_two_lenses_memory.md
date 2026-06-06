# Two lenses on Mamba (linear attention) — and what they require of a "compress-and-recall" memory

*Working note for supervisor meeting. Context: we're building a trainable memory side-car for a frozen Llama — an encoder compresses a window of tokens into a small memory that the LM reads to answer questions.*

---

## The tension we're actually trying to resolve

The memory has to do two things that pull against each other:

- **Compress** — summarize a window into far fewer units than there are tokens (the thesis: a structured/compositional memory beats a flat bank at the same size).
- **Retrieve verbatim** — recover a *specific* stored item (a fact, a relation, a value) on demand, not just "the gist."

Our diagnosis this cycle is that the current encoders do the first but not the second — they keep **gist** (which entities are present) but not **bindings** (which key went with which value). Empirically: with a *ground-truth* memory the read head learns to 1.0 in ~250 steps, but with any of our 5 real encoders it sits at **chance**, with `REAL ≈ SHUF` (scrambling the memory doesn't change the answer) — i.e. the memory is not being *used*. The harness, optimizer, precision, and read head were all ruled out; the failure is the **write**.

Mamba / linear attention is the cleanest model to dissect *why*, because the **same equations** can be read two completely different ways.

---

## Lens 1 — Efficiency (the usual motivation)

Standard attention keeps every past token (the KV cache grows with length) and costs O(N²). Linear attention:

1. starts from attention `out = softmax(q·kᵀ)·v`,
2. **drops the softmax**, replacing the query–key similarity with a *separable* feature dot product `φ(q)·φ(k)`,
3. which lets you rewrite "attend over all past tokens" as a **linear recurrence with a fixed-size state**:
   `Sₜ = Sₜ₋₁ + φ(kₜ)·vₜᵀ`, computable by a **parallel/associative scan**.

Two wins — a **fixed-size state** (compression) and a **parallelizable scan** (speed) — and they're really *one* win: the algebra that lets you carry a fixed `S` forward is the same algebra that turns it into a scannable recurrence. (Linear transformers: Katharopoulos et al. 2020, arXiv:2006.16236. Mamba = Gu & Dao 2023, arXiv:2312.00752, a *selective* SSM in this family.)

`φ` is just the *replacement similarity* once softmax is gone — e.g. `elu(x)+1` — chosen to keep the implied attention weights **non-negative** like softmax did. It is not learned; the learning is in the k/q/v projections.

---

## Lens 2 — Memory (the same object, read differently)

Write the fixed state out: `S = Σₜ φ(kₜ)·vₜᵀ`. That is **literally an associative key–value store**:

- each token writes an **outer product** `φ(kₜ) ⊗ vₜ` — the value *stamped with its key*;
- reading with a query is a matvec `S·φ(q) = Σₜ (φ(q)·φ(kₜ))·vₜ` — matching keys light up, the rest stay quiet.

So the **outer product *is* the "bind"** — the only way to put a `(key → value)` pair into a fixed matrix so a query can pull the right value back out. The read is exactly content-addressed retrieval.

The **naive additive** write has a fatal flaw: pairs **superpose**, so (a) once you store more than the state can hold, cross-talk drowns the signal, and (b) you can **never overwrite** — writing the same key twice returns the *sum* of the two values. The **delta rule** fixes both. Before writing key `k`, read what's currently there and write only the *residual*:

```
Sₜ = Sₜ₋₁ + β·(vₜ − Sₜ₋₁·φ(kₜ))·φ(kₜ)ᵀ
```

This is **one step of online gradient descent on ‖S·φ(k) − v‖²** — the memory is a tiny linear model trained *per token* to "return v when queried with k." It overwrites in place and minimizes interference. (Widrow–Hoff 1960; fast-weight programmers, Schlag et al. 2021, arXiv:2102.11174; DeltaNet, Yang et al. 2024, arXiv:2406.06484.)

---

## The punchline: same equations, two readings

`Sₜ = Sₜ₋₁ + φ(k)·vᵀ` is, in one breath:

- a **fixed-state linear recurrence you can scan** (efficiency), **and**
- an **associative store you bind into and query** (memory).

The field invented it for efficiency; it *happens to be* an associative memory. The delta rule is the single place the two lenses diverge — it's a smarter write *rule* (better recall) at a small cost to the scan's simplicity.

---

## What the memory lens reveals: the checklist for *compress-AND-verbatim*

Reading Mamba as a memory gives a requirements list for **any** mechanism (ours included) that must both compress and recall verbatim:

1. **Fixed / compressed state.** You commit to storing far fewer units than tokens. *(This is the compression.)*
2. **The binding must be installed by the WRITE, as a key-indexed structure** (outer product / discrete code) — never recovered from a pool afterward. A pooled/averaged state is **provably membership-only**: an average over a set encodes *what is present* but not *which key paired with which value* (the Set-Transformer / Perceiver PMA result, Lee et al. 2019, arXiv:1810.00825). **This theorem is exactly our `REAL == SHUF` failure.**
3. **Error-correcting write** (the delta rule), so colliding or updated keys overwrite cleanly instead of superposing into mush.
4. **Distinct, *reachable* addresses.** Keys must be near-orthogonal **and** live in a space the read query can hit. In Mamba the query is the *next token in the same stream*, so it's reachable by construction. For an **external** query (Llama's question), you need a *shared address space* (e.g. a fixed code book) or a query projection *co-trained* with the keys.
5. **Capacity ≈ state dimension** (the Zoology law, Arora et al. 2023, arXiv:2312.04927). Verbatim recall scales with the *size of the state*, not the number of slots you draw. So compression and verbatim recall genuinely trade off, **bounded by how large the addressable state is** — a bigger state or a richer feature map (Based, arXiv:2402.18668) buys more recoverable items.

---

## Where our five mechanisms stand against the checklist

| mechanism | 1 compress | 2 bind-in-write | 3 error-correct | 4 reachable address | what it needs |
|---|:--:|:--:|:--:|:--:|---|
| **Mamba / DeltaNet** | ✅ | ✅ (outer product) | ✅ (delta) | ⚠️ keys are *contextualized* | a query that reaches its keys (in-stream native read, or a co-trained external query) — it **is** the reference |
| **VQ / DKVB** | ✅ | code = address | partial | ✅ a code book is a ready-made *shared* address space | **hard snap** (not soft routing) + anti-collapse |
| **Slot Attention** | ✅ | ❌ update is an **average** | ❌ | ❌ keys == values | replace the averaging update with a binding write — it stops being slot attention |
| **Our graph** | ✅ | ❌ **pools twice** (window→slot cross-attn; soft-pointer endpoint materialization) | ❌ gated blend | ❌ soft-pointer address | replace the two *averages* with *binds*, and the blend with an error-correcting write |

---

## One-line takeaway

**Mamba is, simultaneously, a fixed-size scan (efficiency) and an associative key–value store (memory).** Read as a memory, it spells out exactly what *any* compress-and-recall mechanism must do: **install the binding in the write — not recover it from a pool — correct errors so writes don't smear, and give keys distinct addresses the reader can reach**, with verbatim capacity capped at the state dimension. Our encoders compress but **average where they should bind**. That single substitution — *blend → bind*, plus an error-correcting write — is what would turn a gist-summarizer into a verbatim-retrievable memory **while keeping the compression**.
