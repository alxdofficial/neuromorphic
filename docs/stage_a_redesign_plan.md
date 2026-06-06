# Stage-A Redesign Plan (2026-06-04)

Grounded in `docs/memory_literature_dossier.md`. Goal: an external memory on a **frozen Llama**
that **stores and retrieves by key** (the addressing wall), built toward two value props:
(1) **fixed-footprint memory over unbounded context**, (2) **test-time adaptation**.

## The three fixes every stage carries (why the old setup tied shuffle)
1. **Explicit key/value SPLIT** — each write emits a `key` (address, from the cue) and a `value`
   (payload, the answer); read = `query·key → value`. Address decoupled from the answer gradient.
2. **Co-adapted reader** — Llama base frozen; a small **cross-attention reader** (LongMem/Flamingo
   style, gate init 0) learns to pull from memory. The reader and the memory-writer learn *together*.
3. **Reconstruction objective, closed-book** — the target is **absent** from Llama's input, so it can
   only come from memory (EMAT key→value / ICAE whole-passage). "Reconstruction IS the task," no aux loss.

## Shared building blocks (build once, reused by all stages)
- **`KVMemory` interface:** every mechanism's write returns `keys [B,M,d]`, `values [B,M,d]`; a shared
  read `q·k→v`. Per-mechanism K/V heads:
  - **graph** — un-fuse `GraphV6FactBuilder`: `key = head(src+rel/state)`, `value = head(dst/state)`
    (the native relational triple `(subj,pred)→obj`; recovers the old v5.3/v5.4 split).
  - **MT** — native (stores attention K,V; kNN read). The reference/control.
  - **vqvae → DKVB** — split codebook into a **frozen key-codebook** (VQ-init) + **learnable
    value-codebook**.
  - **mamba → associative matrix** — matrix state `S += φ(k_t)⊗v_t`, read `q·S` (Based/Infini form).
  - **slot** — two heads per slot (long shot; needs EMAT key-AE to tie slot-keys to the query).
- **Cross-attention reader:** gated cross-attn block at one/few Llama layers; `{W_q,W_o,g}` (+opt
  `W_k,W_v`); `g` init 0 (no-op start). Memory is OUT of the token sequence.
  - **Injection LAYER is a swept knob** (papers show it matters): the query into memory is formed from
    the hidden state at the injection layer, so it must be (a) **late enough** that the
    representation is semantically rich (good address), (b) **early enough** that layers remain to
    integrate the retrieved value. **Memorizing Transformers used a single UPPER-MIDDLE layer (9/12) and
    found earlier/later worse** → default ≈ layer 10–12 of Llama-3.2-1B's 16; sweep it; consider a few
    layers (Memory-Layers / Infini-attention style) rather than one. (Distinct from our old graph
    *additive* mid-inject that failed with a FROZEN reader — a TRAINABLE gated cross-attn reader at a
    tuned layer is a different animal.)
- **Controls (the metric, every stage):** **REAL** (correct memory) vs **SHUF** (rolled = a different
  example's memory) vs **OFF** (zero memory). Success = **REAL ≫ SHUF** (content-specific), not just
  REAL < OFF.

---

## STAGE 1 — EMAT: key→value retrieval (does ADDRESSING work at all?)

### 1a — MQAR mechanism gate (fast, NO Llama)
- **Objective:** Multi-Query Associative Recall — stream **N (key→value) pairs**, then query the keys;
  emit the bound values. **Sweep N** {8,16,32,64,128,…} → capacity curve per mechanism.
- **Data:** **single, REAL vocab tokens drawn from a pool of common words, in RANDOM key→value pairings**
  (`table→seventeen`, `ocean→purple`). NOT multi-token coined gibberish — BPE shreds those into subword
  soup whose pooled embedding is off-distribution ("not in language space"). Shortcut-proofing comes from
  the **random pairing** (no semantic reason for the binding → must be stored), not from gibberish; and
  since Llama only EMBEDS at the gate (never decodes), there's no prior to leak, so even entity-like real
  tokens are fine. This is the literature-standard MQAR. (Coined multi-token entities belong to the
  LLM-integrated stages 1b/2, in real prose — not the bare-vector gate.)
- **Setup (KEY POINT — Llama embeds, but does NOT read):** each token is embedded by **Llama's FROZEN
  embedding table** → real vectors in Llama's 2048-d space (so the mechanism is tested in Llama's actual,
  possibly anisotropic, token geometry — that's a feature, it surfaces addressing-in-real-space issues
  early). The mechanism's K/V-split write stores `(Kᵢ,Vᵢ)=(key_head(key_emb), val_head(val_emb))`; read
  matches `q=query_head(query_emb)` against `{Kᵢ}` → retrieve `V`; a **custom MLP head** (NOT Llama) maps
  `V → logits over the value vocab` (CE). So Llama contributes ONLY its embedding table; the *reader/decoder*
  is the custom MLP — which keeps "can the mechanism store & retrieve" cleanly separated from "can Llama
  read it." Trainable = memory + K/V heads + MLP head; Frozen = Llama's embedding table.
  Also run **graph two ways**: emergent (free-attn read, no split) vs K/V-split — does the graph
  self-organize addressing, or does it need the split?
- **Loss:** CE on the value token given the query key.
- **Trainable:** memory write module + K/V heads + the small read head.   **Frozen:** nothing (synthetic toy).
- **Success / gate:** REAL ≫ SHUF; report **pairs-at-≥95%-recall** per mechanism. Mechanisms that can't
  pass MQAR are dead for addressing (expected: MT/graph-split/VQ-DKVB/Mamba-matrix pass; slot & emergent
  variants are the question marks).

### 1b — EMAT key→value with frozen Llama + reader (the real test)
- **Objective:** **closed-book key→value.** Write a passage's facts into the K/V memory; given a key
  (the question), **reconstruct the answer span from memory** — the answer is absent from Llama's input.
- **Setup:** the mechanism(s) that passed 1a; **frozen Llama + cross-attention reader**. Data = our
  coined-entity passages **restructured as multiple (question → guaranteed-extractive-answer-span) QA
  pairs per passage** (recurring associations; shortcut-proof).
- **Loss (reconstruction, no aux):**
  - (i) CE on the **answer span**, regenerated from memory through the reader; **plus**
  - (ii) **EMAT key/value auto-encoding** — the `key` must reconstruct the **question**, the `value`
    must reconstruct the **answer**. This forces each half to encode its content (addressable key,
    retrievable value), not let an entangled blob fake the end-task.
- **Trainable:** memory encoder (write + K/V heads) + cross-attention reader (+ gate `g`).
  **Frozen:** all Llama base weights.
- **Success:** **REAL ≫ SHUF ≫ OFF** on the answer span; the linear probe now reads the binding
  (>> the 0.30 floor, toward the 0.625 oracle). This is the result that proves the addressing fix works.

---

## STAGE 2 — ICAE: whole-passage reconstruction + compression (STORAGE + fixed footprint)

### 2a — whole-passage reconstruction at real compression
- **Objective:** encode the whole passage into a **fixed, compressed** memory (slots << tokens), then
  **regenerate the whole passage from memory** ([AE] token; passage absent). Tests faithful storage
  *under compression* (not the ratio-1 dump).
- **Setup:** same K/V memory + frozen Llama + reader; **sweep compression** (start ~4–8×).
- **Loss:** ICAE-style passage-reconstruction CE; optionally multi-tasked with the Stage-1 key→value
  head (store everything *and* retrieve by key).
- **Trainable:** memory encoder + reader.   **Frozen:** Llama.
- **Success:** passage reconstructable from a *compressed* memory; key→value recall survives compression.

### 2b — streaming long-horizon at fixed footprint (value prop #1)
- **Objective:** stream **many** passages exceeding the context window into a **fixed-size** memory;
  retrieve a fact from far back (scrolled out of attention). Measures **retrieval vs stream length at
  constant footprint** — the actual value prop.
- **Setup:** incremental writes over a long stream; fixed memory; frozen Llama + reader.
- **Loss:** key→value reconstruction of a far-back fact.   **Trainable:** encoder + reader. **Frozen:** Llama.
- **Success:** recall stays high as stream length grows while memory size stays fixed (vs an O(n) KV-cache
  baseline that grows).

---

## STAGE 3 (preview) — test-time adaptation (value prop #2)
Writes happen **at inference, no gradient through the write** (fast-weights / GRACE / Larimar). Test:
a preference/fact written early in a session changes behavior later in the same session. Out of scope
for the immediate build; the K/V + reader design is chosen to support it.

---

## Explicitly DROPPED
- Ratio-1 **whole-passage-recon-only** (the trivial copy/dump — proven for mamba 0.046 vs 6.27, no addressing).
- The **one-shot short-passage probe** with a frozen-decoder + separately-trained reader (no success precedent).
- **slot / mamba as-is** for addressing (literature gaps) — only their K/V-split forms are carried.

## SOLUTION MENU — everything the research surfaced (fallback list if MQAR + Stage 2 don't fix it)
`[✓]` = chosen for the next runs; everything else is a documented fallback to reach for.

**A. Addressing structure (the core lever — how key→value binding is built):**
- `[✓ NOW]` Explicit **key/value SPLIT** (separate key + value vector per slot; **both heads trainable**).
  Immediate change (graph un-fuse). NOTE: the graph already half-decouples by construction — key head reads
  `src+rel` (cue), value head reads `dst` (answer); the answer never feeds the key head's input.
- `[✓ as 1a ablation knob — NOT default-on first]` **Decouple address from content** — freeze / stop-grad
  the keys so the answer gradient can't drift the address (GRACE, DKVB). Test split-only vs split+frozen-keys
  in the MQAR gate; flip on only if split-only ties. (Dossier's strongest signal → likely needed, but verify.)
- `[✓ later refinement]` **Hard read + OFF/defer path** (GRACE ε-ball) — a shuffled key physically can't
  fire. Relevant for the noisy real task (1b); the clean MQAR read can stay standard `q·k→v` at first.
- `[✓ Mamba]` **Associative matrix / delta-rule** `S += φ(k)⊗v`, read `q·S` (Based, Infini-attention, ARMT, Titans).
- `[✓ VQ]` **Discrete VQ codebook KV** with frozen keys + learned values (DKVB, GRACE).
- **kNN / external index** over content-derived keys (Memorizing-T, LongMem, kNN-LM, CAMELoT).
- **Product-key factorized keys** for huge capacity (PKM, Memory Layers at Scale).
- **Closed-form least-squares write** `M=W0†Z`, training-free at inference (Larimar).
- **Modern Hopfield / matrix associative memory** (untried alternative).

**B. Reader / LLM integration (how Llama consumes memory — co-adaptation is non-negotiable):**
- `[✓]` **Dedicated gated cross-attention reader** (LongMem/Flamingo), Llama fully frozen; **gate init 0**.
- `[✓]` **Injection at upper-middle layer** (~12/16); sweep it; single vs a few layers.
- `[✓ detail]` **L2-normalize keys + queries** (Memorizing-T staleness fix).
- **Reader-LoRA** (LoRA on the attention read path) — lighter alt to a new module.
- **LoRA-all** on Llama — diffuse; rejected as primary.
- **Full fine-tune of the decoder** (Gisting, AutoCompressor) — heaviest; departs from frozen thesis.
- **Training-free**: Llama's own attention reads an explicit index (CAMELoT, kNN-LM).

**C. Training objective (what forces the binding INTO memory):**
- `[✓ Stage 1b]` **EMAT key→value reconstruction + key/value AUTO-ENCODING** (key⇒question, value⇒answer).
- `[✓ Stage 2]` **ICAE whole-passage reconstruction** ([AE] token).
- `[✓ always]` **Closed-book** — the target is absent from Llama's input → must come from memory.
- `[✓ Stage 1a]` **MQAR / associative recall** (the diagnostic gate).
- **Next-token LM with memory in-loop** (PKM/Mamba) — for the integrated/streaming setting.
- **Contrastive / retrieval-alignment loss** — aux-loss fallback, deprioritized (aux-loss aversion).
- **Hidden-state matching / JEPA** — embedding-space target (tried earlier).

**D. Task / data regime (make memory NECESSARY + shortcut-proof):**
- `[✓ 1a]` **MQAR: single real common tokens, random pairings** (not coined gibberish — OOD embeddings).
- `[✓ 1b]` **Multi-QA per passage, guaranteed extractive spans** (recurring associations).
- `[✓ 2a]` **Real compression** (slots << tokens).
- `[✓ 2b]` **Long-horizon streaming** beyond the context window (value prop #1).
- `[✓ 3]` **Test-time writes, no gradient** (value prop #2).
- `[✓ have it]` **Coined / unguessable answers** (no semantic shortcut).
- **Scale up** — more recurring associations / steps / tokens (the papers' 100B–1T regime).

**E. Anti-collapse / stability:**
- `[✓ subsumes most]` **Frozen-key / learned-value split** (structural anti-collapse).
- Stochastic slot init + GRU + iteration (Slot canonical); VQ dead-code revival + load-balance;
  σReparam / QK-Norm / LayerScale-init (rank collapse); norm control on memory entering the read;
  diversity losses (aux fallback).

**F. Bigger pivots (if the frozen side-car approach fundamentally fails):**
- **Co-train enc/dec** (Larimar) — buy addressability with a trained encoder+decoder (departs from frozen Llama).
- **Memory Layers at Scale** — train a model from scratch with memory layers (not a side-car).
- **Recurrent fixed-state backbone** (Mamba / RMT / Titans as the model itself, not an add-on).

**G. Cheap diagnostic (do anytime):** bolt a HARD external `(question-key, answer-value)` index onto the same
frozen Llama, change nothing else → if the probe jumps toward the 0.625 oracle, the wall was the missing index.

## First code to build (CPU only, no GPU — pending GPU free)
1. `KVMemory` shared interface + per-mechanism K/V heads (graph un-fuse first; it's the smallest change).
2. MQAR single-token data generator + Stage-1a harness (tiny read head, REAL/SHUF/OFF metric).
3. Cross-attention reader module (gated, init-0) for Stage 1b.
