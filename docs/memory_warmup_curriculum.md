# Memory warmup curriculum — Stage A design + build plan

**Branch:** `memory-warmup`. **Status:** design LOCKED 2026-06-02, building now.

**Why.** v2.1 diagnosis: the side-car memory is a *generic behavioral prior, not stored content*
(zeroing memory ≈ no effect; near rank-1; same structure across inputs). Suspected cause: a
**write/read chicken-and-egg** — the write doesn't know what to store → the read learns to ignore it →
no gradient reaches the write → nothing trains. Cure: a **curriculum** that first makes write+read
mutually useful on the easiest possible storage task, then adds the hard parts.

---

## 1. Curriculum — one task, five knobs

Same task throughout — **write a passage, address a fact by a linguistic key, retrieve its value** —
dialed from trivial (A) to full QA (C):

| knob | **Stage A** (storage+addressing) | **Stage B** | **Stage C** (= our QA) |
|---|---|---|---|
| key | canonical descriptor (`"X's birthplace"`) | + paraphrases | paraphrased / multi-hop question |
| value | short verbatim span (multi-token) | short span | abstractive answer |
| compression | none (passage ≫ #facts) | 2–4× | ~64× |
| reasoning | single direct lookup | + light distractors | multi-hop |
| **read/decode** | shared cross-attn read head → linear (no Llama) | begin routing into Llama | frozen Llama + LoRA |

Stage A is the trivial corner of all five, so a failure there is unambiguous (the substrate can't store,
independent of compression/Llama).

---

## 2. Stage A — the mechanism

**Pipeline (no Llama anywhere, no autoregression):**
```
passage (novel entities, short)  --write (existing encoder.streaming_write)-->  memory  [B, M, d]
for each value slot k:
    query_k = cond(key) + pos_emb(k)                  # key-conditioning + position
    r_k     = CrossAttention(query_k, K=V=memory)     # the READ
    logits_k= Linear(r_k)                              # tiny classifier head, NOT an LM
loss = Σ_k CE(logits_k, value_token_k)                 # + EOS slot for length
```

- **Multi-token short-span values** (a name/place/date — a few tokens), **not** single-token (that's
  classification, not language) and **not** full sentences (that re-introduces paraphrase → needs a
  judge). Verbatim spans keep exact-match meaningful.
- **Non-autoregressive:** each slot's query is `(key, position)` — *not* previous tokens — so there's no
  language model to "bolt on" and nothing to cheat with. Length handled by an **EOS/PAD** token.
  (AR upgrade later = a tiny decoder produces the query from `key + tokens-so-far`; *same read*.)
- **Novel / unguessable entities + no compression** is what makes the memory **load-bearing**: a tiny
  cross-attn+linear head with no priors *cannot* emit the value unless it was read from memory. The
  **zero-memory control is the proof** (zeroing memory must crater recall).

**Objective (minimal).** Token-recon CE only. **Contrastive PARKED** (add as its own term only if
multi-pair addressing fails). **VICReg anti-collapse = silent insurance**, off by default, switched on
only if the effective-rank monitor shows collapse. All warmup terms anneal out by Stage C.

---

## 3. The read across arms — isolates WRITE, plus a graph self-test

The baselines (flat/continuous/MT/mamba) have **no query-driven read** — they just *prepend* their
`memory [M,d]` and let Llama attend. So for Stage A we give **every arm the SAME generic read head**
(the cross-attention above), **separate weights per arm**. Because the only thing differing across arms
is *what's in `memory`*, this is a clean **write-quality** comparison: *does each arm's memory contain
the fact, retrievably?*

graph_v6 additionally runs with its **native** `GraphV6FactReader`. The native-vs-shared comparison
localizes the last question:

| shared read | native read | conclusion |
|---|---|---|
| fail | fail | the **write** doesn't store it (memory problem; all arms) |
| pass | fail | the **graph read** is broken |
| pass | pass | the graph mechanism works |

---

## 4. Data
- **Synthetic generator** (new, self-contained): items = `{owner entity, [(key-phrase, value-span)…],
  short passage}`. Novel entities (unguessable), multi-token short-span values, **#pairs = the difficulty
  axis**. Sanity: a no-memory baseline must score ≈0 (else values are guessable → fix the generator).
- **WikiBio** (`michaelauli/wiki_bio`, CC BY-SA 3.0) — organic Wikipedia bio passage + infobox
  `attribute→value`; the natural-text companion (deferred — synthetic first).
- **zsRE / KnowEdit** (`zjunlp/KnowEdit`, MIT) — native key paraphrases + locality distractors for the
  addressing/no-leak stress (Stage B).

## 5. Metrics, gate, controls (NO LLM judge — verbatim, exact-match)
- **Primary:** exact-match recall of the value (alias list for benign variants).
- **The gate — recall vs. #KV-pairs elbow.** Capacity ∝ params (Nichani 2024), so with 274,944 floats the
  ceiling is large → an **early elbow (single-digit pairs) indicts the write/read MODULES, not the
  bottleneck.** This is the clean "modules vs compression" separation.
- **Controls (attack the v2.1 evidence directly):** (1) zero-memory → recall must crater; (2) shuffle the
  stored bindings → recall must follow the shuffle; (3) retrieved-vector **effective rank** input-dependent
  (watch the singular-value spectrum, not just the loss).
- **Advance to B** only when: high recall on uncompressed few-pair storage AND rank rises AND controls
  behave. A Stage-A *failure* is itself a clean, deep result.

## 6. Stage B / C (sketch)
- **B:** same task; turn up 2–4× compression, multi-pair pick-a-key, paraphrases (zsRE), light distractors
  (KnowEdit locality), multi-token values; add contrastive term *if needed*; route some gradient into Llama.
- **C:** the QA we already run (frozen Llama + LoRA, ~64×, generative + judge), warm-started write+read.

---

## 7. Build plan (file-level)

**Phase 1 — data.**
- `src/repr_learning/data_stage_a.py` (new): `StageAKVDataset` + generator. Emits per item the passage
  token ids, the list of `(key_ids, value_ids)` pairs, owner entity. Novel-entity pool, single→multi-token
  value pool, `n_pairs` knob, short passage (≤256 tok). Collate into batches.
- Smoke: build a few items; verify a no-memory readout scores ≈0 (unguessable check).

**Phase 2 — model.**
- `src/repr_learning/stage_a_read.py` (new): `SharedKVReadHead` — cross-attention `(key_cond + pos) →
  memory[M,d] → Linear→vocab`, multi-slot + EOS. A `StageAModel` wrapper: `arm.streaming_write(passage)
  → memory`; choose read ∈ {`shared`, `native` (graph only)}; returns per-slot logits. Reuses the existing
  `*BaselineEncoder.streaming_write` (write) unchanged; reuses `GraphV6FactReader` for native.
- Zero-memory + shuffle-bindings + effective-rank hooks live here (flags on the forward).

**Phase 3 — trainer + sweep.**
- `scripts/repr_learning/train_stage_a.py` (new): write passage → memory → read by each key → CE on value
  tokens; exact-match recall eval; the three controls; checkpoint by val recall.
- `scripts/repr_learning/sweep_stage_a.py` (new): loop `n_pairs ∈ {1,2,4,8,16,32}` × arm ∈ {graph_v6
  (shared+native), continuous, flat, mamba, MT} → recall-vs-#pairs curve + effective-rank; plot the elbow.

**Defaults:** read head = 1–2 layer cross-attn, d≈256; Llama tokenizer (Stage-C transfer); single passage,
write-once-query-many; VICReg off (monitored); arms = graph_v6 first, then continuous.

---

## 8. Research grounding (see session notes for full citations)
- Reconstruct→QA two-stage, frozen backbone: 500xCompressor (Li 2024), ICAE (Ge 2023), Larimar (Das 2024).
- Names our failure: Deng 2024 "A Silver Bullet or a Compromise?" (content-free compressed memory = the
  canonical synthetic-recall failure; fix = fine-grained autoencoding supervision).
- Text-in / retrieve-vector-out, supervised directly (not via the LM head): Entities-as-Experts (Févry 2020).
- Warmup task + metric: MQAR/Zoology (Arora 2023), Based (Arora 2024), Repeat-After-Me (Jelassi 2024) —
  loss on the value, exact-match recall, #pairs as the axis.
- Output queries (no LM decoder): Perceiver IO (Jaegle 2021), Set-Transformer PMA (Lee 2019).
- Anti-collapse: VICReg (Bardes 2021), Barlow Twins (Zbontar 2021); dimensional collapse is real even under
  contrastive (Jing 2021), so an explicit variance/covariance term is load-bearing for our rank-1.

## 9. Open knobs (tune while building, not benchmark)
VICReg weights (raise variance first) / whether needed; read-head depth; value-head vocab (closed→open);
KV-style vs thin-embedding memory carrier (500xCompressor); optional xRAG-style write warm-start.
