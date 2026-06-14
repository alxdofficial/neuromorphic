# Graph memory — where we are, and what we learned (meeting, 2026-06-01)

## TL;DR

- **The graph led the early QA benchmark.** On the broad (easy-dominated) mix — full composite +
  HotpotQA + NarrativeQA, scoring only the *memory-load-bearing* answer tokens — the graph took the
  top spot (answer-NLL **2.637** / top-1 **49.1%**, edging Mamba 2.674). *Modest, single-seed margin.*
- **The clear lead shrinks to a tie under an honest test.** Tightening three knobs at once — **hard
  families** (multi-hop, no grammar shortcuts), **fair param-matched + tighter compression**, and a
  strict **generative + LLM-judge** metric — collapses the margin: after the read-side fix **graph_v6.1
  (13.5) ties Mamba (13.2)** as the two best memory arms, both leading the pack just above the no-memory
  floor (13.0), with flat/continuous/MT below it. But the full-context ceiling (38.2) towers over them
  all → **compression, not architecture, is the wall.**
- **Every diagnostic converges on one explanation:** the graph builds a **generic, input-invariant
  topology** and works as a **learned behavioral bias / modulator — not grounded memory.** Four
  independent measurements say so (ablation/shuffle, decode-recoverability, effective rank,
  cross-input similarity).
- **This is also an interesting result, not just a miss** (see *Future direction*): implicit
  behavioral adaptation is a real gap in current LMs (RAG / `skills.md` / `soul.md` are all *text*).
  We're **parking** that and first trying to make the memory actually store/retrieve.
- **We've isolated the bottleneck to the WRITE** — the probes show the answer isn't stored in a
  recoverable form. The fix is a **curriculum** (§6): bootstrap write+read on an easy reconstruction
  task *before* piling on compression and QA.

---

## 1. How we measure memory

### Training signal — loss on memory-load-bearing tokens

A QA score only tests *memory* if the answer couldn't have come from grammar / common-sense alone. So
wherever we can, we compute the loss **only on the answer tokens that must come from memory** — the
*memory-load-bearing* tokens — and drop templated echoes and filler.

- **How:** teacher-forced CE on the answer, masked by `answer_content_mask` (`model.py:852-865`); only
  `True` positions are scored. ("`val_recon`"/"`loss_recon`" are legacy names — this is **answer-token
  NLL**, *not* document reconstruction.)
- **composite_v1 (biographical, …):** answers carry an explicit `target_value` span, so we mask to
  *just that span* — in *"Alice is the sister of Bob"* only **Alice** is load-bearing; *"is the sister
  of Bob"* is grammar/echo and is dropped from the loss (`data_qa.py:166-175`).
- **Where we can't isolate it:** hotpot / musique / babilong answers are already pure short spans
  (every token load-bearing → all-`True`); narrative answers are **abstractive paraphrases** with no
  fixed span to mask to. Those families fall back to scoring the whole answer.

> **Consequence — families differ wildly in difficulty.** Where memory isn't strictly required
> (synthetic, templated composite subtasks: biographical / triage / knights), grammar carries the answer
> and *everyone* scores high. The early "graph wins" mix was **dominated by these easy families**
> (composite 0.5 / hotpot 0.25 / narrative 0.25; per-family val loss biographical **0.67** vs narrative
> **5.64**). That is exactly why we later **constrained to the hard families** — and that is where the
> graph's lead narrowed.

### Eval metric — the LLM judge is the headline; EM / containment are sanity

The training NLL above is a *signal*, not a verdict. For evaluation, no string metric is trustworthy on
free-form answers:

- **Exact match (EM)** is brutally strict — *"Paris, France"* ≠ *"Paris"*.
- **Containment / recall** (does the gold contain the prediction, or vice-versa) is looser but still
  string-level — blind to paraphrase, ordering, punctuation. BLEU-4, Hamming distance, etc. are all
  patches on the same string-matching wound.
- So, following current practice, the **LLM judge (Gemini-3-Flash) is our headline metric**; EM and
  containment are kept only as **fast sanity checks** that run in seconds. Anything we actually trust
  goes through the judge.

**The judge returns a rubric, not a scalar.** It sorts each answer into one enumerated failure mode —
`correct` / `correct_verbose` / `incomplete` / `partially_wrong` / `wrong_value` / `irrelevant` /
`no_answer` — and the headline % is *derived* from that. So every score doubles as a **diagnostic**
(v2.1 sweep, 336 answers/arm; the graph row here is the pre-read-fix snapshot):

| arm | correct | verbose | incomplete | partial-wrong | **wrong_value** | irrelevant | no_answer |
|---|--:|--:|--:|--:|--:|--:|--:|
| vanilla (full ctx, ceiling) | 56 | 66 | 12 | 2 | 71 | 17 | 112 |
| mamba | 39 | 1 | 9 | 0 | **255** | 31 | 1 |
| vanilla (no-mem floor) | 38 | 0 | 11 | 1 | **277** | 7 | 2 |
| continuous | 39 | 0 | 6 | 0 | **283** | 6 | 2 |
| MT | 33 | 1 | 7 | 0 | **285** | 6 | 4 |
| graph_v6 (pre-fix) | 32 | 0 | 8 | 2 | **290** | 2 | 2 |
| flat | 31 | 0 | 11 | 0 | **288** | 5 | 1 |

> Every memory arm's failures are overwhelmingly **`wrong_value`** (confident, fluent, *wrong*) — not
> `no_answer` (silence). The model isn't abstaining for lack of memory; it's confabulating a plausible
> value. That's a categorically different failure than "gave up," and only the rubric surfaces it.

**Companion concern — the waste ratio.** A passage is only a fair memory test if the answer is actually
*in* it and not buried in distractors. We track how much of the input is padding/distractor vs.
answer-bearing — a **waste ratio** — so a low score can be attributed correctly: *couldn't compress the
signal* vs. *the signal wasn't there to compress.*

**Why the training NLL can't be the verdict.** Under teacher forcing the model sees the gold answer
prefix, so it often predicts the next answer token from grammar + parametric knowledge *regardless* of
memory. Result: on answer-NLL / top-1, **every arm — the no-context floor included — clusters**, both
overall and family-by-family. Memory barely moves the NLL; the judge is what separates the arms.

![answer-NLL scoreboard — all six arms (floor included) cluster](plots/scoreboard_3d.png)
![per-family answer-NLL — memory ≈ no-context floor in every family](plots/scoreboard_per_family.png)

---

## 2. The arc

| act | when | what we did | result |
|---|---|---|---|
| **Graph leads** | ~05-25 | QA, answer-token NLL, **easy-dominated** broad mix | graph **#1** (NLL 2.637 vs Mamba 2.674; modest, single-seed) |
| **Make it honest** | ~05-28 → 06-01 | hard families **+** fair param-match **+** generative/judge | graph_v6.1 **ties Mamba**; both lead the pack, just above floor, far below ceiling |
| **Diagnose** | 06-01 | ablation + 4 probes | memory = **behavioral prior, not encoding** |

**Act 1 — graph takes the top spot (2026-05-25, commit `094906b`).** Answer-token NLL on `v1h_t4k_v3`,
mix [composite 0.5 / hotpot 0.25 / narrative 0.25]:

| variant | answer-NLL ↓ | top-1 |
|---|---:|---:|
| **graph + load-balance** | **2.637** | **49.1%** |
| recurrent (Mamba) | 2.674 | ~45% |
| continuous | 2.692 | — |
| memorizing (MT) | 2.703 | — |
| flat | 3.396 | — |
| vanilla (full ctx, ceiling) | 3.448 | — |
| vanilla (no memory, floor) | 5.115 | — |

> Graph took top NLL *and* top-1, edging Mamba by 0.037 nat / ~4 pts — but **single-seed, ~1.5σ**, so
> "modest" not "decisive." And the mix is **easy-dominated** (biographical val-loss 0.67 vs narrative
> 5.64). The thesis signal was real (cross-role node reuse 0.38; state diversity jumped 1000× once the
> mode-collapse fix landed) — this is the "graph is ahead" result, on the broad easy mix.

**Act 2 — hard families, fair param-matched (graph_v6, v2.1 sweep, 2026-06-01).** 7 arms, all matched
to 274,944-float state + ~48M memory params + identical rank-16 LoRA on a frozen Llama-3.2-1B:

| model | EM | Containment | F1 | **Judge** |
|---|---:|---:|---:|---:|
| vanilla (full ctx, **ceiling**) | 14.3 | 30.1 | 23.7 | **38.2** |
| **graph_v6.1 (read-fixed)** | 8.0 | 8.0 | — | **13.5** |
| mamba | 8.0 | 8.3 | 14.7 | **13.2** |
| vanilla (no-memory floor) | 7.7 | 7.7 | 14.1 | **13.0** |
| continuous | 8.6 | 8.6 | 14.2 | **12.5** |
| MT | 6.2 | 6.5 | 12.8 | **11.2** |
| graph_v6 (v2.1, *read-bug*) | 6.2 | 6.2 | 12.7 | **10.9** |
| flat | 6.5 | 6.5 | 13.2 | **10.9** |

> **Read the ranking, not just the floor line.** Once the read-side bug is fixed, **graph_v6.1 (13.5)
> ties Mamba (13.2)** — the two strongest memory arms — and they **lead the pack**, coming out just
> above the no-memory floor (13.0), with flat / continuous / MT *below* it. So the graph is competitive
> with the best baseline; **the architecture isn't the problem.** The real gap is to the **full-context
> ceiling (38.2)**, which towers over *every* 274,944-float memory → **compression, not architecture, is
> the wall.** (The 10.9 row is the pre-fix read-bug snapshot, not the model's real standing.)

---

## 3. The diagnostic trail that shaped today's design

Every design decision came from a probe. This is the chain:

| # | What we found | What we changed |
|---|---|---|
| 1 | **Edge-states are read-side dead** — zeroing `edge_state` changed answer F1 by ≈0 on every family. The relation payload was a concatenated input the readout learned to *ignore*. | Adopt the **"no-op-free" principle**: every stored quantity must be consumed by the read on the path to the answer (graph_v6). |
| 2 | **Message-passing = entity-pooling, not relational** — routing helped biographical (+19.5 F1, entity recall) but multi-hop ≈ +0; routing entropy ≈ uniform. | Stop treating it as a relational reasoner; focus on the read/decode side. |
| 3 | **The graph's edge was its question-conditioned *readout*, not the substrate** — on an agnostic-store probe graph ≈ baselines (near chance). | Read mechanism, not graph structure, is load-bearing. |
| 4 | **High write-rank ≠ recall** — graph writes high-rank memory (eff-rank 45–62/128) vs Mamba rank-2, yet Mamba ties → diversity isn't the problem. | Bottleneck is **read/decoder exploitation**, not write diversity. |
| 5 | **Decode probe** (tiny MLP: pooled-question-vector → pooled-answer-vector) — early graph_v5 memory was *somewhat* decodable and **better than baselines**, but a frozen Llama couldn't use it; the appended tokens are too **foreign**. | Add **LoRA fine-tuning for *all* arms** so Llama can adapt to the memory distribution (fairness mandate). |
| 6 | **Proposal mechanism showed unhealthy telemetry** (node collapse across slots, write-gate decay 0.28→0.10). | **Drop the proposal/competitive-write head.** |
| 7 | **Top-k starves gradients** (dead paths on unchosen facts; needs exploration hacks). | Use a **pure-softmax soft blend**, not hard top-k. |
| 8 | Need a principled graph-write. | Adopt **TokenGT-style write** — nodes *and* edges as typed tokens, type + instance-tag embeddings, run a transformer. |

![graph_v5 edge-state / topology diagnostic](plots/v5_topology_diagnostic.png)

---

## 4. The core finding: the memory is a behavioral bias, not stored content

After the read-side rebuild (LoRA-all + graph_v6 + the v6.1 read fixes), the deeper truth showed up.
**Four independent measurements all point the same way:**

**(a) Ablation / shuffle — the read-utilization probe.**

| condition | EM | Containment | reading |
|---|---:|---:|---|
| REAL (correct facts) | 3.1 | 3.1 | — |
| OFF (memory disabled) | 3.1 | 3.1 | **correct memory ≈ no memory** |
| SHUFFLE (another sample's facts) | 1.6 | 1.6 | **wrong memory *degrades*** |

> Injecting the *correct* facts changes nothing; injecting a *random* sample's facts makes it worse.
> The model leans on an input-derived **prior** to answer — it isn't reading stored content. (After the
> v6.1 read fix the model went further and *learned to gate the memory fully off* — REAL = OFF = SHUFFLE.)

**(b) Recoverability — is the answer even in the facts?** A linear probe + max-over-facts control:
the best of 196 fact-tokens is **0.144** cosine to the true answer vs **0.135** to a *random* answer
(signal **+0.009 ≈ noise**). **The answer is not stored in a recoverable form.**

**(c) Effective rank.** Every compressed memory is **near rank-1** (<1% of dimensions used) — including
the "working" arms.

**(d) Cross-input similarity + structure.** The graph builds a **near-identical generic topology for
every input** (two different biographies → the same stats: 71/128 nodes used, ~28 effective, ~122
distinct edges, collapse-cos 0.22). It's structured — but it's the *same* structure regardless of who
the passage is about.

![graph_v6 write structure — two different biographies collapse to the same generic hub topology](plots/graph_v6_structure_pretty.png)

> **Conclusion:** the memory acts as a **learned behavioral modulator** — it builds a prior from the
> input that nudges Llama's answer — rather than encoding *this passage's* facts.

---

## 5. Reconciling "graph clearly led" with "graph ≈ Mamba"

It was **QA the whole way** — the early *clear* lead and the later *tie with Mamba* are the *same task*,
measured under different conditions. Three knobs changed together, and each one removes a crutch:

| knob | early "graph clearly leads" | honest test |
|---|---|---|
| **data mix** | easy-dominated (composite 0.5; biographical val-loss 0.67) | hard families (multi-hop musique, no shortcuts) |
| **compression** | generous, not strictly matched (graph_baseline+LB) | 274,944 floats, ~40× compression, **param-matched** (graph_v6) |
| **metric** | teacher-forced answer-NLL / top-1 (soft, partial-credit) | generative AR-decode + LLM judge (all-or-nothing) |

The diagnostics say *why* the clear lead narrowed to a tie: the graph's real edge was **entity-pooling
on easy recall** (+19.5 F1 on biographical, **+0 on multi-hop**). On an easy-dominated mix scored softly
that edge stands out; tighten the mix, the compression, and the metric and it shrinks to **parity with
Mamba** — the two of them still the best memory arms, but now only a hair above the no-memory floor,
because under the probes the memory holds a **generic behavioral prior, not the answer.** The early
margin wasn't grounded relational memory; it was *pooling helping where memory wasn't really needed.*

---

## 6. Forward plan — a curriculum that bootstraps write+read before QA

**The real problem is a joint optimization, and we've been doing it all at once.** Today we bolt the
memory onto Llama and hope the whole thing trains end-to-end. But it's a chicken-and-egg:

> if the **write** doesn't yet know what to store, the **read** learns to ignore it; and if the read
> ignores it, the write never gets gradient (the only path for `∂loss/∂facts` is *through* the read
> using the stored content). Write and read have to become useful *together* — and that joint optimum
> is hard to reach from a cold start, especially while *also* demanding high compression, LM
> integration, and question-conditioned retrieval all at once.

**So decompose it with a curriculum.** Don't ask the memory to compress, integrate, and answer on day
one. First make write and read mutually useful on a *much easier* task, then add the hard parts.

**Stage A — identity warmup (make write+read useful).**
- Put in a handful of facts (3–4), then ask for them back. Goal: **spit out exactly what was put in** —
  no compression pressure (it fits easily in the hidden-state float budget), no question-conditioning,
  no reasoning.
- **Targets in vector space, not token space.** Reconstructing *tokens* through a frozen Llama is its
  own bottleneck; reconstruct the **hidden-state vectors** directly (continuous target, dense gradient).
  That's the cleanest possible signal that the write stored something the read can recover.
- Run it for **every memory arm**, not just the graph — it's a training-protocol bootstrap that gives
  write and read a non-degenerate starting point and breaks the chicken-and-egg.

**Stage B — the hard parts, layered on.** Once write+read demonstrably carry content, introduce the
three things that make this an actual *memory* problem:
1. **High compression** — squeeze from "fits easily" toward the 274,944-float budget (~40×).
2. **LM integration** — surface the memory through Llama for generation (LoRA).
3. **Question-conditioned reads** — retrieve *what the question needs*, not everything.

Gate-warmup and entity-binding become *tactics inside Stage B* (keep the read open early; bind facts to
entities) rather than the whole fix. An explicit per-fact grounding signal stays the last resort — but
Stage A's vector-space reconstruction is the natural, low-risk way to get there first.

**Validation stays cheap:** the recoverability / structure / ablation probes tell us *at each stage*
whether write+read carry content — on a single-arm ~40-min run, before any full sweep.

*(Mechanical fixes in flight — e.g. a 49× memory-magnitude bug shared by the prepend baselines — are
being corrected but aren't the scientific story.)*

---

## 7. Future direction (parked) — implicit behavioral adaptation

The "behavioral bias" finding is a genuine **gap in current LMs**: RAG, `skills.md`, `soul.md` are all
stored as **text** — there's no *implicit* way for a model to **grow and adapt with a person**, tailoring
its behavior to their interests/preferences. A learned behavioral modulator is a contribution on its own.

Pursuing it would need a **different dataset** — not right/wrong QA, but **style-matching / inference-time
adaptation** (change styles *without* training, measure how the model adapts). **We're parking this** and
first trying to get the memory to actually store/retrieve. Revisit only if grounding proves infeasible.

---

## Appendix — provenance & figures

- **Early QA-NLL era (answer-token-masked):** `v1h_t4k_v3_val_curves.png`; the graph-winning run was
  answer-NLL **2.637** / top-1 **49.1%** on the easy-dominated mix (commit `094906b`). Later graph_v5
  checkpoints reached NLL ~2.0 / top-1 ~57% on their own tranches (v5.1/v5.4, commit `4d78e76`) but were
  never the param-matched comparison.
- **Honest-conditions QA:** `docs/repr_learning_head_to_head.md` (graph_v5 5-model, generative+judge),
  `docs/repr_learning_v2_1_results.md` (7-arm fair sweep, hard families).
- **Scoring protocol:** loss masked to `answer_content_mask` (`model.py:852-865`); per-family masking
  support in `data_qa.py` (composite_v1 selective via `target_value`; others all-token); hard-family mix
  via `--composite-task-weights` / `--mix-weights` (`train_repr_qa.py`, commits `a9bf7c4`, `f694899`).
- **Diagnosis & fixes:** `docs/graph_v6_diagnosis.md` (root causes F1–F6 + v6.1 outcome 10.9→13.5).
- **Probes:** `probe_graph_v6_inject.py` (ablation/shuffle), `probe_recoverability.py` (B), `probe_memory_dashboard.py` (rank/magnitude/diversity), `probe_graph_v6_viz.py` (structure).
- **Plots:** `docs/plots/` — `graph_v6_structure_pretty.png` (+ interactive `graph_v6_structure.html`, regen via `viz_graph_v6_html.py`), `scoreboard_3d.png` (answer-NLL bubble), `scoreboard_per_family.png`, `v5_topology_diagnostic.png`, `v1h_t4k_v3_val_curves.png`, `v5_4_graph_evolution.html` (interactive), `v5_4_per_window/` (write progression across context windows).
