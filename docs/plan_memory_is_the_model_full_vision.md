# Memory IS the Seq2Seq Model — Full Vision (Archived)

**Branch:** `memory-is-the-model` (from `main` @ 84a310f).
**Status:** **ARCHIVED FULL-VISION DESIGN.** The simplified v0 scope actually
being built is in `plan_memory_is_the_model.md`. This file is kept as the
superset specification — every idea here is a candidate future extension
on top of v0. Do not build from this doc.
**Date:** 2026-04-19.

> This document captures the full biological-inspiration design with all
> six laminae per column, predictive-coding ERR units, episodic buffer,
> replay consolidation, structural plasticity, etc. It's too many new
> mechanisms to build in one shot — the rollback surface would be huge.
> The active v0 plan (`plan_memory_is_the_model.md`) trims this to a
> three-module prototype and keeps the rest here as a roadmap.

> **TL;DR.** We delete the LM. The "memory graph" stops being a side-channel
> and becomes the entire seq2seq network: tokens enter via an embedding,
> propagate through a lattice of sparsely-connected cortical columns whose
> synapses are continuously plastic under a three-factor learning rule,
> and exit through a learned readout into vocabulary logits. Inference is
> autoregressive, constant-memory in sequence length, and the network
> **keeps adapting on-line forever** — no "training is over" boundary.
> Think TTT (Sun et al., 2024) [[7](https://consensus.app/papers/details/ea4cdf995acb589f93c13cdfa76dbe7f/?utm_source=claude_code)] × Thousand Brains (Clay et al., 2024)
> [[8](https://consensus.app/papers/details/c46aa9f57b0854b696b1993c4a70008c/?utm_source=claude_code)] × three-factor plasticity (Gerstner et al., 2018)
> [[1](https://consensus.app/papers/details/1d1341ffc747562b85dd37015c1b7459/?utm_source=claude_code)].

---

## 1. Why kill the LM?

Current design (`docs/design.md`, `docs/pretrained_lm_memory.md`) has a
clean split:
- **LM**: dense feedforward / scan transformer (from-scratch custom LM, or
  frozen Llama-3.2). Does sequence modelling.
- **Memory graph**: 8 cells × 32 neurons, side-channel read/write at a
  mid-stack layer. Adds ~4M params.

This works and it's trainable — but the split forces three structural
compromises that directly conflict with the goals ("stupidly fast",
"morph/adapt without re-training", "scalable"):

1. **Gradient SNR is dominated by the LM.** The memory contribution is
   additive on top of a near-perfect teacher (Llama-3.2-3B), so CE moves
   by ≪ 1 nat whatever memory does. verify_01 on the from-scratch branch
   measured within-K rollout spread at 2.8×10⁻⁴ vs across-slot 2.4
   (pretrained_lm_memory.md §6 "Why teacher-forced GRPO fails"). Memory
   can only be a *rounding error* on top of a frozen LM.
2. **The LM is not plastic at inference.** If memory adapts but the LM
   doesn't, the system's *behaviour* is still bounded by the frozen
   weights. "Morph without retraining" is false advertising when 99 %
   of FLOPs come from a fixed network.
3. **Two substrates = two throughput budgets.** The memory runs at
   68K tok/s standalone; bolted onto Llama-3.2-3B, the LM dominates
   compute. The memory graph's beautiful Triton fusion buys very little
   because it's 4 M params sitting next to 3 B.

The biological analogy already points the other way. The mammalian
neocortex is approximately six identical layers of cortical columns, all
the way down — there is no "non-plastic substrate" sitting underneath.
Broca's area is not a frozen scaffold with a plastic memory patch; it
is itself a column-based system, and language is an emergent
specialisation of the same cortical primitive that also does vision,
motor, and everything else (Hagoort, 2014 [[12](https://consensus.app/papers/details/f1c76db2b36f5863bdd30b65bf7997dd/?utm_source=claude_code)];
Tremblay et al., 2016 [[13](https://consensus.app/papers/details/ccffa373bf5e5ee8b311577f7ee50445/?utm_source=claude_code)]).

If we believe the neocortex runs on one repeating primitive, the AI
architecture should too. This is the operating-principle bet behind
the Thousand Brains project [[8](https://consensus.app/papers/details/c46aa9f57b0854b696b1993c4a70008c/?utm_source=claude_code),
[14](https://consensus.app/papers/details/a82c4dedd8e55611af58e23d90c9e3d5/?utm_source=claude_code),
[15](https://consensus.app/papers/details/8077b8384e5056c8b191fd5ac786bd3b/?utm_source=claude_code)]: every
column is a semi-independent sensorimotor learner and the whole brain
is a voting pool of columns.

---

## 2. What the current design does well (preserve)

Before redesigning, catalogue what is *working* so we carry it forward:

1. **Multi-timescale event clocks** (per-token LIF / every-4 msg /
   every-16 modulation). This matches the three-factor learning
   timescale gap that Gerstner et al. (2018)
   [[1](https://consensus.app/papers/details/1d1341ffc747562b85dd37015c1b7459/?utm_source=claude_code)]
   identify as a defining biological constraint — ms-scale spikes have
   to bridge to sec-scale behaviour via eligibility traces. Our
   msg_interval / modulation_interval is the artificial analogue.
2. **LIF state update** (`h = tanh(decay·h + (1-decay)·received)`).
   Simple, cheap, fuses in Triton, and matches a leaky soma.
3. **Block-diagonal per-cell W + shared-trunk modulator.** This is
   already the "repeated canonical column" structure with a shared
   neuromodulator vocabulary. Dopamine is dopamine everywhere
   (Palacios-Filardo et al., 2019 [[2](https://consensus.app/papers/details/a5faacb9a7c453fca43713544fbad4ae/?utm_source=claude_code)]);
   the modulator fires the same code, and receptor profiles per area
   (our per-cell decoder with cell_emb conditioning) differ.
4. **γ-clamped EMA plasticity** (W updates via `W ← (1-γ)·W + γ·ΔW`).
   Keeps (1-γ) above bf16 noise floor and stops runaway learning.
5. **Hebbian rolling trace** = explicit eligibility trace in the
   Gerstner/Shindou three-factor sense
   [[3](https://consensus.app/papers/details/39e2eaa5a0ce50e8a55b075431e66c5e/?utm_source=claude_code)].
   Silent until the neuromodulator (modulator fire) consolidates it.
6. **Triton fusion of the per-token hot path.** 4.2× per-token speedup
   and the kernel design (state tiles in SRAM, one program per
   batch×cell pair) generalises to a bigger-N column system.

These five primitives survive the redesign unchanged.

---

## 3. What's missing (address)

Gaps, ranked by impact on the "memory IS the model" goal:

### 3.1 Laminar structure inside the column

Our cell is 32 all-to-all neurons with 4 "input port" and 4 "output
port" indices. Biologically, a cortical column is **six laminae with
distinct computational roles**:

- **L4** — thalamo-recipient, bottom-up input gate.
- **L2/3** — canonical inference / cortico-cortical output.
- **L5** — motor / subcortical output + apical integration (compartmental).
- **L6** — corticothalamic feedback, gain control.
- **L1** — pure dendritic layer; top-down modulatory feedback lands here.

Schulte to Brinke et al. (2022)
[[16](https://consensus.app/papers/details/9543693f1c365ec4930df215d52c14c0/?utm_source=claude_code)]
replicate the Potjans-Diesmann microcircuit and show that the specific
laminar connectivity *measurably* enhances computation versus random
controls. Cain et al. (2014)
[[17](https://consensus.app/papers/details/0d6ad0e4c70256dbbded399e3377a7d1/?utm_source=claude_code)]
show L4 and L2/3 inputs combine additively into L2/3 but subtract into
L5 — a hard-coded "prior × evidence" operation. George et al. (2025)
[[18](https://consensus.app/papers/details/6bb5f041605256efa73b1ace7797aefe/?utm_source=claude_code)]
derive the laminar roles from a generative-inference objective: L4 =
likelihood factor, L2/3 = posterior, L5 = output decisions, L6 = prior
modulator.

**Implication**: partition our per-column neuron set by role. Not just
"4 input ports + 4 output ports + 24 internal" — give the populations
role-specific connection priors and role-specific plasticity.

### 3.2 Long-range inter-column routing

Currently cells communicate only via the LM scan (inject / readout).
Kill the LM and this fails instantly. Biologically, columns talk via:

- **Thalamo-cortical loops** — thalamus broadcasts gated copies of
  cortical activity (George et al. 2025 make this explicit
  [[18](https://consensus.app/papers/details/6bb5f041605256efa73b1ace7797aefe/?utm_source=claude_code)]).
- **Long-range cortico-cortical** — L2/3 → L4 feedforward, L5/L6 →
  L1 feedback. Small-world topology: high clustering + short path
  length through a handful of hub regions (Achard et al., 2006
  [[19](https://consensus.app/papers/details/a6b931c3a20958a78aff54ba9339237e/?utm_source=claude_code)],
  clustering 0.53 vs random 0.2; van den Heuvel et al., 2008
  [[20](https://consensus.app/papers/details/308533b8390b5b3f874364a28da6c830/?utm_source=claude_code)]).
- **Heterarchy, not strict hierarchy** — Hawkins et al. (2025)
  [[15](https://consensus.app/papers/details/8077b8384e5056c8b191fd5ac786bd3b/?utm_source=claude_code)]
  argue many connections skip levels; voting between same-level columns
  is as important as feedforward.

Mocanu et al. (2017)
[[21](https://consensus.app/papers/details/2c65cd5405d45036ae5824355e5dc489/?utm_source=claude_code)]
showed empirically that sparse scale-free topology matches dense-FC
accuracy at a tiny fraction of parameters. That's the scaling lever we
need.

**Implication**: replace the LM router with an explicit **inter-column
graph** — fixed (learned) sparse topology at init, plastic at inference.
Use sparse block-CSR `W_global` plus a small shared "thalamus" buffer.

### 3.3 Prediction → error → plasticity coupling

Predictive coding (Rao & Ballard 1999; Bogacz, 2017
[[22](https://consensus.app/papers/details/f7210495d7fd5003885659f62e8e7efb/?utm_source=claude_code)])
posits that cortical hierarchies *constantly generate predictions* of
lower-level activity, and the quantity flowing up is *prediction
error*, not raw activity. Ali et al. (2021)
[[23](https://consensus.app/papers/details/7482b7936ccd5c4daff3adc5311ee292/?utm_source=claude_code)]
show that energy-efficiency training *alone* produces prediction/error
unit separation with appropriate inhibition — no hand-coding required.
Chien et al. (2019)
[[24](https://consensus.app/papers/details/6c84f1d136365d348bd8bdaa0f4f642f/?utm_source=claude_code)]
measure gated temporal integration across the human language cortical
hierarchy: each region holds a temporal context that is nonlinearly
integrated with input and **gated by local prediction error**.

Our current "surprise" is a scalar (CE of readout vs ground truth) fed
once to the modulator. That is ~0.01 % of the predictive-coding
signal — we're leaving almost all of it on the floor.

**Implication**: every lamina emits a local prediction of its own
inputs; the prediction error (not the raw input) drives (a) downstream
message passing and (b) plasticity magnitude. Fire-rate of plasticity
writes becomes *proportional to locally observed surprise*, which
matches Palacios-Filardo et al.'s (2019)
[[2](https://consensus.app/papers/details/a5faacb9a7c453fca43713544fbad4ae/?utm_source=claude_code)]
description of neuromodulator-gated hippocampal plasticity.

### 3.4 Hippocampal/episodic fast buffer (complementary learning systems)

McClelland-style CLS theory + modern extensions (Spens et al., 2023
[[25](https://consensus.app/papers/details/0f8c7aa167aa54da88190dff4ec3f157/?utm_source=claude_code)];
Howard et al., 2019
[[26](https://consensus.app/papers/details/d52cec68555e502db2f9dca26dc239cb/?utm_source=claude_code)];
Brodt et al., 2023
[[27](https://consensus.app/papers/details/fe36e064489c5b40b0200d00adaea41c/?utm_source=claude_code)])
split memory into two coupled systems:

- **Hippocampus** — high-plasticity, high-interference, short-lived. An
  autoassociator that rapidly binds new episodes.
- **Neocortex** — slow-plasticity generative model. Updated by
  hippocampal replay during offline periods (sleep SWR).

Our design has only a single cortical-scale system with γ-clamped EMA.
It *cannot* write an episode in one shot (γ is hard-capped at 0.97 and
most neurons sit far lower), and we never do "replay" of past
trajectories.

**Implication**: add an **episodic buffer** — a small high-γ
fast-plasticity sub-graph (think K=4-8 extra columns with γ_max → 1.0,
decay fast) that binds each input chunk to a compressed key. A
periodic replay pass (e.g. every 512 tokens, async) teacher-forces
past episodes to train the slow neocortical columns. This is the same
generative-model-trained-by-autoassociator-replay structure that Spens
et al. (2023) validate [[25](https://consensus.app/papers/details/0f8c7aa167aa54da88190dff4ec3f157/?utm_source=claude_code)].

### 3.5 Continuous adaptation at inference (TTT)

Sun et al. (2024) "TTT: RNNs with Expressive Hidden States"
[[7](https://consensus.app/papers/details/ea4cdf995acb589f93c13cdfa76dbe7f/?utm_source=claude_code)]
and Zhang et al. (2025) "LaCT"
[[28](https://consensus.app/papers/details/fb2c8417ffb05dc0a9a0fd6426418ce4/?utm_source=claude_code)]
make the key observation:

> **The hidden state IS a machine-learning model.** The update rule
> (advancing that hidden state by one token) is a step of self-supervised
> learning.

LaCT scales this to 14 B parameters with 40 % of state capacity in
fast weights, 56 K context. MesaNet (von Oswald et al., 2025)
[[29](https://consensus.app/papers/details/ce96961fa81a5e679cba72cfa49cedbb/?utm_source=claude_code)]
solves an in-context regression objective to optimality with conjugate
gradient per step.

Our current design has plastic W, but (a) it only updates every 16
tokens not every token, (b) ΔW is produced by a separate modulator
network (not derived from a loss at W), and (c) after training, ΔW
dynamics are still gated by frozen modulator weights — limiting true
test-time adaptation.

**Implication**: frame W updates as an *implicit optimization step* of
a per-column local objective. The neuromodulator's job then becomes to
shape that *objective* (not to emit ΔW directly) — it picks what each
column should try to predict next, and the column's W moves one gradient
step in that direction per event. This is exactly the mesa-optimization
framing (von Oswald et al., 2022
[[30](https://consensus.app/papers/details/0689bdbdaf58597481b464be8402f34f/?utm_source=claude_code)];
Zheng et al., 2024
[[31](https://consensus.app/papers/details/ea93c4f0b28c515eb6910c77d2bd9040/?utm_source=claude_code)]),
recast as something biology actually does.

### 3.6 Readout

We currently pool 4 output-port neurons per cell × 8 cells → 32
neurons → 2048 dims → fed into the LM. Kill the LM and the readout has
to go straight to vocabulary. Vocab is 32-128K; a single linear from a
small pool is probably under-parametrised.

**Implication**: dedicate a small set of columns as the **readout pool**
(analogous to left IFG / motor output in language cortex — Hagoort 2014
[[12](https://consensus.app/papers/details/f1c76db2b36f5863bdd30b65bf7997dd/?utm_source=claude_code)]).
Their output goes through a tied-embedding head (Llama-style).

---

## 4. Proposal — CortexNet

Name is a placeholder (branch name is `memory-is-the-model`). Structure:

### 4.1 Lattice of columns

```
  INPUT:  tokens → embedding_table[vocab→D_e=512]
                         │
                         ▼
  SENSORY COLUMNS (C_sense=16, specialised for token embedding)
                         │   heterarchical sparse routing
                         ▼
  ASSOCIATION COLUMNS (C_assoc=128, generic cortex)
                         │
                         ▼
  READOUT COLUMNS (C_out=16, tied to unembedding head)
                         │
                         ▼
  OUTPUT: per-column pooled state → W_out → vocab logits
```

Total columns: `C = 160` at the dev tier. Target scale: `C ≥ 1024`.

### 4.2 Anatomy of one column

Each column has **6 laminar populations** (inspired by George et al.,
2025 [[18](https://consensus.app/papers/details/6bb5f041605256efa73b1ace7797aefe/?utm_source=claude_code)])
and a **per-lamina plasticity budget**:

```
  L1   (N_top=4 neurons) — apical dendrites. Receive top-down / modulation.
  L2/3 (N_main=32)       — canonical inference (our current "internal").
                            Main plasticity target. Emits msg to other cols.
  L4   (N_in=8)          — thalamo-recipient, gates bottom-up input.
                            Receives long-range msgs + sensory embedding.
  L5   (N_out=8)         — generative / output. Drives readout pool for
                            readout columns; emits predictions for all cols.
  L6   (N_thal=4)        — thalamic feedback. Receives cortex, broadcasts
                            via shared thalamic buffer T.
  ERR  (N_err=4, one per predicted lamina) — explicit prediction error
                            units. Their activity = ||L5.predict − L4.in||.
```

Total per column: ~60 neurons (vs current 32). Across `C=160`: ~9,600
neurons; across `C=1024`: ~61K.

All intra-column W is **sparse**. Initialisation: scale-free (preferential
attachment) per Mocanu et al. (2017)
[[21](https://consensus.app/papers/details/2c65cd5405d45036ae5824355e5dc489/?utm_source=claude_code)]
with K ≈ log₂(N) ~ 6 edges per neuron on average. Small-world routing
inside a column.

Inter-lamina connectivity follows the canonical microcircuit:
- L4 → L2/3 (feedforward, plastic)
- L2/3 → L5 (main output pathway, plastic)
- L5 → L1 of upstream columns (feedback predictions, frozen/slowly plastic)
- L6 → thalamic buffer T (broadcast)
- L1 → L2/3 (top-down modulation, plastic gain)

### 4.3 Inter-column graph

- **Sparse scale-free** adjacency `A ∈ {0,1}^{C×C}`. Learned at init via
  a simple preferential-attachment rule, then **plastic** via structural
  plasticity events (add/drop edges at the slow clock).
- **Hub columns** (top ~10 % by degree) serve as "association cortex"
  (Achard 2006 hubs [[19](https://consensus.app/papers/details/a6b931c3a20958a78aff54ba9339237e/?utm_source=claude_code)]).
- **Shared thalamic buffer** `T ∈ [BS, C_thal=16, D_n]`. Any L6 writes
  to `T`; any L1 reads from `T` via a sparse column→thalamic-slot gate.
  Constant memory — `T` does not grow with sequence length.

Inter-column messages are **sparse gather** — `msg_from[c]` is gathered
only for the `~log C` columns that the receiver has edges to. Sparse
CSR bmm is the hot kernel.

### 4.4 Multi-clock event schedule (generalises current)

| Clock | Period | What fires | Analogy |
|---|---:|---|---|
| FAST | 1 tok | LIF integration of `received = W_intra @ msg_prev + inject + T_broadcast`; compute per-lamina prediction & error | somatic membrane |
| SPIKE | 4 tok | `msg = spike_MLP(h)`; sparse gather to neighbour columns; update Hebbian eligibility trace | ms-scale action potential burst |
| MOD | 16 tok | neuromodulator fires per column — emits a **target** (not a ΔW) for each column's local self-supervised loss; column takes one SGD-style step on its W towards that target | DA/ACh/NA release |
| CONSOLIDATE | 512 tok | async: episodic buffer replays recent windows, teacher-forcing slow-column plasticity; prune/grow inter-column edges | sleep SWR replay |

### 4.5 Plasticity — three-factor + mesa-objective

Per column, per MOD event:

1. **Factor 1 — activity.** Accumulated Hebbian eligibility trace
   `E = (1-γ_e)·E + γ_e · msg · msgᵀ` (intra-column; current design has
   this).
2. **Factor 2 — prediction error.** Each lamina's local ERR units fire;
   `err_c = ‖L5.predict_c - L4.input_c‖²` integrated over the mod period.
   (Biology: Bogacz 2017
   [[22](https://consensus.app/papers/details/f7210495d7fd5003885659f62e8e7efb/?utm_source=claude_code)].)
3. **Factor 3 — neuromodulator.** A shared-trunk modulator emits per
   column a **target pattern** `y_c ∈ R^D_n` and a **learning rate**
   `η_c ∈ R⁺`. These are what the codebook encodes now, but interpreted
   differently: instead of decoding to ΔW, they define a local
   regression objective `L_c(W) = ‖W·msg_c − y_c‖²`, and the column
   takes a single gradient step on that objective. The combined update is:

   `ΔW_c = η_c · (y_c − W·msg_c) · msg_cᵀ · gate(E, err_c)`

   `W_c_new = (1-γ_W)·W_c + γ_W · (W_c − ΔW_c)`

   where `gate(E, err_c)` is a learned function of the eligibility trace
   and the local error (0 when nothing surprising, 1 when both active —
   the three-factor rule of Gerstner et al. 2018
   [[1](https://consensus.app/papers/details/1d1341ffc747562b85dd37015c1b7459/?utm_source=claude_code)];
   Shindou et al. 2018
   [[3](https://consensus.app/papers/details/39e2eaa5a0ce50e8a55b075431e66c5e/?utm_source=claude_code)]).

This is **the TTT story** (Sun et al. 2024
[[7](https://consensus.app/papers/details/ea4cdf995acb589f93c13cdfa76dbe7f/?utm_source=claude_code)])
made column-local and neuromodulator-gated. Every column is running its
own online SGD against a target emitted by the modulator, but only
when eligibility × error agree that it's worth updating.

The distinction from current design is crucial: **we stop emitting ΔW
directly** and instead emit the *objective* the column should follow.
This means:
- The modulator is much smaller — it emits a target vector, not an N×N
  matrix.
- The column's ΔW comes from a principled gradient step, not a decoded
  pattern, so it generalises cleanly to arbitrary column sizes.
- At inference, W keeps moving — true test-time adaptation.

### 4.6 Episodic buffer (fast-plasticity columns)

`C_episodic = 8` extra columns with:
- γ_W uncapped (target 0.99+; near-instant writing).
- Decay coupled to a per-column "time since write" scalar — old
  bindings erode unless refreshed (Howard et al., 2019
  [[26](https://consensus.app/papers/details/d52cec68555e502db2f9dca26dc239cb/?utm_source=claude_code)]).
- Output feeds the association cortex, but is **replay-gated**: once
  every CONSOLIDATE interval, an async pass reads a compressed trace of
  the last K episodes and drives slow-column plasticity with them as
  teacher-forced inputs. This is the HC→neocortex consolidation of
  Spens et al. (2023)
  [[25](https://consensus.app/papers/details/0f8c7aa167aa54da88190dff4ec3f157/?utm_source=claude_code)].

### 4.7 Readout

`C_out = 16` output columns. Their L5 population pools into a
shared embedding `h_out ∈ R^{D_e=512}` → unembedding (tied to input
embedding, Llama-style) → softmax over vocab.

One-line summary: **input embedding → sparse-column lattice + thalamic
buffer + episodic buffer → plastic all the way down → tied-embedding
readout.** No transformer blocks. No scan layers. No frozen LM.

---

## 5. Training protocol

### 5.1 Three nested objectives

1. **Global CE** — next-token prediction at the readout. Standard.
2. **Local predictive coding** — each column's ERR unit loss:
   `L_pred = Σ_c ‖L5.predict_c − L4.input_c‖²`. Unsupervised. Runs
   continuously.
3. **Meta-plasticity** — differentiable-plasticity training of the
   modulator and gate networks, using Miconi et al. (2018)
   [[32](https://consensus.app/papers/details/af2e3b8fc73053629c5c72276a207481/?utm_source=claude_code)]
   style backprop-through-plasticity on short TBPTT windows. This
   teaches the modulator *what targets to emit* so that the column's
   subsequent gradient steps help future CE.

Composite loss:  `L = L_CE + λ_pred · L_pred + 0 · L_meta` (meta is
implicit through CE gradient reaching the modulator via the plasticity
chain — no extra loss term needed).

**RL is kept as an optional late-stage wrapper only.** GRPO is only
worth it once the network is stable, because the known failure mode
(SNR 10⁻⁴ per verify_01) comes from adding RL on top of an already
minimizing CE. Here we actually have a strong enough gradient from the
plasticity → column → CE chain.

### 5.2 TBPTT + chunked updates

- Segment length T = 512 tokens (long; matches LaCT-scale
  [[28](https://consensus.app/papers/details/fb2c8417ffb05dc0a9a0fd6426418ce4/?utm_source=claude_code)]).
- `tbptt_block = 32` (W-detach every 32 tokens so backward graph on
  intra-segment W stays bounded).
- Episodic buffer always detached except within a CONSOLIDATE window.

### 5.3 Data (reuses `docs/training_plan.md` stack)

- **Stage 0** — synthetic passkey / K:V / ordered-list (fast
  architecture-sanity loop). Same as before.
- **Stage 1** — conversational memory dialogues.
- **Stage 2** — real long-form (PG-19, LongMemEval, RULER).

The new architecture's test-time adaptation makes passkey-style tests
particularly cheap to validate: memory should learn to bind the passkey
in O(1) MOD events and retrieve it at query time via episodic replay.

---

## 6. Scaling and throughput math

### 6.1 Parameter count (dev tier, C = 160)

| Component | Param count |
|---|---:|
| Token embedding (tied) | 32K × 512 = 16.4 M |
| Per-column intra W (sparse, K=6) | 160 × 60 × 6 × D_n=128 = 7.4 M |
| Per-column lamina MLPs (L4, L2/3, L5 predict, L6) | 160 × 4 × (2 × 128² + small) = ~21 M |
| Inter-column sparse edges | 160 × 12 × D_n² ≈ 32 M |
| Thalamic routing (column→T→column) | 16 × 160 × D_n ≈ 0.3 M |
| Modulator + gates (shared trunk) | ~5 M |
| Episodic buffer (8 cols full + readout) | ~2 M |
| Readout (unembedding is tied) | 16 × 60 × 512 = 0.5 M |
| **Total** | **~85 M** |

~100 M at dev, comparable to Llama-3.2-1B. Target tier (C=1024) scales
to ~1 B with the same primitives.

### 6.2 Throughput target

Per-token cost per column:
- Intra W@msg (sparse, K=6): O(N_col × K × D_n) = 60 × 6 × 128 ≈ 46K ops
- Inter-column gather (sparse, ~log C=8 edges): 8 × D_n = 1K ops
- LIF + tanh: 60 × D_n = 7.7K ops

Total per token, all columns: 160 × (46K + 8K) ≈ 8.6M ops. At 4090
peak ~80 TFLOPS bf16, this is ~1 μs per token — way below memory
bandwidth bound.

**Realistic target**: 100-200K tok/s at dev tier (2-3× current main).
Real-world bottleneck will be the sparse gathers; sparse block-CSR
kernels are 40-60 % tensor core efficiency vs 95 % for dense. We
already have the parked `sparse-memory-graph` branch as a starting
point.

### 6.3 Memory scaling (constant in sequence length)

- No KV cache; RNN-style.
- Only W + Hebbian grow — both are per-column and do NOT depend on T.
- Activation memory inside a TBPTT window: O(BS × T × C × D_n) per
  segment; detached across segments.
- Total inference memory at BS=1, C=1024: ~6 GB — fits on a 4090.

---

## 7. Risks and open questions

### 7.1 Big risks

1. **Without the LM scaffold, does NTP even descend?** The LM is
   absorbing most of the lift today. Kill it and we have to prove that
   a plastic-cortex-only model can learn language from scratch at our
   compute budget. Stage 0 passkey + bigram-entropy diagnostic is the
   gate.
2. **Mesa-optimisation vs direct decoding.** Emitting targets (not ΔW)
   is elegant but unproven at this scale. Could fail silently if the
   learned target + gate doesn't find useful columns. Fallback: hybrid
   — emit a target *plus* a small ΔW correction, anneal the correction
   out.
3. **Sparse kernel perf.** Mocanu et al.'s SET
   [[21](https://consensus.app/papers/details/2c65cd5405d45036ae5824355e5dc489/?utm_source=claude_code)]
   is a reference point, not a recipe. We'll likely need custom
   block-sparse Triton kernels for both intra-column W@msg and
   inter-column gathers. Budget 2-3 weeks for kernel work alone.
4. **Credit assignment across CONSOLIDATE boundaries.** Our current
   TBPTT story stops at 32 tokens. Episodic replay → slow-column
   plasticity is a new gradient path; need to verify that meta-gradient
   through it is informative and doesn't explode.

### 7.2 Open architectural questions

1. **Do we actually need L1 / L6 separately, or is a single top-down
   bus enough?** The TBP theory
   [[15](https://consensus.app/papers/details/8077b8384e5056c8b191fd5ac786bd3b/?utm_source=claude_code)]
   merges them into a "cortical messaging protocol." Simpler is
   better if it works.
2. **Column count sweet spot.** More columns = more distributed
   representation but also more inter-column routing cost. Start at
   C=160, sweep.
3. **Replay schedule.** Every 512 tokens is a guess. Could make it
   surprise-triggered (high-error → more replay).
4. **Modulator vocabulary size.** Currently 2048 codes; with
   target-emission semantics, may need more. Open.

### 7.3 What we deliberately are NOT doing

- **Not parallelising the time dim.** Keep autoregressive recurrence —
  that's the whole point ("stupidly fast, inference-plastic").
- **Not using attention over tokens.** Replaced entirely by column
  state + inter-column routing.
- **Not re-using Llama weights.** Complete break from the pretrained
  pivot on `main`.

---

## 8. Implementation milestones

**M1 — Paper prototype (1 week).** Write the math cleanly in a notebook,
validate gradient flow on a toy: 8 columns, vocabulary 64, passkey task,
full differentiable plasticity loop. No Triton. Confirm loss descends
and W actually moves.

**M2 — Single-column Triton kernel (2 weeks).** Port the column's
intra-lamina fast + spike + mod clocks into a fused kernel. Target:
match current memory-graph fused kernel's per-column throughput.

**M3 — Sparse inter-column routing (2 weeks).** Block-CSR W@msg for
inter-column; structural plasticity (add/drop edges); shared thalamic
buffer. Bench vs dense baseline.

**M4 — Episodic buffer + consolidation (1 week).** Fast-plasticity
columns + async replay. Validate on passkey at 2K distance.

**M5 — Dev-tier training run (2 weeks).** Stage 0 → Stage 1 on real
data. Gate: beats vanilla Llama-3.2-1B on LongMemEval-short at dev
scale.

**M6 — Scale sweep + paper (3 weeks).** C=160 → C=1024 at constant
tok/s, document.

Total optimistic estimate: **11 weeks**. Likely 15-18 weeks real-time.

---

## 9. Relationship to prior work on our tree

- `main` — current attention-neuromod + pretrained-LM pivot. Remains
  the production path; CortexNet is a parallel research bet.
- `abandoned/v9-backprop*` — had per-neuron MLPs and structural
  plasticity; some of those primitives (the phi-based Pearson-corr
  rewiring) come back here.
- `sparse-memory-graph` — already has N×K sparse groundwork. Likely
  the starting branch point for M3.

---

## 10. References

**Three-factor plasticity / eligibility traces**

[1] [Eligibility Traces and Plasticity on Behavioral Time Scales](https://consensus.app/papers/details/1d1341ffc747562b85dd37015c1b7459/?utm_source=claude_code) (Gerstner et al., 2018, 299 citations, Frontiers in Neural Circuits) — foundational review of the three-factor learning framework.

[2] [Neuromodulation of hippocampal long-term synaptic plasticity](https://consensus.app/papers/details/a5faacb9a7c453fca43713544fbad4ae/?utm_source=claude_code) (Palacios-Filardo et al., 2019, 130 citations, Current Opinion in Neurobiology) — DA/ACh/NA/5HT gating of hippocampal LTP.

[3] [A silent eligibility trace enables dopamine‐dependent synaptic plasticity](https://consensus.app/papers/details/39e2eaa5a0ce50e8a55b075431e66c5e/?utm_source=claude_code) (Shindou et al., 2018, 80 citations, Eur J Neuroscience) — direct experimental evidence for 2-second silent eligibility trace in striatum.

**PFC + language cortex**

[4] [Prefrontal cortex as a meta-reinforcement learning system](https://consensus.app/papers/details/6338c79f9dbc5556bf28aeb8cbe15de9/?utm_source=claude_code) (Wang et al., 2018, 570 citations, Nature Neuroscience) — PFC recurrent net trained by dopamine = its own learning system.

[5] [A working memory model based on recurrent neural networks using reinforcement learning](https://consensus.app/papers/details/bcbc9828deeb57ad80ba1710c0f50f8b/?utm_source=claude_code) (Wang et al., 2024, Cognitive Neurodynamics) — PFC-like RNN learns stable low-dim working memory code.

[6] [Pruning recurrent neural networks replicates adolescent changes in working memory and RL](https://consensus.app/papers/details/4cecaee9060d5aa5abe50993698562de/?utm_source=claude_code) (Averbeck, 2022, 24 citations, PNAS) — synaptic pruning improves working memory + RL.

**Test-time training / network IS the model**

[7] [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://consensus.app/papers/details/ea4cdf995acb589f93c13cdfa76dbe7f/?utm_source=claude_code) (Sun et al., 2024, 161 citations, ArXiv) — TTT: hidden state is an ML model, updated by SGD each step.

**Thousand Brains theory**

[8] [The Thousand Brains Project: A New Paradigm for Sensorimotor Intelligence](https://consensus.app/papers/details/c46aa9f57b0854b696b1993c4a70008c/?utm_source=claude_code) (Clay et al., 2024, 4 citations, ArXiv) — the flagship paper on repeated-column sensorimotor AI.

[14] [Thousand-Brains Systems: Sensorimotor Intelligence for Rapid, Robust Learning and Inference](https://consensus.app/papers/details/a82c4dedd8e55611af58e23d90c9e3d5/?utm_source=claude_code) (Leadholm et al., 2025, 1 citation, ArXiv) — Monty evaluation; rapid continual learning from associative Hebbian binding.

[15] [Hierarchy or Heterarchy? A Theory of Long-Range Connections for the Sensorimotor Brain](https://consensus.app/papers/details/8077b8384e5056c8b191fd5ac786bd3b/?utm_source=claude_code) (Hawkins et al., 2025, 1 citation, ArXiv) — cortex is heterarchical, not strict hierarchical; thalamus as pose translator.

**Canonical cortical microcircuit**

[16] [Characteristic columnar connectivity caters to cortical computation](https://consensus.app/papers/details/9543693f1c365ec4930df215d52c14c0/?utm_source=claude_code) (Schulte to Brinke et al., 2022, 6 citations, Frontiers in Integrative Neuroscience) — laminar connectivity sharpens representations vs random control circuits.

[17] [The computational properties of a simplified cortical column model](https://consensus.app/papers/details/0d6ad0e4c70256dbbded399e3377a7d1/?utm_source=claude_code) (Cain et al., 2014, 26 citations, BMC Neuroscience) — L4/L2-3 additive for L2/3, subtractive for L5.

[18] [A detailed theory of thalamic and cortical microcircuits for predictive visual inference](https://consensus.app/papers/details/6bb5f041605256efa73b1ace7797aefe/?utm_source=claude_code) (George et al., 2025, 3 citations, Science Advances) — laminar roles derived from generative inference; explicit column schematic.

**Language cortex functional architecture**

[12] [Nodes and networks in the neural architecture for language: Broca's region and beyond](https://consensus.app/papers/details/f1c76db2b36f5863bdd30b65bf7997dd/?utm_source=claude_code) (Hagoort, 2014, 264 citations, Current Opinion in Neurobiology) — classical model obsolete; language is a dynamic network of regions, shared comprehension/production.

[13] [Broca and Wernicke are dead, or moving past the classic model](https://consensus.app/papers/details/ccffa373bf5e5ee8b311577f7ee50445/?utm_source=claude_code) (Tremblay et al., 2016, 405 citations, Brain and Language) — no anatomical consensus on "Broca's area"; language is a distributed connectome.

[24] [Constructing and Forgetting Temporal Context in the Human Cerebral Cortex](https://consensus.app/papers/details/6c84f1d136365d348bd8bdaa0f4f642f/?utm_source=claude_code) (Chien et al., 2019, 92 citations, bioRxiv) — hierarchical temporal integration gated by local prediction error.

**Brain topology**

[19] [A Resilient, Low-Frequency, Small-World Human Brain Functional Network](https://consensus.app/papers/details/a6b931c3a20958a78aff54ba9339237e/?utm_source=claude_code) (Achard et al., 2006, 2388 citations, J Neuroscience) — sparse small-world cortex with hub structure.

[20] [Small-world and scale-free organization of voxel-based resting-state functional connectivity](https://consensus.app/papers/details/308533b8390b5b3f874364a28da6c830/?utm_source=claude_code) (van den Heuvel et al., 2008, 738 citations, NeuroImage) — cortex is both small-world AND scale-free.

[21] [Scalable training of artificial neural networks with adaptive sparse connectivity inspired by network science](https://consensus.app/papers/details/2c65cd5405d45036ae5824355e5dc489/?utm_source=claude_code) (Mocanu et al., 2017, 688 citations, Nature Communications) — Sparse Evolutionary Training (SET): FC replaced by sparse scale-free, no accuracy loss.

**Predictive coding**

[22] [A tutorial on the free-energy framework for modelling perception and learning](https://consensus.app/papers/details/f7210495d7fd5003885659f62e8e7efb/?utm_source=claude_code) (Bogacz, 2017, 286 citations, J Mathematical Psychology) — accessible Friston free-energy tutorial.

[23] [Predictive coding is a consequence of energy efficiency in recurrent neural networks](https://consensus.app/papers/details/7482b7936ccd5c4daff3adc5311ee292/?utm_source=claude_code) (Ali et al., 2021, 59 citations, Patterns) — prediction/error units self-organise from energy minimisation alone.

**Complementary learning systems**

[25] [A generative model of memory construction and consolidation](https://consensus.app/papers/details/0f8c7aa167aa54da88190dff4ec3f157/?utm_source=claude_code) (Spens et al., 2023, 64 citations, Nature Human Behaviour) — HC replay trains neocortical VAEs; unified episodic + semantic model.

[26] [A model of bi-directional interactions between complementary learning systems for memory consolidation of sequential experiences](https://consensus.app/papers/details/d52cec68555e502db2f9dca26dc239cb/?utm_source=claude_code) (Howard et al., 2019, 5 citations, Frontiers in Systems Neuroscience) — explicit sequence-level CLS model with two-way replay.

[27] [Sleep-A brain-state serving systems memory consolidation](https://consensus.app/papers/details/fe36e064489c5b40b0200d00adaea41c/?utm_source=claude_code) (Brodt et al., 2023, 181 citations, Neuron) — SWR + spindle-driven transformation of HC episodic → cortical schema.

**Large-chunk TTT and fast weights**

[28] [Test-Time Training Done Right](https://consensus.app/papers/details/fb2c8417ffb05dc0a9a0fd6426418ce4/?utm_source=claude_code) (Zhang et al., 2025, 17 citations, ArXiv) — LaCT: 14 B param, 40 % state capacity in fast weights, 56 K–1 M context.

[29] [MesaNet: Sequence Modeling by Locally Optimal Test-Time Training](https://consensus.app/papers/details/ce96961fa81a5e679cba72cfa49cedbb/?utm_source=claude_code) (von Oswald et al., 2025, 10 citations, ArXiv) — Mesa layer solves in-context regression with conjugate gradient per token.

[30] [Transformers learn in-context by gradient descent](https://consensus.app/papers/details/0689bdbdaf58597481b464be8402f34f/?utm_source=claude_code) (von Oswald et al., 2022, 611 citations) — proves linear self-attention = gradient step on regression loss.

[31] [On Mesa-Optimization in Autoregressively Trained Transformers](https://consensus.app/papers/details/ea93c4f0b28c515eb6910c77d2bd9040/?utm_source=claude_code) (Zheng et al., 2024, 5 citations, ArXiv) — non-convex training dynamics converge to mesa-optimiser.

**Differentiable plasticity**

[32] [Differentiable plasticity: training plastic neural networks with backpropagation](https://consensus.app/papers/details/af2e3b8fc73053629c5c72276a207481/?utm_source=claude_code) (Miconi et al., 2018, 170 citations) — 2 M-parameter plastic RNNs trainable end-to-end; beats non-plastic on image memorisation.

**Tolman-Eichenbaum (related memory generalisation)**

[33] [The Tolman-Eichenbaum Machine: Unifying Space and Relational Memory through Generalization in the Hippocampal Formation](https://consensus.app/papers/details/12de2e1b677d533796b8d7cfcc3f03dc/?utm_source=claude_code) (Whittington et al., 2019, 518 citations, Cell) — EC = structural basis, HC = sensory-binding; equivalent to a transformer.

[Paper consensus search link (20 full results per query)](https://consensus.app/sign-up/?utm_source=claude_code&auth=claude_code)
