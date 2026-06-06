# EMAT-first baseline comparison — execution plan

*Branch: `emat-baselines` (off `main`). Direction: [[project_stage_a_direction_emat]]. Diagnosis behind it: docs/mamba_two_lenses_memory.md.*

## Goal

Benchmark memory **mechanisms** on the EMAT **closed-book key→value** objective, with our graph competing against **reputable, published compressor architectures** — all on one frozen backbone, one objective, one matched memory budget, so the *only* variable is the memory mechanism.

## The objective (EMAT, Stage-1b)

- **Write:** encode a passage into a fixed memory `[B, M, d]`.
- **Read:** given a question (the *key*), the frozen LM produces the answer (the *value*) — **closed-book**: the answer span is **absent** from the LM's input, so it can only come from memory.
- **(optional) key/value auto-encoding** (the EMAT loss-ii): key⇒question, value⇒answer — forces each half to encode its content. Defer; add only if the plain closed-book CE underperforms (per [[feedback_avoid_aux_losses]]).
- This tests **content-addressing** (the capability we diagnosed as broken: `REAL == SHUF`), unlike ICAE which tests sequential storage.

## Arms (all retrained, frozen Llama-3.2-1B, identical objective/data/budget)

| arm | what it is | provenance | status |
|---|---|---|---|
| **graph** | our competitor (`graph_v6_baseline`, then bind/delta redesign) | ours | exists in `main` |
| **ICAE** | LoRA-encoder → M soft-prompt slots | Ge et al., **ICLR 2024**, `getao/icae` | port as encoder variant |
| **CCM** | conditional-LoRA → compressed KV memory | Kim et al., **ICLR 2024**, `snu-mllab/Context-Memory` | port as encoder variant |
| **Activation Beacon** | per-layer beacon-activation memory | Zhang et al. (BAAI), FlagEmbedding | port as encoder variant |
| *(opt.)* **LongMem** | explicit out-of-sequence K/V bank | Wang et al., **NeurIPS 2023** | natively key→value; port later |
| **MT** | native-KV memorizing-transformer reference | already a variant | control |
| **vanilla Llama** | no memory | `NullEncoder` | OFF floor |

"Verbatim" = reproduce each baseline's **compressor architecture faithfully as an encoder variant** (we change only objective/data/backbone — same as we do for the graph). We do *not* modify their architecture; we *do* retrain (no 1B checkpoint exists for anyone).

## What we REUSE from `main` (most of the harness already exists)

- **Closed-book read path** — `compute_qa_loss` (memory + question → answer, answer absent).
- **Coined-entity data** — `data/wave1/composite_v1` passages + questions (gist can't reconstruct made-up entities → reconstruction genuinely requires storage).
- **Metric** — EM + containment/recall (verbosity-robust; [[feedback_qa_correctness_metric]]), AR-decode eval (`eval_per_family.py`), TF train.
- **Controls** — `zero_memory` (OFF) exists; verify/add `shuffle_memory` (SHUF) for the `REAL ≫ SHUF ≫ OFF` gate.

## What's NEW

1. **Baseline encoder variants** — ICAE / CCM / Beacon faithful compressors emitting `[B, M, d]` memory through the shared encoder interface.
2. **Matched float budget** — every arm forced to the same `M·d·2` footprint; **report by floats/bytes, not slot count** (Beacon's per-layer K/V silently carries ~30× more capacity per "slot").
3. **REAL/SHUF/OFF** wired into the QA eval if not already.
4. *(later)* graph **bind/delta** write redesign ([[research_memory_sidecar_binding]]).

## Shared interface (the apples-to-apples contract)

Every arm is **an encoder that consumes a passage and emits memory tokens `[B, M, d_llama]`**; the **same** frozen-Llama closed-book reader consumes `memory + question → answer`. Only the encoder differs. This is exactly the existing `ReprLearningModel` contract — baselines slot in as new `VARIANTS` entries.

## Locked config (starter)

- Backbone: **frozen Llama-3.2-1B**.
- Objective: **EMAT closed-book key→value** (key/value-AE deferred).
- Data: `composite_v1` coined-entity passages → closed-book QA, ~96K samples.
- Budget: **M=64 × 2048 × 2 ≈ 262K floats** (matches the graph's known ~275K-float footprint).
- Steps: **BS=16, 6000 steps** ([[feedback_graph_default_bs_steps]]); full BPTT over the write ([[feedback_prefer_full_bptt]]).
- Gate: **REAL ≫ SHUF ≫ OFF** on the answer span + EM/containment off the floor — before scaling or touching the graph redesign.

## Build order

1. **Scope the harness** — confirm closed-book `compute_qa_loss`, SHUF/OFF controls, matched-budget reporting, composite data wiring. Patch gaps. *(next step)*
2. **Validate end-to-end with the graph arm** on the EMAT objective (pipeline shakedown).
3. **Port ICAE** (cleanest: frozen LM + LoRA encoder + slot embeds) as variant #1; verify `REAL ≫ SHUF`.
4. **Port CCM + Beacon**.
5. **Run the matched-budget comparison**; report the table.
6. **Then** the graph bind/delta redesign + re-run (where our engineering goes).

---

## ICAE port blueprint (variant #1)

**Interface mapping (streaming encoder contract):**
- `init_streaming_state(B, device, dtype)` → buffer that accumulates window embeds (list or growing `[B, T, d]`).
- `streaming_write(state, win_emb, win_mask, chunk_offset)` → append `(win_emb, win_mask)` to the buffer; return `(state, {})`. (ICAE isn't natively streaming — we accumulate, then run once at finalize.)
- `finalize_memory(state)` → concat buffer → `[B, T, d]`; append **M learnable slot embeddings** → `[B, T+M, d]`; run the **LoRA-adapted Llama** over it; take the **last-M hidden states** as `memory [B, M, d]`; `_NormMatch(d_llama)` to match token scale (same as other variants). M = `cfg.n_flat_codes` (the shared budget).

**Weight-sharing decision (the crux).** ICAE's encoder = frozen base + *encoder*-LoRA; decoder = the *same* frozen base. Our `apply_lora_to_llama` is non-toggleable, so two LoRAs on one base instance would collide. Two options:
- **(A) Separate frozen base copy for the encoder** — unambiguously faithful (encoder = full Llama + its own LoRA), zero adapter-collision risk; cost = one extra ~2.5GB bf16 base. Simplest/safest.
- **(B) Share the single base by reference + toggleable encoder-LoRA** — memory-efficient (keeps the one-shared-Llama design) but needs `apply_lora_to_llama` to support enable/disable around the encode pass.
- **Decision: start with (A)** for a correct first number; add (B)'s toggle as an optimization once the arm is validated. Document the extra base in the budget table.

**Reuse:** `apply_lora_to_llama(rank, alpha, target_names)` (same helper the decoder uses) for the encoder-LoRA; `_NormMatch` for the output scale; `load_frozen_llama(cfg.llama_model)` for option (A)'s base.

**New config fields:** `icae_lora_rank` (default 32, ICAE uses ~512 on 7B → scale down for 1B), `icae_lora_alpha`, `icae_n_slots` (default = `n_flat_codes` for matched budget).

**Wiring:** new `ICAEBaselineEncoder` in `encoder.py`; add to `ReprLearningModel.VARIANTS`; export in `__init__`. Option (A) means the encoder self-loads its base in `__init__` (no post-construction wiring needed).

**Smoke gate (the safety net — required before declaring done):** forward a tiny batch → `memory.shape == [B, M, d_llama]`; backward → encoder-LoRA + slot-embed grads are finite and non-zero, base grads are None/zero; then a 200-step EMAT run must show `REAL < SHUF` separating (memory used) before scaling.
