# EMAT-first baseline comparison — execution plan

*Branch: `memory-experiment` (off `main`). Direction: [[project_stage_a_direction_emat]]. Diagnosis behind it: docs/mamba_two_lenses_memory.md.*

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

---

## CCM port blueprint (variant #2) — from research a15438 (CCM, Kim et al. ICLR 2024, arXiv:2312.03414, MIT)

**What CCM is:** recurrent **KV-cache** compression. At step t, append `n_comp` `<COMP>` tokens to chunk c(t); they attend over c(t) + the prior memory; harvest their per-layer K,V as the step's compressed memory; fold via **concat** (grows) or **merge** `Mem(t)=(1−1/t)·Mem(t−1)+(1/t)·h(t)` (fixed). **Trainable:** a **conditional LoRA** (rank 8, alpha 16, dropout 0.05, targets **q/k/v/o**) **gated to fire ONLY on `<COMP>` positions** (`out = W·x + 1[x=COMP]·ΔW·x`) + the `<COMP>` token embeddings. Base frozen. The COMP-gate is CCM's signature (keeps text processing frozen so the model can't bypass memory).

**Interface mismatch & resolution (same as ICAE):** native memory = per-layer KV (can't feed our prepend-`[M,2048]` decoder). Port = take the **COMP tokens' last-layer hidden states** as the M memory vectors. Caveat **C1**: drops per-layer KV injection → per-unit capacity `16384→2048` floats (8×) on Llama-3.2-1B; compensate with `n_comp>1`; report by floats.

**Interface map (recommend MERGE fold first — fixed budget, cleanest matched comparison):**
- `init_streaming_state` → `{mem: [B, 0 or n_comp, d], t: 0}`.
- `streaming_write(window)` → run own-copy LoRA-Llama over `[mem_hiddens ++ window ++ COMP]`; take COMP last-layer hiddens `h_comp[B,n_comp,d]`; **merge:** `mem ← (1−1/t)·mem + (1/t)·h_comp` (fixed `M=n_comp`); **concat:** stack (`M=n_comp×n_windows`). Per-window recurrence (BPTT through the fold; checkpoint each window like the others).
- `finalize_memory` → return `mem` (`_NormMatch`).

**The hard part — COMP-gated LoRA:** a custom `CompGatedLoRALinear` wrapping the frozen base q/k/v/o Linears with `lora_A/lora_B` + a per-forward `_comp_mask [B,T,1]` (1 at COMP positions, set on all wrappers before each forward). This is more invasive than ICAE's `apply_lora_to_llama` — implement + unit-test the gate (LoRA contribution is exactly 0 at non-COMP positions).

**Weight-share:** own frozen base copy (option A), as ICAE — the COMP-gate already restricts the adapter, but a separate base removes all collision ambiguity.

**Config (new block):** `ccm_lora_rank=8, ccm_lora_alpha=16, ccm_lora_targets=("q_proj","k_proj","v_proj","o_proj"), ccm_n_comp` (=n_flat_codes for matched M), `ccm_fold="merge"` (vs "concat"). Recipe-faithful rank 8 (not ICAE's 32) — note rank-faithful vs LoRA-budget-matched.

**Smoke gate (required):** `memory==[B,M,d]`; grads reach LoRA+COMP-embeds, base frozen; **COMP-gate unit test** (zero LoRA delta at non-COMP positions); 3-step EMAT run no-NaN; then REAL<SHUF before scaling. Reimplement in our interface (don't lift CCM's HF-attention fork); may lift `peft_custom/lora.py` comp-mask logic as reference (MIT, attribute).
