# graph_v6 diagnosis — why the memory scores below the no-memory floor

Companion to `docs/repr_learning_v2_1_results.md` (the saved 7-arm result table).
Commit context: results `63a678e`; diagnosis from a 5-agent debug sweep + a decisive
read-utilization probe (`scripts/repr_learning/probe_graph_v6_inject.py`) on the trained
`v2_1` graph_v6 checkpoint (best step 3500). 2026-06-01.

## The result being explained
LLM-judge (macro %): ceiling(full-ctx) **38.2** ≫ mamba 13.2, **floor(no-memory) 13.0**,
continuous 12.5, MT 11.2, **graph_v6 10.9**, flat 10.9. graph_v6 sits *at or just below*
the no-memory floor — adding the memory makes Llama no better (and marginally worse) than
having no memory at all.

## Symptom (empirically nailed)
Read-utilization probe — graph_v6 decoded on the same questions with the inject **ON / OFF /
SHUFFLED**:

| condition | EM | Containment |
|---|---|---|
| REAL (correct facts) | 3.1 | 3.1 |
| OFF (inject disabled) | 3.1 | 3.1 |
| SHUFFLE (wrong facts) | 1.6 | 1.6 |

- **REAL ≈ OFF** → injecting the correct facts yields the *same* answers as injecting nothing:
  the decoder routes around the memory; correct facts contribute ~0.
- **SHUFFLE < REAL** → wrong facts make it worse: the inject pathway is *live* (it perturbs
  logits) but its content is noise-like.

Independently, aligning the 288 eval samples shared with the floor: the inject **changes 72%
of predictions** yet the net is **negative** (floor-only correct 11 vs graph_v6-only 5). Failures
are coherent, on-format, same-type/wrong-value flips (`James Taylor`→`Andrew Preston`,
`First person`→`Third person`, `garden`→`kitchen`, `none`→`one`), concentrated in babiLong
state-tracking (8 losses/3 wins) and narrative_qa (2/0). It is **not** gross corruption (0 empty,
0 garbled, floor-matched answer lengths) — it is a **weak/mis-aimed retriever plus an irreducible
noise inject**.

## Root cause (ranked; multi-agent + literature consensus)

1. **The inject can't be a true no-op (the direct cause of *below*-floor).** The ReZero gate
   `scale_raw` (init 0.05) is *floored* by its `tanh` parameterization and the optimizer pushes it
   **down monotonically** across training (`rezero_scale_eff` 0.0495→0.0460) — i.e. SGD found the
   read net-harmful and is trying to delete it, but it can't reach 0, so a ~5%-magnitude residual
   (`fact_norm≈37`) is *always* injected. The M=0 floor has literally zero inject; graph_v6 has an
   irreducible noise residual. *(ReZero, Bachlechner 2020 — residual meant to grow if useful; here it shrinks.)*

2. **Fact-independent corruption path (implementation bug).** `GraphV6FactReader` uses the v5
   `AttnBlock`, whose `forward` returns `q + out` (`graph_substrate.py:317`). So
   `inj = eff·W_out(r)` contains `eff·W_out(q_proj(hidden))` — a learned linear map of the *hidden
   state itself*, re-added to the residual stream **independent of any retrieved fact**. Part of the
   inject is pure noise w.r.t. memory. *(graph_substrate_v6.py:248-252.)*

3. **Ungated, always-on, full-support inject — the core disadvantage vs the prepend baselines.**
   The read adds a residual at *every* position unconditionally (`eff` is a query-independent
   per-channel scalar; full-support softmax over all 196 fact-tokens; no NULL/sink). The frozen+LoRA
   decoder **cannot down-weight it per-position** when irrelevant. The winning baselines
   (mamba/continuous/MT) present memory as *prepended tokens* that pass through Llama's **own
   attention softmax**, which can assign ~0 weight to junk. This is the RAG "distracting/irrelevant
   context degrades the generator" failure reproduced inside the residual stream.
   *(Amiraz; Cuconasu "power of noise"; Yoran; RippleEdits — in-context beats activation edits.)*

4. **Inject at mid-stack layer 8/16 — the "conform zone".** Ben-Artzy et al. show editing hidden
   states in the bottom ~2/3 makes the model *conform* to the edit (flips the answer), while the same
   edit in the top 1/3 is *ignored*. graph_v6 injects at layer 8 — the worst place for an untrusted
   memory. Late layers (12-14) would make a bad read harmless.

5. **Degenerate write substrate (read quality is upper-bounded by write quality).** Node bank
   40-70% collapsed (`node_collapse_cos` 0.39-0.84), `node_active_frac` falls 0.69→0.43 (hub
   collapse), write gates decay (node 0.28, edge 0.21→0.10). Collapsed nodes → soft-pointer src/dst
   endpoints blur → fact-tokens are *entity mixtures* → entity-swap errors. Even a perfect reader
   retrieves mush. (The doc's own anti-collapse fallback — node slot-competition — was never triggered.)

6. **Late-training degradation.** A 176-norm gradient spike in the *memory* pathway at ~step 5000
   (LoRA grad stayed ~5), train loss rebounded 1.90→2.27; best ckpt = 3500. The evaluated mechanism
   is partially degraded.

Note: the no-op-free goal *succeeded mechanically* — `state_effect` grew 0.31→5.2 (state strongly
modulates facts). But making state load-bearing on a collapsed basis just **amplifies a noisy
signal**. Mechanically alive ≠ functionally useful.

## Fix plan (ranked by leverage)

- **F1 — Make the read no-op-able and gated** *(fixes #1, #3, #6=H8)*. Per-position relevance gate
  `gate = sigmoid(g_proj(hidden))`, `inj = gate·eff·W_out(r)`; add a learnable **NULL/sink**
  fact-token so a no-match query pools ~0; let ReZero reach exactly 0 (start at 0, drop the tanh
  floor) so a useless read collapses instead of lingering as noise.
- **F2 — Kill the query-leak** *(fixes #2)*. Zero-init `W_out`; in `GraphV6FactReader` use the raw
  cross-attention output, NOT `AttnBlock`'s `q+out`, so the inject is a pure function of facts.
- **F3 — Move the inject to a late layer (12-14)** *(fixes #4)*. Single highest-leverage hyperparameter;
  sweep `graph_v6_inject_layer`.
- **F4 — Fix the write substrate** *(fixes #5)*. Node decorrelation/whitening (the codebook-orth
  machinery exists) + node slot-competition; target `node_collapse_cos`→0, `active_frac`→1.
- **F5 — Real soft-top-k + `fact_key`** *(fixes #3)*. Sparse (temperature-sharpened / top-k +
  straight-through) selection with a separate query-conditioned readout, instead of full-support blend.
- **F6 — Decisive A/B: prepend vs inject.** Project fact-tokens → d_llama and **prepend** them as K
  memory tokens (the winning baselines' interface). If prepend > inject, the residual channel is the
  corruptor and we switch interfaces. *(RippleEdits: in-context > activation edit.)*
- **Hygiene** — persist graph_v6 health telemetry into the eval summary; add a `ruler_niah` needle
  unit test (if the magic number is in context and the read works, graph_v6 MUST beat vanilla's 0.0);
  clip the memory-pathway grad separately / lower memory LR to kill the step-5000 spike.

**Recommended first cut (v6.1):** F2 + F1 + F3 together (cheap, targeted at the *below-floor*
mechanism) and re-run the graph_v6 arm only vs the saved floor/ceiling. If that clears the floor but
doesn't beat the baselines, escalate to F4/F5/F6 (substrate + interface).

## Provenance
- Probe: `scripts/repr_learning/probe_graph_v6_inject.py` (REAL/OFF/SHUFFLE on v2_1 ckpt).
- Telemetry: `outputs/repr_learning/v2_1_graph_v6_baseline/jsonl/graph_v6_baseline.jsonl`.
- Predictions: `outputs/repr_learning/eval_per_family/eval_v2_1_graph_v6_baseline*_per_sample.jsonl`
  vs `eval_v2_1_vanilla_llama*_per_sample.jsonl`.
- Key cites: Ben-Artzy (layer conform/ignore zones); Amiraz / Cuconasu / Yoran (RAG distractor harm);
  Cohen RippleEdits & Li (KE distortion/ripple; in-context > edit); Bachlechner ReZero; Perez FiLM;
  Grover "Is One Token All It Takes" (pooled graph reads unstable; LoRA only partially stabilizes).
