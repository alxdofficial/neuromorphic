# Trajectory-Memory: diagnostic metrics + per-version baselines

Single source of truth for "is the architecture working?" questions across
versions. Every claim about a model should cite a row in one of these tables.

Status: 2026-05-17, after v2 vocab-trajectory 10K-step run on composite_v1
and the audit that recovered ~7K lines of deleted diagnostics from git.

---

## Section 0 — Recovery status

The `bf86253` "purge v1 from main" commit deleted substantial diagnostic
infrastructure. Status after recovery:

| Asset | Status | Location |
|---|---|---|
| **Output JSONs** (em_vanilla.json, em_ours.json, gap_decomp_*, scale_sweep, etc.) | **PERMANENTLY LOST** — never committed | none — would need re-run |
| Old `docs/baseline_numbers.md` (336 lines) | recovered from git | `docs/baseline_numbers_historical.md` |
| Old `docs/bench_results.md` (402 lines) | recovered | `docs/bench_results_historical.md` |
| Old `docs/wave1_retrieval_pretraining.md` (594 lines) | recovered | `docs/wave1_retrieval_pretraining_historical.md` |
| Old `docs/wave1_v3_protocol.md` (155 lines) | recovered | `docs/wave1_v3_protocol_historical.md` |
| Old `docs/profile_analysis.md` (387 lines) | recovered | `docs/profile_analysis_historical.md` |
| Old `docs/plan_trajectory_memory.md` (1880 lines) | recovered | `docs/plan_trajectory_memory_historical.md` |
| Old diagnostic scripts (13 files, 3,231 lines) | recovered | `scripts/eval/legacy/` |
| v1 trajectory codebase | preserved | branch `abandoned/trajectory-memory-v1` (tip `7bec656`) |

What we have for v1 numbers post-recovery: published numbers in the
historical baseline doc, but the per-doc/per-distance JSONs that backed
them are not on disk. Any new comparison requires re-running.

---

## Section 1 — Complete metric taxonomy

Twelve families. Marked as **[legacy]** if recovered from deleted scripts,
**[live]** if currently emitted by the v2 trainer, or **[planned]** if
still to be wired.

### 1A. Loss / accuracy — headline numbers

| Metric | Definition | Status | Where |
|---|---|---|---|
| `loss` | Total = answer_loss + Σ(coef × aux) | **live** | train.jsonl |
| `answer_loss` | NLL on answer-content tokens (mean per token) | **live** | train+val.jsonl |
| `answer_acc` | argmax-match accuracy on answer-content tokens | **live** | train+val.jsonl |
| `val_loss` / `val_acc` | held-out variants | **live** | val records |
| `per_key_loss` | per-task answer NLL | **live** | val records |
| **NTP CE** (full vocab, all valid tokens) | bench_vanilla_llama.py — unweighted per-source CE | **[legacy]** | needs port to v2 |
| **Answer-span CE** | bench_vanilla_llama.py — CE on `valid_mask & is_answer` only | **[legacy]** | needs port |
| **First-token-only NLL** | NLL on JUST the first answer-content token (kills teacher-forcing leak) | **[planned]** | not wired |

### 1B. Routing health (the v1 collapse failure mode)

| Metric | Definition | Healthy | v1 catastrophic |
|---|---|---|---|
| `w_unique_per_window` | mean unique cells visited by write trajs | > 0.10 · K_max | 0.003 |
| `r_unique_per_window` | same for read | > 0.10 · K_max | collapsed to 1 |
| `w_unique_per_traj`, `r_unique_per_traj` | per-trajectory | > 0.3 · K_hop | 1 |
| `read_entry_entropy`, `write_entry_entropy` | Shannon entropy of entry distribution | > 1 nat | ≈ 0 |
| `aux_lb` | Switch-style load-balance loss | falling, MA < 100 | exploded to 10⁴+ |
| `aux_z` | router z-loss | falling, MA < 300 | unbounded |
| `entry_logits_max` | max entry routing logit | < 10 | > 100 saturated |
| **Lifetime utilization** (% cells ever written / read) | running counter across full run | > 50% / > 20% | **v1: 8.7% / 3.7%** |
| **`r_uf` uniform-random check** | bench_vanilla_llama old: r_uf=1−(N−1/N)^K_read; collapsed routing matches this exactly | should DIFFER from uniform formula | **v1: 0.224 matches `1-(4095/4096)^1024` exactly** ⇒ uniform random |
| `usage_ema` per concept | (legacy) running EMA of cell use | should have peaks | **v1: max 0.0006 = 1/N (flat)** |
| `r_self_overlap` | within-trajectory revisits | > 0 | **v1: 0.0** |

### 1C. Read↔write alignment

| Metric | Definition | Healthy |
|---|---|---|
| `rw_overlap_entry` | Jaccard of read-entry ∩ write-entry for target fact | > 0.5 |
| `rw_overlap_hop` | same for non-entry hops | > 0.1 |
| `rw_overlap_all` | combined | > 0.3 |
| `rw_overlap_target` | restricted to target fact vs distractors | > 0.4 |

### 1D. Edge / state health

| Metric | Definition | Healthy |
|---|---|---|
| `mean_edge_state_norm` | mean L2 of populated edge_state | ≈ √D (32) |
| `n_active_edges` | populated edge count | grows then plateaus 50-80% N·K_max |
| `edge_active_fraction` | / (N·K_max) | as above |
| `mean_visit_count` | mean writes per active edge | grows; not too lopsided |
| `mean_edge_specificity` | cosine-distance EMA per edge | 0.3–0.7 |
| `mean_edge_age` | mean step-count since last visit | growing ⇒ stale edges |
| `mean_fan_out` | mean out-degree | > 0; plateau means topology used |

### 1E. Gradient health

| Metric | Definition | Healthy |
|---|---|---|
| `grad_norm` | total pre-clip | < 10 typical |
| `grad_norm_{module}` | per-module: entry_proj, walker.read/write, mem_inject, lambda_edge, read_attn, concept_ids | non-zero each |
| Grad spike count >50/1000 | how often pathological | < 5 |

### 1F. Contrastive alignment loss

| Metric | Definition | Healthy |
|---|---|---|
| `l_contrast_entry` | entry-cell InfoNCE (read entry positive = write target entry) | ↓ from ~3.7 to <1.5 |
| `l_contrast_per_step` | per-hop trajectory-state InfoNCE | ↓ from ~2.0 to <0.7 |

### 1G. Throughput / efficiency

| Metric | Definition | Healthy |
|---|---|---|
| `step_s` | wall-clock per training step | stable ±10% |
| VRAM peak | max GPU memory during step | < device limit |
| SM utilization | GPU utilization during step | > 80% compute-bound |
| Tok/sec (training) | BS × n_windows × T / step_s | varies by config |

### 1H. Concept-space health (SimVQ-specific, v2)

| Metric | Definition | Healthy |
|---|---|---|
| `concept_ids_norm_mean` | mean L2 of post-projection concept embeddings | stable, ~√D |
| `concept_ids_norm_cv` | CV of concept-ID norms | small (~0.05) |
| `concept_ids_pairwise_cos` | avg pairwise cosine of concept_ids | near 0 |

### 1I. Memory-contribution probes — THE "is memory helping?" tests

The most important section. v2 10K run revealed all gaps to vanilla can be
explained by format-learning + teacher-forced AR — memory itself contributed
zero. These probes are the only way to catch that.

| Probe | Definition | Healthy target | Status |
|---|---|---|---|
| **NLL with-mem vs no-mem (paired)** | same chunks, run twice: full memory vs empty manifold + skipped writes | gap > 0.5 nat per content token | **[planned]** ad-hoc this session showed 0.00 |
| **NLL with-mem vs vanilla no-ctx** | paired comparison to frozen Llama with question only | gap > 1 nat | **[planned]** ad-hoc this session |
| **NLL with-mem vs vanilla full-ctx** | paired comparison to Llama with 8 passages in context | small gap OK | **[planned]** ad-hoc |
| **First-token-only NLL** | score ONLY the first content token (kills teacher-forced leak) | matches with/without-mem gap above | **[planned]** |
| **Generation EM** | autoregressive decode, exact match | passphrase verbatim > 30% | **[legacy]** bench_em_accuracy.py |
| **Generation tok-acc** | argmax-on-answer-tokens (TF-EM upper bound) | > 50% | **[legacy]** |
| **Per-distance CE** | bins: 0-2K, 2K-5K, 5K-12K, 12K-24K, 24K+ | should be near-flat if memory works | **[legacy]** bench_em_accuracy.py |
| **Decode probe** | feed read trajectory to Llama (skip rest of context), inspect emitted tokens | content-relevant tokens, not gibberish | **[legacy]** decode_probe.py + viz_trajectory_decoder.py |
| **Memory readout norm** | hook `mem_inject.memory_fn` output | non-zero (>0.01), varying per question | **[planned]** |
| **Cross-question read divergence** | same chunk + 2 questions: are read trajectories distinct? | Jaccard < 0.5 | **[planned]** |
| **Scale sweep** | multiply trained scale_raw by f ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.5}, measure CE | non-monotonic with sweet spot ≠ 0 | **[legacy]** bench_scale_sweep.py — v1 result was strictly monotonic (memory = noise) |

### 1J. Gap decomposition (v1 legacy methodology)

Critical for honest accounting of WHY our number differs from vanilla.

| Probe | Definition | Status |
|---|---|---|
| **scale_zero (identity scaffold)** | force the bridge scale_raw to 0 → IntegratedLM is byte-identical to vanilla Llama. Measures scaffolding overhead | **[legacy]** bench_gap_decomp.py |
| **scale_trained** | normal memory readout | **[legacy]** |
| **scale_trained_readout_zero** | trained scale, but force memory_fn → 0. Sanity check for W_out bias contribution | **[legacy]** |
| **Memory injection cost** | scale_trained − scale_zero | should be ≤ 0 for "memory helps"; v1 was +0.09 (small cost) |
| **`inject_snr`** | ‖memory_injection‖ / ‖hidden_state‖ | < 0.1 if memory is a gentle modulation; > 0.5 means major rewrite | **[legacy training telemetry]** |
| **`scale_raw` final state** | learned per-feature gate values | should move from init if memory is useful | **[legacy]** v1: 0.0964 vs init 0.1 (never moved) |

### 1K. Interpretability / visualization (legacy, very valuable)

| Probe | Definition | Status |
|---|---|---|
| **Concept dictionary** | for each cell, synthesize length-1 trajectory + decode 100 samples + summarize | **[legacy stub]** viz_trajectory_decoder.py |
| **Interpolation panel** | decode along shortest graph path A→B between concept pairs | **[legacy stub]** |
| **Counterfactual probe** | take real trajectory, swap one hop, decode | **[legacy stub]** |
| **Manifold viz** | 3D projection of concept_ids; nearest-neighbor structure | **[legacy]** viz_manifold.py (311 lines) |
| **Concept↔language** | concept-language correspondence probe | **[legacy stub]** viz_concept_language.py |
| **Needle decode panel** | decode write_trajectory at needle position | **[legacy stub]** viz_needle_recall.py |

### 1L. Architectural ablations (need to rebuild for v2)

| Ablation | Question | Status |
|---|---|---|
| **Flat-bank baseline** | replace trajectory walks with top-K cell attention at same params | **v1 had it; needs port to v2 vocab-trajectory** |
| **Single hop (K=1)** | does multi-hop add anything? | needs flag in walker |
| **No graph (full N×N routing)** | is the small-world graph helping? | needs flag |
| **No decay gate** | is state-magnitude bounding load-bearing? | needs flag |
| **No load-balance loss** | regression test | needs flag |
| **No revival / no W-TinyLFU** | does eviction policy matter? | needs flag |
| **Backbone size up (3B)** | does architecture scale? | needs config |

---

## Section 2 — Required baselines (must be measured for every version)

These are the comparison points for **every** new version's diagnostic run.

| Baseline | Why | How |
|---|---|---|
| **Vanilla Llama no-ctx** | the floor — model has only the question | tokenize Q+A, frozen Llama forward, NLL on A-content tokens |
| **Vanilla Llama full-ctx** | the ceiling — Llama with all passages in context | tokenize 8×P+Q+A, frozen Llama forward, NLL on A-content |
| **Vanilla Llama 2K-cap** | the "memory matters" floor — Llama can't see >2K | rolling KV-cache with 2K cap |
| **scale=0 identity scaffold (ours)** | our scaffolding without memory injection. Isolates scaffold overhead | force `mem_inject.scale_raw = 0` |
| **Flat-bank at same params** | architectural ablation — does the graph buy anything over flat attention? | replace walker with top-K cell attn |

---

## Section 3 — Version inventory (Trajectory-Memory week)

Six distinct v1 versions + the v2 rewrite. All v1 versions are on
`abandoned/trajectory-memory-v1` (tip `7bec656`) — checkout the commit to
get that exact code state.

| # | Version | Commit | Date | Architectural change | Trained on composite_v1? |
|---|---|---|---|---|---|
| V1.0 | Trajectory + softmax-STE | `dcc61d4` | 2026-05-12 | per-cell state + softmax-STE routing | No — trained on older data |
| V1.1 | + Wave 1 v4 protocol | `d95f4b9` | 2026-05-13 | data protocol change, arch unchanged | possible |
| V1.2 | + flat-bank ablation + decode probe | `f2f77ea` | 2026-05-13 | flat-bank module added alongside | yes |
| V1.3 | + Mixtral routing fixes | `311c2aa` | 2026-05-14 | small init, noisy gating, lower temp, stronger LB | **never retrained** |
| V1.4 | + dial-back tuning | `c103ffe` | 2026-05-14 | noise=0, init std 0.01→0.05 | **never retrained** |
| **V1.5** | **+ per-hop contrastive loss** | **`7bec656`** | 2026-05-14/15 | InfoNCE for entry + per-hop trajectory state | **never trained on composite_v1** |
| V2.0 | vocabulary-trajectory rewrite | `6cb713c` | 2026-05-17 | vocab id_basis + SimVQ + sparse edges | Wave 1 v5 smoke |
| **V2** | **current** | **`65fe2f1`** | **2026-05-17** | + bf16 edges + RMS-norm + W-TinyLFU + all v2 fixes | **YES — 10K run we just did** |

---

## Section 3.5 — Comprehensive per-task NLL comparison (composite_v1, 800 paired chunks)

Measured 2026-05-18. All numbers in nats. Same val chunks paired across modes
within each model (so memory contribution Δ within a model is a clean paired
comparison; absolute comparison across models uses the same val sampler seed).

**Memory-required tasks** (where the answer is random content the model can't
infer from format): biographical, passphrase. **Format/state-tracking tasks**
(where the answer is computable from question + format priors): the other 7.

### Mean NLL/tok on content tokens — with memory active

| Task | V1.5 trajectory | flat-bank | V2 vocab | Llama no-ctx | Llama full-ctx |
|---|---:|---:|---:|---:|---:|
| biographical | 3.833 | 4.018 | 3.903 | 5.049 | 3.546 |
| boxes | 3.152 | 1.917 | 1.605 | 4.342 | 3.781 |
| calendar | 1.185 | 3.238 | 0.845 | 4.021 | 3.725 |
| knights | 0.522 | 0.536 | 0.519 | 3.771 | 3.722 |
| passphrase | 2.334 | 2.238 | 2.122 | 6.759 | 3.349 |
| preferences | 1.458 | 4.007 | 1.188 | 5.583 | 4.457 |
| revisions | 2.067 | 3.294 | 1.387 | 3.496 | 2.921 |
| theory_of_mind | 1.715 | 2.224 | 1.185 | 3.425 | 2.816 |
| triage | 1.749 | 2.304 | 0.880 | 6.341 | 5.079 |
| **__overall__** | **2.026** | **2.628** | **1.533** | **4.778** | **3.731** |

### Mean NLL/tok on content tokens — memory disabled

| Task | V1.5 no-mem | flat-bank no-mem | V2 no-mem |
|---|---:|---:|---:|
| biographical | 4.413 | 5.094 | 3.903 |
| boxes | 1.595 | 3.520 | 1.605 |
| calendar | 1.052 | 3.282 | 0.845 |
| knights | 0.521 | 0.517 | 0.519 |
| passphrase | 2.264 | 6.306 | 2.122 |
| preferences | 1.534 | 3.840 | 1.188 |
| revisions | 1.774 | 3.772 | 1.392 |
| theory_of_mind | 1.256 | 4.662 | 1.185 |
| triage | 0.931 | 3.652 | 0.880 |
| **__overall__** | **1.720** | **3.858** | **1.533** |

### Memory contribution Δ = (with-mem) − (no-mem). Negative = memory helps.

| Task | V1.5 Δ | flat-bank Δ | V2 Δ |
|---|---:|---:|---:|
| **biographical** (true retrieval) | **−0.580 ✓** | **−1.076 ✓** | 0.000 |
| **passphrase** (verbatim recall) | +0.069 | **−4.068 ✓** | 0.000 |
| preferences | −0.076 | +0.167 | 0.000 |
| boxes (state tracking) | +1.557 ❌ | **−1.602 ✓** | 0.000 |
| theory_of_mind | +0.459 ❌ | **−2.438 ✓** | 0.000 |
| triage | +0.818 ❌ | **−1.348 ✓** | 0.000 |
| revisions | +0.293 ❌ | **−0.478 ✓** | −0.005 |
| calendar | +0.133 | −0.044 | 0.000 |
| knights | +0.001 | +0.019 | 0.000 |
| **__overall__** | **+0.306 HARMFUL** | **−1.231 HELPFUL** | **−0.000 zero** |

### First-token NLL — the cleanest "did memory retrieve" probe (kills teacher-forcing leak)

| Task | V1.5 with | V1.5 no | V1.5 Δ | flat with | flat no | flat Δ | V2 with | V2 no | V2 Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **biographical** | 3.629 | 3.844 | **−0.215 ✓** | 3.626 | 5.010 | **−1.383 ✓** | 3.478 | 3.478 | 0.000 |
| **passphrase** | 4.262 | 4.155 | +0.106 | 3.822 | 10.017 | **−6.194 ✓** | 3.621 | 3.621 | 0.000 |
| preferences | 2.223 | 2.369 | −0.146 | 4.474 | 4.828 | −0.354 | 1.824 | 1.824 | 0.000 |
| knights | 0.694 | 0.720 | −0.026 | 0.735 | 0.836 | −0.101 | 0.687 | 0.687 | 0.000 |
| calendar | 1.525 | 1.427 | +0.097 | 3.581 | 3.690 | −0.109 | 1.125 | 1.125 | 0.000 |
| boxes | 3.659 | 1.852 | +1.807 ❌ | 2.160 | 4.268 | **−2.109 ✓** | 1.774 | 1.774 | 0.000 |
| theory_of_mind | 2.935 | 2.336 | +0.599 ❌ | 2.562 | 6.261 | **−3.698 ✓** | 2.277 | 2.277 | 0.000 |
| triage | 2.019 | 0.851 | +1.169 ❌ | 2.723 | 4.795 | **−2.072 ✓** | 0.786 | 0.786 | 0.000 |
| revisions | 2.233 | 1.857 | +0.376 | 3.108 | 3.837 | −0.729 | 1.547 | 1.541 | +0.006 |
| **__overall__** | **2.573** | **2.131** | **+0.442** | **2.949** | **4.829** | **−1.880** | **1.887** | **1.886** | **+0.001** |

### Verdict per-task

- **biographical (the canonical retrieval task)**: every architecture that's measured helps. V1.5 −0.58 mean / −0.22 first-token; flat-bank −1.08 / −1.38; V2 0.00 (memory off entirely).
- **passphrase (verbatim random recall)**: only flat-bank does real work (−4.07 mean, −6.19 first-token). V1.5 is essentially noise; V2 is zero.
- **state-tracking tasks (boxes, theory_of_mind, triage)**: flat-bank's memory helps hugely (−1.6 to −2.4 nat). V1.5's memory ACTIVELY HURTS these (+0.5 to +1.6 nat) — the contrastive loss likely overfit the trajectory for retrieval and made it noisy for everything else.
- **format-determined tasks (knights, calendar)**: memory neutral across all models (these don't need memory).

### Reproducibility check

Seed 42 → seed 43 (different val samples, same val set):
- V1.5 v1-v1_no_mem Δ: +0.306 → +0.190 nat (same direction, similar magnitude)
- V1.5 first-token Δ: +0.442 → similar pattern

The result is reproducible across val samples — V1.5's memory really is net-harmful in aggregate but per-task: helpful for biographical, harmful for state-tracking.

### How can V1.5 no-mem reach 1.72 nat without retrieval?

Because most content tokens in composite_v1 are *not* retrieval-required:

| Task | no-mem NLL | True memory required? |
|---|---:|---|
| knights | 0.52 | No — binary answer space |
| triage | 0.93 | No — task name in question structure |
| calendar | 1.05 | No — format-determined date |
| theory_of_mind | 1.26 | No — observable from QA setup |
| preferences | 1.53 | No — in-question hint + format |
| boxes | 1.60 | No — state computable from QA |
| revisions | 1.77 | No — most-recent-value rule |
| **passphrase** | **2.26** | **Yes — random three-word phrase** |
| **biographical** | **4.41** | **Yes — random named-entity fact** |

The 1.72 nat aggregate is heavily diluted by 7 tasks where the trained adapter
predicts format-determined content tokens without needing memory. Vanilla
Llama no-context gets 2.84 because it doesn't know the format space at all
(must predict from full 128K vocab). The 1.12 nat gap is mostly format priors
+ teacher-forced AR continuation within the answer span, NOT retrieval. The
honest "retrieval test" is the biographical + passphrase rows.

---

## Section 3.6 — Per-task read↔write overlap (composite_v1 val, 400 paired chunks)

Measured 2026-05-18. `rw_target_all = |R ∩ W_target| / |R|` — fraction of read
cells the read trajectory visited that are also in the target passage's write
cells. **rw_target_lift = rw_target_all − rw_distractor_mean** (mean overlap against the
7 non-target passages). Positive lift = read is task-specific (lands on target
more than distractors). Random baseline ≈ 0.008 (J·K_w/N = 32/4096).

### Overall

| Architecture | rw_target_all | rw_target_entry | rw_target_hop | rw_distractor_mean | **lift** | NLL Δ (mem) |
|---|---:|---:|---:|---:|---:|---:|
| V1.5 trajectory + per-hop contrastive | 0.124 | **0.442** | **0.005** | 0.126 | **−0.001** | +0.306 HARM |
| flat-bank | 0.169 | 0.414 | 0.092 | 0.172 | **−0.003** | **−1.231 HELP** |
| V2 vocabulary-trajectory | 0.180 | 0.317 | **0.189** | 0.120 | **+0.061** | ~0 |

### Per-task — V1.5 (entry routing precise, hops at random floor)

| Task | rw_target_all | rw_target_entry | rw_target_hop | distract | **lift** |
|---|---:|---:|---:|---:|---:|
| biographical | 0.110 | 0.407 | 0.001 | 0.106 | +0.005 |
| boxes | 0.101 | 0.422 | 0.004 | 0.097 | +0.004 |
| calendar | 0.176 | **0.585** | 0.002 | 0.176 | −0.000 |
| knights | 0.079 | 0.337 | 0.003 | 0.077 | +0.002 |
| **passphrase** | 0.053 | 0.188 | 0.007 | 0.104 | **−0.052** |
| preferences | 0.127 | 0.454 | 0.003 | 0.130 | −0.003 |
| revisions | 0.139 | 0.476 | 0.010 | 0.122 | +0.017 |
| theory_of_mind | 0.076 | 0.315 | 0.010 | 0.089 | −0.014 |
| triage | 0.222 | **0.691** | 0.008 | 0.206 | +0.016 |

**Pattern**: entry routing is highly task-specific (31–69% match vs ~0.008 random
floor). But after the entry, hops are at or below random across every task
(0.001–0.010). The trajectory walker takes random steps after the first cell.
**Net lift = 0** — the entry-cell precision is exactly cancelled by random hops.

### Per-task — flat-bank (lift = 0 by this metric — wrong probe for the architecture)

| Task | rw_target_all | rw_target_entry | rw_target_hop | distract | **lift** |
|---|---:|---:|---:|---:|---:|
| biographical | 0.208 | 0.500 | 0.066 | 0.209 | −0.001 |
| boxes | 0.167 | 0.394 | 0.095 | 0.163 | +0.005 |
| calendar | 0.167 | 0.452 | 0.096 | 0.171 | −0.004 |
| knights | 0.182 | 0.362 | 0.116 | 0.161 | +0.020 |
| passphrase | 0.156 | 0.500 | 0.071 | 0.156 | +0.000 |
| preferences | 0.152 | 0.398 | 0.095 | 0.160 | −0.009 |
| revisions | 0.151 | 0.390 | 0.090 | 0.169 | −0.018 |
| theory_of_mind | 0.158 | 0.393 | 0.093 | 0.175 | −0.017 |
| triage | 0.164 | 0.347 | 0.106 | 0.170 | −0.007 |

**Important caveat**: flat-bank operates via continuous top-K attention over all
N cells, weighted by query–cell similarity. "Visited cells" is the discrete
top-K subset reported for telemetry. The actual readout signal is a
weighted sum over many cells, not a discrete cell-set intersection. So
rw_target_lift ≈ 0 here doesn't mean the memory doesn't work — it means the
discrete top-K metric is *the wrong probe for flat-bank*. The −1.23 nat NLL
contribution proves the bridge IS extracting useful information; it's just
operating on continuous attention weights that this metric doesn't see.

### Per-task — V2 vocabulary-trajectory (POSITIVE lift, especially on state-tracking)

| Task | rw_target_all | rw_target_entry | rw_target_hop | distract | **lift** |
|---|---:|---:|---:|---:|---:|
| **boxes** | **0.495** | 0.226 | **0.922** | 0.343 | **+0.152** |
| **revisions** | **0.442** | 0.207 | **0.545** | 0.321 | **+0.121** |
| **theory_of_mind** | 0.184 | **0.750** | 0.000 | 0.028 | **+0.157** |
| preferences | 0.189 | 0.370 | 0.129 | 0.099 | **+0.090** |
| passphrase | 0.094 | 0.362 | 0.046 | 0.040 | +0.055 |
| knights | 0.061 | 0.426 | 0.000 | 0.020 | +0.040 |
| biographical | **0.027** | **0.048** | 0.004 | 0.020 | +0.007 |
| calendar | 0.098 | 0.479 | 0.000 | 0.095 | +0.003 |
| triage | 0.107 | 0.119 | 0.108 | 0.125 | −0.018 |

**The headline V2 result**: routing **does** find target-specific cells (overall
lift +0.061). Boxes hop overlap is 0.92 — V2 almost perfectly traces the
target passage's state-mutation trajectory. Theory_of_mind entry overlap is
0.75. Revisions hop is 0.55.

**But** V2's per-task NLL memory contribution is 0.000 on every task. So:
- V2's routing landed on the right cells
- The bridge MLP between memory readout and Llama's residual stream is what's
  broken — it can't extract a useful direction from the (correctly-retrieved)
  cell states

**Per-task asymmetry inside V2**: it excels at state-tracking tasks (boxes
0.92 hop overlap, revisions 0.55) but **fails on biographical** (0.027
rw_target_all, 0.048 entry overlap, 0.004 hop — almost no retrieval). The vocab
trajectory architecture is biased toward sequential state mutations over
random-fact lookup, possibly because biographical chunks present 8 unrelated
facts (no inter-fact structure) while state-tracking chunks chain through a
single entity's state.

### The three-architecture diagnosis

| Architecture | Routing | Bridge MLP | Mem Δ NLL | Failure mode |
|---|---|---|---|---|
| **V1.5** | Entry yes, hops random | Could pass entry signal | +0.306 harm | Random hops drown the entry signal |
| **flat-bank** | Continuous (this metric doesn't measure) | **Works** | **−1.231 help** | None — this is what success looks like |
| **V2** | Yes (+0.061 lift, +0.92 hop on boxes) | **Broken** | ~0 | Routing finds info; bridge fails to convey to Llama |

The bottleneck is **architecture-specific**:
- V1.5 needs hop-routing fixed (the trajectory walker is the bug)
- V2 needs the bridge MLP fixed (routing works, integration with Llama doesn't)

### Hop reachability probe (2026-05-18) — the topology is the V1.5 ceiling

#### What reachability measures

The trajectory walker is graph-constrained: at each hop k≥1, it can only
move to a cell that is in the current cell's K_max=32 outgoing-edge
neighbors. So whether the read can reach the target write's hop-k cell
depends on whether that cell IS in the read's current neighborhood. If
it isn't, the trajectory is geometrically blocked — no routing decision
(perfect loss or not) can pick it.

For each (chunk, read trajectory j, hop k≥1):

1. Read's previous-hop cell: `R_prev = read_visited[chunk, j, k-1]`
2. R_prev's K_max neighbors: `neighbors(R_prev)` — from the manifold's
   `edge_indices` (V1.5) or active `edge_dst` (V2)
3. Target write's hop-k cells across J=4 write trajectories:
   `W_target_k = {write_visited[chunk, target_idx, j', k] : j'=0..J-1}`
4. Reachable indicator: 1 if `W_target_k ∩ neighbors(R_prev) ≠ ∅`, else 0

Per-hop reachability at hop k = mean of the indicator across (chunks × J
read trajectories). Overall reachability = mean of per-hop values across
k=1..K_read-1.

**Random baseline** ≈ `K_max·J / N = 32·4 / 4096 ≈ 0.031`. If reach is
at or below this, routing isn't doing better than a random walker. Above
0.031 means the routing has landed on cells whose neighborhoods
preferentially contain target writes.

#### What routing efficiency measures

Derived ratio: **efficiency = rw_target_hop / reachability**.

- `rw_target_hop` = `|R_hop ∩ W_target_hop| / |R_hop|` — fraction of unique
  read non-entry cells that are in the target write's non-entry cell
  set (a precision-style measure).
- `reachability` = fraction of (chunk, j, k) events where the target
  was reachable at all.

Their ratio is interpretable as "given the target IS reachable, how often
does the routing pick it?" — the diagnostic that disambiguates **loss
failure** (low efficiency despite high reach) from **topology failure**
(low reach regardless of efficiency). The ratio can occasionally exceed
100% when the J read trajectories converge to a small set of cells that
all happen to match target writes (V2 boxes hits 146%), so it's a rough
indicator rather than a strict probability. A 4% efficiency unambiguously
means the loss is wasting reachability; 70% efficiency unambiguously
means the loss is using most of what reach allows.

#### Results

**V1.5 (400 chunks, composite_v1 val):**

| Per-hop reach | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1.5 | **0.48** | 0.08 | 0.06 | 0.07 | 0.04 | 0.07 | 0.04 | **0.12** |

V1.5 collapses 6× between hop 1 (0.48) and hop 2 (0.08). Hop 1 is high
because the entry contrastive landed the read's entry cell near the
target's entry cell — their immediate neighborhoods overlap. After one
step the read is at an essentially random cell and reachability drops
to the random-walker floor of `K_max·J/N = 32·4/4096 ≈ 0.031`.

Per-task V1.5 reachability is 0.09–0.20 across all tasks — the fixed
small-world ring + NPMI plasticity has approximately the same shape
regardless of task family.

**V2 (400 chunks, composite_v1 val):**

| Per-hop reach | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 | 0.34 | 0.23 | 0.21 | 0.29 | 0.25 | 0.29 | 0.28 | **0.27** |

V2 is **remarkably flat across hops** — W-TinyLFU eviction + NPMI
plasticity built edges that span the trajectories, not just the immediate
entry neighborhood. The 2.24× higher mean reachability is the topological
gap V1.5 cannot close with loss changes.

**Per-task V2 reachability** reveals the adaptive-topology story:

| Task | V2 reach | V2 rw_target_hop | efficiency = rw_target_hop/reach |
|---|---:|---:|---:|
| **boxes** | **0.633** | 0.922 | 146% (across J trajs) |
| **revisions** | **0.610** | 0.545 | 89% |
| triage | 0.412 | 0.108 | 26% |
| preferences | 0.315 | 0.129 | 41% |
| passphrase | 0.140 | 0.046 | 33% |
| **biographical** | **0.116** | 0.004 | **3%** |
| theory_of_mind | 0.101 | 0.000 | (entry-only) |
| knights | 0.062 | 0.000 | (entry-only) |
| calendar | 0.053 | 0.000 | (entry-only) |

State-tracking tasks (boxes, revisions) get **5× the reachability** of
biographical (0.63 vs 0.12). The dynamic topology builds dense edges
along chains of sequential state mutation, and sparse edges for tasks
without that structure.

**Biographical fails in V2 for the same reason V1.5 fails on everything**:
V2's biographical reachability (0.116) is essentially identical to V1.5's
overall reachability (0.120). When 8 within-chunk facts share no
sequential structure, the dynamic topology can't form the chains hop
routing needs. The W-TinyLFU eviction designed to handle named entities
didn't help here — the problem isn't "cell overloaded" but "no chain to
navigate."

### Routing efficiency: separating loss failure from topology failure

| Architecture | reach | rw_target_hop | **efficiency** | Diagnosis |
|---|---:|---:|---:|---|
| **V1.5** | 0.12 | 0.005 | **4%** | Both broken: bad topology AND bad routing supervision |
| **V2** | 0.27 | 0.189 | **70%** | Topology fixed; routing supervision adequate via edge-state |

V1.5 fails on *both axes*: only 12% of hops can reach the target, AND
within the 12% where it can, the contrastive loss picks correctly 4%
of the time. V2 wins on both: 2.3× reachability from dynamic edges,
plus 70% efficient at converting reachability into actual target hits.

### Implication: V1.5 is not loss-fixable

A perfect per-hop cell-index CE loss on V1.5 could at most push
rw_target_hop from 0.005 → ~0.12 (the reachability ceiling). That's a 24×
improvement and would prove the contrastive failure was real, but it's
still well below V2's 0.19 — and far below the >0.5 "healthy" target.

V1.5 needs both routing supervision AND topology adaptation (i.e.,
something like V2's W-TinyLFU + NPMI plasticity). The fixed small-world
ring + per-cell state architecture genuinely doesn't admit task-specific
trajectories. The thesis "fixed grammar over learned vocabulary" loses
regardless of what loss is applied on top.

V2's measured 0.27 reachability is what V2 actually bought beyond "tune
loss." It bought *the graph the trajectories need to exist on.* The
"learned topology over fixed-vocabulary" compromise wasn't a stylistic
preference — it was necessary.

---

## Section 4 — Per-version results

### Vanilla Llama-3.2-1B (no memory side-car)

| Metric | Value | Val set | Source |
|---|---:|---|---|
| Params total / trainable | 1.24B / 0 | — | architecture |
| NLL/tok full answer (no ctx) | **5.6753** | composite_v1 val, 80 chunks | this session |
| NLL/tok full answer (full ctx, 8 P) | **4.8130** | same | this session |
| Passphrase content NLL (no ctx) | 1.9730 | single chunk | this session |
| **NTP CE on bulk text** (legacy v1 era) | **2.4367** weighted | fineweb/wiki/slim/needle, 800 chunks | `outputs/vanilla_llama_train_floor.json` (lost) |
| NTP CE per source: fineweb_edu | ~2.3 | 200 chunks | lost |
| NTP CE per source: wikipedia | ~2.2 | 200 chunks | lost |
| NTP CE per source: slimpajama | ~2.3 | 200 chunks | lost |
| NTP CE per source: needle | ~2.7 | 200 chunks | lost |
| **Answer-only CE (full-ctx)** | 1.18 | needle val, 100 docs | lost |
| **Answer-only CE (2K-cap)** | 4.08 | same | lost |
| tok-acc (full-ctx) | 73% | needle val | lost |
| tok-acc (2K-cap) | 41% | same | lost |
| Per-doc EM (full-ctx) | 6% | same | lost |
| Per-doc EM (2K-cap) | 3% | same | lost |
| FWD throughput (single fwd, BS=8) | ~0.01s | — | this session |

### V1.5 — trajectory v1 + per-hop contrastive (`7bec656`)

| Metric | Value | Status |
|---|---:|---|
| Trained on composite_v1? | NO | needs retrain |
| Trained on Wave 1 v4? | Possibly (need to check) | TBD |
| All other metrics | unknown | need run |

### V1.0–V1.4 — earlier v1 variants

All from `abandoned/trajectory-memory-v1` parent commits. None trained on
composite_v1. Historical numbers we have from the v1 era (Wave 1 v4 val set,
**not** composite_v1):

| Variant | val_loss @ step 10K | cells written | EM | tok-acc | answer-CE |
|---|---:|---:|---:|---:|---:|
| V1.0 pre-Mixtral | 1.23 | 357/4096 (8.7%) | — | — | — |
| Flat-bank baseline | **1.15** | 2,214/4096 (54%) | — | — | — |
| V1 (older bug-era, KV-cache uncapped) | 3.12 answer-CE | — | 5% | 52% | 3.12 |
| V1 (2K-cap inference, post-patch) | 6.5 answer-CE | — | 0% | 19.5% | ~6.5 |

Note: per-distance CE for V1 (2K-cap inference): {0-2K: 4.00, 2K-5K: 6.22, 5K-12K: 8.09, 12K-24K: 7.32, 24K+: 8.24}. Confirmed memory cannot carry information past the 2K window.

V1 era param state (best ckpt @ step 7000):
- `r_uf = 0.224 ± 0.002` — exactly matches `1 − (4095/4096)^1024` (uniform random)
- `usage_ema` max = 0.0006 (= 1/N, fully flat)
- `r_self_overlap = 0.0` (no revisits)
- `scale_raw` final mean = 0.0964 (init was 0.1 — **never moved**)
- `inject_snr ≈ 0.65` — injection has 65% of hidden-state norm (not gentle)
- Gap decomp: scale_zero CE = 2.37 / scale_trained CE = 2.46 / **memory cost = +0.09 nat**
- Scale sweep: strictly monotonic, no sweet spot — memory readout is noise

### V2 — vocabulary-trajectory (current, `65fe2f1`), 10K-step ckpt

#### Architecture summary
- N=4096 concepts × D=1024 with SimVQ id_basis + id_proj reparameterization
- K_max=32 edges/concept (≈ 131K total edge slots)
- J=4 parallel trajectories, K_read=K_write=8 hops
- Llama-3.2-1B frozen except adapter (~6.3M)
- 48.2M trainable

#### Loss / acc trajectory
| step | train_loss MA | train_ans MA | train_acc | val_loss | val_acc |
|---:|---:|---:|---:|---:|---:|
| 1000 | 2.85 | 2.56 | 0.50 | 2.00 | 0.54 |
| 3000 | 2.02 | 1.80 | 0.55 | 1.66 | 0.57 |
| 5000 | 1.80 | 1.64 | 0.57 | 1.52 | 0.59 |
| 6000 | 1.67 | 1.50 | 0.58 | **1.44** | **0.59** |
| 9500 | — | — | — | 1.42 | 0.62 |
| 10000 | 1.66 | — | — | 1.66 | 0.55 |

#### Routing health
| step | `w_unique_per_window` MA | `aux_lb` MA | `aux_z` MA |
|---:|---:|---:|---:|
| 1K | 5.88 | 710 | 329 |
| 5K | 8.95 | 61 | 116 |
| 10K | ~7.8 | ~50 | ~80 |

#### Memory-contribution probes ⚠️
| Probe | Value | Verdict |
|---|---:|---|
| NLL with-mem | 1.79 | vs vanilla no-ctx 5.67 |
| NLL no-mem (empty manifold, no writes) | **1.79** | **identical** |
| NLL vanilla no-ctx | 5.67 | gap of 3.88 nat looks impressive… |
| NLL vanilla full-ctx | 4.81 | …but |
| Passphrase content NLL: with-mem | **1.974** | |
| Passphrase content NLL: without-mem | **1.974** | **= with-mem to 3 decimals** |
| Passphrase content NLL: vanilla no-ctx | **1.973** | **= both v2 modes** |
| **Verdict** | **memory contributes zero** | the v2→vanilla gap is all teacher-forced AR continuation + format learning, not retrieval |

#### Per-task val NLL (mean across 20 val records during training)
| Task | mean NLL | min | max | spread |
|---|---:|---:|---:|---:|
| calendar.free_at | 0.73 | 0.49 | 0.94 | 0.45 |
| knights.identity_of | 0.59 | 0.45 | 0.82 | 0.37 |
| revisions.how_many_revisions | 0.46 | 0.08 | 1.31 | 1.24 |
| theory_of_mind.has_seen | 0.84 | 0.55 | 2.01 | 1.46 |
| triage.what_blocks | 0.89 | 0.08 | 1.95 | 1.87 |
| triage.is_ready | 0.63 | 0.00 | 2.49 | 2.49 |
| theory_of_mind.where_belief | 1.58 | 1.24 | 2.84 | 1.61 |
| theory_of_mind.where_actually | 1.66 | 1.15 | 4.95 | 3.81 |
| preferences.preference_cancelled | 1.89 | 0.76 | 3.99 | 3.23 |
| boxes.final_state | 1.74 | 1.24 | 2.89 | 1.65 |
| preferences.preference_recall | 1.64 | 0.97 | 2.72 | 1.75 |
| passphrase.verbatim_recall | 2.26 | 2.04 | 3.20 | 1.16 |
| revisions.current_value | 2.40 | 1.44 | 5.50 | 4.06 |
| calendar.next_event_on | 2.66 | 1.42 | 6.61 | 5.19 |
| biographical.atomic | 3.65 | 2.45 | 4.83 | 2.38 |
| biographical.relational_1hop | 4.30 | 2.80 | 5.61 | 2.81 |
| biographical.relational_2hop | 5.86 | 3.77 | 9.00 | 5.24 |
| biographical.aggregation_which | 5.90 | 4.03 | 7.40 | 3.37 |
| biographical.temporal_years_between | 6.59 | 5.27 | 10.19 | 4.92 |

#### Throughput
| Metric | Value |
|---|---|
| Train step time (BS=8) | 1.15 s/step |
| Train tok/sec | ~17.2K |
| VRAM peak | 23.7 / 24.5 GB |
| GPU SM util | ~60% |
| Final wall clock | ~3.2 h |

---

## Section 5 — The plan

### 5A. First — build the unified diagnostic suite

A single command that, given a ckpt, emits **every metric in Section 1**.

```
scripts/eval/eval_v2_full.py \
    --ckpt outputs/wave1_v2/ckpt.10000.pt \
    --val-dir data/wave1/composite_v1/val \
    --output outputs/wave1_v2/eval_full.json
```

Output: JSON with all Section 1 families filled. Plus a markdown report
that drops into Section 4 of this doc as a new row.

Components:
- (port from legacy) bench_vanilla_llama: vanilla Llama no_ctx / full_ctx / 2K-cap CE + tok-acc
- (port from legacy) bench_em_accuracy: per-distance, generation-based EM (priority 2)
- (port from legacy) bench_gap_decomp: scale=0 / scale_trained for v2's mem_inject
- (port from legacy) bench_scale_sweep: scale-factor sweep
- (port from legacy) decode_probe: feed read trajectory directly to Llama
- (new) with-mem vs no-mem paired NLL — true ablation, not the broken scale=0 one
- (new) cross-question read divergence
- (new) memory readout norm + variance
- (new) first-token-only NLL (kills teacher-forcing leak)
- (live, already there) all 42 training JSONL metrics — log final-step values

### 5B. Second — run on V2 ckpts to fill in baseline_numbers.md

Run on `ckpt.7000.pt`, `ckpt.10000.pt`, and the saved `ckpt.pt` (whichever
is best). Estimated: 30-45 min per ckpt.

### 5C. Third — retrain selected v1 versions on composite_v1

User selected for retraining (paired with the new diagnostic suite):
1. **V1.0** (`dcc61d4`) — pre-Mixtral, softmax-STE
2. **V1.3** (`311c2aa`) or **V1.4** (`c103ffe`) — Mixtral routing fixes
3. **V1.5** (`7bec656`) — + per-hop contrastive loss (the final v1 state)
4. **Flat-bank baseline** — to be built inside v2 codebase for fair compare

Cost: each retrain ~3h (BS=8, 10K steps). Plus eval ~30 min each.
4 versions × 3.5h ≈ 14 hours of compute. Overnight job.

Caveats:
- V1.0 → V1.5 used `src/trajectory_memory/` which is on the
  `abandoned/trajectory-memory-v1` branch. Checking out + retraining will
  need worktree to avoid clobbering current v2 work.
- V1's `train_wave1_retrieval.py` uses CompositeRetrievalAdapter (which is
  still in `scripts/data/wave1/common/sampler.py` on the v1 branch).
  Should work out of the box on composite_v1.
- The flat-bank "port to v2" is a new module, not just a config flag.
  Worth doing because it's the cleanest "does the graph help?" ablation.

### 5D. Open questions to confirm with user before launching

1. Train each v1 version with same hyperparameters as v2 (BS=8, 10K, lr=3e-4)? Y/N
2. Use the existing composite_v1 train/val split for fair comparison? Y/N
3. For the flat-bank port: top-K attention over N concept_ids (no edges), parameter-matched to v2's walker. OK?
4. Should the v1 retrains run in a git worktree (`isolate=worktree`) so they
   don't disturb v2 dev on main? Recommended yes.
5. Diagnostic suite build first (~half day) before any retrain starts? Y/N

---

## Section 6 — Diagnostic-suite design notes

What each metric measures (the user's specific ask):

**Loss/accuracy:**
- `loss` / `answer_loss` / `answer_acc`: training signal magnitude — what
  the optimizer is responding to. **Misleading by itself** because the QA
  window contains the gold answer (teacher forcing leaks).
- `per_key_loss`: which tasks the model is learning. Detects task-level
  failure modes (e.g., biographical relational stuck high).

**Routing health:**
- `w_unique_*` / `r_unique_*`: how many distinct cells get used? Low
  values = routing collapse. v1's catastrophic collapse was at 0.003.
- `read/write_entry_entropy`: are entry cells diverse? Distribution
  measure rather than just count.
- `aux_lb` / `aux_z`: training-time regularizers. Should fall during
  training; if they don't, routing isn't optimizing.
- Lifetime utilization: % of N cells ever used. Reveals dead-cell problems.

**Read↔write alignment:**
- `rw_overlap_*`: do reads find cells writes used? Without alignment, the
  graph is decorative. Per-chunk, per-target, per-hop break-downs.

**Edge/state:**
- `mean_edge_state_norm`: bounded? v1 had explosion to 210 before RMS-norm
  fix.
- `n_active_edges` / `edge_active_fraction`: graph coverage.
- `mean_visit_count`: distribution — should be lumpy if some edges matter
  more than others.
- `mean_edge_age` / `mean_edge_specificity`: eviction/freshness health.

**Gradient health:**
- `grad_norm` total: detect spikes (numerical instability).
- Per-module: which modules are getting trained vs starving? Caught the
  v1 write-grad collapse.

**Contrastive losses:**
- `l_contrast_entry`: pushes read entry-cell toward write entry-cell.
- `l_contrast_per_step`: pushes per-hop trajectory states to align.

**Concept space:**
- `concept_ids_*`: SimVQ-specific. Detects embedding-norm collapse.

**Memory contribution (the most important):**
- with-mem − no-mem NLL: direct test. Zero = memory does nothing.
- vs vanilla no-ctx: does our adapter add value beyond Llama?
- vs vanilla full-ctx: how much retrieval are we doing vs just lookup?
- first-token-only NLL: removes teacher-forced AR leak in answer span.
- gen EM / tok-acc: honest decode-based test, no teacher forcing.
- per-distance CE: needle eval, distance bins.
- decode probe: feed read trajectory directly to Llama, inspect tokens.
- readout norm: is the memory bridge actually outputting nonzero values?
- cross-question divergence: does the same chunk's memory route to
  different cells for different questions? (If not, memory ignores Q.)

**Gap decomposition (v1 methodology):**
- scale_zero (scaffolding-only): isolates non-memory overhead.
- scale_trained (memory on): full system.
- Difference = pure memory injection cost. Should be ≤ 0 if memory helps.
- inject_snr: magnitude ratio. > 0.5 means memory is a big perturbation.
- scale_raw final value vs init: did the bridge learn to use memory?
- Scale sweep: monotonic ⇒ memory is noise; non-monotonic with sweet
  spot ⇒ memory carries useful signal.

**Throughput:**
- step_s / VRAM / SM-util: efficiency. Subordinate to correctness.
