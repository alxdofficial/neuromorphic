# Compression-objective direction (2026-06-12)

**Pivot rationale:** separate the two confounded concerns — (A) *what is the best
mechanism for compressing human language* and (B) *how to persist a fixed-footprint
memory without smearing/overwriting/contradiction*. The overnight graph_v9 failure
was substantially a confound: conservation (a persistence mechanism) destroyed the
compression signal. **Work A first; drop fixed-footprint for now** (capacity scales
with input). Householder/operators, conservation, and the apply-to-query read are
RETIRED for this line (they served state-tracking expressivity, which the
compression thesis doesn't need).

## Backbone: SmolLM2 (stress the memory with a weaker prior)
Both confirmed drop-in (`LlamaForCausalLM`, `model.model.layers`, tied head):
- SmolLM2-135M: d_model=576, 30 layers, 9 heads, vocab 49,152 (cached)
- SmolLM2-360M: d_model=960, 32 layers, 15 heads (downloadable)
Weaker prior → lower no-memory floor → wider floor-ceiling band → more pressure on
the code (composes with MAE high-mask). Plan: floor/ceiling band scan at 135M/360M/
1B, develop at the widest band, confirm the winner at ~1B before any absolute claim.

## Data: sentence-length distribution (the Pile dedup, SmolLM2 tokenizer, 1500 docs)
SINGLE sentence (n=69,884): p25=16 p50=27 p75=44 p90=76, mean 41. **≥24 tok keeps
only 58%** (Pile has many short headers/list/code lines). Heavy junk tail (max 44k
tokens = no-punctuation Pile garbage) → NEED an upper bound + punctuation filter.
SENTENCE PAIR (n=68,384): p25=40 p50=59 p75=90 p90=134, mean 77. **≥24 tok keeps
93%** — pairs are naturally longer and more uniform.
At ratio 8: working slot range ~3–12 (single) / 4–18 (pair); slot-count mass at 4–8.
Achieved ratio ≈ 7.6 (ceil rounding).

**Proposed unit/band:** sentence PAIRS, filtered 24 ≤ tokens ≤ 128, ratio 8 →
k = ceil(len/8) ∈ [3, 16], bucketed by k. (Pairs: better coverage + uniformity;
the band drops the junk tail and the too-short.) Note: the Pile is messy
(code/logs); **FineWeb-EDU (already wired in data_continuation) is cleaner prose**
and likely the better corpus — swap and re-measure.

## Capacity-relative baselines (the fairness mechanism)
- **Beacon: already capacity-relative** — α (beacon_ratio) compresses by a factor;
  set α = ratio, done.
- **ICAE / AutoCompressor / CCM: fixed M today** (`= cfg.n_flat_codes`). Make them
  per-bucket: allocate M_max learned slots, use the first k = ceil(len/ratio) per
  example (a prefix/"Matryoshka" code — prefixes must be valid compressions). Same
  k for our compressor → matched slot-for-slot per example.
- Bucket the dataloader by k so each batch is uniform-width.

## Objective
1. **MAE, high mask ratio** (≥85%): ingest sentence(pair) → code → drop input →
   fill masked positions in parallel (NOT teacher-forced NTP — NTP's local prior is
   why the floor was ⅔-guessable). Reuse data_continuation `mae` at sentence scale.
   REPORT the no-memory MAE floor (same mask, memory off) every run.
2. **Contrastive on the CODE** (in-batch InfoNCE, mean-pooled cosine, temperature):
   a sentence's code closer to itself than to other sentences' — the principled form
   of "make SHUF−REAL an objective" + the direct fix for the overnight's sep_cos≈1.
   Pair with MAE (alone it collapses to spread-out-but-useless). Watch recon-vs-
   contrastive tension; small weight.
3. **(future) JEPA** sentence→sentence prediction in the code space (depends on 1).

## THE headline experiment (or the thesis didn't matter)
Our coactivation-vocabulary compressor vs a FLAT bottleneck (ICAE / small Perceiver)
at identical k and identical MAE objective. Must beat, or match-with-a-named-benefit
(compositional generalization to unseen sentence STRUCTURES — hold out templates,
not just sentences; JEPA-readiness; interpretability). Else it's ICAE with extra
steps.

## Decisions (LOCKED 2026-06-12)
- Unit: **sentence PAIR**.
- Corpus: **FineWeb-EDU** (clean prose; already wired in data_continuation).
- Band/ratio: **24 ≤ tokens ≤ 128, ratio 8** → k = ceil(len/8) ∈ [3,16], bucketed by k.
- OUR model readout: **DEFERRED — our compressor is being redesigned.** Build the
  measurement apparatus (data + capacity-relative baselines + SmolLM2) first; the
  baseline floor/ceiling/quality curve is the reference our redesign competes against.
- start flat (no pyramid) when our model returns.

## Build order
1. **DONE: sentence-pair dataloader** (`src/repr_learning/data_sentence.py`).
   FineWeb-EDU local parquet → decode (Llama tok, cached) → segment → SmolLM2
   re-tokenize → filter 24-128 → length-bucket by k=ceil(len/8) (uniform-k
   batches, M=k per batch). Emits QABatch + `k_slots`/`n_tokens`. Smoke: clean
   pairs, batches uniform-k, k range 4-15 (val 720 docs; train has 13k).
2. **NEXT — the one design fork (need a nod):** the OBJECTIVE decode path.
   - AE (free now): teacher-forced reconstruct-the-pair via the existing QA
     forward — BUT teacher forcing lets the local prior cheat (the ⅔-guessable
     trap). Usable immediately as a baseline-of-the-baseline.
   - true MAE (recommended): mask ~85% of the span, predict masked positions in
     ONE forward; the decoder sees [MASK]/anchors at j<i, NOT the true token, so
     the local-prior cheat is removed. Needs a NEW forward (doesn't fit
     context→Q→A). Causal-decoder MAE is valid (position i sees its mostly-masked
     causal prefix + memory). + the no-memory floor control (same mask, mem off).
3. Capacity-relative baselines: ICAE/AutoComp/CCM prefix-of-M_max = k; Beacon α=ratio.
4. SmolLM2 backbone wiring + floor/ceiling band scan (135M/360M/1B).


## Finding — the MAE band requires TRAINING (not a zero-shot probe), 2026-06-12
Validated the true-MAE mechanic + SmolLM2 + data end to end. Key correction:
a FROZEN model can't do MAE zero-shot (it has never seen [MASK] or a memory
prefix), so BOTH floor and ceiling are TRAINED quantities. Evidence: with no
memory + 85% masked, frozen CE = 10.76 ≈ ln(49152) (uniform — correctly nothing
to predict); "ceiling" with a raw memory prefix is WORSE because the frozen model
can't exploit an untrained memory format. Plain next-token CE (which the model
CAN do zero-shot) = 3.46 / ppl 31.7 on the pairs → backbone+data sane, prior is
strong (the thing we bottleneck against). CONSEQUENCE: the floor/ceiling band
scan becomes a short TRAINING run (trained no-memory floor vs trained
full-length-memory ceiling), built alongside the MAE training path.


## Baseline setup (instantiated on SmolLM2-135M, d=576, M_max=16) — 2026-06-12
All four construct cleanly on SmolLM2 (Llama-arch holds; chat template auto-off).

| baseline | trainable | mechanism | capacity/bin (k = ceil(len/8) ∈ [3,16]) |
|---|---|---|---|
| CCM | 0.93M | rank-8 COMP-gated LoRA (120 linears) + k comp slots | k slots × d = k·576 floats |
| ICAE | 1.85M | rank-32 LoRA (60 layers) + M_max learned slots | k of M_max=16 slots, k·576 floats |
| AutoCompressor | 3.71M | LoRA (120 layers) + M_max recurrent slots | k slots, k·576 floats |
| Beacon | 16.59M | full q/k/v beacon projections (NOT LoRA) + α=8 | native α=8 ⇒ ~ceil(len/8) summary vecs |

Code size per bin (floats) = k · d: k=3 → 1,728 … k=16 → 9,216 (135M). Trainable
params are per-MODEL, fixed across bins (slots allocate M_max, use first k).

**FAIRNESS FLAG (echoes the overnight param-asymmetry):** trainable params span
0.93M–16.6M — an 18× spread. Beacon's 16.6M is full q/k/v projections, not LoRA.
A clean compression comparison needs these closer (match LoRA rank / Beacon to a
budget) OR param count reported as a first-class axis. DECISION NEEDED.

## Honest status (2026-06-12 end)
DONE+VALIDATED: sentence-pair dataloader; true-MAE mechanic; SmolLM2 drop-in;
baseline instantiation + param counts. NOT YET BUILT: compute_mae_loss decode
path (the new forward), capacity-relative per-batch-k wiring (use first k slots),
trainer task="sentence_mae", the trained floor/ceiling band scan. Our compressor:
deferred (redesign).


## Param-MATCHED config (~2M trainable, SmolLM2-135M) — applied 2026-06-12
User decision: match trainable params so the comparison measures MECHANISM, not
capacity (label: "param-matched mechanism comparison", NOT paper-faithful).
Anchor 2M (above CCM's natural 0.93M so it gains capacity; matching mostly
SHRINKS the heavy ones — the safe direction). Calibrated on 135M (d=576):

| baseline | setting | trainable |
|---|---|---|
| ICAE | lora_rank 34 / alpha 68 | ~1.97M |
| CCM | lora_rank 17 / alpha 34 | ~1.97M |
| AutoCompressor | lora_rank 17 / alpha 34 | ~1.98M |
| Beacon | wrap 4 layers (0,10,19,29), α=8 | ~2.21M (structural floor; flagged) |

Beacon can't LoRA-shrink (full q/k/v projections); 4 wrapped layers is the
min sensible (3 → 1.66M, too low). Matched within ~10%; reported per-model.
NOTE: ranks are d-dependent — RECALIBRATE for 360M (d=960) before those runs.
Our eventual compressor targets the same ~2M.


## Capacity-matching VERIFIED (2026-06-13) + MAE wiring complete
M (memory vectors) == k = ceil(len/8) EXACTLY for all 4 compressors across
k=3..16 (L=24,40,64,96,128 → M=3,5,8,12,16 for icae/ccm/autocomp/beacon). All
vecs d=576 ⇒ capacity-in-floats = k·576, identical per-example. Enforced by:
data computes k + buckets by it; ICAE/CCM/AutoComp allocate M_max=16 + slice to
first k (prefix/Matryoshka code); Beacon native α=8 also sliced to k. Vanillas
deliberately unmatched (floor M=0, ceiling M=full).

MAE training path complete + trainer-smoke validated end-to-end (train+eval+
save+summary; peak VRAM 1.8GB at bs16). Pre-flight bugs caught & fixed:
pad-token/vocab mismatch (SmolLM2), frozen-decoder-can't-learn-MAE (→ decoder
LoRA on all variants), argparse choices missing sentence_mae, logger keys
(n_content_positions/memory_shape). Decoder LoRA (rank16, ~0.9M) is shared
infra so the protocol is learnable; competitors = ~2M encoder + shared decoder
LoRA (icae total 2.89M). RUN LAUNCHED: 6 variants × 800 steps, SmolLM2-135M,
out-tag mae_135m. Headline metric = val_loss_recon (masked-reconstruction CE).


## RESULTS — 6-variant, 800 steps, SmolLM2-135M, MAE ratio-8 (2026-06-13)
| variant | params | val recon | OFF−REAL | SHUF−REAL | band % |
|---|---|---|---|---|---|
| vanilla_llama (FLOOR) | 0.92M | 7.07 | +0.06 | +0.05 | 0% |
| autocompressor | 2.90M | **6.57** | +0.95 | +1.01 | 12% |
| beacon | 3.14M | 6.67 | +0.80 | +0.82 | 10% |
| ccm | 2.89M | 6.72 | +0.75 | +0.75 | 8% |
| icae | 2.89M | 6.74 | +0.65 | +0.73 | 8% |
| vanilla_full_context (CEILING) | 0.92M | 2.93 | +6.99 | +7.54 | 100% |

Band = floor(7.07) → ceiling(2.93), width 4.14. "band %" = (floor−recon)/4.14.

**Three findings:**
1. **The benchmark is HEALTHY + discriminating** (vs EMAT): wide band (4.14);
   controls sane (floor OFF/SHUF−REAL ≈ 0 — no memory, correctly nothing to
   ablate; ceiling +7 — full context highly passage-specific).
2. **The compressors produce PASSAGE-SPECIFIC codes** — SHUF−REAL > 0 for ALL
   (+0.7 to +1.0), ≈ OFF−REAL. This is EXACTLY the property the EMAT graph runs
   never got (SHUF≈REAL). The MAE compression objective elicits specific codes
   where binding-EMAT could not. Methodological win for the pivot.
3. **But UNDERTRAINED at 800 steps + weak absolute compression:** every curve is
   still dropping (autocomp 6.75→6.57, ceiling 3.23→2.93 — not plateaued), and
   the compressors capture only ~8-12% of the band (barely above the no-memory
   floor). Either much longer training needed or ratio-8 + 85%-mask is hard at
   135M. Ranking (autocomp > beacon > ccm > icae) is PRELIMINARY — gaps ~noise.

**Next:** longer run (3-5k steps) to plateau + get a decisive baseline ranking
and a real "best baseline to beat"; then our redesigned compressor vs this band.


## RESULTS — 4000 steps (CONVERGED), SmolLM2-135M, MAE ratio-8 (2026-06-13)
Band: FLOOR 6.96 → CEILING 0.81 (ppl 2.2), width 6.15. Curves plateaued by ~3500.
| variant | recon ↓ | band% | OFF−REAL | SHUF−REAL |
|---|---|---|---|---|
| vanilla_llama (FLOOR) | 6.96 | 0% | +0.06 | +0.03 |
| **autocompressor** | **4.64** | **38%** | +6.10 | +4.75 |
| beacon | 4.94 | 33% | +4.76 | +4.16 |
| icae | 5.13 | 30% | +5.17 | +3.78 |
| ccm | 5.61 | 22% | +3.53 | +2.87 |
| vanilla_full_context (CEILING) | 0.81 | 100% | +10.40 | +15.31 |

**The benchmark is now proven:** wide band (6.15), converged, DECISIVE ranking
(autocomp > beacon > icae > ccm), healthy gates (OFF−REAL +3.5..+6.1 memory
helps; SHUF−REAL +2.9..+4.8 codes passage-specific). Contrast EMAT where nothing
separated.

**THE TARGET TO BEAT: AutoCompressor (recurrent), 4.64 recon = 38% of band**,
at ~2.9M params / M=k slots × d=576. Compressors capture only 22-38% of band →
big headroom (ratio-8 + 85%-mask is genuinely lossy). Our redesigned vocabulary
compressor competes against this number.
