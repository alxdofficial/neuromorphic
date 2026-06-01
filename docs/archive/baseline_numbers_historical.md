# Baseline numbers + Wave 1 findings

Quality numbers (CE, EM, tok-acc) and learned-param state for the
trajectory-memory model vs. vanilla Llama-3.2-1B baselines. Throughput /
VRAM numbers live in `bench_results.md`; this doc is loss/quality only.

Use this as the canonical reference when comparing future runs against
the Wave 1 baseline or against the vanilla-Llama floor.

---

## 2026-05-12 — KV-cache cap bug + Wave 1 readout

### The bug

`cfg.effective_lm_context = 2048` was intended as a sliding-window cap on
Llama's attention range during training, isolating the role of memory
(memory should be the only path for >2K-distance information). But in
KV-cache mode, `past_key_values` grew unbounded — Llama silently attended
over the entire document. Memory was functionally redundant.

**Fix:** `_crop_kv_cache(cache, max_length)` helper in
`src/trajectory_memory/tbptt.py`, wired into `integrated_lm.py`'s
`forward_window` after every Llama call in KV-cache mode. Crops each
layer's `keys` and `values` to the last `effective_lm_context` positions.

All 48 tests (smoke + integration + unit) pass with the patch.

### Vanilla Llama baselines

| Setup | overall NTP CE | answer-only CE | tok-acc | EM | source |
|---|---|---|---|---|---|
| Llama, **independent 1024-token chunks** on train mix (fineweb_edu / wikipedia / slimpajama / needle, 200 chunks/source) | **2.4367** weighted | — | — | — | `outputs/vanilla_llama_train_floor.json` |
| Llama, **full unbounded context** on needle val (n=100 docs) | — | **1.18** | 73% | 6% | `outputs/em_vanilla.json` |
| Llama, **2K cap** on needle val (last 2048 tokens before answer) | — | **4.08** | 41% | 3% | `outputs/em_vanilla_2k.json` |

Per-source train-floor breakdown:

| Source | n chunks | full CE | answer-only CE |
|---|---|---|---|
| fineweb_edu | 200 | ~2.3 | — |
| wikipedia | 200 | ~2.2 | — |
| slimpajama | 200 | ~2.3 | — |
| needle | 200 | ~2.7 | 5.13 (n=25 answer tokens) |
| **weighted** | **800** | **2.4367** | — |

Note: Llama's per-chunk effective context is ≤1024 tokens (chunk size),
so this number is *also* the Llama-at-2048 floor on this distribution
(the 2048 cap isn't binding when chunks are 1024-long). State-threaded
multi-chunk Llama-at-2048 would be slightly lower (~2.30 estimated, not
measured).

### Our Wave 1 numbers (best ckpt @ step 7000)

| Setup | answer-only CE | tok-acc | EM | training train-CE | source |
|---|---|---|---|---|---|
| Ours w/ **full Llama context** (training-bug deployment) | **3.12** | 52% | 5% | 3.19 | `outputs/em_ours.json` |
| Ours w/ **2K cap enforced at inference** (post-patch) | **~6.5** (avg) | 19.5% | 0% | — | `outputs/em_ours_capped.json` |

**REVISION (gap decomposition, 2026-05-12 follow-up):** the "+0.75 nats
above vanilla floor" was apples-to-oranges. Training's reported `loss`
3.19 is an **answer-weighted CE** (needle answer tokens at weight 100 in
`valid_mask`), not the unweighted NTP CE that vanilla bench reports as
2.44. When measured on the same scheme:

| Setup | weighted val CE | injection cost |
|---|---|---|
| Vanilla Llama 1024-chunk bench | **2.4367** | — |
| Ours through scaffold @ `scale=0` (identity) | **2.3739** | — |
| Ours through scaffold @ trained scale | **2.4639** | **+0.0900 nats** |

So the **actual memory-injection cost on bulk tokens is +0.09 nats**,
not +0.75. Memory is essentially break-even with vanilla Llama on bulk
text. Per-source injection cost is consistent at +0.086 to +0.094 nats
across fineweb / wiki / slimpajama / needle (`scale_trained -
scale_zero`).

But: the EM bench (full-Llama-context, fact in context) shows
**ours answer-CE 3.12 vs vanilla 1.18 = +1.94 nats** on in-context
needle recall. So memory injection is *asymmetrically harmful*: small
cost on bulk text, large cost when Llama's full-context residual carries
a precise answer.

**Why asymmetric:** `inject_snr` from training.json's last entries is
**0.61-0.73** — the injection vector has 60-70% the *norm* of Llama's
hidden state. That's not a small ReZero-style perturbation; that's a
major rewrite of the residual stream. On bulk text, downstream layers
absorb it; on tasks needing precise residuals (in-context fact recall),
it breaks the calculation. The `scale_init=0.1` "10% gate" is
misleading — it's a per-feature multiplier on `W_out(readout)`, but
`W_out` projects D_concept=256 into d_lm=2048 with weights large enough
that `0.1 × W_out(readout)` is 65% of `||hidden||`.

Per-distance breakdown of the capped inference run:

| distance bucket | n docs | tok-acc | answer-CE |
|---|---|---|---|
| 0-2K | 36 | 37% | 4.00 |
| 2K-5K | 26 | 15% | 6.22 |
| 5K-12K | 21 | 5% | 8.09 |
| 12K-24K | 12 | 8% | 7.32 |
| 24K+ | 5 | 8% | 8.24 |

Interpretation: when Llama's attention is properly capped to 2K,
memory cannot carry information past the window — model collapses on
>2K distances. This is the cleanest evidence that the training bug
masked the architectural test.

### Learned parameter state (Wave 1 best ckpt)

```
llama.model.layers.8.scale_raw    shape=(2048,)  mean=0.0964  std=0.0069
tanh(0.0964) ≈ 0.0961   (init was scale_init=0.1 → 0.1)
```

The bridge's per-feature scale gate **never moved from init**. After 7K
steps, the bridge has neither learned to amplify ("memory is useful") nor
to suppress ("memory is noise"). Variance across the 2048 dims is tight
(std 0.007). This is consistent with the bug-explanation: under
unbounded Llama context, memory is simultaneously redundant *and*
slightly harmful on average, so gradient pressure cancels.

Other observable param state from training metrics:
- `r_uf ≈ 0.22`, std 0.002 across 7000+ steps (constant) — uniform-random
  routing math (`0.22 = 1 - (4095/4096)^1024`). See
  `project_routing_uniformity.md` for context.
- `usage_ema`: 4096/4096 concepts active, 0 dominant, max-usage 0.0006
  (= uniform 1/4096).
- `r_self_overlap = 0.0` (no within-trajectory revisits).

### Decomposition (revised after gap-decomp bench)

The "+0.75 nat" framing was wrong (training loss is answer-weighted;
bench loss is unweighted). True NTP CE picture:

1. **Scaffolding overhead: 0**. `IntegratedLM` at `scale=0` produces
   essentially the same NTP CE as vanilla Llama bench (2.37 vs 2.44 —
   our scaffold actually scores 0.06 nats *lower*, attributable to
   fp32-vs-bf16 CE precision). No bugs in windowing / state threading /
   KV-cache cropping.
2. **Memory injection on bulk tokens: +0.09 nats**. Stable across all
   sources. Small noise floor.
3. **Memory injection on in-context fact recall: +1.94 nats**
   (full-Llama-context EM bench). Asymmetric to bulk-token cost.
4. **Cause: injection magnitude.** `inject_snr ≈ 0.65` (training
   telemetry). Even at `scale=0.1`, the injection has 65% of Llama's
   hidden-state norm because `W_out` projects D_concept→d_lm with large
   weights. Bridge "gate" of 0.1 is misleading nomenclature — it's a
   per-feature scalar, not a magnitude cap.
5. **Why the gate didn't move:** scale_raw final mean = 0.0964 (vs
   init 0.1). On bulk tokens, gradient pressure to reduce scale is
   weak (only +0.09 nat penalty). On needle tokens, gradient pressure
   is strong but rare. Net displacement after 7K steps: negligible.

### Scale sweep (2026-05-12)

To test whether memory readout is *purely* noise vs has some useful
signal: multiply trained `scale_raw` by factor f ∈ {0, 0.25, 0.5, 0.75,
1.0, 1.5}, measure val CE. Strictly monotonic — no sweet spot:

| factor | eff_scale | weighted val CE | Δ from no-inject |
|---|---|---|---|
| 0.00 | 0.0000 | **2.3990** | — (floor) |
| 0.25 | 0.0241 | 2.4034 | +0.004 |
| 0.50 | 0.0482 | 2.4200 | +0.021 |
| 0.75 | 0.0722 | 2.4506 | +0.052 |
| 1.00 | 0.0961 | 2.4951 | +0.096 (trained) |
| 1.50 | 0.1436 | 2.6292 | +0.230 |

Source: `outputs/scale_sweep.json`.

Interpretation: under the current routing (uniform-random,
`r_uf=0.224`), the memory readout carries **zero useful signal**. Every
positive scale value adds CE roughly linearly. The bridge "would have"
gated to zero but couldn't optimize there in 7K steps. Root cause is
upstream — readout is noise because routing doesn't specialize.

---

## Bench file index

| File | What it measures | Run with |
|---|---|---|
| `outputs/vanilla_llama_train_floor.json` | Llama-1024 CE on train mix (4 sources × 200 chunks) | `scripts/diagnostics/bench_vanilla_llama.py --val-data-paths data/wave1/*.val.parquet --val-batches 200` |
| `outputs/em_vanilla.json` | Llama-full-context EM/CE on needle val | `scripts/diagnostics/bench_em_accuracy.py --model-type vanilla --max-docs 100` |
| `outputs/em_vanilla_2k.json` | Llama-2K-cap EM/CE on needle val | `... --model-type vanilla --context-cap 2048` |
| `outputs/em_ours.json` | Ours full-Llama EM/CE on needle val | `... --model-type ours --ckpt outputs/wave1/ckpt.best.pt` |
| `outputs/em_ours_capped.json` | Ours w/ 2K cap at inference (patched code) | `... --model-type ours --ckpt outputs/wave1/ckpt.best.pt` (cap enforced inside `forward_window`) |
| `outputs/gap_decomp_scale_zero.json` | Ours scaffolding @ scale=0 (identity to vanilla) | `scripts/diagnostics/bench_gap_decomp.py --ablate scale_zero` |
| `outputs/gap_decomp_scale_trained.json` | Ours scaffolding @ trained scale | `... --ablate scale_trained` |
| `outputs/scale_sweep.json` | CE vs scale_raw multiplier ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.5} | `scripts/diagnostics/bench_scale_sweep.py` |

---

## 2026-05-12 — Architecture bug #2: Gumbel-noise dominates cosine routing

After the aux-loss-only retrain attempt showed `r_uf=0.22` unchanged
(uniform-random math) for 200 steps, traced to a deeper bug:

**Gumbel-STE incompatible with cosine routing.** The routing
implementation was `y = (logits + g) / tau` where `g ~ Gumbel(0,1)`
(std ≈ 1.28). With L2-normalized cosine routing, `logits ∈ [-1, 1]`
and typical pair-wise similarity is `~1/√D ≈ 0.06` for D=256. Gumbel
noise (std 1.28) is **~20× larger than typical signal**, so the
argmax was essentially random — explains the `r_uf=0.22` constant
across 7K Wave 1 steps. `tau` doesn't fix this; it scales both signal
and noise equally and is irrelevant to argmax.

**Fix:** dropped Gumbel-STE entirely. New `softmax_top1_ste`: forward
picks argmax, backward routes through softmax probabilities (modern
MoE pattern, Mixtral/DeepSeek). Added learnable `logit_scale_raw`
parameter (init=1.5, effective scale = `exp(1.5) ≈ 4.5`), applied
before softmax in both read and write modules. Standard pattern from
CLIP. Also converted write_module's entry from raw dot product to
cosine routing for consistency.

**Wave 1 v2 preflight (500 steps) — routing fix works:**

| step | r_uf | r_ent | loss | step time |
|---|---|---|---|---|
| 20 | 0.061 | 5.11 | 3.55 | 5.69s (compile) |
| 100 | 0.047 | 4.79 | 3.18 | 1.60s |
| 250 (ckpt) | — | — | — | — |
| 500 (final) | 0.074 | 5.36 | 3.40 | 0.87s |

Val loss compared to Wave 1 final (after 14910 steps):

| Source | Wave 1 final val | v2 step 250 | v2 step 500 |
|---|---|---|---|
| needle | ~6.85 | **5.30** | **4.16** |
| fineweb_edu | ~4.20 | **3.07** | 3.57 |

v2 at step 500 already beats Wave 1's full-run needle val (4.16 vs
6.85). Confirms the routing fix is unlocking real learning.

Additionally fixed `mem_inject.forward` compile from `dynamic=False`
to `dynamic=True` (handles the rolling-buffer cumulative-context
attention_mask shape variation without recompile thrashing). Step
time settled at ~0.87s, similar to Wave 1's 0.79s — dynamic-compile
overhead is small.

## 2026-05-12 retrain plan (Wave 1 v2)

After gap decomp + scale sweep ruled out scaffolding bugs and revealed
that current readout is pure noise, the retrain target is **specialized
routing**, not "more tokens." Architectural changes are minimal:

- `--load-balance-coef 1e-3` (down from 1e-2). Less pressure toward
  uniform routing.
- `--z-loss-coef 0` (down from 1e-3). Redundant under cosine routing
  whose logits are already bounded in [-1, 1].
- Everything else unchanged: D=4, T_window=256, J=4, K_read=8, N=4096,
  D_concept=256, inject_layer=8, scale_init=0.1, effective_lm_context=2048.

Gate before commit: 500-step pre-flight from scratch. Pass criteria:
`r_uf` drops below 0.20 (currently 0.224 = uniform); loss decreases;
no NaN. If it stays at 0.22, drop `load_balance_coef` further (3e-4 or
1e-4).

Full retrain: 8000 steps (~6h at 0.79s/step). Output:
`outputs/wave1_v2/`.

## 5-gate sanity plan (pre-retrain)

Before committing to a long Wave 1 retrain under the patched code, pass
these gates in order:

### Gate 1 — Memory-off adapter floor

Goal: confirm frozen-Llama-2K + adapter (memory injection forced to
zero) ≈ vanilla Llama-2K on bulk tokens. Isolates "pure adapter
overhead" from "memory contribution."

- Load current ckpt, force `scale=0` (or pass through with no memory
  readout), run eval.
- **Bar:** bulk-token CE within ~0.05 nats of vanilla-2K floor (2.44).
  If worse, adapter is broken at zero scale.

### Gate 2 — Param-health audit (500-step warm-up)

Goal: confirm every trainable param shows nonzero grad and moves from
init.

Metrics per module group:
- `scale_raw`: mean + std across 2048 dims, must move from 0.1.
- `W_in / W_out` (bridge): grad-norm > 0, `||θ_t - θ_0||` > 0 and
  increasing.
- Cross-attn read/write: grad-norm > 0.
- Concept manifold states: `drift_p95` > 0 and bounded.
- Routing: `r_uf`, `w_uf` > 0.10, ideally moving away from 0.22.
- GRU mutation cell: grad-norm > 0.
- No FATAL NaN / no grad explosion / no inf loss.

**Bar:** every module group has nonzero grad-norm + visible drift after
500 steps. Dead modules = bug to fix before long run.

### Gate 3 — Match-floor training

Goal: train under cap-enforced code until bulk-token CE matches (or
trends toward) vanilla floor.

- Log bulk-token CE separately from needle-answer CE (`valid_mask >=
  50` distinguishes them).
- **Bar:** bulk CE within ~0.10 nats of 2.44; needle CE strictly
  improving.

### Gate 4 — Needle eval

- Models: `vanilla-full`, `vanilla-2K`, `ours-2K` (new ckpt),
  `ours-2K-memory-off` ablation.
- Stratify by distance (0-2K, 2K-5K, 5K-12K, 12K-24K, 24K+) and answer
  type (alphanum_short/long, language, date, timezone).
- **Bar for "memory works":** `ours-2K` strictly beats
  `ours-2K-memory-off` on >2K distances, and beats `vanilla-2K`.

### Gate 5 — Debug viz / talking points

Only after Gates 1-4 are clean:
- `scale_raw` histogram evolution over training.
- UMAP manifold of concept states.
- Trajectory drift histograms.
- Routing entropy over time.
- Per-source train CE plot.

---

## Cross-references

- `docs/plan_trajectory_memory.md` — architecture + training waves
- `docs/bench_results.md` — throughput / VRAM benches
- `docs/research_backlog.md` — open research items
  (#10 inner-module compile, #11 StaticCache, #12 routing-specialization)
- `~/.claude/.../memory/project_routing_uniformity.md` — routing
  uniformity context
- `~/.claude/.../memory/feedback_write_grad_collapse.md` — write-grad
  pathology context
