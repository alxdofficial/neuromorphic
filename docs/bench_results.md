# Bench Results — Llama-3.2-1B + graph_walker

Tracks the headline throughput / VRAM numbers for the integration. Each
section dates the run and pins the configuration so future-me can spot
regressions or progress without re-running every time.

Run benches via:
- `scripts/bench_pretrained_gw.py` — frozen-Llama (vanilla fwd, lm-head-only step) + GW (fwd, full integration step) in one process. Use `--compile-block` to enable the production compile path.
- `scripts/bench_llama_full_training.py --mode full` — hot-Llama (all 1.24B params trainable) at the same BS/T.

---

## 2026-05-02 — Three-way comparison at production config

**Setup:** RTX 4090 · BS=4 · T=256 · bf16 autocast · fused AdamW · production walker config (N=1024, K=16, D_s=256, D_id=512, neuromod 6L/8H/D_mod=512/edge_hidden=384) · `_walker_cfg_for` overrides `mod_period = tbptt_block = T = 256` to satisfy the integration's external-surprise constraint · GW path uses `--compile-block`.

| Path | tok/s | Peak VRAM | ms/iter | Trainable params | Notes |
|---|---|---|---|---|---|
| Vanilla Llama fwd-only (`no_grad`) | **54.6k** | 2.78 GB | 18.8 | 0 | Reference: pure Llama inference. |
| Frozen Llama + gradients (lm_head only) | **18.6k** | 5.84 GB | 55.0 | ~262M | Backward flows through all 16 frozen layers; only lm_head receives param-grad. Isolates "activation memory + recompute through frozen Llama" cost. |
| Frozen Llama + GW (full `phase1_pretrained_step`) | **1.8k** | 6.50 GB | 576.3 | 25.3M (24.2M walker + 1.1M inject) | Production integration training step. `--compile-block` on. neuromod-only plasticity. |
| Hot Llama (all 1.24B params trainable) | **9.0k** | 11.52 GB | 113.2 | 1.24B | Full fine-tune reference: fused AdamW state for all params + per-layer param-grads. |

**Slowdown ratios:**
- Vanilla fwd → frozen+grad: **2.9×** slowdown, +3.1 GB VRAM. Cost of running backward through frozen layers (activation memory + recompute), no param-grad cost.
- Frozen+grad → frozen+GW: **10.3×** slowdown, +0.66 GB VRAM. Pure walker compute cost — the dominant term.
- Frozen+GW → hot-Llama: GW is **~5× SLOWER** than hot-Llama training. A 25M-trainable walker is more expensive per token than a 1.24B-trainable Llama.
- Vanilla fwd → frozen+GW: **30.3×** slowdown, +3.7 GB VRAM. End-to-end "walker overhead" vs pure inference.

**Key observations:**
- **Walker is launch-/compute-bound, not memory-bound.** Post the activation-checkpointing commit (`4a19103`) and vocab-agnostic refactor, peak VRAM for the GW path is only +0.66 GB over the lm_head-only baseline (6.50 vs 5.84 GB). The pain is throughput — 576 ms/iter for the walker vs 55 ms/iter for the Llama-only baseline.
- **Walker > Llama in compute cost.** At BS=4, T=256, the walker's per-step cost (~520 ms attributable, after subtracting Llama's 55 ms baseline) exceeds even the cost of training a full 1.24B-param Llama from scratch (113 ms/iter). The walker dominates the integration training step's wall-clock.
- **Walker speedup is the headline lever.** Per `walker_speedup_plan.md`, the target is 10–15× walker speedup via reduce-overhead torch.compile + Triton fusion. At 10× speedup the GW step would land at ~57 ms/iter, comparable to the lm_head-only baseline (55 ms) — i.e., the walker would be effectively free relative to running backward through frozen Llama.
- **VRAM headroom is healthy.** 6.50 GB peak at BS=4 leaves room for at least 2-4× BS increase before hitting the 24 GB 4090 limit, modulo the per-token compute scaling.

---

## 2026-05-02 — Max-BS / per-path optimum on a single 4090

The BS=4 table above is the "fair fixed-config comparison". This section shows each path measured at **its own optimal batch size**, so absolute throughput is what each path can actually deliver when allowed to fill the GPU.

**Setup:** RTX 4090 · T=256 · bf16 autocast · fused AdamW (where applicable) · production walker config · `--compile-block` on the GW path. Each row is a single `bench_pretrained_gw.py` or `bench_llama_full_training.py` subprocess.

| Path | Best BS | tok/s @ best | Peak VRAM | ms/iter | Notes |
|---|---|---|---|---|---|
| Vanilla Llama fwd-only (`no_grad`) | **8** | **54.5k** | 3.08 GB | 37.6 | Plateaus at BS=8; BS=16-256 stays in the 49-54k range — Llama-1B forward is compute-bound at small BS. |
| Frozen Llama + gradients (lm_head + opt.step) | **16** | **17.6k** | 16.94 GB | 232.2 | Sweep with `--mode lmhead-only`; BS=32 OOMs. |
| Frozen Llama + GW (full `phase1_pretrained_step`) | **12** | **4.5k** | 11.52 GB | 675.5 | BS=4: 1.9k → BS=8: 3.3k → BS=12: 4.5k (≈linear scaling, confirming launch-bound). BS=16 was unreliable in the same-process run (cross-test memory fragmentation; reported a degraded 1.3k); should be cleanly measurable with a fresh subprocess and could fit ~14 GB. |
| Hot Llama (all 1.24B trainable) | **16** | **12.7k** | 23.00 GB | 322.6 | At the 24 GB ceiling; BS=32 OOMs. |

**Slowdown ratios at each path's own optimum:**
- Vanilla fwd → frozen+grad: **3.1×** slowdown (54.5k → 17.6k)
- Frozen+grad → frozen+GW: **3.9×** slowdown (17.6k → 4.5k) — the walker's marginal cost
- Frozen+GW vs hot-Llama: walker is **2.8× slower** than full 1.24B fine-tune (4.5k vs 12.7k)
- Vanilla fwd → frozen+GW: **12.1×** slowdown (54.5k → 4.5k) — end-to-end walker overhead vs pure inference

**Comparison to the BS=4 fixed-config slowdowns:**
| Slowdown | At BS=4 fixed | At each path's optimum |
|---|---|---|
| Frozen+grad → frozen+GW | 10.3× | **3.9×** |
| Vanilla fwd → frozen+GW | 30.3× | **12.1×** |
| Frozen+GW vs hot-Llama | 5× slower | **2.8× slower** |

The walker scales much better with BS than the BS=4 numbers suggested — confirming the launch-bound diagnosis. Pure BS scaling alone (BS=4 → BS=12) gave **2.4× speedup** for the GW path with no code changes. The "walker is hopelessly slow" framing from the fair-BS table was overstated; at each path's optimum, the gap is closer to a 4× slowdown vs the lm_head baseline, which is the right neighborhood for an architecture this novel.

**Updated headline numbers:**
- **Achievable now (no code changes):** 4.5k tok/s GW step at BS=12 → 1B-token training run in **~62 hours** (vs ~6.4 days at BS=4).
- **After cudagraph + Triton fusion (10× walker speedup, plan):** ~22ms walker step → ~70 ms total step at BS=12 → ~44k tok/s → 1B-token run in **~6 hours**, comfortably in the iteration regime.

---

## 2026-05-03 — `_checkpoint_block=False` is now the integration default

**Diagnostic finding**: The BS=12 → BS=16 cliff in the GW step path was caused by activation checkpointing on the whole compiled walker block (`compile_block_from_h`). When `_checkpoint_block=True`, backward re-runs the compiled forward to regenerate intermediates; that recompute triggers a separate inductor compilation of the backward gradient kernels. At BS≥16 that backward compile hits a cuBLAS autotuner failure — `select_algorithm.py: "Constructing input/output tensor meta failed for Extern Choice"` warnings fire — and falls back to a much slower kernel path.

**Diagnostic table (single 4090, T=256, --compile-block, BS=16):**

| Config | tok/s | ms/iter | Peak VRAM |
|---|---|---|---|
| `--compile-block` + `_checkpoint_block=True` (old default) | 2.2k | 1826 | 13.99 GB |
| `--compile` (per-step) + ckpt=True | 6.3k | 649 | 16.88 GB |
| `--compile-block` + **`_checkpoint_block=False`** (new default) | **8.8k** | **464** | **14.59 GB** |

Checkpoint OFF is **4× faster** than checkpoint ON at BS=16, with only +0.6 GB VRAM cost. The memory savings ckpt=True buys (~3 GB at BS=16) is not worth the wall-clock cost when there's 9+ GB of GPU headroom.

**Action taken:** `src/graph_walker/pretrained/llm_wrapper.py` `__init__` now sets `self.memory._checkpoint_block = False` after constructing memory. Standalone walker training is unaffected (it uses the per-token `step_core` path, not the compile-block path, so `_checkpoint_block` is irrelevant there).

**Headline at the new default:**
- **Current best: 8.8k tok/s @ BS=16, T=256, 14.59 GB peak VRAM.**
- 69% of the hot-Llama bar (12.7k tok/s @ BS=16). Gap closed from 2.8× → 1.4×.
- 1B-token training run drops from ~62 hours (at the old BS=12 best) to **~32 hours**.
- 9.4 GB VRAM headroom remaining at BS=16 — could push to BS=20+ if scaling continues cleanly.

