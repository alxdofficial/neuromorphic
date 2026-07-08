# docs/ index

The active line is an **always-on implicit memory layer** for a frozen LM: an encoder compresses a
2048-token context into M=96 memory tokens (21:1), trained on a mixed 5-task objective
(mae / babi / qa_rc / continuation / condrecon_bio) on SmolLM2-135M (d=576) — the current sweep
default (`src/memory/data/mixes.py`); some numbers quoted deeper in these docs (e.g. `cohort_results.md`)
are the published cohort's own fixed config (ctx=1024, M=32, 4-task), not this default. The source
of truth for each model is its code under `src/memory/models/<name>/`.

## Active cohort (see `src/memory/model.py` VARIANTS)
- **Published baselines:** `icae`, `ccm`, `autocompressor`, `beacon` — learnable tokens through the
  frozen LM (± LoRA), read their hiddens; prepend M memory tokens.
- **biomem** — gated fast-Hebbian cortical-column grid: memory lives in fast synaptic state
  (chunk-parallel gated-delta write over a C×K×H column/neuron/layer grid, LIF membrane, prepend read).
  Design rationale: `biomem_chunkwise_plan.md`.
- **slotgraph** — ICAE write + a fixed node/edge partition + message-passing read over the predicted
  graph. The graph arm. In the current cohort slotgraph leads biomem (slotgraph ahead on mae +
  continuation; biomem hits the membership wall and does not beat the published compressors).
- **vqicae** — VQ-discretized ICAE; a discreteness probe.
- **vanilla_llama / vanilla_full_context** — loss floor (no memory) / ceiling (full context).

All cohort comparisons must be **same-code, same-config** — cross-era numbers are invalid.

## Docs
- **`cohort_results.md`** — the current head-to-head table (REAL loss + babi EM, the OFF/SHUF binding
  gate, and the exact-match babi_em binding test), aggregated mean ± std across seeds. Regenerate with
  `scripts/diagnostics/cohort_results.py`.
- **`slotgraph_attribution.md`** — 2×2 attribution study isolating the contribution of message-passing
  vs id-tags to slotgraph's reconstruction performance.
- **`slotgraph_metrics.md`** — standing instrument panel: structure canaries (edge-frac, src/dst entropy,
  mem_effrank) logged per checkpoint; regenerate with `scripts/diagnostics/slotgraph_metrics.py`.
- **`biomem_chunkwise_plan.md`** — biomem design rationale (chunk-parallel synaptic write, deep columns).
- **`mamba_two_lenses_memory.md`** — research note: Mamba/linear-attention lenses on compress-and-recall.

## Harness / diagnostics
- `scripts/train/train.py` — trainer (`--task mixed`); `--seed` controls init + train-data order.
- `scripts/diagnostics/cohort_results.py` — builds `cohort_results.md` from the run JSONLs + checkpoints.
- `scripts/diagnostics/mixed_band_gate_eval.py` — REAL/SHUF/OFF band + binding gate over the cohort.
- `scripts/diagnostics/mixed_dashboard.py` — per-task training/val dashboard from the run JSONLs.
